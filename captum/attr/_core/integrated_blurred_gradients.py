#!/usr/bin/env python3
import typing
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..._utils.common import (_expand_additional_forward_args, _expand_target,
                              _format_additional_forward_args, _is_tuple)
from ..._utils.typing import (BaselineType, Literal, TargetType,
                              TensorOrTupleOfTensorsGeneric)
from .._utils.approximation_methods import approximation_parameters
from .._utils.attribution import GradientAttribution
from .._utils.batching import _batch_attribution
from .._utils.common import (_format_attributions, _format_input,
                             _reshape_and_sum, _validate_input)


class GaussianFilter(torch.nn.Module):
    def __init__(self, log_sigma: float, kernel_size: int = 15, channels: int = 3):
        super().__init__()
        self.padding = (kernel_size - 1) // 2
        self.bias = None
        self.stride = 1
        self.dilation = 1
        self.groups = channels
        # Set these to whatever you want for your gaussian filter
        kernel_size = 15
        #self.sigma = torch.autograd.Variable(torch.tensor(sigma, requires_grad=True))
        self.log_sigma = torch.tensor(log_sigma, requires_grad=True)
        variance = torch.exp(2. * self.log_sigma)

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1) / 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * np.pi * variance)) *\
            torch.exp(
            -torch.sum((xy_grid - mean)**2., dim=-1)
            / (2 * variance)
        )
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
        self.weight = gaussian_kernel

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        # return self.conv2d_forward(x, self.weight)

    def get_dgf_dalpha(self, x):
        gf_x = self.forward(x)
        v = torch.ones_like(gf_x, requires_grad=True)
        vjp = torch.autograd.grad(gf_x, self.log_sigma,
                                  grad_outputs=v, create_graph=True)
        return torch.autograd.grad(vjp, v)[0]


class IntegratedBlurredGradients(GradientAttribution):
    r"""
    Integrated Gradients is an axiomatic model interpretability algorithm that
    assigns an importance score to each input feature by approximating the
    integral of gradients of the model's output with respect to the inputs
    along the path (straight line) from given baselines / references to inputs.

    Baselines can be provided as input arguments to attribute method.
    To approximate the integral we can choose to use either a variant of
    Riemann sum or Gauss-Legendre quadrature rule.

    More details regarding the integrated gradients method can be found in the
    original paper:
    https://arxiv.org/abs/1703.01365

    """

    def __init__(self, forward_func: Callable,
                 min_log_sigma: float = -2.0, max_log_sigma: float = 2.0, kernel_size: int = 15) -> None:
        r"""
        Args:

            forward_func (callable):  The forward function of the model or any
                          modification of it
        """
        GradientAttribution.__init__(self, forward_func)
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        self.kernel_size = kernel_size

    # The following overloaded method signatures correspond to the case where
    # return_convergence_delta is False, then only attributions are returned,
    # and when return_convergence_delta is True, the return type is
    # a tuple with both attributions and deltas.
    @typing.overload
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: Literal[False] = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        ...

    @typing.overload
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        *,
        return_convergence_delta: Literal[True],
    ) -> Tuple[TensorOrTupleOfTensorsGeneric, Tensor]:
        ...

    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: bool = False,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, Tensor]
    ]:
        r"""
        This method attributes the output of the model with given target index
        (in case it is provided, otherwise it assumes that output is a
        scalar) to the inputs of the model using the approach described above.

        In addition to that it also returns, if `return_convergence_delta` is
        set to True, integral approximation delta based on the completeness
        property of integrated gradients.

        Args:

            inputs (tensor or tuple of tensors):  Input for which integrated
                        gradients are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            baselines (scalar, tensor, tuple of scalars or tensors, optional):
                        Baselines define the starting point from which integral
                        is computed and can be provided as:

                        - a single tensor, if inputs is a single tensor, with
                          exactly the same dimensions as inputs or the first
                          dimension is one and the remaining dimensions match
                          with inputs.

                        - a single scalar, if inputs is a single tensor, which will
                          be broadcasted for each input value in input tensor.

                        - a tuple of tensors or scalars, the baseline corresponding
                          to each tensor in the inputs' tuple can be:

                          - either a tensor with matching dimensions to
                            corresponding tensor in the inputs' tuple
                            or the first dimension is one and the remaining
                            dimensions match with the corresponding
                            input tensor.

                          - or a scalar, corresponding to a tensor in the
                            inputs' tuple. This scalar value is broadcasted
                            for corresponding input tensor.
                        In the cases when `baselines` is not provided, we internally
                        use zero scalar corresponding to each input tensor.

                        Default: None
            target (int, tuple, tensor or list, optional):  Output indices for
                        which gradients are computed (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the target for the corresponding example.

                        For outputs with > 2 dimensions, targets can be either:

                        - A single tuple, which contains #output_dims - 1
                          elements. This target index is applied to all examples.

                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          target for the corresponding example.

                        Default: None
            additional_forward_args (any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a
                        tuple containing multiple additional arguments including
                        tensors or any arbitrary python types. These arguments
                        are provided to forward_func in order following the
                        arguments in inputs.
                        For a tensor, the first dimension of the tensor must
                        correspond to the number of examples. It will be
                        repeated for each of `n_steps` along the integrated
                        path. For all other types, the given argument is used
                        for all forward evaluations.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            n_steps (int, optional): The number of steps used by the approximation
                        method. Default: 50.
            method (string, optional): Method for approximating the integral,
                        one of `riemann_right`, `riemann_left`, `riemann_middle`,
                        `riemann_trapezoid` or `gausslegendre`.
                        Default: `gausslegendre` if no method is provided.
            internal_batch_size (int, optional): Divides total #steps * #examples
                        data points into chunks of size at most internal_batch_size,
                        which are computed (forward / backward passes)
                        sequentially. internal_batch_size must be at least equal to
                        #examples.
                        For DataParallel models, each batch is split among the
                        available devices, so evaluations on each available
                        device contain internal_batch_size / num_devices examples.
                        If internal_batch_size is None, then all evaluations are
                        processed in one batch.
                        Default: None
            return_convergence_delta (bool, optional): Indicates whether to return
                    convergence delta or not. If `return_convergence_delta`
                    is set to True convergence delta will be returned in
                    a tuple following attributions.
                    Default: False
        Returns:
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*tensor* or tuple of *tensors*):
                    Integrated gradients with respect to each input feature.
                    attributions will always be the same size as the provided
                    inputs, with each value providing the attribution of the
                    corresponding input index.
                    If a single tensor is provided as inputs, a single tensor is
                    returned. If a tuple is provided for inputs, a tuple of
                    corresponding sized tensors is returned.
            - **delta** (*tensor*, returned if return_convergence_delta=True):
                    The difference between the total approximated and true
                    integrated gradients. This is computed using the property
                    that the total sum of forward_func(inputs) -
                    forward_func(baselines) must equal the total sum of the
                    integrated gradient.
                    Delta is calculated per example, meaning that the number of
                    elements in returned delta tensor is equal to the number of
                    of examples in inputs.

        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> net = ImageClassifier()
            >>> ig = IntegratedGradients(net)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Computes integrated gradients for class 3.
            >>> attribution = ig.attribute(input, target=3)
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)
        
        inputs = _format_input(inputs)
        
        baselines = [0 for _ in range(len(inputs))]

        _validate_input(inputs, baselines, n_steps, method)

        if internal_batch_size is not None:
            num_examples = inputs[0].shape[0]
            attributions = _batch_attribution(
                self,
                num_examples,
                internal_batch_size,
                n_steps,
                inputs=inputs,
                baselines=baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                method=method,
            )
        else:
            attributions = self._attribute(
                inputs=inputs,
                target=target,
                additional_forward_args=additional_forward_args,
                n_steps=n_steps,
                method=method,
            )

        if return_convergence_delta:
            baselines = tuple(
                GaussianFilter(self.max_log_sigma)(input)
                for input in inputs
            )
            start_point, end_point = baselines, inputs
            # computes approximation error based on the completeness axiom
            delta = self.compute_convergence_delta(
                attributions,
                start_point,
                end_point,
                additional_forward_args=additional_forward_args,
                target=target,
            )
            return _format_attributions(is_inputs_tuple, attributions), delta
        return _format_attributions(is_inputs_tuple, attributions)

    def rescale_alpha(self, alpha):
        return (self.max_log_sigma - self.min_log_sigma) * alpha + self.min_log_sigma

    def _attribute(
        self,
        inputs: Tuple[Tensor, ...],
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        step_sizes_and_alphas: Union[None, Tuple[List[float], List[float]]] = None,
    ) -> Tuple[Tensor, ...]:
        if step_sizes_and_alphas is None:
            # retrieve step size and scaling factor for specified
            # approximation method
            step_sizes_func, alphas_func = approximation_parameters(method)
            step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)
        else:
            step_sizes, alphas = step_sizes_and_alphas

        # scale features and compute gradients. (batch size is abbreviated as bsz)
        # scaled_features' dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        gaussian_filters = [
            GaussianFilter(self.rescale_alpha(alpha)) for alpha in alphas
        ]

        scaled_features_tpl = tuple(
            torch.cat(
                [gf(input) for gf in gaussian_filters], dim=0
            ).requires_grad_()
            for input in inputs
        )

        dfeatures_dalpha = tuple(
            torch.cat(
                [gf.get_dgf_dalpha(input) for gf in gaussian_filters], dim=0
            ).contiguous().view(n_steps, -1)
            for input in inputs
        )

        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # apply number of steps to additional forward args
        # currently, number of steps is applied only to additional forward arguments
        # that are nd-tensors. It is assumed that the first dimension is
        # the number of batches.
        # dim -> (bsz * #steps x additional_forward_args[0].shape[1:], ...)
        input_additional_args = (
            _expand_additional_forward_args(additional_forward_args, n_steps)
            if additional_forward_args is not None
            else None
        )
        expanded_target = _expand_target(target, n_steps)

        # grads: dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        grads = self.gradient_func(
            forward_fn=self.forward_func,
            inputs=scaled_features_tpl,
            target_ind=expanded_target,
            additional_forward_args=input_additional_args,
        )

        # flattening grads so that we can multilpy it with step-size
        # calling contiguous to avoid `memory whole` problems
        scaled_grads = [
            grad.contiguous().view(n_steps, -1)
            * torch.tensor(step_sizes).view(n_steps, 1).to(grad.device)
            for grad in grads
        ]

        # aggregates across all steps for each tensor in the input tuple
        # total_grads has the same dimensionality as inputs
        total_grads = [
            _reshape_and_sum(
                scaled_grad * dfeature_dalpha, 
                n_steps, grad.shape[0] // n_steps, grad.shape[1:])
            for scaled_grad, grad, dfeature_dalpha in zip(scaled_grads, grads, dfeatures_dalpha)
        ]

        # computes attribution for each tensor in input tuple
        # attributions has the same dimensionality as inputs
        attributions = tuple(total_grads)
        return attributions

    def has_convergence_delta(self) -> bool:
        return True
