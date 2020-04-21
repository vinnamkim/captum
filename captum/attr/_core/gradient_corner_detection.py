#!/usr/bin/env python3
from typing import Any, Callable, Tuple

import torch.nn.functional as F
import torch
from torch import Tensor

from ..._utils.common import _format_input, _is_tuple
from ..._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from .._utils.attribution import GradientAttribution
from .._utils.common import _format_attributions
from .._utils.gradient import apply_gradient_requirements, undo_gradient_requirements


class GradientCornerDetection(GradientAttribution):
    r"""
    A baseline approach for computing the attribution. It multiplies input with
    the gradient with respect to input.
    https://arxiv.org/abs/1611.07270
    """

    def __init__(self, forward_func: Callable) -> None:
        r"""
        Args:

                forward_func (callable):  The forward function of the model or any
                                          modification of it
        """
        GradientAttribution.__init__(self, forward_func)

    def attribute(
            self,
            inputs: TensorOrTupleOfTensorsGeneric,
            target: TargetType = None,
            size: int = 2,
            method: str = 'noble',
            force_double_type: bool = True,
            additional_forward_args: Any = None,
            **kwargs
    ) -> TensorOrTupleOfTensorsGeneric:
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)

        inputs = _format_input(inputs)
        gradient_mask = apply_gradient_requirements(inputs)

        gradients = self.gradient_func(
            self.forward_func, inputs, target, additional_forward_args
        )

        if force_double_type is True:
            gradients = tuple(
                [gradient.to(dtype=torch.float64) for gradient in gradients]
            )

        with torch.no_grad():
            attributions = tuple(
                [self.get_score_gpu(input, gradient, size, method, **kwargs)
                for input, gradient in zip(inputs, gradients)]
            )

        undo_gradient_requirements(inputs, gradient_mask)
        return _format_attributions(is_inputs_tuple, attributions)

    def parse_sampling_kwargs(self, **kwargs) -> Tuple[str, int]:
        if 'sampling_method' not in kwargs:
            raise Exception(
                'If you choose method=sampling, then you should set sampling_method argument.')
        if 'num_samples' not in kwargs:
            raise Exception(
                'If you choose method=sampling, then you should set num_samples argument.')
        
        sampling_method = kwargs.get('sampling_method')
        num_samples = kwargs.get('num_samples')
        
        return sampling_method, num_samples

    def get_score_gpu(self, input: Tensor, gradient: Tensor,
                      size: int = 2, method='noble', **kwargs) -> Tensor:
        channels, height, width = gradient.shape[1], gradient.shape[2], gradient.shape[3]

        # 3-dim : (height x width) x channels x 1
        X = gradient.flatten(2).permute(2, 1, 0)

        # 3-dim : (height x width) x channels x channels
        X = torch.bmm(X, X.permute(0, 2, 1))

        # 2-dim : (height x width) x (channels x channels)
        X = X.reshape(-1, channels * channels)

        # 4-dim : 1 x (channels x channels) x height x width
        X = X.reshape(height, width, channels * channels).permute(2, 0, 1).unsqueeze(0)

        kernel_size = 2 * size + 1
        weight = torch.ones(
            size=[channels * channels, 1, kernel_size, kernel_size],
            dtype=X.dtype, device=X.device)

        # Get a sensitivity matrix X for a rgb perturbation vector
        # 4-dim : 1 x (channels x channels) x height x width
        X = F.conv2d(X, weight=weight, bias=None, stride=1, padding=size,
                     dilation=1, groups=channels * channels)
        if method == 'noble':
            # 3-dim : (height x width) x channels x channels
            X_mat = X.unsqueeze(0).reshape(channels, channels, height, width).permute(
                2, 3, 0, 1).reshape(-1, channels, channels)

            X_det = torch.det(X_mat)
            X_trace = X_mat.diagonal(dim1=-2, dim2=-1).sum(-1)
            score = 2 * X_det / (X_trace + 1e-6)
            score = score.reshape(height, width)
        elif method == 'fro':
            score = X.norm(dim=1, keepdim=True)
        elif method == 'shi-tomasi':
            X_mat = X.unsqueeze(0).reshape(channels, channels, height, width).permute(
                2, 3, 0, 1).reshape(-1, channels, channels)
            
            S, _ = torch.symeig(X_mat, eigenvectors=False)

            return S[:, -1].reshape(height, width)

        elif method == 'sampling':
            sampling_method, num_samples = self.parse_sampling_kwargs(
                **kwargs)

            # unfold = F.unfold(input, kernel_size, dilation=1, padding=size, stride=1).reshape(
            #     1, channels, -1, height * width).squeeze(0).permute(2, 0, 1)
            # diff = unfold - unfold.mean(2, keepdim=True)

            # cov = diff.bmm(diff.permute(0, 2, 1)) / diff.shape[-1]
            # L = torch.cholesky(cov, upper=False).to(dtype=X.dtype)

            # 3-dim : (height x width) x (channels x channels)
            X_mat = X.unsqueeze(0).reshape(channels, channels, height, width).permute(
                2, 3, 0, 1).reshape(-1, channels, channels)

            samples = torch.randn(
                (X_mat.shape[0], num_samples, X_mat.shape[2]),
                dtype=X_mat.dtype, device=X_mat.device
            )
            samples /= samples.norm(dim=-1, keepdim=True)
            #samples = torch.bmm(samples, L.permute(0, 2, 1))

            score = torch.bmm(samples, X_mat)

            if sampling_method == 'std':
                score = (score * samples).sum(-1).sqrt().std(-1).reshape(height, width)
            elif sampling_method == 'mean':
                score = (score * samples).sum(-1).sqrt().mean(-1).reshape(height, width)
            elif sampling_method == 'min':
                score = (score * samples).sum(-1).sqrt().min(-1)[0].reshape(height, width)
            elif sampling_method == 'max':
                score = (score * samples).sum(-1).sqrt().max(-1)[0].reshape(height, width)
        else:
            raise Exception(
                'method should be one of {\'noble\', \'fro\', \'sampling\'}.')

        if score.dim() == 2:
            score = score.unsqueeze(0).unsqueeze(0)

        return score.repeat(1, channels, 1, 1)

    def get_score_cpu(self, gradient: Tensor, size: int = 2, method='noble', **kwargs) -> Tensor:
        channels, height, width = gradient.shape[1], gradient.shape[2], gradient.shape[3]

        padded_gradient = F.pad(
            gradient, (size, size, size, size), 'constant', 0).cpu()

        score = torch.zeros([height, width])

        if method == 'noble':
            def func(M): return 2 * torch.det(M) / (torch.trace(M) + 1e-6)
        elif method == 'fro':
            def func(M): return torch.norm(M)
        elif method == 'sampling':
            def func(M):
                sample = torch.randn([1000, channels], dtype=M.dtype)
                score = torch.mm(sample, M)
                score = score * sample
                return score.sum(-1).mean(-1)
        else:
            raise Exception('method should be one of {\'noble\', \'fro\'}.')

        for h in range(height):
            for w in range(width):
                M = torch.zeros([channels, channels], dtype=gradient.dtype)
                for u in range(-size, size + 1):
                    for v in range(-size, size + 1):
                        vec = padded_gradient[:, :, h + u + size,
                                              w + v + size].reshape(channels, 1)
                        M += torch.mm(vec, vec.T)

                score[h, w] = func(M)

            print('{}/{}'.format(h + 1, height))

        return score
