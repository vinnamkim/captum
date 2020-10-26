#!/usr/bin/env python3
from typing import Any, Callable, Tuple

import torch.nn.functional as F
import torch
from torch import Tensor
import math
from ..._utils.common import _format_input, _is_tuple
from ..._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from .._utils.attribution import GradientAttribution
from .._utils.common import _format_attributions
from .._utils.gradient import apply_gradient_requirements, undo_gradient_requirements


def get_gaussian_kernel(kernel_size, sigma, channels, dtype, device):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) *\
        torch.exp(
        -torch.sum((xy_grid - mean)**2., dim=-1)
        / (2 * variance)
    )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(
        channels * channels, 1, 1, 1).to(dtype=dtype, device=device)
    return gaussian_kernel


class AttributionalCornerDetection(GradientAttribution):
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
            kernel_type: str = 'window',
            kernel_size: int = 5,
            kernel_sigma: float = 0.5,
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
                [self.get_score_gpu(
                    input, gradient,
                    kernel_type, kernel_size, kernel_sigma,
                    method, **kwargs)
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
                      kernel_type: str, kernel_size: int, kernel_sigma: float,
                      method: str, **kwargs) -> Tensor:
        batch_size, channels, height, width = \
            gradient.shape[0], gradient.shape[1], gradient.shape[2], gradient.shape[3]
        # 5-dim : batch_size x height x width x channels x 1
        X = gradient.permute(0, 2, 3, 1).unsqueeze(-1)
        # print(X.shape)

        # 4-dim : batch_size x height x width x (channels x channels)
        X = torch.matmul(X, X.transpose(4, 3)).flatten(3)
        # print(X.shape)

        # 4-dim : batch_size x (channels x channels) x height x width
        X = X.permute(0, 3, 1, 2)
        # print(X.shape)

        if kernel_type == 'window':
            weight = torch.ones(
                size=[channels * channels, 1, kernel_size, kernel_size],
                dtype=X.dtype, device=X.device)
        elif kernel_type == 'gaussian':
            weight = get_gaussian_kernel(
                kernel_size=kernel_size, sigma=kernel_sigma, channels=channels,
                dtype=X.dtype, device=X.device)
        else:
            raise Exception('Unknown kernel type : {}'.format(kernel_type))

        # Get a sensitivity matrix X for a rgb perturbation vector
        # 4-dim : batch_size x (channels x channels) x height x width
        X = F.conv2d(X, weight=weight, bias=None, stride=1, padding=(kernel_size - 1) // 2,
                     dilation=1, groups=channels * channels)
        #print('conv', X.shape)

        # 4-dim : batch_size x height x width x channels x channels
        X = X.permute(0, 2, 3, 1).reshape(
            batch_size, height, width, channels, channels)
        #print('X_mat', X.shape)

        if method == 'noble':
            X_det = torch.det(X)
            # print(X_det.shape)
            X_trace = X.diagonal(dim1=-2, dim2=-1).sum(-1)
            # print(X_trace.shape)
            score = (X_det / (X_trace + 1e-6)).unsqueeze(1)
            # print(score.shape)
        elif method == 'fro':
            score = X.norm(p='fro', dim=(3, 4), keepdim=False).unsqueeze(1)
            # print(score.shape)
        elif method == 'min':
            # S = torch.lobpcg(X, k=1, largest=False)
            # print(S)
            # S, _ = torch.symeig(X, eigenvectors=False)
            # print(S.shape)
            raise Exception('Do not method=min.')
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
                score = (
                    score * samples).sum(-1).sqrt().min(-1)[0].reshape(height, width)
            elif sampling_method == 'max':
                score = (
                    score * samples).sum(-1).sqrt().max(-1)[0].reshape(height, width)
        else:
            raise Exception(
                f'method should be one of [noble, fro, min, sampling]. But it is {method}.')

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
