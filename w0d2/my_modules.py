import torch as t
from torch import nn
import w0d2.convolutions
from typing import Union, Optional, Callable
import w0d2.utils
from w0d2.utils import IntOrPair, Pair, force_pair
from einops import reduce, rearrange, repeat


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Call the functional version of maxpool2d.'''
        return convolutions.maxpool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding
        )

    def extra_repr(self) -> str:
        '''Add additional information to the string representation of this class.'''
        return f'{self.kernel_size} KS, {self.stride} S, {self.padding} P'


class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.maximum(x, t.tensor(0))


class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        '''Flatten out dimensions from start_dim to end_dim, inclusive of both.
        '''
        def num2char(n):
            res = ['a']
            while n > 0:
                n, r = divmod(n, 10)
                res += [chr(ord('a') + r)]
            return ''.join(res)

        dims = len(input.shape)
        r = dims + self.end_dim if self.end_dim < 0 else self.end_dim
        l = self.start_dim
        if not l < r:
             return input
        pattern = [num2char(i) for i in range(dims)]
        out_p = pattern[:]
        out_p[l] = '(' + out_p[l]
        out_p[r] = out_p[r] + ')'
        opspattern = ' '.join(pattern) + ' -> ' + ' '.join(out_p)
        return rearrange(input, opspattern)

    def extra_repr(self) -> str:
        return f'{self.start_dim}-{self.end_dim}'


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        '''A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        '''
        super().__init__()
        self.weight = nn.Parameter(t.randn(out_features, in_features))
        t.nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(t.randn(out_features)) if bias else None

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (*, in_features)
        Return: shape (*, out_features)
        '''
        res = x @ self.weight.T 
        if self.bias is not None:
            res += self.bias
        return res

    def extra_repr(self) -> str:
        return f'{self.weight.shape=} {self.bias.shape if self.bias else ""}'


class Conv2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0
    ):
        '''
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        '''
        super().__init__()
        kh, kw = force_pair(kernel_size)
        self.weight = nn.Parameter(t.randn(out_channels, in_channels, kh, kw))
        t.nn.init.xavier_uniform_(self.weight)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Apply the functional conv2d you wrote earlier.'''
        return convolutions.conv2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        return f'{self.weight.shape=} {self.stride=} {self.padding=}'

if __name__ == '__main__':
    utils.test_maxpool2d_module(MaxPool2d)
    m = MaxPool2d(kernel_size=3, stride=2, padding=1)
    print(f"Manually verify that this is an informative repr: {m}")

    utils.test_relu(ReLU)

    utils.test_flatten(Flatten)

    utils.test_linear_forward(Linear)
    utils.test_linear_parameters(Linear)
    utils.test_linear_no_bias(Linear)

    utils.test_conv2d_module(Conv2d)

