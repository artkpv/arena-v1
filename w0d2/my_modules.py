import torch as t
from torch import nn
import convolutions
from typing import Union, Optional, Callable
import utils
from utils import IntOrPair, Pair, force_pair
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

if __name__ == '__main__':

    utils.test_maxpool2d_module(MaxPool2d)
    m = MaxPool2d(kernel_size=3, stride=2, padding=1)
    print(f"Manually verify that this is an informative repr: {m}")

    utils.test_relu(ReLU)

    utils.test_flatten(Flatten)

