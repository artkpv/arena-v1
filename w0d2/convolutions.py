import numpy as np
from fancy_einsum import einsum
from einops import reduce, rearrange, repeat
from typing import Union, Optional, Callable
import torch as t
import torchvision
import utils

def conv1d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    '''Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    b_num, inc_num, w_val = x.shape
    outc_num, w_inc_num, kw_val = weights.shape
    assert inc_num == w_inc_num, f'{inc_num} == {w_inc_num}'
    xs = x.stride()
    strided_x = t.as_strided(
        x,
        size=(
            b_num,
            inc_num,
            w_val - kw_val + 1,
            kw_val
        ),
        stride=(
            xs[0],  # Next batch.
            xs[1],  # Next in-channel.
            xs[2],  # Next kernel.
            xs[2]  # Next element.
        )
    )
    res = t.einsum('b i j k, o i k -> b o j', strided_x, weights)
    return res

def toy_test():
    x = t.tensor([
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[10, 11, 12, 13], [14, 15, 16, 17]]
    ], dtype=t.double)
    m = t.nn.Conv1d(
        in_channels=x.shape[1],
        out_channels=3,
        kernel_size=3,
        bias=False,
        dtype=t.double
    )
    weights = next(m.parameters()).data.detach()
    res = conv1d_minimal(x, weights)
    exp = m(x).detach()
    t.testing.assert_close(res, exp )
    print('PASS: toy_test')

toy_test()
utils.test_conv1d_minimal(conv1d_minimal)


def conv2d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    '''Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    b_num, inc_num, h_val, w_val = x.shape
    outc_num, w_inc_num, kh_val, kw_val = weights.shape
    assert inc_num == w_inc_num, f'{inc_num} == {w_inc_num}'
    xs = x.stride()
    out_w, out_h = (w_val - kw_val + 1), (h_val - kh_val + 1)
    k_size = kh_val * kw_val
    strided_x = t.as_strided(
        x,
        size=(
            b_num,
            inc_num,
            out_h,
            out_w,
            kh_val,
            kw_val
        ),
        stride=(
            xs[0],  # Next batch.
            xs[1],  # Next in-channel
            w_val,
            1,
            w_val,
            1
        )
    )
    res = t.einsum('b c h w i j, o c i j -> b o h w', strided_x, weights)
    return res

utils.test_conv2d_minimal(conv2d_minimal)


def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    '''
    s = x.shape
    xx = x.new_full((s[0], s[1], left + s[2] + right), pad_value)
    xx[..., left:(-right if right > 0 else None)] = x[..., :]
    return xx

utils.test_pad1d(pad1d)
utils.test_pad1d_multi_channel(pad1d)

def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    '''
    s = x.shape
    xx = x.new_full((s[0], s[1], top + s[2] + bottom, left + s[3] + right), pad_value)
    xx[:,:, top:(-bottom if bottom>0 else None), left:(-right if right > 0 else None)] = x[:,:, :, :]
    return xx

utils.test_pad2d(pad2d)
utils.test_pad2d_multi_channel(pad2d)


def conv1d(x, weights, stride: int = 1, padding: int = 0) -> t.Tensor:
    '''Like torch's conv1d using bias=False.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    x = pad1d(x, padding, padding, 0.0)
    b_num, inc_num, w_val = x.shape
    outc_num, w_inc_num, kw_val = weights.shape
    assert inc_num == w_inc_num, f'{inc_num} == {w_inc_num}'
    xs = x.stride()
    l = (w_val - kw_val) // stride + 1
    strided_x = t.as_strided(
        x,
        size=(
            b_num,
            inc_num,
            l,
            kw_val
        ),
        stride=(
            xs[0],  # Next batch.
            xs[1],  # Next in-channel.
            xs[2] + stride - 1,  # Next kernel.
            xs[2]  # Next element.
        )
    )
    res = t.einsum('b i j k, o i k -> b o j', strided_x, weights)
    return res

utils.test_conv1d(conv1d)


IntOrPair = Union[int, tuple[int, int]]
Pair = tuple[int, int]

def force_pair(v: IntOrPair) -> Pair:
    '''Convert v to a pair of int, if it isn't already.'''
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)

# Examples of how this function can be used:
#       force_pair((1, 2))     ->  (1, 2)
#       force_pair(2)          ->  (2, 2)
#       force_pair((1, 2, 3))  ->  ValueError


def conv2d(x, weights, stride: IntOrPair = 1, padding: IntOrPair = 0) -> t.Tensor:
    '''Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    ver_s, hor_s = force_pair(stride)
    h_pad, w_pad = force_pair(padding)
    x = pad2d(x, w_pad, w_pad, h_pad, h_pad, 0.0)
    b_num, inc_num, h_val, w_val = x.shape
    outc_num, w_inc_num, kh_val, kw_val = weights.shape
    assert inc_num == w_inc_num, f'{inc_num} == {w_inc_num}'
    xs = x.stride()
    out_w = (w_val - kw_val)//hor_s + 1
    out_h = (h_val - kh_val)//ver_s + 1
    k_size = kh_val * kw_val
    strided_x = t.as_strided(
        x,
        size=(
            b_num,
            inc_num,
            out_h,
            out_w,
            kh_val,
            kw_val
        ),
        stride=(
            xs[0],  # Next batch.
            xs[1],  # Next in-channel
            w_val * ver_s,
            hor_s,
            w_val,
            1
        )
    )
    res = t.einsum('b c h w i j, o c i j -> b o h w', strided_x, weights)
    return res

utils.test_conv2d(conv2d)


def maxpool2d(
    x: t.Tensor,
    kernel_size: IntOrPair,
    stride: Optional[IntOrPair] = None,
    padding: IntOrPair = 0
) -> t.Tensor:
    '''Like PyTorch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, out_height, output_width)
    '''
    ver_s, hor_s = force_pair(stride if stride else kernel_size)
    h_pad, w_pad = force_pair(padding)
    x = pad2d(x, w_pad, w_pad, h_pad, h_pad, -float('inf'))
    b_num, c_num, h_val, w_val = x.shape
    kh_val, kw_val = force_pair(kernel_size)
    xs = x.stride()
    out_w = (w_val - kw_val)//hor_s + 1
    out_h = (h_val - kh_val)//ver_s + 1
    strided_x = t.as_strided(
        x,
        size=(
            b_num,
            c_num,
            out_h,
            out_w,
            kh_val,
            kw_val
        ),
        stride=(
            xs[0],  # Next batch.
            xs[1],  # Next in-channel
            w_val * ver_s,
            hor_s,
            w_val,
            1
        )
    )
    res = t.amax(strided_x, (4,5))
    return res

utils.test_maxpool2d(maxpool2d)
