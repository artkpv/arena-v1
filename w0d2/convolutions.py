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
    strided_x = t.as_strided(
        x,
        size=(
            b_num,
            outc_num,
            inc_num,
            w_val - kw_val + 1,
            kw_val
        ),
        stride=(
            inc_num * w_val,  # Next batch.
            0,  # Repeat for out-channels.
            w_val,  # Next in-channel.
            1,  # Next kernel.
            1  # Next element.
        )
    )
    # print(x)
    # print(x.shape)
    # print(strided_x)
    # print(strided_x.shape)
    strided_w = t.as_strided(
        weights,
        size=(
            b_num,
            outc_num,
            inc_num,
            w_val - kw_val + 1,
            kw_val
        ),
        stride=(
            0,  # Repeat for every batch.
            w_inc_num * kw_val,  # Next out-channel.
            kw_val, # Next in-channel
            0,  # Repeat 
            1  # Next element
        )
    )
    # print(weights)
    # print(weights.shape)
    # print(strided_w)
    # print(strided_w.shape)
    res = t.einsum('n o i s w, n o i s w -> n o s', strided_x, strided_w)
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
