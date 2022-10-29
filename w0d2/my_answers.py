import numpy as np
from fancy_einsum import einsum
from einops import reduce, rearrange, repeat
from typing import Union, Optional, Callable
import torch as t
import torchvision
import utils

arr = np.load("numbers.npy")

# utils.display_array_as_img(arr[0])

# utils.display_array_as_img(rearrange(arr, 'b c h w -> c h (b w)'))

# utils.display_array_as_img(repeat(arr[0:2], 'b c h w -> c (b h) (2 w)'))

# utils.display_array_as_img(repeat(arr[0], 'c h w -> c (h 2) w'))

# utils.display_array_as_img(repeat(arr[0], 'c h w -> h (c w)'))
# utils.display_array_as_img(rearrange(arr[0], 'c h w -> h (c w)'))

# utils.display_array_as_img(rearrange(arr, '(b1 b2) c h w -> c (b1 h) (b2 w)', b1=2))

# utils.display_array_as_img(reduce(arr, 'b c h w -> h (b w)', 'max'))
# utils.display_array_as_img(reduce(arr, 'b c h w -> h (b w)', 'min'))
# utils.display_array_as_img(reduce(arr, 'b c h w -> h w', 'min'))

# arr2 = rearrange(arr[0:2], 'b c h w -> c h (b w)')
# utils.display_array_as_img(rearrange(arr2, 'c (h1 h2) w -> c h2 (h1 w)', h1=2))


#utils.display_array_as_img(rearrange(arr, '(b1 b2) c h w -> c (b1 w) (b2 h)', b1=2))

#utils.display_array_as_img(reduce(arr, "(b1 b2) c (h h2) (w w2) -> c (b1 h) (b2 w)", "max", h2=2, w2=2, b1=2))


def einsum_trace(mat: np.ndarray):
    """
    Returns the same as `np.trace`.
    """
    return np.einsum('i i -> ', mat)

def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    """
    return np.einsum('i j,j', mat, vec)

def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    """
    return np.einsum('i j,j k', mat1, mat2)

def einsum_inner(vec1, vec2):
    """
    Returns the same as `np.inner`.
    """
    return np.einsum('i,i', vec1, vec2)

def einsum_outer(vec1, vec2):
    """
    Returns the same as `np.outer`.
    """
    return np.einsum('i,j -> ij', vec1, vec2)

utils.test_einsum_trace(einsum_trace)
utils.test_einsum_mv(einsum_mv)
utils.test_einsum_mm(einsum_mm)
utils.test_einsum_inner(einsum_inner)
utils.test_einsum_outer(einsum_outer)


