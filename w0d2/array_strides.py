import torch as t
import utils
from collections import namedtuple

TestCase = namedtuple("TestCase", ["output", "size", "stride"])

test_input = t.tensor(
    [[0, 1, 2, 3, 4], 
    [5, 6, 7, 8, 9], 
    [10, 11, 12, 13, 14], 
    [15, 16, 17, 18, 19]], dtype=t.float
)

test_cases = [
    TestCase(
        output=t.tensor([0, 1, 2, 3]), 
        size=(4,), 
        stride=(1,)),
    TestCase(
        output=t.tensor([0, 1, 2, 3, 4]), 
        size=(5,), 
        stride=(1,)),
    TestCase(
        output=t.tensor([0, 5, 10, 15]), 
        size=(4,), 
        stride=(5,)),
    TestCase(
        output=t.tensor([[0, 1, 2], [5, 6, 7]]), 
        size=(2,3), 
        stride=(5,1)),
    #4
    TestCase(
        output=t.tensor([[0, 1, 2], [10, 11, 12]]), 
        size=(2,3), 
        stride=(10,1)),
    # 5
    TestCase(
        output=t.tensor([[0, 0, 0], [11, 11, 11]]), 
        size=(2,3),
        stride=(11,0)),    
    #6
    TestCase(
        output=t.tensor([0, 6, 12, 18]), 
        size=(4,), 
        stride=(6,)),
    #7
    TestCase(
        output=t.tensor([[[0, 1, 2]], [[9, 10, 11]]]), 
        size=(2,1,3), 
        stride=(9,0,1)),
    #8
    TestCase(
        output=t.tensor([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]], [[[12, 13], [14, 15]], [[16, 17], [18, 19]]]]),
        size=(2,2,2,2),
        stride=(12,4,2,1))
]
for (i, case) in enumerate(test_cases):
    if (case.size is None) or (case.stride is None):
        print(f"Test {i} failed: attempt missing.")
    else:
        actual = test_input.as_strided(size=case.size, stride=case.stride)
        if (case.output != actual).any():
            print(f"Test {i} failed:")
            print(f"Expected: {case.output}")
            print(f"Actual: {actual} ")
        else:
            print(f"Test {i} passed! ")


def as_strided_trace(mat: t.Tensor) -> t.Tensor:
    '''
    Returns the same as `torch.trace`, using only `as_strided` and `sum` methods.
    '''
    n = min(mat.shape)
    return t.as_strided(
        mat,
        size=(n,),
        stride=(n+1,)
    ).sum()


utils.test_trace(as_strided_trace)

def as_strided_mv(mat: t.Tensor, vec: t.Tensor) -> t.Tensor:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    n, m = mat.shape
    k = vec.shape[0]
    orig_stride = vec.stride(0)
    assert m == k, f'{m} == {k}'
    vec_s = t.as_strided(
            vec,
            size=(n,m),
            stride=(0,orig_stride)
        )
    res = (
        mat * vec_s
    ).sum(dim=1).T
    return res

utils.test_mv(as_strided_mv)
utils.test_mv2(as_strided_mv)

def as_strided_mm(matA: t.Tensor, matB: t.Tensor) -> t.Tensor:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    n,m = matA.shape
    k,p = matB.shape
    assert m == k, f'{m} == {k}'
    matAext = t.as_strided(
        matA,
        size=(n, m, p),
        stride=(matA.stride(0), matA.stride(1), 0)
    )
    matBext = t.as_strided(
        matB,
        size=(n, m, p),
        stride=(0, matB.stride(0), matB.stride(1))
    )
    return (matAext * matBext).sum(dim=1)




utils.test_mm(as_strided_mm)
utils.test_mm2(as_strided_mm)

print('exit')
