import pytest
import torch
from torch import Tensor
from matmul_manual import matmul_scalar

TRY_NUM = 10
MNK_RANGE_MAX = 100

def matmul_verify(A: Tensor, B: Tensor, res: Tensor, tol=1E-4) -> bool:
    (M, N) = A.shape
    (N, K) = B.shape
    assert (M, K) == res.shape, f"Shape incompatible! One has shape {(M, K)} and the other has shape {res.shape}."
    ans = A @ B
    max_diff = torch.max(torch.abs(ans-res))
    return max_diff<tol

def test_matmul_scalar():
    for t in range(TRY_NUM):
        M, N, K = torch.randint(1, MNK_RANGE_MAX, (3,))
        A = torch.randn((M, N))
        B = torch.randn((N, K))
        res = matmul_scalar(A, B)
        assert matmul_verify(A, B, res)

def test_matul_triton():
    A = Tensor([
        [1, 2],
        [4, 5],
        [6, 7]
    ])

    B = Tensor([
        [3, 4, 5, 6],
        [-1, 9, 8, 0]
    ])

    ans = A@B
    res = torch.empty_like(ans)
    
    assert ans == res
