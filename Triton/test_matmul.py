import pytest
import torch
from torch import Tensor
from matmul_manual import matmul_scalar
from matmul_triton import matmul_triton
from util import matmul_verify

TRY_NUM = 10
MNK_RANGE_MIN = 16 # 由于 tl.dot 会要求矩阵的行数和列数不小于16, 故添加此约束
MNK_RANGE_MAX = 100

def test_matmul_tirton():
    for t in range(TRY_NUM):
        M, N, K = torch.randint(MNK_RANGE_MIN, MNK_RANGE_MAX, (3,))
        A = torch.randn((M, N), device='cuda')
        B = torch.randn((N, K), device='cuda')
        res = matmul_triton(A, B)
        print(A, B, res)
        matmul_verify(A, B, res)

def test_matmul_scalar():
    for t in range(TRY_NUM):
        M, N, K = torch.randint(MNK_RANGE_MIN, MNK_RANGE_MAX, (3,))
        A = torch.randn((M, N))
        B = torch.randn((N, K))
        res = matmul_scalar(A, B)
        assert matmul_verify(A, B, res)