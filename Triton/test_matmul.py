import pytest
import torch
from torch import Tensor
from matmul_manual import matmul_scalar
from matmul_triton import matmul_triton
from util import matmul_verify
from tqdm import trange

TRY_NUM = 10
MNK_RANGE_MIN = 16 # 由于 tl.dot 会要求矩阵的行数和列数不小于16, 故添加此约束
MNK_RANGE_MAX = 100

def test_matmul_tirton():
    for t in trange(TRY_NUM):
        M, K, N = torch.randint(MNK_RANGE_MIN, MNK_RANGE_MAX, (3,))
        A = torch.randn((M, K), device='cuda')
        B = torch.randn((K, N), device='cuda')
        res = matmul_triton(A, B)
        matmul_verify(A, B, res)

def test_matmul_scalar():
    for t in trange(TRY_NUM):
        M, K, N = torch.randint(MNK_RANGE_MIN, MNK_RANGE_MAX, (3,))
        A = torch.randn((M, K))
        B = torch.randn((K, N))
        res = matmul_scalar(A, B)
        assert matmul_verify(A, B, res)

if __name__ == '__main__':
    test_matmul_tirton()