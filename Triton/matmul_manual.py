import torch
from torch import Tensor

def matmul_scalar(A: Tensor, B: Tensor) -> Tensor:
    (M, N) = A.shape
    (N, K) = B.shape
    res = torch.zeros((M, K))
    for i in range(0, M):
        for j in range(0, K):
            for k in range(0, N):
                res[i, j] += A[i, k] * B[k, j]
    return res