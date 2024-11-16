import torch
from torch import Tensor

def matmul_scalar(A: Tensor, B: Tensor) -> Tensor:
    (M, K) = A.shape
    (K, N) = B.shape
    res = torch.zeros((M, N))
    for i in range(0, M):
        for j in range(0, N):
            for k in range(0, K):
                res[i, j] += A[i, k] * B[k, j]
    return res
