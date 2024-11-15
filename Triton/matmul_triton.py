from torch import Tensor
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, B0: tl.constexpr, B1: tl.constexpr):
    pass

def matual_triton(A: Tensor, B:Tensor) -> Tensor:
    return a@b

if __name__ == '__main__':
    a = torch.randint(0, 10, (7, 8))
    b = torch.randint(0, 10, (8, 10))
    ans = a @ b
    res = matual_triton(a, b)
    print(torch.max(torch.abs(res-ans)))
