from torch import Tensor
import torch
import triton
import triton.language as tl

"""
v0 版本的向量化矩阵相乘, 且存在问题:
- K 必须是2的幂次, 且没分块, 同时还必须得是 tl.constexpr 类型
"""
@triton.jit
def matmul_kernel_v0(
        a_ptr, b_ptr, c_ptr, 
        M, N, K: tl.constexpr, 
        BM: tl.constexpr, BN: tl.constexpr, # BK: tl.constexpr
    ):
    pid = tl.program_id(axis=0)
    pid_num_Ci = tl.cdiv(M, BM)
    pid_num_Cj = tl.cdiv(N, BN)
    pid_Ci = pid // pid_num_Ci
    pid_Cj = pid % pid_num_Ci
 
    off_Ai = pid_Ci*BM + tl.arange(0, BM)
    off_Bj = pid_Cj*BN + tl.arange(0, BN)
    mask_Ai = off_Ai < M
    mask_Bj = off_Bj < N

    off_Aj = tl.arange(0, K)
    off_Bi = tl.arange(0, K)
    off_Aij = off_Ai[:, None]*K + off_Aj[None, :]
    off_Bij = off_Bi[:, None]*N + off_Bj[None, :]
    mask_Aij = mask_Ai[:, None] & (off_Aj<K)[None, :]
    mask_Bij = mask_Bj[None, :] & (off_Bi<K)[:, None]

    off_Cij = off_Ai[:, None]*N + off_Bj[None, :]
    mask_Cij = mask_Ai[:, None] & mask_Bj[None, :]

    A_batch = tl.load(off_Aij+a_ptr, mask_Aij)
    B_batch = tl.load(off_Bij+b_ptr, mask_Bij)
    C_batch = tl.dot(A_batch, B_batch)
    tl.store(off_Cij+c_ptr, C_batch, mask_Cij)

def matmul_triton_v0(A: Tensor, B:Tensor) -> Tensor:
    assert A.is_cuda and B.is_cuda
    M, K = A.shape
    K_1, N = B.shape
    assert K==K_1, "Shape Error!"
    assert (K > 0 and (K & (K - 1)) == 0), "K must be a power of 2!"
    C = torch.empty((M, N), device='cuda')
    grid = lambda meta: (
        (triton.cdiv(M, meta['BM'])*(triton.cdiv(N, meta['BN']))), 
    )
    matmul_kernel_v0[grid](A, B, C, M, N, K, 16, 16)
    return C

if __name__ == '__main__':
    a = torch.randn((45, 64), device='cuda')
    b = torch.randn((64, 38), device='cuda')
    ans = a @ b
    res = matmul_triton_v0(a, b)
    print(torch.max(torch.abs(res-ans)))
