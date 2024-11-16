from torch import Tensor
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr, 
        M, N, K, 
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
    ):
    """
    v1 版本的向量化矩阵相乘,
    """
    pid = tl.program_id(axis=0)
    pid_num_Ci = tl.cdiv(M, BM) # 行方向的块数
    pid_num_Cj = tl.cdiv(N, BN) # 列方向的块数
    pid_Ci = pid // pid_num_Cj # 目前块所在行数
    pid_Cj = pid % pid_num_Cj # 目前块所在列数
 
    off_Ai = pid_Ci*BM + tl.arange(0, BM)
    off_Bj = pid_Cj*BN + tl.arange(0, BN)
    mask_Ai = off_Ai < M
    mask_Bj = off_Bj < N

    C_batch = tl.zeros((BM, BN), dtype=tl.float32)
    for b in tl.range(0, K, BK):
        off_Aj = b + tl.arange(0, BK)
        off_Bi = b + tl.arange(0, BK)
        off_Aij = off_Ai[:, None]*K + off_Aj[None, :]
        off_Bij = off_Bi[:, None]*N + off_Bj[None, :]
        mask_Aij = mask_Ai[:, None] & (off_Aj<K)[None, :]
        mask_Bij = mask_Bj[None, :] & (off_Bi<K)[:, None]
        
        A_batch = tl.load(off_Aij+a_ptr, mask_Aij)
        B_batch = tl.load(off_Bij+b_ptr, mask_Bij)
        C_batch += tl.dot(A_batch, B_batch)
    
    off_Cij = off_Ai[:, None]*N + off_Bj[None, :]
    mask_Cij = mask_Ai[:, None] & mask_Bj[None, :]

    tl.store(off_Cij+c_ptr, C_batch, mask_Cij)

def matmul_triton(A: Tensor, B:Tensor) -> Tensor:
    assert A.is_cuda and B.is_cuda
    M, K = A.shape
    K_1, N = B.shape
    assert K==K_1, "Shape Error!"
    C = torch.empty((M, N), device='cuda')
    grid = lambda meta: (
        (triton.cdiv(M, meta['BM'])*(triton.cdiv(N, meta['BN']))), 
    )
    matmul_kernel[grid](A, B, C, M, N, K, 16, 16, 16)
    return C
