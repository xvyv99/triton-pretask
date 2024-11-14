from torch import Tensor

def matmul_scalar(A: Tensor, B: Tensor) -> Tensor:
    return A@B