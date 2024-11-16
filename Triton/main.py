import triton
import torch
import numpy as np
import matplotlib.pyplot as plt

from matmul import matmul_triton, matmul_scalar
from config import *

@triton.testing.perf_report(config_2)
def benchmark(M: int, N: int, K: int, provider: str) -> None:
    quantiles = [0.5, 0.2, 0.8]
    if provider=='Torch-cpu':
        A = torch.randn((M, K), device='cpu')
        B = torch.randn((K, N), device='cpu')
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(A, B), quantiles=quantiles)
    elif provider=='Torch-gpu':
        A = torch.randn((M, K), device='cuda')
        B = torch.randn((K, N), device='cuda')
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(A, B), quantiles=quantiles)
    elif provider=='Triton':
        A = torch.randn((M, K), device='cuda')
        B = torch.randn((K, N), device='cuda')
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_triton(A, B), quantiles=quantiles)
    elif provider=='Scalar':
        A = torch.randn((M, K), device='cpu')
        B = torch.randn((K, N), device='cpu')
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_scalar(A, B), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

def main() -> None:
    benchmark.run(print_data=True, save_path='./benchmark')

if __name__=="__main__":
    main()