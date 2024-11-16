import triton
import torch
import pytest

from util import print_header
from matmul import matmul_triton, matmul_scalar
from config import *

# ------------------------------------------------------------------------------
# User Configurable Variables
# ------------------------------------------------------------------------------

benchmark_config = config_2 # 更改此变量来更换配置

# ------------------------------------------------------------------------------
# BenchMark Function
# ------------------------------------------------------------------------------

@triton.testing.perf_report(benchmark_config)
def benchmark(M: int, N: int, K: int, provider: str) -> None:
    quantiles = [0.5, 0.2, 0.8]

    provider_dict = {
        'Torch-cpu': ('cpu', torch.matmul),
        'Torch-gpu': ('cuda', torch.matmul),
        'Triton': ('cuda', matmul_triton),
        'Scalar': ('cpu', matmul_scalar),
    }
    
    device, matmul_func = provider_dict[provider]
    A = torch.randn((M, K), device=device)
    B = torch.randn((K, N), device=device)
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_func(A, B), quantiles=quantiles)
    
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

def main() -> None:
    result = pytest.main()
        
    if result == pytest.ExitCode.OK:
        print()
        print_header("benchmark session starts")
        benchmark.run(print_data=True, save_path='./benchmark')
        print_header("benchmark session finishs", style='green')

if __name__=="__main__":
    main()