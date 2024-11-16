import triton

# 带标量
config_with_scalar = triton.testing.Benchmark(
    x_names=['M', 'N', 'K'],
    x_vals=range(16, 64, 8),
    line_arg='provider',
    line_vals=['Torch-cpu', 'Torch-gpu', 'Triton', 'Scalar'],
    line_names=['Torch-cpu', 'Torch-gpu', 'Triton', 'Scalar'],
    styles=[
        ('orange', '-.'), 
        ('blue', '-.'), 
        ('green', '-.'),
        ('red', '-.'),
    ],
    ylabel="TFLOPS",
    plot_name="Matmul Performance", 
    args={},
)

config_0 = triton.testing.Benchmark(
    x_names=['M', 'N', 'K'],
    x_vals=range(128, 4096, 128),
    line_arg='provider',
    line_vals=['Torch-gpu', 'Triton'],
    line_names=['Torch-gpu', 'Triton'],
    styles=[
        ('orange', '-.'), 
        ('blue', '-.'), 
    ],
    ylabel="TFLOPS",
    plot_name="Matmul Performance", 
    args={},
)

# 
config_1 = triton.testing.Benchmark(
    x_names=['M', 'N', 'K'],
    x_vals=range(128, 4096, 128),
    line_arg='provider',
    line_vals=['Torch-cpu', 'Torch-gpu', 'Triton'],
    line_names=['Torch-cpu', 'Torch-gpu', 'Triton'],
    styles=[
        ('orange', '-.'), 
        ('blue', '-.'), 
        ('green', '-.'),
    ],
    ylabel="TFLOPS",
    plot_name="Matmul Performance", 
    args={},
)

config_2 = triton.testing.Benchmark(
    x_names=['M', 'N', 'K'],
    x_vals=range(1024, 4096, 128),
    line_arg='provider',
    line_vals=['Torch-cpu', 'Torch-gpu', 'Triton'],
    line_names=['Torch-cpu', 'Torch-gpu', 'Triton'],
    styles=[
        ('orange', '-.'), 
        ('blue', '-.'), 
        ('green', '-.'),
    ],
    ylabel="TFLOPS",
    plot_name="Matmul Performance", 
    args={},
)