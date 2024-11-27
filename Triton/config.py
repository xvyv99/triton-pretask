import triton

# 默认配置
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

# 带 Torch-cpu 矩阵乘法的配置
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

# 带 Torch-cpu 矩阵乘法的配置, 且关注矩阵较大的情形
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

# 带标量矩阵乘法的配置, 由于我写的标量矩阵乘法太慢了, 故缩小迭代范围
config_3 = triton.testing.Benchmark(
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
