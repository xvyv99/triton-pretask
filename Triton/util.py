import torch
from torch import Tensor

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from typing import TypeAlias
from matplotlib.axes import Axes

def matmul_verify(A: Tensor, B: Tensor, res: Tensor, tol=1E-4) -> None:
    """
    验证矩阵相乘结果的正确性
    """
    (M, K) = A.shape
    (K, N) = B.shape
    assert (M, N) == res.shape, f"Shape incompatible! One has shape {(M, N)} and the other has shape {res.shape}."

    ans = A @ B
    max_diff = torch.max(torch.abs(ans-res))
    if max_diff>tol:
        check_detail(res, ans)
        raise Exception(f"The maximum difference between matrix elements is {max_diff}, which exceeds the allowed threshold {tol:.2E}.")

def check_detail(res: Tensor, ans: Tensor, 
        i_batch_size=16, j_batch_size=16, tol=1E-4
    ) -> None:
    """
    用于可视化矩阵检验的结果
    """
    Point: TypeAlias = tuple[int, int]
    null_points: list[Point] = []
    same_points: list[Point] = []

    def draw_points(ax: Axes, points: list[Point], color: str) -> None:
        for x, y in points:
            rect = Rectangle(
                (x, y), 1, 1, 
                facecolor=color,
            )
            ax.add_patch(rect)

    M, N = res.shape
    for i in range(M):
        for j in range(N):
            if res[i, j]==0:
                null_points.append((j, i))
            elif abs(res[i, j]-ans[i, j])<tol:
                same_points.append((j, i))

    fig, ax = plt.subplots()
    rect = Rectangle(
        (0, 0), N, M, 
        facecolor='#ff7f0e',
        alpha=0.5
    )
    ax.add_patch(rect)
    
    draw_points(ax, null_points, "#1f77b4")
    draw_points(ax, same_points, "#2ca02c")

    legend_elements = [
        Patch(facecolor='#1f77b4', label='Zero element'),
        Patch(facecolor='#2ca02c', label='Correct element'),
        Patch(facecolor='#ff7f0e', label='Wrong element')
    ]
    ax.legend(
        bbox_to_anchor=(1.05, 1), 
        loc='upper left', 
        handles=legend_elements
    )
    ax.set_xlim(0, N)
    ax.set_ylim(M, 0)

    ax.set_xticks(torch.arange(0, N, j_batch_size))
    ax.set_yticks(torch.arange(0, M, i_batch_size))


    ax.set_xlabel("column direction")
    ax.set_ylabel("row direction")
    ax.set_title(f"{M}x{N} Matrix Check Result")
    ax.grid(True)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('matrix_check.png', dpi=300)
    plt.close