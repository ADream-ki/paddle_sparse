from __future__ import annotations

from typing import Optional

import paddle
from paddle import Tensor

from paddle_sparse.tensor import SparseTensor


def spmm(index: Tensor, value: Tensor, m: int, n: int, matrix: Tensor) -> Tensor:
    assert n == matrix.shape[-2]

    row, col = index[0], index[1]
    matrix = matrix if matrix.ndim > 1 else matrix.unsqueeze(-1)

    out = matrix.index_select(-2, col)
    out = out * value.unsqueeze(-1)
    
    try:
        from paddle_scatter import scatter_add
        out = scatter_add(out, row, dim=-2, dim_size=m)
    except ImportError:
        def scatter_add_fallback(src, index, dim, dim_size):
            if dim == -2:
                dim = src.ndim - 2
            out_shape = list(src.shape)
            out_shape[dim] = dim_size
            out = paddle.zeros(out_shape, dtype=src.dtype)
            for i in range(index.shape[0]):
                idx = index[i].item()
                out[idx] += src[i]
            return out
        out = scatter_add_fallback(out, row, dim=-2, dim_size=m)

    return out


def spspmm(indexA: Tensor, valueA: Tensor, indexB: Tensor, valueB: Tensor, 
           m: int, k: int, n: int, coalesced: bool = False):
    A = SparseTensor(row=indexA[0], col=indexA[1], value=valueA,
                     sparse_sizes=(m, k), is_sorted=not coalesced)
    B = SparseTensor(row=indexB[0], col=indexB[1], value=valueB,
                     sparse_sizes=(k, n), is_sorted=not coalesced)

    from paddle_sparse.matmul import matmul
    C = matmul(A, B)
    row, col, value = C.coo()

    return paddle.stack([row, col], axis=0), value