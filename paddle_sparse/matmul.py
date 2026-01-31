from __future__ import annotations

from typing import Optional, Tuple

import paddle
from paddle import Tensor

from paddle_sparse.tensor import SparseTensor


def spmm_sum(src: SparseTensor, other: paddle.Tensor) -> paddle.Tensor:
    rowptr, col, value = src.csr()
    
    if value is not None:
        value = value.astype(other.dtype)
    
    # Use scatter operations for aggregation
    row = src.storage.row()
    out = other.index_select(-2, col)
    
    if value is not None:
        out = out * value.unsqueeze(-1)
    
    try:
        from paddle_scatter import scatter_add
        out = scatter_add(out, row, dim=-2, dim_size=src.size(0))
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
        out = scatter_add_fallback(out, row, dim=-2, dim_size=src.size(0))
    
    return out


def spmm_add(src: SparseTensor, other: paddle.Tensor) -> paddle.Tensor:
    return spmm_sum(src, other)


def spmm_mean(src: SparseTensor, other: paddle.Tensor) -> paddle.Tensor:
    rowptr, col, value = src.csr()
    
    if value is not None:
        value = value.astype(other.dtype)
    
    row = src.storage.row()
    out = other.index_select(-2, col)
    
    if value is not None:
        out = out * value.unsqueeze(-1)
    
    try:
        from paddle_scatter import scatter_mean
        out = scatter_mean(out, row, dim=-2, dim_size=src.size(0))
    except ImportError:
        def scatter_mean_fallback(src, index, dim, dim_size):
            if dim == -2:
                dim = src.ndim - 2
            out_shape = list(src.shape)
            out_shape[dim] = dim_size
            out = paddle.zeros(out_shape, dtype=src.dtype)
            count = paddle.zeros([dim_size], dtype=src.dtype)
            
            for i in range(index.shape[0]):
                idx = index[i].item()
                count[idx] += 1.0
                out[idx] += src[i]
            
            count = paddle.maximum(count, paddle.ones_like(count))
            out = out / count.unsqueeze(-1)
            return out
        out = scatter_mean_fallback(out, row, dim=-2, dim_size=src.size(0))
    
    return out


def spmm_min(src: SparseTensor, other: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
    rowptr, col, value = src.csr()
    
    if value is not None:
        value = value.astype(other.dtype)
    
    row = src.storage.row()
    out = other.index_select(-2, col)
    
    if value is not None:
        out = out * value.unsqueeze(-1)
    
    try:
        from paddle_scatter import scatter_min
        out, argmin = scatter_min(out, row, dim=-2, dim_size=src.size(0))
    except ImportError:
        def scatter_min_fallback(src, index, dim, dim_size):
            if dim == -2:
                dim = src.ndim - 2
            out_shape = list(src.shape)
            out_shape[dim] = dim_size
            out = paddle.full(out_shape, float('inf'), dtype=src.dtype)
            argmin = paddle.zeros([dim_size], dtype='int64')
            
            for i in range(index.shape[0]):
                idx = index[i].item()
                mask = src[i] < out[idx]
                if paddle.any(mask):
                    out[idx] = paddle.where(mask, src[i], out[idx])
                    argmin[idx] = i
            
            return out, argmin
        out, argmin = scatter_min_fallback(out, row, dim=-2, dim_size=src.size(0))
    
    return out, argmin


def spmm_max(src: SparseTensor, other: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
    rowptr, col, value = src.csr()
    
    if value is not None:
        value = value.astype(other.dtype)
    
    row = src.storage.row()
    out = other.index_select(-2, col)
    
    if value is not None:
        out = out * value.unsqueeze(-1)
    
    try:
        from paddle_scatter import scatter_max
        out, argmax = scatter_max(out, row, dim=-2, dim_size=src.size(0))
    except ImportError:
        def scatter_max_fallback(src, index, dim, dim_size):
            if dim == -2:
                dim = src.ndim - 2
            out_shape = list(src.shape)
            out_shape[dim] = dim_size
            out = paddle.full(out_shape, float('-inf'), dtype=src.dtype)
            argmax = paddle.zeros([dim_size], dtype='int64')
            
            for i in range(index.shape[0]):
                idx = index[i].item()
                mask = src[i] > out[idx]
                if paddle.any(mask):
                    out[idx] = paddle.where(mask, src[i], out[idx])
                    argmax[idx] = i
            
            return out, argmax
        out, argmax = scatter_max_fallback(out, row, dim=-2, dim_size=src.size(0))
    
    return out, argmax


def spmm(src: SparseTensor, other: paddle.Tensor, reduce: str = "sum") -> paddle.Tensor:
    if reduce == 'sum' or reduce == 'add':
        return spmm_sum(src, other)
    elif reduce == 'mean':
        return spmm_mean(src, other)
    elif reduce == 'min':
        return spmm_min(src, other)[0]
    elif reduce == 'max':
        return spmm_max(src, other)[0]
    else:
        raise ValueError


def spspmm_sum(src: SparseTensor, other: SparseTensor) -> SparseTensor:
    try:
        rowA, colA, valueA = src.coo()
        rowB, colB, valueB = other.coo()
        
        A_coo = paddle.sparse.sparse_coo_tensor(
            paddle.stack([rowA, colA]), valueA, src.sparse_sizes()
        )
        B_coo = paddle.sparse.sparse_coo_tensor(
            paddle.stack([rowB, colB]), valueB, other.sparse_sizes()
        )
        
        C_coo = paddle.sparse.mm(A_coo, B_coo)
        
        indices = C_coo.indices()
        row, col = indices[0], indices[1]
        value = C_coo.values() if C_coo.values() is not None else None
        
        return SparseTensor(
            row=row,
            col=col,
            value=value,
            sparse_sizes=(C_coo.shape[0], C_coo.shape[1]),
            is_sorted=True,
            trust_data=True,
        )
    except:
        raise NotImplementedError("Manual sparse-sparse multiplication not yet implemented")


def spspmm_add(src: SparseTensor, other: SparseTensor) -> SparseTensor:
    return spspmm_sum(src, other)


def spspmm(src: SparseTensor, other: SparseTensor, reduce: str = "sum") -> SparseTensor:
    if reduce == 'sum' or reduce == 'add':
        return spspmm_sum(src, other)
    elif reduce == 'mean' or reduce == 'min' or reduce == 'max':
        raise NotImplementedError
    else:
        raise ValueError


def matmul(src, other, reduce="sum"):
    if isinstance(other, paddle.Tensor):
        return spmm(src, other, reduce)
    elif isinstance(other, SparseTensor):
        return spspmm(src, other, reduce)
    raise ValueError


# Add methods to SparseTensor class
SparseTensor.spmm = lambda self, other, reduce="sum": spmm(self, other, reduce)
SparseTensor.spspmm = lambda self, other, reduce="sum": spspmm(self, other, reduce)
SparseTensor.matmul = lambda self, other, reduce="sum": matmul(self, other, reduce)
SparseTensor.__matmul__ = lambda self, other: matmul(self, other, 'sum')