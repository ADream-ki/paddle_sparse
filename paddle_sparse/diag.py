from __future__ import annotations

from typing import Optional

import paddle
from paddle import Tensor

from paddle_sparse.storage import SparseStorage
from paddle_sparse.tensor import SparseTensor


def remove_diag(src: SparseTensor, k: int = 0) -> SparseTensor:
    row, col, value = src.coo()
    inv_mask = row != col if k == 0 else row != (col - k)
    new_row, new_col = row[inv_mask], col[inv_mask]

    if value is not None:
        value = value[inv_mask]

    storage = SparseStorage(
        row=new_row, 
        rowptr=None, 
        col=new_col, 
        value=value,
        sparse_sizes=src.sparse_sizes(), 
        rowcount=None,
        colptr=None, 
        colcount=None, 
        csr2csc=None,
        csc2csr=None, 
        is_sorted=True
    )
    return src.from_storage(storage)


def set_diag(src: SparseTensor, values: Optional[Tensor] = None, k: int = 0) -> SparseTensor:
    src = remove_diag(src, k=k)
    row, col, value = src.coo()

    m, n = src.size(0), src.size(1)
    if k >= 0:
        diag_size = min(m, n - k)
        diag_row = paddle.arange(diag_size, dtype='int64')
        diag_col = diag_row + k
    else:
        diag_size = min(m + k, n)
        diag_col = paddle.arange(diag_size, dtype='int64')
        diag_row = diag_col - k

    new_row = paddle.concat([row, diag_row], axis=0)
    new_col = paddle.concat([col, diag_col], axis=0)

    new_value: Optional[Tensor] = None
    if value is not None or values is not None:
        if value is not None:
            if values is not None:
                diag_values = values
            else:
                diag_values = paddle.ones([diag_size], dtype=value.dtype)
            new_value = paddle.concat([value, diag_values], axis=0)
        else:
            if values is not None:
                new_value = values
            else:
                new_value = paddle.ones([diag_size], dtype='float32')

    storage = SparseStorage(
        row=new_row, 
        rowptr=None, 
        col=new_col, 
        value=new_value,
        sparse_sizes=src.sparse_sizes(), 
        rowcount=None,
        colptr=None, 
        colcount=None, 
        csr2csc=None,
        csc2csr=None, 
        is_sorted=False
    )
    return src.from_storage(storage).coalesce()


def fill_diag(src: SparseTensor, fill_value: float, k: int = 0) -> SparseTensor:
    m, n = src.size(0), src.size(1)
    if k >= 0:
        diag_size = min(m, n - k)
    else:
        diag_size = min(m + k, n)

    value = src.storage.value()
    if value is not None:
        diag_values = paddle.full([diag_size], fill_value, dtype=value.dtype)
    else:
        diag_values = paddle.full([diag_size], fill_value, dtype='float32')
        
    return set_diag(src, diag_values, k)


def get_diag(src: SparseTensor) -> Tensor:
    row, col, value = src.coo()

    if value is None:
        value = paddle.ones(row.shape[0], dtype='float32')

    sizes = list(value.shape)
    sizes[0] = min(src.size(0), src.size(1))

    out = paddle.zeros(sizes, dtype=value.dtype)

    mask = row == col
    out[row[mask]] = value[mask]

    return out


SparseTensor.remove_diag = lambda self, k=0: remove_diag(self, k)
SparseTensor.set_diag = lambda self, values=None, k=0: set_diag(self, values, k)
SparseTensor.fill_diag = lambda self, fill_value, k=0: fill_diag(self, fill_value, k)
SparseTensor.get_diag = lambda self: get_diag(self)