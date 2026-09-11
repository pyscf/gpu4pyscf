# cupyx/__init__.py
# Fake cupyx package root

import numpy as _np
import cupy as _cupy


def empty_pinned(shape, dtype=_np.float64, order='C'):
    '''Equivalent of cupyx.empty_pinned: allocate an uninitialized host
    ndarray backed by pinned (page-locked) memory, for fast host<->device
    transfers. Backed by cupy.cuda.alloc_pinned_memory, the same pinned
    allocator already used elsewhere in the shim (see
    gpu4pyscf/lib/cupy_helper.py:pin_memory and cupy/cuda.py
    :alloc_pinned_memory).
    '''
    shape = tuple(int(s) for s in shape) if isinstance(shape, (tuple, list)) else (int(shape),)
    dtype = _np.dtype(dtype)
    nbytes = int(_np.prod(shape)) * dtype.itemsize if shape else dtype.itemsize
    mem = _cupy.cuda.alloc_pinned_memory(nbytes)
    return _np.ndarray(shape, dtype=dtype, buffer=mem, order=order)


def zeros_pinned(shape, dtype=_np.float64, order='C'):
    '''Equivalent of cupyx.zeros_pinned: like empty_pinned but zero-filled.'''
    out = empty_pinned(shape, dtype=dtype, order=order)
    out.fill(0)
    return out
