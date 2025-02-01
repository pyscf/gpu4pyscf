# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import cupy
import numpy as np
from gpu4pyscf.__config__ import _p2p_access

__all__ = ['p2p_transfer', 'copy_array']

def p2p_transfer(a, b):
    ''' If the direct P2P data transfer is not available, transfer data via CPU memory
    '''
    if a.device == b.device:
        a[:] = b
    elif _p2p_access:
        a[:] = b
        '''
    elif a.strides == b.strides and a.flags.c_contiguous and a.dtype == b.dtype:
        # cupy supports a direct copy from different devices without p2p. See also
        # https://github.com/cupy/cupy/blob/v13.3.0/cupy/_core/_routines_indexing.pyx#L48
        # https://github.com/cupy/cupy/blob/v13.3.0/cupy/_core/_routines_indexing.pyx#L1015
        a[:] = b
        '''
    else:
        copy_array(b, a)
    return a

def find_contiguous_chunks(shape, h_strides, d_strides):
    """
    Find the largest contiguous chunk size based on strides and shape.
    """
    chunk_shape = []
    chunk_size = 1
    for dim, h_stride, d_stride in zip(reversed(shape), reversed(h_strides), reversed(d_strides)):
        if h_stride == chunk_size and d_stride == chunk_size:
            chunk_shape.append(dim)
            chunk_size *= dim
        else:
            break
    chunk_shape = tuple(reversed(chunk_shape))
    return chunk_shape, chunk_size

def copy_array(src_view, out=None):
    ''' Copy cupy/numpy array to cupy array if out is None
        Copy cupy/numpy array to cupy/numpy array (out)
    '''
    if out is None:
        out = cupy.empty_like(src_view)
    else:
        # Ensure both arrays have the same shape
        if src_view.shape != out.shape:
            raise ValueError("Host and device views must have the same shape.")
    return _copy_array(src_view, out)

def _copy_array(src_view, dst_view):
    ''' Copy data from cupy/numpy array to another cupy/numpy array
    Check memory layout, then copy memory chunks by cupy.cuda.runtime.memcpy
    '''
    if src_view.nbytes == 0:
        return dst_view
    
    shape = src_view.shape
    itemsize = src_view.itemsize
    strides_src = [stride // itemsize for stride in src_view.strides]
    strides_dst = [stride // itemsize for stride in dst_view.strides]

    # Find the largest contiguous chunk
    chunk_shape, chunk_size = find_contiguous_chunks(shape, strides_src, strides_dst)

    if isinstance(src_view, cupy.ndarray):
        src_data_ptr = src_view.data.ptr
    else:
        src_data_ptr = src_view.ctypes.data

    if isinstance(dst_view, cupy.ndarray):
        dst_data_ptr = dst_view.data.ptr
    else:
        dst_data_ptr = dst_view.ctypes.data

    if isinstance(src_view, cupy.ndarray) and isinstance(dst_view, cupy.ndarray):
        kind = cupy.cuda.runtime.memcpyDeviceToDevice
    elif isinstance(src_view, cupy.ndarray) and isinstance(dst_view, np.ndarray):
        kind = cupy.cuda.runtime.memcpyDeviceToHost
    elif isinstance(src_view, np.ndarray) and isinstance(dst_view, cupy.ndarray):
        kind = cupy.cuda.runtime.memcpyHostToDevice
    else:
        raise NotImplementedError
        
    assert len(chunk_shape) > 0

    # Transfer data chunk-by-chunk
    outer_dims = shape[:-len(chunk_shape)]
    for outer_index in np.ndindex(*outer_dims):
        # Compute offsets for the current outer slice
        src_offset = sum(outer_index[i] * strides_src[i] for i in range(len(outer_dims)))
        dst_offset = sum(outer_index[i] * strides_dst[i] for i in range(len(outer_dims)))
        # Perform the memcpy for the contiguous chunk
        cupy.cuda.runtime.memcpy(
            dst_data_ptr + dst_offset * dst_view.itemsize,
            src_data_ptr + src_offset * src_view.itemsize,
            chunk_size * src_view.itemsize,
            kind
        )
    return dst_view
