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

import os
import sys
import functools
import ctypes
import numpy as np
import scipy.linalg
import cupy
import dpnp
import dpctl
import dpctl.memory as dpmem
from dpnp.dpnp_array import dpnp_array  # low-level constructor
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cutensor import contract
from gpu4pyscf.lib.onemkl_lapack import eigh as onemkl_eigh, cholesky as onemkl_cholesky
#from gpu4pyscf.lib.onemkl_lapack import eigh, cholesky  #NOQA
from gpu4pyscf.lib.memcpy import copy_array, p2p_transfer  #NOQA
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.lib.utils import load_library
from gpu4pyscf.lib.multi_gpu import lru_cache
from gpu4pyscf.__config__ import num_devices, _p2p_access

LMAX_ON_GPU = 7
DSOLVE_LINDEP = 1e-13

_kernel_registery = {}

libdpnp_helper = load_library('libcupy_helper')

def pin_memory(array):
    mem = dpctl.memory.MemoryUSMHost(array.nbytes)
    ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret

def release_gpu_stack():
    pass
    # cupy.cuda.runtime.deviceSetLimit(0x00, 128)

def print_mem_info():
    total_mem = cupy.cuda.get_total_memory()
    free_mem = cupy.cuda.get_free_memory()
    used_mem = total_mem - free_mem
    GB = 1024 * 1024 * 1024
    msg = f'mem_avail: {mem_avail/GB:.3f} GB, total_mem: {total_mem/GB:.3f} GB, used_mem: {used_mem/GB:.3f} GB,mem_limt: {mem_limit/GB:.3f} GB'
    print(msg)
    return msg

def get_avail_mem():
    return cupy.cuda.get_free_memory()

def concatenate(array_list):
    ''' Concatenate axis=0 only
    '''
    if _p2p_access:
        return cupy.concatenate(array_list)
    else:
        #array_list_cpu = [a.get() for a in array_list]
        n = sum([a.shape[0] for a in array_list])
        a0_shape = list(array_list[0].shape)
        out_shape = tuple([n] + a0_shape[1:])
        out = cupy.empty(out_shape)
        p0 = p1 = 0
        for a in array_list:
            p1 = p0 + a.shape[0]
            #out[p0:p1].set(a)
            copy_array(a, out[p0:p1])
            p0 = p1
        return out

def broadcast_to_devices():
    ''' Broadcast dpnp ndarray to all the devices, return a list of dpnp ndarray
    '''
    raise NotImplementedError

def reduce_to_device(array_list, inplace=False):
    ''' Reduce a list of ndarray in different devices to device 0
    TODO: reduce memory footprint, improve throughput
    '''
    assert len(array_list) == num_devices
    if num_devices == 1:
        return array_list[0]

    out_shape = array_list[0].shape
    for s in _streams:
        s.synchronize()

    if inplace:
        result = array_list[0]
    else:
        result = array_list[0].copy()

    # Transfer data chunk by chunk, reduce memory footprint,
    result = result.reshape(-1)
    for device_id, matrix in enumerate(array_list):
        if device_id == 0:
            continue

        assert matrix.device.id == device_id
        matrix = matrix.reshape(-1)
        blksize = 1024*1024*1024 // matrix.itemsize # 1GB
        for p0, p1 in lib.prange(0,len(matrix), blksize):
            result[p0:p1] += copy_array(matrix[p0:p1])
            #result[p0:p1] += cupy.asarray(matrix[p0:p1])
    return result.reshape(out_shape)

def device2host_2d(a_cpu, a_gpu, stream=None):
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    libdpnp_helper.async_d2h_2d(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        a_cpu.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(a_cpu.strides[0]),
        ctypes.cast(a_gpu.data.ptr, ctypes.c_void_p),
        ctypes.c_int(a_gpu.strides[0]),
        ctypes.c_int(a_gpu.shape[0]),
        ctypes.c_int(a_gpu.shape[1]))

# Define dpnp array with tag using Python class wrapper
class DPNPArrayWithTag:
    def __init__(self, array):
        if not isinstance(array, dpnp.ndarray):
            raise TypeError("Input must be a dpnp.ndarray")
        self.array = array
        self.metadata = {}

    def __getattr__(self, name):
        if name in self.metadata:
            return self.metadata[name]
        return getattr(self.array, name)  # forward to underlying dpnp.ndarray

    def __setattr__(self, name, value):
        if name in ("array", "metadata"):
            super().__setattr__(name, value)
        else:
            self.metadata[name] = value

    def __array__(self, dtype=None):
        """Allow conversion to array (useful for numpy/dpnp functions)"""
        if dtype is None:
            return self.array
        return self.array.astype(dtype)

    def __repr__(self):
        return f"DPNPArrayWithTag({repr(self.array)})"

    def __str__(self):
        return str(self.array)

    def __iter__(self):
        """Make the wrapper iterable like the underlying array"""
        return iter(self.array)

    def __getitem__(self, key):
        """Support indexing to enable unpacking"""
        return self.array[key]

    def __setitem__(self, key, value):
        """Support item assignment"""
        self.array[key] = value

    def __len__(self):
        """Support len() for iteration"""
        return len(self.array)

    def __add__(self, other):
        return self.array + other

    def __sub__(self, other):
        return self.array - other

    def __mul__(self, other):
        return self.array * other

    def __rmul__(self, other):
        return other * self.array

    def __radd__(self, other):
        return other + self.array

    def __rsub__(self, other):
        return other - self.array

    def __rtruediv__(self, other):
        return other / self.array

    def __rfloordiv__(self, other):
        return other // self.array

    def __rpow__(self, other):
        return other ** self.array

    # In-place operations
    def __iadd__(self, other):
        self.array += other
        return self

    def __isub__(self, other):
        self.array -= other
        return self

    def __imul__(self, other):
        self.array *= other
        return self

    def __itruediv__(self, other):
        self.array /= other
        return self

    def __ifloordiv__(self, other):
        self.array //= other
        return self

    def __ipow__(self, other):
        self.array **= other
        return self

    def __eq__(self, other):
        return self.array == other

    def __ne__(self, other):
        return self.array != other

    def __lt__(self, other):
        return self.array < other

    def __le__(self, other):
        return self.array <= other

    def __gt__(self, other):
        return self.array > other

    def __ge__(self, other):
        return self.array >= other

    def __neg__(self):
        return -self.array

    def __pos__(self):
        return +self.array

    def __abs__(self):
        return abs(self.array)

# Define numpy tagged array if needed for compatibility
class NPArrayWithTag:
    def __init__(self, array):
        if not isinstance(array, np.ndarray):
            raise TypeError("Input must be a numpy.ndarray")
        self.array = array
        self.__dict__.update(array.__dict__)

    def __getattr__(self, name):
        if name in self.__dict__.get('metadata', {}):
            return self.metadata[name]
        return getattr(self.array, name)

#@functools.wraps(lib.tag_array)
def tag_array(a, **kwargs):
    '''
    Tag a dpnp/numpy array or tuple of them with additional metadata.
    '''
    # Unwrap if a is already a wrapper
    if isinstance(a, DPNPArrayWithTag):
        base = a.array
    else:
        base = a

    if isinstance(base, dpnp.ndarray) or (isinstance(base, tuple) and isinstance(base[0], dpnp.ndarray)):
        t = DPNPArrayWithTag(dpnp.asarray(base))
        if isinstance(a, DPNPArrayWithTag):
            t.metadata.update(a.metadata) # Copy metadata if already tagged
        t.metadata.update(kwargs)
    elif isinstance(base, np.ndarray):
        t = np.asarray(a).view(lib.NPArrayWithTag)
        if isinstance(a, lib.NPArrayWithTag):
            t.__dict__.update(a.__dict__)
        t.__dict__.update(kwargs)
    else:
        raise TypeError(f"Unsupported input type: {type(a)}")

    return t

def asarray(a, **kwargs):
    '''
    Like cupy.asarray replacement using dpnp and dpctl.
    Transfers numpy arrays to device memory using dpnp.
    '''
    if isinstance(a, np.ndarray):
        allow_fast_transfer = kwargs.get('dtype', a.dtype) == a.dtype
        # a must be C-contiguous or F-contiguous
        if not a.flags.c_contiguous and not a.flags.f_contiguous:
            allow_fast_transfer = False
        if allow_fast_transfer:
            #ABB: cupy.empty_like(a) worked for CUPY where a was of type `numpy.ndarray`
            # but it wouldnt work for DPNP. Since the input is expected of dpnp.ndarray
            return dpnp.asarray(a)

    elif isinstance(a, DPNPArrayWithTag):
        a = a.array

    return dpnp.asarray(a, **kwargs)

ensure_numpy = dpnp.asnumpy

def to_dpnp(a):
    '''Convert numpy → dpnp (handles nested structures)'''
    if isinstance(a, lib.NPArrayWithTag):
        attrs = {k: to_dpnp(v) for k, v in a.__dict__.items()}
        return tag_array(dpnp.asarray(a), **attrs)
    if isinstance(a, np.ndarray):
        return dpnp.asarray(a)
    if isinstance(a, (tuple, list)):
        return type(a)(to_dpnp(x) for x in a)
    if isinstance(a, dict):
        return {k: to_dpnp(v) for k, v in a.items()}
    return a
    # '''Converts a numpy (and subclass) object to a dpnp object'''
    # if isinstance(a, lib.NPArrayWithTag):
    #     attrs = {k: to_dpnp(v) for k, v in a.__dict__.items()}
    #     return tag_array(cupy.asarray(a), **attrs)
    # if isinstance(a, np.ndarray):
    #     return cupy.asarray(a)
    # return a

########################################################################
# This section guards the return_cupy_array() section when a pyscf.cpu method
# is passed with DPNP arrays. It explicitly copies the array to numpy.ndarrat type
# Similar to cupy but it does implictly (hiding the transfer)

def _to_numpy(a):
    '''Convert GPU → NumPy (handles nested structures)'''
    if isinstance(a, cupy.ndarray):
        return cupy.asnumpy(a)
    if hasattr(a, 'asnumpy'):
        return a.asnumpy()
    if isinstance(a, (tuple, list)):
        return type(a)(_to_numpy(x) for x in a)
    if isinstance(a, dict):
        return {k: _to_numpy(v) for k, v in a.items()}
    return a

def _is_cpu_function(fn):
    '''Detect if function is from CPU PySCF (pyscf.scf.*)'''
    fn_module = fn.__module__ or ''
    if 'pyscf' in fn_module and 'gpu4pyscf' not in fn_module:
        return True
    if 'cpu' in (fn.__name__ or '').lower():
        return True
    return False

class _GPUMethodProxy:
    """
    Proxy that wraps an mf object so that any method call
    automatically converts numpy inputs back to dpnp.
    """
    def __init__(self, mf):
        object.__setattr__(self, '_mf', mf)

    def __getattr__(self, name):
        attr = getattr(object.__getattribute__(self, '_mf'), name)
        if callable(attr):
            @functools.wraps(attr)
            def wrapper(*args, **kwargs):
                # Convert numpy arrays back to dpnp before calling GPU method
                args = tuple(to_dpnp(a) for a in args)
                kwargs = {k: to_dpnp(v) for k, v in kwargs.items()}
                return attr(*args, **kwargs)
            return wrapper
        return attr

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, '_mf'), name, value)

def return_cupy_array(fn):
    '''Convert inputs for CPU functions, wrapping mf to auto-convert on callbacks'''
    is_cpu = _is_cpu_function(fn)

    @functools.wraps(fn)
    def filter_ret(*args, **kwargs):
        if is_cpu and args:
            # Wrap mf (first arg) so GPU method calls auto-convert numpy→dpnp
            mf_proxy = _GPUMethodProxy(args[0])
            args = (mf_proxy,) + tuple(_to_numpy(a) for a in args[1:])
            kwargs = {k: _to_numpy(v) for k, v in kwargs.items()}

        ret = fn(*args, **kwargs)

        if isinstance(ret, tuple):
            return tuple(to_dpnp(x) for x in ret)
        return to_dpnp(ret)
    return filter_ret

# ## How this works:
# ```
# Decorator:
#   1. Detects CPU function (pyscf.scf.uhf.get_occ)
#   2. Wraps mf with _GPUMethodProxy
#   3. Converts mo_energy, mo_coeff to numpy
#   4. Calls CPU function

# CPU get_occ:
#   - Works with numpy arrays ✓
#   - numpy.zeros_like(mo_energy) works ✓
#   - Calls mf.spin_square(numpy_arrays)
#        ↓
#   _GPUMethodProxy intercepts:
#   - Converts numpy → dpnp
#   - Calls real GPU spin_square(dpnp_arrays) ✓

########################################################################

def pack_tril(a, stream=None):
    ndim = a.ndim
    assert ndim in (2, 3)
    if ndim == 2:
        a = a[None]

    counts, n = a.shape[:2]
    if a.dtype != np.float64 or not a.flags.c_contiguous:
        idx = cupy.arange(n)
        mask = idx[:,None] >= idx
        a_tril = a[:,mask]
    else:
        if stream is None:
            stream = cupy.cuda.get_current_stream()
        a_tril = cupy.empty((counts, n*(n+1)//2), dtype=np.float64)
        err = libdpnp_helper.pack_tril(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            ctypes.cast(a_tril.data.ptr, ctypes.c_void_p),
            ctypes.cast(a.data.ptr, ctypes.c_void_p),
            ctypes.c_int(n), ctypes.c_int(counts))
        if err != 0:
            raise RuntimeError('pack_tril kernel failed')

    if ndim == 2:
        a_tril = a_tril[0]
    return a_tril

def unpack_tril(cderi_tril, out=None, stream=None, hermi=1):
    assert cderi_tril.flags.c_contiguous
    assert hermi in (1, 2)
    ndim = cderi_tril.ndim
    assert ndim in (1, 2)
    if ndim == 1:
        cderi_tril = cderi_tril[None]
    count = cderi_tril.shape[0]
    nao = int((2*cderi_tril.shape[1])**.5)
    out = ndarray((count,nao,nao), dtype=cderi_tril.dtype, buffer=out)

    if cderi_tril.dtype != np.float64:
        idx = cupy.arange(nao)
        mask = idx[:,None] >= idx
        cderiT = out.transpose(0,2,1)
        if hermi == 1:
            cderiT[:,mask] = cderi_tril.conj()
        else:
            raise NotImplementedError
        out [:,mask] = cderi_tril
        return out

    if stream is None:
        stream = cupy.cuda.get_current_stream()
    err = libdpnp_helper.unpack_tril(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(cderi_tril.data.ptr, ctypes.c_void_p),
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.c_int(nao),
        ctypes.c_int(count),
        ctypes.c_int(hermi))
    if err != 0:
        raise RuntimeError('failed in unpack_tril kernel')
    if ndim == 1:
        out = out[0]
    return out

def unpack_sparse(cderi_sparse, row, col, p0, p1, nao, out=None, stream=None):
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    if out is None:
        out = cupy.zeros([nao,nao,p1-p0])
    nij = len(row)
    naux = cderi_sparse.shape[1]
    nao = out.shape[1]
    err = libdpnp_helper.unpack_sparse(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(cderi_sparse.data.ptr, ctypes.c_void_p),
        ctypes.cast(row.data.ptr, ctypes.c_void_p),
        ctypes.cast(col.data.ptr, ctypes.c_void_p),
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.c_int(nao),
        ctypes.c_int(nij),
        ctypes.c_int(naux),
        ctypes.c_int(p0),
        ctypes.c_int(p1)
    )
    if err != 0:
        raise RuntimeError('failed in unpack_sparse')
    return out

def add_sparse(a, b, indices):
    '''
    a[:,...,:np.ix_(indices, indices)] += b
    '''
    assert a.device == b.device
    assert a.flags.c_contiguous
    assert b.flags.c_contiguous
    if len(indices) == 0: return a
    indices = cupy.asarray(indices, dtype=np.int32)
    n = a.shape[-1]
    m = b.shape[-1]
    if a.ndim > 2:
        count = np.prod(a.shape[:-2])
    elif a.ndim == 2:
        count = 1
    else:
        raise RuntimeError('add_sparse only supports 2d or 3d tensor')

    stream = cupy.cuda.get_current_stream()
    err = libdpnp_helper.add_sparse(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(a.data.ptr, ctypes.c_void_p),
        ctypes.cast(b.data.ptr, ctypes.c_void_p),
        ctypes.cast(indices.data.ptr, ctypes.c_void_p),
        ctypes.c_int(n),
        ctypes.c_int(m),
        ctypes.c_int(count)
    )
    if err != 0:
        raise RuntimeError('failed in sparse_add2d')
    return a

def dist_matrix(x, y, out=None):
    '''np.linalg.norm(x[:,None,:] - y[None,:,:], axis=2)'''
    x = dpnp.asarray(x, dtype=np.float64)
    y = dpnp.asarray(y, dtype=np.float64)    
    assert x.flags.c_contiguous
    assert y.flags.c_contiguous

    m = x.shape[0]
    n = y.shape[0]
    if out is None:
        out = dpnp.empty([m,n])

    stream = cupy.cuda.get_current_stream()
    err = libdpnp_helper.dist_matrix(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.cast(x.data.ptr, ctypes.c_void_p),
        ctypes.cast(y.data.ptr, ctypes.c_void_p),
        ctypes.c_int(m),
        ctypes.c_int(n)
    )
    if err != 0:
        raise RuntimeError('failed in calculating distance matrix')
    return out

@multi_gpu.lru_cache(1)
def _initialize_c2s_data():
    from gpu4pyscf.gto import mole
    c2s_l = [mole.cart2sph_by_l(l) for l in range(LMAX_ON_GPU)]
    c2s_data = dpnp.concatenate([x.ravel() for x in c2s_l])
    c2s_offset = np.cumsum([0] + [x.shape[0]*x.shape[1] for x in c2s_l])
    return c2s_l, c2s_data, c2s_offset

def block_c2s_diag(angular, counts):
    '''
    Diagonal blocked cartesian to spherical transformation
    Args:
        angular (list): angular momentum type, e.g. [0,1,2,3]
        counts (list): count of each angular momentum
    '''
    c2s_l, c2s_data, c2s_offset = _initialize_c2s_data()

    nshells = np.sum(counts)
    rows = [np.array([0], dtype='int32')]
    cols = [np.array([0], dtype='int32')]
    offsets = []
    for l, count in zip(angular, counts):
        r, c = c2s_l[l].shape
        rows.append(rows[-1][-1] + np.arange(1,count+1, dtype='int32') * r)
        cols.append(cols[-1][-1] + np.arange(1,count+1, dtype='int32') * c)
        offsets += [c2s_offset[l]] * count
    rows = dpnp.asarray(np.hstack(rows))
    cols = dpnp.asarray(np.hstack(cols))

    ncart, nsph = int(rows[-1]), int(cols[-1])
    cart2sph = dpnp.zeros([ncart, nsph])
    offsets = dpnp.asarray(offsets, dtype='int32')

    stream = cupy.cuda.get_current_stream()
    err = libdpnp_helper.block_diag(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(cart2sph.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ncart),
        ctypes.c_int(nsph),
        ctypes.cast(c2s_data.data.ptr, ctypes.c_void_p),
        ctypes.c_int(nshells),
        ctypes.cast(offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(rows.data.ptr, ctypes.c_void_p),
        ctypes.cast(cols.data.ptr, ctypes.c_void_p),
    )
    if err != 0:
        raise RuntimeError('failed in block_diag kernel')
    return cart2sph

def block_diag(blocks, out=None):
    '''
    each block size is up to 16x16
    '''
    rows = np.cumsum(np.asarray([0] + [x.shape[0] for x in blocks]))
    cols = np.cumsum(np.asarray([0] + [x.shape[1] for x in blocks]))
    offsets = np.cumsum(np.asarray([0] + [x.shape[0]*x.shape[1] for x in blocks]))

    m, n = rows[-1], cols[-1]
    if out is None: out = cupy.zeros([m, n])
    rows = cupy.asarray(rows, dtype='int32')
    cols = cupy.asarray(cols, dtype='int32')
    offsets = cupy.asarray(offsets, dtype='int32')
    data = cupy.concatenate([x.ravel() for x in blocks])
    stream = cupy.cuda.get_current_stream()
    err = libdpnp_helper.block_diag(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.cast(data.data.ptr, ctypes.c_void_p),
        ctypes.c_int(len(blocks)),
        ctypes.cast(offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(rows.data.ptr, ctypes.c_void_p),
        ctypes.cast(cols.data.ptr, ctypes.c_void_p),
    )
    if err != 0:
        raise RuntimeError('failed in block_diag kernel')
    return out

def take_last2d(a, indices, out=None):
    '''
    Reorder the last 2 dimensions as a[..., indices[:,None], indices]
    '''
    assert a.flags.c_contiguous
    assert a.shape[-1] == a.shape[-2]
    nao = a.shape[-1]
    nidx = len(indices)
    if a.ndim == 2:
        count = 1
    else:
        count = np.prod(a.shape[:-2])
    out = ndarray((count, nidx, nidx), buffer=out)        
    indices_int32 = dpnp.asarray(indices, dtype='int32')
    stream = cupy.cuda.get_current_stream()
    err = libdpnp_helper.take_last2d(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.cast(a.data.ptr, ctypes.c_void_p),
        ctypes.cast(indices_int32.data.ptr, ctypes.c_void_p),
        ctypes.c_int(count),
        ctypes.c_int(nidx),
        ctypes.c_int(nao)
    )
    if err != 0:
        raise RuntimeError('failed in take_last2d kernel')
    if a.ndim == 2:
        out = out.reshape(nidx,nidx)
    return out

def takebak(out, a, indices, axis=-1):
    '''(experimental)
    Take elements from a NumPy array along an axis and write to CuPy array.
    out[..., indices] = a
    '''
    assert axis == -1
    assert isinstance(a, np.ndarray)
    assert isinstance(out, dpnp.ndarray)
    assert out.ndim == a.ndim
    assert a.shape[-1] == len(indices)
    if a.ndim == 1:
        count = 1
    else:
        assert out.shape[:-1] == a.shape[:-1]
        count = np.prod(a.shape[:-1])
    n_a = a.shape[-1]
    n_o = out.shape[-1]
    indices_int32 = dpnp.asarray(indices, dtype=dpnp.int32)
    stream = cupy.cuda.get_current_stream()
    err = libdpnp_helper.takebak(
        ctypes.c_void_p(stream.ptr),
        ctypes.c_void_p(out.data.ptr), a.ctypes,
        ctypes.c_void_p(indices_int32.data.ptr),
        ctypes.c_int(count), ctypes.c_int(n_o), ctypes.c_int(n_a)
    )
    if err != 0: # Not the mapped host memory
        out[...,indices] = dpnp.asarray(a)
    return out

def transpose_sum(a, stream=None, inplace=True):
    '''
    return a + a.transpose(0,2,1) inplace
    '''
    if not inplace:
        a = dpnp.copy(a, order='C')    
    assert isinstance(a, dpnp.ndarray)
    assert a.flags.c_contiguous
    assert a.ndim in (2, 3)
    ndim = a.ndim
    if ndim == 2:
        a = a[None]
    count, m, n = a.shape
    assert m == n
    out = a
    stream = cupy.cuda.get_current_stream()
    if a.dtype == np.float64:
        fn = libdpnp_helper.transpose_dsum
    else:
        fn = libdpnp_helper.transpose_zsum
    err = fn(ctypes.cast(stream.ptr, ctypes.c_void_p),
             ctypes.cast(a.data.ptr, ctypes.c_void_p),
             ctypes.c_int(n), ctypes.c_int(count))
    if err != 0:
        raise RuntimeError('failed in transpose_sum kernel')
    if ndim == 2:
        out = out[0]
    return out

def hermi_triu(mat, hermi=1, inplace=True, stream=None):
    '''
    Use the elements of the lower triangular part to fill the upper triangular part.
    See also pyscf.lib.hermi_triu

    hermi=1 performs symmetric; hermi=2 performs anti-symmetric
    '''
    assert hermi in (1, 2)
    assert mat.dtype == np.float64
    if inplace:
        assert mat.flags.c_contiguous
    else:
        mat = mat.copy('C')

    if mat.ndim == 2:
        n = mat.shape[0]
        counts = 1
    elif mat.ndim == 3:
        counts, n = mat.shape[:2]
    else:
        raise ValueError(f'dimension not supported {mat.ndim}')

    if stream is None:
        stream = cupy.cuda.get_current_stream()
    err = libdpnp_helper.fill_triu(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(mat.data.ptr, ctypes.c_void_p),
        ctypes.c_int(n), ctypes.c_int(counts), ctypes.c_int(hermi))
    if err != 0:
        raise RuntimeError('hermi_triu kernel failed')
    return mat

def cart2sph_cutensor(t, axis=0, ang=1, out=None):
    '''
    transform 'axis' of a tensor from cartesian basis into spherical basis with cutensor
    '''
    from gpu4pyscf.gto import mole
    if(ang <= 1):
        if(out is not None): out[:] = t
        return t
    size = list(t.shape)
    c2s = mole.cart2sph_by_l(ang)
    if(not t.flags['C_CONTIGUOUS']): t = cupy.asarray(t, order='C')
    li_size = c2s.shape
    nli = size[axis] // li_size[0]
    i0 = max(1, np.prod(size[:axis]))
    i3 = max(1, np.prod(size[axis+1:]))
    out_shape = size[:axis] + [nli*li_size[1]] + size[axis+1:]

    t_cart = t.reshape([i0*nli, li_size[0], i3])
    if(out is not None):
        out = out.reshape([i0*nli, li_size[1], i3])
    t_sph = contract('min,ip->mpn', t_cart, c2s, out=out)
    return t_sph.reshape(out_shape)

def cart2sph(t, axis=0, ang=1, out=None, stream=None):
    '''
    transform 'axis' of a tensor from cartesian basis into spherical basis
    '''
    from gpu4pyscf.gto import mole
    if(ang <= 1):
        if(out is not None): out[:] = t
        return t
    size = list(t.shape)
    c2s = mole.cart2sph_by_l(ang)
    if(not t.flags['C_CONTIGUOUS']): t = cupy.asarray(t, order='C')
    li_size = c2s.shape
    nli = size[axis] // li_size[0]
    i0 = max(1, np.prod(size[:axis]))
    i3 = max(1, np.prod(size[axis+1:]))
    out_shape = size[:axis] + [nli*li_size[1]] + size[axis+1:]

    t_cart = t.reshape([i0*nli, li_size[0], i3])
    if(out is not None):
        out = out.reshape([i0*nli, li_size[1], i3])
    else:
        out = cupy.empty(out_shape)
    count = i0*nli*i3
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    err = libdpnp_helper.cart2sph(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(t_cart.data.ptr, ctypes.c_void_p),
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.c_int(i3),
        ctypes.c_int(count),
        ctypes.c_int(ang)
    )
    if err != 0:
        raise RuntimeError('failed in cart2sph kernel')
    return out.reshape(out_shape)

# a copy with modification from
# https://github.com/pyscf/pyscf/blob/9219058ac0a1bcdd8058166cad0fb9127b82e9bf/pyscf/lib/linalg_helper.py#L1536
def krylov(aop, b, x0=None, tol=1e-10, max_cycle=30, dot=cupy.dot,
           lindep=DSOLVE_LINDEP, callback=None, hermi=False,
           verbose=logger.WARN):
    r'''Krylov subspace method to solve  (1+a) x = b.  Ref:
    J. A. Pople et al, Int. J.  Quantum. Chem.  Symp. 13, 225 (1979).
    Args:
        aop : function(x) => array_like_x
            aop(x) to mimic the matrix vector multiplication :math:`\sum_{j}a_{ij} x_j`.
            The argument is a 1D array.  The returned value is a 1D array.
        b : a vector or a list of vectors
    Kwargs:
        x0 : 1D array
            Initial guess
        tol : float
            Tolerance to terminate the operation aop(x).
        max_cycle : int
            max number of iterations.
        lindep : float
            Linear dependency threshold.  The function is terminated when the
            smallest eigenvalue of the metric of the trial vectors is lower
            than this threshold.
        dot : function(x, y) => scalar
            Inner product
        callback : function(envs_dict) => None
            callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            environment.
    Returns:
        x : ndarray like b
    '''
    if isinstance(aop, cupy.ndarray) and aop.ndim == 2:
        return cupy.linalg.solve(aop+cupy.eye(aop.shape[0]), b)

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    if not (isinstance(b, cupy.ndarray) and b.ndim == 1):
        b = cupy.asarray(b)

    if x0 is None:
        x1 = b
    else:
        b = b - (x0 + aop(x0))
        x1 = b
    if x1.ndim == 1:
        x1 = x1.reshape(1, x1.size)
    nroots, ndim = x1.shape
    x1, rmat = _stable_qr(x1, cupy.dot, lindep=lindep)
    if len(x1) == 0:
        return cupy.zeros_like(b)

    x1 *= rmat.diagonal()[:,None]

    innerprod = [rmat[i,i].real ** 2 for i in range(x1.shape[0])]
    max_innerprod = max(innerprod)

    if max_innerprod < lindep or max_innerprod < tol**2:
        if x0 is None:
            return cupy.zeros_like(b)
        else:
            return x0

    xs = []
    ax = []

    max_cycle = min(max_cycle, ndim)
    for cycle in range(max_cycle):
        axt = aop(x1)
        if axt.ndim == 1:
            axt = axt.reshape(1,ndim)
        xs.extend(x1)
        ax.extend(axt)
        if callable(callback):
            callback(cycle, xs, ax)
        x1 = axt.copy()
        for i in range(len(xs)):
            xsi = cupy.asarray(xs[i])
            w = cupy.dot(x1, xsi.conj()) / innerprod[i]
            x1 -= xsi * cupy.expand_dims(w,-1)
        axt = xsi = None
        x1, rmat = _stable_qr(x1, cupy.dot, lindep=lindep)
        x1 *= rmat.diagonal()[:,None]
        innerprod1 = rmat.diagonal().real ** 2
        max_innerprod = max(innerprod1, default=0.)

        log.info(f'krylov cycle {cycle}, r = {max_innerprod**.5:.3e}, {x1.shape[0]} equations')
        if max_innerprod < lindep or max_innerprod < tol**2:
            break
        mask = (innerprod1 > lindep) & (innerprod1 > tol**2)
        x1 = x1[mask]
        innerprod.extend(innerprod1[mask])
        if max_innerprod > 1e10:
            raise RuntimeError('Krylov subspace iterations diverge')

    else:
        raise RuntimeError('Krylov solver failed to converge')

    log.info(f'krylov space size {len(xs)}')
    xs = cupy.asarray(xs)
    ax = cupy.asarray(ax)
    nd = xs.shape[0]

    h = cupy.dot(xs, ax.T)

    # Add the contribution of I in (1+a)
    h += cupy.diag(cupy.asarray(innerprod[:nd]))
    g = cupy.zeros((nd,nroots), dtype=x1.dtype)

    if b.ndim == 1:
        g[0] = innerprod[0]
    else:
        # Restore the first nroots vectors, which are array b or b-(1+a)x0
        for i in range(min(nd, nroots)):
            xsi = cupy.asarray(xs[i])
            g[i] = cupy.dot(xsi.conj(), b.T)

    c = cupy.linalg.solve(h, g)
    x = _gen_x0(c, cupy.asarray(xs))
    if b.ndim == 1:
        x = x[0]

    if x0 is not None:
        x += x0
    return x

def _qr(xs, dot, lindep=1e-14):
    '''QR decomposition for a list of vectors (for linearly independent vectors only).
    xs = (r.T).dot(qs)
    '''
    nvec = len(xs)
    dtype = xs[0].dtype
    qs = cupy.empty((nvec,xs[0].size), dtype=dtype)
    rmat = cupy.eye(nvec, order='F', dtype=dtype)

    nv = 0
    for i in range(nvec):
        xi = cupy.array(xs[i], copy=True)
        prod = dot(qs[:nv].conj(), xi)
        xi -= cupy.dot(qs[:nv].T, prod)

        innerprod = dot(xi.conj(), xi).real
        norm = innerprod**0.5
        if innerprod > lindep:
            rmat[:,nv] -= cupy.dot(rmat[:,:nv], prod)
            qs[nv] = xi/norm
            rmat[:nv+1,nv] /= norm
            nv += 1
    return qs[:nv], cupy.linalg.inv(rmat[:nv,:nv])

def _stable_qr(xs, dot, lindep=1e-14):
    '''QR decomposition for a list of vectors (for linearly independent vectors only).
    using the modified Gram-Schmidt process
    '''
    nvec = len(xs)
    dtype = xs[0].dtype
    Q = cupy.empty((nvec,xs[0].size), dtype=dtype)
    R = cupy.zeros((nvec,nvec), dtype=dtype)
    V = xs.copy()
    nv = 0
    for i in range(nvec):
        norm = cupy.linalg.norm(V[i])
        if norm**2 > lindep:
            R[nv,nv] = norm
            Q[nv] = V[i] / norm
            R[nv, i+1:] = dot(Q[nv], V[i+1:].T)
            V[i+1:] -= cupy.outer(R[nv, i+1:], Q[nv])
            nv += 1
    return Q[:nv], R[:nv,:nv]

def _gen_x0(v, xs):
    ndim = v.ndim
    if ndim == 1:
        v = v[:,None]
    space, nroots = v.shape
    x0 = cupy.einsum('c,x->cx', v[space-1], cupy.asarray(xs[space-1]))
    for i in reversed(range(space-1)):
        xsi = cupy.asarray(xs[i])
        x0 += cupy.expand_dims(v[i],-1) * xsi
    if ndim == 1:
        x0 = x0[0]
    return x0

def empty_mapped(shape, dtype=float, order='C'):
    '''(experimental)
    Returns a new, uninitialized NumPy array with the given shape and dtype.

    This is a convenience function which is just :func:`numpy.empty`,
    except that the underlying buffer is a pinned and mapped memory.
    This array can be used as the buffer of zero-copy memory.
    '''
    nbytes = np.prod(shape) * np.dtype(dtype).itemsize
    q = dpctl.SyclQueue()  # or _create_from_ptr(int(libgpu.sycl_get_queue_ptr()))
    mem = dpmem.MemoryUSMShared(nbytes, queue=q)  # use MemoryUSMHost(...) if you don't need device mapping
    # mem = cupy.cuda.PinnedMemoryPointer(
    #     cupy.cuda.PinnedMemory(nbytes, cupy.cuda.runtime.hostAllocMapped), 0)
    out = np.ndarray(shape, dtype=dtype, buffer=mem, order=order)
    return out

def ndarray(shape, dtype=np.float64, buffer=None):
    '''
    Construct CuPy ndarray object using the NumPy ndarray API
    '''
    if buffer is None:
        return cupy.empty(shape, dtype=dtype)
    else:
        out = cupy.ndarray(shape, dtype, memptr=buffer.data)
        assert buffer.nbytes >= out.nbytes
        return out

def pinv(a, lindep=1e-10):
    '''psudo-inverse with eigh, to be consistent with pyscf
    '''
    a = cupy.asarray(a)
    w, v = cupy.linalg.eigh(a)
    mask = w > lindep
    v1 = v[:,mask]
    j2c = cupy.dot(v1/w[mask], v1.conj().T)
    return j2c

def cond(a, sympos=False):
    """
    Calculate the condition number of a matrix.

    Parameters:
    a (cupy.ndarray): The input matrix.
    sympos : Whether the input matrix is symmetric and positive definite.

    Returns:
    float: The condition number of the matrix.
    """
    if sympos:
        s = cupy.linalg.eigvalsh(a)
        if s[0] <= 0:
            raise RuntimeError('matrix is not positive definite')
        return s[-1] / s[0]
    else:
        _, s, _ = cupy.linalg.svd(a)
        cond_number = s[0] / s[-1]
        return cond_number

def grouped_dot(As, Bs, Cs=None):
    '''
    As: dpnp 2D array list.
    Bs: dpnp 2D array list.
    Cs: dpnp 2D array list.
    einsum('ik,jk->ij', A, B, C) C=A@B.T
    '''
    assert len(As) > 0
    assert len(As) == len(Bs)
    assert As[0].flags.c_contiguous
    assert Bs[0].flags.c_contiguous
    groups = len(As)

    if Cs is None:
        Cs = []
        for a, b in zip(As, Bs):
            Cs.append(cupy.empty((a.shape[0], b.shape[0])))

    # Pure DPNP implementation using matmul with transpose
    # C = A @ B.T  (einsum 'ik,jk->ij')
    for i in range(groups):
        # B.T: transpose B so that (N, K) -> (K, N)
        # Result: (M, K) @ (K, N) -> (M, N)
        Cs[i][...] = cupy.matmul(As[i], Bs[i].T)

    return Cs

# def grouped_dot(As, Bs, Cs=None):
#     '''
#     todo: layout of cutlass kernel
#     As: dpnp 2D array list.
#     Bs: dpnp 2D array list.
#     Cs: dpnp 2D array list.
#     einsum('ik,jk->ij', A, B, C) C=A@B.T
#     '''
#     assert len(As) > 0
#     assert len(As) == len(Bs)
#     assert As[0].flags.c_contiguous
#     assert Bs[0].flags.c_contiguous
#     groups = len(As)
#     Ms, Ns, Ks = [], [], []
#     for a, b in zip(As, Bs):
#         Ms.append(a.shape[0])
#         Ns.append(b.shape[0])
#         Ks.append(a.shape[1])

#     if Cs is None:
#         Cs = []
#         for i in range(groups):
#             Cs.append(cupy.empty((Ms[i], Ns[i])))

#     As_ptr, Bs_ptr, Cs_ptr = [], [], []
#     for a, b, c in zip(As, Bs, Cs):
#         As_ptr.append(a.data.ptr)
#         Bs_ptr.append(b.data.ptr)
#         Cs_ptr.append(c.data.ptr)

#     As_ptr = np.array(As_ptr)
#     Bs_ptr = np.array(Bs_ptr)
#     Cs_ptr = np.array(Cs_ptr)

#     Ms = np.array(Ms)
#     Ns = np.array(Ns)
#     Ks = np.array(Ks)
#     total_size = 68 * groups
#     '''
#     68 is the result of
#     sizeof(cutlass::gemm::GemmCoord) +
#     sizeof(typename DeviceKernel::ElementA*) +
#     sizeof(typename DeviceKernel::ElementB*) +
#     sizeof(typename DeviceKernel::ElementC*) +
#     sizeof(typename DeviceKernel::ElementC*) +
#     sizeof(int64_t) + sizeof(int64_t) + sizeof(int64_t)
#     '''
#     padding = 8 - (total_size % 8)
#     total_size += padding
#     cutlass_space = cupy.empty(total_size, dtype=cupy.uint8)

#     stream = cupy.cuda.get_current_stream()
#     err = libdpnp_helper.grouped_dot(
#         ctypes.cast(stream.ptr, ctypes.c_void_p),
#         ctypes.cast(Cs_ptr.ctypes.data, ctypes.c_void_p),
#         ctypes.cast(As_ptr.ctypes.data, ctypes.c_void_p),
#         ctypes.cast(Bs_ptr.ctypes.data, ctypes.c_void_p),
#         ctypes.cast(Ms.ctypes.data, ctypes.c_void_p),
#         ctypes.cast(Ns.ctypes.data, ctypes.c_void_p),
#         ctypes.cast(Ks.ctypes.data, ctypes.c_void_p),
#         ctypes.cast(cutlass_space.data.ptr, ctypes.c_void_p),
#         ctypes.c_int(groups)
#     )
#     if err != 0:
#         raise RuntimeError('failed in grouped_gemm kernel')
#     return Cs

def grouped_gemm(As, Bs, Cs=None):
    '''
    As: dpnp 2D array list.
    Bs: dpnp 2D array list.
    Cs: dpnp 2D array list.
    assuming (X, 64).T @ (X, Y)
    einsum('ki,kj->ij', A, B, C) C=A.T@B
    Compare with grouped_dot, this function handles the case M < 128
    '''
    assert len(As) > 0
    assert len(As) == len(Bs)
    assert As[0].flags.c_contiguous
    assert Bs[0].flags.c_contiguous
    groups = len(As)
    Ms, Ns, Ks = [], [], []
    for a, b in zip(As, Bs):
        Ms.append(a.shape[1])
        Ns.append(b.shape[1])
        Ks.append(a.shape[0])

    if Cs is None:
        Cs = []
        for i in range(groups):
            Cs.append(cupy.empty((Ms[i], Ns[i])))

    As_ptr, Bs_ptr, Cs_ptr = [], [], []
    for a, b, c in zip(As, Bs, Cs):
        As_ptr.append(a.data.ptr)
        Bs_ptr.append(b.data.ptr)
        Cs_ptr.append(c.data.ptr)
    As_ptr = np.array(As_ptr)
    Bs_ptr = np.array(Bs_ptr)
    Cs_ptr = np.array(Cs_ptr)

    Ms = np.array(Ms)
    Ns = np.array(Ns)
    Ks = np.array(Ks)

    stream = cupy.cuda.get_current_stream()
    err = libdpnp_helper.grouped_gemm(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(Cs_ptr.ctypes.data, ctypes.c_void_p),
        ctypes.cast(As_ptr.ctypes.data, ctypes.c_void_p),
        ctypes.cast(Bs_ptr.ctypes.data, ctypes.c_void_p),
        ctypes.cast(Ms.ctypes.data, ctypes.c_void_p),
        ctypes.cast(Ns.ctypes.data, ctypes.c_void_p),
        ctypes.cast(Ks.ctypes.data, ctypes.c_void_p),
        ctypes.c_int(groups)
    )
    if err != 0:
        raise RuntimeError('failed in grouped_gemm kernel')
    return Cs

# def condense(opname, a, loc_x, loc_y=None):
#     """
#     dpnp version of condense() without any explicit SYCL kernel.

#     Parameters
#     ----------
#     opname : {'sum', 'max', 'min', 'abssum', 'absmax', 'norm'}
#     a      : np.ndarray or dpnp.ndarray, float64, ndim >= 2
#     loc_x  : 1D array-like of ints (partition on axis -2)
#     loc_y  : 1D array-like of ints (partition on axis -1), defaults to loc_x

#     Returns
#     -------
#     out : dpnp.ndarray (or numpy if you convert back)
#         Shape (len(loc_x)-1, len(loc_y)-1) (possibly transposed for Fortran input).
#     """
#     assert opname in ("sum", "max", "min", "abssum", "absmax", "norm")
#     assert a.dtype == np.float64
#     assert a.ndim >= 2
#     if loc_y is None:
#         loc_y = loc_x
#     do_transpose = False
#     loc_x = np.asarray(loc_x, dtype=np.int64)
#     loc_y = np.asarray(loc_y, dtype=np.int64)

#     if a.ndim == 2:
#         if a.flags.f_contiguous:
#             a = dpnp.transpose(a)
#             loc_x, loc_y = loc_y, loc_x
#             do_transpose = True
#         a = a[None]
#     else:
#         nx, ny = a.shape[-2:]
#         a = a.reshape(-1, nx, ny)

#     counts, nx, ny = a.shape
#     assert loc_x[-1] == nx
#     assert loc_y[-1] == ny

#     # Move to device
#     a_dev = dp.asarray(a)  # shape (counts, nx, ny)
#     loc_x_dev = loc_x      # indices are small, fine on host
#     loc_y_dev = loc_y

#     nloc_x = loc_x_dev.size - 1
#     nloc_y = loc_y_dev.size - 1

#     out = dp.zeros((nloc_x, nloc_y), dtype=dp.float64)

#     for i in range(nloc_x):
#         i0, i1 = loc_x_dev[i], loc_x_dev[i + 1]
#         for j in range(nloc_y):
#             j0, j1 = loc_y_dev[j], loc_y_dev[j + 1]

#             # Slice all counts, block in x,y -> shape (counts, i1-i0, j1-j0)
#             block = a_dev[:, i0:i1, j0:j1]

#             if opname == "sum":
#                 val = dp.sum(block)  # over all axes
#             elif opname == "max":
#                 val = dp.max(block)
#             elif opname == "min":
#                 val = dp.min(block)
#             elif opname == "abssum":
#                 val = dp.sum(dp.abs(block))
#             elif opname == "absmax":
#                 val = dp.max(dp.abs(block))
#             elif opname == "norm":
#                 # sqrt of sum of squares over all elements
#                 val = dp.sqrt(dp.sum(block * block))

#             out[i, j] = val

#     if do_transpose:
#         out = dpnp.transpose(out)
#     return out

def condense(opname, a, loc_x, loc_y=None):
    """
    DPNP/SYCL port of condense(): reduce over the last two dims in windows.
    Reduces across counts and the i/j window just like the CUDA kernel.

    """
    assert opname in ('sum', 'max', 'min', 'abssum', 'absmax', 'norm')
    assert a.dtype == np.float64
    assert a.ndim >= 2
    if loc_y is None:
        loc_y = loc_x
    do_transpose = False
    if a.ndim == 2:
        # Match CUDA path: if input is F-contig, transpose and swap locators
        if a.flags.f_contiguous:
            a = dpnp.transpose(a)
            loc_x, loc_y = loc_y, loc_x
            do_transpose = True
        a = a[None, ...]  # shape -> (counts=1, nx, ny)
    else:
        nx, ny = int(a.shape[-2]), int(a.shape[-1])
        a = a.reshape(-1, nx, ny)  # (counts, nx, ny)

    # Work with host-side integer indices; windows stay on device
    a = dpnp.asarray(a, order='C')
    loc_x = np.asarray(loc_x, dtype=np.int32)
    loc_y = np.asarray(loc_y, dtype=np.int32)
    nloc_x = loc_x.size - 1
    nloc_y = loc_y.size - 1
    counts, nx, ny = a.shape
    assert loc_x[-1] == nx
    assert loc_y[-1] == ny

    out = dpnp.zeros((nloc_x, nloc_y), dtype=a.dtype)

    # Helper for a single window reduction
    def _reduce_window(win):
        if   opname == 'sum':    return dpnp.sum(win)
        elif opname == 'max':    return dpnp.max(win)
        elif opname == 'min':    return dpnp.min(win)
        elif opname == 'abssum': return dpnp.sum(dpnp.abs(win))
        elif opname == 'absmax': return dpnp.max(dpnp.abs(win))
        elif opname == 'norm':   return dpnp.sqrt(dpnp.sum(win * win))
        else:
            raise ValueError(opname)

    # Host loops over blocks; device does heavy reductions per window
    for i in range(nloc_x):
        i0, i1 = int(loc_x[i]), int(loc_x[i+1])
        for j in range(nloc_y):
            j0, j1 = int(loc_y[j]), int(loc_y[j+1])
            win = a[:, i0:i1, j0:j1]          # (counts, i1-i0, j1-j0) on device
            out[i, j] = _reduce_window(win)   # device reduction

    if do_transpose:
        out = dpnp.transpose(out)

    return out

def sandwich_dot(a, c, out=None):
    '''Performs c.T.dot(a).dot(c)'''
    a = cupy.asarray(a)
    c = cupy.asarray(c)
    a_ndim = a.ndim
    if a_ndim == 2:
        a = a[None]
    counts = a.shape[0]
    m = c.shape[1]
    dtype = dpnp.result_type(a, c)
    out = cupy.empty((counts, m, m), dtype=dtype)
    tmp = None
    for i in range(counts):
        tmp = cupy.dot(c.conj().T, a[i], out=tmp)
        cupy.dot(tmp, c, out=out[i])
    if a_ndim == 2:
        out = out[0]
    return out

def set_conditional_mempool_malloc(threshold=None):
    """No-op: SYCL/USM manages memory automatically.
    
    In CuPy, this sets conditional memory pool allocation based on size.
    With DPNP/SYCL USM, memory management is handled by the runtime.
    """
    pass
# def set_conditional_mempool_malloc(n_bytes_threshold=100000000):
#     '''
#     Customize CuPy memory allocator.

#     For large memory allocations (>100MB by default), the custom allocator bypasses
#     the CuPy memory pool, directly calling the CUDA malloc API. The large memory
#     chunks will be released back to the system when the associated object is
#     destroyed. Only small memory blocks are allocated from the CuPy memory pool.

#     Execute the following command to restore the default CuPy malloc
#         cupy.cuda.set_allocator(cupy.get_default_memory_pool().malloc)
#     '''
#     cuda_malloc = cupy.cuda.memory._malloc
#     default_mempool_malloc = cupy.get_default_memory_pool().malloc
#     def malloc(size):
#         if size >= n_bytes_threshold:
#             return cuda_malloc(size)
#         return default_mempool_malloc(size)
#     cupy.cuda.set_allocator(malloc)

def batched_vec3_norm2(batched_vec3):
    """
    Compute per-row squared L2 norm for an (n,3) float64 array on a SYCL device.

    Parameters
    ----------
    batched_vec3 : dpnp.ndarray or array-like
        Shape (n,3), float64.
    strict : bool
        If True, enforce the same assumptions as the CuPy version:
        - must already be dpnp.ndarray
        - must be C-contiguous
        - dtype float64, shape (n,3)
        If False, the function will convert/copy as needed.
    device, usm_type, sycl_queue :
        Optional placement controls for dpnp allocations/conversion.
    """
    # if strict:
    assert type(batched_vec3) is dpnp.ndarray
    assert batched_vec3.dtype == dpnp.float64
    assert batched_vec3.ndim == 2
    assert batched_vec3.shape[1] == 3
    assert batched_vec3.flags.c_contiguous
    vec = batched_vec3
    # else:
    #     vec = dpnp.asarray(
    #         batched_vec3,
    #         dtype=dpnp.float64,
    #         order="C",
    #         device=device,
    #         usm_type=usm_type,
    #         sycl_queue=sycl_queue,
    #     )

    if vec.ndim != 2 or vec.shape[1] != 3:
        raise ValueError(f"Expected shape (n,3); got {vec.shape}")

    n = vec.shape[0]
    if n >= np.iinfo(np.int32).max:
        raise ValueError("n must fit in int32 (matches original constraint)")

    # Preallocate output on the same device/queue by default
    out = dpnp.zeros(n, dtype=dpnp.float64)

    # Equivalent to: out[i] = sum_j vec[i,j] * vec[i,j]
    dpnp.einsum("ij,ij->i", vec, vec, out=out)
    return out

cholesky = onemkl_cholesky

def eigh(a, b=None, overwrite=False):
    '''
    Solve a standard or generalized eigenvalue problem for a complex
    Hermitian or real symmetric matrix.

    Note: both a and b matrices are overwritten when overwrite is specified.
    '''
    # if a.shape[0] > cusolver.MAX_EIGH_DIM:
    #     if not SCIPY_EIGH_FOR_LARGE_ARRAYS:
    #         raise RuntimeError(
    #             f'Array size exceeds the maximum size {cusolver.MAX_EIGH_DIM}.')
    #     a = a.get()
    #     if b is not None:
    #         b = b.get()
    #     e, c = scipy.linalg.eigh(a, b, overwrite_a=True)
    #     e = asarray(e)
    #     c = asarray(c)
    #     return e, c

    if b is not None:
        return onemkl_eigh(a, b, overwrite)

    return cupy.linalg.eigh(a)
