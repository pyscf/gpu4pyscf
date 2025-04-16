# gpu4pyscf is a plugin to use Intel GPU in PySCF package
#
# Copyright (C) 2022 Qiming Sun
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import functools
import ctypes
import numpy as np
import dpnp
from dpctl._sycl_device_factory import _cached_default_device as get_default_cached_device
from dpctl._sycl_queue_manager import get_device_cached_queue
import dpctl, dpctl.utils
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.gto import mole

from gpu4pyscf.lib.dptensor import contract
# from gpu4pyscf.lib.cusolver import eigh, cholesky  #NOQA

LMAX_ON_GPU = 7
DSOLVE_LINDEP = 1e-13

c2s_l = mole.get_cart2sph(lmax=LMAX_ON_GPU)
c2s_offset = np.cumsum([0] + [x.shape[0]*x.shape[1] for x in c2s_l])
_data = {'c2s': None}

def load_library(libname):
    try:
        _loaderpath = os.path.dirname(__file__)
        # return np.ctypeslib.load_library(libname, _loaderpath)
        return ctypes.CDLL(f"{_loaderpath}/{libname}.so") # np.ctypeslib.load_library(libname, _loaderpath)
    except OSError:
        raise

# libdpnp_helper = load_library('libdpnp_helper')
print(f"###{os.getcwd()}##")
print(f"###{os.path.dirname(__file__)}##")

path = os.path.dirname(__file__)
libdpnp_helper = ctypes.CDLL(f"{path}/libdpnp_helper.so")  # Adjust the path as needed


def eigh():
    return

# libdpnp_helper.cart2sph.argtypes=[c_void_p, ctypes.POINTER(c_double), ctypes.POINTER(c_double), c_int,  c_int,  c_int]
# libdpnp_helper.unpack_tril.argtypes=[c_void_p, ctypes.POINTER(c_double), ctypes.POINTER(c_double), c_int,  c_int,  c_int]
libdpnp_helper.cart2sph.restype = int
libdpnp_helper.unpack_tril.restype = int

def pin_memory(array):
    mem = dpctl.memory.MemoryUSMHost(array.nbytes)
    ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret

def release_gpu_stack():
    print('release_gpu_stack place holder')
    # dpnp.cuda.runtime.deviceSetLimit(0x00, 128)

def print_mem_info():
    dev = get_default_cached_device()
    dev.print_device_info()
    descr = dpctl.utils.intel_device_info(dev)
    mem_avail = descr['free_memory']
    total_mem = dev.global_mem_size
    GB = 1024 * 1024 * 1024
    print(f'mem_avail: {mem_avail/GB:.3f} GB, total_mem: {total_mem/GB:.3f} GB')

def get_avail_mem():
    dev = get_default_cached_device()
    descr = dpctl.utils.intel_device_info(dev)
    return descr['free_memory']

def device2host_2d(a_cpu, a_gpu, stream=None):
    if stream is None:
        stream = dpctl.get_current_queue()
    libdpnp_helper.async_d2h_2d(
        ctypes.cast(stream.get_queue_ref(), ctypes.c_void_p),
        a_cpu.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(a_cpu.strides[0]),
        ctypes.cast(a_gpu.data.ptr, ctypes.c_void_p),
        ctypes.c_int(a_gpu.strides[0]),
        ctypes.c_int(a_gpu.shape[0]),
        ctypes.c_int(a_gpu.shape[1]))

# define dpnp array with tags
class CPArrayWithTag(dpnp.ndarray):
    pass

@functools.wraps(lib.tag_array)
def tag_array(a, **kwargs):
    '''
    a should be cupy/numpy array or tuple of cupy/numpy array

    attach attributes to dpnp ndarray for dpnp array
    attach attributes to numpy ndarray for numpy array
    '''
    if isinstance(a, dpnp.ndarray) or isinstance(a[0], dpnp.ndarray):
        t = dpnp.asarray(a).view(CPArrayWithTag)
        if isinstance(a, CPArrayWithTag):
            t.__dict__.update(a.__dict__)
    else:
        t = np.asarray(a).view(lib.NPArrayWithTag)
        if isinstance(a, lib.NPArrayWithTag):
            t.__dict__.update(a.__dict__)
    t.__dict__.update(kwargs)
    return t

def to_dpnp(a):
    '''Converts a numpy (and subclass) object to a dpnp object'''
    if isinstance(a, lib.NPArrayWithTag):
        attrs = {k: to_dpnp(v) for k, v in a.__dict__.items()}
        return tag_array(dpnp.asarray(a), **attrs)
    if isinstance(a, np.ndarray):
        return dpnp.asarray(a)
    return a

def return_gpunp_array(fn):
    '''Ensure that arrays in returns are dpnp objects'''
    @functools.wraps(fn)
    def filter_ret(*args, **kwargs):
        ret = fn(*args, **kwargs)
        if isinstance(ret, tuple):
            return tuple(to_dpnp(x) for x in ret)
        return to_dpnp(ret)
    return filter_ret

# def unpack_tril(cderi_tril, cderi, stream=None):
#     nao = cderi.shape[1]
#     count = cderi_tril.shape[0]
#     blk = 32
#     if stream is None:
#         stream = cderil_tril.sycl_queue
    
#     cderi_tril_usm_interface = cderi_tril.__sycl_usm_array_interface__
#     cderi_tril_data_ptr = cderi_tril_usm_interface['data'][0]

#     cderi_usm_interface = cderi.__sycl_usm_array_interface__
#     cderi_data_ptr = cderi_usm_interface['data'][0]

#     err = libdpnp_helper.unpack_tril(
#         ctypes.cast(stream.addressof_ref(), ctypes.POINTER(ctypes.c_size_t)), # stream
#         ctypes.cast(cderi_tril_data_ptr, ctypes.POINTER(ctypes.c_double)),
#         ctypes.cast(cderi_data_ptr, ctypes.POINTER(ctypes.c_double)),
#         ctypes.c_int(nao),
#         ctypes.c_int(count),
#         ctypes.c_int(blk))
#     if err != 0:
#         raise RuntimeError('failed in unpack_tril kernel')
#     return

# def get_ptr(val):
#     _usm_interface = val.__sycl_usm_array_interface__
#     return _usm_interface['data'][0]
     
# def unpack_sparse(cderi_sparse, row, col, p0, p1, nao, out=None, stream=None):
#     if stream is None:
#         stream = cderi_sparse.sycl_queue
#     if out is None:
#         out = dpnp.zeros([nao,nao,p1-p0])
#     nij = len(row)
#     naux = cderi_sparse.shape[1]
#     nao = out.shape[1]
#     cderi_sparse_usm_interface = cderi_sparse.__sycl_usm_array_interface__
#     cderi_sparse_data_ptr = cderi_sparse_usm_interface['data'][0]
#     out_usm_interface = out.__sycl_usm_array_interface__
#     out_data_ptr = out_usm_interface['data'][0]
#     err = libdpnp_helper.unpack_sparse(
#         ctypes.cast(stream.addressof_ref(), ctypes.POINTER(ctypes.c_size_t)), # stream
#         ctypes.cast(cderi_sparse_data_ptr, ctypes.POINTER(ctypes.c_double)),
#         ctypes.cast(row.data.ptr, ctypes.c_void_p), #alvarom unsure
#         ctypes.cast(col.data.ptr, ctypes.c_void_p),
#         ctypes.cast(out_data_ptr, ctypes.POINTER(ctypes.c_double)),
#         ctypes.c_int(nao),
#         ctypes.c_int(nij),
#         ctypes.c_int(naux),
#         ctypes.c_int(p0),
#         ctypes.c_int(p1)
#     )
#     if err != 0:
#         raise RuntimeError('failed in unpack_sparse')
#     return out

def add_sparse(a, b, indices):
    '''
    a[:,...,:np.ix_(indices, indices)] += b
    '''
    assert a.flags.c_contiguous
    assert b.flags.c_contiguous
    if len(indices) == 0: return a
    n = a.shape[-1]
    m = b.shape[-1]
    if a.ndim > 2:
        count = np.prod(a.shape[:-2])
    elif a.ndim == 2:
        count = 1
    else:
        raise RuntimeError('add_sparse only supports 2d or 3d tensor')
    stream = a.sycl_queue
    a_ptr = a.__sycl_usm_array_interface__['data'][0]
    b_ptr = b.__sycl_usm_array_interface__['data'][0]
    indices_ptr = indices.__sycl_usm_array_interface__['data'][0]
    err = libdpnp_helper.add_sparse(
        ctypes.cast(stream.addressof_ref(), ctypes.POINTER(ctypes.c_size_t)), # stream
        ctypes.cast(a_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(b_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(indices_ptr, ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(n),
        ctypes.c_int(m),
        ctypes.c_int(count)
    )
    if err != 0:
        raise RuntimeError('failed in sparse_add2d')
    return a

def dist_matrix(x, y, out=None, stream=None):
    assert x.flags.c_contiguous
    assert y.flags.c_contiguous

    m = x.shape[0]
    n = y.shape[0]

    if stream is None:
        stream = x.sycl_queue
    if out is None:
        out = dpnp.empty([m,n], sycl_queue=stream)
    x_ptr = x.__sycl_usm_array_interface__['data'][0]
    y_ptr = y.__sycl_usm_array_interface__['data'][0]
    out_ptr = out.__sycl_usm_array_interface__['data'][0]
    err = libdpnp_helper.dist_matrix(
        ctypes.cast(stream.addressof_ref(), ctypes.POINTER(ctypes.c_size_t)),
        ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(y_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(m),
        ctypes.c_int(n)
    )
    if err != 0:
        raise RuntimeError('failed in calculating distance matrix')
    return out

def block_c2s_diag(ncart, nsph, angular, counts, stream=None):
    '''
    constract a cartesian to spherical transformation of n shells
    '''
    if _data['c2s'] is None: 
        c2s_data = dpnp.concatenate([dpnp.asarray(x.ravel()) for x in c2s_l])
        _data['c2s'] = c2s_data
    c2s_data = _data['c2s']

    nshells = np.sum(counts)
    rows = [np.array([0], dtype='int32')]
    cols = [np.array([0], dtype='int32')]
    offsets = []
    for l, count in zip(angular, counts):
        r, c = c2s_l[l].shape
        rows.append(rows[-1][-1] + np.arange(1,count+1, dtype='int32') * r)
        cols.append(cols[-1][-1] + np.arange(1,count+1, dtype='int32') * c)
        offsets += [c2s_offset[l]] * count
    if stream is None:
        stream = dpctl.SyclQueue()
    rows = dpnp.hstack(rows)
    cols = dpnp.hstack(cols)

    cart2sph = dpnp.zeros([ncart, nsph],sycl_queue=stream)
    offsets = dpnp.asarray(offsets, dtype='int32', sycl_queue=stream)

    cart2sph_prt = cart2sph.__sycl_usm_array_interface__['data'][0]
    offsets_prt = offsets.__sycl_usm_array_interface__['data'][0]
    c2s_data_prt = c2s_data.__sycl_usm_array_interface__['data'][0]
    rows_prt = rows.__sycl_usm_array_interface__['data'][0]
    cols_prt = cols.__sycl_usm_array_interface__['data'][0]

    err = libdpnp_helper.block_diag(
        ctypes.cast(stream.addressof_ref(), ctypes.POINTER(ctypes.c_size_t)),
        ctypes.cast(cart2sph_prt, ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(ncart),
        ctypes.c_int(nsph),
        ctypes.cast(c2s_data_prt, ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(nshells),
        ctypes.cast(offsets_prt, ctypes.POINTER(ctypes.c_int)),
        ctypes.cast(rows_prt, ctypes.POINTER(ctypes.c_int)),
        ctypes.cast(cols_prt, ctypes.POINTER(ctypes.c_int)),
    )
    if err != 0:
        raise RuntimeError('failed in block_diag kernel')
    print('vama careful with queue')
    return cart2sph

def block_diag(blocks, out=None,stream=None):
    '''
    each block size is up to 16x16
    '''
    rows = np.cumsum(np.asarray([0] + [x.shape[0] for x in blocks]))
    cols = np.cumsum(np.asarray([0] + [x.shape[1] for x in blocks]))
    offsets = np.cumsum(np.asarray([0] + [x.shape[0]*x.shape[1] for x in blocks]))

    m, n = rows[-1], cols[-1]
    if out is None: out = dpnp.zeros([m, n])
    rows = dpnp.asarray(rows, dtype='int32')
    cols = dpnp.asarray(cols, dtype='int32')
    offsets = dpnp.asarray(offsets, dtype='int32')
    data = dpnp.concatenate([x.ravel() for x in blocks])
    if stream is None:
        stream = dpctl.SyclQueue()

    cart2sph_ptr = cart2sph.__sycl_usm_array_interface__['data'][0]
    offsets_ptr = offsets.__sycl_usm_array_interface__['data'][0]
    data_ptr = data.__sycl_usm_array_interface__['data'][0]
    rows_ptr = rows.__sycl_usm_array_interface__['data'][0]
    cols_ptr = cols.__sycl_usm_array_interface__['data'][0]
    err = libdpnp_helper.block_diag(
        ctypes.cast(stream.addressof_ref(), ctypes.POINTER(ctypes.c_size_t)),
        ctypes.cast(cart2sph_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(blocks)),
        ctypes.cast(offsets_ptr, ctypes.POINTER(ctypes.c_int)),
        ctypes.cast(rows_ptr, ctypes.POINTER(ctypes.c_int)),
        ctypes.cast(cols_ptr, ctypes.POINTER(ctypes.c_int)),
    )
    if err != 0:
        raise RuntimeError('failed in block_diag kernel')
    return out

def take_last2d(a, indices, out=None, stream=None):
    '''
    Reorder the last 2 dimensions as a[..., indices[:,None], indices]
    '''
    assert a.flags.c_contiguous
    assert a.shape[-1] == a.shape[-2]
    nao = a.shape[-1]
    assert len(indices) == nao
    if a.ndim == 2:
        count = 1
    else:
        count = np.prod(a.shape[:-2])
    if stream is None:
        stream = a.sycl_queue
    if out is None:
        out = dpnp.zeros_like(a, sycl_queue=stream)
    indices_int32 = dpnp.asarray(indices, dtype='int32', sycl_queue=stream)
    a_ptr = a.__sycl_usm_array_interface__['data'][0]
    out_ptr = out.__sycl_usm_array_interface__['data'][0]
    indices_int32_ptr = indices_int32.__sycl_usm_array_interface__['data'][0]
    err = libdpnp_helper.take_last2d(
        ctypes.cast(stream.addressof_ref(), ctypes.POINTER(ctypes.c_size_t)), # stream
        ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(a_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(a_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(indices_int32_ptr, ctypes.c_void_p),
        ctypes.c_int(count),
        ctypes.c_int(nao)
    )
    if err != 0:
        raise RuntimeError('failed in take_last2d kernel')
    return out

def takebak(out, a, indices, axis=-1, stream=None):
    '''(experimental)
    Take elements from a NumPy array along an axis and write to dpnp array.
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
    indices_int32 = dpnp.asarray(indices, dtype=dpnp.int32, sycl_queue=stream)
    if stream is None:
        stream = out.sycl_queue
    out_ptr = out.__sycl_usm_array_interface__['data'][0]
    indices_int32_ptr = indices_int32.__sycl_usm_array_interface__['data'][0]
    err = libdpnp_helper.takebak(
        ctypes.cast(stream.addressof_ref(), ctypes.POINTER(ctypes.c_size_t)),
        ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_double)),
        a.ctypes,
        ctypes.cast(indices_int32_ptr, ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(count), ctypes.c_int(n_o), ctypes.c_int(n_a)
    )
    if err != 0: # Not the mapped host memory
        out[...,indices] = dpnp.asarray(a)
    return out

def transpose_sum(a, stream=None):
    '''
    return a + a.transpose(0,2,1)
    '''
    assert a.flags.c_contiguous
    n = a.shape[-1]
    if a.ndim == 2:
        a = a.reshape([-1,n,n])
    assert a.ndim == 3
    count = a.shape[0]
    if stream is None:
        stream = a.sycl_queue
    a_ptr = a.__sycl_usm_array_interface__['data'][0]
    err = libdpnp_helper.transpose_sum(
        ctypes.cast(stream.addressof_ref(), ctypes.POINTER(ctypes.c_size_t)),
        ctypes.cast(a_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n),
        ctypes.c_int(count)
    )
    if err != 0:
        raise RuntimeError('failed in transpose_sum kernel')
    return a

# for i > j of 2d mat, mat[j,i] = mat[i,j]
def hermi_triu(mat, hermi=1, inplace=True):
    '''
    Use the elements of the lower triangular part to fill the upper triangular part.
    See also pyscf.lib.hermi_triu
    '''
    if not inplace:
        mat = mat.copy('C')
    assert mat.flags.c_contiguous

    if mat.ndim == 2:
        n = mat.shape[0]
        counts = 1
    elif mat.ndim == 3:
        counts, n = mat.shape[:2]
    else:
        raise ValueError(f'dimension not supported {mat.ndim}')

    if stream is None:
        stream = mat.sycl_queue
    mat_ptr = mat.__sycl_usm_array_interface__['data'][0]
    err = libdpnp_helper.CPdsymm_triu(
        ctypes.cast(stream.addressof_ref(), ctypes.POINTER(ctypes.c_size_t)),
        ctypes.cast(mat_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n), ctypes.c_int(counts))
    if err != 0:
        raise RuntimeError('failed in symm_triu kernel')

    return mat

def cart2sph_cutensor(t, axis=0, ang=1, out=None):
    '''
    transform 'axis' of a tensor from cartesian basis into spherical basis with cutensor
    '''
    if(ang <= 1):
        if(out is not None): out[:] = t
        return t
    size = list(t.shape)
    c2s = dpnp.asarray(c2s_l[ang])
    if(not t.flags['C_CONTIGUOUS']): t = dpnp.asarray(t, order='C')
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
    if(ang <= 1):
        if(out is not None): out[:] = t
        return t
    size = list(t.shape)
    c2s = c2s_l[ang]
    if(not t.flags['C_CONTIGUOUS']): t = dpnp.asarray(t, order='C')
    li_size = c2s.shape
    nli = size[axis] // li_size[0]
    i0 = max(1, np.prod(size[:axis]))
    i3 = max(1, np.prod(size[axis+1:]))
    out_shape = size[:axis] + [nli*li_size[1]] + size[axis+1:]

    t_cart = t.reshape([i0*nli, li_size[0], i3])
    if(out is not None):
        out = out.reshape([i0*nli, li_size[1], i3])
    else:
        out = dpnp.empty(out_shape)
    count = i0*nli*i3
    if stream is None:
        stream = t.sycl_queue

    t_cart_usm_interface = t_cart.__sycl_usm_array_interface__
    out_usm_interface = out.__sycl_usm_array_interface__

    t_cart_data_ptr = t_cart_usm_interface['data'][0]
    out_data_ptr = out_usm_interface['data'][0]

    err = dpnp_helper.cart2sph(
        ctypes.cast(stream.addressof_ref(), ctypes.POINTER(c_size_t)), # stream
        ctypes.cast(t_cart_data_ptr, ctypes.POINTER(c_double)),
        ctypes.cast(out_data_ptr, ctypes.POINTER(c_double)),
        c_int(i3),
        c_int(count),
        c_int(ang),
    )
    if err != 0:
        raise RuntimeError('failed in cart2sph kernel')
    return out.reshape(out_shape)

# a copy with modification from
# https://github.com/pyscf/pyscf/blob/9219058ac0a1bcdd8058166cad0fb9127b82e9bf/pyscf/lib/linalg_helper.py#L1536
def krylov(aop, b, x0=None, tol=1e-10, max_cycle=30, dot=dpnp.dot,
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
            envrionment.
    Returns:
        x : ndarray like b
    '''
    if isinstance(aop, dpnp.ndarray) and aop.ndim == 2:
        return dpnp.linalg.solve(aop+dpnp.eye(aop.shape[0]), b)

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    if not (isinstance(b, dpnp.ndarray) and b.ndim == 1):
        b = dpnp.asarray(b)

    if x0 is None:
        x1 = b
    else:
        b = b - (x0 + aop(x0))
        x1 = b
    if x1.ndim == 1:
        x1 = x1.reshape(1, x1.size)
    nroots, ndim = x1.shape

    # Not exactly QR, vectors are orthogonal but not normalized
    x1, rmat = _qr(x1, dpnp.dot, lindep)
    for i in range(len(x1)):
        x1[i] *= rmat[i,i]

    innerprod = [dpnp.dot(xi.conj(), xi).real for xi in x1]
    if innerprod:
        max_innerprod = max(innerprod)
    else:
        max_innerprod = 0
    if max_innerprod < lindep or max_innerprod < tol**2:
        if x0 is None:
            return dpnp.zeros_like(b)
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
            xsi = dpnp.asarray(xs[i])
            w = dpnp.dot(axt, xsi.conj()) / innerprod[i]
            x1 -= xsi * dpnp.expand_dims(w,-1)
        axt = xsi = None

        x1, rmat = _qr(x1, dpnp.dot, lindep)
        for i in range(len(x1)):
            x1[i] *= rmat[i,i]

        max_innerprod = 0
        idx = []
        for i, xi in enumerate(x1):
            innerprod1 = dpnp.dot(xi.conj(), xi).real
            max_innerprod = max(max_innerprod, innerprod1)
            if innerprod1 > lindep and innerprod1 > tol**2:
                idx.append(i)
                innerprod.append(innerprod1)
        log.info(f'krylov cycle {cycle}  r = {max_innerprod**.5:.3e} {x1.shape[0]} equations')
        if max_innerprod < lindep or max_innerprod < tol**2:
            break
        x1 = x1[idx]

    if len(idx) > 0:
        raise RuntimeError("CPSCF failed to converge.")

    xs = dpnp.asarray(xs)
    ax = dpnp.asarray(ax)
    nd = xs.shape[0]

    h = dpnp.dot(xs, ax.T)

    # Add the contribution of I in (1+a)
    h += dpnp.diag(dpnp.asarray(innerprod[:nd]))
    g = dpnp.zeros((nd,nroots), dtype=x1.dtype)

    if b.ndim == 1:
        g[0] = innerprod[0]
    else:
        # Restore the first nroots vectors, which are array b or b-(1+a)x0
        for i in range(min(nd, nroots)):
            xsi = dpnp.asarray(xs[i])
            g[i] = dpnp.dot(xsi.conj(), b.T)

    c = dpnp.linalg.solve(h, g)
    x = _gen_x0(c, dpnp.asarray(xs))
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
    qs = dpnp.empty((nvec,xs[0].size), dtype=dtype)
    rmat = dpnp.empty((nvec,nvec), order='F', dtype=dtype)

    nv = 0
    for i in range(nvec):
        xi = dpnp.array(xs[i], copy=True)
        rmat[:,nv] = 0
        rmat[nv,nv] = 1

        prod = dot(qs[:nv].conj(), xi)
        xi -= dpnp.dot(qs[:nv].T, prod)
        rmat[:,nv] -= dpnp.dot(rmat[:,:nv], prod)

        innerprod = dot(xi.conj(), xi).real
        norm = dpnp.sqrt(innerprod)
        if innerprod > lindep:
            qs[nv] = xi/norm
            rmat[:nv+1,nv] /= norm
            nv += 1
    return qs[:nv], dpnp.linalg.inv(rmat[:nv,:nv])

def _gen_x0(v, xs):
    ndim = v.ndim
    if ndim == 1:
        v = v[:,None]
    space, nroots = v.shape
    x0 = dpnp.einsum('c,x->cx', v[space-1], dpnp.asarray(xs[space-1]))
    for i in reversed(range(space-1)):
        xsi = dpnp.asarray(xs[i])
        x0 += dpnp.expand_dims(v[i],-1) * xsi
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
    # nbytes = np.prod(shape) * np.dtype(dtype).itemsize
    # mem = dpnp.cuda.PinnedMemoryPointer(
    #     dpnp.cuda.PinnedMemory(nbytes, dpnp.cuda.runtime.hostAllocMapped), 0)
    # out = np.ndarray(shape, dtype=dtype, buffer=mem, order=order)
    out = np.ndarray(shape, dtype=dtype,  order=order)
    return out

def pinv(a, lindep=1e-10):
    '''psudo-inverse with eigh, to be consistent with pyscf
    '''
    a = dpnp.asarray(a)
    w, v = dpnp.linalg.eigh(a)
    mask = w > lindep
    v1 = v[:,mask]
    j2c = dpnp.dot(v1/w[mask], v1.conj().T)
    return j2c

def cond(a):
    return dpnp.linalg.norm(a,2)*dpnp.linalg.norm(dpnp.linalg.inv(a),2)

def grouped_dot(As, Bs, Cs=None, stream=None):
    '''
    todo: layout of cutlass kernel
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
    Ms, Ns, Ks = [], [], []
    for a, b in zip(As, Bs):
        Ms.append(a.shape[0])
        Ns.append(b.shape[0])
        Ks.append(a.shape[1])

    if stream is None:
        stream = As[0].sycl_queue

    if Cs is None:
        Cs = []
        for i in range(groups):
            Cs.append(dpnp.empty((Ms[i], Ns[i]), sycl_queue=stream))

    As_ptr, Bs_ptr, Cs_ptr = [], [], []
    for a, b, c in zip(As, Bs, Cs):
        a_ptr = a.__sycl_usm_array_interface__['data'][0]
        b_ptr = b.__sycl_usm_array_interface__['data'][0]
        c_ptr = c.__sycl_usm_array_interface__['data'][0]
        As_ptr.append(a_ptr)
        Bs_ptr.append(b_ptr)
        Cs_ptr.append(c_ptr)

    As_ptr = np.array(As_ptr)
    Bs_ptr = np.array(Bs_ptr)
    Cs_ptr = np.array(Cs_ptr)

    Ms = np.array(Ms)
    Ns = np.array(Ns)
    Ks = np.array(Ks)
    total_size = 68 * groups
    '''
    68 is the result of
    sizeof(cutlass::gemm::GemmCoord) +
    sizeof(typename DeviceKernel::ElementA*) +
    sizeof(typename DeviceKernel::ElementB*) +
    sizeof(typename DeviceKernel::ElementC*) +
    sizeof(typename DeviceKernel::ElementC*) +
    sizeof(int64_t) + sizeof(int64_t) + sizeof(int64_t)
    '''
    padding = 8 - (total_size % 8)
    total_size += padding
    dptlass_space = dpnp.empty(total_size, dtype=dpnp.uint8, sycl_queue=stream)
    dptlass_space_ptr = dptlass_space.__sycl_usm_array_interface__['data'][0]

    stream = dpctl.get_current_queue()
    err = libdpnp_helper.grouped_dot(
        ctypes.cast(stream.addressof_ref(), ctypes.POINTER(ctypes.c_size_t)),
        ctypes.cast(Cs_ptr.ctypes.data, ctypes.c_void_p),
        ctypes.cast(As_ptr.ctypes.data, ctypes.c_void_p),
        ctypes.cast(Bs_ptr.ctypes.data, ctypes.c_void_p),
        ctypes.cast(Ms.ctypes.data, ctypes.c_void_p),
        ctypes.cast(Ns.ctypes.data, ctypes.c_void_p),
        ctypes.cast(Ks.ctypes.data, ctypes.c_void_p),
        ctypes.cast(dptlass_space_ptr, ctypes.c_void_p),
        ctypes.c_int(groups)
    )
    if err != 0:
        raise RuntimeError('failed in grouped_gemm kernel')
    return Cs

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
    if stream is None:
        stream = As[0].sycl_queue
    Ms, Ns, Ks = [], [], []
    for a, b in zip(As, Bs):
        Ms.append(a.shape[1])
        Ns.append(b.shape[1])
        Ks.append(a.shape[0])

    if Cs is None:
        Cs = []
        for i in range(groups):
            Cs.append(dpnp.empty((Ms[i], Ns[i])))

    As_ptr, Bs_ptr, Cs_ptr = [], [], []
    for a, b, c in zip(As, Bs, Cs):
        a_ptr = a.__sycl_usm_array_interface__['data'][0]
        b_ptr = b.__sycl_usm_array_interface__['data'][0]
        c_ptr = c.__sycl_usm_array_interface__['data'][0]
        As_ptr.append(a_ptr)
        Bs_ptr.append(b_ptr)
        Cs_ptr.append(c_ptr)
    As_ptr = np.array(As_ptr)
    Bs_ptr = np.array(Bs_ptr)
    Cs_ptr = np.array(Cs_ptr)

    Ms = np.array(Ms)
    Ns = np.array(Ns)
    Ks = np.array(Ks)

    err = libdpnp_helper.grouped_gemm(
        ctypes.cast(stream.addressof_ref(), ctypes.POINTER(ctypes.c_size_t)),
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
