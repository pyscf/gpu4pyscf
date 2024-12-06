# gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
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
import cupy
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.gto import mole
from gpu4pyscf.lib.cutensor import contract
from gpu4pyscf.lib.cusolver import eigh, cholesky  #NOQA
from gpu4pyscf.__config__ import _streams, _num_devices, _p2p_access

LMAX_ON_GPU = 7
DSOLVE_LINDEP = 1e-13

c2s_l = mole.get_cart2sph(lmax=LMAX_ON_GPU)
c2s_offset = np.cumsum([0] + [x.shape[0]*x.shape[1] for x in c2s_l])
_data = {'c2s': None}

_kernel_registery = {}

def load_library(libname):
    try:
        _loaderpath = os.path.dirname(__file__)
        return np.ctypeslib.load_library(libname, _loaderpath)
    except OSError:
        raise

libcupy_helper = load_library('libcupy_helper')

pinned_memory_pool = cupy.cuda.PinnedMemoryPool()
cupy.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

def pin_memory(array):
    mem = cupy.cuda.alloc_pinned_memory(array.nbytes)
    ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret

def release_gpu_stack():
    cupy.cuda.runtime.deviceSetLimit(0x00, 128)

def print_mem_info():
    mempool = cupy.get_default_memory_pool()
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()
    mem_avail = cupy.cuda.runtime.memGetInfo()[0]
    total_mem = mempool.total_bytes()
    used_mem = mempool.used_bytes()
    mem_limit = mempool.get_limit()
    #stack_size_per_thread = cupy.cuda.runtime.deviceGetLimit(0x00)
    #mem_stack = stack_size_per_thread
    GB = 1024 * 1024 * 1024
    print(f'mem_avail: {mem_avail/GB:.3f} GB, total_mem: {total_mem/GB:.3f} GB, used_mem: {used_mem/GB:.3f} GB,mem_limt: {mem_limit/GB:.3f} GB')

def get_avail_mem():
    mempool = cupy.get_default_memory_pool()
    used_mem = mempool.used_bytes()
    mem_limit = mempool.get_limit()
    if(mem_limit != 0):
        return mem_limit - used_mem
    else:
        total_mem = mempool.total_bytes()
        # get memGetInfo() is slow
        mem_avail = cupy.cuda.runtime.memGetInfo()[0]
        return mem_avail + total_mem - used_mem

def p2p_transfer(a, b):
    ''' If the direct P2P data transfer is not available, transfer data via CPU memory
    '''
    if a.device == b.device:
        a[:] = b
    elif _p2p_access:
        a[:] = b
    else:
        with cupy.cuda.Device(a.device):
            a[:].set(b.get())

def concatenate(array_list):
    ''' Concatenate axis=0 only
    '''
    if _p2p_access:
        return cupy.concatenate(array_list)
    else:
        array_list_cpu = [a.get() for a in array_list]
        n = sum([a.shape[0] for a in array_list_cpu])
        a0_shape = list(array_list_cpu[0].shape)
        out_shape = tuple([n] + a0_shape[1:])
        out = cupy.empty(out_shape)
        p0 = p1 = 0
        for a in array_list_cpu:
            p1 = p0 + a.shape[0]
            out[p0:p1].set(a)
            p0 = p1
        return out

def broadcast_to_devices():
    ''' Broadcast cupy ndarray to all the devices, return a list of cupy ndarray
    '''
    raise NotImplementedError

def reduce_to_device(array_list, inplace=False):
    ''' Reduce a list of ndarray in different devices to device 0
    TODO: reduce memory footprint, improve throughput
    '''
    assert len(array_list) == _num_devices
    if _num_devices == 1:
        return array_list[0]
    
    out_shape = array_list[0].shape
    for s in _streams:
        s.synchronize()
        
    if inplace:
        result = array_list[0]
    else:
        result = array_list[0].copy()
    result = result.reshape(-1)
    # Asynchronously add each matrix from its device
    for device_id, matrix in enumerate(array_list):
        if device_id == 0:
            continue
        
        assert matrix.device.id == device_id
        matrix = matrix.reshape(-1)
        blksize = 1024*1024*128 # 1GB
        for p0, p1 in lib.prange(0,len(matrix), blksize):
            result[p0:p1] += cupy.asarray(matrix[p0:p1])
    
    return result.reshape(out_shape)
    
def device2host_2d(a_cpu, a_gpu, stream=None):
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    libcupy_helper.async_d2h_2d(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        a_cpu.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(a_cpu.strides[0]),
        ctypes.cast(a_gpu.data.ptr, ctypes.c_void_p),
        ctypes.c_int(a_gpu.strides[0]),
        ctypes.c_int(a_gpu.shape[0]),
        ctypes.c_int(a_gpu.shape[1]))

# define cupy array with tags
class CPArrayWithTag(cupy.ndarray):
    pass

#@functools.wraps(lib.tag_array)
def tag_array(a, **kwargs):
    '''
    a should be cupy/numpy array or tuple of cupy/numpy array

    attach attributes to cupy ndarray for cupy array
    attach attributes to numpy ndarray for numpy array
    '''
    if isinstance(a, cupy.ndarray) or isinstance(a[0], cupy.ndarray):
        t = cupy.asarray(a).view(CPArrayWithTag)
        if isinstance(a, CPArrayWithTag):
            t.__dict__.update(a.__dict__)
    else:
        t = np.asarray(a).view(lib.NPArrayWithTag)
        if isinstance(a, lib.NPArrayWithTag):
            t.__dict__.update(a.__dict__)
    t.__dict__.update(kwargs)
    return t

def to_cupy(a):
    '''Converts a numpy (and subclass) object to a cupy object'''
    if isinstance(a, lib.NPArrayWithTag):
        attrs = {k: to_cupy(v) for k, v in a.__dict__.items()}
        return tag_array(cupy.asarray(a), **attrs)
    if isinstance(a, np.ndarray):
        return cupy.asarray(a)
    return a

def return_cupy_array(fn):
    '''Ensure that arrays in returns are cupy objects'''
    @functools.wraps(fn)
    def filter_ret(*args, **kwargs):
        ret = fn(*args, **kwargs)
        if isinstance(ret, tuple):
            return tuple(to_cupy(x) for x in ret)
        return to_cupy(ret)
    return filter_ret

def unpack_tril(cderi_tril, cderi, stream=None):
    nao = cderi.shape[1]
    count = cderi_tril.shape[0]
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    err = libcupy_helper.unpack_tril(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(cderi_tril.data.ptr, ctypes.c_void_p),
        ctypes.cast(cderi.data.ptr, ctypes.c_void_p),
        ctypes.c_int(nao),
        ctypes.c_int(count))
    if err != 0:
        raise RuntimeError('failed in unpack_tril kernel')
    return

def unpack_sparse(cderi_sparse, row, col, p0, p1, nao, out=None, stream=None):
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    if out is None:
        out = cupy.zeros([nao,nao,p1-p0])
    nij = len(row)
    naux = cderi_sparse.shape[1]
    nao = out.shape[1]
    err = libcupy_helper.unpack_sparse(
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
    n = a.shape[-1]
    m = b.shape[-1]
    if a.ndim > 2:
        count = np.prod(a.shape[:-2])
    elif a.ndim == 2:
        count = 1
    else:
        raise RuntimeError('add_sparse only supports 2d or 3d tensor')
    
    stream = cupy.cuda.get_current_stream()
    err = libcupy_helper.add_sparse(
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
    assert x.flags.c_contiguous
    assert y.flags.c_contiguous

    m = x.shape[0]
    n = y.shape[0]
    if out is None:
        out = cupy.empty([m,n])

    stream = cupy.cuda.get_current_stream()
    err = libcupy_helper.dist_matrix(
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

def block_c2s_diag(angular, counts):
    '''
    Diagonal blocked cartesian to spherical transformation
    Args: 
        angular (list): angular momentum type, e.g. [0,1,2,3]
        counts (list): count of each angular momentum
    '''
    if _data['c2s'] is None:
        c2s_data = cupy.concatenate([cupy.asarray(x.ravel()) for x in c2s_l])
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
    rows = cupy.hstack(rows)
    cols = cupy.hstack(cols)
    
    ncart, nsph = int(rows[-1]), int(cols[-1])
    cart2sph = cupy.zeros([ncart, nsph])
    offsets = cupy.asarray(offsets, dtype='int32')

    stream = cupy.cuda.get_current_stream()
    err = libcupy_helper.block_diag(
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
    err = libcupy_helper.block_diag(
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
    assert len(indices) == nao
    if a.ndim == 2:
        count = 1
    else:
        count = np.prod(a.shape[:-2])
    if out is None:
        out = cupy.zeros_like(a)
    indices_int32 = cupy.asarray(indices, dtype='int32')
    stream = cupy.cuda.get_current_stream()
    err = libcupy_helper.take_last2d(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.cast(a.data.ptr, ctypes.c_void_p),
        ctypes.cast(indices_int32.data.ptr, ctypes.c_void_p),
        ctypes.c_int(count),
        ctypes.c_int(nao)
    )
    if err != 0:
        raise RuntimeError('failed in take_last2d kernel')
    return out

def takebak(out, a, indices, axis=-1):
    '''(experimental)
    Take elements from a NumPy array along an axis and write to CuPy array.
    out[..., indices] = a
    '''
    assert axis == -1
    assert isinstance(a, np.ndarray)
    assert isinstance(out, cupy.ndarray)
    assert out.ndim == a.ndim
    assert a.shape[-1] == len(indices)
    if a.ndim == 1:
        count = 1
    else:
        assert out.shape[:-1] == a.shape[:-1]
        count = np.prod(a.shape[:-1])
    n_a = a.shape[-1]
    n_o = out.shape[-1]
    indices_int32 = cupy.asarray(indices, dtype=cupy.int32)
    stream = cupy.cuda.get_current_stream()
    err = libcupy_helper.takebak(
        ctypes.c_void_p(stream.ptr),
        ctypes.c_void_p(out.data.ptr), a.ctypes,
        ctypes.c_void_p(indices_int32.data.ptr),
        ctypes.c_int(count), ctypes.c_int(n_o), ctypes.c_int(n_a)
    )
    if err != 0: # Not the mapped host memory
        out[...,indices] = cupy.asarray(a)
    return out

def transpose_sum(a, stream=None):
    '''
    return a + a.transpose(0,2,1)
    '''
    assert a.flags.c_contiguous
    out = a
    if a.ndim == 2:
        a = a[None]
    assert a.ndim == 3
    count, m, n = a.shape
    assert m == n
    stream = cupy.cuda.get_current_stream()
    err = libcupy_helper.transpose_sum(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(a.data.ptr, ctypes.c_void_p),
        ctypes.c_int(n),
        ctypes.c_int(count)
    )
    if err != 0:
        raise RuntimeError('failed in transpose_sum kernel')
    return out

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

    err = libcupy_helper.CPdsymm_triu(
        ctypes.cast(mat.data.ptr, ctypes.c_void_p),
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
    c2s = cupy.asarray(c2s_l[ang])
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
    if(ang <= 1):
        if(out is not None): out[:] = t
        return t
    size = list(t.shape)
    c2s = c2s_l[ang]
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
    err = libcupy_helper.cart2sph(
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
            envrionment.
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
    mem = cupy.cuda.PinnedMemoryPointer(
        cupy.cuda.PinnedMemory(nbytes, cupy.cuda.runtime.hostAllocMapped), 0)
    out = np.ndarray(shape, dtype=dtype, buffer=mem, order=order)
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

def cond(a):
    """
    Calculate the condition number of a matrix.

    Parameters:
    a (cupy.ndarray): The input matrix.

    Returns:
    float: The condition number of the matrix.
    """
    _, s, _ = cupy.linalg.svd(a)
    cond_number = s[0] / s[-1]
    return cond_number

def grouped_dot(As, Bs, Cs=None):
    '''
    todo: layout of cutlass kernel
    As: cupy 2D array list.
    Bs: cupy 2D array list.
    Cs: cupy 2D array list.
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
    cutlass_space = cupy.empty(total_size, dtype=cupy.uint8)

    stream = cupy.cuda.get_current_stream()
    err = libcupy_helper.grouped_dot(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(Cs_ptr.ctypes.data, ctypes.c_void_p),
        ctypes.cast(As_ptr.ctypes.data, ctypes.c_void_p),
        ctypes.cast(Bs_ptr.ctypes.data, ctypes.c_void_p),
        ctypes.cast(Ms.ctypes.data, ctypes.c_void_p),
        ctypes.cast(Ns.ctypes.data, ctypes.c_void_p),
        ctypes.cast(Ks.ctypes.data, ctypes.c_void_p),
        ctypes.cast(cutlass_space.data.ptr, ctypes.c_void_p),
        ctypes.c_int(groups)
    )
    if err != 0:
        raise RuntimeError('failed in grouped_gemm kernel')
    return Cs

def grouped_gemm(As, Bs, Cs=None):
    '''
    As: cupy 2D array list.
    Bs: cupy 2D array list.
    Cs: cupy 2D array list.
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
    err = libcupy_helper.grouped_gemm(
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

def condense(opname, a, loc_x, loc_y=None):
    assert opname in ('sum', 'max', 'min', 'abssum', 'absmax', 'norm')
    assert a.dtype == np.float64
    if loc_y is None:
        loc_y = loc_x
    do_transpose = False
    if a.ndim == 2:
        if a.flags.f_contiguous:
            a = a.T
            loc_x, loc_y = loc_y, loc_x
            do_transpose = True
        a = a[None]
    else:
        assert a.flags.c_contiguous
    loc_x = cupy.asarray(loc_x, cupy.int32)
    loc_y = cupy.asarray(loc_y, cupy.int32)
    nloc_x = loc_x.size - 1
    nloc_y = loc_y.size - 1
    counts, nx, ny = a.shape
    assert loc_x[-1] == nx
    assert loc_y[-1] == ny

    #if opname == 'absmax':
    #    out = cupy.zeros((nloc_x, nloc_y))
    #    err = libcupy_helper.dabsmax_condense(
    #        ctypes.cast(out.ctypes.data, ctypes.c_void_p),
    #        ctypes.cast(a.ctypes.data, ctypes.c_void_p),
    #        ctypes.cast(loc_x.ctypes.data, ctypes.c_void_p),
    #        ctypes.cast(loc_y.ctypes.data, ctypes.c_void_p),
    #        ctypes.c_int(nloc_x), ctypes.c_int(nloc_y), ctypes.c_int(counts))
    #    if err != 0:
    #        raise RuntimeError('failed in dabsmax_condense kernel')
    #    if do_transpose:
    #        out = out.T
    #    return out

    fn_name = f'd{opname}_condense'
    if fn_name not in _kernel_registery:
        if opname == 'sum':
            init_code = '0'
            code = 'val += a[ip*nj+jp];'
            result_code = 'val'
        elif opname == 'max':
            init_code = '0'
            code = 'double tmp = a[ip*nj+jp]; val = (val > tmp) ? val : tmp;'
            result_code = 'val'
        elif opname == 'min':
            init_code = '0'
            code = 'double tmp = a[ip*nj+jp]; val = (val < tmp) ? val : tmp;'
            result_code = 'val'
        elif opname == 'abssum':
            init_code = '0'
            code = 'val += fabs(a[ip*nj+jp]);'
            result_code = 'val'
        elif opname == 'absmax':
            init_code = '0'
            code = 'double tmp = fabs(a[ip*nj+jp]); val = (val > tmp) ? val : tmp;'
            result_code = 'val'
        elif opname == 'norm':
            init_code = '0'
            code = 'double tmp = a[ip*nj+jp]; val += tmp * tmp;'
            result_code = 'fsqrt(val)'

        kernel_code = (f'''\
extern "C" __global__
void {fn_name}(double *out, double *a, int *loc_x, int *loc_y,
               long long nloc_x, long long nloc_y, long long counts)'''
'''
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nloc_x || j >= nloc_y) {
        return;
    }
    size_t ni = loc_x[nloc_x];
    size_t nj = loc_y[nloc_y];
    size_t Nloc_y = nloc_y;
    int i0 = loc_x[i];
    int i1 = loc_x[i+1];
    int j0 = loc_y[j];
    int j1 = loc_y[j+1];
    double val = ''' + init_code + ''';
    for (int n = 0; n < counts; ++n) {
        for (int ip = i0; ip < i1; ++ip) {
        for (int jp = j0; jp < j1; ++jp) {
            ''' + code + '''
        } }
        a += ni * nj;
    }
    out[i*Nloc_y+j] = ''' + result_code + ''';
}
''')
        _kernel_registery[fn_name] = cupy.RawKernel(kernel_code, fn_name)

    kernel = _kernel_registery[fn_name]
    out = cupy.zeros((nloc_x, nloc_y))
    blocks = ((nloc_x+15)//16, (nloc_y+15)//16)
    threads = (16, 16)
    kernel(blocks, threads, (out, a, loc_x, loc_y, nloc_x, nloc_y, counts))
    cupy.cuda.Stream.null.synchronize()
    if do_transpose:
        out = out.T
    return out

def sandwich_dot(a, c, out=None):
    '''Performs c.T.dot(a).dot(c)'''
    a = cupy.asarray(a)
    a_ndim = a.ndim
    if a_ndim == 2:
        a = a[None]
    counts = a.shape[0]
    m = c.shape[1]
    dtype = np.result_type(a, c)
    out = cupy.empty((counts, m, m), dtype=dtype)
    tmp = None
    for i in range(counts):
        tmp = cupy.dot(c.conj().T, a[i], out=tmp)
        cupy.dot(tmp, c, out=out[i])
    if a_ndim == 2:
        out = out[0]
    return out
