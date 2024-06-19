# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
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


import copy
import cupy
import ctypes
import numpy as np
from cupyx.scipy.linalg import solve_triangular
from pyscf import lib
from pyscf.df import df, addons, incore
from gpu4pyscf.lib.cupy_helper import (
    cholesky, tag_array, get_avail_mem, cart2sph, take_last2d, transpose_sum)
from gpu4pyscf.df import int3c2e, df_jk
from gpu4pyscf.lib import logger
from gpu4pyscf import __config__
from cupyx import scipy

MIN_BLK_SIZE = getattr(__config__, 'min_ao_blksize', 128)
ALIGNED = getattr(__config__, 'ao_aligned', 32)
GB = 1024*1024*1024

# TODO: reuse the setting in pyscf 2.6
LINEAR_DEP_THR = 1e-7#incore.LINEAR_DEP_THR
GROUP_SIZE = 256

class DF(lib.StreamObject):
    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'intopt', 'mol', 'auxmol'}

    def __init__(self, mol, auxbasis=None):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory
        self._auxbasis = auxbasis

        self.auxmol = None
        self.intopt = None
        self.nao = None
        self.naux = None
        self.cd_low = None
        self._cderi = None
        self._rsh_df = {}

    @property
    def auxbasis(self):
        return self._auxbasis
    @auxbasis.setter
    def auxbasis(self, x):
        if self._auxbasis != x:
            self.reset()
            self._auxbasis = x

    def to_cpu(self):
        from gpu4pyscf.lib.utils import to_cpu
        obj = to_cpu(self)
        return obj.reset()

    def build(self, direct_scf_tol=1e-14, omega=None):
        mol = self.mol
        auxmol = self.auxmol
        self.nao = mol.nao
        log = logger.new_logger(mol, mol.verbose)
        t0 = log.init_timer()
        if auxmol is None:
            self.auxmol = auxmol = addons.make_auxmol(mol, self.auxbasis)

        if omega and omega > 1e-10:
            with auxmol.with_range_coulomb(omega):
                j2c_cpu = auxmol.intor('int2c2e', hermi=1)
        else:
            j2c_cpu = auxmol.intor('int2c2e', hermi=1)
        j2c = cupy.asarray(j2c_cpu, order='C')
        t0 = log.timer_debug1('2c2e', *t0)
        intopt = int3c2e.VHFOpt(mol, auxmol, 'int2e')
        intopt.build(direct_scf_tol, diag_block_with_triu=False, aosym=True, group_size=GROUP_SIZE)
        log.timer_debug1('prepare intopt', *t0)
        self.j2c = j2c.copy()

        j2c = take_last2d(j2c, intopt.aux_ao_idx)
        try:
            self.cd_low = cholesky(j2c)
            self.cd_low = tag_array(self.cd_low, tag='cd')
        except Exception:
            w, v = cupy.linalg.eigh(j2c)
            idx = w > LINEAR_DEP_THR
            self.cd_low = (v[:,idx] / cupy.sqrt(w[idx]))
            self.cd_low = tag_array(self.cd_low, tag='eig')

        v = w = None
        naux = self.naux = self.cd_low.shape[1]
        log.debug('size of aux basis %d', naux)

        self._cderi = cholesky_eri_gpu(intopt, mol, auxmol, self.cd_low, omega=omega)
        log.timer_debug1('cholesky_eri', *t0)
        self.intopt = intopt

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True,
               direct_scf_tol=getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13),
               omega=None):
        if omega is None:
            return df_jk.get_jk(self, dm, hermi, with_j, with_k, direct_scf_tol)
        assert omega >= 0.0

        # A temporary treatment for RSH-DF integrals
        key = '%.6f' % omega
        if key in self._rsh_df:
            rsh_df = self._rsh_df[key]
        else:
            rsh_df = self._rsh_df[key] = copy.copy(self).reset()
            logger.info(self, 'Create RSH-DF object %s for omega=%s', rsh_df, omega)

        return df_jk.get_jk(rsh_df, dm, hermi, with_j, with_k, direct_scf_tol, omega=omega)

    def get_blksize(self, extra=0, nao=None):
        '''
        extra for pre-calculated space for other variables
        '''
        if nao is None: nao = self.nao
        mem_avail = get_avail_mem()
        blksize = int(mem_avail*0.2/8/(nao*nao + extra) / ALIGNED) * ALIGNED
        blksize = min(blksize, MIN_BLK_SIZE)
        log = logger.new_logger(self.mol, self.mol.verbose)
        log.debug(f"GPU Memory {mem_avail/1e9:.3f} GB available, block size = {blksize}")
        if blksize < ALIGNED:
            raise RuntimeError("Not enough GPU memory")
        return blksize

    def loop(self, blksize=None, unpack=True):
        ''' loop over all cderi and unpack the CDERI in (Lij) format '''
        cderi_sparse = self._cderi
        if blksize is None:
            blksize = self.get_blksize()
        nao = self.nao
        naux = self.naux
        rows = self.intopt.cderi_row
        cols = self.intopt.cderi_col
        buf_prefetch = None
        buf_cderi = cupy.zeros([blksize,nao,nao])
        data_stream = cupy.cuda.stream.Stream(non_blocking=True)
        compute_stream = cupy.cuda.get_current_stream()
        for p0, p1 in lib.prange(0, naux, blksize):
            p2 = min(naux, p1+blksize)
            if isinstance(cderi_sparse, cupy.ndarray):
                buf = cderi_sparse[p0:p1,:]
            if isinstance(cderi_sparse, np.ndarray):
                # first block
                if buf_prefetch is None:
                    buf = cupy.asarray(cderi_sparse[p0:p1,:])
                buf_prefetch = cupy.empty([p2-p1,cderi_sparse.shape[1]])
            with data_stream:
                if isinstance(cderi_sparse, np.ndarray) and p1 < p2:
                    buf_prefetch.set(cderi_sparse[p1:p2,:])
                stop_event = data_stream.record()
            if unpack:
                buf_cderi[:p1-p0,rows,cols] = buf
                buf_cderi[:p1-p0,cols,rows] = buf
                buf2 = buf_cderi[:p1-p0]
            else:
                buf2 = None
            yield buf2, buf.T
            compute_stream.wait_event(stop_event)
            if isinstance(cderi_sparse, np.ndarray):
                cupy.cuda.Device().synchronize()

            if buf_prefetch is not None:
                buf = buf_prefetch

    def reset(self, mol=None):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        if mol is not None:
            self.mol = mol
            self.auxmol = None
        self._cderi = None
        self._vjopt = None
        self._rsh_df = {}
        self.intopt = None
        self.nao = None
        self.naux = None
        self.cd_low = None
        return self

    get_ao_eri = get_eri = NotImplemented
    get_mo_eri = ao2mo = NotImplemented

def cholesky_eri_gpu(intopt, mol, auxmol, cd_low, omega=None, sr_only=False):
    '''
    Returns:
        2D array of (naux,nao*(nao+1)/2) in C-contiguous
    '''
    naoaux, naux = cd_low.shape
    npair = len(intopt.cderi_row)
    log = logger.new_logger(mol, mol.verbose)
    nq = len(intopt.log_qs)

    # if the matrix exceeds the limit, store CDERI in CPU memory
    avail_mem = get_avail_mem()
    use_gpu_memory = True
    if naux * npair * 8 < 0.4 * avail_mem:
        try:
            cderi = cupy.empty([naux, npair], order='C')
            log.debug(f"Saving CDERI on GPU. CDERI size {cderi.nbytes/GB}")
        except Exception:
            use_gpu_memory = False
    else:
        use_gpu_memory = False
    if(not use_gpu_memory):
        log.debug("Saving cderi on CPU memory.")
        # TODO: async allocate memory
        try:
            mem = cupy.cuda.alloc_pinned_memory(naux * npair * 8)
            cderi = np.ndarray([naux, npair], dtype=np.float64, order='C', buffer=mem)
        except Exception:
            raise RuntimeError('Out of CPU memory')
    if(not use_gpu_memory):
        data_stream = cupy.cuda.stream.Stream(non_blocking=False)
    count = 0
    nq = len(intopt.log_qs)
    cd_low_f = cupy.array(cd_low, order='F', copy=False)
    for cp_ij_id, _ in enumerate(intopt.log_qs):
        if len(intopt.ao_pairs_row[cp_ij_id]) == 0: continue
        t1 = log.init_timer()
        cpi = intopt.cp_idx[cp_ij_id]
        cpj = intopt.cp_jdx[cp_ij_id]
        li = intopt.angular[cpi]
        lj = intopt.angular[cpj]
        i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi+1]
        j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj+1]
        ni = i1 - i0
        nj = j1 - j0
        if sr_only:
            # TODO: in-place implementation or short-range kernel
            ints_slices = cupy.zeros([naoaux, nj, ni], order='C')
            for cp_kl_id, _ in enumerate(intopt.aux_log_qs):
                k0 = intopt.aux_ao_loc[cp_kl_id]
                k1 = intopt.aux_ao_loc[cp_kl_id+1]
                int3c2e.get_int3c2e_slice(intopt, cp_ij_id, cp_kl_id, out=ints_slices[k0:k1])
            if omega is not None:
                ints_slices_lr = cupy.zeros([naoaux, nj, ni], order='C')
                for cp_kl_id, _ in enumerate(intopt.aux_log_qs):
                    k0 = intopt.aux_ao_loc[cp_kl_id]
                    k1 = intopt.aux_ao_loc[cp_kl_id+1]
                    int3c2e.get_int3c2e_slice(intopt, cp_ij_id, cp_kl_id, out=ints_slices[k0:k1], omega=omega)
                ints_slices -= ints_slices_lr
        else:
            # Initialization is required due to cutensor operations later
            ints_slices = cupy.zeros([naoaux, nj, ni], order='C')
            for cp_kl_id, _ in enumerate(intopt.aux_log_qs):
                k0 = intopt.aux_ao_loc[cp_kl_id]
                k1 = intopt.aux_ao_loc[cp_kl_id+1]

                int3c2e.get_int3c2e_slice(intopt, cp_ij_id, cp_kl_id, out=ints_slices[k0:k1], omega=omega)
        if lj>1 and not mol.cart: ints_slices = cart2sph(ints_slices, axis=1, ang=lj)
        if li>1 and not mol.cart: ints_slices = cart2sph(ints_slices, axis=2, ang=li)

        i0, i1 = intopt.ao_loc[cpi], intopt.ao_loc[cpi+1]
        j0, j1 = intopt.ao_loc[cpj], intopt.ao_loc[cpj+1]

        row = intopt.ao_pairs_row[cp_ij_id] - i0
        col = intopt.ao_pairs_col[cp_ij_id] - j0

        ints_slices = ints_slices[:,col,row]
        if cd_low.tag == 'eig':
            cderi_block = cupy.dot(cd_low.T, ints_slices)
            ints_slices = None
        elif cd_low.tag == 'cd':
            cderi_block = solve_triangular(cd_low_f, ints_slices, lower=True, overwrite_b=True)
        ij0, ij1 = count, count+cderi_block.shape[1]
        count = ij1
        if isinstance(cderi, cupy.ndarray):
            cderi[:,ij0:ij1] = cderi_block
        else:
            with data_stream:
                for i in range(naux):
                    cderi_block[i].get(out=cderi[i,ij0:ij1])
        t1 = log.timer_debug1(f'solve {cp_ij_id} / {nq}', *t1)
    if not use_gpu_memory:
        cupy.cuda.Device().synchronize()
    return cderi


