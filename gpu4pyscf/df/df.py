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
from pyscf import lib, __config__
from pyscf.df import df, addons
from gpu4pyscf.lib.cupy_helper import cart2sph, get_avail_mem, tag_array, solve_triangular, cholesky
from gpu4pyscf.df import int3c2e, df_jk
from gpu4pyscf.lib import logger
from gpu4pyscf import __config__

MIN_BLK_SIZE = getattr(__config__, 'min_ao_blksize', 128)
ALIGNED = getattr(__config__, 'ao_aligned', 32)
LINEAR_DEP_TOL = 1e-8

class DF(df.DF):
    device = 'gpu'
    def __init__(self, mol, auxbasis=None):
        super().__init__(mol, auxbasis)
        self.auxmol = None
        self.intopt = None
        self.nao = None
        self.naux = None
        self.cd_low = None
        self.intopt = None
        self._cderi = None
    
    def build(self, direct_scf_tol=1e-14, omega=None):
        mol = self.mol
        auxmol = self.auxmol
        self.nao = mol.nao
        
        # cache indices for better performance
        nao = mol.nao
        tril_row, tril_col = cupy.tril_indices(nao)
        tril_row = cupy.asarray(tril_row)
        tril_col = cupy.asarray(tril_col)
        
        self.tril_row = tril_row
        self.tril_col = tril_col
        
        idx = np.arange(nao)
        self.diag_idx = cupy.asarray(idx*(idx+1)//2+idx)
        
        t0 = (logger.process_clock(), logger.perf_counter())
        log = logger.new_logger(mol, mol.verbose)
        if auxmol is None:
            self.auxmol = auxmol = addons.make_auxmol(mol, self.auxbasis)
        
        if omega and omega > 1e-10:
            with auxmol.with_range_coulomb(omega):
                j2c_cpu = auxmol.intor('int2c2e', hermi=1)
        else:
            j2c_cpu = auxmol.intor('int2c2e', hermi=1)
        j2c = cupy.asarray(j2c_cpu)
        t0 = log.timer_debug1('2c2e', *t0)    
        intopt = int3c2e.VHFOpt(mol, auxmol, 'int2e')
        intopt.build(direct_scf_tol, diag_block_with_triu=False, aosym=True, group_size=128)
        t1 = log.timer_debug1('prepare intopt', *t0)
        self.j2c = j2c.copy()
        j2c = j2c[cupy.ix_(intopt.sph_aux_idx, intopt.sph_aux_idx)]
        try:
            self.cd_low = cholesky(j2c)
            self.cd_low = tag_array(self.cd_low, tag='cd')
        except:
            w, v = cupy.linalg.eigh(j2c)
            idx = w > LINEAR_DEP_TOL
            self.cd_low = (v[:,idx] / cupy.sqrt(w[idx]))
            self.cd_low = tag_array(self.cd_low, tag='eig')
        
        v = w = None
        naux = self.naux = self.cd_low.shape[1]
        log.debug('size of aux basis %d', naux)
        
        # TODO: calculate short-range CDERI and long-range CDERI separately
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
    
    def get_blksize(self, extra=0):
        '''
        extra for pre-calculated space for other variables
        '''
        nao = self.nao
        mem_avail = get_avail_mem()
        blksize = int(mem_avail*0.2/8/(nao*nao + extra) / ALIGNED) * ALIGNED
        blksize = min(blksize, MIN_BLK_SIZE)
        log = logger.new_logger(self.mol, self.mol.verbose)
        log.debug(f"GPU Memory {mem_avail/1e9:.3f} GB available, block size = {blksize}")
        if blksize < ALIGNED:
            raise RuntimeError("Not enough GPU memory")
        return blksize

    def loop(self, blksize=None):
        cderi = self._cderi
        naux = self.naux
        if blksize is None:
            blksize = self.get_blksize()
        
        for p0, p1 in lib.prange(0, naux, blksize):
            cderi_tril = cderi[p0:p1]
            yield p0, p1, cderi_tril
    
    def reset(self, mol=None):
        '''
        reset object for scanner
        '''
        if mol is not None:
            self.mol = mol
        self.auxmol = None
        self._cderi = None
        self._rsh_df = {}
        self.intopt = None
        return self

def cholesky_eri_gpu(intopt, mol, auxmol, cd_low, omega=None, sr_only=False):
    '''
    Returns:
        2D array of (naux,nao*(nao+1)/2) in C-contiguous
    '''
    nao = mol.nao
    naoaux, naux = cd_low.shape
    npair = (nao + 1) * nao //2
    log = logger.new_logger(mol, mol.verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    
    nq = len(intopt.log_qs)
    avail_mem = get_avail_mem()
    if naux * npair * 8 < 0.7 * avail_mem:
        cderi = cupy.empty([naux, npair], order='C')
    else:
        raise RuntimeError("Not enough GPU memory")
    row, col = np.tril_indices(nao)
    indices_matrix = cupy.zeros([nao, nao], dtype=cupy.int32, order='C')
    indices_matrix[row, col] = cupy.arange(len(row), dtype=cupy.int32)
    indices_matrix[col, row] = cupy.arange(len(row), dtype=cupy.int32)
    
    nq = len(intopt.log_qs)
    for cp_ij_id, _ in enumerate(intopt.log_qs):
        t1 = (logger.process_clock(), logger.perf_counter())
        cpi = intopt.cp_idx[cp_ij_id]
        cpj = intopt.cp_jdx[cp_ij_id]
        li = intopt.angular[cpi]
        lj = intopt.angular[cpj]
        i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi+1]
        j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj+1]
        ni = i1 - i0; nj = j1 - j0
        if sr_only:
            # TODO: in-place implementation or short-range kernel
            ints_slices = cupy.zeros([naoaux, nj, ni], order='C')
            for cp_kl_id, _ in enumerate(intopt.aux_log_qs):
                k0 = intopt.sph_aux_loc[cp_kl_id]
                k1 = intopt.sph_aux_loc[cp_kl_id+1]
                int3c2e.get_int3c2e_slice(intopt, cp_ij_id, cp_kl_id, out=ints_slices[k0:k1])
            if omega is not None:
                ints_slices_lr = cupy.zeros([naoaux, nj, ni], order='C')
                for cp_kl_id, _ in enumerate(intopt.aux_log_qs):
                    k0 = intopt.sph_aux_loc[cp_kl_id]
                    k1 = intopt.sph_aux_loc[cp_kl_id+1]
                    int3c2e.get_int3c2e_slice(intopt, cp_ij_id, cp_kl_id, out=ints_slices[k0:k1], omega=omega)
                ints_slices -= ints_slices_lr
        else:
            ints_slices = cupy.zeros([naoaux, nj, ni], order='C')
            for cp_kl_id, _ in enumerate(intopt.aux_log_qs):
                k0 = intopt.sph_aux_loc[cp_kl_id]
                k1 = intopt.sph_aux_loc[cp_kl_id+1]
                int3c2e.get_int3c2e_slice(intopt, cp_ij_id, cp_kl_id, out=ints_slices[k0:k1], omega=omega)

        if lj>1: ints_slices = cart2sph(ints_slices, axis=1, ang=lj)
        if li>1: ints_slices = cart2sph(ints_slices, axis=2, ang=li)
        
        i0, i1 = intopt.sph_ao_loc[cpi], intopt.sph_ao_loc[cpi+1]
        j0, j1 = intopt.sph_ao_loc[cpj], intopt.sph_ao_loc[cpj+1]
        
        ji_matrix = indices_matrix[j0:j1, i0:i1]
        if cpi == cpj:
            row, col = np.tril_indices(j1-j0)
            ints_slices = ints_slices + ints_slices.transpose([0,2,1])
            ints_slices = ints_slices[:, col, row]
            row_sph, col_sph = np.tril_indices(ji_matrix.shape[0])
            ji = ji_matrix[row_sph, col_sph]
        else:
            ints_slices = ints_slices.reshape([naoaux,-1], order='C')
            ji = ji_matrix.ravel()
        
        if cd_low.tag == 'eig':
            cderi[:, ji] = cupy.dot(cd_low.T, ints_slices)
        elif cd_low.tag == 'cd':
            cderi[:, ji] = solve_triangular(cd_low, ints_slices)

        ji_matrix = ji = ints_slices = None
        t1 = log.timer_debug1(f'solve {cp_ij_id} / {nq}', *t1)
    return cderi


