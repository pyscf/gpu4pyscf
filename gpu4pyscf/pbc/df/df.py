# Copyright 2024-2025 The PySCF Developers. All Rights Reserved.
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

'''
Density fitting

Divide the 3-center Coulomb integrals to two parts.  Compute the local
part in real space, long range part in reciprocal space.
'''

__all__ = ['GDF']

import warnings
import ctypes
import tempfile
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.pbc.df import aft as aft_cpu
from pyscf.pbc.df.rsdf_builder import estimate_ke_cutoff_for_omega
from pyscf.pbc.df import df as df_cpu
from pyscf.pbc.df.gdf_builder import libpbc
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.tools import k2gamma
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import utils
from gpu4pyscf.lib.cupy_helper import (
    return_cupy_array, pack_tril, get_avail_mem, asarray)
from gpu4pyscf.lib.memcpy import copy_array
from gpu4pyscf.df import df as mol_df
from gpu4pyscf.pbc.df import rsdf_builder, df_jk, df_jk_real
from gpu4pyscf.pbc.df.aft import _check_kpts, AFTDF
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.pbc.lib.kpts_helper import reset_kpts, fft_matrix
from gpu4pyscf.__config__ import num_devices

DEBUG = False


class GDF(lib.StreamObject):
    '''Gaussian density fitting
    '''
    blockdim = df_cpu.GDF.blockdim

    _keys = df_cpu.GDF._keys.union({'is_gamma_point', 'nao'})

    def __init__(self, cell, kpts=None):
        df_cpu.GDF.__init__(self, cell, kpts)
        self.is_gamma_point = False
        self.nao = None

    # Some methods inherited from the molecule code tries to access the .mol attribute
    @property
    def mol(self):
        return self.cell
    @mol.setter
    def mol(self, x):
        self.cell = x

    @property
    def kpts(self):
        if isinstance(self._kpts, KPoints):
            return self._kpts
        else:
            return self.cell.get_abs_kpts(self._kpts)

    @kpts.setter
    def kpts(self, val):
        if val is None:
            self._kpts = np.zeros((1, 3))
        elif isinstance(val, KPoints):
            self._kpts = val
        else:
            self._kpts = self.cell.get_scaled_kpts(val)

    def reset(self, cell=None):
        if cell is not None:
            if isinstance(self._kpts, KPoints):
                self.kpts = reset_kpts(self.kpts, cell)
            self.cell = cell
        self._cderi = self._cderip = self._cderi_idx = None
        self._rsh_df = {}
        return self

    __getstate__, __setstate__ = lib.generate_pickle_methods(
        excludes=('_cderi_to_save', '_cderi', '_cderip', '_cderi_idx', '_rsh_df'),
        reset_state=True)

    auxbasis = df_cpu.GDF.auxbasis

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s ********', self.__class__)
        if self.auxcell is None:
            log.info('auxbasis = %s', self.auxbasis)
        else:
            log.info('auxbasis = %s', self.auxcell.basis)
        log.info('exp_to_discard = %s', self.exp_to_discard)
        #log.info('len(kpts) = %d', len(self.kpts))
        log.info('is_gamma_point = %s', self.is_gamma_point)
        return self

    def build(self, j_only=None, kpts_band=None):
        warnings.warn(
            'PBC.df is currently experimental and subject to significant changes.')
        if j_only is not None:
            self._j_only = j_only
        assert kpts_band is None and self.kpts_band is None

        cell = self.cell
        if cell.dimension < 3 and cell.low_dim_ft_type == 'inf_vacuum':
            raise NotImplementedError

        self.check_sanity()
        self.dump_flags()
        auxcell = df_cpu.make_auxcell(cell, self.auxbasis, self.exp_to_discard)
        self.auxcell = auxcell

        kpts = self.kpts
        if self.is_gamma_point:
            assert kpts is None or is_zero(kpts)
            self.kmesh = np.zeros(3, dtype=int)
        else:
            self.kmesh = kpts_to_kmesh(cell, kpts)

        t1 = (logger.process_clock(), logger.perf_counter())
        self._cderi, self._cderip, self._cderi_idx = rsdf_builder.build_cderi(
            cell, auxcell, kpts, j_only=j_only,
            linear_dep_threshold=self.linear_dep_threshold, compress=True)
        t1 = logger.timer_debug1(self, 'j3c', *t1)
        return self

    has_kpts = df_cpu.GDF.has_kpts
    weighted_coulG = return_cupy_array(aft_cpu.weighted_coulG)
    pw_loop = NotImplemented
    ft_loop = df_cpu.GDF.ft_loop
    range_coulomb = aft_cpu.AFTDFMixin.range_coulomb

    def get_naoaux(self):
        if self._cderi is None:
            self.build(j_only=self._j_only)
        return max(x.shape[0] for x in self._cderi.values())

    def sr_loop(self, blksize, compact=True, aux_iter=None):
        '''Iterator for the 3-index cderi tensor over the auxliary dimension.

        Kwargs:
            compact :
                If compact is specified, the output is a compressed CDERI
                tensor. Otherwise, the output is a dense tensor with shape
                [nkpts,*,nao,nao].
            aux_iter :
                Allows multiple GPU executors to share the producer, dynamically
                loading the tesnor blocks as needed.
        '''
        if self._cderi is None:
            self.build(j_only=self._j_only)
        cell = self.cell
        if aux_iter is None:
            naux = self.get_naoaux()
            aux_iter = lib.prange(0, naux, blksize)
        if not compact:
            kmesh = self.kmesh
            expLk = fft_matrix(kmesh)
            nao = cell.nao
            kk_conserv = k2gamma.double_translation_indices(kmesh)

        for k_aux, p0, p1 in aux_iter:
            out = asarray(self._cderi[k_aux][p0:p1,:])
            if out.size == 0:
                return
            if not compact:
                out = rsdf_builder.unpack_cderi_k(
                    out, self._cderi_idx, k_aux, kk_conserv, expLk, nao)
            yield k_aux, out, 1
            if p0 == 0 and cell.dimension == 2 and k_aux in self._cderip:
                out = asarray(self._cderip[k_aux])
                if not compact:
                    out = rsdf_builder.unpack_cderi_k(
                        out, self._cderi_idx, k_aux, kk_conserv, expLk, nao)
                yield k_aux, out, -1

    def get_pp(self, kpts=None):
        kpts, is_single_kpt = _check_kpts(self, kpts)
        if is_single_kpt and is_zero(kpts):
            vpp = rsdf_builder.get_pp(self.cell)
        else:
            vpp = rsdf_builder.get_pp(self.cell, kpts)
            if is_single_kpt:
                vpp = vpp[0]
        return vpp

    def get_nuc(self, kpts=None):
        kpts, is_single_kpt = _check_kpts(self, kpts)
        if is_single_kpt and is_zero(kpts):
            nuc = rsdf_builder.get_nuc(self.cell)
        else:
            nuc = rsdf_builder.get_nuc(self.cell, kpts)
            if is_single_kpt:
                nuc = nuc[0]
        return nuc

    # Note: Special exxdiv by default should not be used for an arbitrary
    # input density matrix. When the df object was used with the molecular
    # post-HF code, get_jk was often called with an incomplete DM (e.g. the
    # core DM in CASCI). An SCF level exxdiv treatment is inadequate for
    # post-HF methods.
    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        if omega is not None and omega != 0:
            cell = self.cell
            if omega > 0:
                mydf = AFTDF(cell, self.kpts)
                ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
                mydf.mesh = cell.cutoff_to_mesh(ke_cutoff)
            else:
                mydf = self
            with mydf.range_coulomb(omega) as rsh_df:
                return rsh_df.get_jk(dm, hermi, kpts, kpts_band, with_j, with_k,
                                     omega=None, exxdiv=exxdiv)

        if self.is_gamma_point:
            return df_jk_real.get_jk(self, dm, hermi, with_j, with_k, exxdiv)
        else:
            kpts, is_single_kpt = _check_kpts(self, kpts)
            if is_single_kpt:
                return df_jk.get_jk(self, dm, hermi, kpts[0], kpts_band, with_j,
                                    with_k, exxdiv)
        vj = vk = None
        if with_k:
            vk = df_jk.get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
        if with_j:
            vj = df_jk.get_j_kpts(self, dm, hermi, kpts, kpts_band)
        return vj, vk

    get_eri = get_ao_eri = NotImplemented
    ao2mo = get_mo_eri = NotImplemented
    ao2mo_7d = NotImplemented

    get_blksize = mol_df.DF.get_blksize

    # TOOD: refactor and reuse the loop method in the molecule df module
    def loop(self, blksize=None, unpack=True, aux_iter=None):
        ''' loop over cderi and unpack the CDERI in (Lij) format

        Kwargs:
            unpack :
                CDERI tensor is compressed for orbital-pair. If unpack is
                specified, a dense tensor with shape [*,nao,nao] will be
                constructed as the first argument in the output
            aux_iter :
                Allows multiple GPU executors to share the producer, dynamically
                loading the tesnor blocks as needed.
        '''
        assert is_zero(self.kpts)
        cell = self.cell

        if blksize is None:
            blksize = self.get_blksize()
        cderi_sparse = self._cderi[0]
        naux, npairs = cderi_sparse.shape
        if aux_iter is None:
            aux_iter = lib.prange(0, naux, blksize)

        if unpack:
            nao = self.nao
            ao_pair_mapping, diag = self._cderi_idx
            ao_pair_mapping = asarray(ao_pair_mapping)
            rows, cols = divmod(ao_pair_mapping, nao)
            buf_cderi = cp.zeros([blksize,nao,nao])

        buf1 = cp.empty(blksize*npairs, dtype=cderi_sparse.dtype)
        out2 = None
        for p0, p1 in aux_iter:
            out = cderi_sparse[p0:p1,:]
            if not isinstance(cderi_sparse, cp.ndarray):
                out = out.get(out=buf1[:out.size].reshape(out.shape))
            if unpack:
                out2 = buf_cderi[:p1-p0]
                out2[:,cols,rows] = out2[:,rows,cols] = out
            yield out2, out.T

            if p0 == 0 and cell.dimension == 2:
                out = asarray(self._cderip[0])
                if unpack:
                    out2 = buf_cderi[:1]
                    out2[:,cols,rows] = out2[:,rows,cols] = out
                yield out2, out.T

    to_gpu = utils.to_gpu
    device = utils.device

    def to_cpu(self):
        from pyscf.pbc.df.df import GDF
        out = GDF(self.cell, kpts=self.kpts)
        return utils.to_cpu(self, out=out)
