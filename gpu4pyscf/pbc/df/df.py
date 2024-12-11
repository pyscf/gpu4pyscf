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

'''
Density fitting

Divide the 3-center Coulomb integrals to two parts.  Compute the local
part in real space, long range part in reciprocal space.
'''

__all__ = ['GDF']

import ctypes
import tempfile
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.pbc.df import aft as aft_cpu
from pyscf.pbc.df import df as df_cpu
from pyscf.pbc.df.aft import _check_kpts
from pyscf.pbc.df.gdf_builder import libpbc
from pyscf.pbc.lib.kpts_helper import is_zero, unique
from pyscf.pbc.df.rsdf_builder import _RSGDFBuilder, _RSNucBuilder
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.df import df_jk
from gpu4pyscf.lib.cupy_helper import return_cupy_array, pack_tril, unpack_tril
from gpu4pyscf.lib import utils

class GDF(lib.StreamObject):
    '''Gaussian density fitting
    '''
    blockdim = df_cpu.GDF.blockdim
    _dataname = 'j3c'
    _prefer_ccdf = False
    force_dm_kbuild = False

    _keys = df_cpu.GDF._keys

    __init__ = df_cpu.GDF.__init__

    __getstate__, __setstate__ = lib.generate_pickle_methods(
            excludes=('_cderi_to_save', '_cderi', '_rsh_df'), reset_state=True)

    auxbasis = df_cpu.GDF.auxbasis
    reset = df_cpu.GDF.reset
    dump_flags = df_cpu.GDF.dump_flags

    def build(self, j_only=None, with_j3c=True, kpts_band=None):
        if j_only is not None:
            self._j_only = j_only
        if self.kpts_band is not None:
            self.kpts_band = np.reshape(self.kpts_band, (-1,3))
        assert kpts_band is None

        self.check_sanity()
        self.dump_flags()

        self.auxcell = df_cpu.make_auxcell(self.cell, self.auxbasis,
                                           self.exp_to_discard)

        if with_j3c and self._cderi_to_save is not None:
            if isinstance(self._cderi_to_save, str):
                cderi = self._cderi_to_save
            else:
                cderi = self._cderi_to_save.name
            self._cderi = cderi
            t1 = (logger.process_clock(), logger.perf_counter())
            self._make_j3c(self.cell, self.auxcell, None, cderi)
            t1 = logger.timer_debug1(self, 'j3c', *t1)
        return self

    def _make_j3c(self, cell=None, auxcell=None, kptij_lst=None, cderi_file=None):
        if cell is None: cell = self.cell
        if auxcell is None: auxcell = self.auxcell
        if cderi_file is None: cderi_file = self._cderi_to_save

        # Remove duplicated k-points. Duplicated kpts may lead to a buffer
        # located in incore.wrap_int3c larger than necessary. Integral code
        # only fills necessary part of the buffer, leaving some space in the
        # buffer unfilled.
        if self.kpts_band is None:
            kpts_union = self.kpts
        else:
            kpts_union = unique(np.vstack([self.kpts, self.kpts_band]))[0]

        dfbuilder = _RSGDFBuilder(cell, auxcell, kpts_union)
        dfbuilder.mesh = self.mesh
        dfbuilder.linear_dep_threshold = self.linear_dep_threshold
        j_only = self._j_only or len(kpts_union) == 1
        dfbuilder.make_j3c(cderi_file, j_only=j_only, dataname=self._dataname,
                           kptij_lst=kptij_lst)

    has_kpts = df_cpu.GDF.has_kpts
    weighted_coulG = return_cupy_array(aft_cpu.weighted_coulG)
    pw_loop = NotImplemented
    ft_loop = df_cpu.GDF.ft_loop
    get_naoaux = df_cpu.GDF.get_naoaux
    range_coulomb = aft_cpu.AFTDFMixin.range_coulomb

    def sr_loop(self, kpti_kptj=np.zeros((2,3)), max_memory=2000,
                compact=True, blksize=None, aux_slice=None):
        '''Short range part'''
        assert aux_slice is None
        if self._cderi is None:
            self.build()
        cell = self.cell
        kpti, kptj = kpti_kptj
        unpack = is_zero(kpti-kptj) and not compact
        nao = cell.nao
        if blksize is None:
            blksize = max_memory*1e6/16/(nao**2*2)
            blksize /= 2  # For prefetch
            blksize = max(16, min(int(blksize), self.blockdim))
            logger.debug2(self, 'max_memory %d MB, blksize %d', max_memory, blksize)

        def load(aux_slice):
            b0, b1 = aux_slice
            naux = b1 - b0
            Lpq = cp.asarray(j3c[b0:b1])
            if compact and Lpq.shape[1] == nao**2:
                Lpq = pack_tril(Lpq.reshape(naux, nao, nao))
            elif unpack and Lpq.shape[1] != nao**2:
                Lpq = unpack_tril(Lpq)
            return Lpq

        with df_cpu._load3c(self._cderi, self._dataname, kpti_kptj) as j3c:
            slices = lib.prange(0, j3c.shape[0], blksize)
            for Lpq in lib.map_with_prefetch(load, slices):
                yield Lpq, 1

        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            # Truncated Coulomb operator is not positive definite. Load the
            # CDERI tensor of negative part.
            with df_cpu._load3c(self._cderi, self._dataname+'-', kpti_kptj,
                                ignore_key_error=True) as j3c:
                slices = lib.prange(0, j3c.shape[0], blksize)
                for Lpq in lib.map_with_prefetch(load, slices):
                    yield Lpq, -1

    get_pp = return_cupy_array(df_cpu.GDF.get_pp)
    get_nuc = return_cupy_array(df_cpu.GDF.get_nuc)

    # Note: Special exxdiv by default should not be used for an arbitrary
    # input density matrix. When the df object was used with the molecular
    # post-HF code, get_jk was often called with an incomplete DM (e.g. the
    # core DM in CASCI). An SCF level exxdiv treatment is inadequate for
    # post-HF methods.
    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        if omega is not None:  # J/K for RSH functionals
            raise NotImplementedError

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

    to_gpu = utils.to_gpu
    device = utils.device
    to_cpu = utils.to_cpu
