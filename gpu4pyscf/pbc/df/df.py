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
from pyscf.pbc.df import df as df_cpu
from pyscf.pbc.df.gdf_builder import libpbc
from pyscf.pbc.lib.kpts_helper import is_zero
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.df import df_jk, rsdf_builder
from gpu4pyscf.pbc.df.aft import _check_kpts
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.lib.cupy_helper import return_cupy_array, pack_tril, get_avail_mem
from gpu4pyscf.lib import utils

class GDF(lib.StreamObject):
    '''Gaussian density fitting
    '''
    blockdim = df_cpu.GDF.blockdim
    _prefer_ccdf = False
    force_dm_kbuild = False

    _keys = df_cpu.GDF._keys

    __init__ = df_cpu.GDF.__init__

    __getstate__, __setstate__ = lib.generate_pickle_methods(
            excludes=('_cderi_to_save', '_cderi', '_rsh_df'), reset_state=True)

    auxbasis = df_cpu.GDF.auxbasis
    reset = df_cpu.GDF.reset
    dump_flags = df_cpu.GDF.dump_flags

    def build(self, j_only=None, kpts_band=None):
        warnings.warn(
            'PBC.df is currently experimental and subject to significant changes.')
        if j_only is not None:
            self._j_only = j_only
        assert kpts_band is None and self.kpts_band is None

        self.check_sanity()
        self.dump_flags()
        cell = self.cell
        auxcell = df_cpu.make_auxcell(cell, self.auxbasis, self.exp_to_discard)
        self.auxcell = auxcell

        t1 = (logger.process_clock(), logger.perf_counter())
        self._cderi, self._cderip = rsdf_builder.build_cderi(
            cell, auxcell, self.kpts, j_only=j_only)
        t1 = logger.timer_debug1(self, 'j3c', *t1)
        return self

    has_kpts = df_cpu.GDF.has_kpts
    weighted_coulG = return_cupy_array(aft_cpu.weighted_coulG)
    pw_loop = NotImplemented
    ft_loop = df_cpu.GDF.ft_loop
    get_naoaux = df_cpu.GDF.get_naoaux
    range_coulomb = aft_cpu.AFTDFMixin.range_coulomb

    def sr_loop(self, ki, kj, compact=True, blksize=None):
        '''Iterator for the 3-index cderi tensor over the auxliary dimension'''
        if self._cderi is None:
            self.build()
        cell = self.cell
        nao = cell.nao
        if blksize is None:
            avail_mem = get_avail_mem() * .8
            blksize = avail_mem/16/(nao**2*3)
            if blksize < 16:
                raise RuntimeError('Insufficient GPU memory')
            blksize = min(int(blksize), self.blockdim)
            logger.debug2(self, 'max_memory %d MB, blksize %d', avail_mem*1e-6, blksize)

        if (ki, kj) in self._cderi:
            req_conj = False
        elif (kj, ki) in self._cderi:
            req_conj = True
        else:
            raise RuntimeError('CDERI for kpoints {ki},{kj} not generated')

        Lpq_kij = self._cderi[ki,kj]
        naux = len(Lpq_kij)
        for b0, b1 in lib.prange(0, naux, blksize):
            if req_conj:
                Lpq = Lpq_kij[b0:b1].transpose(0,2,1).conj()
            else:
                Lpq = Lpq_kij[b0:b1]
            assert Lpq[0].size == nao**2
            if compact:
                Lpq = pack_tril(Lpq.reshape(-1, nao, nao))
            yield Lpq, 1

        if cell.dimension == 2:
            assert cell.low_dim_ft_type != 'inf_vacuum'
            Lpq_kij = self._cderip[ki,kj]
            naux = len(Lpq_kij)
            for b0, b1 in lib.prange(0, naux, blksize):
                if req_conj:
                    Lpq = Lpq_kij[b0:b1].transpose(0,2,1).conj()
                else:
                    Lpq = Lpq_kij[b0:b1]
                assert Lpq[0].size == nao**2
                if compact:
                    Lpq = pack_tril(Lpq.reshape(-1, nao, nao))
                yield Lpq, -1

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
