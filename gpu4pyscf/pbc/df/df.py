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
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import utils
from gpu4pyscf.lib.cupy_helper import (
    return_cupy_array, pack_tril, get_avail_mem, asarray)
from gpu4pyscf.lib.memcpy import copy_array
from gpu4pyscf.df import df as mol_df
from gpu4pyscf.pbc.df import rsdf_builder, df_jk, df_jk_real
from gpu4pyscf.pbc.df.aft import _check_kpts, AFTDF
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.pbc.lib.kpts_helper import reset_kpts
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

        t1 = (logger.process_clock(), logger.perf_counter())
        if self.is_gamma_point:
            with_long_range = cell.omega == 0
            if with_long_range:
                cell_exps, cs = rsdf_builder.extract_pgto_params(cell, 'diffused')
                omega = cell_exps.min()**.5
                logger.debug(cell, 'omega guess for rsdf_builder = %g', omega)
            else:
                assert cell.omega < 0
                omega = abs(cell.omega)
            if DEBUG:
                cderi, cderip = \
                    rsdf_builder.build_cderi_gamma_point(
                        cell, auxcell, omega, with_long_range,
                        self.linear_dep_threshold)
                nao = cell.nao
                rows, cols = np.tril_indices(nao)
                diag_idx = np.arange(nao)
                diag_idx = diag_idx*(diag_idx+1)//2 + diag_idx
                cderi = cderi.popitem()[1]
                cderi = cderi[:, rows, cols]
                self._cderi_idx = rows, cols, diag_idx
            else:
                cderi, self._cderip, self._cderi_idx = \
                    rsdf_builder.compressed_cderi_gamma_point(
                        cell, auxcell, omega, with_long_range,
                        self.linear_dep_threshold)
            self._cderi = [None] * num_devices
            self.nao = cell.nao
            if num_devices == 1:
                self._cderi[0] = cderi
            else:
                # Distribute cderi to other devices
                naux = len(cderi)
                blksize = (naux + num_devices - 1) // num_devices
                ALIGNED = mol_df.ALIGNED
                blksize = (blksize + ALIGNED - 1) // ALIGNED * ALIGNED
                for dev_id in range(num_devices):
                    p0 = dev_id * blksize
                    p1 = min(p0 + blksize, naux)
                    tmp = cp.asarray(cderi[p0:p1], order='C')
                    if dev_id == 0:
                        self._cderi[0] = tmp
                        continue
                    with cp.cuda.Device(dev_id):
                        self._cderi[dev_id] = copy_array(tmp)
        else:
            self._cderi, self._cderip = rsdf_builder.build_cderi(
                cell, auxcell, self.kpts, j_only=j_only,
                linear_dep_threshold=self.linear_dep_threshold)
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
    def loop(self, blksize=None, unpack=True):
        ''' loop over cderi for the current device
            and unpack the CDERI in (Lij) format
        '''
        device_id = cp.cuda.Device().id
        cderi_sparse = self._cderi[device_id]
        naux_slice = len(cderi_sparse)
        if blksize is None:
            blksize = self.get_blksize()
        if unpack:
            nao = self.nao
            rows, cols, diag = self._cderi_idx
            rows = cp.asarray(rows)
            cols = cp.asarray(cols)
            buf_cderi = cp.zeros([blksize,nao,nao])

        for p0, p1 in lib.prange(0, naux_slice, blksize):
            if isinstance(cderi_sparse, cp.ndarray):
                buf = cderi_sparse[p0:p1,:]
            else:
                buf = asarray(cderi_sparse[p0:p1,:])
            if unpack:
                buf2 = buf_cderi[:p1-p0]
                buf2[:,cols,rows] = buf2[:,rows,cols] = buf
            else:
                buf2 = None
            yield buf2, buf.T

    to_gpu = utils.to_gpu
    device = utils.device

    def to_cpu(self):
        from pyscf.pbc.df.df import GDF
        out = GDF(self.cell, kpts=self.kpts)
        return utils.to_cpu(self, out=out)
