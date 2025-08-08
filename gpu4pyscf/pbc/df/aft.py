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

'''Density expansion on plane waves'''

__all__ = [
    'get_pp', 'get_nuc', 'AFTDF'
]

import contextlib
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf import gto
from pyscf.pbc.df import aft as aft_cpu
from pyscf.pbc.gto.pseudo import pp_int
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.df import ft_ao
from pyscf.pbc.tools import k2gamma
from gpu4pyscf.pbc.tools.pbc import get_coulG
from gpu4pyscf.pbc.df import aft_jk
from gpu4pyscf.pbc.df.ft_ao import FTOpt
from gpu4pyscf.pbc.lib.kpts_helper import reset_kpts
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import (return_cupy_array, contract, unpack_tril,
                                       get_avail_mem)

KE_SCALING = aft_cpu.KE_SCALING

def _get_pp_loc_part1(mydf, kpts=None, with_pseudo=True):
    kpts, is_single_kpt = _check_kpts(mydf, kpts)
    log = logger.new_logger(mydf)
    cell = mydf.cell
    mesh = np.asarray(mydf.mesh)

    kpt_allow = np.zeros(3)
    if cell.dimension > 0:
        ke_guess = aft_cpu.estimate_ke_cutoff(cell, cell.precision)
        mesh_guess = cell.cutoff_to_mesh(ke_guess)
        if np.any(mesh < mesh_guess*KE_SCALING):
            logger.warn(mydf, 'mesh %s is not enough for AFTDF.get_nuc function '
                        'to get integral accuracy %g.\nRecommended mesh is %s.',
                        mesh, cell.precision, mesh_guess)
    log.debug1('aft.get_pp_loc_part1 mesh = %s', mesh)
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    assert len(Gv) > 0

    if with_pseudo:
        vpplocG = pp_int.get_gth_vlocG_part1(cell, Gv)
        vpplocG = -np.einsum('ij,ij->j', cell.get_SI(Gv), vpplocG)
        vpplocG = cp.asarray(vpplocG)
    else:
        fakenuc = aft_cpu._fake_nuc(cell, with_pseudo=with_pseudo)
        aoaux = cp.asarray(ft_ao.ft_ao(fakenuc, Gv))
        charges = cp.asarray(cell.atom_charges(), dtype=np.float64)
        coulG = get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv)
        vpplocG = contract('i,xi->x', -charges, aoaux)
        vpplocG *= coulG

    vpplocG *= cp.asarray(kws)
    vj = 0.
    for Gpq, p0, p1 in mydf.ft_loop(mesh, kpt_allow, kpts):
        vj += contract('kGpq,G->kpq', Gpq, vpplocG[p0:p1].conj())

    if is_zero(kpts):
        vj = vj.real
    if is_single_kpt:
        vj = vj[0]
    return vj

def get_pp(mydf, kpts=None):
    '''Get the periodic pseudopotential nuc-el AO matrix, with G=0 removed.

    Kwargs:
        mesh: custom mesh grids. By default mesh is determined by the
        function _guess_eta from module pbc.df.gdf_builder.
    '''
    cell = mydf.cell
    kpts, is_single_kpt = aft_cpu._check_kpts(mydf, kpts)
    vpp = _get_pp_loc_part1(mydf, kpts, with_pseudo=True)
    pp2builder = aft_cpu._IntPPBuilder(cell, kpts)
    vpp += cp.asarray(pp2builder.get_pp_loc_part2())
    vpp += cp.asarray(pp_int.get_pp_nl(cell, kpts))
    if is_single_kpt:
        vpp = vpp[0]
    return vpp


def get_nuc(mydf, kpts=None):
    '''Get the periodic nuc-el AO matrix, with G=0 removed.

    Kwargs:
        function _guess_eta from module pbc.df.gdf_builder.
    '''
    return _get_pp_loc_part1(mydf, kpts, with_pseudo=False)


class AFTDFMixin:

    weighted_coulG = return_cupy_array(aft_cpu.weighted_coulG)
    pw_loop = NotImplemented

    def ft_loop(self, mesh=None, q=np.zeros(3), kpts=None, bvk_kmesh=None,
                max_memory=None, transform_ao=True, **kwargs):
        '''
        Fourier transform iterator for all kpti which satisfy
            2pi*N = (kpts - kpti - q)*a,  N = -1, 0, 1
        The tensors returned by this function is different to the one in PySCF CPU version
        '''
        cell = self.cell
        if mesh is None:
            mesh = self.mesh
        if kpts is None:
            assert is_zero(q)
            kpts = self.kpts

        ft_opt = FTOpt(cell, kpts, bvk_kmesh)
        ft_kern = ft_opt.gen_ft_kernel()

        if ft_opt.bvk_kmesh is None:
            bvk_ncells = 1
        else:
            bvk_ncells = np.prod(ft_opt.bvk_kmesh)

        nao = ft_opt.sorted_cell.nao
        Gv = cell.get_Gv(mesh)
        ngrids = len(Gv)

        if max_memory is None:
            avail_mem = get_avail_mem() * .8
        else:
            avail_mem = max_memory * 1e6
        # the memory estimation is determined by the size of the intermediates
        # in the ft_kern
        blksize = max(16, int(avail_mem/(nao**2*bvk_ncells*16*2)))
        blksize = min(blksize, ngrids, 16384)

        for p0, p1 in lib.prange(0, ngrids, blksize):
            dat = ft_kern(Gv[p0:p1], q, kpts, transform_ao)
            yield dat, p0, p1

    range_coulomb = aft_cpu.AFTDFMixin.range_coulomb


class AFTDF(lib.StreamObject, AFTDFMixin):
    '''Density expansion on plane waves
    '''

    _keys = aft_cpu.AFTDF._keys

    __init__ = aft_cpu.AFTDF.__init__
    dump_flags = aft_cpu.AFTDF.dump_flags
    check_sanity = aft_cpu.AFTDF.check_sanity
    build = aft_cpu.AFTDF.build

    get_nuc = get_nuc
    get_pp = get_pp

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

    # Note: Special exxdiv by default should not be used for an arbitrary
    # input density matrix. When the df object was used with the molecular
    # post-HF code, get_jk was often called with an incomplete DM (e.g. the
    # core DM in CASCI). An SCF level exxdiv treatment is inadequate for
    # post-HF methods.
    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        if omega is not None:  # J/K for RSH functionals
            with self.range_coulomb(omega) as rsh_df:
                return rsh_df.get_jk(dm, hermi, kpts, kpts_band, with_j, with_k,
                                     omega=None, exxdiv=exxdiv)

        kpts, is_single_kpt = _check_kpts(self, kpts)
        if is_single_kpt:
            return aft_jk.get_jk(self, dm, hermi, kpts[0], kpts_band, with_j,
                                  with_k, exxdiv)

        vj = vk = None
        if with_k:
            vk = aft_jk.get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
        if with_j:
            vj = aft_jk.get_j_kpts(self, dm, hermi, kpts, kpts_band)
        return vj, vk

    get_eri = get_ao_eri = NotImplemented
    ao2mo = get_mo_eri = NotImplemented
    ao2mo_7d = NotImplemented
    get_ao_pairs_G = get_ao_pairs = NotImplemented
    get_mo_pairs_G = get_mo_pairs = NotImplemented

    to_gpu = utils.to_gpu
    device = utils.device

    def to_cpu(self):
        from pyscf.pbc.df.aft import AFTDF
        out = AFTDF(self.cell, kpts=self.kpts)
        return utils.to_cpu(self, out=out)

def _check_kpts(mydf, kpts):
    '''Check if the argument kpts is a single k-point'''
    if kpts is None:
        kpts = getattr(mydf, 'kpts', None)
    if kpts is None:
        kpts = np.zeros((1, 3))
        is_single_kpt = True
    else:
        kpts = np.asarray(kpts)
        is_single_kpt = kpts.ndim == 1
    kpts = kpts.reshape(-1,3)
    return kpts, is_single_kpt
