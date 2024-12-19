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
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df.aft import _check_kpts
from pyscf.pbc.tools import k2gamma
from gpu4pyscf.pbc.tools.pbc import get_coulG
from gpu4pyscf.pbc.df import aft_jk
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import return_cupy_array, contract, unpack_tril

KE_SCALING = aft_cpu.KE_SCALING

def _get_pp_loc_part1(mydf, kpts=None, with_pseudo=True):
    kpts, is_single_kpt = _check_kpts(mydf, kpts)
    log = logger.new_logger(mydf)
    cell = mydf.cell
    mesh = np.asarray(mydf.mesh)
    nkpts = len(kpts)
    nao = cell.nao_nr()
    nao_pair = nao * (nao+1) // 2

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
    vj = cp.zeros((nkpts, nao_pair), dtype=np.complex128)
    for Gpq, p0, p1 in mydf.ft_loop(mesh, kpt_allow, kpts, aosym='s2'):
        vj += contract('kGx,G->kx', Gpq, vpplocG[p0:p1].conj())

    vj_kpts = unpack_tril(vj)
    if is_zero(kpts):
        vj_kpts = vj_kpts.real
    if is_single_kpt:
        vj_kpts = vj_kpts[0]
    return vj_kpts

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

    def ft_loop(self, mesh=None, q=np.zeros(3), kpts=None, shls_slice=None,
                max_memory=4000, aosym='s1', intor='GTO_ft_ovlp', comp=1,
                bvk_kmesh=None, return_complex=True):
        '''
        Fourier transform iterator for all kpti which satisfy
            2pi*N = (kpts - kpti - q)*a,  N = -1, 0, 1
        The tensors returned by this function is different to the one in PySCF CPU version
        '''
        assert return_complex
        cell = self.cell
        if mesh is None:
            mesh = self.mesh
        if kpts is None:
            assert (is_zero(q))
            kpts = self.kpts
        kpts = np.asarray(kpts)
        nkpts = len(kpts)

        nao = cell.nao
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        ngrids = gxyz.shape[0]

        assert shls_slice is None
        if aosym == 's2':
            nij = nao * (nao+1) // 2
        else:
            nij = nao * nao

        if bvk_kmesh is None:
            bvk_kmesh = k2gamma.kpts_to_kmesh(cell, kpts)

        rcut = ft_ao.estimate_rcut(cell)
        supmol = ft_ao.ExtendedMole.from_cell(cell, bvk_kmesh, rcut.max())
        supmol = supmol.strip_basis(rcut)
        ft_kern = supmol.gen_ft_kernel(aosym, intor=intor, comp=comp,
                                       return_complex=True)

        blksize = max(16, int(max_memory*.9e6/(nij*nkpts*16*comp)))
        blksize = min(blksize, ngrids, 16384)

        for p0, p1 in lib.prange(0, ngrids, blksize):
            dat = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, q, kpts, shls_slice)
            yield cp.asarray(dat), p0, p1

    range_coulomb = aft_cpu.AFTDFMixin.range_coulomb


class AFTDF(lib.StreamObject, AFTDFMixin):
    '''Density expansion on plane waves
    '''

    _keys = aft_cpu.AFTDF._keys

    __init__ = aft_cpu.AFTDF.__init__
    dump_flags = aft_cpu.AFTDF.dump_flags
    reset = aft_cpu.AFTDF.reset
    check_sanity = aft_cpu.AFTDF.check_sanity
    build = aft_cpu.AFTDF.build

    get_nuc = get_nuc
    get_pp = get_pp

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
    to_cpu = utils.to_cpu
