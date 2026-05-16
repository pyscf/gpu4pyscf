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
from gpu4pyscf.pbc.gto.pseudo.pp_int import get_pp_nl_gpu
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.df import ft_ao
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.pbc.tools.pbc import get_coulG
from gpu4pyscf.pbc.df import aft_jk
from gpu4pyscf.pbc.df.ft_ao import FTOpt
from gpu4pyscf.pbc.lib.kpts_helper import reset_kpts
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import (return_cupy_array, contract, unpack_tril,
                                       get_avail_mem)

KE_SCALING = aft_cpu.KE_SCALING

def _get_pp_loc_part1(mydf, kpts=None, with_pseudo=True):
    log = logger.new_logger(mydf)
    cell = mydf.cell
    mesh = np.asarray(mydf.mesh)
    is_single_kpt = kpts is not None and kpts.ndim == 1
    if kpts is None:
        kpts = np.zeros((1, 3))
    else:
        kpts = kpts.reshape(-1, 3)
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
    is_single_kpt = kpts is not None and kpts.ndim == 1
    if kpts is None:
        kpts = np.zeros((1, 3))
    else:
        kpts = kpts.reshape(-1, 3)
    vpp = _get_pp_loc_part1(mydf, kpts, with_pseudo=True)
    pp2builder = aft_cpu._IntPPBuilder(cell, kpts)
    vpp += cp.asarray(pp2builder.get_pp_loc_part2())
    vpp += cp.asarray(get_pp_nl_gpu(cell, kpts))
    if is_single_kpt:
        vpp = vpp[0]
    return vpp


def get_nuc(mydf, kpts=None):
    '''Get the periodic nuc-el AO matrix, with G=0 removed.

    Kwargs:
        function _guess_eta from module pbc.df.gdf_builder.
    '''
    return _get_pp_loc_part1(mydf, kpts, with_pseudo=False)


class AFTDF(lib.StreamObject):
    '''Density expansion on plane waves
    '''

    time_reversal_symmetry = True

    _keys = aft_cpu.AFTDF._keys

    def __init__(self, cell, kpts=None):
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory
        self.mesh = cell.mesh
        if cell.omega > 0:
            ke_cutoff = aft_cpu.estimate_ke_cutoff_for_omega(cell, cell.omega)
            self.mesh = cell.cutoff_to_mesh(ke_cutoff)
        self.kpts = kpts

        # The following attributes are not input options.
        # self.exxdiv has no effects. It was set in the get_k_kpts function to
        # mimic the KRHF/KUHF object in the call to tools.get_coulG.
        self.exxdiv = None
        self._rsh_df = {}  # Range separated Coulomb DF objects

    dump_flags = aft_cpu.AFTDF.dump_flags
    check_sanity = aft_cpu.AFTDF.check_sanity
    build = aft_cpu.AFTDF.build

    get_nuc = get_nuc
    get_pp = get_pp

    __getstate__, __setstate__ = lib.generate_pickle_methods(
        excludes=('_rsh_df',))

    @property
    def kpts(self):
        if isinstance(self._kpts, KPoints):
            return self._kpts
        else:
            return self.cell.get_abs_kpts(cp.asnumpy(self._kpts))

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

    pw_loop = NotImplemented

    def weighted_coulG(self, kpt=None, exx=None, mesh=None, omega=None,
                       kpts=None, lr_factor=1, sr_factor=1):
        '''Weighted Coulomb kernel'''
        cell = self.cell
        if mesh is None:
            mesh = self.mesh
        if omega is None:
            omega = cell.omega
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)

        if lr_factor == sr_factor:
            coulG = get_coulG(cell, kpt, exx, mesh=mesh, Gv=Gv,
                              wrap_around=True, omega=omega, kpts=kpts)
            if lr_factor is not None and lr_factor != 1:
                coulG *= kws * lr_factor
            else:
                coulG *= kws
            return coulG

        assert omega > 0
        if lr_factor == 0:
            coulG = get_coulG(cell, kpt, exx, mesh=mesh, Gv=Gv,
                              wrap_around=True, omega=-omega, kpts=kpts)
            coulG *= sr_factor * kws
        if sr_factor == 0:
            coulG = get_coulG(cell, kpt, exx, mesh=mesh, Gv=Gv,
                              wrap_around=True, omega=omega, kpts=kpts)
            coulG *= lr_factor * kws
        else:
            coulG = get_coulG(cell, kpt, exx, mesh=mesh, Gv=Gv,
                              wrap_around=True, omega=0., kpts=kpts)
            coulG_LR = get_coulG(cell, kpt, exx, mesh=mesh, Gv=Gv,
                                 wrap_around=True, omega=omega, kpts=kpts)
            coulG -= coulG_LR
            coulG *= sr_factor
            coulG += coulG_LR * lr_factor
            coulG *= kws
        return coulG

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
        if bvk_kmesh is None:
            bvk_kmesh = kpts_to_kmesh(cell, kpts, bound_by_supmol=True)
        if kpts is None:
            assert is_zero(q)
            kpts = self.kpts

        ft_opt = FTOpt(cell, bvk_kmesh).build()
        ft_kern = ft_opt.gen_ft_kernel(transform_ao=transform_ao)

        bvk_ncells = len(ft_opt.bvkmesh_Ls)
        nao = ft_opt.cell.nao
        Gv = cell.get_Gv(mesh)
        ngrids = len(Gv)

        mem_free = cp.cuda.runtime.memGetInfo()[0]
        avail_mem = mem_free * .8
        # the memory estimation is determined by the size of the intermediates
        # in the ft_kern
        blksize = int(avail_mem/(nao**2*bvk_ncells*16*2)) // 32 * 32
        if blksize == 0:
            raise RuntimeError('Insufficient GPU memory')
        blksize = min(blksize, ngrids)

        for p0, p1 in lib.prange(0, ngrids, blksize):
            dat = ft_kern(Gv[p0:p1], q, kpts)
            yield dat, p0, p1

    @contextlib.contextmanager
    def range_coulomb(self, omega):
        '''Creates a temporary density fitting object for RSH-DF integrals.
        In this context, only LR or SR integrals for mol and auxmol are computed.
        '''
        if omega is None or omega == 0:
            yield self
            return

        key = '%.6f' % omega
        if key in self._rsh_df:
            rsh_df = self._rsh_df[key]
        else:
            rsh_df = self._rsh_df[key] = self.copy().reset()
            logger.info(self, 'Create RSH-DF object %s for omega=%s', rsh_df, omega)

        cell = self.cell
        auxcell = getattr(self, 'auxcell', None)

        cell_omega = cell.omega
        cell.omega = omega
        auxcell_omega = None
        if auxcell is not None:
            auxcell_omega = auxcell.omega
            auxcell.omega = omega

        assert rsh_df.cell.omega == omega
        if getattr(rsh_df, 'auxcell', None) is not None:
            assert rsh_df.auxcell.omega == omega

        try:
            yield rsh_df
        finally:
            cell.omega = cell_omega
            if auxcell_omega is not None:
                auxcell.omega = auxcell_omega

    # Note: Special exxdiv by default should not be used for an arbitrary
    # input density matrix. When the df object was used with the molecular
    # post-HF code, get_jk was often called with an incomplete DM (e.g. the
    # core DM in CASCI). An SCF level exxdiv treatment is inadequate for
    # post-HF methods.
    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        kpts, is_single_kpt = _check_kpts(kpts, dm)
        if is_single_kpt:
            return aft_jk.get_jk(self, dm, hermi, kpts[0], kpts_band, with_j,
                                  with_k, exxdiv, omega)

        vj = vk = None
        if with_k:
            vk = aft_jk.get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv,
                                   omega=omega)
        if with_j:
            vj = aft_jk.get_j_kpts(self, dm, hermi, kpts, kpts_band)
        return vj, vk

    get_j_e1 = aft_jk.get_ej_ip1
    get_k_e1 = aft_jk.get_ek_ip1
    get_jk_e1 = NotImplemented

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

def _check_kpts(kpts, dm):
    '''Check if the argument kpts is a single k-point'''
    is_single_kpt =True
    if kpts is None:
        kpts = np.zeros((1, 3))
        if (dm.ndim == 2 or # RHF
            (dm.ndim == 3 and len(dm) == 2)): # UHF
            return kpts, is_single_kpt

    if kpts.ndim == 1:
        kpts = kpts.reshape(1, 3)
        assert (dm.ndim == 2 or # RHF
                (dm.ndim == 3 and len(dm) == 2)) # UHF
    else:
        is_single_kpt = False
        nkpts = len(kpts)
        if dm.ndim == 2:
            raise RuntimeError('dm.ndim == 2, incompatible with kpts')
        elif dm.ndim == 3: # KRHF
            assert len(dm) == nkpts, 'KRHF dm incompatible with kpts. Are you running UHF?'
        else: # KUHF
            assert dm.shape[:2] == (2, nkpts), 'KUHF dm incompatible with kpts'
    return kpts, is_single_kpt
