#!/usr/bin/env python
#
# Copyright 2024 The GPU4PySCF Developers. All Rights Reserved.
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

'''GPW method'''

import numpy as np
import cupy as cp
from pyscf import gto
from pyscf import lib
from pyscf.pbc.df import fft as fft_cpu
from pyscf.pbc.df import aft as aft_cpu
from pyscf.pbc.df.aft import _check_kpts, ft_ao
from pyscf.pbc.gto import pseudo
from pyscf.pbc.lib.kpts_helper import is_zero
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.pbc import tools
from gpu4pyscf.pbc.df import fft_jk

__all__ = [
    'get_nuc', 'get_pp', 'get_SI', 'FFTDF'
]

def get_nuc(mydf, kpts=None):
    from gpu4pyscf.pbc.dft import numint
    kpts, is_single_kpt = _check_kpts(mydf, kpts)
    cell = mydf.cell
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension > 1
    mesh = mydf.mesh
    charge = cp.asarray(-cell.atom_charges())
    Gv = cell.get_Gv(mesh)
    SI = get_SI(cell, mesh=mesh)
    rhoG = charge.dot(SI)

    coulG = tools.get_coulG(cell, mesh=mesh, Gv=Gv)
    vneG = rhoG * coulG
    vneR = tools.ifft(vneG, mesh).real

    nkpts = len(kpts)
    nao = cell.nao
    if is_zero(kpts):
        vne = cp.zeros((nkpts,nao,nao))
    else:
        vne = cp.zeros((nkpts,nao,nao), dtype=np.complex128)
    kpts = np.asarray(kpts)
    ao_ks = numint.eval_ao_kpts(cell, mydf.grids.coords, kpts)
    for k, ao in enumerate(ao_ks):
        vne[k] += (ao.conj().T*vneR).dot(ao)

    if is_single_kpt:
        vne = vne[0]
    return vne

def get_pp(mydf, kpts=None):
    '''Get the periodic pseudopotential nuc-el AO matrix, with G=0 removed.
    '''
    from gpu4pyscf.pbc.dft import numint
    kpts, is_single_kpt = _check_kpts(mydf, kpts)
    cell = mydf.cell
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension > 1
    mesh = mydf.mesh
    Gv = cell.get_Gv(mesh)
    SI = get_SI(cell, mesh=mesh)
    vpplocG = pseudo.get_vlocG(cell, Gv)
    vpplocG = -np.einsum('ij,ij->j', SI, vpplocG)
    vpplocG = cp.asarray(vpplocG)
    # vpploc evaluated in real-space
    vpplocR = tools.ifft(vpplocG, mesh).real

    ngrids = len(vpplocG)
    nkpts = len(kpts)
    nao = cell.nao
    if is_zero(kpts):
        vpp = cp.zeros((nkpts,nao,nao))
    else:
        vpp = cp.zeros((nkpts,nao,nao), dtype=np.complex128)
    kpts = np.asarray(kpts)
    ao_ks = numint.eval_ao_kpts(cell, mydf.grids.coords, kpts)
    for k, ao in enumerate(ao_ks):
        vpp[k] += (ao.conj().T*vpplocR).dot(ao)

    # vppnonloc evaluated in reciprocal space
    fakemol = gto.Mole()
    fakemol._atm = np.zeros((1,gto.ATM_SLOTS), dtype=np.int32)
    fakemol._bas = np.zeros((1,gto.BAS_SLOTS), dtype=np.int32)
    ptr = gto.PTR_ENV_START
    fakemol._env = np.zeros(ptr+10)
    fakemol._bas[0,gto.NPRIM_OF ] = 1
    fakemol._bas[0,gto.NCTR_OF  ] = 1
    fakemol._bas[0,gto.PTR_EXP  ] = ptr+3
    fakemol._bas[0,gto.PTR_COEFF] = ptr+4

    # buf for SPG_lmi upto l=0..3 and nl=3
    buf = np.empty((48,ngrids), dtype=np.complex128)
    def vppnl_by_k(kpt):
        Gk = Gv + kpt
        G_rad = lib.norm(Gk, axis=1)
        aokG = ft_ao.ft_ao(cell, Gv, kpt=kpt) * (1/cell.vol)**.5
        vppnl = 0
        for ia in range(cell.natm):
            symb = cell.atom_symbol(ia)
            if symb not in cell._pseudo:
                continue
            pp = cell._pseudo[symb]
            p1 = 0
            for l, proj in enumerate(pp[5:]):
                rl, nl, hl = proj
                if nl > 0:
                    fakemol._bas[0,gto.ANG_OF] = l
                    fakemol._env[ptr+3] = .5*rl**2
                    fakemol._env[ptr+4] = rl**(l+1.5)*np.pi**1.25
                    pYlm_part = fakemol.eval_gto('GTOval', Gk)

                    p0, p1 = p1, p1+nl*(l*2+1)
                    # pYlm is real, SI[ia] is complex
                    pYlm = np.ndarray((nl,l*2+1,ngrids), dtype=np.complex128, buffer=buf[p0:p1])
                    for k in range(nl):
                        qkl = pseudo.pp._qli(G_rad*rl, l, k)
                        pYlm[k] = pYlm_part.T * qkl
                    #:SPG_lmi = np.einsum('g,nmg->nmg', SI[ia].conj(), pYlm)
                    #:SPG_lm_aoG = np.einsum('nmg,gp->nmp', SPG_lmi, aokG)
                    #:tmp = np.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                    #:vppnl += np.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
            if p1 > 0:
                SPG_lmi = buf[:p1]
                SPG_lmi *= SI[ia].conj()
                SPG_lm_aoGs = lib.zdot(SPG_lmi, aokG)
                p1 = 0
                for l, proj in enumerate(pp[5:]):
                    rl, nl, hl = proj
                    if nl > 0:
                        p0, p1 = p1, p1+nl*(l*2+1)
                        hl = np.asarray(hl)
                        SPG_lm_aoG = SPG_lm_aoGs[p0:p1].reshape(nl,l*2+1,-1)
                        tmp = np.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                        vppnl += np.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
        return vppnl * (1./cell.vol)

    for k, kpt in enumerate(kpts):
        vppnl = vppnl_by_k(kpt)
        if is_zero(kpt):
            vpp[k] += cp.asarray(vppnl.real)
        else:
            vpp[k] += cp.asarray(vppnl)

    if is_single_kpt:
        vpp = vpp[0]
    return vpp

def get_SI(cell, Gv=None, mesh=None, atmlst=None):
    '''Calculate the structure factor (0D, 1D, 2D, 3D) for all atoms; see MH (3.34).

    Args:
        cell : instance of :class:`Cell`

        Gv : (N,3) array
            G vectors

        atmlst : list of ints, optional
            Indices of atoms for which the structure factors are computed.

    Returns:
        SI : (natm, ngrids) ndarray, dtype=np.complex128
            The structure factor for each atom at each G-vector.
    '''
    coords = cp.asarray(cell.atom_coords())
    if atmlst is not None:
        coords = coords[np.asarray(atmlst)]
    if Gv is None:
        if mesh is None:
            mesh = cell.mesh
        basex, basey, basez = cell.get_Gv_weights(mesh)[1]
        basex = cp.asarray(basex)
        basey = cp.asarray(basey)
        basez = cp.asarray(basez)
        b = cp.asarray(cell.reciprocal_vectors())
        rb = coords.dot(b.T)
        SIx = cp.exp(-1j*rb[:,0,None] * basex)
        SIy = cp.exp(-1j*rb[:,1,None] * basey)
        SIz = cp.exp(-1j*rb[:,2,None] * basez)
        SI = SIx[:,:,None,None] * SIy[:,None,:,None] * SIz[:,None,None,:]
        natm = coords.shape[0]
        SI = SI.reshape(natm, -1)
    else:
        SI = cp.exp(-1j*coords.dot(cp.asarray(Gv).T))
    return SI


class FFTDF(lib.StreamObject):
    '''Density expansion on plane waves (GPW method)
    '''

    blockdim = 240

    _keys = fft_cpu.FFTDF._keys

    def __init__(self, cell, kpts=np.zeros((1,3))):
        from gpu4pyscf.pbc.dft import gen_grid
        from gpu4pyscf.pbc.dft import numint
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory
        self.kpts = kpts
        self.grids = gen_grid.UniformGrids(cell)

        # The following attributes are not input options.
        # self.exxdiv has no effects. It was set in the get_k_kpts function to
        # mimic the KRHF/KUHF object in the call to tools.get_coulG.
        self.exxdiv = None
        self._numint = numint.KNumInt()
        self._rsh_df = {}  # Range separated Coulomb DF objects

    mesh = fft_cpu.FFTDF.mesh
    dump_flags = fft_cpu.FFTDF.dump_flags
    check_sanity = fft_cpu.FFTDF.check_sanity
    build = fft_cpu.FFTDF.build
    reset = fft_cpu.FFTDF.reset

    aoR_loop = NotImplemented

    get_pp = get_pp
    get_nuc = get_nuc

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        if omega is not None:  # J/K for RSH functionals
            with self.range_coulomb(omega) as rsh_df:
                return rsh_df.get_jk(dm, hermi, kpts, kpts_band, with_j, with_k,
                                     omega=None, exxdiv=exxdiv)

        kpts, is_single_kpt = _check_kpts(self, kpts)
        if is_single_kpt:
            vj, vk = fft_jk.get_jk(self, dm, hermi, kpts[0], kpts_band,
                                   with_j, with_k, exxdiv)
        else:
            vj = vk = None
            if with_k:
                vk = fft_jk.get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
            if with_j:
                vj = fft_jk.get_j_kpts(self, dm, hermi, kpts, kpts_band)
        return vj, vk

    get_eri = get_ao_eri = NotImplemented
    ao2mo = get_mo_eri = NotImplemented
    ao2mo_7d = NotImplemented
    get_ao_pairs_G = get_ao_pairs = NotImplemented
    get_mo_pairs_G = get_mo_pairs = NotImplemented

    range_coulomb = aft_cpu.AFTDF.range_coulomb

    to_gpu = utils.to_gpu
    device = utils.device

    def to_cpu(self):
        obj = utils.to_cpu(self)
        return obj.reset()
