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

'''GPW method'''

__all__ = [
    'get_nuc', 'get_pp', 'get_SI', 'FFTDF'
]

import numpy as np
import cupy as cp
from pyscf import gto
from pyscf import lib
from pyscf.pbc.df import fft as fft_cpu
from pyscf.pbc.df import aft as aft_cpu
from pyscf.pbc.gto import pseudo
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.lib.kpts import KPoints
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.pbc import tools
from gpu4pyscf.pbc.df import fft_jk
from gpu4pyscf.pbc.df.aft import _check_kpts
from gpu4pyscf.pbc.df.ft_ao import ft_ao
from gpu4pyscf.pbc.lib.kpts_helper import reset_kpts

def get_nuc(mydf, kpts=None):
    from gpu4pyscf.pbc.dft import numint
    kpts, is_single_kpt = _check_kpts(mydf, kpts)
    cell = mydf.cell
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension > 1
    mesh = mydf.mesh
    charge = cp.asarray(-cell.atom_charges(), dtype=np.float64)
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
    vpplocG = -cp.einsum('ij,ij->j', SI, cp.asarray(vpplocG))
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
        aokG = ft_ao(cell, Gv, kpt=kpt) * (1/cell.vol)**.5
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
                SPG_lmi = cp.asarray(buf[:p1])
                SPG_lmi *= SI[ia].conj()
                SPG_lm_aoGs = SPG_lmi.dot(aokG)
                p1 = 0
                for l, proj in enumerate(pp[5:]):
                    rl, nl, hl = proj
                    if nl > 0:
                        p0, p1 = p1, p1+nl*(l*2+1)
                        hl = cp.asarray(hl)
                        SPG_lm_aoG = SPG_lm_aoGs[p0:p1].reshape(nl,l*2+1,-1)
                        tmp = contract('ij,jmp->imp', hl, SPG_lm_aoG)
                        vppnl += contract('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
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

    def __init__(self, cell, kpts=None):
        from gpu4pyscf.pbc.dft import numint
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory
        self.mesh = cell.mesh
        self.kpts = kpts

        # The following attributes are not input options.
        # self.exxdiv has no effects. It was set in the get_k_kpts function to
        # mimic the KRHF/KUHF object in the call to tools.get_coulG.
        self.exxdiv = None
        self._numint = numint.KNumInt()
        self._rsh_df = {}  # Range separated Coulomb DF objects

    @property
    def grids(self):
        from gpu4pyscf.pbc.dft.gen_grid import UniformGrids
        grids = UniformGrids(self.cell)
        grids.mesh = self.mesh
        return grids
    @grids.setter
    def grids(self, val):
        self.mesh = val.mesh

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

    dump_flags = fft_cpu.FFTDF.dump_flags
    check_sanity = fft_cpu.FFTDF.check_sanity
    build = fft_cpu.FFTDF.build

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

    get_j_e1 = fft_jk.get_j_e1_kpts

    get_eri = get_ao_eri = NotImplemented
    ao2mo = get_mo_eri = NotImplemented
    ao2mo_7d = NotImplemented
    get_ao_pairs_G = get_ao_pairs = NotImplemented
    get_mo_pairs_G = get_mo_pairs = NotImplemented

    range_coulomb = aft_cpu.AFTDF.range_coulomb

    to_gpu = utils.to_gpu
    device = utils.device

    # customize to_cpu because attributes grids and kpts are not compatible with pyscf-2.10
    def to_cpu(self):
        from pyscf.pbc.df.fft import FFTDF
        out = FFTDF(self.cell, kpts=self.kpts)
        out.mesh = self.mesh
        return out
