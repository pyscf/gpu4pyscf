#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
Analytical nuclear gradients for RHF with kpoints sampling
'''

import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.gto.mole import PTR_ENV_START, ANG_OF
from pyscf.pbc.grad import krhf as krhf_cpu
from gpu4pyscf.lib import logger
from gpu4pyscf.grad import rhf as molgrad
from pyscf.pbc.gto.pseudo.pp import get_vlocG, get_alphas, _qli
from gpu4pyscf.pbc.dft import numint as pbc_numint
from gpu4pyscf.pbc.dft import UniformGrids
from gpu4pyscf.pbc.df.aft import _check_kpts
from gpu4pyscf.pbc.df import ft_ao
from gpu4pyscf.pbc import tools
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.lib.cupy_helper import contract, ensure_numpy

__all__ = ['Gradients']

def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None):
    '''
    Electronic part of KRHF/KRKS gradients
    Args:
        mf_grad : pbc.grad.krhf.Gradients or pbc.grad.krks.Gradients object
    '''
    mf = mf_grad.base
    cell = mf_grad.cell
    natm = cell.natm
    kpts = mf.kpts
    nkpts = len(kpts)
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff

    if getattr(mf, 'disp', None):
        raise NotImplementedError('dispersion correction')

    log = logger.new_logger(mf_grad)
    t0 = log.init_timer()
    log.debug('Computing Gradients of NR-HF Coulomb repulsion')
    hcore_deriv = mf_grad.hcore_generator(cell, kpts)
    s1 = mf_grad.get_ovlp(cell, kpts)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    dvhf = mf_grad.get_veff(dm0, kpts)
    t1 = log.timer('gradients of 2e part', *t0)

    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)
    aoslices = cell.aoslice_by_atom()
    extra_force = np.empty([natm, 3])
    dh1e = cp.empty([natm, 3])
    for ia in range(natm):
        h1ao = hcore_deriv(ia)
        dh1e[ia] = cp.einsum('kxij,kji->x', h1ao, dm0).real
        extra_force[ia] = ensure_numpy(mf_grad.extra_force(ia, locals()))
    log.timer('gradients of 1e part', *t1)

    # nabla is applied on bra in vhf. *2 for the contributions of nabla|ket>
    ds = contract('kxij,kji->xi', s1, dme0).real
    ds = (-2 * ds).get()
    ds = np.array([ds[:,p0:p1].sum(axis=1) for p0, p1 in aoslices[:,2:]])
    de = (2 * dvhf + dh1e.get() + ds) / nkpts + extra_force

    if log.verbose > logger.DEBUG:
        log.debug('gradients of electronic part')
        mf_grad._write(cell, de, range(natm))
    return de

def get_hcore(cell, kpts):
    '''Part of the nuclear gradients of core Hamiltonian'''
    h1 = int1e.int1e_ipkin(cell, kpts)
    dtype = h1.dtype
    if cell._pseudo:
        SI = cell.get_SI()
        Gv_cpu = cell.Gv
        Gv = cp.asarray(Gv_cpu)
        natom = cell.natm
        coords = cp.asarray(cell.get_uniform_grids())
        ngrids = len(coords)
        vlocG = get_vlocG(cell)
        vpplocG = -cp.einsum('ij,ij->j', SI, vlocG)
        vpplocG[0] = cp.sum(get_alphas(cell))
        vpplocR = tools.ifft(vpplocG, cell.mesh).real
        ni = pbc_numint.KNumInt()
        grids = UniformGrids(cell)
        # block_loop(sort_grids=True) would reorder the grids. Sorting vpplocR
        # accordingly
        vpplocR = vpplocR[grids.argsort()]
        deriv = 1
        grid0 = grid1 = 0
        for ao_ks, weight, coords in ni.block_loop(cell, grids, deriv, kpts,
                                                   sort_grids=True):
            ao_ks = ao_ks.transpose(0,1,3,2) # [nk,comp,nao,nGv]
            grid0, grid1 = grid1, grid1 + len(weight)
            aow = ao_ks[:,0] * vpplocR[grid0:grid1]
            #:h1 += cp.einsum('kxig,kjg->kxij', ao_ks[:,1:].conj(), aow)
            contract('kxig,kjg->kxij', ao_ks[:,1:].conj(), aow, beta=1, out=h1)

        fakemol = krhf_cpu._make_fakemol()
        ptr = PTR_ENV_START
        ft_weight = 1. / cell.vol
        for kn, kpt in enumerate(kpts):
            # Compute aokG analytically. An error ~cell.precision may be
            # encountered compared to the fftk results.
            #:expir = cp.exp(-1j*cp.dot(coords, cp.asarray(kpt)))
            #:aos = pbc_numint.eval_ao(cell, coords, kpt, deriv=1)
            #:aokG = tools.fftk(aos.transpose(0,2,1).reshape(-1,ngrids),
            #                   cell.mesh, expir).reshape(4,nao,ngrids)
            #:aokG, aokG_ip1 = aokG[0], aokG[1:4]
            aokG = ft_ao.ft_ao(cell, Gv, kpt=kpt).T
            aokG *= ft_weight
            Gk = Gv_cpu + kpt
            aokG_ip1 = aokG * (cp.asarray(Gk).T[:,None,:]*1j)
            G_rad = np.linalg.norm(Gk, axis=1)
            for ia in range(natom):
                symb = cell.atom_symbol(ia)
                if symb not in cell._pseudo:
                    continue
                pp = cell._pseudo[symb]
                for l, proj in enumerate(pp[5:]):
                    rl, nl, hl = proj
                    if nl >0:
                        hl = cp.asarray(hl)
                        fakemol._bas[0,ANG_OF] = l
                        fakemol._env[ptr+3] = .5*rl**2
                        fakemol._env[ptr+4] = rl**(l+1.5)*np.pi**1.25
                        pYlm_part = fakemol.eval_gto('GTOval', Gk)
                        pYlm = cp.empty((nl,l*2+1,ngrids))
                        for k in range(nl):
                            qkl = _qli(G_rad*rl, l, k)
                            pYlm[k].set(pYlm_part.T * qkl)
                        SPG_lmi = cp.einsum('g,nmg->nmg', SI[ia].conj(), pYlm)
                        SPG_lm_aoG = contract('nmg,pg->nmp', SPG_lmi, aokG)
                        SPG_lm_aoG_ip1 = contract('nmg,apg->anmp', SPG_lmi, aokG_ip1)
                        tmp = contract('ij,jmp->imp', hl, SPG_lm_aoG)
                        vnl = contract('aimp,imq->apq', SPG_lm_aoG_ip1.conj(), tmp)
                        if dtype == np.float64:
                            h1[kn,:] += vnl.real
                        else:
                            h1[kn,:] += vnl
    else:
        raise NotImplementedError
    return h1

def hcore_generator(mf_grad, cell=None, kpts=None):
    if cell is None: cell = mf_grad.cell
    if kpts is None: kpts = mf_grad.kpts
    h1 = mf_grad.get_hcore(cell, kpts)
    dtype = h1.dtype

    mf = mf_grad.base
    kpts, is_single_kpt = _check_kpts(mf, kpts)

    aoslices = cell.aoslice_by_atom()
    SI = cp.asarray(cell.get_SI())
    mesh = cell.mesh
    Gv_cpu = cell.Gv
    Gv = cp.asarray(Gv_cpu)
    ngrids = len(Gv)
    vlocG = cp.asarray(get_vlocG(cell))
    ni = pbc_numint.KNumInt()
    grids = UniformGrids(cell)
    ptr = PTR_ENV_START

    def hcore_deriv(atm_id):
        hcore = cp.zeros_like(h1)
        symb = cell.atom_symbol(atm_id)
        if symb not in cell._pseudo:
            return hcore
        vloc_g = cp.einsum('ga,g,g->ag', Gv, 1j * SI[atm_id], vlocG[atm_id])
        vloc_R = tools.ifft(vloc_g, mesh).real
        # block_loop(sort_grids=True) would reorder the grids.
        vloc_R = vloc_R[:,grids.argsort()]
        vloc_g = None
        deriv = 0
        grid0 = grid1 = 0
        for ao_ks, weight, coords in ni.block_loop(cell, grids, deriv, kpts,
                                                   sort_grids=True):
            ao_ks = ao_ks.transpose(0,2,1) # [nk,nao,nGv]
            grid0, grid1 = grid1, grid1 + len(weight)
            aow = ao_ks[:,None,:,:] * vloc_R[:,None,grid0:grid1]
            #:hcore += contract('kig,kxjg->kxij',ao_ks.conj(), aow)
            contract('kig,kxjg->kxij', ao_ks.conj(), aow, beta=1, out=hcore)

        fakemol = krhf_cpu._make_fakemol()
        shl0, shl1, p0, p1 = aoslices[atm_id]
        ft_weight = 1. / cell.vol
        for kn, kpt in enumerate(kpts):
            # Compute aokG analytically. An error ~cell.precision may be
            # encountered compared to the fftk results.
            #:aokG = 1./ngrids*fftk(ao.T, mesh, exp(-1j*coords.dot(kpt)))
            aokG = ft_ao.ft_ao(cell, Gv, kpt=kpt).T
            aokG *= ft_weight
            Gk = Gv_cpu + kpt
            G_rad = lib.norm(Gk, axis=1)
            pp = cell._pseudo[symb]
            for l, proj in enumerate(pp[5:]):
                rl, nl, hl = proj
                if nl > 0:
                    hl = cp.asarray(hl)
                    fakemol._bas[0,ANG_OF] = l
                    fakemol._env[ptr+3] = .5*rl**2
                    fakemol._env[ptr+4] = rl**(l+1.5)*np.pi**1.25
                    pYlm_part = fakemol.eval_gto('GTOval', Gk)
                    pYlm = cp.empty((nl,l*2+1,ngrids))
                    for k in range(nl):
                        qkl = _qli(G_rad*rl, l, k)
                        pYlm[k].set(pYlm_part.T * qkl)
                    SPG_lmi = cp.einsum('g,nmg->nmg', SI[atm_id].conj(), pYlm)
                    SPG_lm_aoG = contract('nmg,pg->nmp', SPG_lmi, aokG)
                    SPG_lmi_G = contract('nmg, ga->anmg', SPG_lmi, 1j*Gv)
                    SPG_lm_G_aoG = contract('anmg, pg->anmp', SPG_lmi_G, aokG)
                    tmp_1 = contract('ij,ajmp->aimp', hl, SPG_lm_G_aoG)
                    vppnl = contract('imp,aimq->apq', SPG_lm_aoG.conj(), tmp_1)
                    tmp = contract('ij,jmp->imp', hl, SPG_lm_aoG)
                    vppnl = contract('jmp,ajmq->apq', tmp.conj(), SPG_lm_G_aoG)
                    if dtype == np.float64:
                        hcore[kn] += vppnl.real
                        hcore[kn] += vppnl.real.transpose(0,2,1)
                    else:
                        hcore[kn] += vppnl
                        hcore[kn] += vppnl.conj().transpose(0,2,1)
        hcore[:,:,p0:p1] -= h1[:,:,p0:p1]
        hcore[:,:,:,p0:p1] -= h1[:,:,p0:p1].transpose(0,1,3,2).conj()
        return hcore
    return hcore_deriv

class GradientsBase(molgrad.GradientsBase):
    '''
    Basic nuclear gradient functions for non-relativistic methods
    '''
    def __init__(self, method):
        self.cell = method.cell
        molgrad.GradientsBase.__init__(self, method)

    @property
    def kpts(self):
        return self.base.kpts

    def reset(self, cell=None):
        if cell is not None:
            self.cell = cell
        self.base.reset(cell)
        return self

    def get_hcore(self, cell=None, kpts=None):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        return get_hcore(cell, kpts)

    hcore_generator = hcore_generator

    def get_ovlp(self, cell=None, kpts=None):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        return -int1e.int1e_ipovlp(cell, kpts)

    def get_jk(self, dm=None, kpts=None):
        if kpts is None: kpts = self.kpts
        if dm is None: dm = self.base.make_rdm1()
        exxdiv = self.base.exxdiv
        cpu0 = (logger.process_clock(), logger.perf_counter())
        ej, ek = self.base.with_df.get_jk_e1(dm, kpts, exxdiv=exxdiv)
        logger.timer(self, 'ejk', *cpu0)
        return ej, ek

    def get_j(self, dm=None, kpts=None):
        if kpts is None: kpts = self.kpts
        if dm is None: dm = self.base.make_rdm1()
        cpu0 = (logger.process_clock(), logger.perf_counter())
        ej = self.base.with_df.get_j_e1(dm, kpts)
        logger.timer(self, 'ej', *cpu0)
        return ej

    def get_k(self, dm=None, kpts=None, kpts_band=None):
        if kpts is None: kpts = self.kpts
        if dm is None: dm = self.base.make_rdm1()
        exxdiv = self.base.exxdiv
        cpu0 = (logger.process_clock(), logger.perf_counter())
        ek = self.base.with_df.get_k_e1(dm, kpts, kpts_band, exxdiv)
        logger.timer(self, 'ek', *cpu0)
        return ek

    def get_veff(self, dm=None, kpts=None):
        '''
        Computes the first-order derivatives of the energy contributions from
        Veff per atom.

        NOTE: This function is incompatible to the one implemented in PySCF CPU version.
        In the CPU version, get_veff returns the first order derivatives of Veff matrix.
        '''
        raise NotImplementedError

    def grad_nuc(self, cell=None, atmlst=None):
        if cell is None: cell = self.cell
        return krhf_cpu.grad_nuc(cell, atmlst)

class Gradients(GradientsBase):
    '''Non-relativistic restricted Hartree-Fock gradients'''

    def get_veff(self, dm=None, kpts=None):
        ej, ek = self.get_jk(dm, kpts)
        dvhf = ej - ek * .5
        return dvhf

    def make_rdm1e(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        '''Energy weighted density matrix'''
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        nkpts = len(mo_occ)
        nao = mo_coeff[0].shape[0]
        dtype = mo_coeff[-1].dtype
        dm1e = cp.empty((nkpts, nao, nao), dtype=dtype)
        for k, (e, c, occ) in enumerate(zip(mo_energy, mo_coeff, mo_occ)):
            mask = occ > 0
            c = c[:,mask]
            e_occ = e[mask] * occ[mask]
            dm1e[k] = (c*e_occ).dot(c.conj().T)
        return dm1e

    def extra_force(self, atom_id, envs):
        '''Hook for extra contributions in analytical gradients.

        Contributions like the response of auxiliary basis in density fitting
        method, the grid response in DFT numerical integration can be put in
        this function.
        '''
        #1 force from exxdiv corrections when madelung constant has non-zero derivative
        #2 DFT grid response
        return 0

    grad_elec = grad_elec
    as_scanner = molgrad.as_scanner
    _finalize = krhf_cpu.Gradients._finalize

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        if self.verbose >= logger.INFO:
            self.dump_flags()

        de = self.grad_elec(mo_energy, mo_coeff, mo_occ)
        self.de = de + self.grad_nuc()
        logger.timer(self, 'SCF gradients', *cput0)
        self._finalize()
        return self.de
