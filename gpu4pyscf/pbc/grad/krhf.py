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
from pyscf.pbc.gto.pseudo.pp import get_vlocG, get_alphas, _qli
from gpu4pyscf.lib import logger
from gpu4pyscf.grad import rhf as molgrad
from gpu4pyscf.pbc.dft import numint as pbc_numint
from gpu4pyscf.pbc.dft import UniformGrids
from gpu4pyscf.pbc.df import ft_ao
from gpu4pyscf.pbc.df.fft import get_SI
from gpu4pyscf.pbc import tools
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
from gpu4pyscf.pbc.tools.pbc import get_coulG
from gpu4pyscf.lib.cupy_helper import contract, ensure_numpy
from gpu4pyscf.pbc.grad.pp import vppnl_nuc_grad
from gpu4pyscf.pbc.dft import multigrid, multigrid_v2

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
    s1 = mf_grad.get_ovlp(cell, kpts)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    # derivatives of the Veff contribution
    dvhf = mf_grad.get_veff(dm0, kpts) * 2
    t1 = log.timer('gradients of 2e part', *t0)

    ni = getattr(mf, "_numint", None)
    if isinstance(ni, multigrid.MultiGridNumInt):
        raise NotImplementedError(
            "Gradient with kpts not implemented with multigrid.MultiGridNumInt. "
            "Please use the default KNumInt or multigrid_v2.MultiGridNumInt instead.")
    elif isinstance(ni, multigrid_v2.MultiGridNumInt):
        # Attention: The orbital derivative of vpploc term is in multigrid_v2.get_veff_ip1() function.
        rho_g = multigrid_v2.evaluate_density_on_g_mesh(ni, dm0, kpts)
        rho_g = rho_g[0,0]
        if cell._pseudo:
            dh1e = multigrid.eval_vpplocG_SI_gradient(cell, ni.mesh, rho_g) * nkpts
        else:
            dh1e = multigrid.eval_nucG_SI_gradient(cell, ni.mesh, rho_g) * nkpts

        dm_dmH = dm0 + dm0.transpose(0,2,1).conj()
        dh1e_kin = int1e.int1e_ipkin(cell, kpts)
        aoslices = cell.aoslice_by_atom()
        for ia in range(natm):
            p0, p1 = aoslices[ia, 2:]
            dh1e[ia] -= cp.einsum('kxij,kji->x', dh1e_kin[:,:,p0:p1,:], dm_dmH[:,:,p0:p1]).real
    else:
        hcore_deriv = mf_grad.hcore_generator(cell, kpts)
        dh1e = cp.empty([natm, 3])
        for ia in range(natm):
            h1ao = hcore_deriv(ia)
            dh1e[ia] = cp.einsum('kxij,kji->x', h1ao, dm0).real

    if cell._pseudo:
        dm0_cpu = dm0.get()
        dh1e_pp_nonlocal = vppnl_nuc_grad(cell, dm0_cpu, kpts = kpts)
        dh1e += cp.asarray(dh1e_pp_nonlocal)

    log.timer('gradients of 1e part', *t1)

    extra_force = np.empty([natm, 3])
    for ia in range(natm):
        extra_force[ia] = ensure_numpy(mf_grad.extra_force(ia, locals()))

    # nabla is applied on bra in vhf. *2 for the contributions of nabla|ket>
    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)
    aoslices = cell.aoslice_by_atom()
    ds = contract('kxij,kji->xi', s1, dme0).real
    ds = (-2 * ds).get()
    ds = np.array([ds[:,p0:p1].sum(axis=1) for p0, p1 in aoslices[:,2:]])
    de = (dh1e.get() + ds) / nkpts + dvhf + extra_force

    if log.verbose > logger.DEBUG:
        log.debug('gradients of electronic part')
        mf_grad._write(cell, de, range(natm))
    return de

def get_hcore(cell, kpts):
    '''
        Part of the nuclear gradients of core Hamiltonian
        If pseudo potential is turned on, the local term is included, but the nonlocal term is not included.
    '''
    h1 = int1e.int1e_ipkin(cell, kpts)
    if cell._pseudo:
        SI = cell.get_SI()
        Gv_cpu = cell.Gv
        Gv = cp.asarray(Gv_cpu)
        coords = cp.asarray(cell.get_uniform_grids())
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
    else:
        mesh = cell.mesh
        charge = cp.asarray(-cell.atom_charges(), dtype=np.float64)
        Gv = cell.get_Gv(mesh)
        SI = get_SI(cell, mesh=mesh)
        rhoG = charge.dot(SI)
        coulG = get_coulG(cell, mesh=mesh, Gv=Gv)
        vneG = rhoG * coulG
        vneR = tools.ifft(vneG, mesh).real
        ni = pbc_numint.KNumInt()
        grids = UniformGrids(cell)
        # block_loop(sort_grids=True) would reorder the grids. Sorting vneR
        # accordingly
        vneR = vneR[grids.argsort()]
        deriv = 1
        grid0 = grid1 = 0
        for ao_ks, weight, coords in ni.block_loop(cell, grids, deriv, kpts,
                                                   sort_grids=True):
            ao_ks = ao_ks.transpose(0,1,3,2) # [nk,comp,nao,nGv]
            grid0, grid1 = grid1, grid1 + len(weight)
            aow = ao_ks[:,0] * vneR[grid0:grid1]
            #:h1 += cp.einsum('kxig,kjg->kxij', ao_ks[:,1:].conj(), aow)
            contract('kxig,kjg->kxij', ao_ks[:,1:].conj(), aow, beta=1, out=h1)
    return h1

def hcore_generator(mf_grad, cell=None, kpts=None):
    '''
        If pseudo potential is turned on, the local term is included, but the nonlocal term is not included.
    '''
    if cell is None: cell = mf_grad.cell
    if kpts is None:
        kpts = mf_grad.kpts
    else:
        kpts = kpts.reshape(-1, 3)
    h1 = mf_grad.get_hcore(cell, kpts)

    aoslices = cell.aoslice_by_atom()
    SI = cp.asarray(cell.get_SI())
    mesh = cell.mesh
    Gv_cpu = cell.Gv
    Gv = cp.asarray(Gv_cpu)
    if cell._pseudo:
        vlocG = cp.asarray(get_vlocG(cell))
    else:
        Z = cell.atom_charges()
        coulG = get_coulG(cell, mesh=mesh, Gv=Gv)
    ni = pbc_numint.KNumInt()
    grids = UniformGrids(cell)

    def hcore_deriv(atm_id):
        hcore = cp.zeros_like(h1)
        if cell._pseudo:
            vloc_g = cp.einsum('ga,g,g->ag', Gv, 1j * SI[atm_id], vlocG[atm_id])
        else:
            vloc_g = cp.einsum('ga,g,g->ag', Gv, Z[atm_id]*1j * SI[atm_id], coulG)
        vloc_R = tools.ifft(vloc_g, mesh).real
        vloc_R = vloc_R[:,grids.argsort()]
        vloc_g = None
        deriv = 0
        grid0 = grid1 = 0
        # block_loop(sort_grids=True) would reorder the grids.
        for ao_ks, weight, coords in ni.block_loop(cell, grids, deriv, kpts,
                                                   sort_grids=True):
            ao_ks = ao_ks.transpose(0,2,1) # [nk,nao,nGv]
            grid0, grid1 = grid1, grid1 + len(weight)
            aow = ao_ks[:,None,:,:] * vloc_R[:,None,grid0:grid1]
            #:hcore += contract('kig,kxjg->kxij',ao_ks.conj(), aow)
            contract('kig,kxjg->kxij', ao_ks.conj(), aow, beta=1, out=hcore)

        shl0, shl1, p0, p1 = aoslices[atm_id]
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
        '''The derivatives of the Coulomb and exchange energy per cell'''
        if kpts is None: kpts = self.kpts
        if dm is None: dm = self.base.make_rdm1()
        if self.base.rsjk is not None:
            raise NotImplementedError
        exxdiv = self.base.exxdiv
        cpu0 = (logger.process_clock(), logger.perf_counter())
        ej, ek = self.base.with_df.get_jk_e1(dm, kpts, exxdiv=exxdiv)
        logger.timer(self, 'ejk', *cpu0)
        return ej, ek

    def get_j(self, dm=None, kpts=None):
        '''
        The derivatives of Coulomb energy per cell
        '''
        if kpts is None: kpts = self.kpts
        if dm is None: dm = self.base.make_rdm1()
        cpu0 = (logger.process_clock(), logger.perf_counter())
        with_rsjk = self.base.rsjk
        if with_rsjk is not None:
            assert isinstance(with_rsjk, PBCJKMatrixOpt)
            if with_rsjk.supmol is None:
                with_rsjk.build()
            ej = with_rsjk._get_ejk_sr_ip1(dm, kpts, k_factor=0)
            ej += with_rsjk._get_ejk_lr_ip1(dm, kpts, k_factor=0)
        else:
            ej = self.base.with_df.get_j_e1(dm, kpts)
        logger.timer(self, 'ej', *cpu0)
        return ej

    def get_k(self, dm=None, kpts=None, kpts_band=None):
        '''
        The derivatives of exchange energy per cell
        '''
        if kpts is None: kpts = self.kpts
        if dm is None: dm = self.base.make_rdm1()
        cpu0 = (logger.process_clock(), logger.perf_counter())
        with_rsjk = self.base.rsjk
        if with_rsjk is not None:
            assert isinstance(with_rsjk, PBCJKMatrixOpt)
            if with_rsjk.supmol is None:
                with_rsjk.build()
            exxdiv = self.base.exxdiv
            ek = with_rsjk._get_ejk_sr_ip1(dm, kpts, exxdiv=exxdiv, j_factor=0)
            ek += with_rsjk._get_ejk_lr_ip1(dm, kpts, exxdiv=exxdiv, j_factor=0)
            if dm.ndim == 3: # KRHF
                ek *= 2
            elif dm.ndim == 4: # KUHF
                pass
            else:
                raise RuntimeError('Illegal dm dimension')
        else:
            ek = self.base.with_df.get_k_e1(dm, kpts, kpts_band, exxdiv)
        logger.timer(self, 'ek', *cpu0)
        return ek

    def get_veff(self, dm=None, kpts=None):
        '''
        Computes the first-order derivatives of the energy contributions per
        cell from Veff per atom.

        NOTE: This function is incompatible to the one implemented in PySCF CPU version.
        In the CPU version, get_veff returns the first order derivatives of Veff matrix.
        '''
        raise NotImplementedError

    def grad_nuc(self, cell=None, atmlst=None):
        if cell is None: cell = self.cell
        return krhf_cpu.grad_nuc(cell, atmlst)

    def optimizer(self):
        '''Geometry (atom positions and lattice) optimization solver
        '''
        from gpu4pyscf.geomopt.ase_solver import GeometryOptimizer
        return GeometryOptimizer(self.base)

class Gradients(GradientsBase):
    '''Non-relativistic restricted Hartree-Fock gradients'''

    def get_veff(self, dm, kpts):
        '''
        The energy contribution from the effective potential

        einsum('kxij,kji->x', veff, dm) / nkpts
        '''
        if self.base.rsjk is not None:
            from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
            with_rsjk = self.base.rsjk
            assert isinstance(with_rsjk, PBCJKMatrixOpt)
            if with_rsjk.supmol is None:
                with_rsjk.build()
            ejk = with_rsjk._get_ejk_sr_ip1(dm, kpts, exxdiv=self.base.exxdiv)
            ejk += with_rsjk._get_ejk_lr_ip1(dm, kpts, exxdiv=self.base.exxdiv)
        else:
            ej, ek = self.get_jk(dm, kpts)
            ejk = ej - ek * .5
        return ejk

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

    def get_stress(self):
        from gpu4pyscf.pbc.grad import krhf_stress
        return krhf_stress.kernel(self)
