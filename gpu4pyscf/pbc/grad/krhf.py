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
from pyscf.pbc.grad import krhf as krhf_cpu
from pyscf.pbc.gto.pseudo.pp import get_vlocG, get_alphas
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, asarray, batched_vec_norm2
from gpu4pyscf.grad import rhf as molgrad
from gpu4pyscf.pbc.dft import numint as pbc_numint
from gpu4pyscf.pbc.dft.numint import eval_ao_kpts, _GTOvalOpt
from gpu4pyscf.pbc.dft import UniformGrids, BeckeGrids
from gpu4pyscf.pbc.dft import multigrid_v3
from gpu4pyscf.pbc.df import ft_ao, GDF
from gpu4pyscf.pbc.df.aft import get_SI, _get_ZSI
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
from gpu4pyscf.pbc import tools as pbctools
from gpu4pyscf.pbc.grad.pp import ppnl_derivatives
from gpu4pyscf.pbc.grad.rhf import contract_h1e_dm, _get_ejk_derivatives
from gpu4pyscf.pbc.grad import rhf as pbchf_grad

__all__ = ['Gradients']

def get_hcore(cell, kpts):
    '''
    Part of the nuclear gradients of core Hamiltonian
    If pseudo potential is turned on, the local term is included, but the nonlocal term is not included.
    '''
    h1 = int1e.int1e_ipkin(cell, kpts)
    if cell._pseudo:
        SI = get_SI(cell)
        Gv_cpu = cell.Gv
        Gv = cp.asarray(Gv_cpu)
        coords = cp.asarray(cell.get_uniform_grids())
        vlocG = get_vlocG(cell)
        vpplocG = -cp.einsum('ij,ij->j', SI, vlocG)
        vpplocG[0] = cp.sum(get_alphas(cell))
        vpplocR = pbctools.ifft(vpplocG, cell.mesh).real
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
        rhoG = _get_ZSI(cell, mesh)
        Gv = cell.get_Gv(mesh)
        coulG = pbctools.get_coulG(cell, mesh=mesh, Gv=Gv)
        vneG = rhoG * coulG
        vneR = pbctools.ifft(vneG, mesh).real
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

    if getattr(mf_grad.base, 'with_x2c', None):
        raise NotImplementedError('X2C gradients')

    h1 = get_hcore(cell, kpts)

    aoslices = cell.aoslice_by_atom()
    SI = get_SI(cell)
    mesh = cell.mesh
    Gv_cpu = cell.Gv
    Gv = cp.asarray(Gv_cpu)
    if cell._pseudo:
        vlocG = cp.asarray(get_vlocG(cell))
    else:
        Z = cell.atom_charges()
        coulG = pbctools.get_coulG(cell, mesh=mesh, Gv=Gv)
    ni = pbc_numint.KNumInt()
    grids = UniformGrids(cell)

    def hcore_deriv(atm_id):
        hcore = cp.zeros_like(h1)
        if cell._pseudo:
            vloc_g = cp.einsum('ga,g,g->ag', Gv, 1j * SI[atm_id], vlocG[atm_id])
        else:
            vloc_g = cp.einsum('ga,g,g->ag', Gv, Z[atm_id]*1j * SI[atm_id], coulG)
        vloc_R = pbctools.ifft(vloc_g, mesh).real
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

def get_nuc_strain_deriv(mf_grad, cell, dm, kpts):
    '''Strain derivatives for nuclear attraction or pp-local with k-points sampling

    This function is deprecated.
    '''
    from gpu4pyscf.lib.cupy_helper import sandwich_dot
    from gpu4pyscf.pbc.grad.krks_stress import (
        _eval_ao_strain_derivatives, _get_vpplocG_strain_derivatives,
        _get_coulG_strain_derivatives, ALIGNED)
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension != 1
    assert kpts.ndim == 2
    assert dm.ndim == 3
    if not cell.cart:
        c2s = asarray(cell.cart2sph_coeff())
        dm = sandwich_dot(dm, c2s.T)
        # Ensure all AOs are evaluated in the Cartesian GTOs as ao_ks strain
        # derivatives currently supports Cartesian format only
        cell = cell.copy()
        cell.cart = True
    nkpts, nao = dm.shape[:2]
    assert nkpts == len(kpts)

    grids = UniformGrids(cell)
    grids_idx = grids.argsort(tile=8)
    grids_coords = grids.coords[grids_idx]
    ngrids = len(grids_coords)
    mesh = grids.mesh

    def partial_dot(bra, ket):
        '''conj(ig),ig->g'''
        rho = cp.einsum('ig,ig->g', bra.real, ket.real)
        rho += cp.einsum('ig,ig->g', bra.imag, ket.imag)
        return rho

    eval_gto_opt = _GTOvalOpt(cell, kpts, deriv=1)
    max_memory = 4e9
    blksize = int((max_memory/16/(nkpts*10*nao))/ ALIGNED) * ALIGNED

    rho0 = cp.zeros(ngrids)
    rho1 = cp.zeros((3,3, ngrids))

    for p0, p1 in lib.prange(0, ngrids, blksize):
        coords = cp.asarray(grids_coords[p0:p1].T, order='C').T
        ao_ks = eval_ao_kpts(cell, coords, kpts, deriv=1, opt=eval_gto_opt)
        ao_ks_strain = _eval_ao_strain_derivatives(
            cell, coords, kpts, deriv=0, opt=eval_gto_opt)
        coordsT = coords.T
        for k, dm_k in enumerate(dm):
            ao = ao_ks[k].transpose(0,2,1)
            ao_strain = ao_ks_strain[k]
            ao1 = ao_strain[:,:,0]
            # Adding the response of the grids
            ao1 += contract('xig,yg->xyig', ao[1:4], coordsT)
            c0 = dm_k.T.dot(ao[0])
            rho0[p0:p1] += partial_dot(ao[0], c0).real
            rho1[:,:,p0:p1] += contract('xyig,ig->xyg', ao1, c0.conj()).real

    rho0 *= 1./nkpts
    # *2 for rho1 because the derivatives were applied to the bra only
    rho1 *= 2./nkpts

    rho0_fft_order = cp.empty_like(rho0)
    rho1_fft_order = cp.empty_like(rho1)
    rho0_fft_order[grids_idx] = rho0
    rho1_fft_order[:,:,grids_idx] = rho1
    rho0, rho1 = rho0_fft_order, rho1_fft_order
    rhoG = pbctools.fft(rho0, mesh)

    if cell._pseudo:
        vpplocG_0, vpplocG_1 = _get_vpplocG_strain_derivatives(cell, mesh)
        vpplocR = pbctools.ifft(vpplocG_0, mesh).real
        Ene = contract('xyg,g->xy', rho1, vpplocR).real.get()
        Ene += contract('g,xyg->xy', rhoG.conj(), vpplocG_1).real.get() * (1./ngrids)
    else:
        Gv = cell.get_Gv(mesh)
        coulG_0, coulG_1 = _get_coulG_strain_derivatives(cell, Gv)
        # SI corresponds to Fourier components of the fractional atomic
        # positions within the cell. It does not respond to the strain
        # transformation
        ZG = _get_ZSI(cell, mesh)
        vR = pbctools.ifft(ZG * coulG_0, mesh).real
        Ene = contract('xyg,g->xy', rho1, vR).real.get()
        Ene += contract('xyg,g->xy', coulG_1, rhoG.conj()*ZG).real.get() * (1./ngrids)
    return Ene

class GradientsBase(pbchf_grad.GradientsBase):
    '''
    Basic nuclear gradient functions for non-relativistic methods
    '''

    @property
    def kpts(self):
        return self.base.kpts

    def get_veff(self, dm=None, kpts=None):
        '''
        Computes the first-order derivatives of the per-cell energy contribution
        from Veff per atom. This is equivalent to one half of the two-electron
        energy contribution: self.energy_ee()/2.

        NOTE: This function is provided for backward compatibility only. It is
        not consistent to the one implemented in PySCF CPU version. In the CPU
        version, get_veff returns the first order derivatives of Veff matrix
        rather than the energy contribution.
        '''
        return self.energy_ee(dm, kpts) * .5

    def energy_ee(self, dm, kpts):
        '''
        The contribution of electron-electron interactions per cell to the
        nuclear gradients.
        '''
        raise NotImplementedError

class Gradients(GradientsBase):
    '''Non-relativistic restricted Hartree-Fock gradients'''
    grids = None
    grid_response = False

    _keys = {'grid_response', 'grids'}

    hcore_generator = hcore_generator

    def energy_ee(self, dm, kpts):
        '''
        The contribution of electron-electron interactions per cell to the
        nuclear gradients.
        '''
        mf = self.base
        with_df = mf.with_df
        # When J is evaluated using mf.j_engine or mf.rsjk, it is identical to
        # the J from MultiGridNumInt. The contribution from J matrix can be
        # efficiently evaluated using the MultiGridNumInt integrator.
        j_in_xc = not isinstance(with_df, GDF)
        omega = 0
        j_factor = k_sr = k_lr = 1

        # TODO: handle all-electron+GGA and pseudo+GGA differently
        # pseudo+GGA does not need to evaluate the gradients with PBCJKMatrixOpt
        de = 0
        ni = mf._numint
        spin = 0 if dm.ndim == 3 else 1
        if isinstance(ni, multigrid_v3.MultiGridNumInt):
            de = ni.energy_derivatives(
                'HF', dm, kpts=kpts, spin=spin, with_j=j_in_xc, with_nuc=True)
            if j_in_xc:
                j_factor = 0

        de += _get_ejk_derivatives(mf, dm, kpts, j_factor, omega, k_lr, k_sr)
        return de

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

    def grad_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        '''
        Electronic part of KRHF/KRKS gradients
        '''
        mf = self.base
        cell = self.cell
        if mo_energy is None: mo_energy = mf.mo_energy
        if mo_occ is None:    mo_occ = mf.mo_occ
        if mo_coeff is None:  mo_coeff = mf.mo_coeff

        if mf.istype('KSCF'):
            is_uhf = mf.istype('KUHF')
            kpts = mf.kpts
        else:
            is_uhf = mf.istype('UHF')
            kpts = mf.kpt
        nkpts = len(kpts)

        if getattr(mf, 'disp', None):
            raise NotImplementedError('dispersion correction')

        if getattr(mf, 'with_x2c', None):
            raise NotImplementedError('X2C gradients')

        log = logger.new_logger(self)
        t0 = log.init_timer()
        log.debug('Computing Gradients of NR-HF Coulomb repulsion')
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        # derivatives of the two-electron contribution
        grad_sigma = self.energy_ee(dm0, kpts)
        t1 = log.timer_debug1('gradients of 2e part', *t0)

        if is_uhf:
            dm0 = dm0[0] + dm0[1]

        ni = mf._numint
        if isinstance(ni, multigrid_v3.MultiGridNumInt):
            # Vne or pploc contribution is evaluated in energy_ee
            grad_sigma += int1e.kin_derivatives(cell, dm0, kpts)
        else:
            hcore_deriv = self.hcore_generator(cell, kpts)
            dh1e = cp.empty([cell.natm, 3])
            for ia in range(cell.natm):
                h1ao = hcore_deriv(ia)
                dh1e[ia] = cp.einsum('kxij,kji->x', h1ao, dm0).real
            grad_sigma[:-3] += dh1e.get() / nkpts
            if isinstance(self.grids or getattr(mf, 'grids', None), BeckeGrids):
                grad_sigma[-3:] = np.nan
            else:
                # hcore_generator includes kinetic gradients, but not kinetic strain.
                grad_sigma[-3:] += int1e.kin_derivatives(cell, dm0, kpts)[-3:]
                ni = multigrid_v3.MultiGridNumInt(cell)
                grad_sigma[-3:] += ni.energy_strain_gradient(
                    'HF', dm0, kpts, spin=0, with_j=False, with_nuc=True)

        if cell._pseudo:
            grad_sigma += ppnl_derivatives(cell, dm0, kpts)

        log.timer_debug1('gradients of 1e part', *t1)

        dme0 = self.make_rdm1e(mo_energy, mo_coeff, mo_occ)
        grad_sigma -= int1e.ovlp_derivatives(cell, dme0, kpts)

        if log.verbose > logger.DEBUG:
            log.debug('gradients of electronic part')
            self._write(cell, grad_sigma[:-3], range(cell.natm))
            log.debug('Asymmetric strain tensor of electronic part')
            log.debug('%s', grad_sigma[-3:]/cell.vol)
        return grad_sigma

    as_scanner = molgrad.as_scanner
    _finalize = krhf_cpu.Gradients._finalize
