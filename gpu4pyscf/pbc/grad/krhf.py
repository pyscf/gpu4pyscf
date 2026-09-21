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
from gpu4pyscf.pbc.dft import multigrid, multigrid_v3
from gpu4pyscf.pbc.df import ft_ao, GDF
from gpu4pyscf.pbc.df.aft import get_SI, _get_ZSI
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
from gpu4pyscf.pbc import tools as pbctools
from gpu4pyscf.pbc.grad.pp import ppnl_derivatives
from gpu4pyscf.pbc.grad.rhf import contract_h1e_dm, _get_ejk_derivatives
from gpu4pyscf.pbc.grad import rhf as pbchf_grad
from gpu4pyscf.pbc.scf import hf as pbchf
from gpu4pyscf.pbc.df.grad.krhf import get_nuc

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

def get_nuc_fftdf(mf_grad, cell, dm0, kpts):
    nkpts = len(kpts)
    if nkpts == 1:
        if dm0.ndim == 2:
            dm0 = dm0[None, :, :]
    grad_sigma = np.zeros((cell.natm + 3, 3))
    hcore_deriv = hcore_generator(mf_grad, cell, kpts)
    dh1e = cp.empty([cell.natm, 3])
    for ia in range(cell.natm):
        h1ao = hcore_deriv(ia)
        dh1e[ia] = cp.einsum('kxij,kji->x', h1ao, dm0).real
    grad_sigma[:-3] += dh1e.get() / nkpts
    # hcore_generator includes kinetic gradients, but not kinetic strain.
    grad_sigma[-3:] += int1e.kin_derivatives(cell, dm0, kpts)[-3:]
    ni = multigrid_v3.MultiGridNumInt(cell)
    grad_sigma[-3:] += ni.energy_strain_gradient(
        'HF', dm0, kpts, spin=0, with_j=False, with_nuc=True)
    return grad_sigma

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
        elif isinstance(ni, multigrid.MultiGridNumIntBase):
            grad_sigma += get_nuc_fftdf(self, cell, dm0, kpts)
        elif np.prod(cell.mesh) < pbchf.ALLOWED_FFT_MESH_SIZE:
            grad_sigma += get_nuc_fftdf(self, cell, dm0, kpts)
        else:
            grad_sigma += get_nuc(cell, dm0, kpts)
            grad_sigma += int1e.kin_derivatives(cell, dm0, kpts)

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
