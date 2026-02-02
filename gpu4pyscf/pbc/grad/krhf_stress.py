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
Stress tensor
'''

import numpy as np
import cupy as cp
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.tools import pbc as pbctools
from gpu4pyscf.pbc.dft.gen_grid import UniformGrids
from gpu4pyscf.pbc.dft.numint import eval_ao_kpts, _GTOvalOpt
from gpu4pyscf.pbc.grad import krhf as krhf_grad
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
from gpu4pyscf.pbc.df import aft, aft_jk
from gpu4pyscf.lib.cupy_helper import contract, asarray, sandwich_dot
from gpu4pyscf.pbc.grad.rks_stress import (
    _finite_diff_cells,
    _get_coulG_strain_derivatives,
    _eval_ao_strain_derivatives,
    _get_vpplocG_strain_derivatives,
    _get_pp_nonloc_strain_derivatives,
    ewald)

ALIGNED = 256

def kernel(mf_grad):
    '''Compute the energy derivatives for strain tensor (e_ij)

                1  d E
    sigma_ij = --- ------
                V  d e_ij

    sigma is a asymmetric 3x3 matrix. The symmetric stress tensor in the 6 Voigt
    notation can be transformed from the asymmetric stress tensor

    sigma1 = sigma_11
    sigma2 = sigma_22
    sigma3 = sigma_33
    sigma6 = (sigma_12 + sigma_21)/2
    sigma5 = (sigma_13 + sigma_31)/2
    sigma4 = (sigma_23 + sigma_32)/2

    See K. Doll, Mol Phys (2010), 108, 223
    '''
    assert isinstance(mf_grad, krhf_grad.Gradients)
    mf = mf_grad.base

    log = logger.new_logger(mf_grad)
    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Computing stress tensor')

    cell = mf.cell
    dm0 = mf.make_rdm1()
    dme0 = mf_grad.make_rdm1e()
    sigma = ewald(cell)

    kpts = mf.kpts
    int1e_opt_v2 = int1e._Int1eOptV2(cell)
    sigma -= int1e_opt_v2.get_ovlp_strain_deriv(dme0, kpts)

    scaled_kpts = kpts.dot(cell.lattice_vectors().T)
    nkpts = len(kpts)
    disp = 1e-5
    for x in range(3):
        for y in range(3):
            cell1, cell2 = _finite_diff_cells(cell, x, y, disp)
            kpts1 = scaled_kpts.dot(cell1.reciprocal_vectors(norm_to=1))
            kpts2 = scaled_kpts.dot(cell2.reciprocal_vectors(norm_to=1))
            t1 = int1e.int1e_kin(cell1, kpts1)
            t2 = int1e.int1e_kin(cell2, kpts2)
            t1 = cp.einsum('kij,kji->', t1, dm0).real
            t2 = cp.einsum('kij,kji->', t2, dm0).real
            sigma[x,y] += (t1 - t2).get() / (2*disp) / nkpts

    sigma += get_nuc(mf_grad, cell, dm0, kpts)
    t0 = log.timer_debug1('hcore derivatives', *t0)

    sigma += get_veff(mf_grad, cell, dm0, kpts)
    t0 = log.timer_debug1('vhf derivatives', *t0)

    sigma /= cell.vol
    if log.verbose >= logger.DEBUG:
        log.debug('Asymmetric strain tensor')
        log.debug('%s', sigma)
    return sigma

def get_veff(mf_grad, cell, dm, kpts):
    '''Strain derivatives for Coulomb and exchange energy with k-point samples
    '''
    mf = mf_grad.base
    with_rsjk = mf.rsjk
    if with_rsjk is not None:
        assert isinstance(with_rsjk, PBCJKMatrixOpt)
        if with_rsjk.supmol is None:
            with_rsjk.build()
        sigma = with_rsjk._get_ejk_sr_strain_deriv(dm, kpts, exxdiv=mf.exxdiv)
        sigma+= with_rsjk._get_ejk_lr_strain_deriv(dm, kpts, exxdiv=mf.exxdiv)
    elif isinstance(mf.with_df, aft.AFTDF):
        sigma = aft_jk.get_ej_strain_deriv(mf.with_df, dm, kpts)
        sigma -= aft_jk.get_ek_strain_deriv(mf.with_df, dm, kpts, exxdiv=mf.exxdiv) * .5
    else:
        raise NotImplementedError(f'Stress tensor for KHF for {mf.with_df}')
    return sigma

def get_nuc(mf_grad, cell, dm, kpts):
    '''Strain derivatives for Coulomb and Exc with k-point samples
    '''
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
        for k, dm in enumerate(dm):
            ao = ao_ks[k].transpose(0,2,1)
            ao_strain = ao_ks_strain[k]
            ao1 = ao_strain[:,:,0]
            # Adding the response of the grids
            ao1 += contract('xig,yg->xyig', ao[1:4], coordsT)
            c0 = dm.T.dot(ao[0])
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
        Ene += _get_pp_nonloc_strain_derivatives(cell, mesh, dm, kpts)
    else:
        Gv = cell.get_Gv(mesh)
        coulG_0, coulG_1 = _get_coulG_strain_derivatives(cell, Gv)
        charge = -cell.atom_charges()
        # SI corresponds to Fourier components of the fractional atomic
        # positions within the cell. It does not respond to the strain
        # transformation
        SI = cell.get_SI(mesh=mesh)
        ZG = asarray(np.dot(charge, SI))
        vR = pbctools.ifft(ZG * coulG_0, mesh).real
        Ene = contract('xyg,g->xy', rho1, vR).real.get()
        Ene += contract('xyg,g->xy', coulG_1, rhoG.conj()*ZG).real.get() * (1./ngrids)
    return Ene
