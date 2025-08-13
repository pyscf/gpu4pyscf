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

import numpy as np
import cupy as cp
import pyscf
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.tools import pbc as pbctools
from gpu4pyscf.pbc.dft.gen_grid import UniformGrids
from gpu4pyscf.pbc.df import FFTDF
from gpu4pyscf.pbc.dft.numint import NumInt, eval_ao_kpts, _GTOvalOpt
from gpu4pyscf.pbc.grad import uks as uks_grad
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.lib.cupy_helper import contract, asarray, sandwich_dot
from gpu4pyscf.pbc.grad.rks_stress import (
    strain_tensor_dispalcement,
    _finite_diff_cells,
    _get_weight_strain_derivatives,
    _get_coulG_strain_derivatives,
    _eval_ao_strain_derivatives,
    _get_vpplocG_strain_derivatives,
    _get_pp_nonloc_strain_derivatives,
    ewald)

ALIGNED = 256

def get_vxc(ks_grad, cell, dm, with_j=False, with_nuc=False):
    '''Strain derivatives for Coulomb and XC at gamma point

    Kwargs:
        with_j : Whether to include the electron-electron Coulomb interactions
        with_nuc : Whether to include the electron-nuclear Coulomb interactions
    '''
    mf = ks_grad.base
    if dm is None: dm = mf.make_rdm1()
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension != 1

    ni = mf._numint
    assert isinstance(ni, NumInt)
    if ks_grad.grids is not None:
        grids = ks_grad.grids
    else:
        grids = mf.grids
    assert isinstance(grids, UniformGrids)

    xc_code = mf.xc
    xctype = ni._xc_type(xc_code)
    if xctype == 'LDA':
        deriv = 0
        nvar = 1
    elif xctype == 'GGA':
        deriv = 1
        nvar = 4
    elif xctype == 'MGGA':
        deriv = 1
        nvar = 5
    else:
        raise NotImplementedError

    assert dm.ndim == 3
    if not cell.cart:
        c2s = asarray(cell.cart2sph_coeff())
        dm = sandwich_dot(dm, c2s.T)
        cell = cell.copy()
        cell.cart = True
    nao = dm.shape[1]

    grids_idx = grids.argsort(tile=8)
    grids_coords = grids.coords[grids_idx]
    ngrids = len(grids_coords)
    mesh = grids.mesh
    weight_0, weight_1 = _get_weight_strain_derivatives(cell, grids)

    def partial_dot(bra, ket):
        '''conj(ig),ig->g'''
        rho = cp.einsum('ig,ig->g', bra.real, ket.real)
        rho += cp.einsum('ig,ig->g', bra.imag, ket.imag)
        return rho

    eval_gto_opt = _GTOvalOpt(cell, deriv=deriv)
    max_memory = 4e9
    blksize = int((max_memory/16/(nvar*10*nao))/ ALIGNED) * ALIGNED
    XY, YY, ZY, XZ, YZ, ZZ = 5, 7, 8, 6, 8, 9

    out = np.zeros((3,3))
    rho0 = cp.zeros((2, nvar, ngrids))
    rho1 = cp.zeros((3,3, 2, nvar, ngrids))

    for p0, p1 in lib.prange(0, ngrids, blksize):
        coords = cp.asarray(grids_coords[p0:p1].T, order='C').T
        ao = eval_ao_kpts(cell, coords, deriv=deriv+1, opt=eval_gto_opt)[0]
        ao_strain = _eval_ao_strain_derivatives(
            cell, coords, deriv=deriv, opt=eval_gto_opt)[0]
        coordsT = coords.T
        ao = ao.transpose(0,2,1)
        if xctype == 'LDA':
            ao1 = ao_strain[:,:,0]
            # Adding the response of the grids
            ao1 += contract('xig,yg->xyig', ao[1:4], coordsT)
            for s in range(2):
                c0 = dm[s].T.dot(ao[0])
                rho0[s,0,p0:p1] += partial_dot(ao[0], c0).real
                rho1[:,:,s,0,p0:p1] += contract('xyig,ig->xyg', ao1, c0.conj()).real
        elif xctype == 'GGA':
            ao_strain[:,:,0] += contract('xig,yg->xyig', ao[1:4], coordsT)
            ao_strain[:,:,1] += contract('xig,yg->xyig', ao[4:7], coordsT)
            ao_strain[0,:,2] += contract('ig,yg->yig', ao[XY], coordsT)
            ao_strain[1,:,2] += contract('ig,yg->yig', ao[YY], coordsT)
            ao_strain[2,:,2] += contract('ig,yg->yig', ao[ZY], coordsT)
            ao_strain[0,:,3] += contract('ig,yg->yig', ao[XZ], coordsT)
            ao_strain[1,:,3] += contract('ig,yg->yig', ao[YZ], coordsT)
            ao_strain[2,:,3] += contract('ig,yg->yig', ao[ZZ], coordsT)
            c0 = contract('xig,sij->sxjg', ao[:4], dm)
            for s in range(2):
                for i in range(4):
                    rho0[s,i,p0:p1] += partial_dot(ao[0], c0[s,i]).real
                # TODO: computing density derivatives using FFT
                rho1[:,:,s, : ,p0:p1] += contract('xynig,ig->xyng', ao_strain, c0[s,0].conj()).real
                rho1[:,:,s,1:4,p0:p1] += contract('xyig,nig->xyng', ao_strain[:,:,0], c0[s,1:4].conj()).real
        else: # MGGA
            ao_strain[:,:,0] += contract('xig,yg->xyig', ao[1:4], coordsT)
            ao_strain[:,:,1] += contract('xig,yg->xyig', ao[4:7], coordsT)
            ao_strain[0,:,2] += contract('ig,yg->yig', ao[XY], coordsT)
            ao_strain[1,:,2] += contract('ig,yg->yig', ao[YY], coordsT)
            ao_strain[2,:,2] += contract('ig,yg->yig', ao[ZY], coordsT)
            ao_strain[0,:,3] += contract('ig,yg->yig', ao[XZ], coordsT)
            ao_strain[1,:,3] += contract('ig,yg->yig', ao[YZ], coordsT)
            ao_strain[2,:,3] += contract('ig,yg->yig', ao[ZZ], coordsT)
            c0 = contract('xig,sij->sxjg', ao[:4], dm)
            for s in range(2):
                for i in range(4):
                    rho0[s,i,p0:p1] += partial_dot(ao[0], c0[s,i]).real
                rho0[s,4,p0:p1] += partial_dot(ao[1], c0[s,1]).real
                rho0[s,4,p0:p1] += partial_dot(ao[2], c0[s,2]).real
                rho0[s,4,p0:p1] += partial_dot(ao[3], c0[s,3]).real
                rho1[:,:,s, :4,p0:p1] += contract('xynig,ig->xyng', ao_strain, c0[s,0].conj()).real
                rho1[:,:,s,1:4,p0:p1] += contract('xyig,nig->xyng', ao_strain[:,:,0], c0[s,1:4].conj()).real
                rho1[:,:,s,4,p0:p1] += contract('xynig,nig->xyg', ao_strain[:,:,1:4], c0[s,1:4].conj()).real

    if xctype == 'LDA':
        pass
    elif xctype == 'GGA':
        rho0[:,1:4] *= 2 # dm should be hermitian
    else: # MGGA
        rho0[:,1:4] *= 2 # dm should be hermitian
        rho0[:,4] *= .5 # factor 1/2 for tau
        rho1[:,:,:,4] *= .5

    # *2 for rho1 because the derivatives were applied to the bra only
    rho1 *= 2.

    rho0_fft_order = cp.empty_like(rho0)
    rho1_fft_order = cp.empty_like(rho1)
    rho0_fft_order[:,:,grids_idx] = rho0
    rho1_fft_order[:,:,:,:,grids_idx] = rho1
    rho0, rho1 = rho0_fft_order, rho1_fft_order

    exc, vxc = ni.eval_xc_eff(xc_code, rho0, 1, xctype=xctype, spin=1)[:2]
    out += contract('xysng,sng->xy', rho1, vxc).real.get() * weight_0
    rho0 = rho0[:,0].sum(axis=0)
    rho1 = rho1[:,:,:,0].sum(axis=2)
    out += contract('g,g->', rho0, exc.ravel()).real.get() * weight_1

    Gv = cell.get_Gv(mesh)
    coulG_0, coulG_1 = _get_coulG_strain_derivatives(cell, Gv)
    rhoG = pbctools.fft(rho0, mesh)
    if with_j:
        vR = pbctools.ifft(rhoG * coulG_0, mesh)
        EJ = contract('xyg,g->xy', rho1, vR).real.get() * weight_0 * 2
        EJ += contract('g,g->', rho0, vR).real.get() * weight_1
        EJ += contract('xyg,g->xy', coulG_1, rhoG.conj()*rhoG).real.get() * (weight_0/ngrids)
        out += .5 * EJ

    if with_nuc:
        if cell._pseudo:
            vpplocG_0, vpplocG_1 = _get_vpplocG_strain_derivatives(cell, mesh)
            vpplocR = pbctools.ifft(vpplocG_0, mesh).real
            Ene = contract('xyg,g->xy', rho1, vpplocR).real.get()
            Ene += contract('g,xyg->xy', rhoG.conj(), vpplocG_1).real.get() * (1./ngrids)
            Ene += _get_pp_nonloc_strain_derivatives(cell, mesh, dm.sum(axis=0))
        else:
            charge = -cell.atom_charges()
            # SI corresponds to Fourier components of the fractional atomic
            # positions within the cell. It does not respond to the strain
            # transformation
            SI = cell.get_SI(mesh=mesh)
            ZG = asarray(np.dot(charge, SI))
            vR = pbctools.ifft(ZG * coulG_0, mesh).real
            Ene = contract('xyg,g->xy', rho1, vR).real.get()
            Ene += contract('xyg,g->xy', coulG_1, rhoG.conj()*ZG).real.get() * (1./ngrids)
        out += Ene
    return out

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
    assert isinstance(mf_grad, uks_grad.Gradients)
    mf = mf_grad.base
    with_df = mf.with_df
    assert isinstance(with_df, FFTDF)
    ni = mf._numint
    if ni.libxc.is_hybrid_xc(mf.xc):
        raise NotImplementedError('Stress tensor for hybrid DFT')
    if hasattr(mf, 'U_idx'):
        raise NotImplementedError('Stress tensor for DFT+U')

    log = logger.new_logger(mf_grad)
    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Computing stress tensor')

    cell = mf.cell
    dm0 = mf.make_rdm1().sum(axis=0)
    dme0 = mf_grad.make_rdm1e().sum(axis=0)
    sigma = ewald(cell)

    disp = 1e-5
    for x in range(3):
        for y in range(3):
            cell1, cell2 = _finite_diff_cells(cell, x, y, disp)
            t1 = int1e.int1e_kin(cell1)[0]
            t2 = int1e.int1e_kin(cell2)[0]
            t1 = cp.einsum('ij,ji->', t1, dm0)
            t2 = cp.einsum('ij,ji->', t2, dm0)
            sigma[x,y] += (t1 - t2) / (2*disp)
            s1 = int1e.int1e_ovlp(cell1)[0]
            s2 = int1e.int1e_ovlp(cell2)[0]
            s1 = cp.einsum('ij,ji->', s1, dme0)
            s2 = cp.einsum('ij,ji->', s2, dme0)
            sigma[x,y] -= (s1 - s2) / (2*disp)
    t0 = log.timer_debug1('hcore derivatives', *t0)

    dm0 = mf.make_rdm1()
    sigma += get_vxc(mf_grad, cell, dm0, with_j=True, with_nuc=True)
    t0 = log.timer_debug1('Vxc and Coulomb derivatives', *t0)

    sigma /= cell.vol
    if log.verbose >= logger.DEBUG:
        log.debug('Asymmetric strain tensor')
        log.debug('%s', sigma)
    return sigma
