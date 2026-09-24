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

r'''
The energy derivatives for the strain tensor e_ij is

                1  d E
    sigma_ij = --- ------
                V  d e_ij

The strain tesnor e_ij describes the transformation for real space coordinates
in the crystal

    \sum_j [\deta_ij + e_ij] R_j  [for j = x, y, z]

Due to numerical errors, the strain tensor may slightly break the symmetry
within the stress tensor. The 6 independent components of the stress tensor

    [e1   e6/2 e5/2]
    [e6/2 e2   e4/2]
    [e5/2 e4/2 e3  ]

is constructed by symmetrizing the strain tensor as follows:

    e1 = e_11
    e2 = e_22
    e3 = e_33
    e6 = e_12 + e_21
    e5 = e_13 + e_31
    e4 = e_32 + e_23

See K. Doll, Mol Phys (2010), 108, 223
'''

import ctypes
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.pbc.lib.kpts_helper import is_zero
from gpu4pyscf.pbc.tools import pbc as pbctools
from gpu4pyscf.pbc.dft.gen_grid import UniformGrids
from gpu4pyscf.pbc.df.aft import _get_ZSI
from gpu4pyscf.pbc.df.ft_ao import libpbc
from gpu4pyscf.pbc.dft.numint import KNumInt, eval_ao_kpts, _GTOvalOpt
from gpu4pyscf.pbc.gto.cell import get_Gv
from gpu4pyscf.lib.cupy_helper import (
    contract, asarray, sandwich_dot, batched_vec_norm2)

ALIGNED = 256

def get_vxc(ks_grad, cell, dm_kpts, kpts, with_j=False, with_nuc=False):
    '''Strain derivatives for Coulomb and Exc with k-point samples

    Kwargs:
        with_j : Whether to include the electron-electron Coulomb interactions
        with_nuc : Whether to include the electron-nuclear Coulomb interactions
    '''
    mf = ks_grad.base
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension != 1

    ni = mf._numint
    assert isinstance(ni, KNumInt)
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
    elif xctype == 'HF':
        assert not with_j
        assert not with_nuc
        return np.zeros((3, 3))
    else:
        raise NotImplementedError

    assert kpts.ndim == 2
    assert dm_kpts.ndim == 3
    if not cell.cart:
        c2s = asarray(cell.cart2sph_coeff())
        dm_kpts = sandwich_dot(dm_kpts, c2s.T)
        # Ensure all AOs are evaluated in the Cartesian GTOs as ao_ks strain
        # derivatives currently supports Cartesian format only
        cell = cell.copy()
        cell.cart = True
    nkpts, nao = dm_kpts.shape[:2]
    assert nkpts == len(kpts)

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

    eval_gto_opt = _GTOvalOpt(cell, kpts, deriv=deriv+1)
    max_memory = 4e9
    blksize = int((max_memory/16/(nkpts*nvar*10*nao))/ ALIGNED) * ALIGNED
    XY, YY, ZY, XZ, YZ, ZZ = 5, 7, 8, 6, 8, 9

    out = np.zeros((3,3))
    rho0 = cp.zeros((nvar, ngrids))
    rho1 = cp.zeros((3,3, nvar, ngrids))

    for p0, p1 in lib.prange(0, ngrids, blksize):
        coords = cp.asarray(grids_coords[p0:p1].T, order='C').T
        ao_ks = eval_ao_kpts(cell, coords, kpts, deriv=deriv+1, opt=eval_gto_opt)
        ao_ks_strain = _eval_ao_strain_derivatives(
            cell, coords, kpts, deriv=deriv, opt=eval_gto_opt)
        coordsT = coords.T
        for k, dm in enumerate(dm_kpts):
            ao = ao_ks[k].transpose(0,2,1)
            ao_strain = ao_ks_strain[k]
            if xctype == 'LDA':
                ao1 = ao_strain[:,:,0]
                # Adding the response of the grids
                ao1 += contract('xig,yg->xyig', ao[1:4], coordsT)
                c0 = dm.T.dot(ao[0])
                rho0[0,p0:p1] += partial_dot(ao[0], c0).real
                rho1[:,:,0,p0:p1] += contract('xyig,ig->xyg', ao1, c0.conj()).real
            elif xctype == 'GGA':
                ao_strain[:,:,0] += contract('xig,yg->xyig', ao[1:4], coordsT)
                ao_strain[:,:,1] += contract('xig,yg->xyig', ao[4:7], coordsT)
                ao_strain[0,:,2] += contract('ig,yg->yig', ao[XY], coordsT)
                ao_strain[1,:,2] += contract('ig,yg->yig', ao[YY], coordsT)
                ao_strain[2,:,2] += contract('ig,yg->yig', ao[ZY], coordsT)
                ao_strain[0,:,3] += contract('ig,yg->yig', ao[XZ], coordsT)
                ao_strain[1,:,3] += contract('ig,yg->yig', ao[YZ], coordsT)
                ao_strain[2,:,3] += contract('ig,yg->yig', ao[ZZ], coordsT)
                c0 = contract('xig,ij->xjg', ao[:4], dm)
                for i in range(4):
                    rho0[i,p0:p1] += partial_dot(ao[0], c0[i]).real
                # TODO: computing density derivatives in FT form
                rho1[:,:, : ,p0:p1] += contract('xynig,ig->xyng', ao_strain, c0[0].conj()).real
                rho1[:,:,1:4,p0:p1] += contract('xyig,nig->xyng', ao_strain[:,:,0], c0[1:4].conj()).real
            else: # MGGA
                ao_strain[:,:,0] += contract('xig,yg->xyig', ao[1:4], coordsT)
                ao_strain[:,:,1] += contract('xig,yg->xyig', ao[4:7], coordsT)
                ao_strain[0,:,2] += contract('ig,yg->yig', ao[XY], coordsT)
                ao_strain[1,:,2] += contract('ig,yg->yig', ao[YY], coordsT)
                ao_strain[2,:,2] += contract('ig,yg->yig', ao[ZY], coordsT)
                ao_strain[0,:,3] += contract('ig,yg->yig', ao[XZ], coordsT)
                ao_strain[1,:,3] += contract('ig,yg->yig', ao[YZ], coordsT)
                ao_strain[2,:,3] += contract('ig,yg->yig', ao[ZZ], coordsT)
                c0 = contract('xig,ij->xjg', ao[:4], dm)
                for i in range(4):
                    rho0[i,p0:p1] += partial_dot(ao[0], c0[i]).real
                rho0[4,p0:p1] += partial_dot(ao[1], c0[1]).real
                rho0[4,p0:p1] += partial_dot(ao[2], c0[2]).real
                rho0[4,p0:p1] += partial_dot(ao[3], c0[3]).real
                rho1[:,:, :4,p0:p1] += contract('xynig,ig->xyng', ao_strain, c0[0].conj()).real
                rho1[:,:,1:4,p0:p1] += contract('xyig,nig->xyng', ao_strain[:,:,0], c0[1:4].conj()).real
                rho1[:,:,4,p0:p1] += contract('xynig,nig->xyg', ao_strain[:,:,1:4], c0[1:4].conj()).real

    if xctype == 'LDA':
        pass
    elif xctype == 'GGA':
        rho0[1:4] *= 2 # dm should be hermitian
    else: # MGGA
        rho0[1:4] *= 2 # dm should be hermitian
        rho0[4] *= .5 # factor 1/2 for tau
        rho1[:,:,4] *= .5

    rho0 *= 1./nkpts
    # *2 for rho1 because the derivatives were applied to the bra only
    rho1 *= 2./nkpts

    rho0_fft_order = cp.empty_like(rho0)
    rho1_fft_order = cp.empty_like(rho1)
    rho0_fft_order[:,grids_idx] = rho0
    rho1_fft_order[:,:,:,grids_idx] = rho1
    rho0, rho1 = rho0_fft_order, rho1_fft_order

    exc, vxc = ni.eval_xc_eff(xc_code, rho0, 1, xctype=xctype, spin=0)[:2]
    out += cp.einsum('xyng,ng->xy', rho1, vxc).real.get() * weight_0
    out += cp.einsum('g,g->', rho0[0], exc.ravel()).real.get() * weight_1

    out += _contract_coulomb_and_nuc(cell, mesh, dm_kpts, kpts, rho0[0],
                                     rho1[:,:,0], grids, with_j, with_nuc)
    return out

def _contract_coulomb_and_nuc(cell, mesh, dm, kpts, rho0, rho1, grids, with_j, with_nuc):
    ngrids = rho0.shape[-1]
    rhoG = pbctools.fft(rho0, mesh)
    Gv = get_Gv(cell, mesh)
    coulG_0, coulG_1 = _get_coulG_strain_derivatives(cell, Gv)
    rhoG = pbctools.fft(rho0, mesh)
    weight_0, weight_1 = _get_weight_strain_derivatives(cell, grids)
    out = 0
    if with_j:
        vR = pbctools.ifft(rhoG * coulG_0, mesh)
        EJ = cp.einsum('xyg,g->xy', rho1, vR).real.get() * weight_0 * 2
        EJ += cp.einsum('g,g->', rho0, vR).real.get() * weight_1
        EJ += cp.einsum('xyg,g->xy', coulG_1, rhoG.conj()*rhoG).real.get() * (weight_0/ngrids)
        out += .5 * EJ

    if with_nuc:
        if cell._pseudo:
            vpplocG_0, vpplocG_1 = _get_vpplocG_strain_derivatives(cell, mesh)
            vpplocR = pbctools.ifft(vpplocG_0, mesh).real
            Ene = cp.einsum('xyg,g->xy', rho1, vpplocR).real.get()
            Ene += cp.einsum('g,xyg->xy', rhoG.conj(), vpplocG_1).real.get() * (1./ngrids)
        else:
            # SI corresponds to Fourier components of the fractional atomic
            # positions within the cell. It does not respond to the strain
            # transformation
            ZG = _get_ZSI(cell, mesh)
            vR = pbctools.ifft(ZG * coulG_0, mesh).real
            Ene = cp.einsum('xyg,g->xy', rho1, vR).real.get()
            Ene += cp.einsum('xyg,g->xy', coulG_1, rhoG.conj()*ZG).real.get() * (1./ngrids)
        out += Ene
    return out

def _get_coulG_strain_derivatives(cell, Gv, omega=None):
    '''derivatives of 4pi/G^2'''
    remove_G0 = is_zero(cp.asnumpy(Gv[0]))
    Gv = asarray(Gv)
    G2 = batched_vec_norm2(Gv)
    if remove_G0:
        G2[0] = np.inf
    coulG_0 = 4 * np.pi / G2
    if omega is None:
        omega = cell.omega
    coulGxy = cp.einsum('gx,gy->xyg', Gv, Gv)
    coulGxy *= coulG_0
    coulG_1 = coulGxy * 2/G2
    if omega < 0:
        exp_omega_g2 = cp.exp(-.25/omega**2 * G2)
        coulG_1 *= 1 - exp_omega_g2
        coulG_1 -= exp_omega_g2 * (.25/omega**2*2) * coulGxy
        coulG_0 *= 1 - exp_omega_g2
        #coulG_0[0] = np.pi/omega**2
    elif omega > 0:
        exp_omega_g2 = cp.exp(-.25/omega**2 * G2)
        coulG_1 *= exp_omega_g2
        coulG_1 += exp_omega_g2 * (.25/omega**2*2) * coulGxy
        coulG_0 *= exp_omega_g2
        #coulG_0[0] = -np.pi/omega**2
    return coulG_0, coulG_1

def _get_weight_strain_derivatives(cell, grids):
    ngrids = grids.size
    weight_0 = cell.vol / ngrids
    weight_1 = np.eye(3) * weight_0
    return weight_0, weight_1

def _get_weighted_coulG_strain_derivatives(cell, Gv, omega=None):
    coulG_0, coulG_1 = _get_coulG_strain_derivatives(cell, Gv, omega=omega)
    coulG_0 = asarray(coulG_0)
    coulG_1 = asarray(coulG_1)
    vol = cell.vol
    weight_0 = 1./vol
    weight_1 = -1./vol * cp.eye(3)
    wcoulG_0 = weight_0 * coulG_0
    # wcoulG_1 includes two terms, weight_0*coulG_1 + weight_1*coulG_0
    wcoulG_1 = weight_0 * coulG_1
    wcoulG_1 += weight_1[:,:,None] * coulG_0
    return wcoulG_0, wcoulG_1

def _eval_ao_strain_derivatives(cell, coords, kpts=None, deriv=0, out=None,
                                opt=None):
    '''
    Returns:
        ao_kpts: (nkpts, 3,3,comp, nao, ngrids) ndarray
            AO values at each k-point
    '''
    assert deriv <= 2
    if opt is None:
        opt = _GTOvalOpt(cell, kpts, deriv=deriv)
    else:
        assert kpts is opt.kpts
    bvkcell = opt.bvkcell
    ngrids = len(coords)
    coords = cp.asarray(coords.T, order='C')
    bvk_ncells = opt.bvk_ncells
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    nao = cell.nao_nr()
    cart = cell.cart
    out = cp.empty((3, 3, comp, bvk_ncells, nao, ngrids))

    drv = libpbc.PBCeval_gto_strain_tensor
    err = drv(ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.byref(opt.gto_envs),
        ctypes.cast(coords.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(bvk_ncells*nao), ctypes.c_int(bvkcell.nbas),
        ctypes.c_int(deriv), ctypes.c_int(cart),
        ctypes.cast(opt.bas_rcut.data.ptr, ctypes.c_void_p))
    if err != 0:
        raise RuntimeError('PBCeval_gto_strain_tensor failed')

    if bvk_ncells == 1: # gamma point
        out = out.transpose(3,0,1,2,4,5)
    else:
        bvk_ncells, nkpts = opt.expLk.shape
        expLk = opt.expLk.view(np.float64).reshape(bvk_ncells, nkpts, 2)
        out = contract('Lks,xycLig->kxycigs', expLk, out)
        out = out.view(np.complex128)[:,:,:,:,:,:,0]
    return out

def _get_Gv_bases(mesh, b):
    Gx = cp.array(np.fft.fftfreq(mesh[0], 1./mesh[0]) * b[0,:,None])
    Gy = cp.array(np.fft.fftfreq(mesh[1], 1./mesh[1]) * b[1,:,None])
    Gz = cp.array(np.fft.fftfreq(mesh[2], 1./mesh[2]) * b[2,:,None])
    return (Gx, Gy, Gz)

def _get_vpplocG_strain_derivatives(cell, mesh):
    assert cell.dimension == 3
    Gv_bases = _get_Gv_bases(mesh, cell.reciprocal_vectors())
    coords = cp.asarray(cell.atom_coords())
    SIx = cp.exp(-1j * coords.dot(Gv_bases[0]))
    SIy = cp.exp(-1j * coords.dot(Gv_bases[1]))
    SIz = cp.exp(-1j * coords.dot(Gv_bases[2]))

    ngrids = np.prod(mesh)
    Gx, Gy, Gz = Gv_bases
    GvT = Gx[:,:,None,None] + Gy[:,None,:,None] + Gz[:,None,None,:]
    GvT = GvT.reshape(3, ngrids)
    G2 = cp.einsum('xg,xg->g', GvT, GvT)
    coulG = 4 * np.pi / G2
    coulG[0] = 0
    xyG = cp.einsum('xg,yg->xyg', GvT, GvT)

    charges = cell.atom_charges()

    vlocG0 = 0
    vlocG_0 = cp.zeros(ngrids, dtype=np.complex128)
    vlocG_1 = cp.zeros((3, 3, ngrids), dtype=np.complex128)

    for ia in range(cell.natm):
        symb = cell.atom_symbol(ia)
        if symb not in cell._pseudo:
            continue

        pp = cell._pseudo[symb]
        rloc, nexp, cexp = pp[1:3+1]

        SI = (SIx[ia,:,None,None] * SIy[ia,:,None] * SIz[ia]).ravel()
        x = G2 * rloc**2
        expx = cp.exp(-0.5*x)
        SI *= expx
        Z = charges[ia]

        coef1 = -Z * coulG * SI * (2/G2 + rloc**2)
        coef1[0] = 0

        cfacs = 0
        dcfacs = 0
        if nexp >= 1:
            cfacs += cexp[0]
        if nexp >= 2:
            cfacs += cexp[1] * (3 - x)
            dcfacs -= cexp[1]
        if nexp >= 3:
            cfacs += cexp[2] * (15 - 10*x + x*x)
            dcfacs += cexp[2] * (-10 + 2*x)
        if nexp >= 4:
            cfacs += cexp[3] * (105 - 105*x + 21*x*x - x*x*x)
            dcfacs += cexp[3] * (-105 + 42*x - 3*x*x)

        coef2 = (
            (2*np.pi)**1.5
            * rloc**5
            * SI
            * (cfacs - 2 * dcfacs)
        )

        vlocG0 += 2*np.pi*Z*rloc**2
        vlocG_0 -= Z * coulG * SI
        vlocG_0 += (2*np.pi)**(3/2.)*rloc**3 * cfacs * SI

        vlocG_1 += (coef1 + coef2) * xyG

    vlocG_0[0] += vlocG0
    return vlocG_0, vlocG_1
