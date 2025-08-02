# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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

import cupy as cp
import numpy as np
from gpu4pyscf import scf, dft
from gpu4pyscf.lib import logger
from gpu4pyscf.tdscf import ris


def diagonalize_tda(a, nroots=5):
    nocc, nvir = a.shape[:2]
    nov = nocc * nvir
    a = a.reshape(nov, nov)
    e, xy = np.linalg.eig(np.asarray(a))
    sorted_indices = np.argsort(e)

    e_sorted = e[sorted_indices]
    xy_sorted = xy[:, sorted_indices]

    e_sorted_final = e_sorted[e_sorted > 1e-3]
    xy_sorted = xy_sorted[:, e_sorted > 1e-3]
    return e_sorted_final[:nroots], xy_sorted[:, :nroots]


def change_sign(s12_ao, mo_coeff_b ,mo_coeff):
    mo_coeff_new = mo_coeff*1.0
    s12_mo = mo_coeff_b.T @ s12_ao @ mo_coeff_new
    for i in range(s12_mo.shape[-1]):
        if s12_mo[i,i] < 0.0:
            mo_coeff_new[:,i] *= -1
        if s12_mo[i,i]**2 < 1.0e-10:
            max_norm = -1
            for j in range(s12_mo.shape[-1]):
                if s12_mo[i,j]**2 > max_norm:
                    max_norm = s12_mo[i,j]**2
                    idx = j
            mo_coeff_new[:,i] = mo_coeff[:,idx]
            if mo_coeff_b[:,i].T @ s12_ao @ mo_coeff_new[:,i] < 0.0:
                mo_coeff_new[:,i] *= -1
    return mo_coeff_new


def get_new_mol(mol, coords, delta, iatm, icart):
    coords_new = coords*1.0
    coords_new[iatm, icart] += delta
    mol_new = mol.copy()
    mol_new.set_geom_(coords_new, unit='Ang')
    return mol_new


def get_mf(mol, mf, s, mo_coeff):
    if isinstance(mf, dft.rks.RKS):
        mf_new = dft.RKS(mol)
        mf_new.xc = mf.xc
        if len(mf.grids.atom_grid) > 0:
            mf_new.grids.atom_grid = mf.grids.atom_grid
        else:
            mf_new.grids.level = mf.grids.level
    else:
        mf_new = scf.RHF(mol)
    if getattr(mf, 'with_df', None) is not None:
        mf_new = mf_new.density_fit()
    mf_new.conv_tol = mf.conv_tol
    mf_new.conv_tol_cpscf = mf.conv_tol_cpscf
    mf_new.max_cycle = mf.max_cycle
    mf_new.kernel()
    assert mf_new.converged
    mo_coeff_new = change_sign(s, mo_coeff, mf_new.mo_coeff)
    mf_new.mo_coeff = mo_coeff_new

    return mf_new


def get_mf_td(mol, mf, s, mo_coeff, with_ris=False):
    mf_new = get_mf(mol, mf, s, mo_coeff)
    if with_ris:
        td_new = ris.TDA(mf=mf_new, nstates=5, spectra=False, Ktrunc = 0.0, single=False, gram_schmidt=True)
    else:
        td_new = mf_new.TDA()
    a, b = td_new.get_ab()
    e_diag, xy_diag = diagonalize_tda(a)

    return mf_new, xy_diag


def get_nacv_ge(td_nac, x_yI, delta=0.001, with_ris=False, singlet=True, atmlst=None, verbose=logger.INFO):
    mf = td_nac.base._scf
    mol = mf.mol

    coords = mol.atom_coords(unit='Ang')*1.0
    natm = coords.shape[0]
    nac = np.zeros((natm, 3))
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)
    nao, nmo = mo_coeff.shape
    nocc = int((mo_occ > 0).sum())
    nvir = nmo - nocc
    xI, yI = x_yI
    xI = cp.asarray(xI).reshape(nocc, nvir)
    if not isinstance(yI, np.ndarray) and not isinstance(yI, cp.ndarray):
        yI = cp.zeros_like(xI)
    yI = yI.reshape(nocc, nvir)

    gamma = np.block([[np.zeros((nocc, nocc)), xI.get()], 
                      [(xI.T*0.0).get(), np.zeros((nvir, nvir))]])
    gamma = cp.asarray(gamma)*2
    gamma_ao = mo_coeff @ gamma @ mo_coeff.T
    s = mol.intor('int1e_ovlp')
    s = cp.asarray(s)

    for iatm in range(natm):
        for icart in range(3):
            mol_add = get_new_mol(mol, coords, delta, iatm, icart)
            mf_add = get_mf(mol_add, mf, s, mo_coeff)
            mol_minus = get_new_mol(mol, coords, -delta, iatm, icart)
            mf_minus = get_mf(mol_minus, mf, s, mo_coeff)

            mo_diff = (mf_add.mo_coeff - mf_minus.mo_coeff)/(delta*2.0)*0.52917721092
            dpq = mo_coeff.T @ s @ mo_diff
            nac[iatm, icart] = (gamma*dpq).sum()

    nac2 = np.zeros((natm, 3))
    atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    s12_deriv = mol.intor('int1e_ipovlp')
    s12_deriv = cp.asarray(s12_deriv)
    for k, ia in enumerate(atmlst): 
        shl0, shl1, p0, p1 = offsetdic[ia]
        s12_deriv_tmp = s12_deriv*1.0
        ds1_tmp = s12_deriv_tmp.transpose(0,2,1)
        ds1_tmp[:,:,:p0] = 0
        ds1_tmp[:,:,p1:] = 0
        nac2[k] = cp.einsum('xij,ij->x', ds1_tmp, gamma_ao).get()
    return nac - nac2


def get_nacv_ee(td_nac, x_yI, x_yJ, nJ, delta=0.001, with_ris=False, singlet=True, atmlst=None, verbose=logger.INFO):
    mf = td_nac.base._scf
    mol = mf.mol
    coords = mol.atom_coords(unit='Ang')*1.0
    natm = coords.shape[0]
    nac = np.zeros((natm, 3))
    nac3 = np.zeros((natm, 3))
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)
    nao, nmo = mo_coeff.shape
    nocc = int((mo_occ > 0).sum())
    nvir = nmo - nocc
    xI, yI = x_yI
    xJ, yJ = x_yJ
    xI = cp.asarray(xI).reshape(nocc, nvir)
    if not isinstance(yI, np.ndarray) and not isinstance(yI, cp.ndarray):
        yI = cp.zeros_like(xI)
    yI = cp.asarray(yI).reshape(nocc, nvir)
    xJ = cp.asarray(xJ).reshape(nocc, nvir)
    if not isinstance(yJ, np.ndarray) and not isinstance(yJ, cp.ndarray):
        yJ = cp.zeros_like(xJ)
    yJ = cp.asarray(yJ).reshape(nocc, nvir)
    gamma = np.block([[(-xJ@xI.T).get(), np.zeros((nocc, nvir))], 
                    [np.zeros((nvir, nocc)), (xI.T@xJ).get()]]) * 2
    gamma = cp.asarray(gamma)
    gamma_ao = mo_coeff @ gamma @ mo_coeff.T
    s = mol.intor('int1e_ovlp')
    s = cp.asarray(s)
    for iatm in range(natm):
        for icart in range(3):
            mol_add = get_new_mol(mol, coords, delta, iatm, icart)
            mf_add, xy_diag_add = get_mf_td(mol_add, mf, s, mo_coeff, with_ris)
            mol_minus = get_new_mol(mol, coords, -delta, iatm, icart)
            mf_minus, xy_diag_minus = get_mf_td(mol_minus, mf, s, mo_coeff, with_ris)

            sign1 = 1.0
            sign2 = 1.0
            xJ_add = cp.asarray(xy_diag_add[:, nJ]).reshape(nocc, nvir)*cp.sqrt(0.5)
            xJ_minus = cp.asarray(xy_diag_minus[:, nJ]).reshape(nocc, nvir)*cp.sqrt(0.5)
            if (xJ*xJ_add).sum() < 0.0:
                sign1 = -1.0
            if (xJ*xJ_minus).sum() < 0.0:
                sign2 = -1.0
            
            mo_diff = (mf_add.mo_coeff - mf_minus.mo_coeff)/(delta*2.0)*0.52917721092
            dpq = mo_coeff.T @ s @ mo_diff
            nac[iatm, icart] = (gamma*dpq).sum()

            t_diff = (xJ_add*sign1 - xJ_minus*sign2)/(delta*2.0)*0.52917721092
            nac3[iatm, icart] = (xI*t_diff).sum()*2 # for double occupancy
    
    nac2 = np.zeros((natm, 3))
    atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    s12_deriv = mol.intor('int1e_ipovlp')
    s12_deriv = cp.asarray(s12_deriv)
    for k, ia in enumerate(atmlst): 
        shl0, shl1, p0, p1 = offsetdic[ia]
        s12_deriv_tmp = s12_deriv*1.0
        ds1_tmp = s12_deriv_tmp.transpose(0,2,1)
        ds1_tmp[:,:,:p0] = 0
        ds1_tmp[:,:,p1:] = 0
        nac2[k] = cp.einsum('xij,ij->x', ds1_tmp, gamma_ao).get()
    return nac - nac2 + nac3
   