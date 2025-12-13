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
"""
Nonadiabatic derivetive coupling matrix element calculation is now in experiment.
This module is under development.
"""

from functools import reduce
import cupy as cp
import numpy as np
from pyscf import lib, gto
import pyscf
from gpu4pyscf.lib import logger
from pyscf.grad import rhf as rhf_grad_cpu
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.df import int3c2e
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.scf import cphf
from pyscf import __config__
from gpu4pyscf.lib import utils
from gpu4pyscf import tdscf
from pyscf.scf import _vhf
from scipy.optimize import linear_sum_assignment


def match_and_reorder_mos(s12_ao, mo_coeff_b, mo_coeff, threshold=0.4):
    if mo_coeff_b.shape != mo_coeff.shape:
        raise ValueError("Mo coeff b and mo coeff must have the same shape.")
    if s12_ao.shape[0] != s12_ao.shape[1] or s12_ao.shape[0] != mo_coeff_b.shape[0]:
        raise ValueError("S12 ao must be a square matrix with the same shape as mo coeff b.")
    mo_overlap_matrix = mo_coeff_b.T @ s12_ao @ mo_coeff
    abs_mo_overlap = cp.abs(mo_overlap_matrix)
    cost_matrix = -abs_mo_overlap
    below_threshold_mask = abs_mo_overlap < threshold
    infinity_cost = mo_coeff_b.shape[1] + 1
    cost_matrix[below_threshold_mask] = infinity_cost
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix.get())

    matching_indices = col_ind
    
    mo2_reordered = mo_coeff[:, matching_indices]

    final_chosen_overlaps = abs_mo_overlap[row_ind, col_ind]
    invalid_matches_mask = final_chosen_overlaps < threshold
    
    if cp.any(invalid_matches_mask):
        num_invalid = cp.sum(invalid_matches_mask)
        print(
            f"{num_invalid} orbital below threshold {threshold}."
            "This may indicate significant changes in the properties of these orbitals between the two structures."
        )
        invalid_indices = cp.where(invalid_matches_mask)[0]
        for idx in invalid_indices:
            print(f"Warning: reference coeff #{idx}'s best match is {final_chosen_overlaps[idx]:.4f} (below threshold {threshold})")
    s_mo_new = mo_coeff_b.T @ s12_ao @ mo2_reordered
    sign_array = cp.ones(s_mo_new.shape[-1])
    for i in range(s_mo_new.shape[-1]):
        if s_mo_new[i,i] < 0.0:
            # mo2_reordered[:,i] *= -1
            sign_array[i] = -1
    return mo2_reordered, matching_indices, sign_array


def get_nacv_ge(td_nac, x_yI, EI, singlet=True, atmlst=None, verbose=logger.INFO):
    """
    Calculate non-adiabatic coupling vectors between ground and excited states.
    Now, only supports for singlet states.
    Ref:
    [1] 10.1063/1.4903986 main reference
    [2] 10.1021/acs.accounts.1c00312
    [3] 10.1063/1.4885817

    Args:
        td_nac (gpu4pyscf.tdscf.rhf.TDA): Non-adiabatic coupling object for TDDFT or TDHF.
        x_yI (tuple): (xI, yI), xI and yI are the eigenvectors corresponding to the excitation and de-excitation.
        EI (float): excitation energy for state I

    Kwargs:
        singlet (bool): Whether calculate singlet states.
        atmlst (list): List of atoms to calculate the NAC.
        verbose (int): Verbosity level.

    Returns:
        nacv (np.ndarray): NAC matrix element.
    """
    if singlet is False:
        raise NotImplementedError('Only supports for singlet states')
    mol = td_nac.mol
    mf = td_nac.base._scf
    mf_grad = mf.nuc_grad_method()
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    nao, nmo = mo_coeff.shape
    nocc = int((mo_occ > 0).sum())
    nvir = nmo - nocc
    orbv = mo_coeff[:, nocc:]
    orbo = mo_coeff[:, :nocc]

    xI, yI = x_yI
    xI = cp.asarray(xI).reshape(nocc, nvir).T
    if not isinstance(yI, np.ndarray) and not isinstance(yI, cp.ndarray):
        yI = cp.zeros_like(xI)
    yI = cp.asarray(yI).reshape(nocc, nvir).T
    LI = xI-yI    # eq.(83) in Ref. [1]

    vresp = td_nac.base.gen_response(singlet=None, hermi=1)

    def fvind(x):
        dm = reduce(cp.dot, (orbv, x.reshape(nvir, nocc) * 2, orbo.T)) # double occupency
        v1ao = vresp(dm + dm.T)
        return reduce(cp.dot, (orbv.T, v1ao, orbo)).ravel()

    z1 = cphf.solve(
        fvind,
        mo_energy,
        mo_occ,
        -LI*1.0*EI, # only one spin, negative in cphf
        max_cycle=td_nac.cphf_max_cycle,
        tol=td_nac.cphf_conv_tol)[0] # eq.(83) in Ref. [1]

    z1 = z1.reshape(nvir, nocc)
    z1ao = reduce(cp.dot, (orbv, z1, orbo.T)) * 2 # double occupency
    # eq.(50) in Ref. [1]
    z1aoS = (z1ao + z1ao.T)*0.5 # 0.5 is in the definition of z1aoS
    # eq.(73) in Ref. [1]
    GZS = vresp(z1aoS) # generate the double occupency 
    GZS_mo = reduce(cp.dot, (mo_coeff.T, GZS, mo_coeff))
    W = cp.zeros((nmo, nmo))  # eq.(75) in Ref. [1]
    W[:nocc, :nocc] = GZS_mo[:nocc, :nocc]
    zeta0 = mo_energy[nocc:, cp.newaxis]
    zeta0 = z1 * zeta0
    W[:nocc, nocc:] = GZS_mo[:nocc, nocc:] + 0.5*yI.T*EI + 0.5*zeta0.T #* eq.(43), (56), (28) in Ref. [1]
    zeta1 = mo_energy[cp.newaxis, :nocc]
    zeta1 = z1 * zeta1
    W[nocc:, :nocc] = 0.5*xI*EI + 0.5*zeta1
    W = reduce(cp.dot, (mo_coeff, W , mo_coeff.T)) * 2.0

    mf_grad = mf.nuc_grad_method()
    dmz1doo = z1aoS
    td_nac._dmz1doo = dmz1doo
    oo0 = reduce(cp.dot, (orbo, orbo.T)) * 2.0

    if atmlst is None:
        atmlst = range(mol.natm)
    
    h1 = cp.asarray(mf_grad.get_hcore(mol))  # without 1/r like terms
    s1 = cp.asarray(mf_grad.get_ovlp(mol))
    dh_td = contract("xij,ij->xi", h1, dmz1doo)
    ds = contract("xij,ij->xi", s1, (W + W.T))

    dh1e_td = int3c2e.get_dh1e(mol, dmz1doo)  # 1/r like terms
    if mol.has_ecp():
        dh1e_td += rhf_grad.get_dh1e_ecp(mol, dmz1doo)  # 1/r like terms
    extra_force = cp.zeros((len(atmlst), 3))

    dvhf_all = 0
    dvhf = td_nac.get_veff(mol, dmz1doo + oo0) 
    for k, ia in enumerate(atmlst):
        extra_force[k] += mf_grad.extra_force(ia, locals())
    dvhf_all += dvhf
    dvhf = td_nac.get_veff(mol, dmz1doo)
    for k, ia in enumerate(atmlst):
        extra_force[k] -= mf_grad.extra_force(ia, locals())
    dvhf_all -= dvhf
    dvhf = td_nac.get_veff(mol, oo0)
    for k, ia in enumerate(atmlst):
        extra_force[k] -= mf_grad.extra_force(ia, locals())
    dvhf_all -= dvhf

    delec = dh_td*2 - ds
    aoslices = mol.aoslice_by_atom()
    delec = cp.asarray([cp.sum(delec[:, p0:p1], axis=1) for p0, p1 in aoslices[:, 2:]])

    xIao = reduce(cp.dot, (orbo, xI.T, orbv.T)) * 2
    yIao = reduce(cp.dot, (orbv, yI, orbo.T)) * 2
    ds_x = contract("xij,ji->xi", s1, xIao*EI)
    ds_y = contract("xij,ji->xi", s1, yIao*EI)
    ds_x_etf = contract("xij,ij->xi", s1, (xIao*EI + xIao.T*EI) * 0.5)
    ds_y_etf = contract("xij,ij->xi", s1, (yIao*EI + yIao.T*EI) * 0.5)
    dsxy = cp.asarray([cp.sum(ds_x[:, p0:p1] + ds_y[:, p0:p1], axis=1) for p0, p1 in aoslices[:, 2:]])
    dsxy_etf = cp.asarray([cp.sum(ds_x_etf[:, p0:p1] + ds_y_etf[:, p0:p1], axis=1) for p0, p1 in aoslices[:, 2:]])
    de = 2.0 * dvhf_all + extra_force + dh1e_td + delec 
    de_etf = de + dsxy_etf
    de += dsxy 
    
    de = de.get()
    de_etf = de_etf.get()
    return de, de/EI, de_etf, de_etf/EI


def get_nacv_ee(td_nac, x_yI, x_yJ, EI, EJ, singlet=True, atmlst=None, verbose=logger.INFO):
    """
    Only supports for excited-excited states. 
    Quadratic-response-associated terms are all neglected.
    
    Ref:
    [1] 10.1063/1.4903986 main reference
    [2] 10.1021/acs.accounts.1c00312
    [3] 10.1063/1.4885817

    Args:
        td_nac: TDNAC object
        x_yI: (xI, yI) 
            xI and yI are the eigenvectors corresponding to the excitation and de-excitation for state I
        x_yJ: (xJ, yJ) 
            xJ and yJ are the eigenvectors corresponding to the excitation and de-excitation for state J
        EI: energy of state I
        EJ: energy of state J
    
    Keyword args:
        singlet (bool): Whether calculate singlet states.
        atmlst (list): List of atoms to calculate the NAC.
        verbose (int): Verbosity level.
    """
    if singlet is False:
        raise NotImplementedError('Only supports for singlet states')
    mol = td_nac.mol
    mf = td_nac.base._scf
    mf_grad = mf.nuc_grad_method()
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    nao, nmo = mo_coeff.shape
    nocc = int((mo_occ > 0).sum())
    nvir = nmo - nocc
    orbv = mo_coeff[:, nocc:]
    orbo = mo_coeff[:, :nocc]

    xI, yI = x_yI
    xJ, yJ = x_yJ
    
    xI = cp.asarray(xI).reshape(nocc, nvir).T
    if not isinstance(yI, np.ndarray) and not isinstance(yI, cp.ndarray):
        yI = cp.zeros_like(xI)
    yI = cp.asarray(yI).reshape(nocc, nvir).T
    xJ = cp.asarray(xJ).reshape(nocc, nvir).T
    if not isinstance(yJ, np.ndarray) and not isinstance(yJ, cp.ndarray):
        yJ = cp.zeros_like(xJ)
    yJ = cp.asarray(yJ).reshape(nocc, nvir).T

    xpyI = (xI + yI)
    xmyI = (xI - yI)
    dmxpyI = reduce(cp.dot, (orbv, xpyI, orbo.T))
    dmxmyI = reduce(cp.dot, (orbv, xmyI, orbo.T))
    xpyJ = (xJ + yJ)
    xmyJ = (xJ - yJ)
    dmxpyJ = reduce(cp.dot, (orbv, xpyJ, orbo.T)) 
    dmxmyJ = reduce(cp.dot, (orbv, xmyJ, orbo.T)) 
    td_nac._dmxpyI = dmxpyI
    td_nac._dmxpyJ = dmxpyJ

    rIJoo =-contract('ai,aj->ij', xJ, xI) - contract('ai,aj->ij', yI, yJ)
    rIJvv = contract('ai,bi->ab', xI, xJ) + contract('ai,bi->ab', yJ, yI)
    TIJoo = (rIJoo + rIJoo.T) * 0.5
    TIJvv = (rIJvv + rIJvv.T) * 0.5
    dmzooIJ = reduce(cp.dot, (orbo, TIJoo, orbo.T)) * 2
    dmzooIJ += reduce(cp.dot, (orbv, TIJvv, orbv.T)) * 2

    vj0IJ, vk0IJ = mf.get_jk(mol, dmzooIJ, hermi=0)
    vj1I, vk1I = mf.get_jk(mol, (dmxpyI + dmxpyI.T), hermi=0)
    vj2I, vk2I = mf.get_jk(mol, (dmxmyI - dmxmyI.T), hermi=0)
    vj1J, vk1J = mf.get_jk(mol, (dmxpyJ + dmxpyJ.T), hermi=0)
    vj2J, vk2J = mf.get_jk(mol, (dmxmyJ - dmxmyJ.T), hermi=0)
    if not isinstance(vj0IJ, cp.ndarray):
        vj0IJ = cp.asarray(vj0IJ)
    if not isinstance(vk0IJ, cp.ndarray):
        vk0IJ = cp.asarray(vk0IJ)
    if not isinstance(vj1I, cp.ndarray):
        vj1I = cp.asarray(vj1I)
    if not isinstance(vk1I, cp.ndarray):
        vk1I = cp.asarray(vk1I)
    if not isinstance(vj2I, cp.ndarray):
        vj2I = cp.asarray(vj2I)
    if not isinstance(vk2I, cp.ndarray):
        vk2I = cp.asarray(vk2I)
    if not isinstance(vj1J, cp.ndarray):
        vj1J = cp.asarray(vj1J)
    if not isinstance(vk1J, cp.ndarray):
        vk1J = cp.asarray(vk1J)
    if not isinstance(vj2J, cp.ndarray):
        vj2J = cp.asarray(vj2J)
    if not isinstance(vk2J, cp.ndarray):
        vk2J = cp.asarray(vk2J)

    veff0doo = vj0IJ * 2 - vk0IJ
    veff0doo += td_nac.solvent_response(dmzooIJ)
    wvo = reduce(cp.dot, (orbv.T, veff0doo, orbo)) * 2
    veffI = vj1I * 2 - vk1I
    veffI += td_nac.solvent_response(dmxpyI + dmxpyI.T)
    veffI *= 0.5
    veff0mopI = reduce(cp.dot, (mo_coeff.T, veffI, mo_coeff))
    wvo -= contract("ki,ai->ak", veff0mopI[:nocc, :nocc], xpyJ) * 2  
    wvo += contract("ac,ai->ci", veff0mopI[nocc:, nocc:], xpyJ) * 2
    veffJ = vj1J * 2 - vk1J
    veffJ += td_nac.solvent_response(dmxpyJ + dmxpyJ.T)
    veffJ *= 0.5
    veff0mopJ = reduce(cp.dot, (mo_coeff.T, veffJ, mo_coeff))
    wvo -= contract("ki,ai->ak", veff0mopJ[:nocc, :nocc], xpyI) * 2  
    wvo += contract("ac,ai->ci", veff0mopJ[nocc:, nocc:], xpyI) * 2
    veffI = -vk2I
    veffI *= 0.5
    veff0momI = reduce(cp.dot, (mo_coeff.T, veffI, mo_coeff))
    wvo -= contract("ki,ai->ak", veff0momI[:nocc, :nocc], xmyJ) * 2
    wvo += contract("ac,ai->ci", veff0momI[nocc:, nocc:], xmyJ) * 2
    veffJ = -vk2J
    veffJ *= 0.5
    veff0momJ = reduce(cp.dot, (mo_coeff.T, veffJ, mo_coeff))
    wvo -= contract("ki,ai->ak", veff0momJ[:nocc, :nocc], xmyI) * 2
    wvo += contract("ac,ai->ci", veff0momJ[nocc:, nocc:], xmyI) * 2
    # The up parts are according to eq. (86) and (86) in Ref. [1]

    vresp = td_nac.base.gen_response(singlet=None, hermi=1)

    def fvind(x):
        dm = reduce(cp.dot, (orbv, x.reshape(nvir, nocc) * 2, orbo.T)) # double occupency
        v1ao = vresp(dm + dm.T)
        return reduce(cp.dot, (orbv.T, v1ao, orbo)).ravel()

    z1 = cphf.solve(
        fvind,
        mo_energy,
        mo_occ,
        wvo/(EJ-EI), # only one spin, negative in cphf
        max_cycle=td_nac.cphf_max_cycle,
        tol=td_nac.cphf_conv_tol)[0] # eq.(80) in Ref. [1]

    z1ao = reduce(cp.dot, (orbv, z1, orbo.T))
    veff = vresp((z1ao + z1ao.T))
    fock_mo = cp.diag(mo_energy)
    TFoo = cp.dot(TIJoo, fock_mo[:nocc,:nocc])
    TFov = cp.dot(TIJoo, fock_mo[:nocc,nocc:])
    TFvo = cp.dot(TIJvv, fock_mo[nocc:,:nocc])
    TFvv = cp.dot(TIJvv, fock_mo[nocc:,nocc:])

    # W is calculated, eqs. (75)~(78) in Ref. [1]
    # in which g_{IJ} (86) in Ref. [1] is calculated
    im0 = cp.zeros((nmo, nmo))
    im0[:nocc, :nocc] = reduce(cp.dot, (orbo.T, veff0doo, orbo)) # 1st term in Eq. (81) in Ref. [1]
    im0[:nocc, :nocc]+= TFoo*2.0 # 2nd term in Eq. (81) in Ref. [1]
    im0[:nocc, :nocc]+= contract("ak,ai->ik", veff0mopI[nocc:, :nocc], xpyJ) # 3rd term in Eq. (81) in Ref. [1]
    im0[:nocc, :nocc]+= contract("ak,ai->ik", veff0momI[nocc:, :nocc], xmyJ) # 4th term in Eq. (81) in Ref. [1]
    im0[:nocc, :nocc]+= contract("ak,ai->ik", veff0mopJ[nocc:, :nocc], xpyI) # 5th term in Eq. (81) in Ref. [1]
    im0[:nocc, :nocc]+= contract("ak,ai->ik", veff0momJ[nocc:, :nocc], xmyI) # 6th term in Eq. (81) in Ref. [1]
    im0[:nocc, :nocc]+=rIJoo.T*(EJ-EI) # only gamma^{IJ}(II) in Eq. (29) in Ref. [1] is considered.

    im0[:nocc, nocc:] = reduce(cp.dot, (orbo.T, veff0doo, orbv))
    im0[:nocc, nocc:]+= TFov*2.0
    im0[:nocc, nocc:]+= contract("ab,ai->ib", veff0mopI[nocc:, nocc:], xpyJ)
    im0[:nocc, nocc:]+= contract("ab,ai->ib", veff0momI[nocc:, nocc:], xmyJ)
    im0[:nocc, nocc:]+= contract("ab,ai->ib", veff0mopJ[nocc:, nocc:], xpyI)
    im0[:nocc, nocc:]+= contract("ab,ai->ib", veff0momJ[nocc:, nocc:], xmyI)

    im0[nocc:, :nocc] = TFvo*2
    im0[nocc:, :nocc]+= contract("ij,ai->aj", veff0mopI[:nocc, :nocc], xpyJ)
    im0[nocc:, :nocc]-= contract("ij,ai->aj", veff0momI[:nocc, :nocc], xmyJ)
    im0[nocc:, :nocc]+= contract("ij,ai->aj", veff0mopJ[:nocc, :nocc], xpyI)
    im0[nocc:, :nocc]-= contract("ij,ai->aj", veff0momJ[:nocc, :nocc], xmyI)

    im0[nocc:, nocc:] = TFvv*2.0
    im0[nocc:, nocc:]+= contract("ib,ai->ab", veff0mopI[:nocc, nocc:], xpyJ)
    im0[nocc:, nocc:]-= contract("ib,ai->ab", veff0momI[:nocc, nocc:], xmyJ)
    im0[nocc:, nocc:]+= contract("ib,ai->ab", veff0mopJ[:nocc, nocc:], xpyI)
    im0[nocc:, nocc:]-= contract("ib,ai->ab", veff0momJ[:nocc, nocc:], xmyI)
    im0[nocc:, nocc:]+=rIJvv.T*(EJ-EI)

    im0 = im0*0.5
    im0[:nocc, :nocc]+= reduce(cp.dot, (orbo.T, veff, orbo))*(EJ-EI)*0.5
    im0[:nocc, nocc:]+= reduce(cp.dot, (orbo.T, veff, orbv))*(EJ-EI)*0.5
    im0[:nocc, nocc:]+= cp.dot(fock_mo[nocc:,nocc:],z1).T*(EJ-EI)*0.25
    im0[nocc:, :nocc]+= cp.dot(z1, fock_mo[:nocc,:nocc]*(EJ-EI))*0.25
    # 0.5 * 0.5 first is in the equation,
    # second 0.5 due to z1.
    # The up parts are according to eqs. (75)~(78) in Ref. [1]
    # * It should be noted that, the quadratic response part is omitted!

    im0 = reduce(cp.dot, (mo_coeff, im0, mo_coeff.T))*2

    mf_grad = td_nac.base._scf.nuc_grad_method()
    s1 = mf_grad.get_ovlp(mol)
    z1aoS = (z1ao + z1ao.T)*0.5* (EJ - EI)
    dmz1doo = z1aoS + dmzooIJ  # P
    td_nac._dmz1doo = dmz1doo
    oo0 = reduce(cp.dot, (orbo, orbo.T))*2  # D

    if atmlst is None:
        atmlst = range(mol.natm)
    
    h1 = cp.asarray(mf_grad.get_hcore(mol))  # without 1/r like terms
    s1 = cp.asarray(mf_grad.get_ovlp(mol))
    dh_td = contract("xij,ij->xi", h1, dmz1doo)
    ds = contract("xij,ij->xi", s1, (im0 + im0.T))

    dh1e_td = int3c2e.get_dh1e(mol, dmz1doo)  # 1/r like terms
    if mol.has_ecp():
        dh1e_td += rhf_grad.get_dh1e_ecp(mol, dmz1doo)  # 1/r like terms
    extra_force = cp.zeros((len(atmlst), 3))

    dvhf_all = 0
    dvhf = td_nac.get_veff(mol, dmz1doo + oo0) 
    for k, ia in enumerate(atmlst):
        extra_force[k] += cp.asarray(mf_grad.extra_force(ia, locals()))
    dvhf_all += dvhf
    # minus in the next TWO terms is due to only <g^{(\xi)};{D,P_{IJ}}> is needed, 
    # thus minus the contribution from same DM ({D,D}, {P,P}).
    dvhf = td_nac.get_veff(mol, dmz1doo)
    for k, ia in enumerate(atmlst):
        extra_force[k] -= cp.asarray(mf_grad.extra_force(ia, locals()))
    dvhf_all -= dvhf
    dvhf = td_nac.get_veff(mol, oo0)
    for k, ia in enumerate(atmlst):
        extra_force[k] -= cp.asarray(mf_grad.extra_force(ia, locals()))
    dvhf_all -= dvhf
    j_factor=1.0
    k_factor=1.0
    dvhf = td_nac.get_veff(mol, (dmxpyI + dmxpyI.T + dmxpyJ + dmxpyJ.T), j_factor, k_factor)
    for k, ia in enumerate(atmlst):
        extra_force[k] += cp.asarray(mf_grad.extra_force(ia, locals()))
    dvhf_all += dvhf
    # minus in the next TWO terms is due to only <g^{(\xi)};{R_I^S, R_J^S}> is needed, 
    # thus minus the contribution from same DM ({R_I^S,R_I^S} and {R_J^S,R_J^S}).
    dvhf = td_nac.get_veff(mol, (dmxpyI + dmxpyI.T), j_factor, k_factor)
    for k, ia in enumerate(atmlst):
        extra_force[k] -= cp.asarray(mf_grad.extra_force(ia, locals()))
    dvhf_all -= dvhf # NOTE: minus
    dvhf = td_nac.get_veff(mol, (dmxpyJ + dmxpyJ.T), j_factor, k_factor)
    for k, ia in enumerate(atmlst):
        extra_force[k] -= cp.asarray(mf_grad.extra_force(ia, locals()))
    dvhf_all -= dvhf
    dvhf = td_nac.get_veff(mol, (dmxmyI - dmxmyI.T + dmxmyJ - dmxmyJ.T), 0.0, k_factor, hermi=2)
    for k, ia in enumerate(atmlst):
        extra_force[k] += cp.asarray(mf_grad.extra_force(ia, locals()))
    dvhf_all += dvhf
    dvhf = td_nac.get_veff(mol, (dmxmyI - dmxmyI.T), 0.0, k_factor, hermi=2)
    for k, ia in enumerate(atmlst):
        extra_force[k] -= cp.asarray(mf_grad.extra_force(ia, locals()))
    dvhf_all -= dvhf
    dvhf = td_nac.get_veff(mol, (dmxmyJ - dmxmyJ.T), 0.0, k_factor, hermi=2)
    for k, ia in enumerate(atmlst):
        extra_force[k] -= cp.asarray(mf_grad.extra_force(ia, locals()))
    dvhf_all -= dvhf

    delec = dh_td*2 - ds
    aoslices = mol.aoslice_by_atom()
    delec = cp.asarray([cp.sum(delec[:, p0:p1], axis=1) for p0, p1 in aoslices[:, 2:]])

    rIJoo_ao = reduce(cp.dot, (orbo, rIJoo, orbo.T))*2
    rIJvv_ao = reduce(cp.dot, (orbv, rIJvv, orbv.T))*2
    rIJooS_ao = reduce(cp.dot, (orbo, TIJoo, orbo.T))*2
    rIJvvS_ao = reduce(cp.dot, (orbv, TIJvv, orbv.T))*2
    ds_oo = contract("xij,ji->xi", s1, rIJoo_ao * (EJ - EI))
    ds_vv = contract("xij,ji->xi", s1, rIJvv_ao * (EJ - EI))
    ds_oo_etf = contract("xij,ji->xi", s1, rIJooS_ao * (EJ - EI))
    ds_vv_etf = contract("xij,ji->xi", s1, rIJvvS_ao * (EJ - EI))
    dsxy = cp.asarray([cp.sum(ds_oo[:, p0:p1] + ds_vv[:, p0:p1], axis=1) for p0, p1 in aoslices[:, 2:]])
    dsxy_etf = cp.asarray([cp.sum(ds_oo_etf[:, p0:p1] + ds_vv_etf[:, p0:p1], axis=1) for p0, p1 in aoslices[:, 2:]])
    de = 2.0 * dvhf_all + extra_force + dh1e_td + delec  # Eq. (64) in Ref. [1]
    de_etf = de + dsxy_etf
    de += dsxy 
    
    de = de.get()
    de_etf = de_etf.get()
    return de, de/(EJ - EI), de_etf, de_etf/(EJ - EI)


class NAC(lib.StreamObject):

    cphf_max_cycle = getattr(__config__, "grad_tdrhf_Gradients_cphf_max_cycle", 50)
    cphf_conv_tol = getattr(__config__, "grad_tdrhf_Gradients_cphf_conv_tol", 1e-8)

    to_cpu = utils.to_cpu
    to_gpu = utils.to_gpu
    device = utils.device

    _keys = {
        "cphf_max_cycle",
        "cphf_conv_tol",
        "mol",
        "base",
        "states",
        "atmlst",
        "de",
        "de_scaled",
        "de_etf",
        "de_etf_scaled",
    }

    def __init__(self, td):
        self.verbose = td.verbose
        self.stdout = td.stdout
        self.mol = td.mol
        self.base = td
        self.states = (0, 1)  # between which the NACV to be computed. 0 means ground state.
        self.atmlst = None
        self.de = None  # Known as CIS Force Matrix Element
        self.de_scaled = None # CIS derivative coupling without ETF
        self.de_etf = None  # CIS Force Matrix Element with ETF
        self.de_etf_scaled = None # Knwon as CIS derivative coupling with ETF

    _write      = rhf_grad_cpu.GradientsBase._write

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info("\n")
        log.info(
            "******** LR %s gradients for %s ********",
            self.base.__class__,
            self.base._scf.__class__,
        )
        log.info("cphf_conv_tol = %g", self.cphf_conv_tol)
        log.info("cphf_max_cycle = %d", self.cphf_max_cycle)
        log.info(f"States ID = {self.states}")
        log.info("\n")
        return self

    @lib.with_doc(get_nacv_ge.__doc__)
    def get_nacv_ge(self, x_yI, EI, singlet, atmlst=None, verbose=logger.INFO):
        return get_nacv_ge(self, x_yI, EI, singlet, atmlst, verbose)
    @lib.with_doc(get_nacv_ee.__doc__)
    def get_nacv_ee(self, x_yI, x_yJ, EI, EJ, singlet, atmlst=None, verbose=logger.INFO):
        return get_nacv_ee(self, x_yI, x_yJ, EI, EJ, singlet, atmlst, verbose)

    def kernel(self, xy_I=None, xy_J=None, E_I=None, E_J=None, singlet=None, atmlst=None):

        logger.warn(self, "This module is under development!!")

        if singlet is None:
            singlet = self.base.singlet
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        if xy_I is None or xy_J is None:
            states = sorted(self.states)
            nstates = len(self.base.e)
            I, J = states
            if I == J:
                raise ValueError("I and J should be different.")
            if I < 0 or J < 0:
                raise ValueError("Excited states ID should be non-negetive integers.")
            elif I > nstates or J > nstates:
                raise ValueError(f"Excited state exceeds the number of states {nstates}.")
            elif I == 0:
                logger.info(self, f"NACV between ground and excited state {J}.")
                xy_I = self.base.xy[J-1]
                E_I = self.base.e[J-1]
                self.de, self.de_scaled, self.de_etf, self.de_etf_scaled \
                    = self.get_nacv_ge(xy_I, E_I, singlet, atmlst, verbose=self.verbose)
                self._finalize()
            else:
                logger.info(self, f"NACV between excited state {I} and {J}.")
                xy_I = self.base.xy[I-1]
                E_I = self.base.e[I-1]
                xy_J = self.base.xy[J-1]
                E_J = self.base.e[J-1]
                self.de, self.de_scaled, self.de_etf, self.de_etf_scaled \
                    = self.get_nacv_ee(xy_I, xy_J, E_I, E_J, singlet, atmlst, verbose=self.verbose)
                self._finalize()
        return self.de, self.de_scaled, self.de_etf, self.de_etf_scaled
    
    def get_veff(self, mol=None, dm=None, j_factor=1.0, k_factor=1.0, omega=0.0, hermi=0, verbose=None):
        """
        Computes the first-order derivatives of the energy contributions from
        Veff per atom.

        NOTE: This function is incompatible to the one implemented in PySCF CPU version.
        In the CPU version, get_veff returns the first order derivatives of Veff matrix.
        """
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.base.make_rdm1()
        if omega == 0.0:
            vhfopt = self.base._scf._opt_gpu.get(None, None)
            return rhf_grad._jk_energy_per_atom(mol, dm, vhfopt, j_factor=j_factor, k_factor=k_factor, verbose=verbose)
        else:
            vhfopt = self.base._scf._opt_gpu.get(omega, None)
            with mol.with_range_coulomb(omega):
                return rhf_grad._jk_energy_per_atom(
                    mol, dm, vhfopt, j_factor=j_factor, k_factor=k_factor, verbose=verbose)

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(
                self,
                "--------- %s nonadiabatic derivative coupling for states %d and %d----------",
                self.base.__class__.__name__,
                self.states[0],
                self.states[1],
            )
            self._write(self.mol, self.de, self.atmlst)
            logger.note(
                self,
                "--------- %s nonadiabatic derivative coupling for states %d and %d after E scaled (divided by E)----------",
                self.base.__class__.__name__,
                self.states[0],
                self.states[1],
            )
            self._write(self.mol, self.de_scaled, self.atmlst)
            logger.note(
                self,
                "--------- %s nonadiabatic derivative coupling for states %d and %d with ETF----------",
                self.base.__class__.__name__,
                self.states[0],
                self.states[1],
            )
            self._write(self.mol, self.de_etf, self.atmlst)
            logger.note(
                self,
                "--------- %s nonadiabatic derivative coupling for states %d and %d with ETF after E scaled (divided by E)----------",
                self.base.__class__.__name__,
                self.states[0],
                self.states[1],
            )
            self._write(self.mol, self.de_etf_scaled, self.atmlst)
            logger.note(self, "----------------------------------------------")

    def solvent_response(self, dm):
        return 0.0

    to_gpu = lib.to_gpu

    def reset(self, mol):
        self.base.reset(mol)
        self.mol = mol
        return self

    def as_scanner(nacv_instance, states=None):
        if isinstance(nacv_instance, lib.GradScanner):
            return nacv_instance

        logger.info(nacv_instance, 'Create scanner for %s', nacv_instance.__class__)
        name = nacv_instance.__class__.__name__ + NAC_Scanner.__name_mixin__
        return lib.set_class(NAC_Scanner(nacv_instance, states),
                            (NAC_Scanner, nacv_instance.__class__), name)

    @classmethod
    def from_cpu(cls, method):
        td = method.base.to_gpu()
        out = cls(td)
        out.cphf_max_cycle = method.cphf_max_cycle
        out.cphf_conv_tol = method.cphf_conv_tol
        out.state = method.state
        out.de = method.de
        out.de_scaled = method.de_scaled
        out.de_etf = method.de_etf
        out.de_etf_scaled = method.de_etf_scaled
        return out


def check_phase_modified(mol0, mo_coeff0, mo1_reordered, xy0, xy1, nocc, s):
    nao = mol0.nao
    nvir = nao - nocc
    
    total_s_state = 0.0
    num_to_consider = 5

    top_indices0_flat = np.argsort(np.abs(xy0).flatten())[-num_to_consider:]
    top_indices1_flat = np.argsort(np.abs(xy1).flatten())[-num_to_consider:]

    for i in range(num_to_consider):
        idx_l = top_indices0_flat[i]
        idx_r = top_indices1_flat[i]

        idxo_l = idx_l // nvir
        idxv_l = idx_l % nvir
        idxo_r = idx_r // nvir
        idxv_r = idx_r % nvir

        mo_coeff0_tmp = mo_coeff0[:, :nocc].copy()
        mo_coeff1_tmp = mo1_reordered[:, :nocc].copy()
        
        mo_coeff0_tmp[:, idxo_l] = mo_coeff0[:, idxv_l + nocc]
        mo_coeff1_tmp[:, idxo_r] = mo1_reordered[:, idxv_r + nocc]
        
        s_mo = mo_coeff0_tmp.T @ s @ mo_coeff1_tmp
        
        s_state_contribution = cp.linalg.det(s_mo) \
            * xy0[idxo_l, idxv_l] * xy1[idxo_r, idxv_r] * 2

        total_s_state += s_state_contribution

    return total_s_state

class NAC_Scanner(lib.GradScanner):

    _keys = ['sign']

    def __init__(self, nac_instance, states=None):
        lib.GradScanner.__init__(self, nac_instance)
        self.sign = 1.0
        if states is not None:
            self.states = states
        else:
            self.states = nac_instance.states

    def __call__(self, mol_or_geom, states=None, **kwargs):
        mol0 = self.mol.copy()
        mo_coeff0 = self.base._scf.mo_coeff
        mo_occ = cp.asarray(self.base._scf.mo_occ)
        nao, nmo = mo_coeff0.shape
        nocc = int((mo_occ > 0).sum())
        nvir = nmo - nocc

        if isinstance(mol_or_geom, gto.MoleBase):
            assert mol_or_geom.__class__ == gto.Mole
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        self.reset(mol)

        if states is None:
            states = self.states
        else:
            self.states = states
        if isinstance(self.base, (tdscf.ris.TDDFT, tdscf.ris.TDA)):
            from gpu4pyscf.tdscf.ris import rescale_spin_free_amplitudes
            if states[0] != 0:
                xi0, yi0 = rescale_spin_free_amplitudes(self.base.xy, states[0]-1)
                xi0 = xi0.reshape(nocc, nvir)
                yi0 = yi0.reshape(nocc, nvir)
            xj0, yj0 = rescale_spin_free_amplitudes(self.base.xy, states[1]-1)
            xj0 = xj0.reshape(nocc, nvir)
            yj0 = yj0.reshape(nocc, nvir)
        else:
            if states[0] != 0:
                xi0, yi0 = self.base.xy[states[0]-1]
            xj0, yj0 = self.base.xy[states[1]-1]

        td_scanner = self.base

        assert td_scanner.device == 'gpu'
        assert self.device == 'gpu'
        td_scanner(mol)
        
        s = gto.intor_cross('int1e_ovlp', mol0, mol)
        mo_coeff = cp.asarray(self.base._scf.mo_coeff)
        s = cp.asarray(s)
        mo2_reordered, matching_indices, sign_array = match_and_reorder_mos(s, mo_coeff0, mo_coeff, threshold=0.4)
        if states[0] != 0:
            if isinstance(self.base, tdscf.ris.TDDFT) or isinstance(self.base, tdscf.ris.TDA):
                if self.base.xy[1] is not None:
                    xi1 = self.base.xy[0][states[0]-1]*np.sqrt(0.5)
                    yi1 = self.base.xy[1][states[0]-1]*np.sqrt(0.5)
                else:
                    xi1 = self.base.xy[0][states[0]-1]*np.sqrt(0.5)
                    yi1 = self.base.xy[0][states[0]-1]*0.0
                xi1 = xi1.reshape(nocc, nvir)
                yi1 = yi1.reshape(nocc, nvir)
            else:
                xi1, yi1 = self.base.xy[states[0]-1]
        if isinstance(self.base, tdscf.ris.TDDFT) or isinstance(self.base, tdscf.ris.TDA):
            if self.base.xy[1] is not None:
                xj1 = self.base.xy[0][states[1]-1]*np.sqrt(0.5)
                yj1 = self.base.xy[1][states[1]-1]*np.sqrt(0.5)
            else:
                xj1 = self.base.xy[0][states[1]-1]*np.sqrt(0.5)
                yj1 = self.base.xy[0][states[1]-1]*0.0
            xj1 = xj1.reshape(nocc, nvir)
            yj1 = yj1.reshape(nocc, nvir)
        else:
            xj1, yj1 = self.base.xy[states[1]-1]
        
        mo2_reordered = cp.asarray(mo2_reordered)
        mo_coeff0 = cp.asarray(mo_coeff0)

        # for the first state
        if states[0] != 0: # excited state
            sign = check_phase_modified(mol0, mo_coeff0, mo2_reordered, xi0, xi1, nocc, s)
            self.sign *= np.sign(sign)
        else: # ground state
            s_mo_ground = mo_coeff0[:, :nocc].T @ s @ mo2_reordered[:, :nocc]
            s_ground = cp.linalg.det(s_mo_ground)
            self.sign *= np.sign(s_ground)
        # for the second state
        sign = check_phase_modified(mol0, mo_coeff0, mo2_reordered, xj0, xj1, nocc, s)
        self.sign *= np.sign(sign)
        self.sign = float(self.sign)
        e_tot = self.e_tot
        
        de, de_scaled, de_etf, de_etf_scaled= self.kernel(**kwargs)
        de = de*self.sign
        de_scaled = de_scaled*self.sign
        de_etf = de_etf*self.sign
        de_etf_scaled = de_etf_scaled*self.sign
        return e_tot, de, de_scaled, de_etf, de_etf_scaled
