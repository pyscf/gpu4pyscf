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
from pyscf import lib
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
    if getattr(mf, 'with_solvent', None) is not None:
        raise NotImplementedError('With solvent is not supported yet')

    xI, yI = x_yI
    xI = cp.asarray(xI).reshape(nocc, nvir).T
    if not isinstance(yI, np.ndarray) and not isinstance(yI, cp.ndarray):
        yI = cp.zeros_like(xI)
    yI = cp.asarray(yI).reshape(nocc, nvir).T
    LI = xI-yI    # eq.(83) in Ref. [1]

    vresp = mf.gen_response(singlet=None, hermi=1)

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
    if getattr(mf, 'with_solvent', None) is not None:
        raise NotImplementedError('With solvent is not supported yet')

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
    wvo = reduce(cp.dot, (orbv.T, veff0doo, orbo)) * 2
    veffI = vj1I * 2 - vk1I
    veffI *= 0.5
    veff0mopI = reduce(cp.dot, (mo_coeff.T, veffI, mo_coeff))
    wvo -= contract("ki,ai->ak", veff0mopI[:nocc, :nocc], xpyJ) * 2  
    wvo += contract("ac,ai->ci", veff0mopI[nocc:, nocc:], xpyJ) * 2
    veffJ = vj1J * 2 - vk1J
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

    vresp = mf.gen_response(singlet=None, hermi=1)

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
    fock_matrix = mf.get_fock()
    fock_mo = reduce(cp.dot, (mo_coeff.T, fock_matrix, mo_coeff))
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
        "chkfile",
        "states",
        "atmlst",
        "de",
        "de_scaled",
        "de_etf",
        "de_etf_scaled"
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
        # log.info("chkfile = %s", self.chkfile)
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

    as_scanner = NotImplemented

    to_gpu = lib.to_gpu


