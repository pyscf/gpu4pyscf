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
from gpu4pyscf.lib import logger
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import tdrks, tdrks_ris
from gpu4pyscf.df import int3c2e
from gpu4pyscf.dft import rks
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.scf import cphf
from pyscf import __config__
from gpu4pyscf import tdscf
from gpu4pyscf.nac import tdrks as tdrks_nac
from pyscf.data.nist import HARTREE2EV
from gpu4pyscf.tdscf.ris import get_auxmol


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
    if td_nac.base.Ktrunc != 0.0:
        raise NotImplementedError('Ktrunc or frozen method is not supported yet')
    log = logger.new_logger(td_nac, verbose)
    theta = td_nac.base.theta
    J_fit = td_nac.base.J_fit
    K_fit = td_nac.base.K_fit
    if not singlet:
        raise ValueError('TDDFT ris only supports singlet state')
    mol = td_nac.mol
    mf = td_nac.base._scf
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

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    f1voI, f1ooIJ, vxc1, k1aoIJ = tdrks._contract_xc_kernel(td_nac, mf.xc, dmxpyI, dmzooIJ, True, 
        False, singlet)
    with_k = ni.libxc.is_hybrid_xc(mf.xc)

    auxmol_J = get_auxmol(mol=mol, theta=theta, fitting_basis=J_fit)
    if K_fit == J_fit and (omega == 0 or omega is None):
        log.info('K uese exactly same basis as J, and they share same set of Tensors')
        auxmol_K = auxmol_J
    else:
        log.info('K uese different basis as J')
        auxmol_K = get_auxmol(mol=mol, theta=theta, fitting_basis=K_fit)
    mf_J = rks.RKS(mol).density_fit()
    mf_J.with_df.auxmol = auxmol_J
    mf_K = rks.RKS(mol).density_fit()
    mf_K.with_df.auxmol = auxmol_K

    if with_k:
        vj0IJ, vk0IJ = mf.get_jk(mol, dmzooIJ, hermi=0)
        vj1I = mf_J.get_j(mol, (dmxpyI + dmxpyI.T), hermi=0)
        vk1I = mf_K.get_k(mol, (dmxpyI + dmxpyI.T), hermi=0)
        vk2I = mf_K.get_k(mol, (dmxmyI - dmxmyI.T), hermi=0)
        vj1J = mf_J.get_j(mol, (dmxpyJ + dmxpyJ.T), hermi=0)
        vk1J = mf_K.get_k(mol, (dmxpyJ + dmxpyJ.T), hermi=0)
        vk2J = mf_K.get_k(mol, (dmxmyJ - dmxmyJ.T), hermi=0)


        if not isinstance(vj0IJ, cp.ndarray):
            vj0IJ = cp.asarray(vj0IJ)
        if not isinstance(vk0IJ, cp.ndarray):
            vk0IJ = cp.asarray(vk0IJ)
        if not isinstance(vj1I, cp.ndarray):
            vj1I = cp.asarray(vj1I)
        if not isinstance(vk1I, cp.ndarray):
            vk1I = cp.asarray(vk1I)
        if not isinstance(vk2I, cp.ndarray):
            vk2I = cp.asarray(vk2I)
        if not isinstance(vj1J, cp.ndarray):
            vj1J = cp.asarray(vj1J)
        if not isinstance(vk1J, cp.ndarray):
            vk1J = cp.asarray(vk1J)
        if not isinstance(vk2J, cp.ndarray):
            vk2J = cp.asarray(vk2J)
        vk0IJ *= hyb
        vk1I *= hyb
        vk2I *= hyb
        vk1J *= hyb
        vk2J *= hyb
        if omega != 0:
            vk0IJ_omega = mf.get_k(mol, dmzooIJ, hermi=0, omega=omega)
            vk1I_omega = mf_K.get_k(mol, (dmxpyI + dmxmyI.T), hermi=0, omega=omega)
            vk2I_omega = mf_K.get_k(mol, (dmxmyI - dmxmyI.T), hermi=0, omega=omega)
            vk1J_omega = mf_K.get_k(mol, (dmxpyJ + dmxpyJ.T), hermi=0, omega=omega)
            vk2J_omega = mf_K.get_k(mol, (dmxmyJ - dmxmyJ.T), hermi=0, omega=omega)
            if not isinstance(vk0IJ, cp.ndarray):
                vk0IJ = cp.asarray(vk0IJ)
            if not isinstance(vk1I, cp.ndarray):
                vk1I = cp.asarray(vk1I)
            if not isinstance(vk2I, cp.ndarray):
                vk2I = cp.asarray(vk2I)
            if not isinstance(vk1J, cp.ndarray):
                vk1J = cp.asarray(vk1J)
            if not isinstance(vk2J, cp.ndarray):
                vk2J = cp.asarray(vk2J)
            vk0IJ += vk0IJ_omega * (alpha - hyb)
            vk1I += vk1I_omega * (alpha - hyb)
            vk2I += vk2I_omega * (alpha - hyb)
            vk1J += vk1J_omega * (alpha - hyb)
            vk2J += vk2J_omega * (alpha - hyb)

        veff0doo = vj0IJ * 2 - vk0IJ + f1ooIJ[0]
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
    else:
        vj0IJ = mf.get_j(mol, dmzooIJ, hermi=1)
        vj1I = mf_J.get_j(mol, (dmxpyI + dmxpyI.T), hermi=1)
        vj1J = mf_J.get_j(mol, (dmxpyJ + dmxpyJ.T), hermi=1)
        if not isinstance(vj0IJ, cp.ndarray):
            vj0IJ = cp.asarray(vj0IJ)
        if not isinstance(vj1I, cp.ndarray):
            vj1I = cp.asarray(vj1I)
        if not isinstance(vj1J, cp.ndarray):
            vj1J = cp.asarray(vj1J)

        veff0doo = vj0IJ * 2 + f1ooIJ[0]
        wvo = reduce(cp.dot, (orbv.T, veff0doo, orbo)) * 2
        veffI = vj1I
        veff0mopI = reduce(cp.dot, (mo_coeff.T, veffI, mo_coeff))
        wvo -= contract("ki,ai->ak", veff0mopI[:nocc, :nocc], xpyJ) * 2  
        wvo += contract("ac,ai->ci", veff0mopI[nocc:, nocc:], xpyJ) * 2
        veffJ = vj1J
        veff0mopJ = reduce(cp.dot, (mo_coeff.T, veffJ, mo_coeff))
        wvo -= contract("ki,ai->ak", veff0mopJ[:nocc, :nocc], xpyI) * 2  
        wvo += contract("ac,ai->ci", veff0mopJ[nocc:, nocc:], xpyI) * 2
        veff0momI = cp.zeros((nmo, nmo))
        veff0momJ = cp.zeros((nmo, nmo))

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

    im0[nocc:, :nocc] = TFvo*2.0
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
    
    j_factor = 1.0
    k_factor = 0.0
    if with_k:
        k_factor = hyb
    
    extra_force = cp.zeros((len(atmlst), 3))
    dvhf_all = 0
    dvhf = td_nac.get_veff(mol, dmz1doo + oo0, j_factor=j_factor, k_factor=k_factor) 
    for k, ia in enumerate(atmlst):
        extra_force[k] += cp.asarray(mf_grad.extra_force(ia, locals()))
    dvhf_all += dvhf
    # minus in the next TWO terms is due to only <g^{(\xi)};{D,P_{IJ}}> is needed, 
    # thus minus the contribution from same DM ({D,D}, {P,P}).
    dvhf = td_nac.get_veff(mol, dmz1doo, j_factor=j_factor, k_factor=k_factor)
    for k, ia in enumerate(atmlst):
        extra_force[k] -= cp.asarray(mf_grad.extra_force(ia, locals()))
    dvhf_all -= dvhf
    dvhf = td_nac.get_veff(mol, oo0, j_factor=j_factor, k_factor=k_factor)
    for k, ia in enumerate(atmlst):
        extra_force[k] -= cp.asarray(mf_grad.extra_force(ia, locals()))
    dvhf_all -= dvhf
    dvhf = tdrks_ris.get_veff_ris(mf_J, mf_K, mol, (dmxpyI + dmxpyI.T + dmxpyJ + dmxpyJ.T), j_factor=j_factor, k_factor=k_factor)
    for k, ia in enumerate(atmlst):
        extra_force[k] += cp.asarray(tdrks_ris.get_extra_force(ia, locals()))
    dvhf_all += dvhf
    # minus in the next TWO terms is due to only <g^{(\xi)};{R_I^S, R_J^S}> is needed, 
    # thus minus the contribution from same DM ({R_I^S,R_I^S} and {R_J^S,R_J^S}).
    dvhf = tdrks_ris.get_veff_ris(mf_J, mf_K, mol, (dmxpyI + dmxpyI.T), j_factor=j_factor, k_factor=k_factor)
    for k, ia in enumerate(atmlst):
        extra_force[k] -= cp.asarray(tdrks_ris.get_extra_force(ia, locals()))
    dvhf_all -= dvhf # NOTE: minus
    dvhf = tdrks_ris.get_veff_ris(mf_J, mf_K, mol, (dmxpyJ + dmxpyJ.T), j_factor=j_factor, k_factor=k_factor)
    for k, ia in enumerate(atmlst):
        extra_force[k] -= cp.asarray(tdrks_ris.get_extra_force(ia, locals()))
    dvhf_all -= dvhf
    dvhf = tdrks_ris.get_veff_ris(mf_J, mf_K, mol, (dmxmyI - dmxmyI.T + dmxmyJ - dmxmyJ.T), j_factor=0.0, k_factor=k_factor, hermi=2)
    for k, ia in enumerate(atmlst):
        extra_force[k] += cp.asarray(tdrks_ris.get_extra_force(ia, locals()))
    dvhf_all += dvhf
    dvhf = tdrks_ris.get_veff_ris(mf_J, mf_K, mol, (dmxmyI - dmxmyI.T), j_factor=0.0, k_factor=k_factor, hermi=2)
    for k, ia in enumerate(atmlst):
        extra_force[k] -= cp.asarray(tdrks_ris.get_extra_force(ia, locals()))
    dvhf_all -= dvhf
    dvhf = tdrks_ris.get_veff_ris(mf_J, mf_K, mol, (dmxmyJ - dmxmyJ.T), j_factor=0.0, k_factor=k_factor, hermi=2)
    for k, ia in enumerate(atmlst):
        extra_force[k] -= cp.asarray(tdrks_ris.get_extra_force(ia, locals()))
    dvhf_all -= dvhf

    if with_k and omega != 0:
        j_factor = 0.0
        k_factor = alpha - hyb
        dvhf = td_nac.get_veff(mol, dmz1doo + oo0, j_factor=j_factor, k_factor=k_factor, omega=omega) 
        for k, ia in enumerate(atmlst):
            extra_force[k] += cp.asarray(mf_grad.extra_force(ia, locals()))
        dvhf_all += dvhf
        # minus in the next TWO terms is due to only <g^{(\xi)};{D,P_{IJ}}> is needed, 
        # thus minus the contribution from same DM ({D,D}, {P,P}).
        dvhf = td_nac.get_veff(mol, dmz1doo, j_factor=j_factor, k_factor=k_factor, omega=omega)
        for k, ia in enumerate(atmlst):
            extra_force[k] -= cp.asarray(mf_grad.extra_force(ia, locals()))
        dvhf_all -= dvhf
        dvhf = td_nac.get_veff(mol, oo0, j_factor=j_factor, k_factor=k_factor, omega=omega)
        for k, ia in enumerate(atmlst):
            extra_force[k] -= cp.asarray(mf_grad.extra_force(ia, locals()))
        dvhf_all -= dvhf

        dvhf = tdrks_ris.get_veff_ris(mf_J, mf_K, mol, (dmxpyI + dmxpyI.T + dmxpyJ + dmxpyJ.T), j_factor=j_factor, k_factor=k_factor, omega=omega)
        for k, ia in enumerate(atmlst):
            extra_force[k] += cp.asarray(tdrks_ris.get_extra_force(ia, locals()))
        dvhf_all += dvhf
        # minus in the next TWO terms is due to only <g^{(\xi)};{R_I^S, R_J^S}> is needed, 
        # thus minus the contribution from same DM ({R_I^S,R_I^S} and {R_J^S,R_J^S}).
        dvhf = tdrks_ris.get_veff_ris(mf_J, mf_K, mol, (dmxpyI + dmxpyI.T), j_factor=j_factor, k_factor=k_factor, omega=omega)
        for k, ia in enumerate(atmlst):
            extra_force[k] -= cp.asarray(tdrks_ris.get_extra_force(ia, locals()))
        dvhf_all -= dvhf # NOTE: minus
        dvhf = tdrks_ris.get_veff_ris(mf_J, mf_K, mol, (dmxpyJ + dmxpyJ.T), j_factor=j_factor, k_factor=k_factor, omega=omega)
        for k, ia in enumerate(atmlst):
            extra_force[k] -= cp.asarray(tdrks_ris.get_extra_force(ia, locals()))
        dvhf_all -= dvhf
        dvhf = tdrks_ris.get_veff_ris(mf_J, mf_K, mol, (dmxmyI - dmxmyI.T + dmxmyJ - dmxmyJ.T), j_factor=0.0, k_factor=k_factor, omega=omega, hermi=2)
        for k, ia in enumerate(atmlst):
            extra_force[k] += cp.asarray(tdrks_ris.get_extra_force(ia, locals()))
        dvhf_all += dvhf
        dvhf = tdrks_ris.get_veff_ris(mf_J, mf_K, mol, (dmxmyI - dmxmyI.T), j_factor=0.0, k_factor=k_factor, omega=omega, hermi=2)
        for k, ia in enumerate(atmlst):
            extra_force[k] -= cp.asarray(tdrks_ris.get_extra_force(ia, locals()))
        dvhf_all -= dvhf
        dvhf = tdrks_ris.get_veff_ris(mf_J, mf_K, mol, (dmxmyJ - dmxmyJ.T), j_factor=0.0, k_factor=k_factor, omega=omega, hermi=2)
        for k, ia in enumerate(atmlst):
            extra_force[k] -= cp.asarray(tdrks_ris.get_extra_force(ia, locals()))
        dvhf_all -= dvhf

    fxcz1 = tdrks._contract_xc_kernel(td_nac, mf.xc, z1aoS, None, False, False, True)[0]
    veff1_0 = vxc1[1:]          # from <g^{XC[1](\xi)};P_{IJ}> in Eq. (64) in Ref.[1]
    # First two terms from <g^{XC[1](\xi)};P_{IJ}> in Eq. (64) in Ref.[1]
    # Final term from <g^{XC[2](\xi)};\{R^{S}_{I},R^{S}_{J}\}> in Eq. (64) in Ref.[1]
    veff1_1 = f1ooIJ[1:] + fxcz1[1:]

    delec = dh_td*2 - ds
    aoslices = mol.aoslice_by_atom()
    delec = cp.asarray([cp.sum(delec[:, p0:p1], axis=1) for p0, p1 in aoslices[:, 2:]])
    dveff1_0 = cp.asarray(
        [contract("xpq,pq->x", veff1_0[:, p0:p1], dmz1doo[p0:p1]) for p0, p1 in aoslices[:, 2:]])
    dveff1_0 += cp.asarray([
            contract("xpq,pq->x", veff1_0[:, p0:p1].transpose(0, 2, 1), dmz1doo[:, p0:p1],)
            for p0, p1 in aoslices[:, 2:]])
    dveff1_1 = cp.asarray([contract("xpq,pq->x", veff1_1[:, p0:p1], oo0[p0:p1]) for p0, p1 in aoslices[:, 2:]])

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
    
    de = 2.0 * dvhf_all + extra_force + dh1e_td + delec + dveff1_0 + dveff1_1 # Eq. (64) in Ref. [1]

    de_etf = de + dsxy_etf
    de += dsxy

    de = de.get()
    de_etf = de_etf.get()
    return de, de/(EJ - EI), de_etf, de_etf/(EJ - EI)


class NAC(tdrks_nac.NAC):

    @lib.with_doc(get_nacv_ee.__doc__)
    def get_nacv_ee(self, x_yI, x_yJ, EI, EJ, singlet, atmlst=None, verbose=logger.INFO):
        return get_nacv_ee(self, x_yI, x_yJ, EI, EJ, singlet, atmlst, verbose)

    as_scanner = NotImplemented

    def kernel(self, xy_I=None, xy_J=None, E_I=None, E_J=None, singlet=None, atmlst=None):

        logger.warn(self, "This module is under development!!")
        if self.base.Ktrunc != 0.0:
            raise NotImplementedError('Ktrunc or frozen method is not supported yet')
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
            nstates = len(self.base.energies)
            I, J = states
            if I == J:
                raise ValueError("I and J should be different.")
            if I < 0 or J < 0:
                raise ValueError("Excited states ID should be non-negetive integers.")
            elif I > nstates or J > nstates:
                raise ValueError(f"Excited state exceeds the number of states {nstates}.")
            elif I == 0:
                logger.info(self, f"NACV between ground and excited state {J}.")
                if self.base.xy[1] is not None:
                    xy_I = (self.base.xy[0][J-1]*np.sqrt(0.5), self.base.xy[1][J-1]*np.sqrt(0.5))
                else:
                    xy_I = (self.base.xy[0][J-1]*np.sqrt(0.5), self.base.xy[0][J-1]*0.0)
                E_I = self.base.energies[J-1]/HARTREE2EV
                E_I = float(E_I)
                self.de, self.de_scaled, self.de_etf, self.de_etf_scaled \
                    = self.get_nacv_ge(xy_I, E_I, singlet, atmlst, verbose=self.verbose)
                self._finalize()
            else:
                logger.info(self, f"NACV between excited state {I} and {J}.")
                if self.base.xy[1] is not None:
                    xy_I = (self.base.xy[0][I-1]*np.sqrt(0.5), self.base.xy[1][I-1]*np.sqrt(0.5))
                else:
                    xy_I = (self.base.xy[0][I-1]*np.sqrt(0.5), self.base.xy[0][I-1]*0.0)
                E_I = self.base.energies[I-1]/HARTREE2EV
                E_I = float(E_I)
                if self.base.xy[1] is not None:
                    xy_J = (self.base.xy[0][J-1]*np.sqrt(0.5), self.base.xy[1][J-1]*np.sqrt(0.5))
                else:
                    xy_J = (self.base.xy[0][J-1]*np.sqrt(0.5), self.base.xy[0][J-1]*0.0)
                E_J = self.base.energies[J-1]/HARTREE2EV
                E_J = float(E_J)
                self.de, self.de_scaled, self.de_etf, self.de_etf_scaled \
                    = self.get_nacv_ee(xy_I, xy_J, E_I, E_J, singlet, atmlst, verbose=self.verbose)
                self._finalize()
        return self.de, self.de_scaled, self.de_etf, self.de_etf_scaled


