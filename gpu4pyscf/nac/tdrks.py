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
from pyscf import __config__
from gpu4pyscf.lib import logger
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import tdrks
from gpu4pyscf.df.grad import tdrhf as tdrhf_grad_df
from gpu4pyscf.df import int3c2e
from gpu4pyscf.df.df_jk import (
    _tag_factorize_dm, _DFHF, _make_factorized_dm, _aggregate_dm_factor_l,
    _transpose_dm)
from gpu4pyscf.lib.cupy_helper import contract, tag_array
from gpu4pyscf.lib import utils
from gpu4pyscf.scf import cphf
from gpu4pyscf import tdscf
from gpu4pyscf.nac import tdrhf


def get_nacv_ge(td_nac, x_yI, EI, singlet=True, atmlst=None, verbose=logger.INFO):
    """
    Calculate non-adiabatic coupling vectors between ground and excited states.
    Now, only supports for singlet states.
    Only supports for ground-excited states.
    Ref:
    [1] 10.1063/1.4903986 main reference
    [2] 10.1021/acs.accounts.1c00312
    [3] 10.1063/1.4885817

    Args:
        td_nac (gpu4pyscf.tdscf.rhf.TDA): Non-adiabatic coupling object for TDDFT or TDHF.
        x_yI (tuple): (xI, yI), xI and YI are the eigenvectors corresponding to the excitation and de-excitation.
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
    log = logger.new_logger(td_nac, verbose)
    time0 = logger.init_timer(td_nac)

    mol = td_nac.mol
    mf = td_nac.base._scf
    mf_grad = mf.nuc_grad_method()
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    nao, nmo = mo_coeff.shape
    orbo = mo_coeff[:, mo_occ > 0]
    orbv = mo_coeff[:, mo_occ ==0]
    nocc = orbo.shape[1]
    nvir = orbv.shape[1]

    xI, yI = x_yI
    xI = cp.asarray(xI).reshape(nocc, nvir).T
    if not isinstance(yI, np.ndarray) and not isinstance(yI, cp.ndarray):
        yI = cp.zeros_like(xI)
    yI = cp.asarray(yI).reshape(nocc, nvir).T
    LI = xI-yI    # eq.(83) in Ref. [1]

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    with_k = ni.libxc.is_hybrid_xc(mf.xc)
    if td_nac.ris_zvector_solver:
        log.note('Use ris-approximated Z-vector solver')
        if isinstance(td_nac.base, tdscf.ris.TDDFT) or isinstance(td_nac.base, tdscf.ris.TDA):
            from gpu4pyscf.dft import rks
            from gpu4pyscf.tdscf.ris import get_auxmol
            from gpu4pyscf.grad import tdrks_ris

            theta = td_nac.base.theta
            J_fit = td_nac.base.J_fit
            K_fit = td_nac.base.K_fit
        else:
            from gpu4pyscf.dft import rks
            from gpu4pyscf.tdscf.ris import get_auxmol
            from gpu4pyscf.grad import tdrks_ris
            from gpu4pyscf.tdscf.ris import RisBase
            tdris = RisBase(mf)
            theta = tdris.theta
            J_fit = tdris.J_fit
            K_fit = tdris.K_fit
            assert getattr(mf, 'with_solvent', None) is None, 'with_solvent is not supported for ris-approximated Z-vector solver'
        auxmol_J = get_auxmol(mol=mol, theta=theta, fitting_basis=J_fit)
        if K_fit == J_fit and (omega == 0 or omega is None):
            auxmol_K = auxmol_J
        else:
            auxmol_K = get_auxmol(mol=mol, theta=theta, fitting_basis=K_fit)
        mf_J = rks.RKS(mol).density_fit()
        mf_J.with_df.auxmol = auxmol_J
        mf_K = rks.RKS(mol).density_fit()
        mf_K.with_df.auxmol = auxmol_K
        vresp = tdrks_ris.gen_response_ris(mf, mf_J, mf_K, mo_coeff, mo_occ, singlet=None, hermi=1)
    else:
        log.note('Use standard Z-vector solver')
        if isinstance(td_nac.base, tdscf.ris.TDDFT) or isinstance(td_nac.base, tdscf.ris.TDA):
            vresp = td_nac.base._scf.gen_response(singlet=None, hermi=1)
        else:
            vresp = td_nac.base.gen_response(singlet=None, hermi=1)

    t_debug_1 = log.timer_silent(*time0)[2]
    def fvind(x):
        x = orbv.dot(x.reshape(nvir,nocc)) * 2 # *2 for double occupency
        dm = _make_factorized_dm(x, orbo, symmetrize=1)
        v1ao = vresp(dm)
        return reduce(cp.dot, (orbv.T, v1ao, orbo)).ravel()

    z1 = cphf.solve(
        fvind,
        mo_energy,
        mo_occ,
        -LI*1.0*EI, # only one spin, negative in cphf
        max_cycle=td_nac.cphf_max_cycle,
        tol=td_nac.cphf_conv_tol)[0] # eq.(83) in Ref. [1]
    t_debug_2 = log.timer_silent(*time0)[2]
    z1 = z1.reshape(nvir, nocc)
    z1ao = _make_factorized_dm(orbv.dot(z1), orbo, symmetrize=1)
    GZS = vresp(z1ao)
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
    dmz1doo = z1ao
    td_nac._dmz1doo = dmz1doo
    oo0 = _make_factorized_dm(orbo*2, orbo, symmetrize=0)
    t_debug_3 = log.timer_silent(*time0)[2]

    h1 = cp.asarray(mf_grad.get_hcore(mol))  # without 1/r like terms
    s1 = cp.asarray(mf_grad.get_ovlp(mol))
    dh_td = rhf_grad.contract_h1e_dm(mol, h1, dmz1doo, hermi=1)
    ds = rhf_grad.contract_h1e_dm(mol, s1, W, hermi=0)

    dh1e_td = int3c2e.get_dh1e(mol, dmz1doo)  # 1/r like terms
    if len(mol._ecpbas) > 0:
        dh1e_td += rhf_grad.get_dh1e_ecp(mol, dmz1doo)  # 1/r like terms
    t_debug_4 = log.timer_silent(*time0)[2]
    if mol._pseudo:
        raise NotImplementedError("Pseudopotential gradient not supported for molecular system yet")

    j_factor = [1.]
    k_factor = None
    if with_k:
        k_factor = [hyb]
    ejk = td_nac.jk_energies_per_atom(
        [[dmz1doo, oo0]], j_factor, k_factor, sum_results=True) * 2

    if with_k and omega != 0:
        j_factor = None
        beta = alpha - hyb
        k_factor = [beta]
        ejk += td_nac.jk_energies_per_atom(
            [[dmz1doo, oo0]], j_factor, k_factor, omega=omega, sum_results=True) * 2
    t_debug_5 = log.timer_silent(*time0)[2]
    f1ooP, _, vxc1, _ = tdrks._contract_xc_kernel(td_nac, mf.xc, dmz1doo, dmz1doo, True, False, singlet)
    veff1_0 = vxc1[1:]
    veff1_1 = f1ooP[1:]
    t_debug_6 = log.timer_silent(*time0)[2]
    de = dh_td - ds + ejk

    xIao = reduce(cp.dot, (orbo, xI.T, orbv.T))
    yIao = reduce(cp.dot, (orbv, yI, orbo.T))
    dsxy  = tdrhf._contract_h1e_dm_asymmetric(mol, s1, xIao*EI) * 2
    dsxy += tdrhf._contract_h1e_dm_asymmetric(mol, s1, yIao*EI) * 2
    dsxy_etf  = rhf_grad.contract_h1e_dm(mol, s1, xIao*EI, hermi=0)
    dsxy_etf += rhf_grad.contract_h1e_dm(mol, s1, yIao*EI, hermi=0)
    dveff1_0 = rhf_grad.contract_h1e_dm(mol, veff1_0, dmz1doo, hermi=0)
    dveff1_1 = rhf_grad.contract_h1e_dm(mol, veff1_1, oo0, hermi=1) * .5
    de += cp.asnumpy(dh1e_td) + dveff1_0 + dveff1_1
    de_etf = de + dsxy_etf
    de += dsxy
    t_debug_7 = log.timer_silent(*time0)[2]
    if log.verbose >= logger.DEBUG:
        time_list = [0, t_debug_1, t_debug_2, t_debug_3, t_debug_4, t_debug_5, t_debug_6, t_debug_7]
        time_list = [time_list[i+1] - time_list[i] for i in range(len(time_list)-1)]
        for i, t in enumerate(time_list):
            logger.note(td_nac, f"Time for step {i}: {t*1e-3:.6f}s")
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
    log = logger.new_logger(td_nac, verbose)
    time0 = logger.init_timer(td_nac)

    mol = td_nac.mol
    mf = td_nac.base._scf
    mf_grad = mf.nuc_grad_method()
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    nao, nmo = mo_coeff.shape
    orbo = mo_coeff[:, mo_occ > 0]
    orbv = mo_coeff[:, mo_occ ==0]
    nocc = orbo.shape[1]
    nvir = orbv.shape[1]

    xI, yI = x_yI
    xJ, yJ = x_yJ

    is_tda = False
    xI = cp.asarray(xI).reshape(nocc, nvir).T
    if not isinstance(yI, np.ndarray) and not isinstance(yI, cp.ndarray):
        yI = cp.zeros_like(xI)
        is_tda = True
    yI = cp.asarray(yI).reshape(nocc, nvir).T
    xJ = cp.asarray(xJ).reshape(nocc, nvir).T
    if not isinstance(yJ, np.ndarray) and not isinstance(yJ, cp.ndarray):
        yJ = cp.zeros_like(xJ)
    yJ = cp.asarray(yJ).reshape(nocc, nvir).T

    xpyI = (xI + yI)
    xmyI = (xI - yI)
    xpyJ = (xJ + yJ)
    xmyJ = (xJ - yJ)
    dmxpyI = _make_factorized_dm(orbv.dot(xpyI), orbo, symmetrize=0)
    dmxpyJ = _make_factorized_dm(orbv.dot(xpyJ), orbo, symmetrize=0)
    dmxmyI = _make_factorized_dm(orbv.dot(xmyI), orbo, symmetrize=0)
    dmxmyJ = _make_factorized_dm(orbv.dot(xmyJ), orbo, symmetrize=0)
    td_nac._dmxpyI = dmxpyI
    td_nac._dmxpyJ = dmxpyJ

    rIJoo =-contract('ai,aj->ij', xJ, xI) - contract('ai,aj->ij', yI, yJ)
    rIJvv = contract('ai,bi->ab', xI, xJ) + contract('ai,bi->ab', yJ, yI)
    TIJoo = (rIJoo + rIJoo.T) * 0.5
    TIJvv = (rIJvv + rIJvv.T) * 0.5
    dmzooIJ = reduce(cp.dot, (orbo, TIJoo, orbo.T)) * 2
    dmzooIJ += reduce(cp.dot, (orbv, TIJvv, orbv.T)) * 2
    t_debug_1 = log.timer_silent(*time0)[2]
    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    f1voI, f1ooIJ, vxc1, k1aoIJ = tdrks._contract_xc_kernel(td_nac, mf.xc, dmxpyI, dmzooIJ, True,
        True, singlet, with_nac=True, dmvo_2=dmxpyJ)
    f1voJ, _, _, _ = tdrks._contract_xc_kernel(td_nac, mf.xc, dmxpyJ, None, False, False, singlet)
    with_k = ni.libxc.is_hybrid_xc(mf.xc)
    t_debug_2 = log.timer_silent(*time0)[2]
    if with_k:
        if not isinstance(mf, _DFHF):
            dm = cp.stack([dmzooIJ,
                           dmxpyI + dmxpyI.T, dmxpyJ + dmxpyJ.T,
                           dmxmyI - dmxmyI.T, dmxmyJ - dmxmyJ.T])
            vj, vk = mf.get_jk(mol, dm[:3], hermi=1)
            vk *= hyb
            vj0IJ, vj1I, vj1J = vj
            vk0IJ, vk1I, vk1J = vk
            vj, vk = mf.get_jk(mol, dm[3:], hermi=2)
            vk *= hyb
            vk2I, vk2J = vk
        else:
            dmzooIJ = _tag_factorize_dm(dmzooIJ, hermi=1)
            vj0IJ, vk0IJ = mf.get_jk(mol, dmzooIJ, hermi=1)
            vk0IJ *= hyb
            if is_tda:
                dm = _aggregate_dm_factor_l([dmxpyI, dmxpyJ])
                vj, vk = mf.get_jk(mol, dm, hermi=0)
                vk *= hyb
                vk2I = vk[0] - vk[0].T
                vk2J = vk[1] - vk[1].T
            else:
                dm = _aggregate_dm_factor_l([dmxpyI, dmxpyJ, dmxmyI, dmxmyJ])
                vj, vk = mf.get_jk(mol, dm, hermi=0)
                vk *= hyb
                vk2I = vk[2] - vk[2].T
                vk2J = vk[3] - vk[3].T
            vj1I = vj[0] * 2
            vj1J = vj[1] * 2
            vk1I = vk[0] + vk[0].T
            vk1J = vk[1] + vk[1].T

        if omega != 0:
            beta = alpha - hyb
            if not isinstance(mf, _DFHF):
                vk = mf.get_k(mol, dm[:3], hermi=1, omega=omega)
                vk *= beta
                vk0IJ += vk[0]
                vk1I += vk[1]
                vk1J += vk[2]
                vk = mf.get_k(mol, dm[3:], hermi=2, omega=omega)
                vk *= beta
                vk2I += vk[0]
                vk2J += vk[1]
            else:
                vk0IJ += mf.get_k(mol, dmzooIJ, hermi=1, omega=omega) * beta
                if is_tda:
                    vk = mf.get_k(mol, dm, hermi=0, omega=omega)
                    vk *= beta
                    vk2I += vk[0] - vk[0].T
                    vk2J += vk[1] - vk[1].T
                else:
                    vk = mf.get_k(mol, dm, hermi=0, omega=omega)
                    vk *= beta
                    vk2I += vk[2] - vk[2].T
                    vk2J += vk[3] - vk[3].T
                vk1I += vk[0] + vk[0].T
                vk1J += vk[1] + vk[1].T
        dm = vj = vk = None
        dmzooIJ = dmzooIJ.view(cp.ndarray)

        veff0doo = vj0IJ * 2 - vk0IJ + f1ooIJ[0] + k1aoIJ[0] * 2
        veff0doo += td_nac.solvent_response(dmzooIJ)
        wvo = reduce(cp.dot, (orbv.T, veff0doo, orbo)) * 2
        veffI = vj1I * 2 - vk1I + f1voI[0] * 2
        veffI += td_nac.solvent_response(dmxpyI + dmxpyI.T)
        veffI *= 0.5
        veff0mopI = reduce(cp.dot, (mo_coeff.T, veffI, mo_coeff))
        wvo -= contract("ki,ai->ak", veff0mopI[:nocc, :nocc], xpyJ) * 2
        wvo += contract("ac,ai->ci", veff0mopI[nocc:, nocc:], xpyJ) * 2
        veffJ = vj1J * 2 - vk1J + f1voJ[0] * 2
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
    else:
        vj0IJ, vj1I, vj1J = mf.get_j(
            mol, cp.stack([dmzooIJ, dmxpyI+dmxpyI.T, dmxpyJ + dmxpyJ.T]), hermi=1)

        veff0doo = vj0IJ * 2 + f1ooIJ[0] + k1aoIJ[0] * 2
        veff0doo += td_nac.solvent_response(dmzooIJ)
        wvo = reduce(cp.dot, (orbv.T, veff0doo, orbo)) * 2
        veffI = vj1I * 2 + f1voI[0] * 2
        veffI += td_nac.solvent_response(dmxpyI + dmxpyI.T)
        veffI *= 0.5
        veff0mopI = reduce(cp.dot, (mo_coeff.T, veffI, mo_coeff))
        wvo -= contract("ki,ai->ak", veff0mopI[:nocc, :nocc], xpyJ) * 2
        wvo += contract("ac,ai->ci", veff0mopI[nocc:, nocc:], xpyJ) * 2
        veffJ = vj1J * 2 + f1voJ[0] * 2
        veffJ += td_nac.solvent_response(dmxpyJ + dmxpyJ.T)
        veffJ *= 0.5
        veff0mopJ = reduce(cp.dot, (mo_coeff.T, veffJ, mo_coeff))
        wvo -= contract("ki,ai->ak", veff0mopJ[:nocc, :nocc], xpyI) * 2
        wvo += contract("ac,ai->ci", veff0mopJ[nocc:, nocc:], xpyI) * 2
        veff0momI = cp.zeros((nmo, nmo))
        veff0momJ = cp.zeros((nmo, nmo))

    if td_nac.ris_zvector_solver:
        log.note('Use ris-approximated Z-vector solver')
        if isinstance(td_nac.base, tdscf.ris.TDDFT) or isinstance(td_nac.base, tdscf.ris.TDA):
            from gpu4pyscf.dft import rks
            from gpu4pyscf.tdscf.ris import get_auxmol
            from gpu4pyscf.grad import tdrks_ris

            theta = td_nac.base.theta
            J_fit = td_nac.base.J_fit
            K_fit = td_nac.base.K_fit
        else:
            from gpu4pyscf.dft import rks
            from gpu4pyscf.tdscf.ris import get_auxmol
            from gpu4pyscf.grad import tdrks_ris
            from gpu4pyscf.tdscf.ris import RisBase
            tdris = RisBase(mf)
            theta = tdris.theta
            J_fit = tdris.J_fit
            K_fit = tdris.K_fit
            assert getattr(mf, 'with_solvent', None) is None, 'with_solvent is not supported for ris-approximated Z-vector solver'
        auxmol_J = get_auxmol(mol=mol, theta=theta, fitting_basis=J_fit)
        if K_fit == J_fit and (omega == 0 or omega is None):
            auxmol_K = auxmol_J
        else:
            auxmol_K = get_auxmol(mol=mol, theta=theta, fitting_basis=K_fit)
        mf_J = rks.RKS(mol).density_fit()
        mf_J.with_df.auxmol = auxmol_J
        mf_K = rks.RKS(mol).density_fit()
        mf_K.with_df.auxmol = auxmol_K
        vresp = tdrks_ris.gen_response_ris(mf, mf_J, mf_K, mo_coeff, mo_occ, singlet=None, hermi=1)
    else:
        log.note('Use standard Z-vector solver')
        if isinstance(td_nac.base, tdscf.ris.TDDFT) or isinstance(td_nac.base, tdscf.ris.TDA):
            vresp = td_nac.base._scf.gen_response(singlet=None, hermi=1)
        else:
            vresp = td_nac.base.gen_response(singlet=None, hermi=1)
    
    t_debug_3 = log.timer_silent(*time0)[2]
    def fvind(x):
        x = orbv.dot(x.reshape(nvir,nocc)) * 2 # *2 for double occupency
        dm = _make_factorized_dm(x, orbo, symmetrize=1)
        v1ao = vresp(dm)
        return reduce(cp.dot, (orbv.T, v1ao, orbo)).ravel()

    z1 = cphf.solve(
        fvind,
        mo_energy,
        mo_occ,
        wvo,
        max_cycle=td_nac.cphf_max_cycle,
        tol=td_nac.cphf_conv_tol)[0] # eq.(80) in Ref. [1]
    t_debug_4 = log.timer_silent(*time0)[2]
    z1 /= EJ-EI # only one spin, negative in cphf

    z1ao = _make_factorized_dm(orbv.dot(z1), orbo, symmetrize=1)
    veff = vresp(z1ao)
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
    z1aoS = z1ao * ((EJ - EI)/2)
    dmz1doo = z1aoS + dmzooIJ  # P
    td_nac._dmz1doo = dmz1doo
    oo0 = _make_factorized_dm(orbo*2, orbo, symmetrize=0)  # D, the ground state density matrix
    t_debug_5 = log.timer_silent(*time0)[2]

    h1 = cp.asarray(mf_grad.get_hcore(mol))  # without 1/r like terms
    s1 = cp.asarray(mf_grad.get_ovlp(mol))
    dh_td = rhf_grad.contract_h1e_dm(mol, h1, dmz1doo, hermi=1)
    ds = rhf_grad.contract_h1e_dm(mol, s1, im0, hermi=0)

    dh1e_td = int3c2e.get_dh1e(mol, dmz1doo)  # 1/r like terms
    if len(mol._ecpbas) > 0:
        dh1e_td += rhf_grad.get_dh1e_ecp(mol, dmz1doo)  # 1/r like terms
    t_debug_6 = log.timer_silent(*time0)[2]
    if mol._pseudo:
        raise NotImplementedError("Pseudopotential gradient not supported for molecular system yet")

    cp.get_default_memory_pool().free_all_blocks()
    k_factor = None
    if not is_tda:
        j_factor = [1., 2.,  0.]
        if with_k:
            k_factor = np.array([1, 2., -2.])
        dms = [[_tag_factorize_dm(dmz1doo, hermi=1), oo0],
               [dmxpyI, dmxpyJ + dmxpyJ.T],
               [dmxmyI, dmxmyJ - dmxmyJ.T]]
    else:
        j_factor = [1., 4.]
        if with_k:
            k_factor = np.array([1., 4.])
        dms = [[_tag_factorize_dm(dmz1doo, hermi=1), oo0],
               [dmxpyI, _transpose_dm(dmxpyJ)]]

    if with_k:
        ejk = td_nac.jk_energies_per_atom(
            dms, j_factor, k_factor*hyb, sum_results=True) * 2
    else:
        ejk = td_nac.jk_energies_per_atom(
            dms, j_factor, None, sum_results=True) * 2

    if with_k and omega != 0:
        j_factor = None
        beta = alpha - hyb
        ejk += td_nac.jk_energies_per_atom(
            dms, j_factor, k_factor*beta, omega=omega, sum_results=True) * 2

    t_debug_7 = log.timer_silent(*time0)[2]
    fxcz1 = tdrks._contract_xc_kernel(td_nac, mf.xc, z1aoS, None, False, False, True)[0]
    veff1_0 = vxc1[1:]          # from <g^{XC[1](\xi)};P_{IJ}> in Eq. (64) in Ref.[1]
    # First two terms from <g^{XC[1](\xi)};P_{IJ}> in Eq. (64) in Ref.[1]
    # Final term from <g^{XC[2](\xi)};\{R^{S}_{I},R^{S}_{J}\}> in Eq. (64) in Ref.[1]
    veff1_1 = f1ooIJ[1:] + fxcz1[1:] + k1aoIJ[1:] * 2
    veff1_2I = f1voI[1:] # term from <g^{XC[2](\xi)};\{R^{S}_{I},R^{S}_{J}\}> in Eq. (64) in Ref.[1]
    veff1_2J = f1voJ[1:] # term from <g^{XC[2](\xi)};\{R^{S}_{I},R^{S}_{J}\}> in Eq. (64) in Ref.[1]
    t_debug_8 = log.timer_silent(*time0)[2]
    de = dh_td - ds + ejk
    dveff1_0 = rhf_grad.contract_h1e_dm(mol, veff1_0, dmz1doo, hermi=0)
    dveff1_1 = rhf_grad.contract_h1e_dm(mol, veff1_1, oo0, hermi=1) * .5
    dveff1_2  = rhf_grad.contract_h1e_dm(mol, veff1_2I, dmxpyJ, hermi=0) * 2
    dveff1_2 += rhf_grad.contract_h1e_dm(mol, veff1_2J, dmxpyI, hermi=0) * 2

    rIJoo_ao = reduce(cp.dot, (orbo, rIJoo, orbo.T))
    rIJvv_ao = reduce(cp.dot, (orbv, rIJvv, orbv.T))
    rIJooS_ao = reduce(cp.dot, (orbo, TIJoo, orbo.T))
    rIJvvS_ao = reduce(cp.dot, (orbv, TIJvv, orbv.T))
    dsxy  = rhf_grad.contract_h1e_dm(mol, s1, rIJoo_ao * (EJ - EI), hermi=1)
    dsxy += rhf_grad.contract_h1e_dm(mol, s1, rIJvv_ao * (EJ - EI), hermi=1)
    dsxy_etf  = rhf_grad.contract_h1e_dm(mol, s1, rIJooS_ao * (EJ - EI), hermi=1)
    dsxy_etf += rhf_grad.contract_h1e_dm(mol, s1, rIJvvS_ao * (EJ - EI), hermi=1)
    de += cp.asnumpy(dh1e_td) + dveff1_0 + dveff1_1 + dveff1_2 # Eq. (64) in Ref. [1]
    de_etf = de + dsxy_etf
    de += dsxy
    t_debug_9 = log.timer_silent(*time0)[2]
    if log.verbose >= logger.DEBUG:
        time_list = [0, t_debug_1, t_debug_2, t_debug_3, t_debug_4, t_debug_5, t_debug_6, t_debug_7, t_debug_8, t_debug_9]
        time_list = [time_list[i+1] - time_list[i] for i in range(len(time_list) - 1)]
        for i, t in enumerate(time_list):
            logger.note(td_nac, f"Time for step {i}: {t*1e-3:.6f}s")

    return de, de/(EJ - EI), de_etf, de_etf/(EJ - EI)

class NAC(tdrhf.NAC):

    _keys = {'ris_zvector_solver'}

    ris_zvector_solver = False

    @lib.with_doc(get_nacv_ge.__doc__)
    def get_nacv_ge(self, x_yI, EI, singlet, atmlst=None, verbose=logger.INFO):
        return get_nacv_ge(self, x_yI, EI, singlet, atmlst, verbose)

    @lib.with_doc(get_nacv_ee.__doc__)
    def get_nacv_ee(self, x_yI, x_yJ, EI, EJ, singlet, atmlst=None, verbose=logger.INFO):
        return get_nacv_ee(self, x_yI, x_yJ, EI, EJ, singlet, atmlst, verbose)
