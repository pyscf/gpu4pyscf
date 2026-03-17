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
from pyscf import lib
from gpu4pyscf.dft import rks
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.df import int3c2e
from gpu4pyscf.df.df_jk import (
    _tag_factorize_dm, _DFHF, _make_factorized_dm, _aggregate_dm_factor_l,
    _transpose_dm)
from gpu4pyscf.scf import cphf
from gpu4pyscf.nac.tdrhf_grad_nacv import contract_h1e_dm_batched
from gpu4pyscf.tdscf.ris import get_auxmol, rescale_spin_free_amplitudes
from gpu4pyscf.nac.tdrks_grad_nacv import NAC_multistates as NAC_multistates_tdrks
from gpu4pyscf.nac.tdrks_grad_nacv import (
    _contract_xc_kernel_batched, contract_veff_dm_batched, _solve_zvector,
    _c_mat_cT, _cT_mat_c, _aggregate_dms_factor_l, _dms_to_list)
from pyscf.data.nist import HARTREE2EV
from gpu4pyscf.grad import tdrks_ris

def get_nacv_ee_multi(td_nac, x_list, y_list, E_list, singlet=True, atmlst=None, verbose=logger.INFO, grad_state_idx=None):
    if td_nac.base.Ktrunc != 0.0:
        raise NotImplementedError('Ktrunc or frozen method is not supported yet')
    log = logger.new_logger(td_nac, verbose)
    time0 = logger.init_timer(td_nac)

    theta = td_nac.base.theta
    J_fit = td_nac.base.J_fit
    K_fit = td_nac.base.K_fit

    if not singlet:
        raise NotImplementedError('Only supports for singlet states')

    mol = td_nac.mol
    natm = mol.natm
    mf = td_nac.base._scf
    if getattr(mf, 'with_solvent', None) is not None:
        raise NotImplementedError('NACv gradient calculation is not supported for solvent models')

    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)

    nao, nmo = mo_coeff.shape
    orbo = mo_coeff[:, mo_occ > 0]
    orbv = mo_coeff[:, mo_occ ==0]
    nocc = orbo.shape[1]
    nvir = orbv.shape[1]

    n_states = len(E_list)

    is_tda = False
    X_stack = cp.asarray(x_list).reshape(n_states, nocc, nvir).transpose(0, 2, 1)
    if not isinstance(y_list[0], (np.ndarray, cp.ndarray)):
        Y_stack = cp.zeros_like(X_stack)
        is_tda = True
    else:
        Y_stack = cp.asarray(y_list).reshape(n_states, nocc, nvir).transpose(0, 2, 1)
    E_stack = cp.asarray(E_list)

    idx_i, idx_j = [], []
    for i in range(n_states):
        for j in range(i + 1, n_states):
            idx_i.append(i)
            idx_j.append(j)
    n_tasks = n_pairs = len(idx_i)
    if grad_state_idx is not None:
        idx_i.append(grad_state_idx)
        idx_j.append(grad_state_idx)
        n_tasks += 1

    xI = X_stack[idx_i]
    yI = Y_stack[idx_i]
    xJ = X_stack[idx_j]
    yJ = Y_stack[idx_j]

    EI = E_stack[idx_i]
    EJ = E_stack[idx_j]
    dE = EJ - EI
    if grad_state_idx is not None:
        # This effective denominator can make the code for NACV and gradients
        # almost identical.
        dE[-1] = 1.

    xpy_stack = X_stack + Y_stack
    xmy_stack = X_stack - Y_stack
    xpyI = xpy_stack[idx_i]
    xmyI = xmy_stack[idx_i]
    xpyJ = xpy_stack[idx_j]
    xmyJ = xmy_stack[idx_j]

    def transform_to_ao(amp_batch):
        amp_batch = contract('ua,nai->nui', orbv, amp_batch)
        return _make_factorized_dm(amp_batch, orbo, symmetrize=0)

    dmxpy_stack = transform_to_ao(xpy_stack)
    dmxmy_stack = transform_to_ao(xmy_stack)

    rIJoo = -cp.einsum('nai, naj -> nij', xJ, xI) - cp.einsum('nai, naj -> nij', yI, yJ)
    rIJvv = cp.einsum('nai, nbi -> nab', xI, xJ) + cp.einsum('nai, nbi -> nab', yJ, yI)

    TIJoo = (rIJoo + rIJoo.transpose(0, 2, 1)) * 0.5
    TIJvv = (rIJvv + rIJvv.transpose(0, 2, 1)) * 0.5
    dmzooIJ = _c_mat_cT(orbo, TIJoo, orbo) * 2.0
    dmzooIJ += _c_mat_cT(orbv, TIJvv, orbv) * 2.0

    t_debug_1 = log.timer_silent(*time0)[2]

    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    with_k = ni.libxc.is_hybrid_xc(mf.xc)
    f1ooIJ, _, _, vxc1, _ = _contract_xc_kernel_batched(
        td_nac, mf.xc, dmzooIJ, None, None, True, True, singlet)
    t_debug_2 = log.timer_silent(*time0)[2]

    auxmol_J = get_auxmol(mol=mol, theta=theta, fitting_basis=J_fit)
    if K_fit == J_fit and (omega == 0 or omega is None):
        auxmol_K = auxmol_J
    else:
        auxmol_K = get_auxmol(mol=mol, theta=theta, fitting_basis=K_fit)
    mf_J = rks.RKS(mol).density_fit()
    mf_J.with_df.auxmol = auxmol_J
    mf_K = rks.RKS(mol).density_fit()
    mf_K.with_df.auxmol = auxmol_K

    if with_k:
        beta = alpha - hyb
        dmzooIJ = _tag_factorize_dm(dmzooIJ, hermi=1)
        vj0IJ, vk0IJ = mf.get_jk(mol, dmzooIJ, hermi=1)
        vk0IJ *= hyb
        if omega != 0:
            vk0IJ += mf.get_k(mol, dmzooIJ, hermi=1, omega=omega) * beta
        vj = mf_J.get_j(mol, dmxpy_stack, hermi=0)
        vk_sym = mf_K.get_k(mol, dmxpy_stack, hermi=0) * hyb
        if omega != 0:
            vk_sym += mf_K.get_k(mol, dmxpy_stack, hermi=0, omega=omega) * beta
        if is_tda:
            vk_asym = vk_sym
        else:
            vk_asym = mf_K.get_k(mol, dmxmy_stack, hermi=0) * hyb
            if omega != 0:
                vk_asym += mf_K.get_k(mol, dmxmy_stack, hermi=0, omega=omega) * beta
        vj *= 2
        vk_sym = vk_sym + vk_sym.transpose(0,2,1)
        vk_asym = vk_asym - vk_asym.transpose(0,2,1)
        vj1I = vj[idx_i]
        vj1J = vj[idx_j]
        vk1I = vk_sym[idx_i]
        vk1J = vk_sym[idx_j]
        vk2I = vk_asym[idx_i]
        vk2J = vk_asym[idx_j]
    else:
        vj0IJ = mf.get_j(mol, _tag_factorize_dm(dmzooIJ, hermi=1), hermi=1)
        vj = mf_J.get_j(mol, dmxpy_stack, hermi=0)
        vj *= 2
        vj1I = vj[idx_i]
        vj1J = vj[idx_j]
        vk0IJ = vk1I = vk1J = 0

    # Extract Gradient specific VJ/VK
    if grad_state_idx is not None:
        if not singlet:
            vj1I[-1] = vj1J[-1] = 0
    vj = vk_sym = vk_asym = None

    def trans_veff_batch(veff_batch):
        return _cT_mat_c(mo_coeff, veff_batch, mo_coeff)

    veff0doo = vj0IJ * 2 - vk0IJ + f1ooIJ[:, 0]
    wvo = cp.einsum('pi, npq, qj -> nij', orbv, veff0doo, orbo) * 2.0
    veffI = vj1I * 2 - vk1I
    veffI *= 0.5
    veff0mopI = trans_veff_batch(veffI)
    wvo -= contract('nki, nai -> nak', veff0mopI[:, :nocc, :nocc], xpyJ) * 2.0
    wvo += contract('nac, nai -> nci', veff0mopI[:, nocc:, nocc:], xpyJ) * 2.0
    veffJ = vj1J * 2 - vk1J
    veffJ *= 0.5
    veff0mopJ = trans_veff_batch(veffJ)
    wvo -= contract('nki, nai -> nak', veff0mopJ[:, :nocc, :nocc], xpyI) * 2.0
    wvo += contract('nac, nai -> nci', veff0mopJ[:, nocc:, nocc:], xpyI) * 2.0
    if with_k:
        veffI = -vk2I * 0.5
        veff0momI = trans_veff_batch(veffI)
        wvo -= contract('nki, nai -> nak', veff0momI[:, :nocc, :nocc], xmyJ) * 2.0
        wvo += contract('nac, nai -> nci', veff0momI[:, nocc:, nocc:], xmyJ) * 2.0
        veffJ = -vk2J * 0.5
        veff0momJ = trans_veff_batch(veffJ)
        wvo -= contract('nki, nai -> nak', veff0momJ[:, :nocc, :nocc], xmyI) * 2.0
        wvo += contract('nac, nai -> nci', veff0momJ[:, nocc:, nocc:], xmyI) * 2.0
    else:
        veff0momI = cp.zeros((n_tasks, nmo, nmo))
        veff0momJ = cp.zeros((n_tasks, nmo, nmo))

    t_debug_3 = log.timer_silent(*time0)[2]

    rhs = wvo
    # rhs = wvo / dE[:, None, None]

    if td_nac.ris_zvector_solver:
        logger.note(td_nac, 'Use ris-approximated Z-vector solver')
        vresp = tdrks_ris.gen_response_ris(mf, mf_J, mf_K, singlet=None, hermi=1)
    else:
        logger.note(td_nac, 'Use standard Z-vector solver')
        vresp = mf.gen_response(singlet=None, hermi=1)

    z1 = _solve_zvector(td_nac, rhs, vresp)
    t_debug_4 = log.timer_silent(*time0)[2]

    z1 /= dE[:, None, None]
    z1ao_sym = _make_factorized_dm(
        contract('ua,nai->nui', orbv, z1), orbo, symmetrize=1)

    z1aoS = z1ao_sym * 0.5 * dE[:, None, None]
    dmz1doo = z1aoS + dmzooIJ

    veff = vresp(z1ao_sym)
    fock_mo = cp.diag(mo_energy)
    TFoo = cp.matmul(TIJoo, fock_mo[:nocc, :nocc])
    TFov = cp.matmul(TIJoo, fock_mo[:nocc, nocc:])
    TFvo = cp.matmul(TIJvv, fock_mo[nocc:, :nocc])
    TFvv = cp.matmul(TIJvv, fock_mo[nocc:, nocc:])

    im0 = cp.zeros((n_tasks, nmo, nmo))

    term_oo = cp.einsum('ui, nuv, vj -> nij', orbo, veff0doo, orbo)
    term_oo += TFoo * 2.0
    term_oo += cp.einsum('nak, nai -> nik', veff0mopI[:, nocc:, :nocc], xpyJ)
    term_oo += cp.einsum('nak, nai -> nik', veff0momI[:, nocc:, :nocc], xmyJ)
    term_oo += cp.einsum('nak, nai -> nik', veff0mopJ[:, nocc:, :nocc], xpyI)
    term_oo += cp.einsum('nak, nai -> nik', veff0momJ[:, nocc:, :nocc], xmyI)
    term_oo[:n_pairs] += rIJoo[:n_pairs].transpose(0, 2, 1) * dE[:n_pairs, None, None]
    im0[:, :nocc, :nocc] = term_oo

    term_ov = cp.einsum('ui, nuv, va -> nia', orbo, veff0doo, orbv)
    term_ov += TFov * 2.0
    term_ov += cp.einsum('nab, nai -> nib', veff0mopI[:, nocc:, nocc:], xpyJ)
    term_ov += cp.einsum('nab, nai -> nib', veff0momI[:, nocc:, nocc:], xmyJ)
    term_ov += cp.einsum('nab, nai -> nib', veff0mopJ[:, nocc:, nocc:], xpyI)
    term_ov += cp.einsum('nab, nai -> nib', veff0momJ[:, nocc:, nocc:], xmyI)
    im0[:, :nocc, nocc:] = term_ov

    term_vo = TFvo * 2.0
    term_vo += cp.einsum('nij, nai -> naj', veff0mopI[:, :nocc, :nocc], xpyJ)
    term_vo -= cp.einsum('nij, nai -> naj', veff0momI[:, :nocc, :nocc], xmyJ)
    term_vo += cp.einsum('nij, nai -> naj', veff0mopJ[:, :nocc, :nocc], xpyI)
    term_vo -= cp.einsum('nij, nai -> naj', veff0momJ[:, :nocc, :nocc], xmyI)
    im0[:, nocc:, :nocc] = term_vo

    term_vv = TFvv * 2.0
    term_vv += cp.einsum('nib, nai -> nab', veff0mopI[:, :nocc, nocc:], xpyJ)
    term_vv -= cp.einsum('nib, nai -> nab', veff0momI[:, :nocc, nocc:], xmyJ)
    term_vv += cp.einsum('nib, nai -> nab', veff0mopJ[:, :nocc, nocc:], xpyI)
    term_vv -= cp.einsum('nib, nai -> nab', veff0momJ[:, :nocc, nocc:], xmyI)
    term_vv[:n_pairs] += rIJvv[:n_pairs].transpose(0, 2, 1) * dE[:n_pairs, None, None]
    im0[:, nocc:, nocc:] = term_vv

    im0 *= 0.5

    im0[:, :nocc, :nocc] += cp.einsum('ui, nuv, vj -> nij', orbo, veff, orbo) * dE[:, None, None] * 0.5
    im0[:, :nocc, nocc:] += cp.einsum('ui, nuv, va -> nia', orbo, veff, orbv) * dE[:, None, None] * 0.5
    z1_fock_ov = cp.einsum('ab, nbi -> nai', fock_mo[nocc:, nocc:], z1)
    im0[:, :nocc, nocc:] += z1_fock_ov.transpose(0, 2, 1) * dE[:, None, None] * 0.25
    z1_fock_vo = cp.einsum('nai, ij -> naj', z1, fock_mo[:nocc, :nocc])
    im0[:, nocc:, :nocc] += z1_fock_vo * dE[:, None, None] * 0.25

    im0 *= 2

    if grad_state_idx is not None:
        im0_g = im0[-1]
        # The energy weighted DM
        im0_g[:nocc,:nocc] += np.diag(mo_energy[:nocc]) * 2.

    im0_ao = _c_mat_cT(mo_coeff, im0, mo_coeff)

    t_debug_5 = log.timer_silent(*time0)[2]

    oo0 = _make_factorized_dm(orbo*2, orbo, symmetrize=0) # *2 for double occupancy
    if grad_state_idx is not None:
        dmz1doo[-1] += oo0

    mf_grad = td_nac.base._scf.nuc_grad_method()
    h1 = cp.asarray(mf_grad.get_hcore(mol))
    s1 = cp.asarray(mf_grad.get_ovlp(mol))
    dh_td = contract_h1e_dm_batched(mol, h1, dmz1doo, hermi=1)
    ds = contract_h1e_dm_batched(mol, s1, im0_ao, hermi=0)

    dh1e_td_list = []
    for k in range(n_tasks):
        dh1e_k = int3c2e.get_dh1e(mol, dmz1doo[k])
        if len(mol._ecpbas) > 0:
            dh1e_k += rhf_grad.get_dh1e_ecp(mol, dmz1doo[k])
        dh1e_td_list.append(dh1e_k)

    de = dh_td - ds + cp.array(dh1e_td_list)

    if mol._pseudo:
        raise NotImplementedError("Pseudopotential gradient not supported for molecular system yet")
    t_debug_6 = log.timer_silent(*time0)[2]

    dmxpy_stack = _dms_to_list(dmxpy_stack)
    dmxmy_stack = _dms_to_list(dmxmy_stack)

    j_factor = [1.] * n_tasks
    k_factor = None
    if with_k:
        k_factor = [1.] * n_tasks
    dms_tasks = [[_tag_factorize_dm(dmz1doo[k], hermi=1), oo0]
                 for k in range(n_pairs)]
    if grad_state_idx is not None:
        dms_tasks.append(
            [_tag_factorize_dm(dmz1doo[-1] - oo0*.5, hermi=1), oo0])

    if with_k:
        k_factor = np.array(k_factor)
        ejk = td_nac.jk_energies_per_atom(
            dms_tasks, j_factor, k_factor*hyb, sum_results=False)
    else:
        ejk = td_nac.jk_energies_per_atom(
            dms_tasks, j_factor, None, sum_results=False)

    if with_k and omega != 0:
        beta = alpha - hyb
        ejk += td_nac.jk_energies_per_atom(
            dms_tasks, None, k_factor*beta, omega=omega, sum_results=False)
    ejk *= 2

    if not is_tda:
        dms_tasks = []
        j_factor = [2., 0.] * n_tasks
        if with_k:
            k_factor = [2.,-2.] * n_tasks
        for k, (I, J) in enumerate(zip(idx_i, idx_j)):
            dms_tasks.extend(
                [[dmxpy_stack[I], dmxpy_stack[J] + dmxpy_stack[J].T],
                 [dmxmy_stack[I], dmxmy_stack[J] - dmxmy_stack[J].T]])
        if grad_state_idx is not None:
            if not singlet:
                j_factor[-2] = 0.
        if not with_k:
            j_factor = j_factor[::2]
            dms_tasks = dms_tasks[::2]
    else:
        dms_tasks = []
        j_factor = [4.] * n_tasks
        if with_k:
            k_factor = [4.] * n_tasks
        for k, (I, J) in enumerate(zip(idx_i, idx_j)):
            dms_tasks.append([dmxpy_stack[I], _transpose_dm(dmxpy_stack[J])])
        if grad_state_idx is not None:
            if not singlet:
                j_factor[-1] = 0.

    if with_k:
        k_factor = np.array(k_factor)
        ejk_ris = tdrks_ris.jk_energies_per_atom(
            mf_J, mf_K, mol, dms_tasks, j_factor, k_factor*hyb, sum_results=False)
        ejk += ejk_ris.reshape(n_tasks, -1, natm, 3).sum(axis=1) * 2
    else:
        ejk += tdrks_ris.jk_energies_per_atom(
            mf_J, mf_K, mol, dms_tasks, j_factor, None, sum_results=False) * 2

    if with_k and omega != 0:
        beta = alpha - hyb
        ejk_ris = tdrks_ris.jk_energies_per_atom(
            mf_J, mf_K, mol, dms_tasks, None, k_factor*beta, omega=omega,
            sum_results=False)
        ejk += ejk_ris.reshape(n_tasks, -1, natm, 3).sum(axis=1) * 2

    de += cp.asarray(ejk)
    t_debug_7 = log.timer_silent(*time0)[2]

    fxcz1 = _contract_xc_kernel_batched(
        td_nac, mf.xc, z1aoS, None, None, False, False, True)[0]
    t_debug_8 = log.timer_silent(*time0)[2]

    veff1_1_batch = f1ooIJ[:, 1:] + fxcz1[:, 1:]
    veff1_0_batch = cp.repeat(vxc1[None,1:], n_tasks, axis=0)
    dveff1_0 = contract_veff_dm_batched(mol, veff1_0_batch, dmz1doo, hermi=0)
    oo0_batch = cp.repeat(oo0[None, ...], n_tasks, axis=0)
    dveff1_1 = contract_veff_dm_batched(mol, veff1_1_batch, oo0_batch, hermi=1) * 0.5

    de += dveff1_0 + dveff1_1

    results = {}
    if grad_state_idx is not None:
        de, de_grad = de[:-1], de[-1].get()
        rIJoo = rIJoo[:-1]
        rIJvv = rIJvv[:-1]
        TIJoo = TIJoo[:-1]
        TIJvv = TIJvv[:-1]
        dE = dE[:-1]
        idx_i = idx_i[:-1]
        idx_j = idx_j[:-1]
        de_grad += mf_grad.grad_nuc(mol)
        results['gradient'] = de_grad

    rIJoo_ao = _c_mat_cT(orbo, rIJoo, orbo)
    rIJvv_ao = _c_mat_cT(orbv, rIJvv, orbv)
    TIJoo_ao = _c_mat_cT(orbo, TIJoo, orbo)
    TIJvv_ao = _c_mat_cT(orbv, TIJvv, orbv)
    dsxy = contract_h1e_dm_batched(mol, s1, rIJoo_ao * dE[:, None, None], hermi=1)
    dsxy += contract_h1e_dm_batched(mol, s1, rIJvv_ao * dE[:, None, None], hermi=1)
    dsxy_etf = contract_h1e_dm_batched(mol, s1, TIJoo_ao * dE[:, None, None], hermi=1)
    dsxy_etf += contract_h1e_dm_batched(mol, s1, TIJvv_ao * dE[:, None, None], hermi=1)

    de_etf = de + dsxy_etf
    de += dsxy

    de = de.get()
    de_etf = de_etf.get()
    dE = dE.get()
    for k, (i, j) in enumerate(zip(idx_i, idx_j)):
        results[(int(i), int(j))] = {
            'de': de[k],
            'de_scaled': de[k] / dE[k],
            'de_etf': de_etf[k],
            'de_etf_scaled': de_etf[k] / dE[k]
        }

    t_debug_9 = log.timer_silent(*time0)[2]
    if log.verbose >= logger.DEBUG:
        time_list = [0, t_debug_1, t_debug_2, t_debug_3, t_debug_4, t_debug_5, t_debug_6, t_debug_7, t_debug_8, t_debug_9]
        time_list = [time_list[i+1] - time_list[i] for i in range(len(time_list) - 1)]
        for i, t in enumerate(time_list):
            logger.note(td_nac, f"Time for step {i}: {t*1e-3:.6f}s")
    return results


class NAC_multistates(NAC_multistates_tdrks):

    _keys = {'ris_zvector_solver'}

    def __init__(self, td):
        super().__init__(td)
        self.ris_zvector_solver = False

    def get_nacv_ee_multi(self, x_list, y_list, E_list, singlet, atmlst=None, verbose=logger.INFO, grad_state_idx=None):
        return get_nacv_ee_multi(self, x_list, y_list, E_list, singlet, atmlst, verbose, grad_state_idx)

    def kernel(self, states=None, singlet=None, atmlst=None, grad_state=None):

        logger.warn(self, "NAC Multi-State Module (Experimental)")

        if singlet is None:
            singlet = self.base.singlet
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if states is not None:
            self.states = states

        if grad_state is not None:
            self.grad_state = grad_state

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        target_states = sorted(list(set(self.states)))
        if len(target_states) < 2:
            raise ValueError("Must provide at least 2 states for NACV calculation.")
        if any(s < 0 for s in target_states):
            raise ValueError("State indices must be non-negative.")
        nstates = len(self.base.energies)
        if any(s > nstates for s in target_states):
            raise ValueError(f"State index exceeds number of roots ({nstates}).")
        if len(target_states) > nstates:
            raise ValueError(f"Only {nstates} states available, but requested {len(target_states)}.")

        # Ensure that the chosen grad_state is within the evaluated target_states.
        if self.grad_state is not None and self.grad_state not in target_states:
            raise ValueError(f"grad_state {self.grad_state} is requested, ",
                "but it is not within the provided target states {target_states} for NACV calculation.")

        self.results = {}

        has_ground = (0 in target_states)
        excited_states = [s for s in target_states if s > 0]

        if len(excited_states) >= 2:
            logger.info(self, f"Computing Vectorized NACV for excited states EE: {excited_states}")

            x_list, y_list, E_list = [], [], []
            for s in excited_states:
                x_list.append(rescale_spin_free_amplitudes(self.base.xy, s-1)[0])
                y_list.append(rescale_spin_free_amplitudes(self.base.xy, s-1)[1])
                E_list.append(self.base.energies[s-1]/HARTREE2EV)

            grad_idx = excited_states.index(self.grad_state) if (self.grad_state is not None and self.grad_state > 0) else None

            ee_results = self.get_nacv_ee_multi(
                x_list, y_list, E_list, singlet, atmlst, verbose=self.verbose, grad_state_idx=grad_idx
            )

            if 'gradient' in ee_results:
                self.grad_result = ee_results.pop('gradient')

            for (local_i, local_j), res in ee_results.items():
                global_i = excited_states[local_i]
                global_j = excited_states[local_j]
                self.results[(global_i, global_j)] = res

        if has_ground and len(excited_states) > 0:
            logger.info(self, f"Computing Vectorized NACV for Ground (0) - Excited GE: {excited_states}")

            x_ge_list, y_ge_list, E_ge_list = [], [], []
            for s in excited_states:
                x_ge_list.append(rescale_spin_free_amplitudes(self.base.xy, s-1)[0])
                y_ge_list.append(rescale_spin_free_amplitudes(self.base.xy, s-1)[1])
                E_ge_list.append(self.base.energies[s-1]/HARTREE2EV)

            ge_results = self.get_nacv_ge_multi(
                x_ge_list, y_ge_list, E_ge_list, singlet, atmlst, verbose=self.verbose
            )

            for local_idx, res in ge_results.items():
                global_s = excited_states[local_idx]
                self.results[(0, global_s)] = res

        if self.grad_state == 0:
            self.grad_result = self.base._scf.nuc_grad_method().kernel(atmlst=atmlst)

        self._finalize()
        return self.results

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(self, "\n" + "="*60)
            logger.note(self, " NACV Calculation Summary using RIS approximation")
            logger.note(self, "="*60)

            for (i, j) in sorted(self.results.keys()):
                res = self.results[(i, j)]
                logger.note(self, f"\nPair ({i}, {j}):")
                logger.note(self, f"  - DE (CIS Force)      : \n{res['de']}")
                logger.note(self, f"  - DE (CIS Force) (Scaled)  : \n{res['de_scaled']}")
                logger.note(self, f"  - DE (ETF Force)      : \n{res['de_etf']}")
                logger.note(self, f"  - DE (ETF Force) (Scaled)  : \n{res['de_etf_scaled']}")

            if self.grad_state is not None and self.grad_result is not None:
                logger.note(self, f"\nGradient (Energy Derivative) for State ({self.grad_state}):\n{self.grad_result}")

            logger.note(self, "-"*60 + "\n")
