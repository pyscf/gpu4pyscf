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
    _tag_factorize_dm, _make_factorized_dm,
    _transpose_dm)
from gpu4pyscf.nac.tdrhf_grad_nacv import contract_h1e_dm_batched, contract_h1e_dm_asym_batched
from gpu4pyscf.tdscf.ris import get_auxmol, rescale_spin_free_amplitudes
from gpu4pyscf.nac.tdrks_grad_nacv import NAC_multistates as NAC_multistates_tdrks
from gpu4pyscf.nac.tdrks_grad_nacv import (
    _contract_xc_kernel_batched, contract_veff_dm_batched, _solve_zvector,
    _c_mat_cT, _cT_mat_c, _aggregate_dms_factor_l, _dms_to_list)
from pyscf.data.nist import HARTREE2EV
from gpu4pyscf.grad import tdrks_ris


def get_nacv_multi(td_nac, x_list, y_list, E_list, singlet=True, calc_ge=False, 
    calc_ee=False, grad_state_idx=None, atmlst=None, verbose=logger.INFO):
    """
    Unified function to calculate Non-Adiabatic Coupling Vectors (NACV) 
    for Ground-Excited (GE), Excited-Excited (EE), and energy gradients simultaneously.
    Designed for TDRKS with RIS approximated Z-vector and JK evaluations.
    """
    if td_nac.base.Ktrunc != 0.0:
        raise NotImplementedError('Ktrunc or frozen method is not supported yet')
    if not singlet:
        raise NotImplementedError('Only supports for singlet states')
        
    log = logger.new_logger(td_nac, verbose)
    time0 = log.init_timer()

    mol = td_nac.mol
    natm = mol.natm
    mf = td_nac.base._scf
    if getattr(mf, 'with_solvent', None) is not None:
        raise NotImplementedError('NACv gradient calculation is not supported for solvent models')

    mf_grad = mf.nuc_grad_method()
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)

    nao, nmo = mo_coeff.shape
    orbo = mo_coeff[:, mo_occ > 0]
    orbv = mo_coeff[:, mo_occ == 0]
    nocc = orbo.shape[1]
    nvir = orbv.shape[1]

    n_states = len(E_list)
    is_tda = not isinstance(y_list[0], (np.ndarray, cp.ndarray))

    X_stack = cp.asarray(x_list).reshape(n_states, nocc, nvir).transpose(0, 2, 1)
    if is_tda:
        Y_stack = cp.zeros_like(X_stack)
    else:
        Y_stack = cp.asarray(y_list).reshape(n_states, nocc, nvir).transpose(0, 2, 1)
    E_stack = cp.asarray(E_list)

    n_tasks_ge = n_states if calc_ge else 0
    
    idx_i, idx_j = [], []
    if calc_ee:
        for i in range(n_states):
            for j in range(i + 1, n_states):
                idx_i.append(i)
                idx_j.append(j)
    n_pairs = len(idx_i)
    
    has_grad = grad_state_idx is not None
    if has_grad:
        idx_i.append(grad_state_idx)
        idx_j.append(grad_state_idx)
    n_tasks_ee = len(idx_i)
    total_tasks = n_tasks_ge + n_tasks_ee
    if total_tasks == 0:
        return {}

    theta = td_nac.base.theta
    J_fit = td_nac.base.J_fit
    K_fit = td_nac.base.K_fit

    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    with_k = ni.libxc.is_hybrid_xc(mf.xc)

    auxmol_J = get_auxmol(mol=mol, theta=theta, fitting_basis=J_fit)
    if K_fit == J_fit and (omega == 0 or omega is None):
        auxmol_K = auxmol_J
    else:
        auxmol_K = get_auxmol(mol=mol, theta=theta, fitting_basis=K_fit)
        
    mf_J = rks.RKS(mol).density_fit()
    mf_J.with_df.auxmol = auxmol_J
    mf_K = rks.RKS(mol).density_fit()
    mf_K.with_df.auxmol = auxmol_K

    rhs_list = []

    if calc_ge:
        LI = X_stack - Y_stack
        rhs_ge = -LI * E_stack[:, None, None]
        rhs_list.append(rhs_ge)

    if n_tasks_ee > 0:
        xI, yI = X_stack[idx_i], Y_stack[idx_i]
        xJ, yJ = X_stack[idx_j], Y_stack[idx_j]

        EI, EJ = E_stack[idx_i], E_stack[idx_j]
        dE = EJ - EI
        if has_grad:
            dE[-1] = 1.0 

        xpy_stack, xmy_stack = X_stack + Y_stack, X_stack - Y_stack
        xpyI, xmyI = xpy_stack[idx_i], xmy_stack[idx_i]
        xpyJ, xmyJ = xpy_stack[idx_j], xmy_stack[idx_j]

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

        f1ooIJ, _, _, vxc1_ee, _ = _contract_xc_kernel_batched(
            td_nac, mf.xc, dmzooIJ, None, None, True, True, singlet)

        if with_k:
            beta = alpha - hyb
            dmzooIJ_factorized = _tag_factorize_dm(dmzooIJ, hermi=1)
            vj0IJ, vk0IJ = mf.get_jk(mol, dmzooIJ_factorized, hermi=1)
            vk0IJ *= hyb
            if omega != 0:
                vk0IJ += mf.get_k(mol, dmzooIJ_factorized, hermi=1, omega=omega) * beta
                
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
            
            vj1I, vj1J = vj[idx_i], vj[idx_j]
            vk1I, vk1J = vk_sym[idx_i], vk_sym[idx_j]
            vk2I, vk2J = vk_asym[idx_i], vk_asym[idx_j]
        else:
            vj0IJ = mf.get_j(mol, _tag_factorize_dm(dmzooIJ, hermi=1), hermi=1)
            vj = mf_J.get_j(mol, dmxpy_stack, hermi=0)
            vj *= 2
            vj1I, vj1J = vj[idx_i], vj[idx_j]
            vk0IJ = vk1I = vk1J = 0

        if has_grad and not singlet:
            vj1I[-1] = vj1J[-1] = 0

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
            veffI_k = -vk2I * 0.5
            veff0momI = trans_veff_batch(veffI_k)
            wvo -= contract('nki, nai -> nak', veff0momI[:, :nocc, :nocc], xmyJ) * 2.0
            wvo += contract('nac, nai -> nci', veff0momI[:, nocc:, nocc:], xmyJ) * 2.0
            veffJ_k = -vk2J * 0.5
            veff0momJ = trans_veff_batch(veffJ_k)
            wvo -= contract('nki, nai -> nak', veff0momJ[:, :nocc, :nocc], xmyI) * 2.0
            wvo += contract('nac, nai -> nci', veff0momJ[:, nocc:, nocc:], xmyI) * 2.0
        else:
            veff0momI = cp.zeros((n_tasks_ee, nmo, nmo))
            veff0momJ = cp.zeros((n_tasks_ee, nmo, nmo))

        rhs_list.append(wvo)

    rhs_all = cp.concatenate(rhs_list, axis=0)
    t_debug_1 = log.timer_silent(*time0)[2]

    if td_nac.ris_zvector_solver:
        logger.note(td_nac, 'Use ris-approximated Z-vector solver')
        vresp = tdrks_ris.gen_response_ris(mf, mf_J, mf_K, singlet=None, hermi=1)
    else:
        logger.note(td_nac, 'Use standard Z-vector solver')
        vresp = mf.gen_response(singlet=None, hermi=1)

    z1_all = _solve_zvector(td_nac, rhs_all, vresp)
    # for i in range(z1_all.shape[0]):
    #     z1_all[i] = _solve_zvector(td_nac, rhs_all[i][None, :, :], vresp)
    t_debug_2 = log.timer_silent(*time0)[2]

    dmz1doo_list = []
    W_ao_list = []
    offset = 0

    oo0 = _make_factorized_dm(orbo*2, orbo, symmetrize=0)

    if calc_ge:
        z1_ge = z1_all[offset:offset+n_tasks_ge]
        offset += n_tasks_ge

        z1aoS_ge = _make_factorized_dm(contract('ua,nai->nui', orbv, z1_ge), orbo, symmetrize=1)
        GZS_mo_ge = _cT_mat_c(mo_coeff, vresp(z1aoS_ge), mo_coeff)

        W_ge = cp.zeros((n_tasks_ge, nmo, nmo))
        W_ge[:, :nocc, :nocc] = GZS_mo_ge[:, :nocc, :nocc]
        zeta0 = z1_ge * mo_energy[nocc:][None, :, None]
        W_ge[:, :nocc, nocc:] = GZS_mo_ge[:, :nocc, nocc:] + 0.5 * Y_stack.transpose(0, 2, 1) * E_stack[:, None, None] + 0.5 * zeta0.transpose(0, 2, 1)
        zeta1 = z1_ge * mo_energy[None, None, :nocc]
        W_ge[:, nocc:, :nocc] = 0.5 * X_stack * E_stack[:, None, None] + 0.5 * zeta1

        W_ao_ge = _c_mat_cT(mo_coeff, W_ge, mo_coeff) * 2.0

        dmz1doo_list.append(cp.asarray(z1aoS_ge)) 
        W_ao_list.append(W_ao_ge)

    if n_tasks_ee > 0:
        z1_ee = z1_all[offset:]
        z1_ee /= dE[:, None, None]

        z1ao_sym = _make_factorized_dm(contract('ua,nai->nui', orbv, z1_ee), orbo, symmetrize=1)
        z1aoS_ee = z1ao_sym * 0.5 * dE[:, None, None]
        dmz1doo_ee = z1aoS_ee + dmzooIJ

        veff_ee = vresp(z1ao_sym)
        fock_mo = cp.diag(mo_energy)

        im0 = cp.zeros((n_tasks_ee, nmo, nmo))
        term_oo = cp.einsum('ui, nuv, vj -> nij', orbo, veff0doo, orbo)
        term_oo += cp.matmul(TIJoo, fock_mo[:nocc, :nocc]) * 2.0
        term_oo += cp.einsum('nak, nai -> nik', veff0mopI[:, nocc:, :nocc], xpyJ)
        term_oo += cp.einsum('nak, nai -> nik', veff0momI[:, nocc:, :nocc], xmyJ)
        term_oo += cp.einsum('nak, nai -> nik', veff0mopJ[:, nocc:, :nocc], xpyI)
        term_oo += cp.einsum('nak, nai -> nik', veff0momJ[:, nocc:, :nocc], xmyI)
        term_oo[:n_pairs] += rIJoo[:n_pairs].transpose(0, 2, 1) * dE[:n_pairs, None, None]
        im0[:, :nocc, :nocc] = term_oo

        term_ov = cp.einsum('ui, nuv, va -> nia', orbo, veff0doo, orbv)
        term_ov += cp.matmul(TIJoo, fock_mo[:nocc, nocc:]) * 2.0
        term_ov += cp.einsum('nab, nai -> nib', veff0mopI[:, nocc:, nocc:], xpyJ)
        term_ov += cp.einsum('nab, nai -> nib', veff0momI[:, nocc:, nocc:], xmyJ)
        term_ov += cp.einsum('nab, nai -> nib', veff0mopJ[:, nocc:, nocc:], xpyI)
        term_ov += cp.einsum('nab, nai -> nib', veff0momJ[:, nocc:, nocc:], xmyI)
        im0[:, :nocc, nocc:] = term_ov

        term_vo = cp.matmul(TIJvv, fock_mo[nocc:, :nocc]) * 2.0
        term_vo += cp.einsum('nij, nai -> naj', veff0mopI[:, :nocc, :nocc], xpyJ)
        term_vo -= cp.einsum('nij, nai -> naj', veff0momI[:, :nocc, :nocc], xmyJ)
        term_vo += cp.einsum('nij, nai -> naj', veff0mopJ[:, :nocc, :nocc], xpyI)
        term_vo -= cp.einsum('nij, nai -> naj', veff0momJ[:, :nocc, :nocc], xmyI)
        im0[:, nocc:, :nocc] = term_vo

        term_vv = cp.matmul(TIJvv, fock_mo[nocc:, nocc:]) * 2.0
        term_vv += cp.einsum('nib, nai -> nab', veff0mopI[:, :nocc, nocc:], xpyJ)
        term_vv -= cp.einsum('nib, nai -> nab', veff0momI[:, :nocc, nocc:], xmyJ)
        term_vv += cp.einsum('nib, nai -> nab', veff0mopJ[:, :nocc, nocc:], xpyI)
        term_vv -= cp.einsum('nib, nai -> nab', veff0momJ[:, :nocc, nocc:], xmyI)
        term_vv[:n_pairs] += rIJvv[:n_pairs].transpose(0, 2, 1) * dE[:n_pairs, None, None]
        im0[:, nocc:, nocc:] = term_vv

        im0 *= 0.5
        im0[:, :nocc, :nocc] += cp.einsum('ui, nuv, vj -> nij', orbo, veff_ee, orbo) * dE[:, None, None] * 0.5
        im0[:, :nocc, nocc:] += cp.einsum('ui, nuv, va -> nia', orbo, veff_ee, orbv) * dE[:, None, None] * 0.5
        z1_fock_ov = cp.einsum('ab, nbi -> nai', fock_mo[nocc:, nocc:], z1_ee)
        im0[:, :nocc, nocc:] += z1_fock_ov.transpose(0, 2, 1) * dE[:, None, None] * 0.25
        z1_fock_vo = cp.einsum('nai, ij -> naj', z1_ee, fock_mo[:nocc, :nocc])
        im0[:, nocc:, :nocc] += z1_fock_vo * dE[:, None, None] * 0.25

        im0 *= 2.0

        if has_grad:
            dmz1doo_ee[-1] += cp.asarray(oo0)
            im0[-1, :nocc, nocc:] = 0
            im0[-1, nocc:, :nocc] *= 2.0
            im0[-1, :nocc, :nocc] += np.diag(mo_energy[:nocc]) * 2.0

        im0_ao_ee = _c_mat_cT(mo_coeff, im0, mo_coeff)

        dmz1doo_list.append(cp.asarray(dmz1doo_ee))
        W_ao_list.append(im0_ao_ee)

    P_all = cp.concatenate(dmz1doo_list, axis=0)
    W_all = cp.concatenate(W_ao_list, axis=0)
    t_debug_3 = log.timer_silent(*time0)[2]

    h1 = cp.asarray(mf_grad.get_hcore(mol))
    s1 = cp.asarray(mf_grad.get_ovlp(mol))

    dh_td_all = contract_h1e_dm_batched(mol, h1, P_all, hermi=1)
    ds_all = contract_h1e_dm_batched(mol, s1, W_all, hermi=0)

    dh1e_td_list = []
    for k in range(total_tasks):
        dh1e_k = int3c2e.get_dh1e(mol, P_all[k])
        if len(mol._ecpbas) > 0:
            dh1e_k += rhf_grad.get_dh1e_ecp(mol, P_all[k])
        dh1e_td_list.append(dh1e_k)
    dh1e_td_all = cp.array(dh1e_td_list)

    dms_tasks_exact = []
    j_factor_exact = [1.0] * total_tasks
    k_factor_exact = [1.0] * total_tasks if with_k else None

    for k in range(total_tasks):
        if has_grad and k == total_tasks - 1:
            dms_tasks_exact.append([_tag_factorize_dm(P_all[k] - cp.asarray(oo0)*.5, hermi=1), oo0])
        else:
            dms_tasks_exact.append([_tag_factorize_dm(P_all[k], hermi=1), oo0])

    t_debug_4 = log.timer_silent(*time0)[2]

    if with_k:
        k_factor_exact_np = np.array(k_factor_exact)
        ejk_exact_raw = td_nac.jk_energies_per_atom(
            dms_tasks_exact, j_factor_exact, k_factor_exact_np * hyb, sum_results=False)
    else:
        ejk_exact_raw = td_nac.jk_energies_per_atom(
            dms_tasks_exact, j_factor_exact, None, sum_results=False)
    
    ejk_exact_raw = cp.asarray(ejk_exact_raw) * 2.0

    if with_k and omega != 0:
        beta = alpha - hyb
        ejk_exact_lr = td_nac.jk_energies_per_atom(
            dms_tasks_exact, None, k_factor_exact_np * beta, omega=omega, sum_results=False)
        ejk_exact_raw += cp.asarray(ejk_exact_lr) * 2.0

    de_all = dh_td_all - ds_all + dh1e_td_all + ejk_exact_raw

    if n_tasks_ee > 0:
        dms_tasks_ris = []
        j_factor_ris = []
        k_factor_ris = [] if with_k else None
        
        dmxpy_stack = _dms_to_list(dmxpy_stack)
        dmxmy_stack = _dms_to_list(dmxmy_stack)

        if not is_tda:
            j_factor_ris.extend([2., 0.] * n_tasks_ee)
            if with_k: k_factor_ris.extend([2., -2.] * n_tasks_ee)
            for k, (I, J) in enumerate(zip(idx_i, idx_j)):
                dms_tasks_ris.extend([
                    [dmxpy_stack[I], dmxpy_stack[J] + dmxpy_stack[J].T],
                    [dmxmy_stack[I], dmxmy_stack[J] - dmxmy_stack[J].T]
                ])
            if has_grad:
                if not singlet: j_factor_ris[-2] = 0.
        else:
            j_factor_ris.extend([4.] * n_tasks_ee)
            if with_k: k_factor_ris.extend([4.] * n_tasks_ee)
            for k, (I, J) in enumerate(zip(idx_i, idx_j)):
                dms_tasks_ris.append([dmxpy_stack[I], _transpose_dm(dmxpy_stack[J])])
            if has_grad:
                if not singlet: j_factor_ris[-1] = 0.

        if not with_k and not is_tda:
            j_factor_ris = j_factor_ris[::2]
            dms_tasks_ris = dms_tasks_ris[::2]

        if with_k:
            k_factor_ris_np = np.array(k_factor_ris)
            ejk_ris_raw = tdrks_ris.jk_energies_per_atom(
                mf_J, mf_K, mol, dms_tasks_ris, j_factor_ris, k_factor_ris_np * hyb, sum_results=False)
        else:
            ejk_ris_raw = tdrks_ris.jk_energies_per_atom(
                mf_J, mf_K, mol, dms_tasks_ris, j_factor_ris, None, sum_results=False)
        
        ejk_ris_raw = cp.asarray(ejk_ris_raw) * 2.0
        
        ejk_ee_ris = ejk_ris_raw.reshape(n_tasks_ee, -1, natm, 3).sum(axis=1)

        if with_k and omega != 0:
            beta = alpha - hyb
            ejk_ris_lr = tdrks_ris.jk_energies_per_atom(
                mf_J, mf_K, mol, dms_tasks_ris, None, k_factor_ris_np * beta, omega=omega, sum_results=False)
            ejk_ee_ris += cp.asarray(ejk_ris_lr).reshape(n_tasks_ee, -1, natm, 3).sum(axis=1) * 2.0

        de_all[n_tasks_ge:] += ejk_ee_ris

    t_debug_5 = log.timer_silent(*time0)[2]

    if calc_ge:
        f1ooP_batch, _, _, vxc1_ge, _ = _contract_xc_kernel_batched(
            td_nac, mf.xc, dmz1doo_list[0], None, None, True, False, singlet)
        vxc1_batch_ge = cp.repeat(vxc1_ge[None], n_tasks_ge, axis=0)
        
        veff1_0_batch_ge = vxc1_batch_ge[:, 1:]
        veff1_1_batch_ge = f1ooP_batch[:, 1:]
        
        dveff1_0_ge = contract_veff_dm_batched(mol, veff1_0_batch_ge, dmz1doo_list[0], hermi=0)
        oo0_batch_ge = cp.repeat(oo0[None, ...], n_tasks_ge, axis=0)
        dveff1_1_ge = contract_veff_dm_batched(mol, veff1_1_batch_ge, oo0_batch_ge, hermi=1) * 0.5
        
        de_all[:n_tasks_ge] += dveff1_0_ge + dveff1_1_ge

    if n_tasks_ee > 0:
        fxcz1_ee = _contract_xc_kernel_batched(
            td_nac, mf.xc, z1aoS_ee, None, None, False, False, True)[0]
        
        veff1_1_batch_ee = f1ooIJ[:, 1:] + fxcz1_ee[:, 1:]
        veff1_0_batch_ee = cp.repeat(vxc1_ee[None, 1:], n_tasks_ee, axis=0)
        
        dveff1_0_ee = contract_veff_dm_batched(mol, veff1_0_batch_ee, dmz1doo_ee, hermi=0)
        oo0_batch_ee = cp.repeat(oo0[None, ...], n_tasks_ee, axis=0)
        dveff1_1_ee = contract_veff_dm_batched(mol, veff1_1_batch_ee, oo0_batch_ee, hermi=1) * 0.5
        
        de_all[n_tasks_ge:] += dveff1_0_ee + dveff1_1_ee

    t_debug_6 = log.timer_silent(*time0)[2]

    results = {}
    E_stack_cpu = E_stack.get()

    if calc_ge:
        xIao_ge = _c_mat_cT(orbo, X_stack.transpose(0, 2, 1), orbv)
        yIao_ge = _c_mat_cT(orbv, Y_stack, orbo)

        dsxy_x = contract_h1e_dm_asym_batched(mol, s1, xIao_ge * E_stack[:, None, None]) * 2.0
        dsxy_y = contract_h1e_dm_asym_batched(mol, s1, yIao_ge * E_stack[:, None, None]) * 2.0
        dsxy_ge = dsxy_x + dsxy_y

        dsxy_etf_x = contract_h1e_dm_batched(mol, s1, xIao_ge * E_stack[:, None, None])
        dsxy_etf_y = contract_h1e_dm_batched(mol, s1, yIao_ge * E_stack[:, None, None])
        dsxy_etf_ge = dsxy_etf_x + dsxy_etf_y

        base_de_ge = de_all[:n_tasks_ge]
        de_etf_ge_val = (base_de_ge + dsxy_etf_ge).get()
        de_ge_val = (base_de_ge + dsxy_ge).get()

        for local_idx in range(n_tasks_ge):
            E_val = E_stack_cpu[local_idx]
            results[local_idx] = {
                'de': de_ge_val[local_idx],
                'de_scaled': de_ge_val[local_idx] / E_val,
                'de_etf': de_etf_ge_val[local_idx],
                'de_etf_scaled': de_etf_ge_val[local_idx] / E_val
            }

    if n_tasks_ee > 0:
        base_de_ee = de_all[n_tasks_ge:]
        
        if has_grad:
            n_pairs -= 1
            de_grad = base_de_ee[-1].get() + mf_grad.grad_nuc(mol)
            results['gradient'] = de_grad

        if n_pairs > 0:
            dE_pairs = dE[:n_pairs]
            dE_cpu = dE_pairs.get()
            
            rIJoo_ao = _c_mat_cT(orbo, rIJoo[:n_pairs], orbo) * 2.0
            rIJvv_ao = _c_mat_cT(orbv, rIJvv[:n_pairs], orbv) * 2.0
            TIJoo_ao = _c_mat_cT(orbo, TIJoo[:n_pairs], orbo) * 2.0
            TIJvv_ao = _c_mat_cT(orbv, TIJvv[:n_pairs], orbv) * 2.0

            dsxy_ee = contract_h1e_dm_batched(mol, s1, rIJoo_ao * dE_pairs[:, None, None], hermi=1) * 0.5
            dsxy_ee += contract_h1e_dm_batched(mol, s1, rIJvv_ao * dE_pairs[:, None, None], hermi=1) * 0.5
            
            dsxy_etf_ee = contract_h1e_dm_batched(mol, s1, TIJoo_ao * dE_pairs[:, None, None], hermi=1) * 0.5
            dsxy_etf_ee += contract_h1e_dm_batched(mol, s1, TIJvv_ao * dE_pairs[:, None, None], hermi=1) * 0.5

            de_pairs = base_de_ee[:n_pairs]
            de_etf_ee_val = (de_pairs + dsxy_etf_ee).get()
            de_ee_val = (de_pairs + dsxy_ee).get()

            for k in range(n_pairs):
                i = idx_i[k]
                j = idx_j[k]
                results[(int(i), int(j))] = {
                    'de': de_ee_val[k],
                    'de_scaled': de_ee_val[k] / dE_cpu[k],
                    'de_etf': de_etf_ee_val[k],
                    'de_etf_scaled': de_etf_ee_val[k] / dE_cpu[k]
                }

    t_debug_7 = log.timer_silent(*time0)[2]
    if log.verbose >= logger.DEBUG:
        time_list = [0, t_debug_1, t_debug_2, t_debug_3, t_debug_4, t_debug_5, t_debug_6, t_debug_7]
        time_list = [time_list[idx] - time_list[idx-1] for idx in range(1, len(time_list))]
        for idx, t in enumerate(time_list):
            logger.note(td_nac, f"Time for step {idx}: {t*1e-3:.6f}s")
            
    return results


class NAC_multistates(NAC_multistates_tdrks):

    _keys = {'ris_zvector_solver'}

    def __init__(self, td):
        super().__init__(td)
        self.ris_zvector_solver = False

    def kernel(self, states=None, singlet=None, atmlst=None, grad_state=None):

        logger.warn(self, "NAC Multi-State Module with RIS Approximation (Experimental)")

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

        if self.grad_state is not None and self.grad_state not in target_states:
            raise ValueError(f"grad_state {self.grad_state} is requested, ",
                "but it is not within the provided target states {target_states} for NACV calculation.")

        self.results = {}

        has_ground = (0 in target_states)
        excited_states = [s for s in target_states if s > 0]

        calc_ee = len(excited_states) >= 2
        calc_ge = has_ground and len(excited_states) > 0

        if calc_ee or calc_ge:
            logger.info(self, f"Computing Unified Vectorized NACV using RIS for states: {target_states}")

            x_list, y_list, E_list = [], [], []
            for s in excited_states:
                x_list.append(rescale_spin_free_amplitudes(self.base.xy, s-1)[0])
                y_list.append(rescale_spin_free_amplitudes(self.base.xy, s-1)[1])
                E_list.append(self.base.energies[s-1]/HARTREE2EV)

            grad_idx = None
            if self.grad_state is not None and self.grad_state > 0:
                grad_idx = excited_states.index(self.grad_state)

            all_results = get_nacv_multi(
                self, x_list, y_list, E_list, 
                singlet=singlet, calc_ge=calc_ge, calc_ee=calc_ee,
                atmlst=atmlst, verbose=self.verbose, grad_state_idx=grad_idx
            )

            if 'gradient' in all_results:
                self.grad_result = all_results.pop('gradient')

            for key, res in all_results.items():
                if isinstance(key, int):
                    global_s = excited_states[key]
                    self.results[(0, global_s)] = res
                else:
                    local_i, local_j = key
                    global_i = excited_states[local_i]
                    global_j = excited_states[local_j]
                    self.results[(global_i, global_j)] = res

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
