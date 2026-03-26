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

from functools import reduce
import cupy as cp
import numpy as np
from pyscf import lib, gto
from pyscf import __config__
from gpu4pyscf.lib import logger
from pyscf.grad import rhf as rhf_grad_cpu
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.df import int3c2e
from gpu4pyscf.df.df_jk import (
    _tag_factorize_dm, _DFHF, _make_factorized_dm, _aggregate_dm_factor_l,
    _transpose_dm)
from gpu4pyscf.lib.cupy_helper import contract, tag_array
from gpu4pyscf.scf import cphf
from gpu4pyscf.lib import utils
from gpu4pyscf.grad import tdrhf as tdrhf_grad


def contract_h1e_dm_batched(mol, h1e, dms, hermi=0):
    assert h1e.ndim == 3
    assert dms.ndim == 3

    n_batch, nao, _ = dms.shape
    natm = mol.natm

    ao_loc = mol.ao_loc
    dims = ao_loc[1:] - ao_loc[:-1]
    atm_id_for_ao = np.repeat(mol._bas[:, gto.ATOM_OF], dims)
    atm_id_for_ao = cp.asarray(atm_id_for_ao)

    de_partial = cp.einsum('xij, nji -> nix', h1e, dms).real

    if hermi != 1:
        de_partial += cp.einsum('xij, nij -> nix', h1e, dms).real

    atm_ids = cp.arange(natm)[:, None]
    mask = (atm_ids == atm_id_for_ao[None, :])
    mask = mask.astype(de_partial.dtype)

    de = cp.einsum('ai, nix -> nax', mask, de_partial)

    if hermi == 1:
        de *= 2

    return de


def contract_h1e_dm_asym_batched(mol, h1e, dm_batch):
    natm = mol.natm
    de_part = cp.einsum('xuv, nvu -> nux', h1e, dm_batch).real

    ao_loc = mol.ao_loc
    dims = ao_loc[1:] - ao_loc[:-1]
    atm_id_for_ao = cp.asarray(np.repeat(mol._bas[:, gto.ATOM_OF], dims))
    atm_ids = cp.arange(natm)[:, None]
    mask = (atm_ids == atm_id_for_ao[None, :]).astype(de_part.dtype)

    return cp.einsum('au, nux -> nax', mask, de_part)


def get_nacv_multi(td_nac, x_list, y_list, E_list, singlet=True, ge_targets=None, 
        ee_pairs=None, grad_state_idx=None, atmlst=None, verbose=logger.INFO):
    """
    Unified function to calculate Non-Adiabatic Coupling Vectors (NACV) 
    for selective Ground-Excited (GE) and Excited-Excited (EE) targets.
    It batches only the required RHS structures into a single Z-Vector solve.
    """
    if not singlet:
        raise NotImplementedError('Only supports for singlet states')
    log = logger.new_logger(td_nac, verbose)
    time0 = log.init_timer()

    mol = td_nac.mol
    natm = mol.natm
    mf = td_nac.base._scf
    mf_grad = mf.nuc_grad_method()

    if getattr(mf, 'with_solvent', None) is not None:
        raise NotImplementedError('NACv gradient calculation is not supported for solvent models')

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

    ge_targets = ge_targets or []
    ee_pairs = ee_pairs or []

    n_tasks_ge = len(ge_targets)
    idx_i = [p[0] for p in ee_pairs]
    idx_j = [p[1] for p in ee_pairs]
    
    has_grad = grad_state_idx is not None
    if has_grad:
        idx_i.append(grad_state_idx)
        idx_j.append(grad_state_idx)
    n_tasks_ee = len(idx_i)
    n_pairs = len(ee_pairs)
    
    total_tasks = n_tasks_ge + n_tasks_ee
    if total_tasks == 0:
        return {}

    rhs_list = []

    # GE RHS
    if n_tasks_ge > 0:
        LI = X_stack[ge_targets] - Y_stack[ge_targets]
        rhs_ge = -LI * E_stack[ge_targets, None, None]
        rhs_list.append(rhs_ge)

    # EE / Gradient RHS
    if n_tasks_ee > 0:
        xI, yI = X_stack[idx_i], Y_stack[idx_i]
        xJ, yJ = X_stack[idx_j], Y_stack[idx_j]

        EI, EJ = E_stack[idx_i], E_stack[idx_j]
        dE = EJ - EI
        if has_grad: 
            dE[-1] = 1.0 # Effective denominator for gradient evaluation

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

        if not isinstance(mf, _DFHF):
            dm = cp.vstack([dmxpy_stack + dmxpy_stack.transpose(0,2,1), dmzooIJ])
            vj, vk = mf.get_jk(mol, dm, hermi=1)
            vj, vj0IJ = vj[:n_states], vj[n_states:]
            vk_sym, vk0IJ = vk[:n_states], vk[n_states:]
            vk_asym = mf.get_k(mol, dmxmy_stack - dmxmy_stack.transpose(0,2,1), hermi=2)
        else:
            vj0IJ, vk0IJ = mf.get_jk(mol, _tag_factorize_dm(dmzooIJ, hermi=1), hermi=1)
            vj, vk_sym = mf.get_jk(mol, dmxpy_stack, hermi=0)
            if is_tda: 
                vk_asym = vk_sym
            else: 
                vk_asym = mf.get_k(mol, dmxmy_stack, hermi=0)
            vj *= 2
            vk_sym = vk_sym + vk_sym.transpose(0,2,1)
            vk_asym = vk_asym - vk_asym.transpose(0,2,1)

        vj1I = vj[idx_i]
        vj1J = vj[idx_j]
        vk1I = vk_sym[idx_i]
        vk1J = vk_sym[idx_j]
        vk2I = vk_asym[idx_i]
        vk2J = vk_asym[idx_j]

        if has_grad and not singlet:
            vj1I[-1] = vj1J[-1] = 0.

        def trans_veff_batch(veff_batch):
            return _cT_mat_c(mo_coeff, veff_batch, mo_coeff)

        veff0doo = vj0IJ * 2 - vk0IJ
        wvo = cp.einsum('pi, npq, qj -> nij', orbv, veff0doo, orbo) * 2.0

        veffI = (vj1I * 2 - vk1I) * 0.5
        veff0mopI = trans_veff_batch(veffI)
        wvo -= contract('nki, nai -> nak', veff0mopI[:, :nocc, :nocc], xpyJ) * 2.0
        wvo += contract('nac, nai -> nci', veff0mopI[:, nocc:, nocc:], xpyJ) * 2.0

        veffJ = (vj1J * 2 - vk1J) * 0.5
        veff0mopJ = trans_veff_batch(veffJ)
        wvo -= contract('nki, nai -> nak', veff0mopJ[:, :nocc, :nocc], xpyI) * 2.0
        wvo += contract('nac, nai -> nci', veff0mopJ[:, nocc:, nocc:], xpyI) * 2.0

        veffI = -vk2I * 0.5
        veff0momI = trans_veff_batch(veffI)
        wvo -= contract('nki, nai -> nak', veff0momI[:, :nocc, :nocc], xmyJ) * 2.0
        wvo += contract('nac, nai -> nci', veff0momI[:, nocc:, nocc:], xmyJ) * 2.0

        veffJ = -vk2J * 0.5
        veff0momJ = trans_veff_batch(veffJ)
        wvo -= contract('nki, nai -> nak', veff0momJ[:, :nocc, :nocc], xmyI) * 2.0
        wvo += contract('nac, nai -> nci', veff0momJ[:, nocc:, nocc:], xmyI) * 2.0

        rhs_list.append(wvo)

    rhs_all = cp.concatenate(rhs_list, axis=0)
    t_debug_1 = log.timer_silent(*time0)[2]

    vresp = td_nac.base.gen_response(singlet=None, hermi=1)
    z1_all = _solve_zvector(td_nac, rhs_all, vresp)
    # for i in range(z1_all.shape[0]):
    #     z1_all[i] = _solve_zvector(td_nac, rhs_all[i][None, :, :], vresp)
    t_debug_2 = log.timer_silent(*time0)[2]


    dmz1doo_list = []
    W_ao_list = []
    offset = 0

    oo0 = _make_factorized_dm(orbo*2, orbo, symmetrize=0)

    if n_tasks_ge > 0:
        z1_ge = z1_all[offset:offset+n_tasks_ge]
        offset += n_tasks_ge

        z1aoS_ge = _make_factorized_dm(contract('ua,nai->nui', orbv, z1_ge), orbo, symmetrize=1)
        GZS_mo_ge = _cT_mat_c(mo_coeff, vresp(z1aoS_ge), mo_coeff)

        W_ge = cp.zeros((n_tasks_ge, nmo, nmo))
        W_ge[:, :nocc, :nocc] = GZS_mo_ge[:, :nocc, :nocc]
        zeta0 = z1_ge * mo_energy[nocc:][None, :, None]
        W_ge[:, :nocc, nocc:] = (GZS_mo_ge[:, :nocc, nocc:] 
            + 0.5 * Y_stack[ge_targets].transpose(0, 2, 1) * E_stack[ge_targets, None, None] + 0.5 * zeta0.transpose(0, 2, 1))
        zeta1 = z1_ge * mo_energy[None, None, :nocc]
        W_ge[:, nocc:, :nocc] = 0.5 * X_stack[ge_targets] * E_stack[ge_targets, None, None] + 0.5 * zeta1

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

    dms_tasks, j_factor, k_factor = [], [], []

    if n_tasks_ge > 0:
        for k in range(n_tasks_ge):
            dms_tasks.append([_tag_factorize_dm(P_all[k], hermi=1), oo0])
            j_factor.append(1.0)
            k_factor.append(1.0)

    if n_tasks_ee > 0:
        dmxpy_stack = _dms_to_list(dmxpy_stack)
        dmxmy_stack = _dms_to_list(dmxmy_stack)
        
        offset_ee = n_tasks_ge
        dmz1doo_ee = P_all[offset_ee:]

        if not is_tda:
            j_factor.extend([1., 2., 0.] * n_tasks_ee)
            k_factor.extend([1., 2.,-2.] * n_tasks_ee)
            for k, (I, J) in enumerate(zip(idx_i, idx_j)):
                dms_tasks.extend([
                    [_tag_factorize_dm(dmz1doo_ee[k], hermi=1), oo0],
                    [dmxpy_stack[I], dmxpy_stack[J] + dmxpy_stack[J].T],
                    [dmxmy_stack[I], dmxmy_stack[J] - dmxmy_stack[J].T]
                ])
            if has_grad:
                if not singlet: j_factor[-2] = 0.
                dms_tasks[-3][0] = _tag_factorize_dm(dmz1doo_ee[-1] - cp.asarray(oo0)*.5, hermi=1)
        else:
            j_factor.extend([1., 4.] * n_tasks_ee)
            k_factor.extend([1., 4.] * n_tasks_ee)
            for k, (I, J) in enumerate(zip(idx_i, idx_j)):
                dms_tasks.extend([
                    [_tag_factorize_dm(dmz1doo_ee[k], hermi=1), oo0],
                    [dmxpy_stack[I], _transpose_dm(dmxpy_stack[J])]
                ])
            if has_grad:
                if not singlet: j_factor[-1] = 0.
                dms_tasks[-2][0] = _tag_factorize_dm(dmz1doo_ee[-1] - cp.asarray(oo0)*.5, hermi=1)

    t_debug_4 = log.timer_silent(*time0)[2]

    ejk_all = td_nac.jk_energies_per_atom(dms_tasks, j_factor, k_factor, sum_results=False)
    ejk_all_raw = cp.asarray(ejk_all)
    
    de_all = dh_td_all - ds_all + dh1e_td_all
    
    if n_tasks_ge > 0:
        de_all[:n_tasks_ge] += ejk_all_raw[:n_tasks_ge] * 2.0
        
    if n_tasks_ee > 0:
        ejk_ee = ejk_all_raw[n_tasks_ge:]
        ejk_ee = ejk_ee.reshape(n_tasks_ee, -1, natm, 3).sum(axis=1) * 2.0
        de_all[n_tasks_ge:] += ejk_ee

    t_debug_5 = log.timer_silent(*time0)[2]

    results = {}
    offset = 0
    E_stack_cpu = E_stack.get()

    if n_tasks_ge > 0:
        xIao_ge = _c_mat_cT(orbo, X_stack[ge_targets].transpose(0, 2, 1), orbv)
        yIao_ge = _c_mat_cT(orbv, Y_stack[ge_targets], orbo)

        dsxy_x = contract_h1e_dm_asym_batched(mol, s1, xIao_ge * E_stack[ge_targets, None, None]) * 2.0
        dsxy_y = contract_h1e_dm_asym_batched(mol, s1, yIao_ge * E_stack[ge_targets, None, None]) * 2.0
        dsxy_ge = dsxy_x + dsxy_y

        dsxy_etf_x = contract_h1e_dm_batched(mol, s1, xIao_ge * E_stack[ge_targets, None, None])
        dsxy_etf_y = contract_h1e_dm_batched(mol, s1, yIao_ge * E_stack[ge_targets, None, None])
        dsxy_etf_ge = dsxy_etf_x + dsxy_etf_y

        base_de_ge = de_all[:n_tasks_ge]
        de_etf_ge_val = (base_de_ge + dsxy_etf_ge).get()
        de_ge_val = (base_de_ge + dsxy_ge).get()

        for k, local_idx in enumerate(ge_targets):
            E_val = E_stack_cpu[local_idx]
            results[local_idx] = {
                'de': de_ge_val[k],
                'de_scaled': de_ge_val[k] / E_val,
                'de_etf': de_etf_ge_val[k],
                'de_etf_scaled': de_etf_ge_val[k] / E_val
            }
        offset += n_tasks_ge

    if n_tasks_ee > 0:
        base_de_ee = de_all[n_tasks_ge:]
        
        if has_grad:
            n_tasks_ee -= 1  # Excluding the gradient state from ETF calculations
            de_grad = base_de_ee[-1].get() + mf_grad.grad_nuc(mol, atmlst)
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

    t_debug_6 = log.timer_silent(*time0)[2]
    if log.verbose >= logger.DEBUG:
        time_list = [0, t_debug_1, t_debug_2, t_debug_3, t_debug_4, t_debug_5, t_debug_6]
        time_list = [time_list[idx] - time_list[idx-1] for idx in range(1, len(time_list))]
        for idx, t in enumerate(time_list):
            print(f"Time for step {idx}: {t*1e-3:.6f}s")
            
    return results


def _solve_zvector(td_nac, rhs, vresp):
    mf = td_nac.base._scf
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    nvir, nocc = rhs.shape[-2:]
    orbo = mo_coeff[:, mo_occ > 0]
    orbv = mo_coeff[:, mo_occ == 0]

    def fvind(x_flat):
        n_vecs = x_flat.shape[0]
        x_batch = x_flat.reshape(n_vecs, nvir, nocc)
        x_batch = contract('ua,nai->nui', orbv, x_batch)
        dm = _make_factorized_dm(x_batch, orbo*2, symmetrize=1)
        v1ao = vresp(dm)
        resp_mo = contract('nuv,vi->nui', v1ao, orbo, out=x_batch)
        resp_mo = cp.einsum('ua,nui->nai', orbv, resp_mo)
        return resp_mo.reshape(n_vecs, -1)

    z1 = cphf.solve(
        fvind, mo_energy, mo_occ, rhs,
        max_cycle=td_nac.cphf_max_cycle,
        tol=td_nac.cphf_conv_tol
    )[0]
    return z1.reshape(-1, nvir, nocc)

def _c_mat_cT(a, b, c):
    if a.shape[1] > c.shape[1]:
        return contract('nui,vi->nuv', contract('ua,nai->nui', a, b), c)
    else:
        return contract('ui,niv->nuv', a, contract('nia,va->niv', b, c))

def _cT_mat_c(a, b, c):
    if a.shape[1] > c.shape[1]:
        return contract('pa,npi->nai', a, contract('npq,qi->npi', b, c))
    else:
        return contract('niq,qa->nia', contract('pi,npq->niq', a, b), c)

def _aggregate_dms_factor_l(dms):
    factor_l = cp.vstack([x.factor_l for x in dms])
    factor_r = dms[0].factor_r
    assert all(x.symmetrize == 0 for x in dms)
    return tag_array(cp.stack(dms), factor_l=factor_l, factor_r=factor_r,
                     symmetrize=0)

def _dms_to_list(dms):
    dms_list = []
    factor_l = dms.factor_l
    factor_r = dms.factor_r
    if factor_r.ndim < factor_l.ndim:
        factor_r = [factor_r] * len(factor_l)
    symmetrize = dms.symmetrize
    for i, dm in enumerate(dms):
        dm.factor_l = factor_l[i]
        dm.factor_r = factor_r[i]
        dm.symmetrize = symmetrize
        dms_list.append(dm)
    return dms_list


class NAC_multistates(lib.StreamObject):

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
        "results",
        "grad_state",
        "grad_result",
        "target_state"
    }

    def __init__(self, td):
        self.verbose = td.verbose
        self.stdout = td.stdout
        self.mol = td.mol
        self.base = td
        self.states = [1, 2]  # Default to pair (1, 2)
        self.atmlst = None
        self.results = {}     # Dictionary to store results: {(i, j): {data}}

        self.grad_state = None  # Add indicator for the specific state to evaluate energy gradient
        self.grad_result = None
        
        self.target_state = None # the target state, between this state and given states will be calculated.

    _write = rhf_grad_cpu.GradientsBase._write

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
        if self.target_state is not None:
            log.info(f"Target State = {self.target_state} Coupled with States = {self.states}")
        else:
            log.info(f"States List = {self.states}")
        if self.grad_state is not None:
            log.info(f"Computing Energy Gradient for State = {self.grad_state}")
        log.info("\n")
        return self

    def get_nacv_multi(self, x_list, y_list, E_list, singlet=True, ge_targets=None, 
            ee_pairs=None, grad_state_idx=None, atmlst=None, verbose=logger.INFO):
        return get_nacv_multi(self, x_list, y_list, E_list, singlet=singlet, ge_targets=ge_targets, 
            ee_pairs=ee_pairs, grad_state_idx=grad_state_idx, atmlst=atmlst, verbose=verbose)

    def kernel(self, states=None, singlet=None, atmlst=None, grad_state=None, target_state=None):

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
            
        if target_state is not None:
            self.target_state = target_state

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        target_states = sorted(list(set(self.states)))
        if self.target_state is not None and len(target_states) < 1:
            raise ValueError("Must provide at least 1 state in 'states' when target_state is specified.")
        elif self.target_state is None and len(target_states) < 2:
            raise ValueError("Must provide at least 2 states for NACV calculation.")
            
        # Collect all required states to extract from td object
        fetch_states = set(target_states)
        if self.target_state is not None:
            fetch_states.add(self.target_state)
        if self.grad_state is not None:
            fetch_states.add(self.grad_state)
            
        fetch_states = sorted(list(fetch_states))
        
        if any(s < 0 for s in fetch_states):
            raise ValueError("State indices must be non-negative.")
        nstates = len(self.base.e)
        if any(s > nstates for s in fetch_states):
            raise ValueError(f"State index exceeds number of roots ({nstates}).")

        self.results = {}

        has_ground = (0 in fetch_states)
        excited_states = [s for s in fetch_states if s > 0]
        
        if len(excited_states) > 0:
            
            global2local = {s: i for i, s in enumerate(excited_states)}
            
            ge_targets = []
            ee_pairs = []
            
            if self.target_state is not None:
                t = self.target_state
                for s in target_states:
                    if t == s:
                        continue
                    if t == 0:
                        ge_targets.append(global2local[s])
                    elif s == 0:
                        ge_targets.append(global2local[t])
                    else:
                        i, j = global2local[t], global2local[s]
                        ee_pairs.append((min(i, j), max(i, j)))
            else:
                if has_ground:
                    # Original default: compute GE against all requested states
                    ge_targets = [global2local[s] for s in target_states if s > 0]
                    # Original default: compute all combinations within requested target_states
                local_ee_targets = [global2local[s] for s in target_states if s > 0]
                for i in range(len(local_ee_targets)):
                    for j in range(i + 1, len(local_ee_targets)):
                        ee_pairs.append((local_ee_targets[i], local_ee_targets[j]))
                        
            # Keep unique tasks
            ge_targets = sorted(list(set(ge_targets)))
            ee_pairs = sorted(list(set(ee_pairs)))

            if len(ee_pairs) > 0 or len(ge_targets) > 0 or self.grad_state is not None:
                logger.info(self, f"Extracting Base States for Vectorized NACV: {excited_states}")
                
                x_list, y_list, E_list = [], [], []
                for s in excited_states:
                    x_list.append(self.base.xy[s-1][0])
                    y_list.append(self.base.xy[s-1][1])
                    E_list.append(self.base.e[s-1])

                grad_idx = None
                if self.grad_state is not None and self.grad_state > 0:
                    grad_idx = global2local[self.grad_state]

                all_results = self.get_nacv_multi(
                    x_list, y_list, E_list, 
                    singlet=singlet, ge_targets=ge_targets, ee_pairs=ee_pairs,
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

    jk_energy_per_atom = tdrhf_grad.Gradients.jk_energy_per_atom
    jk_energies_per_atom = tdrhf_grad.Gradients.jk_energies_per_atom

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(self, "\n" + "="*60)
            logger.note(self, " NACV Calculation Summary")
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

    def solvent_response(self, dm):
        raise NotImplementedError("Solvent response is not yet implemented.")

    def reset(self, mol):
        self.base.reset(mol)
        self.mol = mol
        return self

    def as_scanner(nacv_instance, states=None):
        raise NotImplementedError("Multi-state NAC scanner is not yet implemented.")

    @classmethod
    def from_cpu(cls, method):
        td = method.base.to_gpu()
        out = cls(td)
        out.cphf_max_cycle = method.cphf_max_cycle
        out.cphf_conv_tol = method.cphf_conv_tol
        return out