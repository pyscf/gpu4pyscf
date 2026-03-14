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


def get_nacv_ge_multi(td_nac, x_list, y_list, E_list, singlet=True, atmlst=None, verbose=logger.INFO):
    """
    Calculate Non-Adiabatic Coupling Vectors (NACV) between Ground State (0)
    and multiple Excited States simultaneously in a batched manner.
    """
    if singlet is False:
        raise NotImplementedError('Only supports for singlet states')
    log = logger.new_logger(td_nac, verbose)
    time0 = log.init_timer()

    mol = td_nac.mol
    mf = td_nac.base._scf
    mf_grad = mf.nuc_grad_method()

    if getattr(mf, 'with_solvent', None) is not None:
        raise NotImplementedError('NACv gradient calculation is not supported for solvent models')

    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    nao, nmo = mo_coeff.shape
    nocc = int((mo_occ > 0).sum())
    nvir = nmo - nocc
    orbv = mo_coeff[:, nocc:]
    orbo = mo_coeff[:, :nocc]

    n_states = len(E_list)

    X_stack = cp.asarray(x_list).reshape(n_states, nocc, nvir).transpose(0, 2, 1)
    if not isinstance(y_list[0], (np.ndarray, cp.ndarray)):
        Y_stack = cp.zeros_like(X_stack)
    else:
        Y_stack = cp.asarray(y_list).reshape(n_states, nocc, nvir).transpose(0, 2, 1)
    E_stack = cp.asarray(E_list)

    LI = X_stack - Y_stack
    t_debug_1 = log.timer_silent(*time0)[2]
    vresp = td_nac.base.gen_response(singlet=None, hermi=1)

    def fvind(x_flat):
        n_vecs = x_flat.shape[0]
        x_batch = x_flat.reshape(n_vecs, nvir, nocc)
        x_batch = contract('ua,nai->nui', orbv, x_batch)
        dm = _make_factorized_dm(x_batch, orbo*2, symmetrize=1)
        v1ao = vresp(dm)
        resp_mo = contract('nuv,vi->nui', v1ao, orbo, out=x_batch)
        resp_mo = cp.einsum('ua,nui->nai', orbv, resp_mo)
        return resp_mo.reshape(n_vecs, -1)

    rhs = (-LI * E_stack[:, None, None])
    rhs = cp.ascontiguousarray(rhs)
    # z1_flat = cp.zeros((n_states, nvir, nocc))
    # for istate in range(n_states):
    #     z1_flat[istate] = cphf.solve(
    #         fvind,
    #         mo_energy,
    #         mo_occ,
    #         rhs[istate],
    #         max_cycle=td_nac.cphf_max_cycle,
    #         tol=td_nac.cphf_conv_tol
    #     )[0]
    z1_flat = cphf.solve(
        fvind,
        mo_energy,
        mo_occ,
        rhs,
        max_cycle=td_nac.cphf_max_cycle,
        tol=td_nac.cphf_conv_tol
    )[0]
    t_debug_2 = log.timer_silent(*time0)[2]

    z1 = z1_flat.reshape(n_states, nvir, nocc)
    #:z1ao = cp.einsum('ua, nai, vi -> nuv', orbv, z1, orbo) * 2.0
    #:z1aoS = (z1ao + z1ao.transpose(0, 2, 1)) * 0.5
    z1aoS = _make_factorized_dm(
        contract('ua,nai->nui', orbv, z1), orbo, symmetrize=1)

    GZS = vresp(z1aoS)
    GZS_mo = _cT_mat_c(mo_coeff, GZS, mo_coeff)

    W = cp.zeros((n_states, nmo, nmo))

    W[:, :nocc, :nocc] = GZS_mo[:, :nocc, :nocc]

    zeta0 = z1 * mo_energy[nocc:][None, :, None]
    W[:, :nocc, nocc:] = GZS_mo[:, :nocc, nocc:] \
                       + 0.5 * Y_stack.transpose(0, 2, 1) * E_stack[:, None, None] \
                       + 0.5 * zeta0.transpose(0, 2, 1)

    zeta1 = z1 * mo_energy[:nocc][None, None, :]
    W[:, nocc:, :nocc] = 0.5 * X_stack * E_stack[:, None, None] + 0.5 * zeta1

    #:W_ao = cp.einsum('up, npq, vq -> nuv', mo_coeff, W, mo_coeff) * 2.0
    W_ao = _c_mat_cT(mo_coeff, W, mo_coeff) * 2.0

    dmz1doo = z1aoS
    td_nac._dmz1doo = dmz1doo
    oo0 = _make_factorized_dm(orbo*2, orbo, symmetrize=0)
    t_debug_3 = log.timer_silent(*time0)[2]

    h1 = cp.asarray(mf_grad.get_hcore(mol))
    s1 = cp.asarray(mf_grad.get_ovlp(mol))

    dh_td = contract_h1e_dm_batched(mol, h1, dmz1doo, hermi=1)
    ds = contract_h1e_dm_batched(mol, s1, W_ao, hermi=0)

    dh1e_td_list = []
    for k in range(n_states):
        dh1e_k = int3c2e.get_dh1e(mol, dmz1doo[k])
        if len(mol._ecpbas) > 0:
            dh1e_k += rhf_grad.get_dh1e_ecp(mol, dmz1doo[k])
        dh1e_td_list.append(dh1e_k)
    dh1e_td = cp.array(dh1e_td_list)
    t_debug_4 = log.timer_silent(*time0)[2]
    if mol._pseudo:
        raise NotImplementedError("Pseudopotential gradient not supported for molecular system yet")

    dms_tasks = [[dmz1doo[k], oo0] for k in range(n_states)]
    j_tasks = [1.] * n_states
    k_tasks = [1.] * n_states

    if getattr(td_nac, 'jk_energies_per_atom', None) is None:
        raise NotImplementedError("jk_energies_per_atom is not implemented for TDRHF.")

    ejk_all = td_nac.jk_energies_per_atom(dms_tasks, j_tasks, k_tasks, sum_results=False)
    ejk_all = cp.asarray(ejk_all) * 2.0
    t_debug_5 = log.timer_silent(*time0)[2]

    de = dh_td - ds + ejk_all

    #:xIao = cp.einsum('ui, nia, va -> nuv', orbo, X_stack.transpose(0, 2, 1), orbv)
    #:yIao = cp.einsum('ua, nai, vi -> nuv', orbv, Y_stack, orbo)
    xIao = _c_mat_cT(orbo, X_stack.transpose(0, 2, 1), orbv)
    yIao = _c_mat_cT(orbv, Y_stack, orbo)

    dsxy_x = contract_h1e_dm_asym_batched(mol, s1, xIao * E_stack[:, None, None]) * 2.0
    dsxy_y = contract_h1e_dm_asym_batched(mol, s1, yIao * E_stack[:, None, None]) * 2.0
    dsxy = dsxy_x + dsxy_y

    dsxy_etf_x = contract_h1e_dm_batched(mol, s1, xIao * E_stack[:, None, None])
    dsxy_etf_y = contract_h1e_dm_batched(mol, s1, yIao * E_stack[:, None, None])
    dsxy_etf = dsxy_etf_x + dsxy_etf_y

    de += dh1e_td
    de_etf = de + dsxy_etf
    de += dsxy

    de = de.get()
    de_etf = de_etf.get()
    E_stack = E_stack.get()
    results = {}
    for local_idx in range(n_states):
        results[local_idx] = {
            'de': de[local_idx],
            'de_scaled': de[local_idx] / E_stack[local_idx],
            'de_etf': de_etf[local_idx],
            'de_etf_scaled': de_etf[local_idx] / E_stack[local_idx]
        }
    t_debug_6 = log.timer_silent(*time0)[2]
    if log.verbose >= logger.DEBUG:
        time_list = [0, t_debug_1, t_debug_2, t_debug_3, t_debug_4, t_debug_5, t_debug_6]
        time_list = [time_list[i] - time_list[i-1] for i in range(1, len(time_list))]
        for i, t in enumerate(time_list):
            print(f"Time for step {i}: {t*1e-3:.5f}s")
    return results


def get_nacv_ee_multi(td_nac, x_list, y_list, E_list, singlet=True, atmlst=None, verbose=logger.INFO, grad_state_idx=None):
    """
    Calculate Non-Adiabatic Coupling Vectors (NACV) for multiple excited-excited state pairs simultaneously.
    If grad_state_idx is provided, it bathes the TDHF gradient evaluation for that specific state alongside NACV.
    """
    if not singlet:
        raise NotImplementedError('Only supports for singlet states')
    log = logger.new_logger(td_nac, verbose)
    time0 = logger.init_timer(td_nac)

    mol = td_nac.mol
    natm = mol.natm
    mf = td_nac.base._scf

    if getattr(mf, 'with_solvent', None) is not None:
        raise NotImplementedError('NACv gradient calculation is not supported for solvent models')

    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)

    nao, nmo = mo_coeff.shape
    nocc = int((mo_occ > 0).sum())
    nvir = nmo - nocc

    orbv = mo_coeff[:, nocc:]
    orbo = mo_coeff[:, :nocc]

    n_states = len(E_list)

    is_tda = False
    X_stack = cp.asarray(x_list).reshape(n_states, nocc, nvir).transpose(0, 2, 1)
    if not isinstance(y_list[0], (np.ndarray, cp.ndarray)):
        Y_stack = cp.zeros_like(X_stack)
        is_tda = True
    else:
        Y_stack = cp.asarray(y_list).reshape(n_states, nocc, nvir).transpose(0, 2, 1)
    E_stack = cp.asarray(E_list)

    idx_i = []
    idx_j = []
    pairs = []
    for i in range(n_states):
        for j in range(i + 1, n_states):
            idx_i.append(i)
            idx_j.append(j)
            pairs.append((i, j))
    n_tasks = n_pairs = len(pairs)
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
    #:dmzooIJ = cp.einsum('ui, nij, vj -> nuv', orbo, TIJoo, orbo) * 2.0
    #:dmzooIJ += cp.einsum('ua, nab, vb -> nuv', orbv, TIJvv, orbv) * 2.0
    dmzooIJ = _c_mat_cT(orbo, TIJoo, orbo) * 2.0
    dmzooIJ += _c_mat_cT(orbv, TIJvv, orbv) * 2.0

    t_debug_1 = log.timer_silent(*time0)[2]
    if not isinstance(mf, _DFHF):
        dm = cp.vstack([dmxpy_stack+dmxpy_stack.transpose(0,2,1), dmzooIJ])
        vj, vk = mf.get_jk(mol, dm, hermi=1)
        vj    , vj0IJ = vj[:n_states], vj[n_states:]
        vk_sym, vk0IJ = vk[:n_states], vk[n_states:]
        vk_asym = mf.get_k(mol, dmxmy_stack-dmxmy_stack.transpose(0,2,1), hermi=2)
    else:
        vj0IJ, vk0IJ = mf.get_jk(mol, _tag_factorize_dm(dmzooIJ, hermi=1), hermi=1)
        if is_tda:
            vj, vk = mf.get_jk(mol, dmxpy_stack, hermi=0)
            vk_sym = vk + vk.transpose(0,2,1)
            vk_asym = vk - vk.transpose(0,2,1)
        else:
            vj, vk = mf.get_jk(mol, dmxpy_stack, hermi=0)
            vk_sym = vk + vk.transpose(0,2,1)
            vk = mf.get_k(mol, dmxmy_stack, hermi=0)
            vk_asym = vk - vk.transpose(0,2,1)
        vj *= 2
    vj1I = vj[idx_i]
    vj1J = vj[idx_j]
    vk1I = vk_sym[idx_i]
    vk1J = vk_sym[idx_j]
    vk2I = vk_asym[idx_i]
    vk2J = vk_asym[idx_j]
    idx_i = idx_i[:-1]
    idx_j = idx_j[:-1]

    # Extract Gradient specific VJ/VK
    if grad_state_idx is not None:
        if not singlet:
            vj1I[-1] = vj1J[-1] = 0
    dm = vj = vk = vk_sym = vk_asym = None

    def trans_veff(veff, C):
        return reduce(cp.dot, (C.T, veff, C))

    def trans_veff_batch(veff_batch):
        return _cT_mat_c(mo_coeff, veff_batch, mo_coeff)

    # NACV Right Hand Side components
    veff0doo = vj0IJ * 2 - vk0IJ
    # TODO: Solvent response batching.
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

    rhs = wvo
    # rhs = (wvo / dE[:, None, None])
    vresp = td_nac.base.gen_response(singlet=None, hermi=1)
    t_debug_2 = log.timer_silent(*time0)[2]
    def fvind(x_flat):
        n_vecs = x_flat.shape[0]
        x_batch = x_flat.reshape(n_vecs, nvir, nocc)
        x_batch = contract('ua,nai->nui', orbv, x_batch)
        dm = _make_factorized_dm(x_batch, orbo*2, symmetrize=1)
        v1ao = vresp(dm)
        resp_mo = contract('nuv,vi->nui', v1ao, orbo, out=x_batch)
        resp_mo = cp.einsum('ua,nui->nai', orbv, resp_mo)
        return resp_mo.reshape(n_vecs, -1)

    # if grad_state_idx is not None:
    #     ndim = n_pairs + 1
    # else:
    #     ndim = n_pairs
    # z1_flat = cp.zeros((ndim, nvir, nocc))
    # for ipair in range(n_pairs):
    #     z1_flat[ipair] = cphf.solve(
    #         fvind,
    #         mo_energy,
    #         mo_occ,
    #         rhs[ipair],
    #         max_cycle=td_nac.cphf_max_cycle,
    #         tol=td_nac.cphf_conv_tol
    #     )[0]
    # if grad_state_idx is not None:
    #     z1_flat[-1] = cphf.solve(
    #         fvind,
    #         mo_energy,
    #         mo_occ,
    #         rhs[-1],
    #         max_cycle=td_nac.cphf_max_cycle,
    #         tol=td_nac.cphf_conv_tol
    #     )[0]

    z1 = cphf.solve(
        fvind,
        mo_energy,
        mo_occ,
        rhs,
        max_cycle=td_nac.cphf_max_cycle,
        tol=td_nac.cphf_conv_tol
    )[0]
    t_debug_3 = log.timer_silent(*time0)[2]
    z1 = z1.reshape(-1, nvir, nocc) / dE[:, None, None]

    z1ao_sym = _make_factorized_dm(
        contract('ua,nai->nui', orbv, z1), orbo, symmetrize=1)

    z1aoS = z1ao_sym * 0.5 * dE[:, None, None]
    dmz1doo = z1aoS + dmzooIJ # P matrix

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
    # This term does not contributes to gradients
    term_oo[:n_pairs] += rIJoo[:n_pairs].transpose(0, 2, 1) * dE[:n_pairs, None, None]
    im0[:, :nocc, :nocc] = term_oo

    # term_ov does not contributes to gradients
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
    # This term does not contributes to gradients
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
        im0_g[:nocc,nocc:] = 0
        im0_g[nocc:,:nocc] *= 2.
        # The energy weighted DM
        im0_g[:nocc,:nocc] += np.diag(mo_energy[:nocc]) * 2.

    im0_ao = _c_mat_cT(mo_coeff, im0, mo_coeff)

    t_debug_4 = log.timer_silent(*time0)[2]

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
    t_debug_5 = log.timer_silent(*time0)[2]

    dmxpy_stack = _dms_to_list(dmxpy_stack)
    dmxmy_stack = _dms_to_list(dmxmy_stack)

    if not is_tda:
        dms_tasks = []
        j_factor = [1., 2., 0.] * n_pairs
        k_factor = [1., 2.,-2.] * n_pairs

        for k, (I, J) in enumerate(zip(idx_i, idx_j)):
            dms_tasks.extend(
                [[_tag_factorize_dm(dmz1doo[k], hermi=1), oo0],
                 [dmxpy_stack[I], dmxpy_stack[J] + dmxpy_stack[J].T],
                 [dmxmy_stack[I], dmxmy_stack[J] - dmxmy_stack[J].T]]
            )
        if grad_state_idx is not None:
            dmz1doo_g = dmz1doo[-1] - oo0*.5
            _dmxpy = dmxpy_stack[grad_state_idx]
            _dmxmy = dmxmy_stack[grad_state_idx]
            dms_tasks.extend(
                [[_tag_factorize_dm(dmz1doo_g, hermi=1), oo0],
                 [_dmxpy, _dmxpy + _dmxpy.T],
                 [_dmxmy, _dmxmy - _dmxmy.T]]
            )
            k_factor.extend([1., 2.,-2.])
            j_factor.extend([1., 2., 0.])
            if not singlet:
                j_factor[-2] = 0.

        ejk = td_nac.jk_energies_per_atom(
            dms_tasks, j_factor, k_factor, sum_results=False)
        ejk = ejk.reshape(-1, 3, natm, 3).sum(axis=1) * 2

    else: # TDA
        dms_tasks = []
        j_factor = [1., 4.] * n_pairs
        k_factor = [1., 4.] * n_pairs

        for k, (I, J) in enumerate(zip(idx_i, idx_j)):
            dms_tasks.extend(
                [[_tag_factorize_dm(dmz1doo[k], hermi=1), oo0],
                 [dmxpy_stack[I], _transpose_dm(dmxpy_stack[J])]]
            )
        if grad_state_idx is not None:
            dmz1doo_g = dmz1doo[-1] - oo0*.5
            _dmxpy = dmxpy_stack[grad_state_idx]
            dms_tasks.extend(
                [[_tag_factorize_dm(dmz1doo_g, hermi=1), oo0],
                 [_dmxpy, _transpose_dm(_dmxpy)]]
            )
            k_factor.extend([1., 4.])
            j_factor.extend([1., 4.])
            if not singlet:
                j_factor[-1] = 0.

        ejk = td_nac.jk_energies_per_atom(
            dms_tasks, j_factor, k_factor, sum_results=False)
        ejk = ejk.reshape(-1, 2, natm, 3).sum(axis=1) * 2

    de += cp.asarray(ejk)
    t_debug_6 = log.timer_silent(*time0)[2]

    results = {}
    if grad_state_idx is not None:
        de, de_grad = de[:-1], de[-1].get()
        rIJoo = rIJoo[:-1]
        rIJvv = rIJvv[:-1]
        TIJoo = TIJoo[:-1]
        TIJvv = TIJvv[:-1]
        dE = dE[:-1]
        de_grad += mf_grad.grad_nuc(mol)
        results['gradient'] = de_grad

    rIJoo_ao = cp.einsum('ui, nij, vj -> nuv', orbo, rIJoo, orbo) * 2.0
    rIJvv_ao = cp.einsum('ua, nab, vb -> nuv', orbv, rIJvv, orbv) * 2.0

    TIJoo_ao = cp.einsum('ui, nij, vj -> nuv', orbo, TIJoo, orbo) * 2.0
    TIJvv_ao = cp.einsum('ua, nab, vb -> nuv', orbv, TIJvv, orbv) * 2.0

    dsxy = contract_h1e_dm_batched(mol, s1, rIJoo_ao * dE[:, None, None], hermi=1) * 0.5
    dsxy += contract_h1e_dm_batched(mol, s1, rIJvv_ao * dE[:, None, None], hermi=1) * 0.5

    dsxy_etf = contract_h1e_dm_batched(mol, s1, TIJoo_ao * dE[:, None, None], hermi=1) * 0.5
    dsxy_etf += contract_h1e_dm_batched(mol, s1, TIJvv_ao * dE[:, None, None], hermi=1) * 0.5

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

    t_debug_7 = log.timer_silent(*time0)[2]
    if log.verbose >= logger.DEBUG:
        time_list = [0, t_debug_1, t_debug_2, t_debug_3, t_debug_4, t_debug_5, t_debug_6, t_debug_7]
        time_list = [time_list[i+1] - time_list[i] for i in range(len(time_list) - 1)]
        for i, t in enumerate(time_list):
            print(f"Time for step {i}: {t*1e-3:.6f}s")
    return results

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
        "grad_result"
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
        log.info(f"States List = {self.states}")
        if self.grad_state is not None:
            log.info(f"Computing Energy Gradient for State = {self.grad_state}")
        log.info("\n")
        return self

    def get_nacv_ge_multi(self, x_list, y_list, E_list, singlet, atmlst=None, verbose=logger.INFO):
        return get_nacv_ge_multi(self, x_list, y_list, E_list, singlet, atmlst, verbose)

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
        nstates = len(self.base.e)
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
                x_list.append(self.base.xy[s-1][0])
                y_list.append(self.base.xy[s-1][1])
                E_list.append(self.base.e[s-1])

            grad_idx = None
            if self.grad_state is not None and self.grad_state > 0:
                grad_idx = excited_states.index(self.grad_state)

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
                x_ge_list.append(self.base.xy[s-1][0])
                y_ge_list.append(self.base.xy[s-1][1])
                E_ge_list.append(self.base.e[s-1])

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
