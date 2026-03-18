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
import os
import cupy as cp
import numpy as np
from pyscf import lib, gto
from pyscf import __config__
from pyscf.dft.numint import NumInt as numint_cpu
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, add_sparse
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import tdrks as tdrks_grad
from gpu4pyscf.df import int3c2e
from gpu4pyscf.df.df_jk import (
    _tag_factorize_dm, _DFHF, _make_factorized_dm, _aggregate_dm_factor_l,
    _transpose_dm)
from gpu4pyscf.scf import cphf
from gpu4pyscf import tdscf
from gpu4pyscf.nac.tdrhf_grad_nacv import NAC_multistates as NAC_multistates_hf
from gpu4pyscf.nac.tdrhf_grad_nacv import (
    contract_h1e_dm_batched, contract_h1e_dm_asym_batched, _solve_zvector,
    _c_mat_cT, _cT_mat_c, _aggregate_dms_factor_l, _dms_to_list)


def contract_veff_dm_batched(mol, veff_batch, dm_batch, hermi=0):
    """
    Contract 4D batched effective potential derivatives with batched density matrices.
    veff_batch: (n_batch, 3, nao, nao)
    dm_batch: (n_batch, nao, nao)
    """
    assert veff_batch.ndim == 4
    assert dm_batch.ndim == 3

    natm = mol.natm
    ao_loc = mol.ao_loc
    dims = ao_loc[1:] - ao_loc[:-1]
    atm_id_for_ao = cp.asarray(np.repeat(mol._bas[:, gto.ATOM_OF], dims))

    de_partial = cp.einsum('nxij, nji -> nix', veff_batch, dm_batch).real
    if hermi != 1:
        de_partial += cp.einsum('nxij, nij -> nix', veff_batch, dm_batch).real

    atm_ids = cp.arange(natm)[:, None]
    mask = (atm_ids == atm_id_for_ao[None, :]).astype(de_partial.dtype)

    de = cp.einsum('ai, nix -> nax', mask, de_partial)
    if hermi == 1:
        de *= 2.0
    return de


def _contract_xc_kernel_batched(td_grad, xc_code, dmvoI, dmvoJ=None, dmoo_batch=None,
            with_vxc=True, with_kxc=True, singlet=True, with_nac=False):
    """
    Batched version of _contract_xc_kernel to process multiple states simultaneously.
    dmvo_batch: (n_batch, nao, nao)
    dmoo_batch: (n_batch, nao, nao)
    dmvo_2_batch: (n_batch, nao, nao)
    """
    mol = td_grad.mol
    mf = td_grad.base._scf
    grids = mf.grids

    ni = mf._numint
    xctype = ni._xc_type(xc_code)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    opt = getattr(ni, "gdftopt", None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    _sorted_mol = opt._sorted_mol
    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[0])

    n_batch = dmvoI.shape[0]

    dmvoI = (dmvoI + dmvoI.transpose(0, 2, 1)) * 0.5
    dmvoI = opt.sort_orbitals(dmvoI, axis=[1, 2])
    f1voI = cp.zeros((n_batch, 4, nao, nao))
    deriv = 2

    if dmvoJ is not None:
        dmvoJ = (dmvoJ + dmvoJ.transpose(0, 2, 1)) * 0.5
        dmvoJ = opt.sort_orbitals(dmvoJ, axis=[1, 2])
        f1voJ = cp.zeros((n_batch, 4, nao, nao))
    else:
        f1voJ = None

    if dmoo_batch is not None:
        f1oo_batch = cp.zeros((n_batch, 4, nao, nao))
        dmoo_batch = opt.sort_orbitals(dmoo_batch, axis=[1, 2])
    else:
        f1oo_batch = None

    if with_vxc:
        v1ao_batch = cp.zeros((4, nao, nao))
    else:
        v1ao_batch = None

    if with_kxc:
        k1ao_batch = cp.zeros((n_batch, 4, nao, nao))
        deriv = 3
    else:
        k1ao_batch = None

    if xctype == "HF":
        return f1voI, f1voJ, f1oo_batch, v1ao_batch, k1ao_batch
    elif xctype == "LDA":
        fmat_, ao_deriv = tdrks_grad._lda_eval_mat_, 1
    elif xctype == "GGA":
        fmat_, ao_deriv = tdrks_grad._gga_eval_mat_, 2
    elif xctype == "MGGA":
        fmat_, ao_deriv = tdrks_grad._mgga_eval_mat_, 2
        logger.warn(td_grad, "TDRKS-MGGA Gradients may be inaccurate due to grids response")
    else:
        raise NotImplementedError(f"td-rks for functional {xc_code}")

    if (not td_grad.base.exclude_nlc) and mf.do_nlc():
        raise NotImplementedError("TDDFT gradient with NLC contribution is not supported yet.")

    for ao, mask, weight, coords in ni.block_loop(_sorted_mol, grids, nao, ao_deriv):
        ao0 = ao[0] if xctype == "LDA" else ao
        mo_coeff_mask = mo_coeff[mask, :]
        rho = ni.eval_rho2(_sorted_mol, ao0, mo_coeff_mask, mo_occ, mask, xctype, with_lapl=False)

        if not singlet:
            rho *= 0.5
            rho = cp.repeat(rho[cp.newaxis], 2, axis=0)

        if deriv > 2 and os.environ.get('LIBXC_ON_GPU', '0') != '1':
            ni_cpu = numint_cpu()
            vxc, fxc, kxc = ni_cpu.eval_xc_eff(xc_code, rho.get(), deriv, xctype=xctype)[1:]
            vxc, fxc, kxc = cp.asarray(vxc), cp.asarray(fxc), cp.asarray(kxc)
        else:
            vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[1:]

        # Pre-calculate Non-singlet coupling factors outside batch loop
        if not singlet:
            fxc_t = fxc[:, :, 0] - fxc[:, :, 1]
            fxc_t = fxc_t[0] - fxc_t[1]
            if dmoo_batch is not None:
                fxc_s = fxc[0, :, 0] + fxc[0, :, 1]
            vxc_s = vxc[0]
            if with_kxc:
                kxc_t = kxc[0, :, 0] - kxc[0, :, 1]
                kxc_t = kxc_t[:, :, 0] - kxc_t[:, :, 1]

        # v1ao evaluation
        if with_vxc:
            vxc_to_use = vxc if singlet else vxc_s
            fmat_(_sorted_mol, v1ao_batch, ao, vxc_to_use * weight, mask, shls_slice, ao_loc)

        # Process each state in the batch, this cannot be simplified
        for i in range(n_batch):
            dmvo_mask = dmvoI[i][mask[:, None], mask]
            rho1 = ni.eval_rho(_sorted_mol, ao0, dmvo_mask, mask, xctype, hermi=1, with_lapl=False)
            if singlet:
                rho1 *= 2.0  # *2 for alpha + beta
            if xctype == "LDA":
                rho1 = rho1[cp.newaxis].copy()

            if singlet:
                tmp = contract("yg, xyg -> xg", rho1, fxc)
            else:
                tmp = contract("yg, xyg -> xg", rho1, fxc_t)
            wv = contract("xg, g -> xg", tmp, weight)
            fmat_(_sorted_mol, f1voI[i], ao, wv, mask, shls_slice, ao_loc)

            if f1voJ is not None:
                dmvo_2_mask = dmvoJ[i][mask[:, None], mask]
                rho2 = ni.eval_rho(_sorted_mol, ao0, dmvo_2_mask, mask, xctype, hermi=1, with_lapl=False)
                if singlet:
                    rho2 *= 2.0
                if xctype == "LDA":
                    rho2 = rho2[cp.newaxis].copy()

                if singlet:
                    tmp = contract("yg, xyg -> xg", rho2, fxc)
                else:
                    tmp = contract("yg, xyg -> xg", rho2, fxc_t)
                wv = contract("xg, g -> xg", tmp, weight)
                fmat_(_sorted_mol, f1voJ[i], ao, wv, mask, shls_slice, ao_loc)

            # k1ao evaluation
            if with_kxc:
                if with_nac:
                    kxc_to_use = kxc if singlet else kxc_t
                    tmp = contract("yg, xyzg -> xzg", rho1, kxc_to_use)
                    tmp = contract("zg, xzg -> xg", rho2, tmp)
                else:
                    kxc_to_use = kxc if singlet else kxc_t
                    tmp = contract("yg, xyzg -> xzg", rho1, kxc_to_use)
                    tmp = contract("zg, xzg -> xg", rho1, tmp)
                wv = contract("xg, g -> xg", tmp, weight)
                fmat_(_sorted_mol, k1ao_batch[i], ao, wv, mask, shls_slice, ao_loc)

            # f1oo evaluation
            if dmoo_batch is not None:
                dmoo_mask = dmoo_batch[i][mask[:, None], mask]
                rho2 = ni.eval_rho(_sorted_mol, ao0, dmoo_mask, mask, xctype, hermi=1, with_lapl=False)
                if singlet:
                    rho2 *= 2.0
                if xctype == "LDA":
                    rho2 = rho2[cp.newaxis].copy()
                if singlet:
                    tmp = contract("yg, xyg -> xg", rho2, fxc)
                else:
                    tmp = contract("yg, xyg -> xg", rho2, fxc_s)
                wv = contract("xg, g -> xg", tmp, weight)
                fmat_(_sorted_mol, f1oo_batch[i], ao, wv, mask, shls_slice, ao_loc)

    f1voI[:, 1:] *= -1
    f1voI = opt.unsort_orbitals(f1voI, axis=[2, 3])

    if f1voJ is not None:
        f1voJ[:, 1:] *= -1
        f1voJ = opt.unsort_orbitals(f1voJ, axis=[2, 3])

    if f1oo_batch is not None:
        f1oo_batch[:, 1:] *= -1
        f1oo_batch = opt.unsort_orbitals(f1oo_batch, axis=[2, 3])

    if v1ao_batch is not None:
        v1ao_batch[1:] *= -1
        v1ao_batch = opt.unsort_orbitals(v1ao_batch, axis=[1, 2])

    if k1ao_batch is not None:
        k1ao_batch[:, 1:] *= -1
        k1ao_batch = opt.unsort_orbitals(k1ao_batch, axis=[2, 3])

    return f1voI, f1voJ, f1oo_batch, v1ao_batch, k1ao_batch


def get_nacv_multi(td_nac, x_list, y_list, E_list, singlet=True, ge_targets=None, 
    ee_pairs=None, grad_state_idx=None, atmlst=None, verbose=logger.INFO):
    """
    Unified function to calculate Non-Adiabatic Coupling Vectors (NACV) 
    for Ground-Excited (GE), Excited-Excited (EE), and energy gradients simultaneously.
    Designed for TDRKS with batched xc kernel evaluation.
    """
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

    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    with_k = ni.libxc.is_hybrid_xc(mf.xc)

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

        dmxpyI_batch = dmxpy_stack[idx_i]
        dmxpyJ_batch = dmxpy_stack[idx_j]
        f1voI, f1voJ, f1ooIJ, vxc1_ee, k1aoIJ = _contract_xc_kernel_batched(
            td_nac, mf.xc, dmxpyI_batch, dmxpyJ_batch, dmzooIJ, True, True, singlet, with_nac=True)

        if with_k:
            beta = alpha - hyb
            if not isinstance(mf, _DFHF):
                dm = cp.vstack([dmxpy_stack+dmxpy_stack.transpose(0,2,1), dmzooIJ])
                vj, vk = mf.get_jk(mol, dm, hermi=1)
                vk *= hyb
                vj, vj0IJ = vj[:n_states], vj[n_states:]
                vk_sym, vk0IJ = vk[:n_states], vk[n_states:]
                vk_asym = mf.get_k(mol, dmxmy_stack-dmxmy_stack.transpose(0,2,1), hermi=2)
                vk_asym *= hyb
                if omega != 0:
                    vk = mf.get_k(mol, dm, hermi=1, omega=omega) * beta
                    vk_sym += vk[:n_states]
                    vk0IJ += vk[n_states:]
                    vk_asym += mf.get_k(mol, dmxmy_stack-dmxmy_stack.transpose(0,2,1), hermi=2, omega=omega) * beta
            else:
                dmzooIJ_factorized = _tag_factorize_dm(dmzooIJ, hermi=1)
                vj0IJ, vk0IJ = mf.get_jk(mol, dmzooIJ_factorized, hermi=1)
                vk0IJ *= hyb
                if omega != 0:
                    vk0IJ += mf.get_k(mol, dmzooIJ_factorized, hermi=1, omega=omega) * beta
                vj, vk_sym = mf.get_jk(mol, dmxpy_stack, hermi=0)
                vk_sym *= hyb
                if omega != 0:
                    vk_sym += mf.get_k(mol, dmxpy_stack, hermi=0, omega=omega) * beta
                if is_tda:
                    vk_asym = vk_sym
                else:
                    vk_asym = mf.get_k(mol, dmxmy_stack, hermi=0) * hyb
                    if omega != 0:
                        vk_asym += mf.get_k(mol, dmxmy_stack, hermi=0, omega=omega) * beta
                vj *= 2
                vk_sym = vk_sym + vk_sym.transpose(0,2,1)
                vk_asym = vk_asym - vk_asym.transpose(0,2,1)
            
            vj1I, vj1J = vj[idx_i], vj[idx_j]
            vk1I, vk1J = vk_sym[idx_i], vk_sym[idx_j]
            vk2I, vk2J = vk_asym[idx_i], vk_asym[idx_j]
        else:
            if not isinstance(mf, _DFHF):
                dm = cp.vstack([dmxpy_stack+dmxpy_stack.transpose(0,2,1), dmzooIJ])
                vj = mf.get_j(mol, dm, hermi=1)
                vj, vj0IJ = vj[:n_states], vj[n_states:]
            else:
                vj0IJ = mf.get_j(mol, _tag_factorize_dm(dmzooIJ, hermi=1), hermi=1)
                vj = mf.get_j(mol, dmxpy_stack, hermi=0)
                vj *= 2
            vj1I, vj1J = vj[idx_i], vj[idx_j]
            vk0IJ = vk1I = vk1J = 0

        if has_grad and not singlet:
            vj1I[-1] = vj1J[-1] = 0

        def trans_veff_batch(veff_batch):
            return _cT_mat_c(mo_coeff, veff_batch, mo_coeff)

        veff0doo = vj0IJ * 2 - vk0IJ + f1ooIJ[:,0] + k1aoIJ[:, 0] * 2
        wvo = cp.einsum('pi, npq, qj -> nij', orbv, veff0doo, orbo) * 2.0
        veffI = vj1I * 2 - vk1I + f1voI[:, 0] * 2
        veffI *= 0.5
        veff0mopI = trans_veff_batch(veffI)
        wvo -= contract('nki, nai -> nak', veff0mopI[:, :nocc, :nocc], xpyJ) * 2.0
        wvo += contract('nac, nai -> nci', veff0mopI[:, nocc:, nocc:], xpyJ) * 2.0
        veffJ = vj1J * 2 - vk1J + f1voJ[:, 0] * 2
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

    vresp = td_nac.base.gen_response(singlet=None, hermi=1)

    z1_all = _solve_zvector(td_nac, rhs_all, vresp)
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
        W_ge[:, :nocc, nocc:] = GZS_mo_ge[:, :nocc, nocc:] + 0.5 * Y_stack[ge_targets].transpose(0, 2, 1) * E_stack[ge_targets, None, None] + 0.5 * zeta0.transpose(0, 2, 1)
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
            k_factor.extend([1., 2., -2.] * n_tasks_ee)
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

    if with_k:
        k_factor_np = np.array(k_factor)
        ejk_all_raw = td_nac.jk_energies_per_atom(
            dms_tasks, j_factor, k_factor_np * hyb, sum_results=False)
    else:
        ejk_all_raw = td_nac.jk_energies_per_atom(
            dms_tasks, j_factor, None, sum_results=False)
    
    ejk_all_raw = cp.asarray(ejk_all_raw)

    if with_k and omega != 0:
        beta = alpha - hyb
        ejk_lr_raw = td_nac.jk_energies_per_atom(
            dms_tasks, None, k_factor_np * beta, omega=omega, sum_results=False)
        ejk_all_raw += cp.asarray(ejk_lr_raw)

    ejk_all_raw *= 2.0
    de_all = dh_td_all - ds_all + dh1e_td_all
    
    if n_tasks_ge > 0:
        de_all[:n_tasks_ge] += ejk_all_raw[:n_tasks_ge]
    if n_tasks_ee > 0:
        ejk_ee = ejk_all_raw[n_tasks_ge:]
        ejk_ee = ejk_ee.reshape(n_tasks_ee, -1, natm, 3).sum(axis=1)
        de_all[n_tasks_ge:] += ejk_ee

    t_debug_5 = log.timer_silent(*time0)[2]

    if n_tasks_ge > 0:
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
        
        veff1_1_batch_ee = f1ooIJ[:, 1:] + fxcz1_ee[:, 1:] + k1aoIJ[:, 1:] * 2
        veff1_2I_batch = f1voI[:, 1:]
        veff1_2J_batch = f1voJ[:, 1:]
        veff1_0_batch_ee = cp.repeat(vxc1_ee[None, 1:], n_tasks_ee, axis=0)
        
        dveff1_0_ee = contract_veff_dm_batched(mol, veff1_0_batch_ee, dmz1doo_ee, hermi=0)
        oo0_batch_ee = cp.repeat(oo0[None, ...], n_tasks_ee, axis=0)
        dveff1_1_ee = contract_veff_dm_batched(mol, veff1_1_batch_ee, oo0_batch_ee, hermi=1) * 0.5
        dveff1_2_ee = contract_veff_dm_batched(mol, veff1_2I_batch, dmxpyJ_batch, hermi=0) * 2.0
        dveff1_2_ee += contract_veff_dm_batched(mol, veff1_2J_batch, dmxpyI_batch, hermi=0) * 2.0
        
        de_all[n_tasks_ge:] += dveff1_0_ee + dveff1_1_ee + dveff1_2_ee

    t_debug_6 = log.timer_silent(*time0)[2]

    results = {}
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

    if n_tasks_ee > 0:
        base_de_ee = de_all[n_tasks_ge:]
        
        if has_grad:
            n_tasks_ee -= 1
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

    t_debug_7 = log.timer_silent(*time0)[2]
    if log.verbose >= logger.DEBUG:
        time_list = [0, t_debug_1, t_debug_2, t_debug_3, t_debug_4, t_debug_5, t_debug_6, t_debug_7]
        time_list = [time_list[idx] - time_list[idx-1] for idx in range(1, len(time_list))]
        for idx, t in enumerate(time_list):
            logger.note(td_nac, f"Time for step {idx}: {t*1e-3:.6f}s")
            
    return results


class NAC_multistates(NAC_multistates_hf):
    
    _keys = {'ris_zvector_solver'}

    def __init__(self, td):
        super().__init__(td)
        self.ris_zvector_solver = False

    def get_nacv_multi(self, x_list, y_list, E_list, singlet=True, ge_targets=None, 
        ee_pairs=None, grad_state_idx=None, atmlst=None, verbose=logger.INFO):
        return get_nacv_multi(self, x_list, y_list, E_list, singlet=singlet, ge_targets=ge_targets, 
            ee_pairs=ee_pairs, grad_state_idx=grad_state_idx, atmlst=atmlst, verbose=verbose)