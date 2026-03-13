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
from gpu4pyscf.scf import cphf
from gpu4pyscf import tdscf
from gpu4pyscf.nac.tdrhf_grad_nacv import NAC_multistates
from gpu4pyscf.nac.tdrhf_grad_nacv import contract_h1e_dm_batched, contract_h1e_dm_asym_batched


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


def _contract_xc_kernel_batched(td_grad, xc_code, dmvo_batch, dmoo_batch=None,
            with_vxc=True, with_kxc=True, singlet=True, with_nac=False, dmvo_2_batch=None):
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

    n_batch = dmvo_batch.shape[0]

    dmvo_batch = (dmvo_batch + dmvo_batch.transpose(0, 2, 1)) * 0.5
    dmvo_batch = opt.sort_orbitals(dmvo_batch, axis=[1, 2])

    f1vo_batch = cp.zeros((n_batch, 4, nao, nao))
    deriv = 2

    if dmoo_batch is not None:
        f1oo_batch = cp.zeros((n_batch, 4, nao, nao))
        dmoo_batch = opt.sort_orbitals(dmoo_batch, axis=[1, 2])
    else:
        f1oo_batch = None

    if with_vxc:
        v1ao_batch = cp.zeros((n_batch, 4, nao, nao))
    else:
        v1ao_batch = None

    if with_kxc:
        k1ao_batch = cp.zeros((n_batch, 4, nao, nao))
        deriv = 3
        if with_nac:
            assert dmvo_2_batch is not None
            dmvo_2_batch = (dmvo_2_batch + dmvo_2_batch.transpose(0, 2, 1)) * 0.5
            dmvo_2_batch = opt.sort_orbitals(dmvo_2_batch, axis=[1, 2])
    else:
        k1ao_batch = None

    if xctype == "HF":
        return f1vo_batch, f1oo_batch, v1ao_batch, k1ao_batch
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

        # Process each state in the batch, this cannot be simplified
        for i in range(n_batch):
            dmvo_mask = dmvo_batch[i][mask[:, None], mask]
            rho1 = ni.eval_rho(_sorted_mol, ao0, dmvo_mask, mask, xctype, hermi=1, with_lapl=False)
            
            if singlet:
                rho1 *= 2.0  # *2 for alpha + beta
            if xctype == "LDA":
                rho1 = rho1[cp.newaxis].copy()

            # f1vo evaluation
            if singlet:
                tmp = contract("yg, xyg -> xg", rho1, fxc)
            else:
                tmp = contract("yg, xyg -> xg", rho1, fxc_t)
            wv = contract("xg, g -> xg", tmp, weight)
            fmat_(_sorted_mol, f1vo_batch[i], ao, wv, mask, shls_slice, ao_loc)

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

            # v1ao evaluation
            if with_vxc:
                vxc_to_use = vxc if singlet else vxc_s
                fmat_(_sorted_mol, v1ao_batch[i], ao, vxc_to_use * weight, mask, shls_slice, ao_loc)

            # k1ao evaluation
            if with_kxc:
                if with_nac:
                    dmvo_2_mask = dmvo_2_batch[i][mask[:, None], mask]
                    rho_dmvo_2 = ni.eval_rho(_sorted_mol, ao0, dmvo_2_mask, mask, xctype, hermi=1, with_lapl=False)
                    if singlet:
                        rho_dmvo_2 *= 2.0
                    if xctype == "LDA":
                        rho_dmvo_2 = rho_dmvo_2[cp.newaxis].copy()
                        
                    kxc_to_use = kxc if singlet else kxc_t
                    tmp = contract("yg, xyzg -> xzg", rho1, kxc_to_use)
                    tmp = contract("zg, xzg -> xg", rho_dmvo_2, tmp)
                else:
                    kxc_to_use = kxc if singlet else kxc_t
                    tmp = contract("yg, xyzg -> xzg", rho1, kxc_to_use)
                    tmp = contract("zg, xzg -> xg", rho1, tmp)
                    
                wv = contract("xg, g -> xg", tmp, weight)
                fmat_(_sorted_mol, k1ao_batch[i], ao, wv, mask, shls_slice, ao_loc)

    f1vo_batch[:, 1:] *= -1
    f1vo_batch = opt.unsort_orbitals(f1vo_batch, axis=[2, 3])
    
    if f1oo_batch is not None:
        f1oo_batch[:, 1:] *= -1
        f1oo_batch = opt.unsort_orbitals(f1oo_batch, axis=[2, 3])
        
    if v1ao_batch is not None:
        v1ao_batch[:, 1:] *= -1
        v1ao_batch = opt.unsort_orbitals(v1ao_batch, axis=[2, 3])
        
    if k1ao_batch is not None:
        k1ao_batch[:, 1:] *= -1
        k1ao_batch = opt.unsort_orbitals(k1ao_batch, axis=[2, 3])

    return f1vo_batch, f1oo_batch, v1ao_batch, k1ao_batch


def get_nacv_ge_multi(td_nac, x_list, y_list, E_list, singlet=True, atmlst=None, verbose=logger.INFO):
    if singlet is False:
        raise NotImplementedError('Only supports for singlet states')
    log = logger.new_logger(td_nac, verbose)
    time0 = logger.init_timer(td_nac)

    mol = td_nac.mol
    mf = td_nac.base._scf
    if getattr(mf, 'with_solvent', None) is not None:
        raise NotImplementedError('NACv gradient calculation is not supported for solvent models')
    mf_grad = mf.nuc_grad_method()
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
    if not isinstance(y_list[0], np.ndarray) and not isinstance(y_list[0], cp.ndarray):
        Y_stack = cp.zeros_like(X_stack)
    else:
        Y_stack = cp.asarray(y_list).reshape(n_states, nocc, nvir).transpose(0, 2, 1)
    E_stack = cp.asarray(E_list)

    LI = X_stack - Y_stack
    t_debug_1 = log.timer_silent(*time0)[2]
    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    with_k = ni.libxc.is_hybrid_xc(mf.xc)

    if isinstance(td_nac.base, tdscf.ris.TDDFT) or isinstance(td_nac.base, tdscf.ris.TDA):
        if td_nac.ris_zvector_solver:
            logger.note(td_nac, 'Use ris-approximated Z-vector solver')
            from gpu4pyscf.dft import rks
            from gpu4pyscf.tdscf.ris import get_auxmol
            from gpu4pyscf.grad import tdrks_ris

            theta = td_nac.base.theta
            J_fit = td_nac.base.J_fit
            K_fit = td_nac.base.K_fit
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
            logger.note(td_nac, 'Use standard Z-vector solver')
            vresp = td_nac.base._scf.gen_response(singlet=None, hermi=1)
    else:
        if getattr(td_nac, 'ris_zvector_solver', None) is not None:
            raise NotImplementedError('Ris-approximated Z-vector solver is not supported for standard TDDFT or TDA')
        vresp = td_nac.base.gen_response(singlet=None, hermi=1)

    def fvind(x_flat):
        n_vecs = x_flat.shape[0]
        x_batch = x_flat.reshape(n_vecs, nvir, nocc)
        dm = cp.einsum('ua, nai, vi -> nuv', orbv, x_batch * 2.0, orbo)
        dm_sym = dm + dm.transpose(0, 2, 1)
        v1ao = vresp(dm_sym) 
        resp_mo = cp.einsum('ua, nuv, vi -> nai', orbv, v1ao, orbo)
        return resp_mo.reshape(n_vecs, -1)

    rhs = (-LI * E_stack[:, None, None])
    rhs = cp.ascontiguousarray(rhs)
    z1_flat = cp.zeros((n_states, nvir, nocc))
    for istate in range(n_states):
        z1_flat[istate] = cphf.solve(
            fvind,
            mo_energy,
            mo_occ,
            rhs[istate],
            max_cycle=td_nac.cphf_max_cycle,
            tol=td_nac.cphf_conv_tol
        )[0] 
    t_debug_2 = log.timer_silent(*time0)[2]
    z1 = z1_flat.reshape(n_states, nvir, nocc)

    z1ao = cp.einsum('ua, nai, vi -> nuv', orbv, z1, orbo) * 2.0
    z1aoS = (z1ao + z1ao.transpose(0, 2, 1)) * 0.5 
    
    GZS = vresp(z1aoS) 
    GZS_mo = cp.einsum('up, nuv, vq -> npq', mo_coeff, GZS, mo_coeff)
    
    W = cp.zeros((n_states, nmo, nmo))
    W[:, :nocc, :nocc] = GZS_mo[:, :nocc, :nocc]
    
    zeta0 = z1 * mo_energy[nocc:][None, :, None]
    W[:, :nocc, nocc:] = GZS_mo[:, :nocc, nocc:] \
                       + 0.5 * Y_stack.transpose(0, 2, 1) * E_stack[:, None, None] \
                       + 0.5 * zeta0.transpose(0, 2, 1)
                       
    zeta1 = z1 * mo_energy[:nocc][None, None, :]
    W[:, nocc:, :nocc] = 0.5 * X_stack * E_stack[:, None, None] + 0.5 * zeta1
    
    W_ao = cp.einsum('up, npq, vq -> nuv', mo_coeff, W, mo_coeff) * 2.0

    dmz1doo = z1aoS
    td_nac._dmz1doo = dmz1doo
    oo0 = reduce(cp.dot, (orbo, orbo.T)) * 2.0 
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
    k_tasks = [hyb] * n_states if with_k else None
    hermi_tasks = [1] * n_states

    ejk_all = td_nac.jk_energies_per_atom(dms_tasks, j_tasks, k_tasks, hermi=hermi_tasks, sum_results=False)
    ejk_all = cp.asarray(ejk_all) * 2.0

    if with_k and omega != 0:
        beta = alpha - hyb
        k_omega = [beta] * n_states
        ejk_temp = td_nac.jk_energies_per_atom(dms_tasks, None, k_omega, hermi=hermi_tasks, omega=omega, sum_results=False) * 2.0
        ejk_all += cp.asarray(ejk_temp)
    t_debug_5 = log.timer_silent(*time0)[2]
    # Batched XC Kernel, this will save xc evaluations for ground state based density
    f1ooP_batch, _, vxc1_batch, _ = _contract_xc_kernel_batched(
        td_nac, mf.xc, dmz1doo, dmz1doo, True, False, singlet)
    t_debug_6 = time.log.timer_silent(*time0)[2]
    veff1_0_batch = vxc1_batch[:, 1:]
    veff1_1_batch = f1ooP_batch[:, 1:]

    de = dh_td - ds + ejk_all

    xIao = cp.einsum('ui, nia, va -> nuv', orbo, X_stack.transpose(0, 2, 1), orbv)
    yIao = cp.einsum('ua, nai, vi -> nuv', orbv, Y_stack, orbo)

    dsxy_x = contract_h1e_dm_asym_batched(mol, s1, xIao * E_stack[:, None, None]) * 2.0
    dsxy_y = contract_h1e_dm_asym_batched(mol, s1, yIao * E_stack[:, None, None]) * 2.0
    dsxy = dsxy_x + dsxy_y
    
    dsxy_etf_x = contract_h1e_dm_batched(mol, s1, xIao * E_stack[:, None, None])
    dsxy_etf_y = contract_h1e_dm_batched(mol, s1, yIao * E_stack[:, None, None])
    dsxy_etf = dsxy_etf_x + dsxy_etf_y
    
    dveff1_0 = contract_veff_dm_batched(mol, veff1_0_batch, dmz1doo, hermi=0)
    oo0_batch = cp.repeat(oo0[None, ...], n_states, axis=0)
    dveff1_1 = contract_veff_dm_batched(mol, veff1_1_batch, oo0_batch, hermi=1) * 0.5

    de += dh1e_td + dveff1_0 + dveff1_1
    de_etf = de + dsxy_etf
    de += dsxy
    
    results = {}
    for local_idx in range(n_states):
        results[local_idx] = {
            'de': de[local_idx].get(),
            'de_scaled': de[local_idx].get() / E_stack[local_idx].get(),
            'de_etf': de_etf[local_idx].get(),
            'de_etf_scaled': de_etf[local_idx].get() / E_stack[local_idx].get()
        }
    t_debug_7 = log.timer_silent(*time0)[2]
    if log.verbose >= logger.DEBUG:
        time_list = [0, t_debug_1, t_debug_2, t_debug_3, t_debug_4, t_debug_5, t_debug_6, t_debug_7]
        time_list = [time_list[i] - time_list[i - 1] for i in range(1, len(time_list))]
        for i, t in enumerate(time_list):
            logger.note(td_nac, f"Time for step {i}: {t*1e-3:.6f}s")
    return results


def get_nacv_ee_multi(td_nac, x_list, y_list, E_list, singlet=True, atmlst=None, verbose=logger.INFO, grad_state_idx=None):
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
    
    X_stack = cp.asarray(x_list).reshape(n_states, nocc, nvir).transpose(0, 2, 1)
    if not isinstance(y_list[0], np.ndarray) and not isinstance(y_list[0], cp.ndarray):
        Y_stack = cp.zeros_like(X_stack)
    else:
        Y_stack = cp.asarray(y_list).reshape(n_states, nocc, nvir).transpose(0, 2, 1)
    E_stack = cp.asarray(E_list)

    idx_i, idx_j, pairs = [], [], []
    for i in range(n_states):
        for j in range(i + 1, n_states):
            idx_i.append(i)
            idx_j.append(j)
            pairs.append((i, j))
    
    idx_i = cp.asarray(idx_i)
    idx_j = cp.asarray(idx_j)
    n_pairs = len(pairs)

    xI, yI = X_stack[idx_i], Y_stack[idx_i]
    xJ, yJ = X_stack[idx_j], Y_stack[idx_j]
    EI, EJ = E_stack[idx_i], E_stack[idx_j]
    dE = EJ - EI
    
    xpyI, xmyI = xI + yI, xI - yI
    xpyJ, xmyJ = xJ + yJ, xJ - yJ
    
    def transform_to_ao(amp_batch):
        return cp.einsum('ua, nai, vi -> nuv', orbv, amp_batch, orbo)

    dmxpyI, dmxmyI = transform_to_ao(xpyI), transform_to_ao(xmyI)
    dmxpyJ, dmxmyJ = transform_to_ao(xpyJ), transform_to_ao(xmyJ)
    
    rIJoo = -cp.einsum('nai, naj -> nij', xJ, xI) - cp.einsum('nai, naj -> nij', yI, yJ)
    rIJvv = cp.einsum('nai, nbi -> nab', xI, xJ) + cp.einsum('nai, nbi -> nab', yJ, yI)

    TIJoo = (rIJoo + rIJoo.transpose(0, 2, 1)) * 0.5
    TIJvv = (rIJvv + rIJvv.transpose(0, 2, 1)) * 0.5

    dmzooIJ = cp.einsum('ui, nij, vj -> nuv', orbo, TIJoo, orbo) * 2.0
    dmzooIJ += cp.einsum('ua, nab, vb -> nuv', orbv, TIJvv, orbv) * 2.0

    dms_to_stack = [
        dmzooIJ,
        dmxpyI + dmxpyI.transpose(0, 2, 1),
        dmxmyI - dmxmyI.transpose(0, 2, 1),
        dmxpyJ + dmxpyJ.transpose(0, 2, 1),
        dmxmyJ - dmxmyJ.transpose(0, 2, 1)
    ]
    
    if grad_state_idx is not None:
        x_g, y_g = X_stack[grad_state_idx], Y_stack[grad_state_idx]
        xpy_g, xmy_g = x_g + y_g, x_g - y_g
        
        dvv_g = cp.einsum("ai, bi -> ab", xpy_g, xpy_g) + cp.einsum("ai, bi -> ab", xmy_g, xmy_g)
        doo_g = -cp.einsum("ai, aj -> ij", xpy_g, xpy_g) - cp.einsum("ai, aj -> ij", xmy_g, xmy_g)
        
        dmxpy_g = cp.einsum('ua, ai, vi -> uv', orbv, xpy_g, orbo)
        dmxmy_g = cp.einsum('ua, ai, vi -> uv', orbv, xmy_g, orbo)
        
        dmzoo_g = cp.einsum('ui, ij, vj -> uv', orbo, doo_g, orbo)
        dmzoo_g += cp.einsum('ua, ab, vb -> uv', orbv, dvv_g, orbv)
        
        dms_to_stack.extend([
            dmzoo_g[None, ...],
            (dmxpy_g + dmxpy_g.T)[None, ...],
            (dmxmy_g - dmxmy_g.T)[None, ...]
        ])

    full_dms = cp.concatenate(dms_to_stack, axis=0)
    
    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    with_k = ni.libxc.is_hybrid_xc(mf.xc)

    if grad_state_idx is not None:
        dmxpyI_ext = cp.concatenate([dmxpyI, dmxpy_g[None, ...]], axis=0)
        dmzooIJ_ext = cp.concatenate([dmzooIJ, dmzoo_g[None, ...]], axis=0)
        dmxpyJ_ext = cp.concatenate([dmxpyJ, dmxpy_g[None, ...]], axis=0)
    else:
        dmxpyI_ext = dmxpyI
        dmzooIJ_ext = dmzooIJ
        dmxpyJ_ext = dmxpyJ
    t_debug_1 = log.timer_silent(*time0)[2]
    f1voI_all, f1ooIJ_all, vxc1_all, k1aoIJ_all = _contract_xc_kernel_batched(
        td_nac, mf.xc, dmxpyI_ext, dmzooIJ_ext, True, True, singlet, with_nac=True, dmvo_2_batch=dmxpyJ_ext)
    
    f1voJ_all, _, _, _ = _contract_xc_kernel_batched(
        td_nac, mf.xc, dmxpyJ, None, False, False, singlet)

    f1voI = f1voI_all[:n_pairs]
    f1ooIJ = f1ooIJ_all[:n_pairs]
    vxc1 = vxc1_all[:n_pairs]
    k1aoIJ = k1aoIJ_all[:n_pairs]
    f1voJ = f1voJ_all[:n_pairs]
    t_debug_2 = log.timer_silent(*time0)[2]
    if with_k:
        vj_all, vk_all = mf.get_jk(mol, full_dms, hermi=0)
        vk_all *= hyb
        if omega != 0:
            vk_omega_all = mf.get_k(mol, full_dms, hermi=0, omega=omega)
            vk_all += vk_omega_all * (alpha - hyb)

        vj_split = cp.split(vj_all[:5 * n_pairs], 5, axis=0)
        vk_split = cp.split(vk_all[:5 * n_pairs], 5, axis=0)
        
        vj0IJ, vk0IJ = vj_split[0], vk_split[0]
        vj1I,  vk1I  = vj_split[1], vk_split[1]
        vk2I  = vk_split[2]
        vj1J,  vk1J  = vj_split[3], vk_split[3]
        vk2J  = vk_split[4]

        if grad_state_idx is not None:
            vj_g = vj_all[5 * n_pairs:]
            vk_g = vk_all[5 * n_pairs:]
            vj0_g, vj1_g = vj_g[0], vj_g[1]
            vk0_g, vk1_g, vk2_g = vk_g[0], vk_g[1], vk_g[2]
            
            f1vo_g, f1oo_g, vxc1_g, k1ao_g = f1voI_all[-1], f1ooIJ_all[-1], vxc1_all[-1], k1aoIJ_all[-1]

        def trans_veff_batch(veff_batch):
            return cp.einsum('up, nuv, vq -> npq', mo_coeff, veff_batch, mo_coeff)

        veff0doo = vj0IJ * 2 - vk0IJ + f1ooIJ[:, 0] + k1aoIJ[:, 0] * 2
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
        veffI = -vk2I * 0.5
        veff0momI = trans_veff_batch(veffI)
        wvo -= contract('nki, nai -> nak', veff0momI[:, :nocc, :nocc], xmyJ) * 2.0
        wvo += contract('nac, nai -> nci', veff0momI[:, nocc:, nocc:], xmyJ) * 2.0
        veffJ = -vk2J * 0.5
        veff0momJ = trans_veff_batch(veffJ)
        wvo -= contract('nki, nai -> nak', veff0momJ[:, :nocc, :nocc], xmyI) * 2.0
        wvo += contract('nac, nai -> nci', veff0momJ[:, nocc:, nocc:], xmyI) * 2.0

        if grad_state_idx is not None:
            def trans_veff(veff, C): 
                return reduce(cp.dot, (C.T, veff, C))
            veff0doo_g = vj0_g * 2 - vk0_g + f1oo_g[0] + k1ao_g[0] * 2
            wvo_g = reduce(cp.dot, (orbv.T, veff0doo_g, orbo)) * 2.0
            veff1_g = vj1_g * 2 - vk1_g + f1vo_g[0] * 2
            veff0mop_g = trans_veff(veff1_g, mo_coeff)
            wvo_g -= contract("ki, ai -> ak", veff0mop_g[:nocc, :nocc], xpy_g) * 2.0
            wvo_g += contract("ac, ai -> ci", veff0mop_g[nocc:, nocc:], xpy_g) * 2.0
            veff2_g = -vk2_g
            veff0mom_g = trans_veff(veff2_g, mo_coeff)
            wvo_g -= contract("ki, ai -> ak", veff0mom_g[:nocc, :nocc], xmy_g) * 2.0
            wvo_g += contract("ac, ai -> ci", veff0mom_g[nocc:, nocc:], xmy_g) * 2.0
            
    else:
        # Non-hybrid
        vj_all = mf.get_j(mol, full_dms, hermi=1)
        vj_split = cp.split(vj_all[:5 * n_pairs], 5, axis=0)
        vj0IJ, vj1I, vj1J = vj_split[0], vj_split[1], vj_split[3]

        if grad_state_idx is not None:
            vj_g = vj_all[5 * n_pairs:]
            vj0_g, vj1_g = vj_g[0], vj_g[1]
            f1vo_g, f1oo_g, vxc1_g, k1ao_g = f1voI_all[-1], f1ooIJ_all[-1], vxc1_all[-1], k1aoIJ_all[-1]

        def trans_veff_batch(veff_batch): 
            return cp.einsum('up, nuv, vq -> npq', mo_coeff, veff_batch, mo_coeff)

        veff0doo = vj0IJ * 2 + f1ooIJ[:, 0] + k1aoIJ[:, 0] * 2
        wvo = cp.einsum('pi, npq, qj -> nij', orbv, veff0doo, orbo) * 2.0
        veffI = vj1I * 2 + f1voI[:, 0] * 2
        veffI *= 0.5
        veff0mopI = trans_veff_batch(veffI)
        wvo -= contract('nki, nai -> nak', veff0mopI[:, :nocc, :nocc], xpyJ) * 2.0
        wvo += contract('nac, nai -> nci', veff0mopI[:, nocc:, nocc:], xpyJ) * 2.0
        veffJ = vj1J * 2 + f1voJ[:, 0] * 2
        veffJ *= 0.5
        veff0mopJ = trans_veff_batch(veffJ)
        wvo -= contract('nki, nai -> nak', veff0mopJ[:, :nocc, :nocc], xpyI) * 2.0
        wvo += contract('nac, nai -> nci', veff0mopJ[:, nocc:, nocc:], xpyI) * 2.0
        veff0momI = cp.zeros((n_pairs, nmo, nmo))
        veff0momJ = cp.zeros((n_pairs, nmo, nmo))

        if grad_state_idx is not None:
            def trans_veff(veff, C): 
                return reduce(cp.dot, (C.T, veff, C))
            veff0doo_g = vj0_g * 2 + f1oo_g[0] + k1ao_g[0] * 2
            wvo_g = reduce(cp.dot, (orbv.T, veff0doo_g, orbo)) * 2.0
            veff1_g = vj1_g * 2 + f1vo_g[0] * 2
            veff0mop_g = trans_veff(veff1_g, mo_coeff)
            wvo_g -= contract("ki, ai -> ak", veff0mop_g[:nocc, :nocc], xpy_g) * 2.0
            wvo_g += contract("ac, ai -> ci", veff0mop_g[nocc:, nocc:], xpy_g) * 2.0
            veff0mom_g = cp.zeros((nmo, nmo))

    vresp = td_nac.base.gen_response(singlet=None, hermi=1)
    t_debug_3 = log.timer_silent(*time0)[2]
    def fvind(x_flat):
        n_vecs = x_flat.shape[0]
        x_batch = x_flat.reshape(n_vecs, nvir, nocc)
        dm = cp.einsum('ua, nai, vi -> nuv', orbv, x_batch * 2, orbo)
        dm_sym = dm + dm.transpose(0, 2, 1)
        v1ao = vresp(dm_sym) 
        resp_mo = cp.einsum('ua, nuv, vi -> nai', orbv, v1ao, orbo)
        return resp_mo.reshape(n_vecs, -1)

    # rhs = wvo / dE[:, None, None]
    rhs = wvo
    if grad_state_idx is not None:
        rhs = cp.concatenate([rhs, wvo_g[None, ...]], axis=0)

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
    
    z1_flat = cphf.solve(
        fvind, mo_energy, mo_occ, rhs,
        max_cycle=td_nac.cphf_max_cycle,
        tol=td_nac.cphf_conv_tol
    )[0]
    t_debug_4 = log.timer_silent(*time0)[2]
    if grad_state_idx is not None:
        z1 = z1_flat[:-1].reshape(n_pairs, nvir, nocc)
        z1_g = z1_flat[-1].reshape(nvir, nocc)
    else:
        z1 = z1_flat.reshape(n_pairs, nvir, nocc)

    z1 = z1 / dE[:, None, None]

    z1ao = cp.einsum('ua, nai, vi-> nuv', orbv, z1, orbo)
    z1ao_sym = z1ao + z1ao.transpose(0, 2, 1)
    z1aoS = z1ao_sym * 0.5 * dE[:, None, None]
    dmz1doo = z1aoS + dmzooIJ
    
    if grad_state_idx is not None:
        z1ao_g = reduce(cp.dot, (orbv, z1_g, orbo.T))
        z1ao_g_sym = z1ao_g + z1ao_g.T
        z1ao_sym_all = cp.concatenate([z1ao_sym, z1ao_g_sym[None, ...]], axis=0)
        veff_all = vresp(z1ao_sym_all)
        veff = veff_all[:-1]
        veff_z_g = veff_all[-1]
    else:
        veff = vresp(z1ao_sym)

    fock_mo = cp.diag(mo_energy)
    TFoo = cp.matmul(TIJoo, fock_mo[:nocc, :nocc])
    TFov = cp.matmul(TIJoo, fock_mo[:nocc, nocc:])
    TFvo = cp.matmul(TIJvv, fock_mo[nocc:, :nocc])
    TFvv = cp.matmul(TIJvv, fock_mo[nocc:, nocc:])

    im0 = cp.zeros((n_pairs, nmo, nmo))
    
    term_oo = cp.einsum('ui, nuv, vj -> nij', orbo, veff0doo, orbo)
    term_oo += TFoo * 2.0
    term_oo += cp.einsum('nak, nai -> nik', veff0mopI[:, nocc:, :nocc], xpyJ)
    term_oo += cp.einsum('nak, nai -> nik', veff0momI[:, nocc:, :nocc], xmyJ)
    term_oo += cp.einsum('nak, nai -> nik', veff0mopJ[:, nocc:, :nocc], xpyI)
    term_oo += cp.einsum('nak, nai -> nik', veff0momJ[:, nocc:, :nocc], xmyI)
    term_oo += rIJoo.transpose(0, 2, 1) * dE[:, None, None]
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
    term_vv += rIJvv.transpose(0, 2, 1) * dE[:, None, None]
    im0[:, nocc:, nocc:] = term_vv

    im0 *= 0.5
    
    im0[:, :nocc, :nocc] += cp.einsum('ui, nuv, vj -> nij', orbo, veff, orbo) * dE[:, None, None] * 0.5
    im0[:, :nocc, nocc:] += cp.einsum('ui, nuv, va -> nia', orbo, veff, orbv) * dE[:, None, None] * 0.5
    z1_fock_ov = cp.einsum('ab, nbi -> nai', fock_mo[nocc:, nocc:], z1)
    im0[:, :nocc, nocc:] += z1_fock_ov.transpose(0, 2, 1) * dE[:, None, None] * 0.25
    z1_fock_vo = cp.einsum('nai, ij -> naj', z1, fock_mo[:nocc, :nocc])
    im0[:, nocc:, :nocc] += z1_fock_vo * dE[:, None, None] * 0.25

    im0_ao = cp.einsum('up, npq, vq-> nuv', mo_coeff, im0, mo_coeff) * 2.0
    
    if grad_state_idx is not None:
        im0_g = cp.zeros((nmo, nmo))
        im0_g[:nocc, :nocc] = reduce(cp.dot, (orbo.T, veff0doo_g + veff_z_g, orbo))
        im0_g[:nocc, :nocc] += contract("ak, ai -> ki", veff0mop_g[nocc:, :nocc], xpy_g)
        im0_g[:nocc, :nocc] += contract("ak, ai -> ki", veff0mom_g[nocc:, :nocc], xmy_g)
        im0_g[nocc:, nocc:] = contract("ci, ai -> ac", veff0mop_g[nocc:, :nocc], xpy_g)
        im0_g[nocc:, nocc:] += contract("ci, ai -> ac", veff0mom_g[nocc:, :nocc], xmy_g)
        im0_g[nocc:, :nocc] = contract("ki, ai -> ak", veff0mop_g[:nocc, :nocc], xpy_g) * 2.0
        im0_g[nocc:, :nocc] += contract("ki, ai -> ak", veff0mom_g[:nocc, :nocc], xmy_g) * 2.0

        zeta = (mo_energy[:, None] + mo_energy) * 0.5
        zeta[nocc:, :nocc] = mo_energy[:nocc]
        zeta[:nocc, nocc:] = mo_energy[nocc:]

        dm1_g = cp.zeros((nmo, nmo))
        dm1_g[:nocc, :nocc] = doo_g
        dm1_g[nocc:, nocc:] = dvv_g
        dm1_g[nocc:, :nocc] = z1_g
        dm1_g[:nocc, :nocc] += cp.eye(nocc) * 2.0

        im0_g = im0_g + zeta * dm1_g
        im0_g = reduce(cp.dot, (mo_coeff, im0_g, mo_coeff.T))

    oo0 = reduce(cp.dot, (orbo, orbo.T)) * 2.0
    t_debug_5 = log.timer_silent(*time0)[2]
    mf_grad = td_nac.base._scf.nuc_grad_method()
    h1 = cp.asarray(mf_grad.get_hcore(mol))
    s1 = cp.asarray(mf_grad.get_ovlp(mol))
    
    dh_td = contract_h1e_dm_batched(mol, h1, dmz1doo, hermi=1)
    ds = contract_h1e_dm_batched(mol, s1, im0_ao, hermi=0)
    
    dh1e_td_list = []
    for k in range(n_pairs):
        dh1e_k = int3c2e.get_dh1e(mol, dmz1doo[k])
        if len(mol._ecpbas) > 0:
            dh1e_k += rhf_grad.get_dh1e_ecp(mol, dmz1doo[k])
        dh1e_td_list.append(dh1e_k)
    dh1e_td = cp.array(dh1e_td_list)
    t_debug_6 = log.timer_silent(*time0)[2]
    dm_xpyI_sym = dmxpyI + dmxpyI.transpose(0, 2, 1)
    dm_xpyJ_sym = dmxpyJ + dmxpyJ.transpose(0, 2, 1)
    dm_xmyI_asym = dmxmyI - dmxmyI.transpose(0, 2, 1)
    dm_xmyJ_asym = dmxmyJ - dmxmyJ.transpose(0, 2, 1)

    dms_tasks, j_tasks, k_tasks, hermi_tasks = [], [], [], []

    for k in range(n_pairs):
        if with_k:
            dms_tasks.extend([[dmz1doo[k], oo0], [dm_xpyI_sym[k], dm_xpyJ_sym[k]], [dm_xmyI_asym[k], dm_xmyJ_asym[k]]])
            j_tasks.extend([1., 1., 0.])
            k_tasks.extend([hyb, hyb, -hyb])
            hermi_tasks.extend([1, 1, 2])
        else:
            dms_tasks.extend([[dmz1doo[k], oo0], [dm_xpyI_sym[k], dm_xpyJ_sym[k]]])
            j_tasks.extend([1., 1.])
            k_tasks.extend([0., 0.])
            hermi_tasks.extend([1, 1])

    if grad_state_idx is not None:
        dmz1doo_g = z1ao_g + dmzoo_g
        dm1_g = (dmz1doo_g + dmz1doo_g.T) * 0.5 + oo0
        dm2_g = (dmz1doo_g + dmz1doo_g.T) * 0.5
        dm3_g = dmxpy_g + dmxpy_g.T
        dm4_g = dmxmy_g - dmxmy_g.T
        
        dms_g = [dm1_g, dm2_g, dm3_g, dm4_g]
        j_g = [1., -1., 2., 0.]
        
        if with_k:
            k_g = [hyb, -hyb, 2*hyb, -2*hyb]
        else:
            k_g = [0., 0., 0., 0.]
            
        hermi_g = [1, 1, 1, 2]

        dms_tasks.extend(dms_g)
        j_tasks.extend(j_g)
        k_tasks.extend(k_g)
        hermi_tasks.extend(hermi_g)

    ejk_all = td_nac.jk_energies_per_atom(dms_tasks, j_tasks, k_tasks if with_k else None, hermi=hermi_tasks, sum_results=False)
    ejk_all = cp.asarray(ejk_all)

    if with_k and omega != 0:
        beta = alpha - hyb
        k_omega = []
        for k in range(n_pairs):
            k_omega.extend([beta, beta, -beta])
        if grad_state_idx is not None:
            k_omega.extend([beta, -beta, 2*beta, -2*beta])
            
        ejk_temp = td_nac.jk_energies_per_atom(dms_tasks, None, k_omega, hermi=hermi_tasks, omega=omega, sum_results=False)
        ejk_all += cp.asarray(ejk_temp)

    n_dms_per_pair = 3 if with_k else 2
    ejk_nacv = ejk_all[:n_dms_per_pair*n_pairs].reshape(n_pairs, n_dms_per_pair, natm, 3).sum(axis=1) * 2.0
    t_debug_7 = log.timer_silent(*time0)[2]
    fxcz1_all = _contract_xc_kernel_batched(
        td_nac, mf.xc, cp.concatenate([z1aoS, z1ao_g[None, ...]], axis=0) if grad_state_idx is not None else z1aoS, 
        None, False, False, True)[0]
        
    veff1_0_batch = vxc1[:, 1:]
    veff1_1_batch = f1ooIJ[:, 1:] + fxcz1_all[:n_pairs, 1:] + k1aoIJ[:, 1:] * 2
    veff1_2I_batch = f1voI[:, 1:]
    veff1_2J_batch = f1voJ[:, 1:]
    t_debug_8 = log.timer_silent(*time0)[2]
    de = dh_td - ds + ejk_nacv

    dveff1_0 = contract_veff_dm_batched(mol, veff1_0_batch, dmz1doo, hermi=0)
    oo0_batch = cp.repeat(oo0[None, ...], n_pairs, axis=0)
    dveff1_1 = contract_veff_dm_batched(mol, veff1_1_batch, oo0_batch, hermi=1) * 0.5
    dveff1_2 = contract_veff_dm_batched(mol, veff1_2I_batch, dmxpyJ, hermi=0) * 2.0
    dveff1_2 += contract_veff_dm_batched(mol, veff1_2J_batch, dmxpyI, hermi=0) * 2.0

    rIJoo_ao = cp.einsum('ui, nij, vj -> nuv', orbo, rIJoo, orbo)
    rIJvv_ao = cp.einsum('ua, nab, vb -> nuv', orbv, rIJvv, orbv)
    TIJoo_ao = cp.einsum('ui, nij, vj -> nuv', orbo, TIJoo, orbo)
    TIJvv_ao = cp.einsum('ua, nab, vb -> nuv', orbv, TIJvv, orbv)
    
    dsxy = contract_h1e_dm_batched(mol, s1, rIJoo_ao * dE[:, None, None], hermi=1)
    dsxy += contract_h1e_dm_batched(mol, s1, rIJvv_ao * dE[:, None, None], hermi=1)
    
    dsxy_etf = contract_h1e_dm_batched(mol, s1, TIJoo_ao * dE[:, None, None], hermi=1)
    dsxy_etf += contract_h1e_dm_batched(mol, s1, TIJvv_ao * dE[:, None, None], hermi=1)
    
    de += dh1e_td + dveff1_0 + dveff1_1 + dveff1_2
    de_etf = de + dsxy_etf
    de += dsxy

    results = {}
    pair_indices = zip(idx_i.get(), idx_j.get())
    
    for k, (i, j) in enumerate(pair_indices):
        results[(int(i), int(j))] = {
            'de': de[k].get(),
            'de_scaled': de[k].get() / dE[k].get(),
            'de_etf': de_etf[k].get(),
            'de_etf_scaled': de_etf[k].get() / dE[k].get()
        }
        
    if grad_state_idx is not None:
        dh_ground = contract_h1e_dm_batched(mol, h1, oo0[None, ...], hermi=1)[0]
        dh_td_g = contract_h1e_dm_batched(mol, h1, dmz1doo_g[None, ...], hermi=0)[0]
        ds_g = contract_h1e_dm_batched(mol, s1, im0_g[None, ...], hermi=0)[0]
        
        dh1e_ground = int3c2e.get_dh1e(mol, oo0)
        if len(mol._ecpbas) > 0:
            dh1e_ground += rhf_grad.get_dh1e_ecp(mol, oo0)
            
        dh1e_td_g = int3c2e.get_dh1e(mol, (dmz1doo_g + dmz1doo_g.T) * 0.5)
        if len(mol._ecpbas) > 0:
            dh1e_td_g += rhf_grad.get_dh1e_ecp(mol, (dmz1doo_g + dmz1doo_g.T) * 0.5)

        ejk_g = ejk_all[n_dms_per_pair*n_pairs:].sum(axis=0)

        fxcz1_g = fxcz1_all[-1]
        veff1_0_g = vxc1_g[1:]
        veff1_1_g = (f1oo_g[1:] + fxcz1_g[1:] + k1ao_g[1:] * 2) * 2.0
        veff1_2_g = f1vo_g[1:] * 2.0

        dveff1_0_g = rhf_grad.contract_h1e_dm(mol, veff1_0_g, oo0 + dmz1doo_g, hermi=0)
        dveff1_1_g = rhf_grad.contract_h1e_dm(mol, veff1_1_g, oo0, hermi=1) * .25
        dveff1_2_g = rhf_grad.contract_h1e_dm(mol, veff1_2_g, dmxpy_g, hermi=0) * 2.0

        de_grad = dh_ground + dh_td_g - ds_g + ejk_g
        de_grad += dh1e_ground + dh1e_td_g + cp.asarray(dveff1_0_g + dveff1_1_g + dveff1_2_g)

        if atmlst is not None:
            de_grad = de_grad[atmlst]
        
        de_grad += cp.asarray(mf_grad.grad_nuc(mol, atmlst))
        if mol.symmetry:
            de_grad = cp.asarray(mf_grad.symmetrize(de_grad.get(), atmlst))
            
        results['gradient'] = de_grad.get()
    t_debug_9 = log.timer_silent(*time0)[2]
    if log.verbose >= logger.DEBUG:
    if verbose >= logger.NOTE:
        time_list = [0, t_debug_1, t_debug_2, t_debug_3, t_debug_4, t_debug_5, t_debug_6, t_debug_7, t_debug_8, t_debug_9]
        time_list = [time_list[i+1] - time_list[i] for i in range(len(time_list) - 1)]
        for i, t in enumerate(time_list):
            logger.note(td_nac, f"Time for step {i}: {t*1e-3:.6f}s")
    return results


class NAC_multistates(NAC_multistates):
    
    def get_nacv_ge_multi(self, x_list, y_list, E_list, singlet, atmlst=None, verbose=logger.INFO):
        return get_nacv_ge_multi(self, x_list, y_list, E_list, singlet, atmlst, verbose)
    
    def get_nacv_ee_multi(self, x_list, y_list, E_list, singlet, atmlst=None, verbose=logger.INFO, grad_state_idx=None):
        return get_nacv_ee_multi(self, x_list, y_list, E_list, singlet, atmlst, verbose, grad_state_idx)
