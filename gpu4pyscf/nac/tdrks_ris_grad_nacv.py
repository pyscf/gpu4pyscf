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
from gpu4pyscf.dft import rks
from pyscf import __config__
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.df import int3c2e
from gpu4pyscf.scf import cphf
from gpu4pyscf.nac.tdrhf_grad_nacv import contract_h1e_dm_batched
import time
from gpu4pyscf.tdscf.ris import get_auxmol, rescale_spin_free_amplitudes
from gpu4pyscf.nac.tdrks_grad_nacv import NAC_multistates as NAC_multistates_tdrks
from gpu4pyscf.nac.tdrks_grad_nacv import _contract_xc_kernel_batched, contract_veff_dm_batched
from pyscf.data.nist import HARTREE2EV
from gpu4pyscf.grad import tdrks_ris


def get_nacv_ee_multi(td_nac, x_list, y_list, E_list, singlet=True, atmlst=None, verbose=logger.INFO, grad_state_idx=None):
    
    if td_nac.base.Ktrunc != 0.0:
        raise NotImplementedError('Ktrunc or frozen method is not supported yet')
    theta = td_nac.base.theta
    J_fit = td_nac.base.J_fit
    K_fit = td_nac.base.K_fit
    
    t_debug_0 = time.time()
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
        dmxpyI + dmxpyI.transpose(0, 2, 1),
        dmxmyI - dmxmyI.transpose(0, 2, 1),
        dmxpyJ + dmxpyJ.transpose(0, 2, 1),
        dmxmyJ - dmxmyJ.transpose(0, 2, 1)
    ]

    sym_dms_list = [
        dmxpyI + dmxpyI.transpose(0, 2, 1),
        dmxpyJ + dmxpyJ.transpose(0, 2, 1)
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
            (dmxpy_g + dmxpy_g.T)[None, ...],
            (dmxmy_g - dmxmy_g.T)[None, ...]
        ])
        sym_dms_list.extend([
            (dmxpy_g + dmxpy_g.T)[None, ...]
        ])

    full_dms = cp.concatenate(dms_to_stack, axis=0)
    full_dms_sym = cp.concatenate(sym_dms_list, axis=0)
    
    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    with_k = ni.libxc.is_hybrid_xc(mf.xc)

    if grad_state_idx is not None:
        dmzooIJ_ext = cp.concatenate([dmzooIJ, dmzoo_g[None, ...]], axis=0)
    else:
        dmzooIJ_ext = dmzooIJ
    t_debug_1 = time.time()
    f1ooIJ_all, _, vxc1_all, _ = _contract_xc_kernel_batched(
        td_nac, mf.xc, dmzooIJ_ext, None, True, False, singlet)

    f1ooIJ = f1ooIJ_all[:n_pairs]
    vxc1 = vxc1_all[:n_pairs]
    t_debug_2 = time.time()

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
        vj_all, vk_all = mf.get_jk(mol, dmzooIJ_ext, hermi=0)
        vj_ris_all = mf_J.get_j(mol, full_dms_sym, hermi=0)
        vk_ris_all = mf_K.get_k(mol, full_dms, hermi=0)
        vk_all *= hyb
        vk_ris_all *= hyb
        if omega != 0:
            vk_omega_all = mf.get_k(mol, dmzooIJ_ext, hermi=0, omega=omega)
            vk_all += vk_omega_all * (alpha - hyb)
            vk_ris_all_omega = mf_K.get_k(mol, full_dms, hermi=0, omega=omega)
            vk_ris_all += vk_ris_all_omega * (alpha - hyb)

        vj_ris_split = cp.split(vj_ris_all[:2 * n_pairs], 2, axis=0)
        vk_ris_split = cp.split(vk_ris_all[:4 * n_pairs], 4, axis=0)

        
        vj0IJ, vk0IJ = vj_all[:n_pairs], vk_all[:n_pairs]
        vj1I,  vk1I  = vj_ris_split[0], vk_ris_split[0]
        vk2I  = vk_ris_split[1]
        vj1J,  vk1J  = vj_ris_split[1], vk_ris_split[2]
        vk2J  = vk_ris_split[3]

        if grad_state_idx is not None:
            vj0_g = vj_all[-1]
            vk0_g = vk_all[-1]
            vj1_g = vj_ris_all[-1]
            vk1_g, vk2_g = vk_ris_all[4*n_pairs:]
            
            f1oo_g, vxc1_g = f1ooIJ_all[-1], vxc1_all[-1]

        def trans_veff_batch(veff_batch):
            return cp.einsum('up, nuv, vq -> npq', mo_coeff, veff_batch, mo_coeff)

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
            veff0doo_g = vj0_g * 2 - vk0_g + f1oo_g[0]
            wvo_g = reduce(cp.dot, (orbv.T, veff0doo_g, orbo)) * 2.0
            veff1_g = vj1_g * 2 - vk1_g
            veff0mop_g = trans_veff(veff1_g, mo_coeff)
            wvo_g -= contract("ki, ai -> ak", veff0mop_g[:nocc, :nocc], xpy_g) * 2.0
            wvo_g += contract("ac, ai -> ci", veff0mop_g[nocc:, nocc:], xpy_g) * 2.0
            veff2_g = -vk2_g
            veff0mom_g = trans_veff(veff2_g, mo_coeff)
            wvo_g -= contract("ki, ai -> ak", veff0mom_g[:nocc, :nocc], xmy_g) * 2.0
            wvo_g += contract("ac, ai -> ci", veff0mom_g[nocc:, nocc:], xmy_g) * 2.0
            
    else:
        vj_all = mf.get_j(mol, dmzooIJ_ext, hermi=1)
        vj0IJ = vj_all[:n_pairs]
        vj_ris_all = mf_J.get_j(mol, full_dms_sym, hermi=0)
        vj_ris_split = cp.split(vj_ris_all[:2 * n_pairs], 2, axis=0)
        vj1I, vj1J = vj_ris_split[0], vj_ris_split[1]

        if grad_state_idx is not None:
            vj0_g = vj_all[n_pairs:]
            vj1_g = vj_ris_all[2 * n_pairs:]
            vj0_g = vj0_g[0]
            vj1_g = vj1_g[0]
            f1oo_g, vxc1_g = f1ooIJ_all[-1], vxc1_all[-1]

        def trans_veff_batch(veff_batch): 
            return cp.einsum('up, nuv, vq -> npq', mo_coeff, veff_batch, mo_coeff)

        veff0doo = vj0IJ * 2 + f1ooIJ[:, 0]
        wvo = cp.einsum('pi, npq, qj -> nij', orbv, veff0doo, orbo) * 2.0
        veffI = vj1I * 2
        veffI *= 0.5
        veff0mopI = trans_veff_batch(veffI)
        wvo -= contract('nki, nai -> nak', veff0mopI[:, :nocc, :nocc], xpyJ) * 2.0
        wvo += contract('nac, nai -> nci', veff0mopI[:, nocc:, nocc:], xpyJ) * 2.0
        veffJ = vj1J * 2
        veffJ *= 0.5
        veff0mopJ = trans_veff_batch(veffJ)
        wvo -= contract('nki, nai -> nak', veff0mopJ[:, :nocc, :nocc], xpyI) * 2.0
        wvo += contract('nac, nai -> nci', veff0mopJ[:, nocc:, nocc:], xpyI) * 2.0
        veff0momI = cp.zeros((n_pairs, nmo, nmo))
        veff0momJ = cp.zeros((n_pairs, nmo, nmo))

        if grad_state_idx is not None:
            def trans_veff(veff, C): 
                return reduce(cp.dot, (C.T, veff, C))
            veff0doo_g = vj0_g * 2 + f1oo_g[0]
            wvo_g = reduce(cp.dot, (orbv.T, veff0doo_g, orbo)) * 2.0
            veff1_g = vj1_g * 2
            veff0mop_g = trans_veff(veff1_g, mo_coeff)
            wvo_g -= contract("ki, ai -> ak", veff0mop_g[:nocc, :nocc], xpy_g) * 2.0
            wvo_g += contract("ac, ai -> ci", veff0mop_g[nocc:, nocc:], xpy_g) * 2.0
            veff0mom_g = cp.zeros((nmo, nmo))
    if td_nac.ris_zvector_solver:
        logger.note(td_nac, 'Use ris-approximated Z-vector solver')
        vresp = tdrks_ris.gen_response_ris(mf, mf_J, mf_K, singlet=None, hermi=1)
    else:
        logger.note(td_nac, 'Use standard Z-vector solver')
        vresp = mf.gen_response(singlet=None, hermi=1)

    t_debug_3 = time.time()
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
    t_debug_4 = time.time()
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
    t_debug_5 = time.time()
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
    t_debug_6 = time.time()
    dm_xpyI_sym = dmxpyI + dmxpyI.transpose(0, 2, 1)
    dm_xpyJ_sym = dmxpyJ + dmxpyJ.transpose(0, 2, 1)
    dm_xmyI_asym = dmxmyI - dmxmyI.transpose(0, 2, 1)
    dm_xmyJ_asym = dmxmyJ - dmxmyJ.transpose(0, 2, 1)

    dms_tasks_full, j_tasks_full, k_tasks_full, hermi_tasks_full = [], [], [], []
    dms_tasks_ris, j_tasks_ris, k_tasks_ris, hermi_tasks_ris = [], [], [], []

    for k in range(n_pairs):
        if with_k:
            dms_tasks_full.extend([[dmz1doo[k], oo0]])
            j_tasks_full.extend([1.])
            k_tasks_full.extend([hyb])
            hermi_tasks_full.extend([1])

            dms_tasks_ris.extend([[dm_xpyI_sym[k], dm_xpyJ_sym[k]], [dm_xmyI_asym[k], dm_xmyJ_asym[k]]])
            j_tasks_ris.extend([1., 0.])
            k_tasks_ris.extend([hyb, -hyb])
            hermi_tasks_ris.extend([1, 2])
        else:
            dms_tasks_full.extend([[dmz1doo[k], oo0]])
            j_tasks_full.extend([1.])
            k_tasks_full.extend([0.])
            hermi_tasks_full.extend([1])

            dms_tasks_ris.extend([[dm_xpyI_sym[k], dm_xpyJ_sym[k]]])
            j_tasks_ris.extend([1.])
            k_tasks_ris.extend([0.])
            hermi_tasks_ris.extend([1])

    if grad_state_idx is not None:
        dmz1doo_g = z1ao_g + dmzoo_g
        dm1_g = (dmz1doo_g + dmz1doo_g.T) * 0.5 + oo0
        dm2_g = (dmz1doo_g + dmz1doo_g.T) * 0.5
        dm3_g = dmxpy_g + dmxpy_g.T
        dm4_g = dmxmy_g - dmxmy_g.T
        
        dms_g_full = [dm1_g, dm2_g]
        j_g_full = [1., -1.]
        
        if with_k:
            k_g_full = [hyb, -hyb]
        else:
            k_g_full = [0., 0.]
            
        hermi_g_full = [1, 1]

        dms_g_ris = [dm3_g, dm4_g]
        j_g_ris = [2., 0.]
        
        if with_k:
            k_g_ris = [2*hyb, -2*hyb]
        else:
            k_g_ris = [0., 0.]
            
        hermi_g_ris = [1, 2]

        dms_tasks_full.extend(dms_g_full)
        j_tasks_full.extend(j_g_full)
        k_tasks_full.extend(k_g_full)
        hermi_tasks_full.extend(hermi_g_full)

        dms_tasks_ris.extend(dms_g_ris)
        j_tasks_ris.extend(j_g_ris)
        k_tasks_ris.extend(k_g_ris)
        hermi_tasks_ris.extend(hermi_g_ris)

    ejk_all = td_nac.jk_energies_per_atom(dms_tasks_full, j_tasks_full, k_tasks_full if with_k else None, 
        hermi=hermi_tasks_full, sum_results=False)
    ejk_ris = tdrks_ris.jk_energies_per_atom(
        mf_J, mf_K, mol, dms_tasks_ris, j_tasks_ris, k_tasks_ris, hermi=hermi_tasks_ris, sum_results=False)
    ejk_all = cp.asarray(ejk_all)
    ejk_ris = cp.asarray(ejk_ris)

    if with_k and omega != 0:
        beta = alpha - hyb
        k_omega_full = []
        k_omega_ris = []
        for k in range(n_pairs):
            k_omega_full.extend([beta])
        if grad_state_idx is not None:
            k_omega_full.extend([beta, -beta])

        for k in range(n_pairs):
            k_omega_ris.extend([beta, -beta])
        if grad_state_idx is not None:
            k_omega_ris.extend([2*beta, -2*beta])
            
        ejk_temp_all = td_nac.jk_energies_per_atom(dms_tasks_full, None, k_omega_full, hermi=hermi_tasks_full, omega=omega, sum_results=False)
        ejk_temp_ris = tdrks_ris.jk_energies_per_atom(
            mf_J, mf_K, mol, dms_tasks_ris, None, k_omega_ris, hermi=hermi_tasks_ris, omega=omega,
            sum_results=False)
        ejk_all += cp.asarray(ejk_temp_all)
        ejk_ris += cp.asarray(ejk_temp_ris)
    if with_k:
        n_dms_per_pair = 1
        n_dms_per_pari_ris = 2
        ejk_nacv = ejk_all[:n_dms_per_pair*n_pairs].reshape(n_pairs, n_dms_per_pair, natm, 3).sum(axis=1) * 2.0
        ejk_nacv+= ejk_ris[:n_dms_per_pari_ris*n_pairs].reshape(n_pairs, n_dms_per_pari_ris, natm, 3).sum(axis=1) * 2.0
    else:
        n_dms_per_pair = 1
        ejk_all += ejk_ris
        ejk_nacv = ejk_all[:n_dms_per_pair*n_pairs].reshape(n_pairs, n_dms_per_pair, natm, 3).sum(axis=1) * 2.0
    t_debug_7 = time.time()
    fxcz1_all = _contract_xc_kernel_batched(
        td_nac, mf.xc, cp.concatenate([z1aoS, z1ao_g[None, ...]], axis=0) if grad_state_idx is not None else z1aoS, 
        None, False, False, True)[0]
        
    veff1_0_batch = vxc1[:, 1:]
    veff1_1_batch = f1ooIJ[:, 1:] + fxcz1_all[:n_pairs, 1:]
    t_debug_8 = time.time()
    de = dh_td - ds + ejk_nacv

    dveff1_0 = contract_veff_dm_batched(mol, veff1_0_batch, dmz1doo, hermi=0)
    oo0_batch = cp.repeat(oo0[None, ...], n_pairs, axis=0)
    dveff1_1 = contract_veff_dm_batched(mol, veff1_1_batch, oo0_batch, hermi=1) * 0.5

    rIJoo_ao = cp.einsum('ui, nij, vj -> nuv', orbo, rIJoo, orbo)
    rIJvv_ao = cp.einsum('ua, nab, vb -> nuv', orbv, rIJvv, orbv)
    TIJoo_ao = cp.einsum('ui, nij, vj -> nuv', orbo, TIJoo, orbo)
    TIJvv_ao = cp.einsum('ua, nab, vb -> nuv', orbv, TIJvv, orbv)
    
    dsxy = contract_h1e_dm_batched(mol, s1, rIJoo_ao * dE[:, None, None], hermi=1)
    dsxy += contract_h1e_dm_batched(mol, s1, rIJvv_ao * dE[:, None, None], hermi=1)
    
    dsxy_etf = contract_h1e_dm_batched(mol, s1, TIJoo_ao * dE[:, None, None], hermi=1)
    dsxy_etf += contract_h1e_dm_batched(mol, s1, TIJvv_ao * dE[:, None, None], hermi=1)
    
    de += dh1e_td + dveff1_0 + dveff1_1
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
        if with_k:
            n_dms_per_pair = 1
            n_dms_per_pari_ris = 2
            ejk_g = ejk_all[n_dms_per_pair*n_pairs:].sum(axis=0)
            ejk_g+= ejk_ris[n_dms_per_pari_ris*n_pairs:].sum(axis=0)
        else:
            n_dms_per_pair = 1
            ejk_g = ejk_all[n_dms_per_pair*n_pairs:].sum(axis=0)

        fxcz1_g = fxcz1_all[-1]
        veff1_0_g = vxc1_g[1:]
        veff1_1_g = (f1oo_g[1:] + fxcz1_g[1:]) * 2.0

        dveff1_0_g = rhf_grad.contract_h1e_dm(mol, veff1_0_g, oo0 + dmz1doo_g, hermi=0)
        dveff1_1_g = rhf_grad.contract_h1e_dm(mol, veff1_1_g, oo0, hermi=1) * .25

        de_grad = dh_ground + dh_td_g - ds_g + ejk_g
        de_grad += dh1e_ground + dh1e_td_g + cp.asarray(dveff1_0_g + dveff1_1_g)

        if atmlst is not None:
            de_grad = de_grad[atmlst]
        
        de_grad += cp.asarray(mf_grad.grad_nuc(mol, atmlst))
        if mol.symmetry:
            de_grad = cp.asarray(mf_grad.symmetrize(de_grad.get(), atmlst))
            
        results['gradient'] = de_grad.get()
    t_debug_9 = time.time()
    time_list = [t_debug_0, t_debug_1, t_debug_2, t_debug_3, t_debug_4, t_debug_5, t_debug_6, t_debug_7, t_debug_8, t_debug_9]
    time_list = [time_list[i+1] - time_list[i] for i in range(len(time_list) - 1)]
    if verbose >= logger.NOTE:
        for i, t in enumerate(time_list):
            logger.note(td_nac, f"Time for step {i}: {t:.6f}s")
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