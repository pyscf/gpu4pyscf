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
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.scf import cphf
from gpu4pyscf.lib import utils
from gpu4pyscf.nac.tdrhf import NAC
from scipy.optimize import linear_sum_assignment


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

def get_nacv_ee_multi(td_nac, x_list, y_list, E_list, singlet=True, atmlst=None, verbose=logger.INFO):
    """
    Calculate Non-Adiabatic Coupling Vectors (NACV) for multiple excited-excited state pairs simultaneously.
    """
    print("get_nacv_ee_multi")
    if not singlet:
        raise NotImplementedError('Only supports for singlet states')

    mol = td_nac.mol
    natm = mol.natm
    mf = td_nac.base._scf
    
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

    idx_i = []
    idx_j = []
    pairs = []
    for i in range(n_states):
        for j in range(i + 1, n_states):
            idx_i.append(i)
            idx_j.append(j)
            pairs.append((i, j))
    
    idx_i = cp.asarray(idx_i)
    idx_j = cp.asarray(idx_j)
    n_pairs = len(pairs)

    xI = X_stack[idx_i]
    yI = Y_stack[idx_i]
    xJ = X_stack[idx_j]
    yJ = Y_stack[idx_j]
    
    EI = E_stack[idx_i]
    EJ = E_stack[idx_j]
    dE = EJ - EI
    
    xpyI = xI + yI
    xmyI = xI - yI
    xpyJ = xJ + yJ
    xmyJ = xJ - yJ
    
    def transform_to_ao(amp_batch):
        return cp.einsum('ua, nai, vi -> nuv', orbv, amp_batch, orbo)

    dmxpyI = transform_to_ao(xpyI)
    dmxmyI = transform_to_ao(xmyI)
    dmxpyJ = transform_to_ao(xpyJ)
    dmxmyJ = transform_to_ao(xmyJ)
    
    rIJoo = -cp.einsum('nai, naj -> nij', xJ, xI) - cp.einsum('nai, naj -> nij', yI, yJ)
    rIJvv = cp.einsum('nai, nbi -> nab', xI, xJ) + cp.einsum('nai, nbi -> nab', yJ, yI)

    TIJoo = (rIJoo + rIJoo.transpose(0, 2, 1)) * 0.5
    TIJvv = (rIJvv + rIJvv.transpose(0, 2, 1)) * 0.5

    dmzooIJ = cp.einsum('ui, nij, vj -> nuv', orbo, TIJoo, orbo) * 2.0
    dmzooIJ += cp.einsum('ua, nab, vb -> nuv', orbv, TIJvv, orbv) * 2.0

    dms_to_stack = [
        dmzooIJ,                                # 0IJ
        dmxpyI + dmxpyI.transpose(0, 2, 1),     # 1I
        dmxmyI - dmxmyI.transpose(0, 2, 1),     # 2I
        dmxpyJ + dmxpyJ.transpose(0, 2, 1),     # 1J
        dmxmyJ - dmxmyJ.transpose(0, 2, 1)      # 2J
    ]
    
    full_dms = cp.concatenate(dms_to_stack, axis=0)
    
    vj_all, vk_all = mf.get_jk(mol, full_dms, hermi=0)
    
    vj_split = cp.split(vj_all, 5, axis=0)
    vk_split = cp.split(vk_all, 5, axis=0)
    
    vj0IJ, vk0IJ = vj_split[0], vk_split[0]
    vj1I,  vk1I  = vj_split[1], vk_split[1]
    vj2I,  vk2I  = vj_split[2], vk_split[2]
    vj1J,  vk1J  = vj_split[3], vk_split[3]
    vj2J,  vk2J  = vj_split[4], vk_split[4]

    def trans_veff(veff, C):
        return reduce(cp.dot, (C.T, veff, C))
    
    def trans_veff_batch(veff_batch):
        return cp.einsum('up, nuv, vq -> npq', mo_coeff, veff_batch, mo_coeff)

    veff0doo = vj0IJ * 2 - vk0IJ
    # TODO: Solvent response batching.
    wvo = cp.einsum('pi, npq, qj -> nij', orbv, veff0doo, orbo) * 2.0
    veffI = (vj1I * 2 - vk1I) * 0.5
    veff0mopI = trans_veff_batch(veffI)
    wvo -= cp.einsum('nki, nai -> nak', veff0mopI[:, :nocc, :nocc], xpyJ) * 2.0
    wvo += cp.einsum('nac, nai -> nci', veff0mopI[:, nocc:, nocc:], xpyJ) * 2.0
    veffJ = (vj1J * 2 - vk1J) * 0.5
    veff0mopJ = trans_veff_batch(veffJ)
    wvo -= cp.einsum('nki, nai -> nak', veff0mopJ[:, :nocc, :nocc], xpyI) * 2.0
    wvo += cp.einsum('nac, nai -> nci', veff0mopJ[:, nocc:, nocc:], xpyI) * 2.0
    veffI = -vk2I * 0.5
    veff0momI = trans_veff_batch(veffI)
    wvo -= cp.einsum('nki, nai -> nak', veff0momI[:, :nocc, :nocc], xmyJ) * 2.0
    wvo += cp.einsum('nac, nai -> nci', veff0momI[:, nocc:, nocc:], xmyJ) * 2.0
    veffJ = -vk2J * 0.5
    veff0momJ = trans_veff_batch(veffJ)
    wvo -= cp.einsum('nki, nai -> nak', veff0momJ[:, :nocc, :nocc], xmyI) * 2.0
    wvo += cp.einsum('nac, nai -> nci', veff0momJ[:, nocc:, nocc:], xmyI) * 2.0

    vresp = td_nac.base.gen_response(singlet=None, hermi=1)

    def fvind(x_flat):
        n_vecs = x_flat.shape[0]
        x_batch = x_flat.reshape(n_vecs, nvir, nocc)
        dm = cp.einsum('ua, nai, vi -> nuv', orbv, x_batch * 2, orbo)
        dm_sym = dm + dm.transpose(0, 2, 1)
        
        v1ao = vresp(dm_sym) 
        resp_mo = cp.einsum('ua, nuv, vi -> nai', orbv, v1ao, orbo)
        return resp_mo.reshape(n_vecs, -1)

    rhs = (wvo / dE[:, None, None])
    
    z1_flat = cphf.solve(
        fvind,
        mo_energy,
        mo_occ,
        rhs,
        max_cycle=td_nac.cphf_max_cycle,
        tol=td_nac.cphf_conv_tol
    )[0]
    
    z1 = z1_flat.reshape(n_pairs, nvir, nocc)


    z1ao = cp.einsum('ua, nai, vi-> nuv', orbv, z1, orbo)
    z1ao_sym = z1ao + z1ao.transpose(0, 2, 1)
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
    
    z1aoS = z1ao_sym * 0.5 * dE[:, None, None]
    dmz1doo = z1aoS + dmzooIJ # P matrix
    oo0 = reduce(cp.dot, (orbo, orbo.T)) * 2.0
    
    mf_grad = td_nac.base._scf.nuc_grad_method()
    h1 = cp.asarray(mf_grad.get_hcore(mol))
    s1 = cp.asarray(mf_grad.get_ovlp(mol))
    
    dh_td = contract_h1e_dm_batched(mol, h1, dmz1doo, hermi=1)
    ds = contract_h1e_dm_batched(mol, s1, im0_ao, hermi=0)
    
    # TODO: check whether this can be batched
    dh1e_td_list = []
    for k in range(n_pairs):
        dh1e_k = int3c2e.get_dh1e(mol, dmz1doo[k])
        if len(mol._ecpbas) > 0:
            dh1e_k += rhf_grad.get_dh1e_ecp(mol, dmz1doo[k])
        dh1e_td_list.append(dh1e_k)
    dh1e_td = cp.array(dh1e_td_list)
    
    dm_xpyI_sym = dmxpyI + dmxpyI.transpose(0, 2, 1)
    dm_xpyJ_sym = dmxpyJ + dmxpyJ.transpose(0, 2, 1)
    dm_xmyI_asym = dmxmyI - dmxmyI.transpose(0, 2, 1)
    dm_xmyJ_asym = dmxmyJ - dmxmyJ.transpose(0, 2, 1)

    oo0_stack = cp.repeat(oo0[None, ...], n_pairs, axis=0)
    
    dms_grad_stack = cp.concatenate([
        dmz1doo + oo0_stack,
        dmz1doo,
        oo0_stack,
        dm_xpyI_sym + dm_xpyJ_sym,
        dm_xpyI_sym,
        dm_xpyJ_sym,
        dm_xmyI_asym + dm_xmyJ_asym,
        dm_xmyI_asym,
        dm_xmyJ_asym
    ], axis=0)
    
    j_factors = [1, -1, -1, 1, -1, -1, 0, 0, 0] * n_pairs
    k_factors = [1, -1, -1, 1, -1, -1, -1, 1, 1] * n_pairs
    
    dms_interleaved = cp.zeros((9 * n_pairs, nao, nao))
    for k in range(9):
        dms_interleaved[k::9] = dms_grad_stack[k*n_pairs : (k+1)*n_pairs]
    # TODO: temporary method
    dvhf = np.zeros((n_pairs, natm, 3))
    if getattr(td_nac, 'jk_energy_per_atom', None) is None:
        raise NotImplementedError("jk_energy_per_atom is not implemented for TDRHF, using density fitting")
    for ipair in range(n_pairs):
        dvhf[ipair] = td_nac.jk_energy_per_atom(dms_interleaved[ipair*9 : (ipair+1)*9], 
            j_factors[ipair*9 : (ipair+1)*9], k_factors[ipair*9 : (ipair+1)*9]) * 0.5
    dvhf = cp.asarray(dvhf)
    # dvhf_flat = td_nac.jk_energy_per_atom(dms_interleaved, j_factors, k_factors) * 0.5
    # dvhf = dvhf_flat.reshape(n_pairs, 9, -1, 3).sum(axis=1) # (n_pairs, natm, 3)

    de = dh_td - ds + 2 * dvhf
    
    rIJoo_ao = cp.einsum('ui, nij, vj -> nuv', orbo, rIJoo, orbo) * 2.0
    rIJvv_ao = cp.einsum('ua, nab, vb -> nuv', orbv, rIJvv, orbv) * 2.0
    
    TIJoo_ao = cp.einsum('ui, nij, vj -> nuv', orbo, TIJoo, orbo) * 2.0
    TIJvv_ao = cp.einsum('ua, nab, vb -> nuv', orbv, TIJvv, orbv) * 2.0
    
    dsxy = contract_h1e_dm_batched(mol, s1, rIJoo_ao * dE[:, None, None], hermi=1) * 0.5
    dsxy += contract_h1e_dm_batched(mol, s1, rIJvv_ao * dE[:, None, None], hermi=1) * 0.5
    
    dsxy_etf = contract_h1e_dm_batched(mol, s1, TIJoo_ao * dE[:, None, None], hermi=1) * 0.5
    dsxy_etf += contract_h1e_dm_batched(mol, s1, TIJvv_ao * dE[:, None, None], hermi=1) * 0.5
    
    de += dh1e_td
    de_etf = de + dsxy_etf
    de += dsxy

    results = {}
    pair_indices = zip(idx_i.get(), idx_j.get())
    
    for k, (i, j) in enumerate(pair_indices):
        results[(int(i), int(j))] = {
            'de': de[k],
            'de_scaled': de[k] / dE[k],
            'de_etf': de_etf[k],
            'de_etf_scaled': de_etf[k] / dE[k]
        }
        
    return results


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
    }

    def __init__(self, td):
        self.verbose = td.verbose
        self.stdout = td.stdout
        self.mol = td.mol
        self.base = td
        self.states = [1, 2]  # Default to pair (1, 2)
        self.atmlst = None
        self.results = {}     # Dictionary to store results: {(i, j): {data}}

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
        log.info("\n")
        return self

    def get_nacv_ge(self, x_list, y_list, E_list, singlet, atmlst=None, verbose=logger.INFO):
        raise NotImplementedError("NACV GE calculation is not implemented for multi-state module.")

    def get_nacv_ee_multi(self, x_list, y_list, E_list, singlet, atmlst=None, verbose=logger.INFO):
        return get_nacv_ee_multi(self, x_list, y_list, E_list, singlet, atmlst, verbose)

    def kernel(self, states=None, singlet=None, atmlst=None):

        logger.warn(self, "NAC Multi-State Module (Experimental)")

        if singlet is None:
            singlet = self.base.singlet
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst
        
        if states is not None:
            self.states = states

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

        self.results = {}
        
        has_ground = (0 in target_states)
        excited_states = [s for s in target_states if s > 0]
        
        if len(excited_states) >= 2:
            logger.info(self, f"Computing Vectorized NACV for excited states: {excited_states}")
            
            x_list = []
            y_list = []
            E_list = []
            

            for s in excited_states:
                x_list.append(self.base.xy[s-1][0])
                y_list.append(self.base.xy[s-1][1])
                E_list.append(self.base.e[s-1])
            
            ee_results = self.get_nacv_ee_multi(
                x_list, y_list, E_list, singlet, atmlst, verbose=self.verbose
            )
            
            for (local_i, local_j), res in ee_results.items():
                global_i = excited_states[local_i]
                global_j = excited_states[local_j]
                self.results[(global_i, global_j)] = res

        if has_ground:
            for s in excited_states:
                logger.info(self, f"Computing NACV for Ground (0) - Excited ({s})")
                xy_I = self.base.xy[s-1]
                E_I = self.base.e[s-1]
                
                de, de_scaled, de_etf, de_etf_scaled = self.get_nacv_ge(
                    xy_I, E_I, singlet, atmlst, verbose=self.verbose
                )
                
                self.results[(0, s)] = {
                    'de': de,
                    'de_scaled': de_scaled,
                    'de_etf': de_etf,
                    'de_etf_scaled': de_etf_scaled
                }

        self._finalize()
        return self.results

    def get_veff(self, mol=None, dm=None, j_factor=1.0, k_factor=1.0, omega=0.0, hermi=0, verbose=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.base.make_rdm1()
        vhfopt = self.base._scf._opt_gpu.get(omega, None)
        with mol.with_range_coulomb(omega):
            return rhf_grad._jk_energy_per_atom(
                mol, dm, vhfopt, j_factor=j_factor, k_factor=k_factor,
                verbose=verbose) * .5

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(self, "\n" + "="*60)
            logger.note(self, " NACV Calculation Summary")
            logger.note(self, "="*60)
            
            # Sort keys for consistent output
            for (i, j) in sorted(self.results.keys()):
                res = self.results[(i, j)]
                logger.note(self, f"\nPair ({i}, {j}):")
                logger.note(self, f"  - DE (CIS Force)      : \n{res['de']}")
                logger.note(self, f"  - DE (CIS Force) (Scaled)  : \n{res['de_scaled']}")
                logger.note(self, f"  - DE (ETF Force)      : \n{res['de_etf']}")
                logger.note(self, f"  - DE (ETF Force) (Scaled)  : \n{res['de_etf_scaled']}")

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
