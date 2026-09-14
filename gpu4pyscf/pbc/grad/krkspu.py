#!/usr/bin/env python
# Copyright 2025-2026 The PySCF Developers. All Rights Reserved.
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

'''
Analytical derivatives for DFT+U with kpoints sampling
'''

import numpy as np
import cupy as cp
from pyscf.pbc import gto
from pyscf.pbc.lib.kpts_helper import is_zero
from gpu4pyscf.lib.cupy_helper import asarray, contract
from gpu4pyscf.pbc.dft.krkspu import _set_U, _make_minao_lo, reference_mol
from gpu4pyscf.pbc.grad import krks as krks_grad
from gpu4pyscf.pbc.grad.rhf import _finite_diff_cells
from gpu4pyscf.pbc.gto import int1e

def generate_first_order_local_orbitals(cell, minao_ref='MINAO', kpts=None):
    kpts = kpts.reshape(-1, 3)
    nkpts = len(kpts)
    if isinstance(minao_ref, str):
        pcell = reference_mol(cell, minao_ref)
    else:
        pcell = minao_ref
    nao = cell.nao
    s = int1e.int1e_ovlp(cell+pcell, kpts)
    sAA = s[:,:nao,:nao]
    sAB = s[:,:nao,nao:]

    C0_minao = []
    wv_ks = []
    S0_lowdin = []
    for k in range(nkpts):
        C0_minao.append(cp.linalg.solve(sAA[k], sAB[k]))

        # Lowdin orthogonalization coefficients = S^{-1/2}
        S0 = sAB[k].conj().T.dot(C0_minao[k])
        w2, v = cp.linalg.eigh(S0)
        w = np.sqrt(w2)
        wv_ks.append((w, v))
        S0_lowdin.append((v/w).dot(v.conj().T))

    s = int1e.int1e_ipovlp(cell+pcell, kpts)
    sAA_ip1 = s[:,:,:nao,:nao]
    sAB_ip1 = s[:,:,:nao,nao:]
    sBA_ip1 = s[:,:,nao:,:nao]

    nao, n_minao = C0_minao[0].shape
    ao_slice = cell.aoslice_by_atom()
    minao_slice = pcell.aoslice_by_atom()
    dtype = np.result_type(*C0_minao)

    def make_coeff(atm_id):
        p0, p1 = ao_slice[atm_id,2:]
        q0, q1 = minao_slice[atm_id,2:]
        C1 = cp.empty((nkpts, 3, nao, n_minao), dtype=dtype)
        for k in range(nkpts):
            w, v = wv_ks[k]
            for n in range(3):
                sAA1 = cp.zeros((nao, nao), dtype=dtype)
                sAA1[p0:p1,:] -= sAA_ip1[k][n,p0:p1]
                sAA1[:,p0:p1] -= sAA_ip1[k][n,p0:p1].conj().T
                sAB1 = cp.zeros((nao, n_minao), dtype=dtype)
                sAB1[p0:p1,:] -= sAB_ip1[k][n,p0:p1]
                sAB1[:,q0:q1] -= sBA_ip1[k][n,q0:q1].conj().T

                S1 = C0_minao[k].conj().T.dot(sAB1)
                S1 = S1 + S1.conj().T
                S1 -= C0_minao[k].conj().T.dot(sAA1).dot(C0_minao[k])
                S1 = v.conj().T.dot(-S1).dot(v)
                S1 /= (w[:,None] + w)
                vw = v / w
                S1_lowdin = vw.dot(S1).dot(vw.conj().T)

                C1_minao = cp.linalg.solve(sAA[k], sAB1 - sAA1.dot(C0_minao[k]))
                C1[k,n] = C1_minao.dot(S0_lowdin[k])
                C1[k,n] += C0_minao[k].dot(S1_lowdin)
        return C1
    return make_coeff

def _hubbard_U_derivatives(mf, dm=None, kpts=None):
    """Return Hubbard-U atomic and strain derivatives in (natm+3, 3) rows."""
    assert mf.alpha is None
    assert mf.C_ao_lo is None
    assert mf.minao_ref is not None
    if dm is None:
        dm = mf.make_rdm1()
    if kpts is None:
        kpts = mf.kpts.reshape(-1, 3)
    nkpts = len(kpts)
    cell = mf.cell

    # Construct orthogonal minao local orbitals.
    pcell = reference_mol(cell, mf.minao_ref)
    C_ao_lo = _make_minao_lo(cell, pcell, kpts=kpts)
    U_idx, U_val = _set_U(cell, pcell, mf.U_idx, mf.U_val)[:2]
    U_idx_stack = np.hstack(U_idx)
    C0 = [C_k[:,U_idx_stack] for C_k in C_ao_lo]

    ovlp0 = int1e.int1e_ovlp(cell, kpts)
    C_inv = [C_k.conj().T.dot(S_k) for C_k, S_k in zip(C0, ovlp0)]
    # Both responses differentiate the same projected density. Build its
    # zeroth-order factors once, including C_inv D for the first-order terms.
    # Restricted densities include both spins; unrestricted densities keep
    # them separate and have twice the quadratic occupation coefficient.
    dm = cp.asarray(dm)
    unrestricted = dm.ndim == 4
    if not unrestricted:
        assert dm.ndim == 3
        dm = dm[None]
    else:
        assert dm.shape[0] == 2
    occupation_factor = 2 if unrestricted else 1
    C_dm = [[C_k.dot(dm_k) for C_k, dm_k in zip(C_inv, dm_s)] for dm_s in dm]
    dm_deriv0 = [[D_k.dot(C_k.conj().T) for D_k, C_k in zip(D_s, C_inv)]
                 for D_s in C_dm]
    dE_U = cp.zeros((cell.natm+3, 3))

    def contract_response(k, SC1):
        # Flatten derivative components: three for an atom, nine for strain.
        # This keeps the Hubbard trace contraction and its factors identical.
        SC1 = SC1.reshape(-1, *SC1.shape[-2:])
        response = cp.zeros(SC1.shape[0])
        for spin, D_s in enumerate(C_dm):
            dm_deriv1 = contract('pj,xjq->xpq', D_s[k], SC1)
            i0 = i1 = 0
            for idx, val in zip(U_idx, U_val):
                i0, i1 = i1, i1 + len(idx)
                P0 = dm_deriv0[spin][k][i0:i1,i0:i1]
                P1 = dm_deriv1[:,i0:i1,i0:i1]
                # The factor two accounts for P1 + P1.H; energy has a factor 1/2.
                response += (val / nkpts) * (
                    cp.einsum('xii->x', P1).real
                    - occupation_factor * cp.einsum('xij,ji->x', P1, P0).real)
        return response

    ovlp1 = int1e.int1e_ipovlp(cell, kpts)
    f_local_ao = generate_first_order_local_orbitals(cell, pcell, kpts)
    for atm_id, (p0, p1) in enumerate(cell.aoslice_by_atom()[:,2:]):
        C1 = f_local_ao(atm_id)
        for k in range(nkpts):
            C1_k = C1[k][:,:,U_idx_stack]
            SC1 = contract('pq,xqi->xpi', ovlp0[k], C1_k)
            SC1 -= contract('xqp,qi->xpi', ovlp1[k][:,p0:p1].conj(), C0[k][p0:p1])
            SC1[:,p0:p1] -= contract('xpq,qi->xpi', ovlp1[k][:,p0:p1], C0[k])
            dE_U[atm_id] += contract_response(k, SC1)
    ovlp1 = f_local_ao = C1 = None

    C1 = _strain_deriv_local_orbitals(cell, pcell, kpts)
    ovlp1 = cp.asarray(ovlp_strain_deriv(cell, kpts))
    nao = ovlp0.shape[-1]
    ovlp1 = ovlp1.reshape(3,3,nkpts,nao,nao)
    for k in range(nkpts):
        C1_k = C1[:,:,k][:,:,:,U_idx_stack]
        SC1 = contract('pq,xyqi->xypi', ovlp0[k], C1_k)
        SC1 += contract('xypq,qi->xypi', ovlp1[:,:,k], C0[k])
        dE_U[-3:] += contract_response(k, SC1).reshape(3, 3)
    return dE_U.get()


def ovlp_strain_deriv(cell, kpts):
    '''Strain derivatives for overlap matrix
    '''
    disp = 1e-5
    scaled_kpts = kpts.dot(cell.lattice_vectors().T)
    s = []
    for x in range(3):
        for y in range(3):
            cell1, cell2 = _finite_diff_cells(cell, x, y, disp)
            kpts1 = scaled_kpts.dot(cell1.reciprocal_vectors(norm_to=1))
            kpts2 = scaled_kpts.dot(cell2.reciprocal_vectors(norm_to=1))
            s1 = int1e.int1e_ovlp(cell1, kpts1)
            s2 = int1e.int1e_ovlp(cell2, kpts2)
            s.append((s1 - s2) / (2*disp))
    return cp.array(s)

def _strain_deriv_local_orbitals(cell, minao_ref='MINAO', kpts=None):
    if isinstance(minao_ref, str):
        pcell = reference_mol(cell, minao_ref)
    else:
        pcell = minao_ref
    scaled_kpts = kpts.dot(cell.lattice_vectors().T)
    nkpts = len(kpts)

    nao = cell.nao
    naop = pcell.nao
    if is_zero(kpts):
        C1_minao = cp.empty((3, 3, nkpts, nao, naop))
    else:
        C1_minao = cp.empty((3, 3, nkpts, nao, naop), dtype=np.complex128)
    disp = 1e-5
    for x in range(3):
        for y in range(3):
            cell1, cell2 = _finite_diff_cells(cell, x, y, disp)
            pcell1, pcell2 = _finite_diff_cells(pcell, x, y, disp)
            kpts1 = scaled_kpts.dot(cell1.reciprocal_vectors(norm_to=1))
            kpts2 = scaled_kpts.dot(cell2.reciprocal_vectors(norm_to=1))
            C1 = _make_minao_lo(cell1, pcell1, kpts=kpts1)
            C2 = _make_minao_lo(cell2, pcell2, kpts=kpts2)
            C1_minao[x,y] = (C1 - C2) / (2*disp)
    return C1_minao

class Gradients(krks_grad.Gradients):
    def energy_ee(self, dm, kpts):
        # Share local orbitals, overlap, and projected densities between responses.
        dE = _hubbard_U_derivatives(self.base, dm, kpts)
        return krks_grad.Gradients.energy_ee(self, dm, kpts) + dE
