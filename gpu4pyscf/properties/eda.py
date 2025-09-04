# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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

from pyscf import gto
from gpu4pyscf.dft import rks
from gpu4pyscf.scf import hf as rhf
import numpy as np
import cupy as cp
from pyscf.data import nist
from pyscf.gto.mole import conc_mol
from gpu4pyscf.gto.int3c1e import int1e_grids
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.lib.cupy_helper import eigh as generalized_eigh
from cupyx.scipy.sparse.linalg import LinearOperator, cg
from cupyx.scipy.linalg import expm as matrix_exp
from gpu4pyscf.scf import cphf
from gpu4pyscf.lib.cupy_helper import pack_tril, unpack_tril
from gpu4pyscf.lib.diis import DIIS
from gpu4pyscf.lib import logger
import time
import warnings

# np.set_printoptions(linewidth = np.iinfo(np.int32).max, threshold = np.iinfo(np.int32).max, precision = 16, suppress = True)

def merge_mol(mol_list):
    n_frag = len(mol_list)
    assert n_frag >= 1
    merged = mol_list[0]
    for i in range(1, n_frag):
        merged = conc_mol(merged, mol_list[i])
    merged.stdout = mol_list[0].stdout # Same change as https://github.com/pyscf/pyscf/pull/2900
    return merged

def _get_total_system_Fock_and_energy(mf_sum, dm, H1e):
    vhf = mf_sum.get_veff(mf_sum.mol, dm)
    F = mf_sum.get_fock(h1e = H1e, dm = dm, vhf = vhf)
    E = mf_sum.energy_elec(dm = dm, h1e = H1e, vhf = vhf)[0] + mf_sum.energy_nuc()
    return F, E

def _get_fragment_Fock_and_energy(mf_list, mf_sum, H1e_list, nocc_offsets, mocc_sum):
    n_frag = len(mf_list)

    Fock_list = []
    energy_list = []
    for i_frag in range(n_frag):
        mf_i = mf_list[i_frag]

        H1e_i = H1e_list[i_frag]
        mocc_i = mocc_sum[:, nocc_offsets[i_frag] : nocc_offsets[i_frag + 1]]
        dm_i = 2 * mocc_i @ mocc_i.T
        vhf_i = mf_sum.get_veff(mf_sum.mol, dm_i)
        F_i = mf_sum.get_fock(h1e = H1e_i, dm = dm_i, vhf = vhf_i)
        E_i = mf_sum.energy_elec(dm = dm_i, h1e = H1e_i, vhf = vhf_i)[0] + mf_i.energy_nuc()
        if mf_i.do_disp():
            E_i += mf_i.get_dispersion()

        dm_i = None
        vhf_i = None
        Fock_list.append(F_i)
        energy_list.append(E_i)

    return Fock_list, energy_list

def _get_total_system_xc_energy(mf_sum, dm):
    # This function computes the K+XC energy of the given functional (specified in mf_sum)
    # and the given density matrix.
    # The algorithm is: First compute J+K+XC energy, then subtract J energy.
    # This is a hacky way to make it compatible with both HF and KS objects,
    # because in both cases the J,K,XC matrices are summed into one vhf matrix,
    # it is hard to extract the K+XC component, especially for HF (in KS, K+XC energy is stored in exc).
    E_j_plus_xc = mf_sum.energy_elec(dm = dm, h1e = dm * 0)[0]
    J = mf_sum.get_j(mf_sum.mol, dm, hermi = 1)
    E_j = 0.5 * cp.einsum('ij,ji->', J, dm)
    return float(E_j_plus_xc - E_j)

def _get_fragment_xc_energy_sum(mf_sum, nocc_offsets, mocc_sum):
    # See comments in the above function.
    n_frag = len(nocc_offsets) - 1
    E_sum = 0
    for i_frag in range(n_frag):
        mocc_i = mocc_sum[:, nocc_offsets[i_frag] : nocc_offsets[i_frag + 1]]
        D_i = 2 * mocc_i @ mocc_i.T
        E_j_plus_xc_i = mf_sum.energy_elec(dm = D_i, h1e = D_i * 0)[0]
        J_i = mf_sum.get_j(mf_sum.mol, D_i, hermi = 1)
        E_j_i = 0.5 * cp.einsum('ij,ji->', J_i, D_i)
        D_i = None
        E_sum += E_j_plus_xc_i - E_j_i
    return E_sum

def get_eda_classical_electrostatic_energy(mf_list, _make_mf, eda_cache):
    n_frag = len(mf_list)
    assert n_frag >= 1

    if "mol_sum" in eda_cache:
        mol_sum = eda_cache["mol_sum"]
    else:
        mol_sum = merge_mol([mf.mol for mf in mf_list])
        eda_cache["mol_sum"] = mol_sum
    if "mf_sum" in eda_cache:
        mf_sum = eda_cache["mf_sum"]
    else:
        mf_sum = _make_mf(mol_sum, if_kernel = False)
        eda_cache["mf_sum"] = mf_sum

    classical_electrostatic_energy_pair = np.zeros((n_frag, n_frag))

    for i_frag in range(n_frag):
        for j_frag in range(i_frag + 1, n_frag):
            mf_i = mf_list[i_frag]
            mf_j = mf_list[j_frag]

            nao_i = mf_i.mol.nao
            nao_j = mf_j.mol.nao
            dm_i = mf_i.make_rdm1()
            dm_j = mf_j.make_rdm1()

            dm_i_resized = cp.zeros([nao_i + nao_j, nao_i + nao_j])
            dm_i_resized[0 : nao_i, 0 : nao_i] = dm_i
            dm_j_resized = cp.zeros([nao_i + nao_j, nao_i + nao_j])
            dm_j_resized[nao_i : nao_i+nao_j, nao_i : nao_i+nao_j] = dm_j

            mol_merged = merge_mol([mf_i.mol, mf_j.mol])
            mf_merged = _make_mf(mol_merged, if_kernel = False)
            J_j = mf_merged.get_j(mol_merged, dm_i_resized)
            E_ee_ij = contract('ij,ij->', dm_j_resized, J_j)
            mf_merged = None
            dm_i_resized = None
            dm_j_resized = None

            nucleus_position_i = mf_i.mol.atom_coords(unit = "B")
            nucleus_charge_i = mf_i.mol.atom_charges()
            nucleus_position_j = mf_j.mol.atom_coords(unit = "B")
            nucleus_charge_j = mf_j.mol.atom_charges()

            nucleus_position_i = cp.asarray(nucleus_position_i)
            nucleus_charge_i = cp.asarray(nucleus_charge_i)
            nucleus_position_j = cp.asarray(nucleus_position_j)
            nucleus_charge_j = cp.asarray(nucleus_charge_j)

            V1e_j = int1e_grids(mf_i.mol, nucleus_position_j, dm = dm_i)
            E_en_ij = V1e_j.T @ nucleus_charge_j
            V1e_i = int1e_grids(mf_j.mol, nucleus_position_i, dm = dm_j)
            E_en_ji = V1e_i.T @ nucleus_charge_i
            dm_i = None
            dm_j = None

            E_nn_ij = mol_merged.enuc - mf_i.mol.enuc - mf_j.mol.enuc

            classical_electrostatic_energy_pair[i_frag, j_frag] = E_ee_ij - E_en_ij - E_en_ji + E_nn_ij
            logger.debug(mf_sum, f"Classical electrostatic energy between fragment {i_frag} and {j_frag} is "
                                 f"{classical_electrostatic_energy_pair[i_frag, j_frag]} Hartree")

    classical_electrostatic_energy = float(np.sum(classical_electrostatic_energy_pair))
    eda_cache["classical_electrostatic_energy_pair"] = classical_electrostatic_energy_pair
    eda_cache["classical_electrostatic_energy"] = classical_electrostatic_energy
    return classical_electrostatic_energy

def get_eda_electrostatic_energy(mf_list, _make_mf, eda_cache, build_orbital_hessian = False):
    n_frag = len(mf_list)
    assert n_frag >= 1

    if "mol_sum" in eda_cache:
        mol_sum = eda_cache["mol_sum"]
    else:
        mol_sum = merge_mol([mf.mol for mf in mf_list])
        eda_cache["mol_sum"] = mol_sum
    if "mf_sum" in eda_cache:
        mf_sum = eda_cache["mf_sum"]
    else:
        mf_sum = _make_mf(mol_sum, if_kernel = False)
        eda_cache["mf_sum"] = mf_sum

    cp.cuda.runtime.deviceSynchronize()
    time_electrostatic_start = time.time()

    mocc_list = []
    for i_frag in range(n_frag):
        mf_i = mf_list[i_frag]
        mo_coeff_i = mf_i.mo_coeff
        assert mo_coeff_i.ndim == 2
        mo_occ_i = mf_i.mo_occ
        assert mo_occ_i.ndim == 1
        mocc_i = mo_coeff_i[:, mo_occ_i > 0]
        mocc_list.append(mocc_i)
    nao_offsets  = np.cumsum([0] + [mocc.shape[0] for mocc in mocc_list])
    nocc_offsets = np.cumsum([0] + [mocc.shape[1] for mocc in mocc_list])
    nao_sum  =  nao_offsets[-1]
    nocc_sum = nocc_offsets[-1]

    mocc_sum = cp.zeros([nao_sum, nocc_sum])
    for i_frag in range(n_frag):
        mocc_i = mocc_list[i_frag]
        mocc_i = cp.asarray(mocc_i)
        nao_i, nocc_i = mocc_i.shape
        i_ao_offset, i_occ_offset = nao_offsets[i_frag], nocc_offsets[i_frag]
        mocc_sum[i_ao_offset : i_ao_offset + nao_i, i_occ_offset : i_occ_offset + nocc_i] = mocc_i

    S = mol_sum.intor_symmetric('int1e_ovlp')
    S = cp.asarray(S)

    CTSC = mocc_sum.T @ S @ mocc_sum
    # D_frozen = 2 * mocc_sum @ cp.linalg.solve(CTSC, mocc_sum.T)

    ### Note: The (C^T S C)^-1/2 result must be near identity
    CTSC_eigenvalues, CTSC_eigenvectors = cp.linalg.eigh(CTSC)
    assert cp.min(CTSC_eigenvalues) > 1e-6
    CTSC_minus_half = CTSC_eigenvectors @ cp.diag(CTSC_eigenvalues**-0.5) @ CTSC_eigenvectors.T

    mocc_renormalized = mocc_sum @ CTSC_minus_half
    mocc_sum = mocc_renormalized
    # D_frozen = 2 * mocc_renormalized @ mocc_renormalized.T
    # print(cp.max(cp.abs(mocc_renormalized.T @ S @ mocc_renormalized - cp.eye(nocc_sum))))
    # print(cp.max(cp.abs(2 * mocc_renormalized @ mocc_renormalized.T - D_frozen)))

    K1e = cp.asarray(mol_sum.intor_symmetric('int1e_kin'))
    H1e_list = []
    for i_frag in range(n_frag):
        assert not mol_sum._pseudo, "Pseudo potential not implemented for EDA"
        assert not mol_sum.nucmod
        assert len(mol_sum._ecpbas) == 0, "ECP not implemented for EDA"
        mf_i = mf_list[i_frag]
        H1e_i = K1e + int1e_grids(mol_sum, mf_i.mol.atom_coords(unit = "B"), charges = -mf_i.mol.atom_charges())
        H1e_list.append(H1e_i)
    K1e = None

    logger.info(mf_sum, "Orthogonal Decomposition of the Initial Supersystem Wavefunction")
    Fock_list, energy_list = _get_fragment_Fock_and_energy(mf_list, mf_sum, H1e_list, nocc_offsets, mocc_sum)
    energy_sum = float(sum(energy_list))
    logger.info(mf_sum, f"Cycle {0:2d}: energy = {energy_sum}")
    energy_unrelaxed = energy_sum
    scf_conv = False

    cp.cuda.runtime.deviceSynchronize()
    time_electrostatic_before_scf = time.time()
    logger.debug(mf_sum, f"EDA electrostatic time: before SCF = {time_electrostatic_before_scf - time_electrostatic_start} s")

    for cycle in range(mf_sum.max_cycle):
        cp.cuda.runtime.deviceSynchronize()
        time_electrostatic_scf_start = time.time()

        def upper_trinagular_to_pair_index(i, j, n):
            return (2 * n - 2 - i) * (i - 1) // 2 + n - 1 + j - i - 1
        nocc_count = nocc_offsets[1:] - nocc_offsets[:-1]
        nocc_frag_pair_count = [int(nocc_count[i] * nocc_count[j]) for i in range(n_frag) for j in range(i+1, n_frag)]
        nocc_frag_pair_offsets = np.cumsum([0] + nocc_frag_pair_count)
        nocc_frag_pair_sum = nocc_frag_pair_offsets[-1]

        orbital_gradient = cp.zeros(nocc_frag_pair_sum)
        for i_frag in range(n_frag):
            mocc_i = mocc_sum[:, nocc_offsets[i_frag] : nocc_offsets[i_frag + 1]]
            F_i = Fock_list[i_frag]

            for j_frag in range(i_frag + 1, n_frag):
                mocc_j = mocc_sum[:, nocc_offsets[j_frag] : nocc_offsets[j_frag + 1]]
                F_j = Fock_list[j_frag]

                orbital_gradient_ij = 2 * mocc_i.T @ (F_j - F_i) @ mocc_j
                ij_frag_pair = upper_trinagular_to_pair_index(i_frag, j_frag, n_frag)
                orbital_gradient[nocc_frag_pair_offsets[ij_frag_pair] : nocc_frag_pair_offsets[ij_frag_pair + 1]] = \
                    orbital_gradient_ij.reshape(nocc_count[i_frag] * nocc_count[j_frag])
                orbital_gradient_ij = None

        if build_orbital_hessian:
            orbital_hessian = cp.zeros([nocc_frag_pair_sum, nocc_frag_pair_sum])
            for i_frag in range(n_frag):
                mocc_i = mocc_sum[:, nocc_offsets[i_frag] : nocc_offsets[i_frag + 1]]
                F_i = Fock_list[i_frag]

                for j_frag in range(i_frag + 1, n_frag):
                    mocc_j = mocc_sum[:, nocc_offsets[j_frag] : nocc_offsets[j_frag + 1]]
                    F_j = Fock_list[j_frag]

                    ij_frag_pair = upper_trinagular_to_pair_index(i_frag, j_frag, n_frag)

                    for k_frag in range(0, n_frag):
                        mocc_k = mocc_sum[:, nocc_offsets[k_frag] : nocc_offsets[k_frag + 1]]
                        F_k = Fock_list[k_frag]

                        for l_frag in range(k_frag + 1, n_frag):
                            mocc_l = mocc_sum[:, nocc_offsets[l_frag] : nocc_offsets[l_frag + 1]]
                            F_l = Fock_list[l_frag]

                            kl_frag_pair = upper_trinagular_to_pair_index(k_frag, l_frag, n_frag)
                            orbital_hessian_ijkl = cp.zeros([nocc_count[i_frag], nocc_count[j_frag], nocc_count[k_frag], nocc_count[l_frag]])

                            if k_frag == j_frag:
                                orbital_hessian_il = mocc_i.T @ (F_l - 2 * F_k + F_i) @ mocc_l
                                for i_occ in range(nocc_count[j_frag]):
                                    orbital_hessian_ijkl[:, i_occ, i_occ, :] += orbital_hessian_il
                                orbital_hessian_il = None
                            if l_frag == j_frag:
                                orbital_hessian_ik = - mocc_i.T @ (F_k - 2 * F_l + F_i) @ mocc_k
                                for i_occ in range(nocc_count[j_frag]):
                                    orbital_hessian_ijkl[:, i_occ, :, i_occ] += orbital_hessian_ik
                                orbital_hessian_ik = None
                            if l_frag == i_frag:
                                orbital_hessian_jk = mocc_j.T @ (F_k - 2 * F_l + F_j) @ mocc_k
                                for i_occ in range(nocc_count[i_frag]):
                                    orbital_hessian_ijkl[i_occ, :, :, i_occ] += orbital_hessian_jk
                                orbital_hessian_jk = None
                            if k_frag == i_frag:
                                orbital_hessian_jl = - mocc_j.T @ (F_l - 2 * F_k + F_j) @ mocc_l
                                for i_occ in range(nocc_count[i_frag]):
                                    orbital_hessian_ijkl[i_occ, :, i_occ, :] += orbital_hessian_jl
                                orbital_hessian_jl = None

                            orbital_hessian[nocc_frag_pair_offsets[ij_frag_pair] : nocc_frag_pair_offsets[ij_frag_pair + 1],
                                            nocc_frag_pair_offsets[kl_frag_pair] : nocc_frag_pair_offsets[kl_frag_pair + 1]] = \
                                orbital_hessian_ijkl.reshape([nocc_count[i_frag] * nocc_count[j_frag], nocc_count[k_frag] * nocc_count[l_frag]])
                            orbital_hessian_ijkl = None
            F_i = None
            F_j = None
            F_k = None
            F_l = None

            newton_direction = -cp.linalg.solve(orbital_hessian, orbital_gradient)
            assert not np.isnan(newton_direction).any()
            orbital_hessian = None

        else:
            conjugate_gradient_initial_guess = cp.zeros(nocc_frag_pair_sum)
            for i_frag in range(n_frag):
                mocc_i = mocc_sum[:, nocc_offsets[i_frag] : nocc_offsets[i_frag + 1]]
                F_i = Fock_list[i_frag]

                for j_frag in range(i_frag + 1, n_frag):
                    mocc_j = mocc_sum[:, nocc_offsets[j_frag] : nocc_offsets[j_frag + 1]]
                    F_j = Fock_list[j_frag]

                    preconditioner_ii = 2 * mocc_i.T @ (F_j - F_i) @ mocc_i
                    preconditioner_jj = 2 * mocc_j.T @ (F_i - F_j) @ mocc_j

                    preconditioner_ii_eigenvalues, preconditioner_ii_eigenvectors = cp.linalg.eigh(preconditioner_ii)
                    preconditioner_jj_eigenvalues, preconditioner_jj_eigenvectors = cp.linalg.eigh(preconditioner_jj)

                    preconditioner_ijij_diagonal = preconditioner_ii_eigenvalues[:, cp.newaxis] + preconditioner_jj_eigenvalues[cp.newaxis, :]
                    preconditioner_ijij_diagonal_inv = preconditioner_ijij_diagonal**-1
                    preconditioner_ijij_diagonal_inv[cp.abs(preconditioner_ijij_diagonal) < 1e-14] = 0
                    preconditioner_ijij_diagonal = None
                    preconditioner_ii_eigenvalues = None
                    preconditioner_jj_eigenvalues = None

                    ij_frag_pair = upper_trinagular_to_pair_index(i_frag, j_frag, n_frag)
                    orbital_gradient_ij = orbital_gradient[nocc_frag_pair_offsets[ij_frag_pair] : nocc_frag_pair_offsets[ij_frag_pair + 1]]
                    orbital_gradient_ij = orbital_gradient_ij.reshape(nocc_count[i_frag], nocc_count[j_frag])

                    conjugate_gradient_initial_guess_ij = \
                        preconditioner_ii_eigenvectors.T @ orbital_gradient_ij @ preconditioner_jj_eigenvectors
                    conjugate_gradient_initial_guess_ij = \
                        preconditioner_ijij_diagonal_inv * conjugate_gradient_initial_guess_ij
                    conjugate_gradient_initial_guess_ij = \
                        preconditioner_ii_eigenvectors @ conjugate_gradient_initial_guess_ij @ preconditioner_jj_eigenvectors.T
                    orbital_gradient_ij = None
                    preconditioner_ii_eigenvectors = None
                    preconditioner_jj_eigenvectors = None
                    preconditioner_ijij_diagonal_inv = None

                    conjugate_gradient_initial_guess[nocc_frag_pair_offsets[ij_frag_pair] : nocc_frag_pair_offsets[ij_frag_pair + 1]] = \
                        conjugate_gradient_initial_guess_ij.reshape(nocc_count[i_frag] * nocc_count[j_frag])
                    conjugate_gradient_initial_guess_ij = None
            F_i = None
            F_j = None

            def left_multiple_orbital_hessian(x):
                y = cp.zeros_like(x)
                for i_frag in range(n_frag):
                    mocc_i = mocc_sum[:, nocc_offsets[i_frag] : nocc_offsets[i_frag + 1]]
                    F_i = Fock_list[i_frag]

                    for j_frag in range(i_frag + 1, n_frag):
                        mocc_j = mocc_sum[:, nocc_offsets[j_frag] : nocc_offsets[j_frag + 1]]
                        F_j = Fock_list[j_frag]

                        ij_frag_pair = upper_trinagular_to_pair_index(i_frag, j_frag, n_frag)
                        x_ij = x[nocc_frag_pair_offsets[ij_frag_pair] : nocc_frag_pair_offsets[ij_frag_pair + 1]]
                        x_ij = x_ij.reshape(nocc_count[i_frag], nocc_count[j_frag])

                        for k_frag in range(0, n_frag):
                            mocc_k = mocc_sum[:, nocc_offsets[k_frag] : nocc_offsets[k_frag + 1]]
                            F_k = Fock_list[k_frag]

                            for l_frag in range(k_frag + 1, n_frag):
                                mocc_l = mocc_sum[:, nocc_offsets[l_frag] : nocc_offsets[l_frag + 1]]
                                F_l = Fock_list[l_frag]

                                kl_frag_pair = upper_trinagular_to_pair_index(k_frag, l_frag, n_frag)
                                y_kl = cp.zeros([nocc_count[k_frag], nocc_count[l_frag]])

                                if k_frag == j_frag:
                                    orbital_hessian_il = mocc_i.T @ (F_l - 2 * F_k + F_i) @ mocc_l
                                    y_kl += (orbital_hessian_il.T @ x_ij).T
                                    orbital_hessian_il = None
                                if l_frag == j_frag:
                                    orbital_hessian_ik = - mocc_i.T @ (F_k - 2 * F_l + F_i) @ mocc_k
                                    y_kl += orbital_hessian_ik.T @ x_ij
                                    orbital_hessian_ik = None
                                if l_frag == i_frag:
                                    orbital_hessian_jk = mocc_j.T @ (F_k - 2 * F_l + F_j) @ mocc_k
                                    y_kl += (x_ij @ orbital_hessian_jk).T
                                    orbital_hessian_jk = None
                                if k_frag == i_frag:
                                    orbital_hessian_jl = - mocc_j.T @ (F_l - 2 * F_k + F_j) @ mocc_l
                                    y_kl += x_ij @ orbital_hessian_jl
                                    orbital_hessian_jl = None

                                y[nocc_frag_pair_offsets[kl_frag_pair] : nocc_frag_pair_offsets[kl_frag_pair + 1]] += \
                                    y_kl.reshape(nocc_count[k_frag] * nocc_count[l_frag])
                                y_kl = None

                        x_ij = None

                F_i = None
                F_j = None
                F_k = None
                F_l = None

                return y

            conjugate_gradient_threshold = 1e-14
            orbital_hessian = LinearOperator(shape = (nocc_frag_pair_sum, nocc_frag_pair_sum),
                                             matvec = left_multiple_orbital_hessian,
                                             dtype = orbital_gradient.dtype)
            newton_direction, conjugate_gradient_info = cg(orbital_hessian,
                                                           orbital_gradient,
                                                           conjugate_gradient_initial_guess,
                                                           conjugate_gradient_threshold)
            newton_direction *= -1
            assert conjugate_gradient_info == 0, "Conjugate gradient for orbital hessian inverse " \
                                                 "in EDA orthogonal decomposition not converged!"
            conjugate_gradient_initial_guess = None

        Fock_list = None
        orbital_gradient = None

        orbital_rotation = cp.zeros([nocc_sum, nocc_sum])
        for i_frag in range(n_frag):
            for j_frag in range(i_frag + 1, n_frag):
                ij_frag_pair = upper_trinagular_to_pair_index(i_frag, j_frag, n_frag)

                orbital_rotation_ij = newton_direction[nocc_frag_pair_offsets[ij_frag_pair] : nocc_frag_pair_offsets[ij_frag_pair + 1]]
                orbital_rotation_ij = orbital_rotation_ij.reshape([nocc_count[i_frag], nocc_count[j_frag]])
                orbital_rotation[nocc_offsets[i_frag] : nocc_offsets[i_frag + 1],
                                 nocc_offsets[j_frag] : nocc_offsets[j_frag + 1]] = orbital_rotation_ij
                orbital_rotation[nocc_offsets[j_frag] : nocc_offsets[j_frag + 1],
                                 nocc_offsets[i_frag] : nocc_offsets[i_frag + 1]] = -orbital_rotation_ij.T
        newton_direction = None

        U = matrix_exp(orbital_rotation)
        orbital_rotation = None

        mocc_sum = mocc_sum @ U
        U = None

        cp.cuda.runtime.deviceSynchronize()
        time_electrostatic_scf_mo = time.time()
        logger.debug(mf_sum, f"EDA electrostatic time: SCF update MO = {time_electrostatic_scf_mo - time_electrostatic_scf_start} s")

        energy_previous = energy_sum
        Fock_list, energy_list = _get_fragment_Fock_and_energy(mf_list, mf_sum, H1e_list, nocc_offsets, mocc_sum)
        energy_sum = float(sum(energy_list))
        delta_energy = energy_sum - energy_previous
        logger.info(mf_sum, f"Cycle {cycle + 1:2d}: energy = {energy_sum}, delta energy = {delta_energy}")

        cp.cuda.runtime.deviceSynchronize()
        time_electrostatic_scf_fock = time.time()
        logger.debug(mf_sum, f"EDA electrostatic time: SCF update Fock = {time_electrostatic_scf_fock - time_electrostatic_scf_mo} s")

        if (abs(delta_energy) < mf_sum.conv_tol):
            scf_conv = True
            break

    if not scf_conv:
        raise RuntimeError("Orthogonal decomposition not converged!")

    cp.cuda.runtime.deviceSynchronize()
    time_electrostatic_after_scf = time.time()
    logger.debug(mf_sum, f"EDA electrostatic time: SCF total = {time_electrostatic_after_scf - time_electrostatic_before_scf} s")

    electrostatic_energy_pair = np.zeros((n_frag, n_frag))

    for i_frag in range(n_frag):
        mocc_i = mocc_sum[:, nocc_offsets[i_frag] : nocc_offsets[i_frag + 1]]
        D_i = 2 * mocc_i @ mocc_i.T

        for j_frag in range(i_frag + 1, n_frag):
            mocc_j = mocc_sum[:, nocc_offsets[j_frag] : nocc_offsets[j_frag + 1]]
            D_j = 2 * mocc_j @ mocc_j.T

            J_j = mf_sum.get_j(mol_sum, D_i)
            E_ee_ij = contract('ij,ij->', D_j, J_j)

            mf_i = mf_list[i_frag]
            mf_j = mf_list[j_frag]
            nucleus_position_i = mf_i.mol.atom_coords(unit = "B")
            nucleus_charge_i = mf_i.mol.atom_charges()
            nucleus_position_j = mf_j.mol.atom_coords(unit = "B")
            nucleus_charge_j = mf_j.mol.atom_charges()

            nucleus_position_i = cp.asarray(nucleus_position_i)
            nucleus_charge_i = cp.asarray(nucleus_charge_i)
            nucleus_position_j = cp.asarray(nucleus_position_j)
            nucleus_charge_j = cp.asarray(nucleus_charge_j)

            V1e_j = int1e_grids(mol_sum, nucleus_position_j, dm = D_i)
            E_en_ij = V1e_j.T @ nucleus_charge_j
            V1e_i = int1e_grids(mol_sum, nucleus_position_i, dm = D_j)
            E_en_ji = V1e_i.T @ nucleus_charge_i

            nucleus_rij = nucleus_position_i[:, np.newaxis, :] - nucleus_position_j[np.newaxis, :, :]
            nucleus_rij = np.linalg.norm(nucleus_rij, axis = -1)
            nucleus_qiqj_rij = nucleus_charge_i[:, np.newaxis] * nucleus_charge_j[np.newaxis, :] / nucleus_rij
            E_nn_ij = cp.sum(nucleus_qiqj_rij)

            electrostatic_energy_pair[i_frag, j_frag] = E_ee_ij - E_en_ij - E_en_ji + E_nn_ij
            logger.debug(mf_sum, f"Electrostatic energy between fragment {i_frag} and {j_frag} is "
                                 f"{electrostatic_energy_pair[i_frag, j_frag]} Hartree")

        D_i = None
        D_j = None

    cp.cuda.runtime.deviceSynchronize()
    time_electrostatic_end = time.time()
    logger.debug(mf_sum, f"EDA electrostatic time: after SCF = {time_electrostatic_end - time_electrostatic_after_scf} s")
    logger.debug(mf_sum, f"EDA electrostatic time: total = {time_electrostatic_end - time_electrostatic_start} s")

    electrostatic_energy = float(np.sum(electrostatic_energy_pair))
    eda_cache["electrostatic_energy_pair"] = electrostatic_energy_pair
    eda_cache["electrostatic_energy"] = electrostatic_energy
    eda_cache["kinetic_energy_pressure"] = energy_sum - energy_unrelaxed
    eda_cache["mocc_pauli"] = mocc_sum
    return electrostatic_energy

def get_eda_dispersion_energy(mf_list, _make_mf, eda_cache):
    n_frag = len(mf_list)
    assert n_frag >= 1

    if "mol_sum" in eda_cache:
        mol_sum = eda_cache["mol_sum"]
    else:
        mol_sum = merge_mol([mf.mol for mf in mf_list])
        eda_cache["mol_sum"] = mol_sum
    if "mf_sum" in eda_cache:
        mf_sum = eda_cache["mf_sum"]
    else:
        mf_sum = _make_mf(mol_sum, if_kernel = False)
        eda_cache["mf_sum"] = mf_sum

    cp.cuda.runtime.deviceSynchronize()
    time_dispersion_start = time.time()

    assert "mocc_pauli" in eda_cache

    S = mol_sum.intor_symmetric('int1e_ovlp')
    S = cp.asarray(S)

    mocc_list = []
    for i_frag in range(n_frag):
        mf_i = mf_list[i_frag]
        mo_coeff_i = mf_i.mo_coeff
        assert mo_coeff_i.ndim == 2
        mo_occ_i = mf_i.mo_occ
        assert mo_occ_i.ndim == 1
        mocc_i = mo_coeff_i[:, mo_occ_i >  0]

        mocc_list.append(mocc_i)

    # nao_offsets  = np.cumsum([0] + [mocc.shape[0] for mocc in mocc_list])
    nocc_offsets = np.cumsum([0] + [mocc.shape[1] for mocc in mocc_list])
    # nao_sum  =  nao_offsets[-1]
    # nocc_sum = nocc_offsets[-1]
    mocc_list = None

    mocc_sum = eda_cache["mocc_pauli"]
    CTSC = mocc_sum.T @ S @ mocc_sum
    D_frozen = 2 * mocc_sum @ cp.linalg.solve(CTSC, mocc_sum.T)

    logger.info(mf_sum, "Using Hartree-Fock XC as dispersion-free XC in EDA dispersion energy calculation")
    mf_dispersion_free_sum = _make_mf(mol_sum, dispersion_free_xc = "HF", if_kernel = False)

    if hasattr(mf_sum, "with_df") and hasattr(mf_dispersion_free_sum, "with_df"):
        # This is a hack to save memory for df cderi, it works because mf_sum and mf_dispersion_free_sum have the same mol
        # and thus same full-range 3-center integral.
        # It does NOT necessarily work with other dispersion-free functionals!
        mf_dispersion_free_sum.with_df = mf_sum.with_df

    E_frozen = _get_total_system_xc_energy(mf_sum, D_frozen)
    E_fragment_sum = _get_fragment_xc_energy_sum(mf_sum, nocc_offsets, mocc_sum)

    E_dispersion_free_frozen = _get_total_system_xc_energy(mf_dispersion_free_sum, D_frozen)
    E_dispersion_free_fragment_sum = _get_fragment_xc_energy_sum(mf_dispersion_free_sum, nocc_offsets, mocc_sum)

    cp.cuda.runtime.deviceSynchronize()
    time_dispersion_end = time.time()
    logger.debug(mf_sum, f"EDA dispersion time: total = {time_dispersion_end - time_dispersion_start} s")

    E_dispersion = (E_frozen - E_fragment_sum) - (E_dispersion_free_frozen - E_dispersion_free_fragment_sum)
    eda_cache["dispersion_energy"] = E_dispersion
    eda_cache["interfragment_dfxc_energy"] = E_dispersion_free_frozen - E_dispersion_free_fragment_sum
    return E_dispersion

def get_eda_polarization_energy(mf_list, _make_mf, eda_cache,
                                field_order = 2, virtual_singular_value_threshold = 1e-4, uncoupled_ferf = False):
    """
    Attention: The result is very sensetive to virtual_singular_value_threshold!
               If a near-zero singular vector that does not belong to FERF virtual space
               is mixed into the FERF virtual space, the result can be off by 1 kJ/mol!
    """

    n_frag = len(mf_list)
    assert n_frag >= 1

    if "mol_sum" in eda_cache:
        mol_sum = eda_cache["mol_sum"]
    else:
        mol_sum = merge_mol([mf.mol for mf in mf_list])
        eda_cache["mol_sum"] = mol_sum
    if "mf_sum" in eda_cache:
        mf_sum = eda_cache["mf_sum"]
    else:
        mf_sum = _make_mf(mol_sum, if_kernel = False)
        eda_cache["mf_sum"] = mf_sum

    cp.cuda.runtime.deviceSynchronize()
    time_polarization_start = time.time()

    assert type(field_order) is int
    if field_order == 1:
        logger.info(mf_sum, "Dipole response included for FERF (nD)")
    elif field_order == 2:
        logger.info(mf_sum, "Dipole and quadrupole response included for FERF (nDQ)")
    elif field_order == 3:
        logger.info(mf_sum, "Dipole, quadrupole and octupole response included for FERF (nDQO)")
    else:
        raise ValueError(f"Incorrect field_order ({field_order}) specified for get_eda_polarization_energy()")

    logger.info(mf_sum, "FERF Constrained Virtual Space Construction")
    G_projector_list = []
    mocc_list = []
    for i_frag in range(n_frag):
        cp.cuda.runtime.deviceSynchronize()
        time_polarization_ferf_i_start = time.time()

        mf_i = mf_list[i_frag]
        mo_coeff_i = mf_i.mo_coeff
        assert mo_coeff_i.ndim == 2
        mo_occ_i = mf_i.mo_occ
        assert mo_occ_i.ndim == 1
        mocc_i = mo_coeff_i[:, mo_occ_i >  0]
        mvir_i = mo_coeff_i[:, mo_occ_i == 0]
        mo_energy_i = mf_i.mo_energy

        mocc_list.append(mocc_i)

        mass_i = mf_i.mol.atom_mass_list()
        mass_i = np.asarray(mass_i, dtype = np.float32)
        coords_i = mf_i.mol.atom_coords(unit = "B")
        center_of_mass_i = (mass_i @ coords_i) / mass_i.sum()

        with mf_i.mol.with_common_orig(center_of_mass_i):
            assert field_order >= 1
            dipole_integral = mf_i.mol.intor('int1e_r')
            dipole_integral = cp.asarray(dipole_integral)
            dipole_integral_ai = -2 * contract('ap,dpj->daj', mvir_i.T, dipole_integral @ mocc_i)
            dipole_integral = None
            multipole_integral_ai = dipole_integral_ai
            dipole_integral_ai = None
            if field_order >= 2:
                quadrupole_integral = mf_i.mol.intor('int1e_rr')

                quadrupole_integral_trace = quadrupole_integral[0] + quadrupole_integral[4] + quadrupole_integral[8]
                quadrupole_integral[0] = quadrupole_integral[0] - quadrupole_integral_trace / 3
                quadrupole_integral[4] = quadrupole_integral[4] - quadrupole_integral_trace / 3
                quadrupole_integral[8] = quadrupole_integral[8] - quadrupole_integral_trace / 3
                quadrupole_integral_trace = None
                quadrupole_integral *= 1.5

                quadrupole_integral_spherical = np.zeros([5, mf_i.mol.nao, mf_i.mol.nao])
                quadrupole_integral_spherical[0] = (2.0/np.sqrt(3.0)) * quadrupole_integral[1] # xy
                quadrupole_integral_spherical[1] = (2.0/np.sqrt(3.0)) * quadrupole_integral[5] # yz
                quadrupole_integral_spherical[2] = quadrupole_integral[8] # z^2
                quadrupole_integral_spherical[3] = (2.0/np.sqrt(3.0)) * quadrupole_integral[2] # xz
                quadrupole_integral_spherical[4] = (1.0/np.sqrt(3.0)) * (quadrupole_integral[0] - quadrupole_integral[4]) # x^2 - y^2
                quadrupole_integral = quadrupole_integral_spherical
                quadrupole_integral_spherical = None

                quadrupole_integral = cp.asarray(quadrupole_integral)
                quadrupole_integral_ai = -2 * contract('ap,dpj->daj', mvir_i.T, quadrupole_integral @ mocc_i)
                quadrupole_integral = None
                multipole_integral_ai = cp.concatenate([multipole_integral_ai, quadrupole_integral_ai], axis=0)
                quadrupole_integral_ai = None
            if field_order >= 3:
                raise NotImplementedError("EDA polarization term field response with octupole is not tested")
                octupole_integral = mf_i.mol.intor('int1e_rrr')

                octupole_integral_trace = octupole_integral[0] + octupole_integral[4] + octupole_integral[8] # xr^2
                octupole_integral[0] -= octupole_integral_trace / 5
                octupole_integral[4] -= octupole_integral_trace / 5
                octupole_integral[8] -= octupole_integral_trace / 5
                octupole_integral_trace = octupole_integral[1] + octupole_integral[13] + octupole_integral[17] # yr^2
                octupole_integral[1] -= octupole_integral_trace / 5
                octupole_integral[13] -= octupole_integral_trace / 5
                octupole_integral[17] -= octupole_integral_trace / 5
                octupole_integral_trace = octupole_integral[2] + octupole_integral[14] + octupole_integral[26] # zr^2
                octupole_integral[2] -= octupole_integral_trace / 5
                octupole_integral[14] -= octupole_integral_trace / 5
                octupole_integral[26] -= octupole_integral_trace / 5
                octupole_integral_trace = None
                quadrupole_integral *= 2.5

                octupole_integral_spherical = np.zeros([7, mf_i.mol.nao, mf_i.mol.nao])
                octupole_integral_spherical[0] = (1.0/np.sqrt(10.0)) * (3 * octupole_integral[1] - octupole_integral[13]) # 3x^2y - y^3
                octupole_integral_spherical[1] = (2.0*np.sqrt(3.0/5.0)) * octupole_integral[5] # xyz
                octupole_integral_spherical[2] = np.sqrt(3.0/2.0) * octupole_integral[17] # yz^2
                octupole_integral_spherical[3] = octupole_integral[26] # z^3
                octupole_integral_spherical[4] = np.sqrt(3.0/2.0) * octupole_integral[8] # xz^2
                octupole_integral_spherical[5] = np.sqrt(3.0/5.0) * (octupole_integral[2] - octupole_integral[14]) # x^2z - y^2z
                octupole_integral_spherical[6] = (1.0/np.sqrt(10.0)) * (octupole_integral[0] - 3 * octupole_integral[4]) # x^3 - 3xy^2
                octupole_integral = octupole_integral_spherical
                octupole_integral_spherical = None

                octupole_integral = cp.asarray(octupole_integral)
                octupole_integral_ai = -2 * contract('ap,dpj->daj', mvir_i.T, octupole_integral @ mocc_i)
                octupole_integral = None
                multipole_integral_ai = cp.concatenate([multipole_integral_ai, octupole_integral_ai], axis=0)
                octupole_integral_ai = None
            if field_order >= 4:
                raise NotImplementedError("EDA polarization term field response higher than 3rd order (octupole) is not implemented")

        if not uncoupled_ferf:
            from gpu4pyscf.properties.polarizability import gen_vind
            fx = gen_vind(mf_i, mo_coeff_i, mo_occ_i, with_nlc = True)
            kappa_ai, _ = cphf.solve(fx, mo_energy_i, mo_occ_i, multipole_integral_ai, max_cycle = mf_i.max_cycle, tol = mf_i.conv_tol_cpscf)
        else:
            nocc_i = mocc_i.shape[1]
            epsilon_a = mo_energy_i[nocc_i:]
            epsilon_i = mo_energy_i[:nocc_i]
            epsilon_ai = 1.0 / (epsilon_a[:, cp.newaxis] - epsilon_i[cp.newaxis, :])
            kappa_ai = multipole_integral_ai * -epsilon_ai
            epsilon_ai = None
        multipole_integral_ai = None

        polarization_subspace = mocc_i.copy()
        n_field = kappa_ai.shape[0]
        for i_field in range(n_field):
            kappa_ai_singularvector_left, kappa_ai_singularvalue, kappa_ai_singularvector_right = \
                cp.linalg.svd(kappa_ai[i_field, :, :], full_matrices = False)
            del kappa_ai_singularvector_right
            kappa_ai_singularvector_left = kappa_ai_singularvector_left[:, kappa_ai_singularvalue > virtual_singular_value_threshold]
            kappa_ai_singularvalue = None
            C_kappa_pi = mvir_i @ kappa_ai_singularvector_left
            kappa_ai_singularvector_left = None
            polarization_subspace = cp.concatenate([polarization_subspace, C_kappa_pi], axis=1)
            C_kappa_pi = None
        kappa_ai = None

        ### Don't use QR, it makes the result unstable.
        polarization_subspace_singularvector_left, polarization_subspace_singularvalue, polarization_subspace_singularvector_right =  \
            cp.linalg.svd(polarization_subspace, full_matrices = False)
        del polarization_subspace_singularvector_right
        G = polarization_subspace_singularvector_left[:, polarization_subspace_singularvalue > virtual_singular_value_threshold]
        logger.info(mf_sum, f"Fragment {i_frag} FERF cutoff = {virtual_singular_value_threshold}, "
                            f"FERF singular value = {cp.array2string(polarization_subspace_singularvalue, precision = 1)}, "
                            f"the last {polarization_subspace_singularvalue.shape[0] - G.shape[1]} singular vectors are discarded.")
        polarization_subspace_singularvalue = None
        polarization_subspace_singularvector_left = None

        G_projector_list.append(G)

        cp.cuda.runtime.deviceSynchronize()
        time_polarization_ferf_i_end = time.time()
        logger.debug(mf_sum, f"EDA polarization time: fragment {i_frag} FERF construction = {time_polarization_ferf_i_end - time_polarization_ferf_i_start} s")

    nao_offsets        = np.cumsum([0] + [G.shape[0] for G in G_projector_list])
    nprojector_offsets = np.cumsum([0] + [G.shape[1] for G in G_projector_list])
    nao_sum        =        nao_offsets[-1]
    nprojector_sum = nprojector_offsets[-1]

    G = cp.zeros([nao_sum, nprojector_sum])
    for i_frag in range(n_frag):
        G[nao_offsets[i_frag] : nao_offsets[i_frag + 1],
          nprojector_offsets[i_frag] : nprojector_offsets[i_frag + 1]] = G_projector_list[i_frag]
    G_projector_list = None

    cp.cuda.runtime.deviceSynchronize()
    time_polarization_ferf_end = time.time()
    logger.debug(mf_sum, f"EDA polarization time: FERF construction total = {time_polarization_ferf_end - time_polarization_start} s")

    nocc_offsets = np.cumsum([0] + [mocc.shape[1] for mocc in mocc_list])
    nocc_sum = nocc_offsets[-1]

    mocc_sum = cp.zeros([nao_sum, nocc_sum])
    for i_frag in range(n_frag):
        mocc_i = mocc_list[i_frag]
        mocc_i = cp.asarray(mocc_i)
        mocc_sum[nao_offsets[i_frag] : nao_offsets[i_frag + 1],
                 nocc_offsets[i_frag] : nocc_offsets[i_frag + 1]] = mocc_i
    mocc_list = None

    logger.info(mf_sum, "SCF-MI for the Polarized Fragment Wavefunction")

    S = mol_sum.intor_symmetric('int1e_ovlp')
    S = cp.asarray(S)
    gamma = G.T @ S @ G

    def get_full_density(mocc_sum_projected, inv_sigma):
        D = cp.zeros([nao_sum, nao_sum])
        for i_frag in range(n_frag):
            for j_frag in range(n_frag):
                D += G[:, nprojector_offsets[i_frag] : nprojector_offsets[i_frag + 1]] @ \
                     mocc_sum_projected[nprojector_offsets[i_frag] : nprojector_offsets[i_frag + 1],
                                        nocc_offsets[i_frag] : nocc_offsets[i_frag + 1]] @ \
                     inv_sigma[nocc_offsets[i_frag] : nocc_offsets[i_frag + 1],
                               nocc_offsets[j_frag] : nocc_offsets[j_frag + 1]] @ \
                     mocc_sum_projected[nprojector_offsets[j_frag] : nprojector_offsets[j_frag + 1],
                                        nocc_offsets[j_frag] : nocc_offsets[j_frag + 1]].T @ \
                     G[:, nprojector_offsets[j_frag] : nprojector_offsets[j_frag + 1]].T
        D *= 2
        ### The expression above is identical to
        # D = 2 * G @ mocc_sum_projected @ inv_sigma @ mocc_sum_projected.T @ G.T
        ### If G is full rank orthogonal matrix, then the expression above is identical to
        # D = 2 * mocc_sum @ cp.linalg.inv(mocc_sum.T @ S @ mocc_sum) @ mocc_sum.T
        return D

    def get_Stoll_density(i_frag, mocc_sum_projected, inv_sigma):
        D_stoll_i = cp.zeros([nao_sum, nao_sum])
        for j_frag in range(n_frag):
            D_stoll_i += G[:, nprojector_offsets[j_frag] : nprojector_offsets[j_frag + 1]] @ \
                         mocc_sum_projected[nprojector_offsets[j_frag] : nprojector_offsets[j_frag + 1],
                                            nocc_offsets[j_frag] : nocc_offsets[j_frag + 1]] @ \
                         inv_sigma[nocc_offsets[j_frag] : nocc_offsets[j_frag + 1],
                                   nocc_offsets[i_frag] : nocc_offsets[i_frag + 1]] @ \
                         mocc_sum_projected[nprojector_offsets[i_frag] : nprojector_offsets[i_frag + 1],
                                            nocc_offsets[i_frag] : nocc_offsets[i_frag + 1]].T @ \
                         G[:, nprojector_offsets[i_frag] : nprojector_offsets[i_frag + 1]].T
        D_stoll_i *= 2
        ### The expression above is identical to
        # D_stoll_i = 2 * G @ mocc_sum_projected @ inv_sigma[:, nocc_offsets[i_frag] : nocc_offsets[i_frag + 1]] @ \
        #             mocc_sum_projected[nprojector_offsets[i_frag] : nprojector_offsets[i_frag + 1],
        #                                nocc_offsets[i_frag] : nocc_offsets[i_frag + 1]].T @ \
        #             G[:, nprojector_offsets[i_frag] : nprojector_offsets[i_frag + 1]].T
        return D_stoll_i

    logger.info(mf_sum, "Stoll algorithm is used for fragment constrained SCF")

    diis_list = []
    for i_frag in range(n_frag):
        diis_i = DIIS(mf_list[i_frag])
        diis_list.append(diis_i)

    # Initial guess
    mocc_sum_projected = G.T @ mocc_sum

    # Step 0
    sigma = mocc_sum_projected.T @ gamma @ mocc_sum_projected
    inv_sigma = cp.linalg.inv(sigma)
    D = get_full_density(mocc_sum_projected, inv_sigma)

    H1e = mf_sum.get_hcore()
    F, energy_sum = _get_total_system_Fock_and_energy(mf_sum, D, H1e)
    logger.info(mf_sum, f"Cycle {0:2d}: energy = {energy_sum}")
    energy_frozen = energy_sum
    scf_conv = False

    cp.cuda.runtime.deviceSynchronize()
    time_polarization_before_scf = time.time()
    logger.debug(mf_sum, f"EDA polarization time: SCF preparation = {time_polarization_before_scf - time_polarization_ferf_end} s")

    for cycle in range(mf_sum.max_cycle):
        cp.cuda.runtime.deviceSynchronize()
        time_polarization_scf_start = time.time()

        F_S_list = []
        new_mocc_sum = cp.zeros_like(mocc_sum_projected)
        for i_frag in range(n_frag):
            D_stoll_i = get_Stoll_density(i_frag, mocc_sum_projected, inv_sigma)
            F_stoll_i = G.T @ (cp.eye(nao_sum)*2 - S @ D + S @ D_stoll_i.T) @ F @ (cp.eye(nao_sum)*2 - D @ S + D_stoll_i @ S) @ G
            S_stoll_i = gamma
            D_stoll_i = None

            F_stoll_i_ii = F_stoll_i[nprojector_offsets[i_frag] : nprojector_offsets[i_frag + 1],
                                     nprojector_offsets[i_frag] : nprojector_offsets[i_frag + 1]]
            S_stoll_i_ii = S_stoll_i[nprojector_offsets[i_frag] : nprojector_offsets[i_frag + 1],
                                     nprojector_offsets[i_frag] : nprojector_offsets[i_frag + 1]]
            F_stoll_i = None
            S_stoll_i = None

            F_S_list.append((F_stoll_i_ii, S_stoll_i_ii))
            F_stoll_i_ii = None
            S_stoll_i_ii = None

        Err = cp.linalg.solve(gamma, G.T @ S @ D @ F @ (D @ S - cp.eye(nao_sum)*2) @ G) / 4

        new_mocc_sum = cp.zeros_like(mocc_sum_projected)
        for i_frag in range(n_frag):
            gamma_ii = gamma[nprojector_offsets[i_frag] : nprojector_offsets[i_frag + 1],
                             nprojector_offsets[i_frag] : nprojector_offsets[i_frag + 1]]
            Err_ii = Err[nprojector_offsets[i_frag] : nprojector_offsets[i_frag + 1],
                         nprojector_offsets[i_frag] : nprojector_offsets[i_frag + 1]]
            Err_ii = gamma_ii @ Err_ii - Err_ii.T @ gamma_ii.T
            F_ii, S_ii = F_S_list[i_frag]

            nprojector_i = nprojector_offsets[i_frag + 1] - nprojector_offsets[i_frag]
            Err_tril = pack_tril(Err_ii.reshape(-1, nprojector_i, nprojector_i))
            F_tril = pack_tril(F_ii.reshape(-1, nprojector_i, nprojector_i))
            F_new_tril = diis_list[i_frag].update(F_tril, xerr = Err_tril)
            F_new = unpack_tril(F_new_tril).reshape(F_ii.shape)
            Err_tril = None
            F_tril = None
            F_new_tril = None
            F_ii = None

            epsilon_i, C_ii = generalized_eigh(F_new, S_ii)
            assert all(epsilon_i[i] <= epsilon_i[i+1] for i in range(len(epsilon_i)-1))
            epsilon_i = None
            F_new = None
            S_ii = None

            nocc_i = nocc_offsets[i_frag + 1] - nocc_offsets[i_frag]
            new_mocc_sum[nprojector_offsets[i_frag] : nprojector_offsets[i_frag + 1],
                         nocc_offsets[i_frag] : nocc_offsets[i_frag + 1]] = C_ii[:, 0 : nocc_i]
            C_ii = None

        F_S_list = None

        mocc_sum_projected = new_mocc_sum
        new_mocc_sum = None

        cp.cuda.runtime.deviceSynchronize()
        time_polarization_scf_mo = time.time()
        logger.debug(mf_sum, f"EDA polarization time: SCF update MO = {time_polarization_scf_mo - time_polarization_scf_start} s")

        sigma = mocc_sum_projected.T @ gamma @ mocc_sum_projected
        inv_sigma = cp.linalg.inv(sigma)
        D = get_full_density(mocc_sum_projected, inv_sigma)

        energy_previous = energy_sum
        F, energy_sum = _get_total_system_Fock_and_energy(mf_sum, D, H1e)
        delta_energy = energy_sum - energy_previous
        logger.info(mf_sum, f"Cycle {cycle + 1:2d}: energy = {energy_sum}, delta energy = {delta_energy}")

        cp.cuda.runtime.deviceSynchronize()
        time_polarization_scf_fock = time.time()
        logger.debug(mf_sum, f"EDA polarization time: SCF update Fock = {time_polarization_scf_fock - time_polarization_scf_mo} s")

        if (abs(delta_energy) < mf_sum.conv_tol):
            scf_conv = True
            break

    if not scf_conv:
        raise RuntimeError("FERF subspace SCF-MI not converged!")

    cp.cuda.runtime.deviceSynchronize()
    time_polarization_end = time.time()
    logger.debug(mf_sum, f"EDA polarization time: total = {time_polarization_end - time_polarization_start} s")

    eda_cache["total_frozen_energy"] = energy_frozen
    eda_cache["polarization_energy"] = energy_sum - energy_frozen
    eda_cache["mocc_polarized"] = G @ mocc_sum_projected
    return energy_sum - energy_frozen

def get_eda_charge_transfer_energy(mf_list, _make_mf, eda_cache):
    n_frag = len(mf_list)
    assert n_frag >= 1

    assert "mocc_polarized" in eda_cache
    assert "total_frozen_energy" in eda_cache
    assert "polarization_energy" in eda_cache

    if "mol_sum" in eda_cache:
        mol_sum = eda_cache["mol_sum"]
    else:
        mol_sum = merge_mol([mf.mol for mf in mf_list])
        eda_cache["mol_sum"] = mol_sum
    if "mf_sum" in eda_cache:
        mf_sum = eda_cache["mf_sum"]
    else:
        mf_sum = _make_mf(mol_sum, if_kernel = False)
        eda_cache["mf_sum"] = mf_sum

    S = mol_sum.intor_symmetric('int1e_ovlp')
    S = cp.asarray(S)
    mocc_sum = eda_cache["mocc_polarized"]

    dm_polarized = 2 * mocc_sum @ cp.linalg.solve(mocc_sum.T @ S @ mocc_sum, mocc_sum.T)
    sum_energy = mf_sum.kernel(dm0 = dm_polarized)

    charge_transfer_energy = sum_energy - eda_cache["polarization_energy"] - eda_cache["total_frozen_energy"]
    eda_cache["total_system_energy"] = sum_energy
    eda_cache["charge_transfer_energy"] = charge_transfer_energy
    return charge_transfer_energy

def eval_ALMO_EDA_2_energies(mol_list, if_compute_gradient = False,
                             xc = "wB97X-V", xc_grid = (99,590), nlc_grid = (50,194), auxbasis = None,
                             conv_tol = 1e-10, conv_tol_cpscf = 1e-8, max_cycle = 100, verbose = 4, chkfile = None,
                             grid_response = False, auxbasis_response = True):
    """
    Main driver of absolutely localized molecular orbital (ALMO) energy decomposition analysis (EDA) version 2

    Args:
        mol_list: a list of pyscf.gto.mole.Mole objects, each mol is a fragment with atoms and basis functions specified
        if_compute_gradient: whether to compute gradients of each fragment and the total system
        other: specification of SCF

    Returns:
        (eda_result, dft_result)
        eda_result: a dict with EDA components in kJ/mol
        dft_result: a dict with field "energy", referring to fragments energies + total system energy (in order),
                    and field "gradient", referring to corresponding gradients (in order), if if_compute_gradient is True

    Computation cost:
        n fragment SCF + 1 second order SCF for frozen terms + 1 constrained SCF for polarization term + 1 total SCF

    Reference:
        - Not-so-clear definition of FERF and polarization energy:
        Horn, P. R.; Head-Gordon, M. Polarization contributions to intermolecular interactions revisited
        with fragment electric-field response functions. The Journal of Chemical Physics 2015, 143.
        doi: https://doi.org/10.1063/1.4930534

        - Clear definition of electrostatic and dispersion energy:
        Horn, P. R.; Mao, Y.; Head-Gordon, M. Defining the contributions of permanent electrostatics, Pauli repulsion,
        and dispersion in density functional theory calculations of intermolecular interaction energies.
        The Journal of chemical physics 2016, 144.
        doi: https://doi.org/10.1063/1.4942921

        - Clear definition of frozen density:
        Horn, P. R.; Head-Gordon, M. Alternative definitions of the frozen energy in energy decomposition analysis
        of density functional theory calculations. The Journal of chemical physics 2016, 144.
        doi: https://doi.org/10.1063/1.4941849

        - Overall procedure, with clear definition of Pauli and charge transfer terms:
        Horn, P. R.; Mao, Y.; Head-Gordon, M. Probing non-covalent interactions with a second generation
        energy decomposition analysis using absolutely localized molecular orbitals.
        Physical Chemistry Chemical Physics 2016, 18, 23067-23079.
        doi: https://doi.org/10.1039/C6CP03784D

        - An approximation to FERF:
        Aldossary, A.; Shen, H.; Wang, Z.; Head-Gordon, M. Uncoupled fragment electric-field response functions:
        An accelerated model for the polarization energy in energy decomposition analysis of intermolecular interactions.
        Chemical Physics Letters 2025, 862, 141825.
        doi: https://doi.org/10.1016/j.cplett.2024.141825

        - TODO: Gradient of each EDA term, frozen and polarization terms:
        Mao, Y.; Horn, P. R.; Head-Gordon, M. Energy decomposition analysis in an adiabatic picture.
        Physical Chemistry Chemical Physics 2017, 19, 5944-5958.
        doi: https://doi.org/10.1039/C6CP08039A

        - TODO: Gradient of each EDA term, classical electrostatic term:
        Aldossary, A.; Gimferrer, M.; Mao, Y.; Hao, H.; Das, A. K.; Salvador, P.; Head-Gordon, T.; Head-Gordon, M.
        Force Decomposition Analysis: A method to decompose intermolecular forces into physically relevant component contributions.
        The Journal of Physical Chemistry A 2023, 127, 1760-1774.
        doi: https://doi.org/10.1021/acs.jpca.2c08061
    """

    assert len(mol_list) > 1

    def _make_mf(mol, if_kernel = True, dispersion_free_xc = None):
        _xc = xc if dispersion_free_xc is None else dispersion_free_xc
        if _xc is None or _xc.upper() == "HF":
            mf = rhf.RHF(mol)
        else:
            mf = rks.RKS(mol, xc = _xc)
            mf.grids.atom_grid = xc_grid
            mf.nlcgrids.atom_grid = nlc_grid
        mf.conv_tol = conv_tol
        mf.conv_tol_cpscf = conv_tol_cpscf
        mf.max_cycle = max_cycle
        mf.verbose = verbose
        mf.chkfile = chkfile
        if auxbasis is not None:
            mf = mf.density_fit(auxbasis = auxbasis)
        mf.direct_scf_tol = 1e-16
        if if_kernel:
            energy = mf.kernel()
            mf.mol.stdout.flush()
            assert mf.converged
            return mf, energy
        else:
            return mf

    def _get_gradient(mf):
        grad_obj = mf.Gradients()
        grad_obj.grid_response = grid_response
        grad_obj.auxbasis_response = auxbasis_response
        gradient = grad_obj.kernel()
        if isinstance(gradient, cp.ndarray):
            gradient = gradient.get()
        mf.mol.stdout.flush()
        return grad_obj.kernel()

    n_frag = len(mol_list)
    for i_frag in range(n_frag):
        for j_frag in range(i_frag + 1, n_frag):
            if mol_list[i_frag].stdout != mol_list[j_frag].stdout:
                warnings.warn("The stdout of each mol in mol_list is not consistent. We do not guarantee which stdout to write. "
                              "Notice if the mol objects share the same \"output\" value, then the same output file is opened "
                              "more than once, and the outputs of earlier-created mol will be lost.")

    mf_list = []
    frag_energy_list = []
    frag_gradient_list = []
    for i_frag in range(n_frag):
        frag_i_mf, frag_i_energy = _make_mf(mol_list[i_frag])
        mf_list.append(frag_i_mf)
        frag_energy_list.append(float(frag_i_energy))
        if if_compute_gradient:
            frag_i_gradient = _get_gradient(frag_i_mf)
            frag_gradient_list.append(frag_i_gradient)

        if hasattr(frag_i_mf, "with_df"):
            # This is a hack to save memory for df cderi, we will never need to build JK for fragments again
            frag_i_mf.with_df = None

    log = logger.new_logger(mf_list[0], verbose)
    if if_compute_gradient:
        log.note("Force decomposition analysis not supported, only fragment and total system force calculated.")

    eda_cache = {}
    eda_classical_electrostatic = get_eda_classical_electrostatic_energy(mf_list, _make_mf, eda_cache)
    eda_electrostatic = get_eda_electrostatic_energy(mf_list, _make_mf, eda_cache)
    eda_dispersion = get_eda_dispersion_energy(mf_list, _make_mf, eda_cache)
    eda_pauli = eda_cache["kinetic_energy_pressure"] + eda_cache["interfragment_dfxc_energy"]
    eda_polarization = get_eda_polarization_energy(mf_list, _make_mf, eda_cache)
    eda_charge_transfer = get_eda_charge_transfer_energy(mf_list, _make_mf, eda_cache)
    eda_frozen = eda_cache["total_frozen_energy"] - sum(frag_energy_list)
    eda_frozen_reminder = eda_frozen - eda_dispersion - eda_electrostatic

    assert "mf_sum" in eda_cache
    total_system_energy = float(eda_cache["total_system_energy"])
    dft_result = { "energy" : frag_energy_list + [total_system_energy], "unit" : "au" }
    if if_compute_gradient:
        total_system_gradient = _get_gradient(eda_cache["mf_sum"])
        dft_result["gradient"] = frag_gradient_list + [total_system_gradient]

    hartree_to_kjmol = 10**-3 * nist.HARTREE2J * nist.AVOGADRO

    for i_frag in range(len(frag_energy_list)):
        log.log(f"Fragment {i_frag} energy = {frag_energy_list[i_frag]:.10f} Hartree")
    log.log(f"Total system energy = {total_system_energy:.10f} Hartree")
    eda_total = total_system_energy - sum(frag_energy_list)
    log.log(f"EDA frozen energy = {eda_frozen:.10f} Hartree = {eda_frozen * hartree_to_kjmol:.10f} kJ/mol")
    log.log(f"EDA total = {eda_total:.10f} Hartree = {eda_total * hartree_to_kjmol:.10f} kJ/mol")
    log.log(f"EDA classical electrostatic = {eda_classical_electrostatic:.10f} Hartree = {eda_classical_electrostatic * hartree_to_kjmol:.10f} kJ/mol")
    log.log(f"EDA electrostatic = {eda_electrostatic:.10f} Hartree = {eda_electrostatic * hartree_to_kjmol:.10f} kJ/mol")
    log.log(f"EDA dispersion = {eda_dispersion:.10f} Hartree = {eda_dispersion * hartree_to_kjmol:.10f} kJ/mol")
    log.log(f"EDA Pauli (kinetic energy pressure + interfragment exchange) = {eda_pauli:.10f} Hartree = {eda_pauli * hartree_to_kjmol:.10f} kJ/mol")
    log.log(f"EDA Pauli (frozen - electrostatic - dispersion) = {eda_frozen_reminder:.10f} Hartree = {eda_frozen_reminder * hartree_to_kjmol:.10f} kJ/mol")
    log.log(f"EDA polarization = {eda_polarization:.10f} Hartree = {eda_polarization * hartree_to_kjmol:.10f} kJ/mol")
    log.log(f"EDA charge transfer = {eda_charge_transfer:.10f} Hartree = {eda_charge_transfer * hartree_to_kjmol:.10f} kJ/mol")

    eda_result = {
        "total"                   : float(eda_total                  ) * hartree_to_kjmol,
        "frozen"                  : float(eda_frozen                 ) * hartree_to_kjmol,
        "electrostatic"           : float(eda_electrostatic          ) * hartree_to_kjmol,
        "classical electrostatic" : float(eda_classical_electrostatic) * hartree_to_kjmol,
        "dispersion"              : float(eda_dispersion             ) * hartree_to_kjmol,
        "pauli"                   : float(eda_frozen_reminder        ) * hartree_to_kjmol,
        "polarization"            : float(eda_polarization           ) * hartree_to_kjmol,
        "charge transfer"         : float(eda_charge_transfer        ) * hartree_to_kjmol,
        "unit"                    : "kJ/mol",
    }
    return eda_result, dft_result
