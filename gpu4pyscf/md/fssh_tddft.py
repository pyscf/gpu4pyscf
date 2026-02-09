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

import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.lib import logger
from gpu4pyscf.md.fssh_o1 import FSSH
from gpu4pyscf.nac.tdrhf import _wfn_overlap

class FSSH_TDDFT(FSSH):
    def __init__(self, td, states):
        self.ris_zvector_solver = True

        nstates = len(states)
        assert td.nstates >= nstates-1
        self.td_scanner = td.as_scanner()
        # to track the phase of the ground state and excited states
        self.sign = np.ones(td.nstates+1)
        super().__init__(td.mol, states)

    def compute_electronic(self, position, with_nacv=True):
        """
        Calculate electronic energies, nuclear forces and nonadiabatic coupling for all states.

        This method computes the potential energies and nonadiabatic coupling for all electronic
        states at the given nuclear configuration. The forces are obtained as the
        negative gradient of the potential energy surface at the current state.

        Args:
            position (np.ndarray): Nuclear coordinates in Bohr (Natoms * 3)

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - energy: Electronic energies for all states (Nstates,) in Hartree
                - force: Nuclear forces for current state (Natoms * 3) in Ha/Bohr
                - Nacv: Nonadiabatic coupling vectors for all states (Nstates, Nstates, Natoms, 3) in 1/bohr
        """
        from gpu4pyscf.tdscf.ris import RisBase, rescale_spin_free_amplitudes
        td_scanner = self.td_scanner
        mf = td_scanner._scf
        mol0 = td_scanner.mol

        if not with_nacv:
            # Calculate energy and gradient for the current state
            mol = mol0.set_geom_(position, unit='Bohr', inplace=False)
            excited_energies = cp.asnumpy(td_scanner(mol))
            ground_energy = mf.e_tot
            energies = np.append(ground_energy, excited_energies)
            energy = energies[self.states]  # (Nstates,)  Unit: Ha
            converged = np.append(mf.converged, td_scanner.converged)
            assert all(converged[self.states])
            force = -td_scanner.Gradients().kernel(state=self.cur_state)
            return energy, force, None

        # Save the previous states
        mo_coeff0 = cp.asnumpy(mf.mo_coeff)
        nmo = mo_coeff0.shape[1]
        nocc = int(np.count_nonzero(mf.mo_occ > 0))
        nvir = nmo - nocc
        if isinstance(td_scanner, RisBase):
            xs0 = td_scanner.xy[0]
            xs0 = [xs0[i-1].reshape(nocc, nvir) for i in self.states]
        else:
            xs0 = [td_scanner.xy[i-1][0] for i in self.states]

        # Calculate energy and gradient for the current state
        mol = mol0.set_geom_(position, unit='Bohr', inplace=False)
        excited_energies = cp.asnumpy(td_scanner(mol))
        ground_energy = mf.e_tot
        energies = np.append(ground_energy, excited_energies)
        energy = energies[self.states]  # (Nstates,)  Unit: Ha
        converged = np.append(mf.converged, td_scanner.converged)
        assert all(converged[self.states])

        mo_coeff = cp.asnumpy(mf.mo_coeff)
        if isinstance(td_scanner, RisBase):
            xs1 = td_scanner.xy[0]
            xs1 = [xs1[i-1].reshape(nocc, nvir) for i in self.states]
        else:
            xs1 = [td_scanner.xy[i-1][0] for i in self.states]

        s = gto.intor_cross('int1e_ovlp', mol0, mol)
        s_mo_ground = mo_coeff0[:, :nocc].T.dot(s).dot(mo_coeff[:, :nocc])
        self.sign[0] *= np.sign(np.linalg.det(s_mo_ground))

        for i in self.states:
            state_ovlp = _wfn_overlap(mo_coeff0, mo_coeff, xs0[i-1], xs1[i-1], s)
            if abs(state_ovlp) < 0.3:
                logger.warn(mol0, f'Possible state flip detected for state {i}. '
                            f'Overlap with the previous step is {state_ovlp:.3f}.')
            self.sign[i] *= np.sign(state_ovlp)

        states = self.states
        nstates = len(states)
        nac_idx = self.nac_idx
        if nac_idx is None:
            nac_idx = [(i,j) for i in range(nstates-1) for j in range(i+1, nstates)]

        nac_pairs = [(states[i], states[j]) for i, j in nac_idx]
        force, nacv_dic = td_scanner.force_and_nacv(
            self.cur_state, nac_pairs, self.ris_zvector_solver)

        natm = mol.natm
        Nacv = np.zeros((nstates, nstates, natm, 3))  # (Ns, Ns, Na, D)  Unit: 1/bohr
        for i, j in nac_idx:
            state_i = states[i]
            state_j = states[j]
            sign = self.sign[state_i] * self.sign[state_j]
            de_etf_scaled = nacv_dic[state_i,state_j][1] * sign
            Nacv[i,j] = de_etf_scaled
            Nacv[j,i] = -de_etf_scaled

        return energy, force, Nacv

FSSH = FSSH_TDDFT
