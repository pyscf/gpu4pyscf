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
from gpu4pyscf.md.fssh_o1 import FSSH
from gpu4pyscf.nac.tdrhf import _wfn_overlap

class FSSH_TDDFT(FSSH):
    def __init__(self, td, states):
        nstates = len(states)
        assert td.nstates >= nstates-1
        self.td_scanner = td.as_scanner()
        # to track the phase of the ground state and excited states
        self.sign = np.ones(td.nstates+1)
        super().__init__(td.mol, states)

    def compute_electronic(self, position):
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
        # Save the previous states
        td_scanner = self.td_scanner
        mf = td_scanner._scf
        mol0 = td_scanner.mol
        mo_coeff0 = cp.asnumpy(mf.mo_coeff)
        if isinstance(td_scanner, RisBase):
            xs0 = [rescale_spin_free_amplitudes(td_scanner.xy, i-1)[0] for i in self.states]
        else:
            xs0 = [td_scanner.xy[i-1][0] for i in self.states]

        # Calculate energy and gradient for the current state
        mol = mol0.set_geom_(position, unit='Bohr', inplace=False)
        excited_energies = td_scanner(mol)
        ground_energy = mf.e_tot
        energies = np.append(ground_energy, excited_energies)
        energy = energies[self.states]  # (Nstates,)  Unit: Ha
        converged = np.append(mf.converged, td_scanner.converged)
        assert all(converged[self.states])

        tdgrad = td_scanner.Gradients()
        grad = tdgrad.kernel(state=self.cur_state)
        force = -grad

        mo_coeff = cp.asnumpy(mf.mo_coeff)
        if isinstance(td_scanner, RisBase):
            xs1 = [rescale_spin_free_amplitudes(td_scanner.xy, i-1)[0] for i in self.states]
        else:
            xs1 = [td_scanner.xy[i-1][0] for i in self.states]

        s = gto.intor_cross('int1e_ovlp', mol0, mol)

        nocc = xs0[0].shape[0]
        s_mo_ground = mo_coeff0[:, :nocc].T.dot(s).dot(mo_coeff[:, :nocc])
        self.sign[0] *= np.sign(np.linalg.det(s_mo_ground))

        for i in self.states:
            self.sign[i] *= np.sign(_wfn_overlap(mo_coeff0, mo_coeff, xs0[i-1], xs1[i-1], s))

        nstates = len(self.states)
        natm = len(position)
        Nacv = np.zeros((nstates, nstates, natm, 3))  # (Ns, Ns, Na, D)  Unit: 1/bohr
        tdnac = td_scanner.NAC()
        for i, j in self.nac_idx:
            states = (self.states[i], self.states[j])
            tdnac.states = states
            de, de_scaled, de_etf, de_etf_scaled = tdnac.kernel()
            de_etf_scaled *= self.sign[i] * self.sign[j]
            Nacv[i,j] = de_etf_scaled
            Nacv[j,i] = -de_etf_scaled

        return energy, force, Nacv

FSSH = FSSH_TDDFT
