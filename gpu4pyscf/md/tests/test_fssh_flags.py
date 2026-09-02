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

import unittest
from unittest import mock

import numpy as np

from gpu4pyscf.md import fssh_tddft
from gpu4pyscf.tdscf import rhf


class FSSHGuessFlags(unittest.TestCase):
    def test_disable_initial_guess_reuse(self):
        mol = mock.Mock()
        mol.set_geom_.return_value = mol

        scanner = mock.Mock()
        scanner.mol = mol
        scanner._scf = mock.Mock(e_tot=-1.0, converged=True)
        scanner.converged = np.array([True])
        scanner.return_value = np.array([0.1])

        nac = mock.Mock(base=scanner)
        nac._z_prev = object()
        nac._z_tasks = object()
        nac.grad_result = np.zeros((1, 3))

        fssh = object.__new__(fssh_tddft.FSSH_TDDFT)
        fssh.tddft = scanner
        fssh.tdnac_grad = nac
        fssh.states = [0, 1]
        fssh.reuse_xy_z = False
        fssh.reuse_scf_dm = False

        with mock.patch.object(fssh_tddft, 'TD_Scanner', type(scanner)):
            fssh.evaluate_pes(
                np.zeros((1, 3)), cur_state=1, with_nacv=False)

        scanner.assert_called_once_with(
            mol, reuse_scf_dm=False, reuse_td_guess=False)
        self.assertIsNone(nac._z_prev)
        self.assertIsNone(nac._z_tasks)

    def test_disable_xy_and_scf_initial_guesses(self):
        mol = mock.Mock()
        scanner = mock.Mock(device='gpu', verbose=0)
        scanner._scf.mo_coeff = np.eye(2)
        scanner._scf.mo_occ = np.array([2, 0])
        scanner._scf.return_value = -1.0
        scanner.xy = object()
        scanner.e = np.array([0.1])
        fresh_xy = object()
        scanner.init_guess.return_value = fresh_xy

        with mock.patch.object(rhf.gto, 'MoleBase', type(mol)):
            rhf.TD_Scanner.__call__(
                scanner, mol, reuse_scf_dm=False, reuse_td_guess=False)

        scanner._scf.assert_called_once_with(mol, dm0=None)
        scanner._transfer_initial_guess.assert_not_called()
        scanner.kernel.assert_called_once_with(x0=fresh_xy)


if __name__ == '__main__':
    unittest.main()
