#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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

import sys
import unittest
import numpy as np
from pyscf.data.nist import BOHR as BOHR_TO_ANGSTROM, HARTREE2WAVENUMBER
from pyscf.pbc import gto
from gpu4pyscf.pbc import dft
from gpu4pyscf.pbc.hessian import (
    GammaHessian,
    HAS_PHONOPY,
)
from gpu4pyscf.pbc.hessian.gamma_hessian import HARTREE_TO_THZ


@unittest.skipUnless(HAS_PHONOPY, "phonopy is not installed")
class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        a = 5.431
        cls.cell = gto.Cell()
        cls.cell.a = np.array(
            [
                [0.0, a / 2, a / 2],
                [a / 2, 0.0, a / 2],
                [a / 2, a / 2, 0.0],
            ]
        )
        cls.cell.atom = [
            ["Si", [0.0, 0.0, 0.0]],
            ["Si", [a / 4, a / 4, a / 4]],
        ]

        cls.cell.unit = "Angstrom"
        # the basis is to align with cp2k's basis
        cls.cell.basis = """
        Si  DZVP-GTH-PADE DZVP-GTH-PADE-q4
        2
        3 0 1 4 2 2
        1.203242234500  0.329035044500  0.000000000000  0.047453912600  0.000000000000
        0.468840978600 -0.253311832300  0.000000000000 -0.259447357300  0.000000000000
        0.167986323400 -0.787094627700  0.000000000000 -0.544092930300  0.000000000000
        0.057561952600 -0.190989847900  1.000000000000 -0.362401036400  1.000000000000
        3 2 2 1 1
        0.450000000000  1.000000000000
        """
        cls.cell.pseudo = "GTH-PBE-q4"
        cls.cell.verbose = 0
        cls.cell.ke_cutoff = 200
        cls.cell.build()

        cls.mf = dft.RKS(cls.cell, xc="pbe")
        cls.mf.conv_tol = 1e-10 
        cls.mf.kernel()
        if not cls.mf.converged:
            raise RuntimeError("Reference Si SCF did not converge")

        cls.coords = cls.cell.atom_coords().copy()
        cls.unit = cls.cell.unit
        cls.hessian_001 = GammaHessian(
            cls.mf,
            primitive_matrix=np.eye(3),
            displacement=0.01,
            force_central_diff=True,
        )
        cls.fc_001 = cls.hessian_001.kernel()
        cls.hessian_002 = GammaHessian(
            cls.mf,
            primitive_matrix=np.eye(3),
            displacement=0.02,
            force_central_diff=True,
        )
        cls.fc_002 = cls.hessian_002.kernel()
        cls.frequencies, cls.eigenvectors, cls.dyn_mat = (
            cls.hessian_001.phonon_modes()
        )
        cls.frequencies_phonopy, _, cls.dyn_mat_phonopy = (
            cls.hessian_001.phonopy_modes()
        )

    def test_force_constant_symmetry_and_geometry_restore(self):
        self.assertTrue(np.allclose(self.fc_001, self.fc_001.T, atol=1e-7))
        self.assertTrue(np.allclose(self.cell.atom_coords(), self.coords))
        self.assertEqual(self.cell.unit, self.unit)
        self.assertIsNotNone(self.hessian_001.phonon)
        self.assertEqual(len(self.hessian_001.phonon.primitive), 2)
        self.assertTrue(
            np.array_equal(
                self.hessian_001.primitive_matrix,
                np.eye(3),
            )
        )

    def test_acoustic_sum_rule(self):
        self.assertLess(np.max(np.abs(self.fc_001.sum(axis=1))), 1e-7)

    def test_displacement_convergence(self):
        np.testing.assert_allclose(
            self.fc_001,
            self.fc_002,
            rtol=0.25,
            atol=5e-4,
        )

    def test_manual_central_difference_benchmark(self):
        displacement = self.hessian_001.displacement
        displacement_bohr = displacement / BOHR_TO_ANGSTROM
        reference_coords = self.cell.atom_coords(unit="Angstrom")
        manual_fc = np.empty_like(self.fc_001)
        reference_dm = self.mf.make_rdm1()

        for column in range(3 * self.cell.natm):
            atom, axis = divmod(column, 3)
            gradients = []
            for sign in (1.0, -1.0):
                displaced_cell = self.cell.copy()
                displaced_coords = reference_coords.copy()
                displaced_coords[atom, axis] += sign * displacement
                displaced_cell.set_geom_(
                    displaced_coords,
                    unit="Angstrom",
                )

                mf_disp = dft.RKS(
                    displaced_cell,
                    xc=self.mf.xc,
                )
                mf_disp.conv_tol = self.mf.conv_tol
                mf_disp.kernel(dm0=reference_dm)
                self.assertTrue(mf_disp.converged)

                gradient = mf_disp.nuc_grad_method().kernel()
                if hasattr(gradient, "get"):
                    gradient = gradient.get()
                gradients.append(np.asarray(gradient).reshape(-1))

                del gradient, mf_disp, displaced_cell
            manual_fc[:, column] = (gradients[0] - gradients[1]) / (2.0 * displacement_bohr)

        np.testing.assert_allclose(
            manual_fc,
            self.fc_001,
            rtol=5e-2,
            atol=5e-4,
        )

    def test_dynamical_matrix(self):
        self.assertEqual(self.frequencies.shape, (6,))
        self.assertLess(np.max(np.abs(self.frequencies[:3])), 5.0)
        self.assertTrue(
            np.allclose(
                self.eigenvectors.T @ self.eigenvectors,
                np.eye(6),
                atol=1e-10,
            )
        )
        self.assertTrue(np.allclose(self.dyn_mat, self.dyn_mat.T, atol=1e-12))
        self.assertTrue(
            np.allclose(
                self.frequencies,
                self.frequencies_phonopy,
                atol=1e-4,
            )
        )
        self.assertTrue(
            np.allclose(
                self.dyn_mat,
                self.dyn_mat_phonopy,
                atol=1e-12,
            )
        )

    def test_vs_cp2k(self):
        """
        &GLOBAL
        PROJECT Si_phonon
        RUN_TYPE ENERGY_FORCE
        PRINT_LEVEL MEDIUM
        &END GLOBAL

        &FORCE_EVAL
        METHOD QS
        &PRINT
            &FORCES ON
            &END FORCES
        &END PRINT

        &DFT
            BASIS_SET_FILE_NAME BASIS_SET
            POTENTIAL_FILE_NAME POTENTIAL
            &MGRID
            CUTOFF 400
            REL_CUTOFF 60
            &END MGRID
            &QS
            EPS_DEFAULT 1.0E-10
            &END QS
            &SCF
            EPS_SCF 1.0E-10
            MAX_SCF 100
            &END SCF
            &XC
            &XC_FUNCTIONAL PBE
            &END XC_FUNCTIONAL
            &END XC
        &END DFT
        &SUBSYS
            &CELL
            A  0.000000000  2.715500000  2.715500000
            B  2.715500000  0.000000000  2.715500000
            C  2.715500000  2.715500000  0.000000000
            &END CELL
            &COORD
            Si  0.000000000  0.000000000  0.000000000
            Si  1.357750000  1.357750000  1.357750000
            &END COORD
            &KIND Si
            BASIS_SET DZVP-GTH-PADE
            POTENTIAL GTH-PBE-q4
            &END KIND
        &END SUBSYS
        &END FORCE_EVAL

        $ phonopy-init --cp2k -c si.inp -d --dim="1 1 1" --amplitude=0.01 --pm
        $ cp2k -i si-supercell-001.inp -o supercell-001.out
        $ cp2k -i si-supercell-002.inp -o supercell-002.out
        $ phonopy-init --cp2k -f Si_phonon-supercell-001-forces-1_0.xyz Si_phonon-supercell-002-forces-1_0.xyz
        $ phonopy --qpoints="0 0 0"
        $ phonopy --writefc

        In file FORCE_CONSTANTS
           2    2
        1 1
            0.590763674050798     0.000000000000000     0.000000000000000
            0.000000000000000     0.590763674050798     0.000000000000000
            0.000000000000000     0.000000000000000     0.590763674050798
        1 2
            -0.590763674050798     0.000000000000000     0.000000000000000
            0.000000000000000    -0.590763674050798     0.000000000000000
            0.000000000000000     0.000000000000000    -0.590763674050798
        2 1
            -0.590763674050798     0.000000000000000     0.000000000000000
            0.000000000000000    -0.590763674050798     0.000000000000000
            0.000000000000000     0.000000000000000    -0.590763674050798
        2 2
            0.590763674050798     0.000000000000000     0.000000000000000
            0.000000000000000     0.590763674050798     0.000000000000000
            0.000000000000000     0.000000000000000     0.590763674050798

        In file qpoints.yaml
        nqpoint: 1      
        natom:   2      
        reciprocal_lattice:
        - [  -0.18412815,   0.18412815,   0.18412815 ] # a*
        - [   0.18412815,  -0.18412815,   0.18412815 ] # b*
        - [   0.18412815,   0.18412815,  -0.18412815 ] # c*
        phonon:
        - q-position: [    0.0000000,    0.0000000,    0.0000000 ]
        band:
        - # 1
            frequency:   -0.0000002881
        - # 2
            frequency:   -0.0000002037
        - # 3
            frequency:    0.0000000000
        - # 4
            frequency:   22.9935696695
        - # 5
            frequency:   22.9935696695
        - # 6
            frequency:   22.9935696695

        """
        # unit in hartree/Angstrom.au
        ref = np.array([[[  0.590763674050798,  0.000000000000000,  0.000000000000000 ],
                         [  0.000000000000000,  0.590763674050798,  0.000000000000000 ],
                         [  0.000000000000000,  0.000000000000000,  0.590763674050798 ],],
                         [[-0.590763674050798,  0.000000000000000,  0.000000000000000 ],
                         [  0.000000000000000, -0.590763674050798,  0.000000000000000 ],
                         [  0.000000000000000,  0.000000000000000, -0.590763674050798 ],],
                         [[-0.590763674050798,  0.000000000000000,  0.000000000000000 ],
                         [  0.000000000000000, -0.590763674050798,  0.000000000000000 ],
                         [  0.000000000000000,  0.000000000000000, -0.590763674050798 ],],
                         [[ 0.590763674050798,  0.000000000000000,  0.000000000000000 ],
                         [  0.000000000000000,  0.590763674050798,  0.000000000000000 ],
                         [  0.000000000000000,  0.000000000000000,  0.590763674050798 ],]])
        
        ref_4d = ref.reshape(2, 2, 3, 3)
        ref_swapped = ref_4d.swapaxes(1, 2)
        ref = ref_swapped.reshape(6, 6)
        ref_in_atomic_unit = ref*BOHR_TO_ANGSTROM
        ref_frequencies =  22.9935696695 *  HARTREE2WAVENUMBER / HARTREE_TO_THZ # original in THz

        assert np.abs(ref_in_atomic_unit - self.fc_001).max() < 2.0E-3
        assert np.abs(self.frequencies_phonopy[3:] - ref_frequencies).max() < 1.0
        assert np.abs(self.frequencies[3:] - ref_frequencies).max() < 1.0



if __name__ == "__main__":
    print("Full Tests for PBC Hessian for gamma point with phonopy.")
    unittest.main()
