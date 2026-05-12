# Copyright 2025 The PySCF Developers. All Rights Reserved.
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

"""
Basic correctness tests for the single-shot DMET driver.

The cancellation property used here:

    For a closed-shell system computed at the SAME mean-field level
    (i.e. ``mf_inner`` and ``mf_outer`` share the same method and the
    same orbital basis), the single-shot DMET total energy must
    reproduce the full-system mean-field total energy exactly.
"""

import unittest
import numpy as np
from pyscf import gto, scf

from gpu4pyscf.dmet import DMET


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom='''
            H 0.0 0.0 0.00
            H 0.0 0.0 0.74
            H 0.0 0.0 2.20
            H 0.0 0.0 2.94
            ''',
            basis='sto-3g',
            verbose=0,
        )
        cls.mf_ref = scf.RHF(cls.mol)
        cls.e_ref = cls.mf_ref.kernel()

    def test_self_consistency_two_atom_fragment(self):
        # A single-shot DMET with the same low- and high-level method
        # must reproduce the full-system mean-field energy.
        mf_outer = scf.RHF(self.mol)
        mf_outer.kernel()

        mf_inner_template = scf.RHF(self.mol)

        dmet = DMET(
            mf_outer=mf_outer,
            mf_inner=mf_inner_template,
            frag_atoms=[0, 1],
            threshold=1e-8,
        )
        e_dmet = dmet.kernel()

        self.assertAlmostEqual(e_dmet, self.e_ref, places=7)

    def test_self_consistency_single_atom_fragment(self):
        mf_outer = scf.RHF(self.mol)
        mf_outer.kernel()

        mf_inner_template = scf.RHF(self.mol)

        dmet = DMET(
            mf_outer=mf_outer,
            mf_inner=mf_inner_template,
            frag_atoms=[0],
            threshold=1e-8,
        )
        e_dmet = dmet.kernel()
        self.assertAlmostEqual(e_dmet, self.e_ref, places=7)

    def test_bath_summary(self):
        mf_outer = scf.RHF(self.mol)
        mf_outer.kernel()

        dmet = DMET(
            mf_outer=mf_outer,
            mf_inner=scf.RHF(self.mol),
            frag_atoms=[0, 1],
            threshold=1e-6,
        )
        dmet.build_bath()
        info = dmet.bath_summary()
        # Two H atoms in STO-3G means 2 fragment AOs.
        self.assertEqual(info['n_fragment_aos'], 2)
        # Number of (bath + core + virtual) eigenvalues equals the
        # environment AO count.
        self.assertEqual(
            info['n_bath'] + info['n_core'] + info['n_virtual'],
            self.mol.nao_nr() - info['n_fragment_aos'],
        )

    def test_decomposition_keys(self):
        mf_outer = scf.RHF(self.mol)
        mf_outer.kernel()

        dmet = DMET(
            mf_outer=mf_outer,
            mf_inner=scf.RHF(self.mol),
            frag_atoms=[0, 1],
            threshold=1e-8,
        )
        dmet.kernel()
        decomp = dmet.energy_decomposition()
        for key in ('E_nuc', 'E_core', 'E_inner', 'E_DMET'):
            self.assertIn(key, decomp)


if __name__ == '__main__':
    print("Tests for single-shot DMET")
    unittest.main()
