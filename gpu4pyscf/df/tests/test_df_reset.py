# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
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
Regression tests for issue #827: DF.reset() must invalidate geometry-dependent
caches (auxmol, intopt, j_engine) even when called without a new mol, and
DF.build() must place the auxiliary basis at the runtime coordinates of
self.mol. Otherwise an in-place mol.set_geom_() + mf.reset() + mf.kernel()
sequence (the standard finite-difference / scanner reuse pattern) silently
returns wrong SCF energies.
'''

import unittest
import numpy as np
import pyscf
from pyscf import dft

atom = '''
O       0.0000000000     0.0000000000     0.0000000000
H       0.9572000000     0.0000000000     0.0000000000
H      -0.2400000000     0.9266000000     0.0000000000
'''

bas = 'def2-svp'
DISP_BOHR = 1e-3


def setUpModule():
    global mol
    mol = pyscf.M(atom=atom, basis=bas, output='/dev/null', verbose=1)


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


def _displaced_coords(m):
    coords = m.atom_coords(unit='Bohr').copy()
    coords[0, 0] += DISP_BOHR
    return coords


class KnownValues(unittest.TestCase):

    def test_reset_after_inplace_geometry_change(self):
        # Reference: fresh mf built directly at the displaced geometry
        mol1 = mol.copy()
        mol1.set_geom_(_displaced_coords(mol1), unit='Bohr')
        mol1.build(dump_input=False)
        e_fresh = float(dft.RKS(mol1, xc='wb97x').density_fit().to_gpu().kernel())

        # Reuse: kernel at the original geometry, then displace in place,
        # reset and rerun — must match the fresh reference.
        mf = dft.RKS(mol.copy(), xc='wb97x').density_fit().to_gpu()
        mf.kernel()
        mf.mol.set_geom_(_displaced_coords(mf.mol), unit='Bohr')
        mf.mol.build(dump_input=False)
        mf.reset()
        e_reused = float(mf.kernel())

        assert abs(e_reused - e_fresh) < 1e-8

    def test_reset_preserves_auxmol_reference(self):
        # Several internal callers (df/grad, hessian, tddft) hold a reference
        # to with_df.auxmol across reset(), using reset() as a memory-release
        # idiom. reset() without a geometry change must keep the object (and
        # its identity) valid.
        mf = dft.RKS(mol.copy(), xc='wb97x').density_fit().to_gpu()
        mf.kernel()
        aux_ref = mf.with_df.auxmol
        assert aux_ref is not None
        mf.reset()
        assert mf.with_df.auxmol is aux_ref
        # The kept object must still be usable (the CI failure mode read
        # auxmol.with_range_coulomb right after reset()).
        with aux_ref.with_range_coulomb(0.3):
            pass

    def test_reset_repairs_geometry_caches_in_place(self):
        # After an in-place displacement, reset() re-anchors auxmol to the
        # runtime coordinates (same object) and drops the stale integral
        # engines.
        mf = dft.RKS(mol.copy(), xc='wb97x').density_fit().to_gpu()
        mf.kernel()
        aux_ref = mf.with_df.auxmol
        mf.mol.set_geom_(_displaced_coords(mf.mol), unit='Bohr')
        mf.mol.build(dump_input=False)
        mf.reset()
        assert mf.with_df.auxmol is aux_ref
        dev = abs(aux_ref.atom_coords() - mf.mol.atom_coords()).max()
        assert dev < 1e-10
        assert mf.with_df.intopt is None

    def test_auxmol_follows_runtime_coordinates(self):
        # After an in-place displacement + reset, the rebuilt auxmol must sit
        # at the runtime coordinates of mf.mol (not at the coordinates its
        # _atom input record was parsed from).
        mf = dft.RKS(mol.copy(), xc='wb97x').density_fit().to_gpu()
        mf.kernel()
        mf.mol.set_geom_(_displaced_coords(mf.mol), unit='Bohr')
        mf.mol.build(dump_input=False)
        mf.reset()
        mf.kernel()
        aux = mf.with_df.auxmol
        dev = abs(aux.atom_coords(unit='Bohr')
                  - mf.mol.atom_coords(unit='Bohr')).max()
        assert dev < 1e-10


if __name__ == "__main__":
    print("Full Tests for DF reset after in-place geometry change (issue #827)")
    unittest.main()
