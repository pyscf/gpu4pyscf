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

import unittest
import numpy as np
import cupy as cp
import pyscf
from pyscf import lib, gto, scf, dft
from gpu4pyscf import tdscf, nac
from gpu4pyscf.nac.mecp import ConicalIntersectionOptimizer
import gpu4pyscf

atom = [
    ['C', ( 0.0000,  1.3970, 0.1000)],
    ['C', ( 1.2100,  0.6985, 0.0000)],
    ['C', ( 1.2100, -0.6985, 0.0000)],
    ['C', ( 0.0000, -1.3970, 0.0000)],
    ['C', (-1.2100, -0.6985, 0.0000)],
    ['C', (-1.2100,  0.6985, 0.0000)],
    ['H', ( 0.0000,  2.4770, 0.0000)],
    ['H', ( 2.1450,  1.2385, 0.0000)],
    ['H', ( 2.1450, -1.2385, 0.0000)],
    ['H', ( 0.0000, -2.4770, 0.0000)],
    ['H', (-2.1450, -1.2385, 0.0000)],
    ['H', (-2.1450,  1.2385, 0.0000)],
]

bas0 = "cc-pvdz"

def setUpModule():
    global mol
    mol = pyscf.M(
        atom=atom, basis=bas0, max_memory=32000, output="/dev/null", verbose=1)


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


def project_on_plane_lstsq(x3, x1, x2):
    x3 = x3.reshape(-1)
    x1 = x1.reshape(-1)
    x2 = x2.reshape(-1)
    A = np.column_stack([x1, x2])
    c, _, _, _ = np.linalg.lstsq(A, x3, rcond=None)
    projection = A @ c
    return projection


def calc_energy(mol):
    mf = scf.RHF(mol).to_gpu()
    mf.kernel()
    td = mf.TDA()
    td.nstates=5
    td.kernel()
    return td.e


class KnownValues(unittest.TestCase):
    def test_mecp_hf_tda_singlet(self):
        mf = scf.RHF(mol).to_gpu()
        mf.kernel()
        td = mf.TDA()
        td.nstates = 5
        td.kernel()
        ci_optimizer = ConicalIntersectionOptimizer(td, states=(1, 2), crossing_type='n-2')
            
        optimized_mol = ci_optimizer.optimize()
        mff = scf.RHF(optimized_mol).to_gpu()
        mff.kernel()
        tdf = mff.TDA()
        tdf.nstates = 5
        tdf.kernel()
        gf = tdf.nuc_grad_method()
        nac_obj = tdf.nac_method()
        nac_obj.states = (1, 2)
        nac_obj.kernel()

        gf.state = 1
        g1 = gf.kernel()
        gf.state = 2
        g2 = gf.kernel()

        x1 = g1 - g2
        x1_norm_val = np.linalg.norm(x1)
        x1_norm_vec = x1 / x1_norm_val if x1_norm_val > 1e-9 else np.zeros_like(x1)
        x2 = nac_obj.de_scaled
        x2_norm_val = np.linalg.norm(x2)
        x2_norm_vec = x2 / x2_norm_val if x2_norm_val > 1e-9 else np.zeros_like(x2)
        natom = g2.shape[0]
        g2_proj = project_on_plane_lstsq(g2, x1_norm_vec, x2_norm_vec)
        g2_proj = g2_proj.reshape(natom, 3)
        g2_proj = g2 - g2_proj
        g2_proj_norm_val = np.linalg.norm(g2_proj)
        g2_proj_norm_vec = g2_proj / g2_proj_norm_val if g2_proj_norm_val > 1e-9 else np.zeros_like(g2_proj)

        delta = 10.0E-4
        v1 = delta * x1_norm_vec + delta * x2_norm_vec
        v2 = g2_proj_norm_vec*delta

        atom_coords = optimized_mol.atom_coords(unit='a')
        mol1 = optimized_mol.copy()
        mol2 = optimized_mol.copy()
        mol1.set_geom_(atom_coords + v1, unit='a')
        mol2.set_geom_(atom_coords + v2, unit='a')

        e1 = calc_energy(mol1)
        e2 = calc_energy(mol2)

        e_mecp = tdf.e[1] - tdf.e[0]
        delta_e1 = e1[1] - e1[0]
        delta_e2 = e2[1] - e2[0]

        assert e_mecp <= 1.0E-5
        assert delta_e1 >= 1.0E-5
        assert delta_e2 <= 1.0E-5
        assert delta_e2 <= 1.5 * e_mecp


if __name__ == "__main__":
    print("Full Tests for MECP search between excited state.")
    unittest.main()
