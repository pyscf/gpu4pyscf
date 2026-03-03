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
from gpu4pyscf.nac.mecp import MECPScanner, ConicalIntersectionOptimizer
import pytest

atom = [
    ['C', ( 1.08714538e-07,  1.42742925e+00,  1.66180082e-02)],
    ['C', ( 1.20863220e+00,  7.37682299e-01,  1.26124030e-02)],
    ['C', ( 1.20863229e+00, -7.37682827e-01,  4.05547048e-03)],
    ['C', (-1.10080950e-07, -1.42742890e+00,  4.18955561e-05)],
    ['C', (-1.20863232e+00, -7.37682428e-01,  4.05542079e-03)],
    ['C', (-1.20863217e+00,  7.37682696e-01,  1.26126415e-02)],
    ['H', ( 3.30517487e-07,  2.50912129e+00,  2.28905128e-02)],
    ['H', ( 2.15206376e+00,  1.26465626e+00,  1.56466006e-02)],
    ['H', ( 2.15206372e+00, -1.26465701e+00,  1.02499797e-03)],
    ['H', (-3.28717469e-07, -2.50912092e+00, -6.22998716e-03)],
    ['H', (-2.15206383e+00, -1.26465649e+00,  1.02491283e-03)],
    ['H', (-2.15206365e+00,  1.26465678e+00,  1.56471236e-02)],
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
    @pytest.mark.slow
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

        ci_optimizer_new = ConicalIntersectionOptimizer(tdf, states=(1, 2), crossing_type='n-2')
        mecp_obj = MECPScanner(ci_optimizer_new)
        g_bar = mecp_obj(optimized_mol)[1]

        assert np.linalg.norm(g_bar) <= 1.0E-5
        assert e_mecp <= 1.0E-5
        assert delta_e1 >= 1.0E-5
        assert delta_e2 <= 1.0E-5
        assert delta_e2 <= 1.5 * e_mecp


if __name__ == "__main__":
    print("Full Tests for MECP search between excited state.")
    unittest.main()
