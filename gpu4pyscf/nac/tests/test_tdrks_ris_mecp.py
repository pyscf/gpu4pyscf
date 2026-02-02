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
import pyscf
from pyscf import dft
from gpu4pyscf import tdscf
from pyscf.data.nist import HARTREE2EV
from gpu4pyscf.nac.mecp import MECPScanner, ConicalIntersectionOptimizer
from gpu4pyscf.lib.multi_gpu import num_devices

atom = """
 H   0.451616   0.760462   1.270585
 H  -0.075939  -1.025987   1.104107
 H  -0.187363   1.030696  -1.090653
 H   0.344198  -0.749472  -1.304979
 C  -0.233997   0.032263   0.789767
 C  -0.303805  -0.029177  -0.763879
"""

bas0 = "3-21g"

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
    mf = dft.RKS(mol, xc='pbe0').to_gpu()
    mf.kernel()
    td = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True, Ktrunc=0.0)
    td.conv_tol=1.0E-5
    td.nstates=5
    td.kernel()
    return td.energies/HARTREE2EV


class KnownValues(unittest.TestCase):
    @unittest.skipIf(num_devices > 1, '')
    def test_mecp_pbe0_tda_singlet(self):
        mf = dft.RKS(mol, xc='pbe0').to_gpu().density_fit()
        mf.kernel()
        td = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True, Ktrunc=0.0)
        td.conv_tol=1.0E-4
        td.nstates = 5
        td.kernel()
        ci_optimizer = ConicalIntersectionOptimizer(td, states=(1, 2), crossing_type='n-2')
            
        optimized_mol = ci_optimizer.optimize()
        mff = dft.RKS(optimized_mol, xc='pbe0').to_gpu().density_fit()
        mff.kernel()
        tdf = tdscf.ris.TDA(mf=mff, nstates=5, spectra=False, single=False, gram_schmidt=True, Ktrunc=0.0)
        td.conv_tol=1.0E-4
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

        e_mecp = tdf.energies[1]/HARTREE2EV - tdf.energies[0]/HARTREE2EV
        delta_e1 = e1[1] - e1[0]
        delta_e2 = e2[1] - e2[0]

        ci_optimizer_new = ConicalIntersectionOptimizer(tdf, states=(1, 2), crossing_type='n-2')
        mecp_obj = MECPScanner(ci_optimizer_new)
        g_bar = mecp_obj(optimized_mol)[1]
        assert np.linalg.norm(g_bar) <= 5.0E-5
        assert e_mecp <= 2.0E-5
        assert delta_e1 >= 1.0E-5
        #assert delta_e2 <= 1.0E-5
        assert delta_e2 <= 1.5 * e_mecp


if __name__ == "__main__":
    print("Full Tests for MECP search between excited state using TDA-ris.")
    unittest.main()
