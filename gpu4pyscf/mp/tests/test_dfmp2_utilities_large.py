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


"""
Unittest for DF-MP2 utility functions.

This will only test intermediate results, not the final MP2 energy and the full process correctness.

Some additional assumptions:
- no cartesian
- restricted reference
"""

import pytest
import unittest
import pyscf
import gpu4pyscf

# IntelliSense hinting for imported modules
import pyscf.df
import gpu4pyscf.df.int3c2e_bdiv
import pyscf.mp.dfmp2

from gpu4pyscf.mp import dfmp2_drivers


def setUpModule():
    global mol, aux, mf, mp, with_df, intopt, vhfopt
    token = """
    C                 -0.07551087    1.68127663   -0.10745193
    O                  1.33621755    1.87147409   -0.39326987
    C                  1.67074668    2.95729545    0.49387976
    C                  0.41740763    3.77281969    0.78495878
    C                 -0.60481480    3.07572636    0.28906224
    H                 -0.19316298    1.01922455    0.72486113
    O                  0.35092043    5.03413298    1.45545728
    H                  0.42961487    5.74279041    0.81264173
    O                 -1.95331750    3.53349874    0.15912025
    H                 -2.55333895    2.78846397    0.23972698
    O                  2.81976302    3.20110148    0.94542226
    C                 -0.81772499    1.09230218   -1.32146482
    H                 -0.70955636    1.74951833   -2.15888136
    C                 -2.31163857    0.93420736   -0.98260166
    H                 -2.72575463    1.89080093   -0.74107186
    H                 -2.41980721    0.27699120   -0.14518512
    O                 -0.26428017   -0.18613595   -1.64425697
    H                 -0.72695910   -0.55328886   -2.40104423
    O                 -3.00083741    0.38730252   -2.10989934
    H                 -3.93210821    0.28874990   -1.89865997
    """
    mol = pyscf.gto.Mole(atom=token, basis='6-31G', max_memory=32000, output='/dev/null', cart=False).build()
    aux = pyscf.gto.Mole(atom=token, basis='cc-pVDZ-ri', max_memory=32000, output='/dev/null', cart=False).build()
    mol.output = aux.output = '/dev/null'
    mol.incore_anyway = True
    mf = pyscf.scf.RHF(mol).density_fit().run()
    mp = pyscf.mp.dfmp2.DFMP2(mf)
    mp.with_df = pyscf.df.DF(mol, auxbasis='cc-pVDZ-ri')
    mp.run()
    intopt = gpu4pyscf.df.int3c2e_bdiv.Int3c2eOpt(mol, aux)
    
def tearDownModule():
    global mol, aux, mf, mp, intopt
    mol.stdout.close()
    aux.stdout.close()
    del mol, aux, mf, mp, intopt

class Intermediates(unittest.TestCase):
    def test_dfmp2_kernel_multi_gpu(self):
        nocc = mol.nelectron // 2
        occ_coeff = mf.mo_coeff[:, :nocc]
        vir_coeff = mf.mo_coeff[:, nocc:]
        occ_energy = mf.mo_energy[:nocc]
        vir_energy = mf.mo_energy[nocc:]

        result = dfmp2_drivers.dfmp2_kernel_multi_gpu_cderi_cpu(mol, aux, occ_coeff, vir_coeff, occ_energy, vir_energy, j3c_backend='vhfopt')
        self.assertAlmostEqual(result['e_corr_os'], mp.e_corr_os, 7)
        self.assertAlmostEqual(result['e_corr_os'], -0.9626136186267932, 7)
        self.assertAlmostEqual(result['e_corr_ss'], mp.e_corr_ss, 7)
        self.assertAlmostEqual(result['e_corr_ss'], -0.3337013459521023, 7)

        result = dfmp2_drivers.dfmp2_kernel_multi_gpu_cderi_cpu(mol, aux, occ_coeff, vir_coeff, occ_energy, vir_energy, j3c_backend='bdiv')
        self.assertAlmostEqual(result['e_corr_os'], mp.e_corr_os, 7)
        self.assertAlmostEqual(result['e_corr_os'], -0.9626136186267932, 7)
        self.assertAlmostEqual(result['e_corr_ss'], mp.e_corr_ss, 7)
        self.assertAlmostEqual(result['e_corr_ss'], -0.3337013459521023, 7)
