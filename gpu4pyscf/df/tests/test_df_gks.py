# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
import cupy
import pyscf
from pyscf import lib
from pyscf import dft as cpu_dft
from pyscf.df import df_jk as cpu_df_jk
from gpu4pyscf import dft as gpu_dft
from gpu4pyscf.df import df_jk as gpu_df_jk
try:
    import mcfun
except ImportError:
    mcfun = None


atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''


bas='def2tzvpp'


def setUpModule():
    global mol
    mol = pyscf.M(atom=atom, basis=bas, charge=1, spin=1, max_memory=32000,
                      output='/dev/null', verbose=1)


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


class KnownValues(unittest.TestCase):
    '''
    known values are obtained by PySCF
    '''
    def test_gks(self):
        mf_gpu = gpu_dft.GKS(mol, xc='b3lyp').density_fit(auxbasis='def2-tzvpp-jkfit')
        mf_gpu.collinear = 'm'
        mf_gpu._numint.spin_samples = 6
        e_tot = mf_gpu.kernel()
        if mcfun is not None:
            mf_cpu = cpu_dft.GKS(mol, xc='b3lyp').density_fit(auxbasis='def2-tzvpp-jkfit')
            mf_cpu.collinear = 'm'
            mf_cpu._numint.spin_samples = 6
            e_pyscf = mf_cpu.kernel()
            assert np.abs(e_tot - e_pyscf) < 1e-5
            assert np.abs(lib.fp(mf_cpu.mo_energy) - lib.fp(mf_gpu.mo_energy.get())) < 1e-5
        assert np.abs(e_tot - -75.99882822956384) < 1e-5
        assert np.abs(-96.56444462841858 - lib.fp(mf_gpu.mo_energy.get())) < 1e-5

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_to_cpu(self):
        mf = gpu_dft.GKS(mol, xc='b3lyp').density_fit()
        mf.collinear = 'm'
        mf._numint.spin_samples = 6
        e_gpu = mf.kernel()
        mf = mf.to_cpu()
        e_cpu = mf.kernel()
        assert isinstance(mf, cpu_df_jk._DFHF)
        assert np.abs(e_gpu - e_cpu) < 1e-5

    @unittest.skip("skip test_to_gpu")
    def test_to_gpu(self):
        mf = cpu_dft.GKS(mol, xc='b3lyp').density_fit()
        mf.collinear = 'm'
        mf._numint.spin_samples = 6
        e_cpu = mf.kernel()
        mf = mf.to_gpu()
        e_gpu = mf.kernel()
        assert isinstance(mf, gpu_df_jk._DFHF)
        assert np.abs(e_gpu - e_cpu) < 1e-5


if __name__ == "__main__":
    print("Full Tests for Generalized Hartree-Fock")
    unittest.main()
