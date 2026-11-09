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

import unittest
import numpy as np
import pyscf
from pyscf import dft as cpu_dft
from gpu4pyscf import dft
from gpu4pyscf.scf import gcp
try:
    from pyscf.scf import gcp as cpu_gcp
    from pyscf.grad import gcp as cpu_gcp_grad
    from pyscf.hessian import gcp as cpu_gcp_hess
except ImportError:
    cpu_gcp = None
try:
    from gpu4pyscf.dispersion import dftd3, dftd4
except (ImportError, OSError):
    dftd3 = dftd4 = None

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''
# NH2 radical: its singly occupied orbital is non-degenerate, unlike OH's
nh2 = 'N 0 0 0; H 0 0.8014 -0.6374; H 0 -0.8014 -0.6374'
# PySCF and GPU4PySCF choose different default auxiliary bases for def2-mTZVP(P)
auxbasis = 'def2-universal-jfit'

def setUpModule():
    global mol, mol_b97
    mol = pyscf.M(atom=atom, basis='def2-mtzvpp', verbose=0, output='/dev/null')
    mol_b97 = pyscf.M(atom=atom, basis='def2-mtzvp', verbose=0, output='/dev/null')

def tearDownModule():
    global mol, mol_b97
    mol.stdout.close()
    mol_b97.stdout.close()
    del mol, mol_b97

def _run(mf, gcp=None):
    mf.conv_tol = 1e-12
    if gcp is not None:
        mf.gcp = gcp
    mf.kernel()
    assert mf.converged
    return mf

@unittest.skipIf(cpu_gcp is None, 'gCP requires PySCF >= 2.15')
@unittest.skipIf(dftd3 is None or dftd4 is None, 'requires dftd3 and dftd4')
class KnownValues(unittest.TestCase):
    def test_r2scan3c_energy_includes_gcp(self):
        mf = _run(dft.RKS(mol, xc='r2scan-3c').density_fit())
        mf_off = _run(dft.RKS(mol, xc='r2scan-3c').density_fit(), gcp=False)
        e_gcp = cpu_gcp.get_gcp(cpu_dft.RKS(mol, xc='r2scan-3c'))
        self.assertTrue(abs(e_gcp) > 1e-3)
        self.assertAlmostEqual(mf.scf_summary['gcp'], e_gcp, 10)
        # gCP depends on the geometry only, so it is the whole energy difference
        self.assertAlmostEqual(mf.e_tot - mf_off.e_tot, e_gcp, 9)

    def test_r2scan3c_energy_vs_pyscf(self):
        e_gpu = _run(dft.RKS(mol, xc='r2scan-3c').density_fit(auxbasis)).e_tot
        e_cpu = _run(cpu_dft.RKS(mol, xc='r2scan-3c').density_fit(auxbasis)).e_tot
        self.assertAlmostEqual(e_gpu, e_cpu, 7)

    def test_r2scan3c_uks_energy_vs_pyscf(self):
        mol_nh2 = pyscf.M(atom=nh2, spin=1, basis='def2-mtzvpp', verbose=0)
        e_gpu = _run(dft.UKS(mol_nh2, xc='r2scan-3c').density_fit(auxbasis)).e_tot
        e_cpu = _run(cpu_dft.UKS(mol_nh2, xc='r2scan-3c').density_fit(auxbasis)).e_tot
        self.assertAlmostEqual(e_gpu, e_cpu, 7)

    def test_b973c_energy_vs_pyscf(self):
        e_gpu = _run(dft.RKS(mol_b97, xc='b97-3c').density_fit(auxbasis)).e_tot
        e_cpu = _run(cpu_dft.RKS(mol_b97, xc='b97-3c').density_fit(auxbasis)).e_tot
        self.assertAlmostEqual(e_gpu, e_cpu, 7)

    def test_r2scan3c_gradient_includes_gcp(self):
        mf = _run(dft.RKS(mol, xc='r2scan-3c').density_fit())
        mf_off = _run(dft.RKS(mol, xc='r2scan-3c').density_fit(), gcp=False)
        g = mf.Gradients().kernel()
        g_off = mf_off.Gradients().kernel()
        g_gcp = cpu_gcp_grad.get_gcp(cpu_dft.RKS(mol, xc='r2scan-3c').Gradients())
        self.assertTrue(abs(g_gcp).max() > 1e-4)
        np.testing.assert_allclose(g - g_off, g_gcp, atol=1e-7)

    def test_r2scan3c_hessian_includes_gcp(self):
        mf = _run(dft.RKS(mol, xc='r2scan-3c').density_fit())
        mf_off = _run(dft.RKS(mol, xc='r2scan-3c').density_fit(), gcp=False)
        h = mf.Hessian().kernel()
        h_off = mf_off.Hessian().kernel()
        h_gcp = cpu_gcp_hess.get_gcp(cpu_dft.RKS(mol, xc='r2scan-3c').Hessian())
        self.assertTrue(abs(h_gcp).max() > 1e-4)
        np.testing.assert_allclose(h - h_off, h_gcp, atol=1e-6)

    def test_tda_gradient_includes_scf_dispersion(self):
        # The excited-state energy includes the SCF's dispersion energy, so
        # its gradient must include the dispersion gradient.
        mf = _run(dft.RKS(mol, xc='b3lyp-d3bj').density_fit())
        mf_off = _run(dft.RKS(mol, xc='b3lyp').density_fit())
        g = mf.TDA().set(nstates=3).run().Gradients().kernel(state=1)
        g_off = mf_off.TDA().set(nstates=3).run().Gradients().kernel(state=1)
        g_disp = mf.Gradients().get_dispersion()
        self.assertTrue(abs(g_disp).max() > 1e-6)
        # D3 does not change the density, so the difference is the D3 gradient alone
        np.testing.assert_allclose(g - g_off, g_disp, atol=1e-9)

    def test_uks_tda_gradient_includes_scf_dispersion(self):
        # The unrestricted TD gradients share the restricted kernel without
        # inheriting from its class
        mol_nh2 = pyscf.M(atom=nh2, spin=1, basis='def2-svp', verbose=0)
        mf = _run(dft.UKS(mol_nh2, xc='b3lyp-d3bj').density_fit())
        mf_off = _run(dft.UKS(mol_nh2, xc='b3lyp').density_fit())
        g = mf.TDA().set(nstates=3).run().Gradients().kernel(state=1)
        g_off = mf_off.TDA().set(nstates=3).run().Gradients().kernel(state=1)
        g_disp = mf.Gradients().get_dispersion()
        self.assertTrue(abs(g_disp).max() > 1e-6)
        # The unrestricted TD solve leaves ~3e-9 of noise between the two runs
        np.testing.assert_allclose(g - g_off, g_disp, atol=1e-7)

@unittest.skipIf(cpu_gcp is not None, 'PySCF < 2.15 only')
class OldPySCF(unittest.TestCase):
    def test_3c_without_gcp_is_refused(self):
        # PySCF < 2.15 parses 'r2scan-3c' as bare r2SCAN; running it without
        # gCP would silently give a different method.
        with self.assertRaises(NotImplementedError):
            gcp.check_gcp(dft.RKS(mol, xc='r2scan-3c'))
        self.assertFalse(gcp.check_gcp(dft.RKS(mol, xc='r2scan-3c').set(gcp=False)))
        self.assertFalse(gcp.check_gcp(dft.RKS(mol, xc='r2scan')))

if __name__ == "__main__":
    print("Full Tests for gCP/SRB corrections")
    unittest.main()
