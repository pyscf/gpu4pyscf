import unittest
import numpy as np
import pyscf
import pytest
from packaging import version
from gpu4pyscf.cc import ccsd_incore

def setUpModule():
    global mol, mf
    mol = pyscf.M(atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)],
    ], basis = 'cc-pvdz', verbose=0)
    mf = mol.RHF().run()

def tearDownModule():
    global mol, mf
    del mol, mf

class KnownValues(unittest.TestCase):
    def test_eris(self):
        ref = mf.CCSD().ao2mo()
        mcc = ccsd_incore.CCSD(mf)
        eris = mcc.ao2mo()
        self.assertAlmostEqual(abs(eris.oooo - ref.oooo).max(), 0, 11)
        self.assertAlmostEqual(abs(eris.ovoo - ref.ovoo).max(), 0, 11)
        self.assertAlmostEqual(abs(eris.oovv - ref.oovv).max(), 0, 11)
        self.assertAlmostEqual(abs(eris.ovvo - ref.ovvo).max(), 0, 10)

    def test_ccsd_incore_update_amps(self):
        ref = mf.CCSD()
        mcc = ccsd_incore.CCSD(mf)
        nocc = mcc.nocc
        nvir = mcc.nmo - ref.nocc

        np.random.seed(1)
        inp1 = np.random.rand(nocc, nvir)
        inp2 = np.random.rand(nocc, nocc, nvir, nvir)
        inp2 = inp2 + inp2.transpose(1, 0, 3, 2)

        r1, r2 = ref.update_amps(inp1, inp2, ref.ao2mo())
        t1, t2 = mcc.update_amps(inp1, inp2, mcc.ao2mo())
        self.assertAlmostEqual(abs(r1 - t1).max(), 0, 9)
        self.assertAlmostEqual(abs(r2 - t2).max(), 0, 9)

    @pytest.mark.skipif(version.parse(pyscf.__version__) <= version.parse('2.4.0'), reason='requires pyscf 2.5 or higher')
    def test_ccsd_incore_kernel(self):
        ref = mf.CCSD().run()
        mcc = ccsd_incore.CCSD(mf.to_gpu()).run()
        self.assertAlmostEqual(mcc.e_tot, -76.2401089190058, 8)
        self.assertAlmostEqual(mcc.e_tot, ref.e_tot, 8)
        self.assertAlmostEqual(abs(mcc.t1 - ref.t1).max(), 0, 6)
        self.assertAlmostEqual(abs(mcc.t2 - ref.t2).max(), 0, 6)

    def test_to_gpu(self):
        mcc = ccsd_incore.CCSD(mf.to_gpu()).run()
        mcc = mcc.run()
        e_gpu = mcc.e_tot
        mcc = mcc.to_gpu()
        mcc = mcc.run()
        e_cpu = mcc.e_tot
        assert (e_cpu - e_gpu) < 1e-6


if __name__ == '__main__':
    print("Full Tests for CCSD")
    unittest.main()
