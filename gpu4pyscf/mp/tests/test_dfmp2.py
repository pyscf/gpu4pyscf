import pytest
import unittest
import pyscf
import gpu4pyscf
import cupy as cp

# IntelliSense hinting for imported modules
import pyscf.df
import gpu4pyscf.df.int3c2e_bdiv
import gpu4pyscf.mp.dfmp2


def setUpModule():
    global mol, aux, mf, mp, with_df, intopt, vhfopt
    token = """
    O    0.    0.    0.  
    H    0.94  0.    0.  
    H   -0.24  0.    0.91
    """
    mol = pyscf.gto.Mole(atom=token, basis='def2-TZVPP', max_memory=32000, output='/dev/null', cart=False).build()
    aux = pyscf.gto.Mole(atom=token, basis='def2-TZVPP-ri', max_memory=32000, output='/dev/null', cart=False).build()
    mol.output = aux.output = '/dev/null'
    mol.incore_anyway = True
    mf = pyscf.scf.RHF(mol).density_fit().run(conv_tol=1e-13)

    with_df = pyscf.df.DF(mol, auxbasis='def2-TZVPP-ri').build()
    mf._eri = with_df.get_ao_eri()
    mp = pyscf.mp.mp2.MP2(mf)
    mp.kernel(with_t2=True)
    intopt = gpu4pyscf.df.int3c2e_bdiv.Int3c2eOpt(mol, aux)


def tearDownModule():
    global mol, aux, mf, mp, intopt
    mol.stdout.close()
    aux.stdout.close()
    del mol, aux, mf, mp, intopt


class KnownValues(unittest.TestCase):
    def test_dfmp2(self):
        mf_gpu = mf.to_gpu()
        mp_gpu = gpu4pyscf.mp.dfmp2.DFMP2(mf_gpu, auxbasis='def2-TZVPP-ri').run()
        print(mp.e_corr_os, mp.e_corr_ss, mp.e_corr)
        e_corr_ref = -0.27863248239139604
        self.assertAlmostEqual(mp_gpu.e_corr_os, -0.21320345319720685, 8)
        self.assertAlmostEqual(mp_gpu.e_corr_ss, -0.06542902919418918, 8)
        self.assertAlmostEqual(mp_gpu.e_corr, e_corr_ref, 8)

        mp_gpu = gpu4pyscf.mp.dfmp2.DFMP2(mf_gpu, auxbasis='def2-TZVPP-ri').run(j2c_decomp_alg='eig')
        self.assertAlmostEqual(mp_gpu.e_corr, e_corr_ref, 8)

        mp_gpu = gpu4pyscf.mp.dfmp2.DFMP2(mf_gpu, auxbasis='def2-TZVPP-ri').run(j3c_backend='vhfopt')
        self.assertAlmostEqual(mp_gpu.e_corr, e_corr_ref, 8)

        mp_gpu = gpu4pyscf.mp.dfmp2.DFMP2(mf_gpu, auxbasis='def2-TZVPP-ri').run(fp_type='FP32')
        self.assertAlmostEqual(mp_gpu.e_corr, e_corr_ref, 6)

        mp_gpu = gpu4pyscf.mp.dfmp2.DFMP2(mf_gpu, auxbasis='def2-TZVPP-ri').run(with_t2=True)
        self.assertTrue(cp.allclose(mp_gpu.t2, mp.t2, atol=1e-6))

    def test_dfmp2_frozen(self):
        mf_gpu = mf.to_gpu()
        mp_frz = pyscf.mp.mp2.MP2(mf).run(frozen=[1, 2])
        mp_gpu = gpu4pyscf.mp.dfmp2.DFMP2(mf_gpu, auxbasis='def2-TZVPP-ri').run(frozen=mp_frz.frozen)
        self.assertAlmostEqual(mp_gpu.e_corr, -0.09885825743893081, 8)
        self.assertAlmostEqual(mp_gpu.e_corr, mp_frz.e_corr, 8)

    def test_scf_from_gpu(self):
        mf_gpu = mf.to_gpu()
        mp_gpu = gpu4pyscf.mp.dfmp2.DFMP2(mf_gpu, auxbasis='def2-TZVPP-ri').run()
        self.assertAlmostEqual(mp_gpu.e_corr, -0.27863248239139604, 8)

    def test_to_gpu(self):
        mp_gpu = pyscf.mp.dfmp2.DFMP2(mf).to_gpu()
        self.assertTrue(isinstance(mp_gpu, gpu4pyscf.mp.dfmp2.DFMP2))
        e_gpu, _ = mp_gpu.kernel()
        e_corr_ref = -0.27863248239139604
        self.assertAlmostEqual(e_gpu, e_corr_ref, 8)
