import pytest
import unittest
import pyscf
import gpu4pyscf
import cupy as cp

# IntelliSense hinting for imported modules
import pyscf.df
import gpu4pyscf.df.int3c2e_bdiv
import gpu4pyscf.mp.dfump2


def setUpModule():
    global mol, aux, mf, mp, with_df, intopt, vhfopt
    token = """
    O    0.    0.    0.  
    H    0.94  0.    0.  
    H   -0.24  0.    0.91
    """
    mol = pyscf.gto.Mole(atom=token, basis='def2-TZVPP', max_memory=32000, spin=2, output='/dev/null', cart=False).build()
    aux = pyscf.gto.Mole(atom=token, basis='def2-TZVPP-ri', max_memory=32000, spin=2, output='/dev/null', cart=False).build()
    mol.output = aux.output = '/dev/null'
    mol.incore_anyway = True
    mf = pyscf.scf.UHF(mol).density_fit().run()

    with_df = pyscf.df.DF(mol, auxbasis='def2-TZVPP-ri').build()
    mf._eri = with_df.get_ao_eri()
    mp = pyscf.mp.ump2.UMP2(mf)
    mp.kernel(with_t2=True)
    intopt = gpu4pyscf.df.int3c2e_bdiv.Int3c2eOpt(mol, aux)


def tearDownModule():
    global mol, aux, mf, mp, intopt
    mol.stdout.close()
    aux.stdout.close()
    del mol, aux, mf, mp, intopt


class KnownValues(unittest.TestCase):
    def test_dfump2(self):
        mp_gpu = gpu4pyscf.mp.dfump2.DFUMP2(mf, auxbasis='def2-TZVPP-ri').run()
        print(mp.e_corr_os, mp.e_corr_ss, mp.e_corr)
        e_corr_ref = -0.22761931525036094
        self.assertAlmostEqual(mp_gpu.e_corr_os, -0.1741764261645391, 9)
        self.assertAlmostEqual(mp_gpu.e_corr_ss, -0.05344288908582186, 9)
        self.assertAlmostEqual(mp_gpu.e_corr, e_corr_ref, 9)

        mp_gpu = gpu4pyscf.mp.dfump2.DFUMP2(mf, auxbasis='def2-TZVPP-ri').run(j2c_decomp_alg='eig')
        self.assertAlmostEqual(mp_gpu.e_corr, e_corr_ref, 9)

        mp_gpu = gpu4pyscf.mp.dfump2.DFUMP2(mf, auxbasis='def2-TZVPP-ri').run(j3c_backend='vhfopt')
        self.assertAlmostEqual(mp_gpu.e_corr, e_corr_ref, 9)

        mp_gpu = gpu4pyscf.mp.dfump2.DFUMP2(mf, auxbasis='def2-TZVPP-ri').run(fp_type='FP32')
        self.assertNotAlmostEqual(mp_gpu.e_corr, e_corr_ref, 9)  # FP32 is not accurate enough
        self.assertAlmostEqual(mp_gpu.e_corr, e_corr_ref, 6)

        mp_gpu = gpu4pyscf.mp.dfump2.DFUMP2(mf, auxbasis='def2-TZVPP-ri').run(with_t2=True)
        for t2_gpu, t2_cpu in zip(mp_gpu.t2, mp.t2):
            self.assertTrue(cp.allclose(t2_gpu, t2_cpu, atol=1e-6))

    def test_dfump2_frozen(self):
        mp_frz = pyscf.mp.ump2.UMP2(mf).run(frozen=[1, 2])
        mp_gpu = gpu4pyscf.mp.dfump2.DFUMP2(mf, auxbasis='def2-TZVPP-ri').run(frozen=mp_frz.frozen)
        self.assertAlmostEqual(mp_gpu.e_corr, -0.06374760208934793, 9)
        self.assertAlmostEqual(mp_gpu.e_corr, mp_frz.e_corr, 9)

        mp_frz = pyscf.mp.ump2.UMP2(mf).run(frozen=[[0, 1], [1, 2]])
        mp_gpu = gpu4pyscf.mp.dfump2.DFUMP2(mf, auxbasis='def2-TZVPP-ri').run(frozen=mp_frz.frozen)
        self.assertAlmostEqual(mp_gpu.e_corr, -0.08095769088357245, 9)
        self.assertAlmostEqual(mp_gpu.e_corr, mp_frz.e_corr, 9)
