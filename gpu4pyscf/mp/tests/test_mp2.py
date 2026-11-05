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
from functools import reduce
import numpy
import cupy
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import mp as mp_cpu
from gpu4pyscf import mp as mp_gpu

def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.verbose = 1
    mol.output = '/dev/null'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()
    mol.incore_anyway = True
    mol.max_memory = 32000
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.scf()

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf

class OrbitalInitialization(unittest.TestCase):
    def assert_eris_matches_cpu(self, cpu_mf, pt, coeff=None, gpu_coeff=None):
        cpu_pt = mp_cpu.mp2.MP2(cpu_mf, frozen=pt.frozen)
        ref = mp_cpu.mp2._ChemistsERIs()._common_init_(cpu_pt, coeff)
        original_coeff = pt.mo_coeff
        eris = pt.ao2mo(gpu_coeff)
        self.assertIs(eris.mol, pt.mol)
        self.assertIs(pt.mo_coeff, original_coeff)
        numpy.testing.assert_array_equal(pt.mo_coeff.get(), cpu_mf.mo_coeff)
        for name in ('mo_coeff', 'mo_energy', 'fock'):
            array = getattr(eris, name)
            self.assertIsInstance(array, cupy.ndarray)
            self.assertEqual(array.dtype, getattr(ref, name).dtype)
            numpy.testing.assert_allclose(array.get(), getattr(ref, name),
                                          atol=1e-9, rtol=0)
        return eris

    def rotated_virtuals(self):
        # Rotating virtual orbitals leaves the density unchanged but creates
        # off-diagonal MO Fock elements, so a diagonal shortcut is incorrect.
        nocc = mol.nelectron // 2
        rotation = numpy.eye(mf.mo_energy.size)
        c, s = numpy.cos(0.2), numpy.sin(0.2)
        rotation[nocc:nocc+2,nocc:nocc+2] = [[c, -s], [s, c]]
        return mf.mo_coeff.dot(rotation)

    def test_default_eris_with_frozen_orbitals(self):
        pt = mp_gpu.mp2.MP2(mf.to_gpu(), frozen=[0, mf.mo_energy.size-1])
        # This shared-array path triggered the PySCF 2.14 helper-contract error.
        self.assertIs(pt.mo_coeff, pt._scf.mo_coeff)
        self.assert_eris_matches_cpu(mf, pt)

    def test_old_df_default_eris_with_frozen_orbitals(self):
        mf_df = mf.density_fit('weigend')
        pt = mp_gpu.dfmp2_old.DFMP2(mf_df.to_gpu(), frozen=[0, mf.mo_energy.size-1])
        self.assertIs(pt.mo_coeff, pt._scf.mo_coeff)
        self.assert_eris_matches_cpu(mf_df, pt)

    def test_explicit_numpy_orbitals_rebuild_fock(self):
        coeff = self.rotated_virtuals()
        pt = mp_gpu.mp2.MP2(mf.to_gpu(), frozen=[0, mf.mo_energy.size-1])
        eris = self.assert_eris_matches_cpu(mf, pt, coeff, coeff)
        self.assertGreater(abs(eris.fock[pt.nocc,pt.nocc+1].item()), 1e-4)


class KnownValues(unittest.TestCase):
    def test_mp2(self):
        nocc = mol.nelectron//2
        nmo = mf.mo_energy.size
        nvir = nmo - nocc

        co = mf.mo_coeff[:,:nocc]
        cv = mf.mo_coeff[:,nocc:]
        g = ao2mo.incore.general(mf._eri, (co,cv,co,cv)).ravel()
        eia = mf.mo_energy[:nocc,None] - mf.mo_energy[nocc:]
        t2ref0 = g/(eia.reshape(-1,1)+eia.reshape(-1)).ravel()
        t2ref0 = t2ref0.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3)

        pt = mp_gpu.MP2(mf.to_gpu())

        emp2, t2 = pt.kernel(mf.mo_energy, mf.mo_coeff)
        self.assertAlmostEqual(emp2, -0.204019967288338, 8)
        self.assertAlmostEqual(pt.e_corr_ss, -0.05153088565639835, 8)
        self.assertAlmostEqual(pt.e_corr_os, -0.15248908163191538, 8)
        self.assertAlmostEqual(abs(t2.get() - t2ref0).max(), 0, 8)

        pt.max_memory = 1
        pt.frozen = None
        emp2, t2 = pt.kernel()
        self.assertAlmostEqual(emp2, -0.204019967288338, 8)
        self.assertAlmostEqual(pt.e_corr_ss, -0.05153088565639835, 8)
        self.assertAlmostEqual(pt.e_corr_os, -0.15248908163191538, 8)
        self.assertAlmostEqual(abs(t2.get() - t2ref0).max(), 0, 8)

    def test_mp2_frozen(self):
        pt = mp_gpu.mp2.MP2(mf.to_gpu())
        pt.frozen = [0]
        pt.kernel(with_t2=False)
        self.assertAlmostEqual(pt.emp2, -0.20168270592254167, 8)
        pt.set_frozen()
        pt.kernel(with_t2=False)
        self.assertAlmostEqual(pt.emp2, -0.20168270592254167, 8)

    def test_mp2_with_df(self):
        nocc = mol.nelectron//2
        nmo = mf.mo_energy.size
        nvir = nmo - nocc

        mf_df = mf.density_fit('weigend')
        pt = mp_gpu.dfmp2_old.DFMP2(mf_df.to_gpu())
        e, t2 = pt.kernel(mf.mo_energy, mf.mo_coeff)

        pt_cpu = mp_cpu.dfmp2.DFMP2(mf_df)
        eris = mp_cpu.mp2._make_eris(pt_cpu, mo_coeff=mf.mo_coeff, ao2mofn=mf_df.with_df.ao2mo)
        g = eris.ovov.ravel()
        eia = mf.mo_energy[:nocc,None] - mf.mo_energy[nocc:]
        t2ref0 = g/(eia.reshape(-1,1)+eia.reshape(-1)).ravel()
        t2ref0 = t2ref0.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3)
        e, t2 = pt.kernel(mf.mo_energy, mf.mo_coeff)
        self.assertAlmostEqual(e, -0.20425449198334983, 8)
        self.assertAlmostEqual(abs(t2.get() - t2ref0).max(), 0, 8)

        pt = mp_gpu.MP2(mf.density_fit('weigend').to_gpu())
        pt.frozen = [1]
        e = pt.kernel(with_t2=False)[0]
        self.assertAlmostEqual(e, -0.14708846352674113, 8)

        pt = mp_gpu.dfmp2_old.DFMP2(mf.density_fit('weigend').to_gpu())
        e = pt.kernel(mf.mo_energy, mf.mo_coeff)[0]
        self.assertAlmostEqual(e, -0.20425449198334983, 8)

        pt.frozen = [1]
        e = pt.kernel()[0]
        self.assertAlmostEqual(e, -0.14708846352674113, 8)

        pt = mp_gpu.dfmp2_old.DFMP2(mf.to_gpu())
        pt.frozen = [1]
        pt.with_df = mf.to_gpu().density_fit('weigend').with_df
        e = pt.kernel()[0]
        self.assertAlmostEqual(e, -0.14708846352674113, 8)

    def test_to_cpu(self):
        pt = mp_gpu.mp2.MP2(mf.to_gpu())
        e_gpu = pt.kernel()[0]
        pt = pt.to_cpu()
        e_cpu = pt.kernel()[0]
        assert abs(e_cpu - e_gpu) < 1e-6

        pt = mp_gpu.dfmp2.DFMP2(mf.to_gpu())
        e_gpu = pt.kernel()[0]
        pt = pt.to_cpu()
        e_cpu = pt.kernel()[0]
        assert abs(e_cpu - e_gpu) < 1e-6

    def test_to_gpu(self):
        pt = mp_cpu.mp2.MP2(mf)
        e_cpu = pt.kernel()[0]
        pt = pt.to_gpu()
        e_gpu = pt.kernel()[0]
        assert abs(e_cpu - e_gpu) < 1e-6

        pt = mp_cpu.dfmp2.DFMP2(mf)
        e_cpu = pt.kernel()[0]
        pt = pt.to_gpu()
        e_gpu = pt.kernel()[0]
        assert abs(e_cpu - e_gpu) < 1e-6

class MP2Amplitudes(unittest.TestCase):
    def test_mp2_energy_and_gamma_utilities(self):
        import copy
        small_mol = gto.M(atom='O 0 0 0; H .7 0 .6; H -.7 0 .6',
                          basis='sto-3g', verbose=0)
        reference = scf.RHF(small_mol).run(conv_tol=1e-12)
        assert reference.converged
        cpu = mp_cpu.MP2(reference, frozen=1)
        cpu.kernel()
        gpu = mp_gpu.mp2.MP2(reference.to_gpu(), frozen=1)
        original_eris = gpu.ao2mo()
        energy, original_t2 = mp_gpu.mp2.kernel(gpu, eris=original_eris)
        numpy.testing.assert_allclose(original_t2.get(), cpu.t2, atol=2e-12, rtol=2e-11)
        assert abs(float(energy) - cpu.e_corr) < 2e-12
        for layout in ['matrix', 'tensor']:
            for complex_phase in [False, True]:
                with self.subTest(layout=layout, complex_phase=complex_phase):
                    eris = copy.copy(original_eris)
                    nocc, nmo = gpu.nocc, gpu.nmo
                    nvir = nmo - nocc
                    t2 = original_t2.copy()
                    integrals = original_eris.ovov.reshape(nocc, nvir, nocc, nvir).copy()
                    phase = numpy.ones(nmo, dtype=complex)
                    if complex_phase:
                        # These are equivalent complex canonical orbitals, not arbitrary complex
                        # tensors: the amplitudes acquire the conjugate two-electron phase.
                        phase = numpy.exp(1j * numpy.linspace(.1, 1.2, nmo))
                        occupied, virtual = cupy.asarray(phase[:nocc]), cupy.asarray(phase[nocc:])
                        factor = (occupied.conj()[:, None, None, None] * virtual[None, :, None, None]
                                  * occupied.conj()[None, None, :, None] * virtual[None, None, None, :])
                        integrals = integrals * factor
                        t2 = t2 * factor.transpose(0, 2, 1, 3).conj()
                        eris.mo_coeff = eris.mo_coeff * cupy.asarray(phase)
                    eris.ovov = integrals if layout == 'tensor' else integrals.reshape(nocc*nvir, nocc*nvir)
                    saved_t2, saved_eri = t2.copy(), eris.ovov.copy()
                    result = mp_gpu.mp2.energy(gpu, t2, eris)
                    assert abs(float(result) - cpu.e_corr) < 2e-12
                    assert abs(float(result.e_corr_ss) - cpu.e_corr_ss) < 2e-12
                    assert abs(float(result.e_corr_os) - cpu.e_corr_os) < 2e-12

                    actual = mp_gpu.mp2._gamma1_intermediates(gpu, t2=t2)
                    reference = mp_cpu.mp2._gamma1_intermediates(cpu, t2=t2.get())
                    real_reference = mp_cpu.mp2._gamma1_intermediates(cpu, t2=cpu.t2)
                    for block, expected, real_block, orbital_phase in zip(
                            actual, reference, real_reference, [phase[:nocc], phase[nocc:]]):
                        numpy.testing.assert_allclose(block.get(), expected, atol=2e-13, rtol=2e-11)
                        numpy.testing.assert_allclose(block.get(), block.get().conj().T, atol=2e-13)
                        transformed = orbital_phase.conj()[:, None] * real_block * orbital_phase[None, :]
                        numpy.testing.assert_allclose(block.get(), transformed, atol=2e-13, rtol=2e-11)
                    assert abs(float((cupy.trace(actual[0]) + cupy.trace(actual[1])).real)) < 2e-13
                    assert numpy.linalg.eigvalsh(actual[0].get()).max() < 2e-13
                    assert numpy.linalg.eigvalsh(actual[1].get()).min() > -2e-13

                    # The lower-storage helper must reconstruct the same unrelaxed density
                    # correction directly from physical integrals without retaining T2.
                    stored = copy.copy(gpu)
                    stored.t2 = t2
                    for a, b in zip(mp_gpu.mp2._gamma1_intermediates(stored), actual):
                        numpy.testing.assert_allclose(a.get(), b.get(), atol=2e-13, rtol=2e-11)
                    no_amplitudes = copy.copy(gpu)
                    no_amplitudes.t2 = None
                    reconstructed = mp_gpu.mp2._gamma1_intermediates(no_amplitudes, eris=eris)
                    for a, b in zip(reconstructed, actual):
                        numpy.testing.assert_allclose(a.get(), b.get(), atol=2e-13, rtol=2e-11)
                    if not complex_phase:
                        for a, b in zip(mp_gpu.mp2._gamma1_intermediates(no_amplitudes), actual):
                            numpy.testing.assert_allclose(a.get(), b.get(), atol=2e-13, rtol=2e-11)
                    numpy.testing.assert_array_equal(t2.get(), saved_t2.get())
                    numpy.testing.assert_array_equal(eris.ovov.get(), saved_eri.get())


if __name__ == "__main__":
    print("Full Tests for mp2")
    unittest.main()
