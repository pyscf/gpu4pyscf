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
        mf_gpu._numint.spin_samples = 50
        mf_gpu.conv_tol = 1e-12
        e_tot = mf_gpu.kernel()
        if mcfun is not None:
            mf_cpu = cpu_dft.GKS(mol, xc='b3lyp').density_fit(auxbasis='def2-tzvpp-jkfit')
            mf_cpu.collinear = 'm'
            mf_cpu._numint.spin_samples = 50
            mf_cpu.conv_tol = 1e-12
            e_pyscf = mf_cpu.kernel()
            assert np.abs(e_tot - e_pyscf) < 1e-5
            assert np.abs(mf_cpu.mo_energy - mf_gpu.mo_energy.get()).max() < 1e-5
        assert np.abs(e_tot - -75.99877509321838) < 1e-5

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


class SmallDFGKSPotentials(unittest.TestCase):
    auxbasis = [[0, [1.6, 1.]], [0, [.4, 1.]], [1, [.8, 1.]], [2, [.6, 1.]]]
    long_range = '0.4*LR_HF(0.3)+0.6*LDA_X,LDA_C_VWN'
    short_range = '0.25*SR_HF(0.3)+0.75*LDA_X,LDA_C_VWN'
    functionals = ('LDA,VWN', 'PBE', 'PBE0', 'CAM-B3LYP', long_range, short_range)

    @classmethod
    def setUpClass(cls):
        from scipy.linalg import eigh
        cls.integrals = {}
        mol = pyscf.M(atom='Li 0 0 0', basis='sto-3g', spin=1, verbose=0)
        hcore = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        _, coeff = eigh(hcore, mol.intor('int1e_ovlp'))
        na, nb = mol.nelec
        blocks = [coeff[:, :n] @ coeff[:, :n].T for n in (na, nb)]
        dm = np.zeros((2*mol.nao, 2*mol.nao))
        dm[:mol.nao, :mol.nao], dm[mol.nao:, mol.nao:] = blocks
        grids = cpu_dft.gen_grid.Grids(mol)
        grids.atom_grid = (20, 50)
        grids.prune = None
        grids.build()
        cls.mol, cls.hcore, cls.coeff = mol, hcore, coeff
        cls.dm, cls.grids = dm, grids

    def _method(self, xc, mode='mix_outside_kernel'):
        mol, grids = self.mol, self.grids
        mf = gpu_dft.GKS(mol, xc=xc).density_fit(auxbasis=self.auxbasis)
        mf.collinear = 'm'
        mf._numint.spin_samples = 50
        mf.small_rho_cutoff = 0
        mf.range_separated_mode = mode
        mf.grids.atom_grid = (20, 50)
        mf.grids.prune = None
        mf.grids.coords = cupy.asarray(grids.coords)
        mf.grids.weights = cupy.asarray(grids.weights)
        return mf

    def _three_center(self, omega):
        from pyscf import df
        if omega not in self.integrals:
            mol = self.mol
            aux = df.addons.make_auxmol(mol, self.auxbasis)
            with mol.with_range_coulomb(omega), aux.with_range_coulomb(omega):
                three = df.incore.aux_e2(mol, aux)
                metric = aux.intor('int2c2e')
            self.integrals[omega] = three.reshape(mol.nao**2, -1), metric
        return self.integrals[omega]

    def _eri(self, omega=0., lr=1., sr=1.):
        three, metric = self._three_center(0.)
        if omega:
            three_lr, metric_lr = self._three_center(omega)
            three = sr*three + (lr-sr)*three_lr
            metric = sr*metric + (lr-sr)*metric_lr
        # These deliberately small independent bases have a well-conditioned metric.
        from pyscf.df.incore import LINEAR_DEP_THR
        assert np.linalg.eigvalsh(metric).min() > LINEAR_DEP_THR
        nao = self.mol.nao
        return (three @ np.linalg.solve(metric, three.T)).reshape((nao,)*4)

    def _cpu_df_terms(self, xc, dm, inside=False):
        # inside=True matches mix_inside_kernel: mix the range-separated
        # three-center integrals and auxiliary metrics before density fitting.
        # False fits each range separately, then mixes the exchange contributions.
        mol = self.mol
        omega, alpha, hyb = cpu_dft.numint.NumInt().rsh_and_hybrid_coeff(xc, spin=mol.spin)
        full = self._eri()
        if omega and inside:
            exchange = self._eri(omega, alpha, hyb)
        elif omega:
            exchange = hyb*full + (alpha-hyb)*self._eri(omega, 1., 0.)
        else:
            exchange = hyb*full
        n = mol.nao
        charge = dm[:n, :n] + dm[n:, n:]
        j = np.einsum('pqrs,sr->pq', full, charge)
        vj = np.kron(np.eye(2), j)
        vk = np.empty_like(dm)
        for a in range(2):
            for b in range(2):
                vk[a*n:(a+1)*n, b*n:(b+1)*n] = np.einsum(
                    'prsq,rs->pq', exchange, dm[a*n:(a+1)*n, b*n:(b+1)*n])
        return vj, vk

    def _check_components(self, xc, dm=None, inside=False):
        hcore, original = self.hcore, self.dm
        if dm is None:
            dm = original
        mode = 'mix_inside_kernel' if inside else 'mix_outside_kernel'
        mf = self._method(xc, mode)
        gpu_dm = cupy.asarray(dm)
        result = mf.get_veff(dm=gpu_dm)
        # Use the actual unchanged NumInt2C integration result, not a fake XC
        # callback. This isolates DF assembly from known finite-spin quadrature errors.
        _, exc_xc, vxc = mf._numint.get_vxc(mf.mol, mf.grids, xc, gpu_dm)
        j, k = self._cpu_df_terms(xc, dm, inside)
        np.testing.assert_allclose(result.get()-vxc.get(), j-k, atol=1e-9, rtol=0)
        ecoul = .5*np.einsum('ij,ji->', dm, j).real
        exx = -.5*np.einsum('ij,ji->', dm, k).real
        self.assertAlmostEqual(float(result.ecoul), ecoul, delta=1e-9)
        self.assertAlmostEqual(float(result.exc-exc_xc), exx, delta=1e-9)
        hspin = np.kron(np.eye(2), hcore)
        energy = mf.energy_elec(gpu_dm, h1e=cupy.asarray(hspin), vhf=result)[0]
        expected = np.einsum('ij,ji->', dm, hspin).real + ecoul + exx + float(exc_xc)
        self.assertAlmostEqual(float(energy), expected, delta=1e-9)
        assert mf.mo_coeff is None
        return result.get()-vxc.get(), ecoul, exx

    def test_open_shell_df_assembly(self):
        for xc in self.functionals:
            with self.subTest(xc=xc):
                self._check_components(xc)

    def test_inside_kernel_rsh_assembly(self):
        self._check_components('cam-b3lyp', inside=True)

    def test_complex_spin_rotation_of_df_terms(self):
        mol, dm = self.mol, self.dm
        theta, phi = .67, .31
        u = np.array([[np.cos(theta/2), -np.sin(theta/2)*np.exp(-1j*phi)],
                      [np.sin(theta/2)*np.exp(1j*phi), np.cos(theta/2)]])
        rotation = np.kron(u, np.eye(mol.nao))
        rotated_dm = rotation @ dm @ rotation.conj().T
        for xc in ('LDA,VWN', 'PBE0', 'CAM-B3LYP'):
            with self.subTest(xc=xc):
                potential, ej, ek = self._check_components(xc)
                rotated, rej, rek = self._check_components(xc, rotated_dm)
                np.testing.assert_allclose(rotated, rotation @ potential @ rotation.conj().T,
                                           atol=1e-9, rtol=0)
                self.assertAlmostEqual(ej, rej, delta=1e-10)
                self.assertAlmostEqual(ek, rek, delta=1e-10)

    def test_antihermitian_zero_charge(self):
        mol, coeff = self.mol, self.coeff
        transition = np.outer(coeff[:, 0], coeff[:, 1])
        dm = np.zeros((mol.nao*2,)*2)
        dm[:mol.nao, :mol.nao] = transition-transition.T
        mf = self._method('LDA,VWN')
        result = mf.get_veff(dm=cupy.asarray(dm), hermi=2)
        np.testing.assert_allclose(result.get(), 0, atol=1e-12, rtol=0)
        self.assertAlmostEqual(float(result.ecoul), 0, delta=1e-12)
        self.assertAlmostEqual(float(result.exc), 0, delta=1e-12)

    def test_unsupported_xc_modes(self):
        dm = cupy.asarray(self.dm)
        mf = self._method('LDA,VWN')
        mf.collinear = 'col'
        with self.assertRaisesRegex(NotImplementedError, 'multi-colinear'):
            mf.get_veff(dm=dm)
        mf.collinear = 'm'
        mf.nlc = 'vv10'
        assert mf.do_nlc()
        with self.assertRaisesRegex(NotImplementedError, 'non-collinear nlc'):
            mf.get_veff(dm=dm)
        mf.nlc = ''
        mf.xc = 'CAM-B3LYP'
        mf.range_separated_mode = 'invalid'
        with self.assertRaisesRegex(ValueError, 'range_separated_mode'):
            mf.get_veff(dm=dm)


if __name__ == "__main__":
    print("Full Tests for Generalized Hartree-Fock")
    unittest.main()
