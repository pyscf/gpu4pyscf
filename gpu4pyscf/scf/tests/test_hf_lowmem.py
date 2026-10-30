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
import tempfile
import numpy as np
import cupy
import pyscf
from pyscf import lib
from gpu4pyscf.scf import hf_lowmem
from gpu4pyscf.lib.cupy_helper import unpack_tril

mol = pyscf.M(
    atom = '''
    O       0.0000000000    -0.0000000000     0.1174000000
    H      -0.7570000000    -0.0000000000    -0.4696000000
    H       0.7570000000     0.0000000000    -0.4696000000
    ''',
    basis='ccpvdz',
    spin=None,
    output = '/dev/null'
)

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    def test_rhf_scf(self):
        e_tot = hf_lowmem.RHF(mol).kernel()
        e_ref = -76.02676567311744
        assert np.abs(e_tot - e_ref) < 1e-8

    def test_diis_on_cpu(self):
        mf = hf_lowmem.RHF(mol)
        mf.diis = mf.DIIS()
        mf.diis.incore = False
        e_tot = mf.kernel()
        e_ref = -76.02676567311744
        assert np.abs(e_tot - e_ref) < 1e-8
        assert all(isinstance(x, np.ndarray) and not isinstance(x, cupy.ndarray)
                   for x in mf.diis._buffer.values())



class SmallLowMemoryContracts(unittest.TestCase):
    def test_kernel_zero_cycles(self):
        mol = pyscf.M(atom='H 0 0 0; H 0 0 .8', basis='sto-3g', verbose=0)
        cpu = mol.RHF().run(conv_tol=1e-12)
        density = cupy.asarray(cpu.make_rdm1())
        mf = hf_lowmem.RHF(mol)
        mf.max_cycle = 0
        np.testing.assert_allclose(mf.kernel(dm0=density), cpu.e_tot, atol=1e-10)
        self.assertFalse(mf.converged)
        coeff = mf.mo_coeff.copy()
        np.testing.assert_allclose(mf.kernel(), cpu.e_tot, atol=1e-10)
        np.testing.assert_array_equal(mf.mo_coeff.get(), coeff.get())

    def _lih_reference(self):
        mol = pyscf.M(atom='Li 0 0 0; H .1 -.2 1.6', basis='sto-3g', verbose=0)
        cpu = mol.RHF().run(conv_tol=1e-12)
        mf = hf_lowmem.RHF(mol)
        mf.mo_coeff = cupy.asarray(cpu.mo_coeff)
        mf.mo_occ = cupy.asarray(cpu.mo_occ)
        return mf, cpu

    def _check_incremental_potential(self, old_wavefunction, new_wavefunction):
        mf, cpu = self._lih_reference()
        mol = mf.mol
        density = mf.make_rdm1()
        rotation = np.eye(mol.nao)
        angle = .13
        rotation[np.ix_([0, 2], [0, 2])] = [[np.cos(angle), -np.sin(angle)],
                                          [np.sin(angle), np.cos(angle)]]
        previous = mf.make_wfn(mf.mo_coeff @ cupy.asarray(rotation), mf.mo_occ)
        previous_density = previous.make_rdm1()
        v0 = mf.get_veff(mol, previous)
        old = previous if old_wavefunction else previous_density
        new = mf.make_wfn(mf.mo_coeff, mf.mo_occ) if new_wavefunction else density
        actual = mf.get_veff(mol, new, dm_last=old, vhf_last=v0)
        full = mf.get_veff(mol, new)
        reference = cpu.get_veff(mol, density.get())
        np.testing.assert_allclose(actual, full, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(unpack_tril(cupy.asarray(actual)).get(), reference, atol=1e-11)
        np.testing.assert_allclose(mf.energy_elec(new, mf.get_hcore(), actual),
                                   cpu.energy_elec(density.get()), atol=1e-11)
        np.testing.assert_array_equal(previous.make_rdm1().get(), previous_density.get())

    def test_incremental_potential_wavefunctions(self):
        self._check_incremental_potential(True, True)

    def test_incremental_potential_densities(self):
        self._check_incremental_potential(False, False)

    def test_incremental_potential_wavefunction_to_density(self):
        self._check_incremental_potential(True, False)

    def test_incremental_potential_density_to_wavefunction(self):
        self._check_incremental_potential(False, True)

    def test_default_fock(self):
        mf, cpu = self._lih_reference()
        expected = cpu.get_hcore() + cpu.get_veff()
        np.testing.assert_allclose(mf.get_fock().get(), expected, atol=1e-11)

    def test_level_shift_with_packed_overlap(self):
        mf, cpu = self._lih_reference()
        mol = mf.mol
        density = mf.make_rdm1()
        wavefunction = mf.make_wfn(mf.mo_coeff, mf.mo_occ)
        overlap = mol.intor('int1e_ovlp')
        packed_s = overlap[np.tril_indices(mol.nao)]
        potential = mf.get_veff(mol, wavefunction)
        shift = .2
        actual = mf.get_fock(s1e=packed_s, vhf=potential, cycle=0,
                             dm_or_wfn=wavefunction, level_shift_factor=shift)
        expected = pyscf.scf.hf.level_shift(overlap, density.get()*.5,
                                            cpu.get_hcore()+cpu.get_veff(), shift)
        np.testing.assert_allclose(actual.get(), expected, atol=1e-11)


if __name__ == "__main__":
    print("Full Tests for hf_lowmem")
    unittest.main()
