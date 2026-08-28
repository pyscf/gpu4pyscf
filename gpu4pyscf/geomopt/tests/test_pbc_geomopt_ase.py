# Copyright 2025 The PySCF Developers. All Rights Reserved.
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

try:
    import ase
except ImportError:
    ase = None
import numpy as np
import pyscf
import pytest
from pyscf import lib
from pyscf.data.nist import BOHR, HARTREE2EV

if ase is not None:
    from pyscf.pbc.tools.pyscf_ase import pyscf_to_ase_atoms


class _FakeGradients:
    def kernel(self):
        return np.ones((self.cell.natm, 3))

    def get_stress(self):
        return np.eye(3)


class _FakeScanner:
    converged = True

    def __call__(self, cell):
        self.cell = cell
        return 0.

    def Gradients(self):
        gradients = _FakeGradients()
        gradients.cell = self.cell
        return gradients


class _FakeMethod(lib.StreamObject):
    def __init__(self, cell):
        self.cell = cell
        self.mo_coeff = None
        self.converged = True
        self.dm0 = None
        self.dm = np.ones((1, 1))

    def as_scanner(self):
        return _FakeScanner()

    def kernel(self, dm0=None):
        self.dm0 = dm0
        return 0.

    def make_rdm1(self):
        return self.dm

    def Gradients(self):
        gradients = _FakeGradients()
        gradients.cell = self.cell
        return gradients


@pytest.mark.skipif(ase is None, reason='ASE not available')
def test_ase_stress_units():
    from gpu4pyscf.tools.ase_interface import PySCF

    cell = pyscf.M(
        atom='He 0 0 0', a=np.eye(3) * 4., unit='Angstrom',
        basis='gth-szv', pseudo='gth-pade', precision=1e-8, verbose=0)
    atoms = pyscf_to_ase_atoms(cell)

    calculator = PySCF(method=_FakeMethod(cell))
    calculator.calculate(atoms, properties=['stress'])

    assert np.allclose(
        calculator.results['stress'], np.eye(3) * HARTREE2EV / BOHR**3)


@pytest.mark.skipif(ase is None, reason='ASE not available')
def test_ase_method_factory_rebuilds_each_geometry():
    from gpu4pyscf.tools.ase_interface import PySCF

    cell = pyscf.M(
        atom='He 0 0 0', a=np.eye(3) * 4., unit='Angstrom',
        basis='gth-szv', pseudo='gth-pade', precision=1e-8, verbose=0)
    initial_mesh = cell.mesh.copy()
    methods = []

    def method_factory(new_cell):
        method = _FakeMethod(new_cell)
        method.dm[:] = len(methods) + 1
        methods.append(method)
        return method

    atoms = pyscf_to_ase_atoms(cell)
    calculator = PySCF(
        method=_FakeMethod(cell), method_factory=method_factory)
    atoms.set_cell(atoms.cell * 1.4, scale_atoms=True)
    calculator.calculate(atoms, properties=['stress'])
    atoms.set_cell(atoms.cell * 1.1, scale_atoms=True)
    calculator.calculate(atoms, properties=['stress'])

    assert len(methods) == 2
    assert methods[0] is not methods[1]
    assert methods[0].cell is not methods[1].cell
    assert not np.array_equal(methods[0].cell.mesh, initial_mesh)
    assert not np.array_equal(methods[0].cell.mesh, methods[1].cell.mesh)
    assert methods[0].dm0 is None
    assert np.array_equal(methods[1].dm0, methods[0].dm)


@pytest.mark.skipif(ase is None, reason='ASE not available')
def test_ase_multigrid_method_factory():
    from gpu4pyscf.tools.ase_interface import PySCF

    cell = pyscf.M(
        atom='He 0 0 0', a=np.eye(3) * 4., unit='Angstrom',
        basis='gth-szv', pseudo='gth-pade', precision=1e-8,
        mesh=[15] * 3, verbose=0)
    methods = []

    def method_factory(new_cell):
        method = new_cell.RKS(xc='lda,vwn').to_gpu().multigrid_numint()
        methods.append(method)
        return method

    atoms = pyscf_to_ase_atoms(cell)
    atoms.calc = PySCF(
        method=method_factory(cell), method_factory=method_factory)
    stress0 = atoms.get_stress()
    atoms.set_cell(atoms.cell * 1.01, scale_atoms=True)
    stress1 = atoms.get_stress()

    assert np.isfinite(stress0).all()
    assert np.isfinite(stress1).all()
    assert len(methods) == 3
    assert methods[1] is not methods[2]
    for method in methods[1:]:
        ni = method._numint
        assert ni.mg_envs.nimgs == len(ni.bvkcell.get_lattice_Ls())
        assert np.array_equal(ni.mesh, method.cell.mesh)


@pytest.mark.skipif(ase is None, reason='ASE not available')
def test_pbc_optimizer_removes_translation(monkeypatch):
    from gpu4pyscf.geomopt import ase_solver

    captured = []

    class FakeBFGS:
        def __init__(self, atoms, logfile=None):
            captured.append(atoms)

        def run(self, fmax, steps):
            return True

    monkeypatch.setattr(ase_solver, 'BFGS', FakeBFGS)
    cell = pyscf.M(
        atom='He 0 0 0; He 1 1 1', a=np.eye(3) * 4., unit='Angstrom',
        basis='gth-szv', pseudo='gth-pade', mesh=[15] * 3, verbose=0)

    for target in (None, 'atoms', 'cell', 'lattice'):
        ase_solver.kernel(
            _FakeMethod(cell), target=target, logfile=None, max_steps=0)
        system = captured[-1]
        atoms = getattr(system, 'atoms', system)
        total_force = atoms.get_forces().sum(axis=0)
        if target == 'lattice':
            assert not np.allclose(total_force, 0.)
        else:
            assert np.allclose(total_force, 0.)

@pytest.mark.skipif(ase is None, reason='ASE not available')
def test_ase_optimize_cell():
    cell = pyscf.M(
        atom='''
    C 0.  0.  0.
    C 1.1 1.1 1.1
    ''', a='''
    0. , 2.2, 2.2
    2.2, 0. , 2.2
    2.2, 2.2, 0.
    ''', basis='gth-dzv', pseudo='gth-pade', mesh=[29]*3,
        output='/dev/null', verbose=5)

    mf = cell.KRKS(xc='pbe').to_gpu()
    opt = mf.Gradients().optimizer().run()
    cell = opt.cell
    a = cell.lattice_vectors()
    atom_coords = cell.atom_coords()
    assert abs(atom_coords[0,0]) < 1e-5
    assert abs(atom_coords[1,0] - 2.10721898) < 5e-4
    assert abs(atom_coords[1,0]*2 - a[0,1]) < 1e-7

@pytest.mark.skipif(ase is None, reason='ASE not available')
def test_ase_optimize_mol():
    from gpu4pyscf.geomopt.ase_solver import GeometryOptimizer
    mol = pyscf.M(
        atom = '''
O      0.000    0.    0.
H     -0.757    0.    0.58
H      0.757    0.    0.58
''', basis='def2-svp', output='/dev/null', verbose=5)

    mf = mol.RHF().to_gpu().density_fit()
    opt = GeometryOptimizer(mf).run()
    mol = opt.mol
    atom_coords = mol.atom_coords()
    assert abs(atom_coords[2,0] - 1.42162605) < 1e-5
