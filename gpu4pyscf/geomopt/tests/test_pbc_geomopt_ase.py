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
    from ase.constraints import FixCom
    from pyscf.pbc.tools.pyscf_ase import pyscf_to_ase_atoms


class _FakeGradients:
    def __init__(self, cell):
        self.cell = cell

    def kernel(self):
        return np.arange(1, self.cell.natm * 3 + 1).reshape(-1, 3)

    def get_stress(self):
        return np.eye(3)


class _FakeScanner:
    converged = True

    def __call__(self, cell):
        self.cell = cell
        return 0.

    def Gradients(self):
        return _FakeGradients(self.cell)


class _FakeMethod(lib.StreamObject):
    def __init__(self, cell):
        self.cell = cell

    def as_scanner(self):
        return _FakeScanner()


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
def test_pbc_optimizer_freezes_automatic_mesh():
    from gpu4pyscf.geomopt import ase_solver

    cell = pyscf.M(
        atom='He 0 0 0', a=np.eye(3) * 4., unit='Angstrom',
        basis='gth-szv', pseudo='gth-pade', precision=1e-8, verbose=0)
    mesh = np.asarray(cell.mesh).copy()
    assert cell._mesh_from_build

    method = _FakeMethod(cell)
    _, optimized_cell = ase_solver.kernel(method, max_steps=0)

    assert not cell._mesh_from_build
    assert method._geomopt_mesh == tuple(mesh)
    np.testing.assert_array_equal(cell.mesh, mesh)
    np.testing.assert_array_equal(optimized_cell.mesh, mesh)

    strained_cell = optimized_cell.set_geom_(
        optimized_cell.atom_coords(),
        a=optimized_cell.lattice_vectors() * 1.01,
        unit='Bohr',
        inplace=False,
    )
    np.testing.assert_array_equal(strained_cell.mesh, mesh)


@pytest.mark.skipif(ase is None, reason='ASE not available')
def test_pbc_optimizer_fixcom(monkeypatch):
    from gpu4pyscf.geomopt import ase_solver

    optimized = []

    class FakeBFGS:
        def __init__(self, atoms, logfile=None):
            optimized.append(atoms)

        def run(self, fmax, steps):
            return True

    monkeypatch.setattr(ase_solver, 'BFGS', FakeBFGS)

    for target in (None, 'atoms', 'cell', 'lattice'):
        cell = pyscf.M(
            atom='He 0 0 0; He 1 1 1',
            a=np.eye(3) * 4.,
            unit='Angstrom',
            basis='gth-szv',
            pseudo='gth-pade',
            mesh=[15] * 3,
            verbose=0,
        )
        ase_solver.kernel(
            _FakeMethod(cell),
            target=target,
            max_steps=0,
        )

        system = optimized[-1]
        atoms = getattr(system, 'atoms', system)
        has_fixcom = any(
            isinstance(constraint, FixCom)
            for constraint in atoms.constraints
        )
        assert has_fixcom == (target != 'lattice')

        if target == 'atoms':
            assert np.allclose(atoms.get_forces().sum(axis=0), 0.)

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
