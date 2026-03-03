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
import pyscf
import pytest

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
