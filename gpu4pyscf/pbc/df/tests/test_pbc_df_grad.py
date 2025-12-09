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

import unittest
import numpy as np
import cupy as cp
import pyscf
from pyscf import lib
from gpu4pyscf.pbc.df import int3c2e
from gpu4pyscf.pbc.df.grad.rhf import _jk_energy_per_atom

def test_ej_ip1_gamma_point():
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]),
               'C2': 'ccpvdz'},
        precision = 1e-10,
        a=np.diag([2.5, 1.9, 2.2])*3)

    auxcell = cell.copy()
    auxcell.basis = {
        'C1':'''
C    S
      0.5000000000           1.0000000000
C    P
    102.9917624900           1.0000000000
C    P
     28.1325940100           1.0000000000
C    P
      9.8364318200           1.0000000000
C    P
      3.3490545000           1.0000000000
C    P
      1.4947618600           1.0000000000
C    P
      0.5769010900           1.0000000000
C    D
      0.1995412500           1.0000000000 ''',
        'C2':[[0, [.5, 1.]]],
    }
    auxcell.build()
    omega = -0.2

    np.random.seed(8)
    nao = cell.nao
    nocc = 4
    mo_coeff = np.random.rand(nao, nao) - .5
    mo_occ = np.zeros(nao)
    mo_occ[:nocc] = 2
    opt = int3c2e.SRInt3c2eOpt_v2(cell, auxcell, omega).build()
    ej = _jk_energy_per_atom(opt, mo_coeff, mo_occ, k_factor=0)
    assert abs(ej.sum(axis=0)).max() < 1e-12

    dm = (mo_coeff*mo_occ).dot(mo_coeff.T)
    disp = 1e-3
    atom_coords = cell.atom_coords()
    def eval_j(i, x, disp):
        atom_coords[i,x] += disp
        cell1 = cell.set_geom_(atom_coords, unit='Bohr')
        auxcell1 = auxcell.set_geom_(atom_coords, unit='Bohr')
        opt = int3c2e.SRInt3c2eOpt_v2(cell1, auxcell1, omega).build()
        jaux = opt.contract_dm(dm)
        cp.cuda.get_current_stream().synchronize()
        j2c = int3c2e.sr_int2c2e(auxcell1, omega)[0]
        atom_coords[i,x] -= disp
        return cp.linalg.solve(j2c, jaux).dot(jaux) * .5

    for i, x in [(0, 0), (0, 1), (0, 2)]:
        e1 = eval_j(i, x, disp)
        e2 = eval_j(i, x, -disp)
        assert abs((e1 - e2)/(2*disp)- ej[i,x]) < 5e-6
