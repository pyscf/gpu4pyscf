#!/usr/bin/env python
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

import numpy as np
from pyscf import lib
from pyscf.pbc import gto as pgto
from pyscf.pbc import dft
import pytest

def test_reset():
    cell = pgto.Cell()
    cell.unit = 'A'
    cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
    cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''
    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.verbose = 7
    cell.output = '/dev/null'
    cell.build()
    kpts0 = cell.make_kpts([3,1,1])
    mf = cell.KRKS(kpts=kpts0).to_gpu()

    cell1 = pgto.Cell()
    cell1.atom = 'C 0.,  0.,  0.; C 0.95,  0.95,  0.95'
    cell1.a = '''0.   1.9  1.9
                 1.9  0.   1.9
                 1.9  1.9  0.    '''
    cell1.basis = 'gth-dzvp'
    cell1.pseudo = 'gth-pade'
    cell1.verbose = 7
    cell1.output = '/dev/null'
    cell1.build()
    mf.reset(cell1)
    assert abs(mf.kpts - kpts0).sum() > 0.1
    ref = cell1.make_kpts([3,1,1])
    assert abs(mf.kpts - ref).max() < 1e-9

    cell1.set_geom_(a='''0.   2.0  2.0
                         2.0  0.   2.0
                         2.0  2.0  0.    ''')
    ref = cell1.make_kpts([3,1,1])
    mf.reset(cell1)
    assert abs(mf.kpts - kpts0).sum() > 0.1
    assert abs(mf.kpts - ref).max() < 1e-9

@pytest.mark.skip('KsymAdaptedKRKS for GPU is not avail')
def test_reset_ksym():
    cell = pgto.Cell()
    cell.unit = 'A'
    cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
    cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''
    cell.basis = 'gth-dzvp'
    cell.space_group_symmetry = True
    cell.pseudo = 'gth-pade'
    cell.verbose = 7
    cell.output = '/dev/null'
    cell.build()
    kpts0 = cell.make_kpts([3,1,1], space_group_symmetry=True)
    mf = dft.KRKS(cell, kpts=kpts0).to_gpu()

    ref = pgto.M(
        unit = 'A',
        atom = 'C 0.,  0.,  0.; C 0.95,  0.95,  0.95',
        a = '''0.   1.9  1.9
               1.9  0.   1.9
               1.9  1.9  0. ''',
        basis = 'gth-dzvp',
        space_group_symmetry = True,
        pseudo = 'gth-pade',
        verbose = 0).make_kpts([3,1,1], space_group_symmetry=True)
    cell.set_geom_(
        'C 0.,  0.,  0.; C 0.95,  0.95,  0.95',
        a = '''0.   1.9  1.9
               1.9  0.   1.9
               1.9  1.9  0. ''')
    mf.reset(cell)
    assert abs(mf.kpts.kpts_ibz - ref.kpts_ibz).max() < 1e-9

    ref = pgto.M(
        unit = 'A',
        atom = 'C 0.,  0.,  0.; C 0.95,  0.95,  0.95',
        a = '''0.   2.0  2.0
               2.0  0.   2.0
               2.0  2.0  0. ''',
        basis = 'gth-dzvp',
        space_group_symmetry = True,
        pseudo = 'gth-pade',
        verbose = 0).make_kpts([3,1,1], space_group_symmetry=True)
    cell.set_geom_(a='''0.   2.0  2.0
                        2.0  0.   2.0
                        2.0  2.0  0. ''')
    ref = cell.make_kpts([3,1,1], space_group_symmetry=True)
    mf.reset(cell)
    assert abs(mf.kpts.kpts_ibz - ref.kpts_ibz).max() < 1e-9

    ref = pgto.M(
        unit = 'A',
        atom = 'C 0.,  0.,  0.; C 1.,  1.,  1.',
        a = '''0.   2.0  2.0
               2.0  0.   2.0
               2.0  2.0  0. ''',
        basis = 'gth-dzvp',
        space_group_symmetry = True,
        pseudo = 'gth-pade',
        verbose = 0).make_kpts([3,1,1], space_group_symmetry=True)
    cell.set_geom_(
        'C 0.,  0.,  0.; C 1.,  1.,  1.')
    mf.reset(cell)
    assert abs(mf.kpts.kpts_ibz - ref.kpts_ibz).max() < 1e-9
