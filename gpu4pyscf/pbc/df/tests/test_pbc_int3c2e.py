# Copyright 2024-2025 The PySCF Developers. All Rights Reserved.
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
from pyscf.pbc.df import rsdf_builder
from gpu4pyscf.pbc.df import int3c2e
from gpu4pyscf.pbc.df.int3c2e import sr_aux_e2, fill_triu_bvk
from gpu4pyscf.pbc.df.int2c2e import sr_int2c2e
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.pbc.lib.kpts_helper import conj_images_in_bvk_cell


def test_int3c2e_gamma_point():
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]),
               'C2': 'ccpvdz'},
        precision = 1e-8,
        a=np.diag([2.5, 1.9, 2.2])*3)

    auxcell = cell.copy()
    auxcell.basis = {
        'C1':'''
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
    dat = int3c2e.sr_aux_e2(cell, auxcell, omega).get()

    cell.precision=1e-10
    cell.build()
    df = rsdf_builder._RSGDFBuilder(cell, auxcell).build(omega=abs(omega))
    int3c = df.gen_int3c_kernel('int3c2e', aosym='s1', return_complex=True)
    ref = int3c().reshape(dat.shape)
    assert abs(dat - ref).max() < 1e-8

def test_int3c2e_kpoints():
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis='ccpvdz',
        precision = 1e-8,
        a=np.diag([2.5, 1.9, 2.2])*2)
    auxcell = cell.copy()
    auxcell.basis = [[0, [3.5, 1.]],
                     [0, [1.1, 1.]],
                     [1, [0.7, 1.]],
                     [2, [1.5, 1.]]]
    auxcell.build()
    kpts = cell.make_kpts([2,5,1])
    omega = -0.2
    dat = int3c2e.sr_aux_e2(cell, auxcell, omega, kpts).get()

    cell.precision=1e-10
    cell.build()
    df = rsdf_builder._RSGDFBuilder(cell, auxcell, kpts).build(omega=abs(omega))
    int3c = df.gen_int3c_kernel('int3c2e', aosym='s1', return_complex=True)
    ref = int3c().reshape(dat.shape)
    assert abs(dat - ref).max() < 1e-8

def test_minor_diffused_basis():
    cell = pyscf.M(
        atom='''H   1.3    .2       .3
                H   .19   .1      1.1
        ''',
        basis='''
C    S
      7.5                    0.40
      2.6                    0.90
      0.5                    0.08''',
        precision = 1e-8,
        a=np.diag([2.5, 1.9, 2.2])*3)
    auxcell = cell.copy()
    auxcell.basis = '''
C    P
      1.4947618600           1.0000000000
C    P
      0.5769010900           1.0000000000
C    D
      0.1995412500           1.0000000000 '''
    auxcell.build()
    omega = -0.2
    dat = int3c2e.sr_aux_e2(cell, auxcell, omega).get()

    cell.precision=1e-12
    cell.build()
    df = rsdf_builder._RSGDFBuilder(cell, auxcell).build(omega=abs(omega))
    int3c = df.gen_int3c_kernel('int3c2e', aosym='s1', return_complex=True)
    ref = int3c().reshape(dat.shape)
    assert abs(dat - ref).max() < 1e-8

def test_ignorable_diffused_basis():
    cell = pyscf.M(
        atom='''H   1.3    .2       .3
                H   .19   .1      1.1
        ''',
        basis='''
C    S
      7.5                    0.4000000
      2.6                    0.9000000
      0.5                    0.0000002''',
        precision = 1e-8,
        a=np.diag([2.5, 1.9, 2.2])*3)
    auxcell = cell.copy()
    auxcell.basis = '''
C    P
      1.4947618600           1.0000000000
C    P
      0.5769010900           1.0000000000
C    D
      0.1995412500           1.0000000000 '''
    auxcell.build()
    omega = -0.2
    cell.verbose = 6
    dat = int3c2e.sr_aux_e2(cell, auxcell, omega).get()

    cell.basis='''
C S
      7.5                    0.4000000
      2.6                    0.9000000'''
    cell.build()
    df = rsdf_builder._RSGDFBuilder(cell, auxcell).build(omega=abs(omega))
    int3c = df.gen_int3c_kernel('int3c2e', aosym='s1', return_complex=True)
    ref = int3c().reshape(dat.shape)
    assert abs(dat - ref).max() < 1e-6

def test_aopair_fill_triu():
    bvk_ncells, nao = 6, 13
    out = cp.random.rand(bvk_ncells,nao,nao)
    conj_mapping = cp.asarray(conj_images_in_bvk_cell([bvk_ncells,1,1]), dtype=np.int32)
    ix, iy = cp.tril_indices(nao, -1)
    ref = out.copy()
    for k, ck in enumerate(conj_mapping):
        ref[ck,iy,ix] = ref[k,ix,iy]
    out = fill_triu_bvk(out, nao, [bvk_ncells,1,1])
    assert abs(out-ref).max() == 0.

def test_sr_int2c2e():
    cell = pyscf.M(
        atom='''C1  1.3    .2       .3
                C2  .19   .1      1.1
                C3  0.  0.  0.
        ''',
        precision = 1e-8,
        a=np.diag([2.5, 1.9, 2.2])*3,
        basis='def2-universal-jkfit')
    omega = 0.2
    dat = sr_int2c2e(cell, -omega).get()

    kmesh = [6, 1, 1]
    kpts = cell.make_kpts(kmesh)
    auxcell_sr = cell.copy()
    auxcell_sr.precision = 1e-14
    auxcell_sr.rcut = 50
    with auxcell_sr.with_short_range_coulomb(omega):
        ref = auxcell_sr.pbc_intor('int2c2e', hermi=1, kpts=kpts)
    assert abs(dat - ref[0]).max() < 1e-10

    dat = sr_int2c2e(cell, -omega, kpts=kpts, bvk_kmesh=kmesh).get()
    assert abs(dat - ref).max() < 1e-10

    cell = cell.copy()
    cell.basis = {
        'C1':'''
C    S
     12.9917624900           1.0000000000
C    S
      2.1325940100           1.0000000000
C    P
      9.8364318200           1.0000000000
C    P
      3.3490545000           1.0000000000
C    P
      1.4947618600           1.0000000000
C    P
      0.5769010900           1.0000000000
C    D
      0.6                    1.0000000000 ''',
        'C2':[[0, [.5, 1.]]],
    }
    cell.build()
    omega = 0.2
    dat = sr_int2c2e(cell, -omega).get()
    auxcell_sr = cell.copy()
    auxcell_sr.precision = 1e-14
    auxcell_sr.rcut = 50
    with auxcell_sr.with_short_range_coulomb(omega):
        ref = auxcell_sr.pbc_intor('int2c2e', hermi=1, kpts=kpts)
    assert abs(dat - ref[0]).max() < 1e-10

    dat = sr_int2c2e(cell, -omega, kpts=kpts, bvk_kmesh=kmesh).get()
    assert abs(dat - ref).max() < 1e-10

def test_contract_dm_gamma_point():
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]),
               'C2': 'ccpvdz'},
        precision = 1e-8,
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
      9.8364318200           1.0000000000
      3.3490545000           1.0000000000
C    P
      1.4947618600           1.0000000000
C    P
      0.5769010900           1.0000000000
C    D
      0.1995412500           1.0000000000 ''',
        'C2': ('unc-weigend', [[0, [.5, 1.]], [1, [.8, 1.]], [3, [.9, 1]]]),
    }
    auxcell.build()
    omega = -0.2

    np.random.seed(9)
    nao = cell.nao
    dm = np.random.rand(nao, nao) - .5
    dm = cp.asarray(dm.dot(dm.T))
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega).build()
    jaux = opt.contract_dm(opt.cell.apply_C_mat_CT(dm))

    j3c = int3c2e.sr_aux_e2(cell, auxcell, omega)
    ref = cp.einsum('pqr,qp->r', j3c, dm)
    assert abs(jaux - ref).max() < 1e-9

    np.random.seed(9)
    auxvec = cp.asarray(np.random.rand(auxcell.nao))
    vj = opt.contract_auxvec(opt.auxcell.apply_C_dot(auxvec))
    ref = cp.einsum('pqr,r->pq', j3c, auxvec)
    assert abs(vj - ref).max() < 1e-10

def test_contract_dm_kpts():
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]),
               'C2': 'ccpvdz'},
        precision = 1e-8,
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
      9.8364318200           1.0000000000
      3.3490545000           1.0000000000
C    P
      1.4947618600           1.0000000000
C    P
      0.5769010900           1.0000000000
C    D
      0.1995412500           1.0000000000 ''',
        'C2': ('unc-weigend', [[0, [.5, 1.]], [1, [.8, 1.]], [3, [.9, 1]]]),
    }
    auxcell.build()
    omega = -0.2

    kmesh = [3,1,4]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    dm = cp.asarray(cell.pbc_intor('int1e_ovlp', kpts=kpts))
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega, kmesh).build()
    jaux = opt.contract_dm(opt.cell.apply_C_mat_CT(dm), kpts=kpts)

    j3c = int3c2e.sr_aux_e2(cell, auxcell, omega, kpts, kmesh, j_only=True)
    ref = cp.einsum('kpqr,kqp->r', j3c, dm) / nkpts
    assert abs(jaux - ref).max() < 3e-10

    np.random.seed(9)
    auxvec = np.random.rand(auxcell.nao)
    vj = opt.contract_auxvec(opt.auxcell.apply_C_dot(auxvec), kpts=kpts)
    ref = cp.einsum('kpqr,r->kpq', j3c, auxvec)
    assert abs(vj - ref).max() < 1e-10

def test_int3c2e_batch_evaluation():
    from gpu4pyscf.df.int3c2e_bdiv import argsort_aux
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]),
               'C2': 'ccpvdz'},
        precision = 1e-8,
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
        'C2': ('unc-weigend', [[0, [.5, 1.]], [1, [.8, 1.]], [3, [.9, 1]]]),
    }
    auxcell.build()
    omega = -0.2
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega).build()
    eval_j3c, aux_sorting = opt.int3c2e_evaluator()[:2]
    dat = eval_j3c()

    nao = cell.nao
    naux = auxcell.nao
    pair_address = opt.pair_and_diag_indices()[0]
    i, j = divmod(pair_address, nao)
    j3c = cp.zeros((nao, nao, naux))
    j3c[j, i] = j3c[i, j] = dat[:,aux_sorting,0].dot(opt.auxcell.ctr_coeff)

    cell.precision=1e-10
    cell.build()
    df = rsdf_builder._RSGDFBuilder(cell, auxcell).build(omega=abs(omega))
    int3c = df.gen_int3c_kernel('int3c2e', aosym='s1', return_complex=True)
    ref = int3c().reshape(j3c.shape)
    assert abs(j3c.get() - ref).max() < 1e-8

    ref = dat[:,aux_sorting]
    batch_size = int(ref.shape[0] *.23)
    eval_j3c, aux_sorting, ao_pair_offsets = opt.int3c2e_evaluator(
        ao_pair_batch_size=batch_size)[:3]
    dat = cp.empty_like(ref)
    for i, (p0, p1) in enumerate(zip(ao_pair_offsets[:-1],
                                     ao_pair_offsets[1:])):
        dat[p0:p1] = eval_j3c(i)
    assert abs(dat[:,aux_sorting] - ref).max() < 1e-12

    batch_size = int(ref.shape[1] * 0.22)
    eval_j3c, aux_sorting, ao_pair_offsets, aux_offsets = opt.int3c2e_evaluator(
        aux_batch_size=batch_size)[:4]
    dat = cp.empty_like(ref)
    for i, (p0, p1) in enumerate(zip(aux_offsets[:-1], aux_offsets[1:])):
        dat[:,p0:p1] = eval_j3c(aux_batch_id=i)
    assert abs(dat[:,aux_sorting] - ref).max() < 2e-10

    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega, bvk_kmesh=[3,1,2]).build()
    eval_j3c, aux_sorting = opt.int3c2e_evaluator()[:2]
    ref = eval_j3c()[:,aux_sorting]
    batch_size = int(ref.shape[0] *.23)

    eval_j3c, aux_sorting, ao_pair_offsets = opt.int3c2e_evaluator(
        ao_pair_batch_size=batch_size)[:3]
    dat = cp.empty_like(ref)
    for i, (p0, p1) in enumerate(zip(ao_pair_offsets[:-1],
                                     ao_pair_offsets[1:])):
        dat[p0:p1] = eval_j3c(i)
    assert abs(dat[:,aux_sorting] - ref).max() < 1e-12

    batch_size = int(ref.shape[1] * 0.22)
    eval_j3c, aux_sorting, ao_pair_offsets, aux_offsets = opt.int3c2e_evaluator(
        aux_batch_size=batch_size)[:4]
    dat = cp.empty_like(ref)
    for i, (p0, p1) in enumerate(zip(aux_offsets[:-1], aux_offsets[1:])):
        dat[:,p0:p1] = eval_j3c(aux_batch_id=i)
    assert abs(dat[:,aux_sorting] - ref).max() < 1e-10
