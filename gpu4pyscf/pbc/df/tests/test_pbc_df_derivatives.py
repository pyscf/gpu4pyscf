# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
import cupy as cp
import pyscf
from pyscf.pbc import gto
from pyscf.pbc.df.df import make_auxcell
from gpu4pyscf.lib.cupy_helper import tag_array, contract
from gpu4pyscf.pbc.df import int3c2e
from gpu4pyscf.pbc.df.grad import rhf, rhf_stress, uhf_stress
from gpu4pyscf.pbc.df.grad import krhf_stress, kuhf_stress
from gpu4pyscf.pbc.df.int2c2e import sr_int2c2e
from gpu4pyscf.pbc.df import rsdf_builder
from gpu4pyscf.pbc.grad.rks_stress import _finite_diff_cells

def create_cell_auxcell():
    np.random.seed(3)
    cell = pyscf.M(
        atom='''C1   1.3   2.2       .3
                C2   .19   .1      1.1
        ''',
        unit = 'Bohr',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]),
               'C2': 'ccpvdz'},
        a=np.diag([8.5, 7.5, 9.2])+np.random.rand(3,3)
    )

    auxcell = cell.copy()
    auxcell.basis = {
        'C1':'''
C    S
      0.5000000000           1.0000000000
C    P
    102.9917624900           1.0000000000
     28.1325940100           1.0000000000
      9.8364318200           1.0000000000
C    P
      3.3490545000           1.0000000000
C    P
      1.4947618600           1.0000000000
C    P
      0.4000000000           1.0000000000
C    D
      0.1995412500           1.0000000000 ''',
        'C2': ('unc-weigend', [[0, [.5, 1.]], [1, [.8, 1.]], [3, [.9, 1]]]),
    }
    auxcell.build()
    return cell, auxcell


def _check_gradient(grad, cell, auxcell, eval_energy, disp=1e-3, tol=5e-6):
    atom_coords = cell.atom_coords()
    for ia, axis in [(0, 0), (0, 1), (0, 2)]:
        coords = atom_coords.copy()
        coords[ia, axis] += disp
        cell1 = cell.set_geom_(coords, unit='Bohr', inplace=False)
        auxcell1 = auxcell.set_geom_(coords, unit='Bohr', inplace=False)
        e1 = eval_energy(cell1, auxcell1)

        coords[ia, axis] -= 2 * disp
        cell2 = cell.set_geom_(coords, unit='Bohr', inplace=False)
        auxcell2 = auxcell.set_geom_(coords, unit='Bohr', inplace=False)
        e2 = eval_energy(cell2, auxcell2)
        assert abs((e1-e2)/(2*disp) - grad[ia,axis]) < tol

def test_ej_strain_deriv_gamma_point_without_long_range():
    cell, auxcell = create_cell_auxcell()
    np.random.seed(8)
    nao = cell.nao
    nocc = 4
    mo_coeff = np.random.rand(nao, nao) - .5
    mo_occ = np.zeros(nao)
    mo_occ[:nocc] = 2
    dm = (mo_coeff*mo_occ).dot(mo_coeff.T)
    dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    omega = -0.3
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega).build()
    grad, sigma = rhf_stress._get_ejk_strain_deriv(opt, dm, hermi=1, k_factor=0, omega=omega)
    assert abs(grad.sum(axis=0)).max() < 1e-11

    disp = 1e-4
    dm_cart = opt.cell.apply_C_mat_CT(dm)
    def eval_j(c, ac):
        opt = int3c2e.SRInt3c2eOpt(c, ac, omega).build()
        jaux = opt.contract_dm(dm_cart)
        j2c = sr_int2c2e(ac, omega)
        return float(cp.linalg.solve(j2c, jaux).dot(jaux).get()) * .5

    _check_gradient(grad, cell, auxcell, eval_j)

    for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
        cell1, cell2 = _finite_diff_cells(cell, i, j, disp=disp)
        acell1, acell2 = _finite_diff_cells(auxcell, i, j, disp=disp)
        e1 = eval_j(cell1, acell1)
        e2 = eval_j(cell2, acell2)
        assert abs(sigma[i, j] - (e1-e2)/2/disp) < 5e-7

def test_ej_strain_deriv_gamma_point_with_long_range():
    cell, auxcell = create_cell_auxcell()
    np.random.seed(8)
    nao = cell.nao
    nocc = 4
    mo_coeff = np.random.rand(nao, nao) - .5
    mo_occ = np.zeros(nao)
    mo_occ[:nocc] = 2
    dm = (mo_coeff*mo_occ).dot(mo_coeff.T)
    dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    omega = -0.3
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega).build()
    grad, sigma = rhf_stress._get_ejk_strain_deriv(opt, dm, hermi=1, k_factor=0)
    assert abs(grad.sum(axis=0)).max() < 1e-11

    disp = 1e-4
    def eval_j(c, ac):
        cderi = rsdf_builder.build_cderi(c, ac)[0][0,0]
        jaux = cp.einsum('rpq,qp->r', cderi, dm)
        return float(jaux.dot(jaux).get()) * .5

    _check_gradient(grad, cell, auxcell, eval_j)

    for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
        cell1, cell2 = _finite_diff_cells(cell, i, j, disp=disp)
        acell1, acell2 = _finite_diff_cells(auxcell, i, j, disp=disp)
        e1 = eval_j(cell1, acell1)
        e2 = eval_j(cell2, acell2)
        assert abs(sigma[i, j] - (e1-e2)/2/disp) < 5e-7

def test_ejk_strain_deriv_gamma_point_without_long_range():
    cell, auxcell = create_cell_auxcell()
    np.random.seed(8)
    nao = cell.nao
    nocc = 4
    mo_coeff = np.random.rand(nao, nao) - .5
    mo_occ = np.zeros(nao)
    mo_occ[:nocc] = 2
    dm = (mo_coeff*mo_occ).dot(mo_coeff.T)
    dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    omega = -0.3
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega).build()
    ek, sigma = rhf_stress._get_ejk_strain_deriv(
        opt, dm, hermi=1, j_factor=1, k_factor=1, omega=omega)
    assert abs(ek.sum(axis=0)).max() < 1e-11

    disp = 1e-4
    def eval_jk(c, ac):
        cderi = rsdf_builder.build_cderi(c, ac, omega=omega)[0][0,0]
        cderi = cderi.transpose(1,2,0)
        ref = .5 * cp.einsum('ijp,klp,ji,lk->', cderi, cderi, dm, dm, optimize=True)
        ref -= .25 * cp.einsum('ijp,klp,jk,li->', cderi, cderi, dm, dm, optimize=True)
        return float(ref.get())

    _check_gradient(ek, cell, auxcell, eval_jk)

    for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
        cell1, cell2 = _finite_diff_cells(cell, i, j, disp=disp)
        acell1, acell2 = _finite_diff_cells(auxcell, i, j, disp=disp)
        e1 = eval_jk(cell1, acell1)
        e2 = eval_jk(cell2, acell2)
        assert abs(sigma[i, j] - (e1-e2)/2/disp) < 5e-7


def test_metric_solver_with_linear_dependency():
    rng = np.random.default_rng(12)
    a = rng.standard_normal((3, 3)) + 1j*rng.standard_normal((3, 3))
    vectors = np.linalg.qr(a)[0]
    eigenvalues = np.array([2., 1e-12, -.5])
    j2c = vectors @ np.diag(eigenvalues) @ vectors.conj().T
    rhs = rng.standard_normal((3, 2)) + 1j*rng.standard_normal((3, 2))

    solve_j2c = rhf._gen_metric_solver(
        cp.asarray(j2c), 1e-10, dimension=2)
    result = solve_j2c(cp.asarray(rhs)).get()

    keep = np.array([0, 2])
    expected = vectors[:,keep] @ np.diag(1/eigenvalues[keep])
    expected = expected @ vectors[:,keep].conj().T @ rhs
    assert abs(result - expected).max() < 1e-12


def test_task_pool_batch_size_includes_bvk_cells():
    nksh_per_batch = np.array([12, 36, 24])
    batch_size = rhf._get_shl_pair_batch_size(
        nksh_per_batch, bvk_ncells=4)

    # 16383 // (36 * 4) = 113; nearest smaller power of two is 64.
    assert batch_size == 64
    assert batch_size * nksh_per_batch.max() * 4 <= int3c2e.POOL_SIZE

def test_ejk_strain_deriv_gamma_point_with_long_range():
    cell, auxcell = create_cell_auxcell()
    omega = -0.3

    np.random.seed(8)
    nao = cell.nao
    nocc = 4
    mo_coeff = np.random.rand(nao, nao) - .5
    mo_occ = np.zeros(nao)
    mo_occ[:nocc] = 2
    dm = (mo_coeff*mo_occ).dot(mo_coeff.T)
    dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega).build()
    hermi = 1
    j_factor = 1
    k_factor = 1
    ek, sigma = rhf_stress._get_ejk_strain_deriv(opt, dm, hermi, j_factor, k_factor)
    assert abs(ek.sum(axis=0)).max() < 1e-11

    disp = 1e-4
    def eval_jk(c, ac):
        cderi = rsdf_builder.build_cderi(c, ac)[0][0,0]
        cderi = cderi.transpose(1,2,0)
        ref = .5 * cp.einsum('ijp,klp,ji,lk->', cderi, cderi, dm, dm, optimize=True)
        ref -= .25 * cp.einsum('ijp,klp,jk,li->', cderi, cderi, dm, dm, optimize=True)
        return float(ref.get())

    _check_gradient(ek, cell, auxcell, eval_jk)

    for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
        cell1, cell2 = _finite_diff_cells(cell, i, j, disp=disp)
        acell1, acell2 = _finite_diff_cells(auxcell, i, j, disp=disp)
        e1 = eval_jk(cell1, acell1)
        e2 = eval_jk(cell2, acell2)
        assert abs(sigma[i, j] - (e1-e2)/2/disp) < 5e-7

def test_ej_strain_deriv_kpts_without_long_range():
    cell, auxcell = create_cell_auxcell()
    kmesh = [3,1,4]
    kpts = cell.make_kpts(kmesh)
    dm = cp.asarray(np.linalg.inv(cell.pbc_intor('int1e_ovlp', kpts=kpts))*.5)
    omega = -0.3
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega, kmesh).build()
    grad, sigma = krhf_stress._get_ejk_strain_deriv(
        opt, dm, hermi=1, kpts=kpts, k_factor=0, omega=omega)
    assert abs(grad.sum(axis=0)).max() < 1e-11

    disp = 1e-4
    dm = opt.cell.apply_C_mat_CT(dm)
    def eval_j(c, ac):
        opt = int3c2e.SRInt3c2eOpt(c, ac, omega, kmesh).build()
        jaux = opt.contract_dm(dm, kpts=c.make_kpts(kmesh))
        j2c = sr_int2c2e(ac, omega)
        return float(cp.linalg.solve(j2c, jaux).dot(jaux).get()) * .5

    _check_gradient(grad, cell, auxcell, eval_j)

    for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
        cell1, cell2 = _finite_diff_cells(cell, i, j, disp=disp)
        acell1, acell2 = _finite_diff_cells(auxcell, i, j, disp=disp)
        e1 = eval_j(cell1, acell1)
        e2 = eval_j(cell2, acell2)
        assert abs(sigma[i, j] - (e1-e2)/2/disp) < 5e-7

def test_ej_strain_deriv_kpts_with_long_range():
    cell, auxcell = create_cell_auxcell()

    kmesh = [3,1,4]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    dm = cp.asarray(np.linalg.inv(cell.pbc_intor('int1e_ovlp', kpts=kpts))*.5)
    omega = -0.3
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega, kmesh).build()
    grad, sigma = krhf_stress._get_ejk_strain_deriv(
        opt, dm, hermi=1, kpts=kpts, k_factor=0)
    assert abs(grad.sum(axis=0)).max() < 1e-11

    disp = 1e-4
    def eval_j(c, ac):
        kpts = c.make_kpts(kmesh)
        cderi = rsdf_builder.build_cderi(c, ac, kpts, kmesh, j_only=True)[0]
        jaux = 0
        for ki in range(nkpts):
            jaux += cp.einsum('pij,ji->p', cderi[ki,ki], dm[ki])
        ref = .5/nkpts**2 * jaux.dot(jaux).real.get()
        return ref

    _check_gradient(grad, cell, auxcell, eval_j)

    for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
        cell1, cell2 = _finite_diff_cells(cell, i, j, disp=disp)
        acell1, acell2 = _finite_diff_cells(auxcell, i, j, disp=disp)
        e1 = eval_j(cell1, acell1)
        e2 = eval_j(cell2, acell2)
        assert abs(sigma[i, j] - (e1-e2)/2/disp) < 5e-7

def test_ejk_strain_deriv_kpts_without_long_range():
    cell, auxcell = create_cell_auxcell()
    kmesh = [3,1,4]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    mo_coeff = np.linalg.eigh(cell.pbc_intor('int1e_ovlp', kpts=kpts))[1]
    nao = cell.nao
    nocc = 9
    mo_occ = np.zeros((nkpts, nao))
    mo_occ[:,:nocc] = 2
    dm = cp.einsum('kpi,ki,kqi->kpq', mo_coeff, mo_occ, mo_coeff.conj())
    omega = -0.3
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega, kmesh).build()
    j_factor = 1
    k_factor = 1

    ejk0, sigma0 = krhf_stress._get_ejk_strain_deriv(
        opt, dm, kpts, hermi=1, j_factor=j_factor, k_factor=k_factor, omega=omega)
    assert abs(ejk0.sum(axis=0)).max() < 2e-11

    dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    ejk, sigma = krhf_stress._get_ejk_strain_deriv(
        opt, dm, kpts, hermi=1, j_factor=j_factor, k_factor=k_factor, omega=omega)
    assert abs(ejk0 - ejk).max() < 1e-9
    assert abs(sigma0 - sigma).max() < 1e-9

    disp = 1e-4
    def eval_jk(c, ac):
        kpts = c.make_kpts(kmesh)
        cderi = rsdf_builder.build_cderi(c, ac, kpts=kpts, kmesh=kmesh, omega=omega)[0]
        jaux = 0
        for ki in range(nkpts):
            jaux += cp.einsum('pij,ji->p', cderi[ki,ki], dm[ki])
        ref = j_factor * .5/nkpts**2 * jaux.dot(jaux).real.get()
        ek = 0
        for ki in range(nkpts):
            for kj in range(nkpts):
                if (ki, kj) in cderi:
                    cderi_ij = cderi[ki,kj]
                else:
                    cderi_ij = cderi[kj,ki].transpose(0,2,1).conj()
                if (kj, ki) in cderi:
                    cderi_ji = cderi[kj,ki]
                else:
                    cderi_ji = cderi[ki,kj].transpose(0,2,1).conj()
                ek += cp.einsum('pij,jk,li,pkl->', cderi_ij, dm[kj],
                                dm[ki], cderi_ji, optimize=True)
        ek = float(ek.real.get())
        ref -= ek * .25 / nkpts**2 * k_factor
        return ref

    _check_gradient(ejk, cell, auxcell, eval_jk, tol=1e-6)

    for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
        cell1, cell2 = _finite_diff_cells(cell, i, j, disp=disp)
        acell1, acell2 = _finite_diff_cells(auxcell, i, j, disp=disp)
        e1 = eval_jk(cell1, acell1)
        e2 = eval_jk(cell2, acell2)
        assert abs(sigma[i, j] - (e1-e2)/2/disp) < 5e-7

def test_ejk_strain_deriv_kpts_with_long_range():
    cell, auxcell = create_cell_auxcell()
    kmesh = [3,1,2]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    mo_coeff = np.linalg.eigh(cell.pbc_intor('int1e_ovlp', kpts=kpts))[1]
    nao = cell.nao
    nocc = 9
    mo_occ = np.zeros((nkpts, nao))
    mo_occ[:,:nocc] = 2
    dm = cp.einsum('kpi,ki,kqi->kpq', mo_coeff, mo_occ, mo_coeff.conj())
    omega = -0.3
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega, kmesh).build()
    j_factor = 1
    k_factor = 1

    dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    ejk, sigma = krhf_stress._get_ejk_strain_deriv(
        opt, dm, kpts, hermi=1, j_factor=j_factor, k_factor=k_factor)
    assert abs(ejk.sum(axis=0)).max() < 1e-11

    disp = 1e-4
    def eval_jk(c, ac):
        kpts = c.make_kpts(kmesh)
        cderi = rsdf_builder.build_cderi(c, ac, kpts=kpts, kmesh=kmesh)[0]
        jaux = 0
        for ki in range(nkpts):
            jaux += cp.einsum('pij,ji->p', cderi[ki,ki], dm[ki])
        ref = j_factor * .5/nkpts**2 * jaux.dot(jaux).real.get()
        ek = 0
        for ki in range(nkpts):
            for kj in range(nkpts):
                if (ki, kj) in cderi:
                    cderi_ij = cderi[ki,kj]
                else:
                    cderi_ij = cderi[kj,ki].transpose(0,2,1).conj()
                if (kj, ki) in cderi:
                    cderi_ji = cderi[kj,ki]
                else:
                    cderi_ji = cderi[ki,kj].transpose(0,2,1).conj()
                ek += cp.einsum('pij,jk,li,pkl->', cderi_ij, dm[kj],
                                dm[ki], cderi_ji, optimize=True)
        ek = float(ek.real.get())
        ref -= ek * .25 / nkpts**2 * k_factor
        return ref

    _check_gradient(ejk, cell, auxcell, eval_jk, tol=1e-6)

    for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
        cell1, cell2 = _finite_diff_cells(cell, i, j, disp=disp)
        acell1, acell2 = _finite_diff_cells(auxcell, i, j, disp=disp)
        e1 = eval_jk(cell1, acell1)
        e2 = eval_jk(cell2, acell2)
        assert abs(sigma[i, j] - (e1-e2)/2/disp) < 5e-7

def test_ejk_strain_deriv_kpts_with_long_range1():
    cell = gto.Cell()
    cell.atom= [['H1', [0.0, 0.0, 0.0]], ['H2', [1.685,1.685,1.6]]]
    cell.a = '''
    0.00, 3.37, 3.37
    3.37, 0.00, 4.
    2.  , 3.37, 0.00'''
    cell.verbose = 0
    cell.basis = [[0, [3.3, 1]], [0, [1.1, 1]], [1, [0.8, 1]]]
    cell.unit = 'bohr'
    cell.build()
    auxcell = make_auxcell(cell)

    kmesh = [1,1,3]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)

    s = cell.pbc_intor('int1e_ovlp', kpts=kpts)
    mo_coeff = cp.array(np.linalg.eigh(s)[1])
    mo_occ = cp.zeros((nkpts, cell.nao))
    mo_occ[:,:3] = 2
    omega = -0.3

    dm = cp.einsum('kpi,ki,kqi->kpq', mo_coeff, mo_occ, mo_coeff.conj())
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega, kmesh).build()
    j_factor = 1
    k_factor = 1

    dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    ejk, sigma = krhf_stress._get_ejk_strain_deriv(
        opt, dm, kpts, hermi=1, j_factor=j_factor, k_factor=k_factor)
    assert abs(ejk.sum(axis=0)).max() < 1e-11

    disp = 1e-4
    def eval_jk(c, ac):
        kpts = c.make_kpts(kmesh)
        cderi = rsdf_builder.build_cderi(c, ac, kpts=kpts, kmesh=kmesh)[0]
        jaux = 0
        for ki in range(nkpts):
            jaux += cp.einsum('pij,ji->p', cderi[ki,ki], dm[ki])
        ref = j_factor * .5/nkpts**2 * jaux.dot(jaux).real.get()
        ek = 0
        for ki in range(nkpts):
            for kj in range(nkpts):
                if (ki, kj) in cderi:
                    cderi_ij = cderi[ki,kj]
                else:
                    cderi_ij = cderi[kj,ki].transpose(0,2,1).conj()
                if (kj, ki) in cderi:
                    cderi_ji = cderi[kj,ki]
                else:
                    cderi_ji = cderi[ki,kj].transpose(0,2,1).conj()
                ek += cp.einsum('pij,jk,li,pkl->', cderi_ij, dm[kj], dm[ki], cderi_ji)
        ek = float(ek.real.get())
        ref -= ek * .25 / nkpts**2 * k_factor
        return ref

    _check_gradient(ejk, cell, auxcell, eval_jk, tol=1e-6)

    for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
        cell1, cell2 = _finite_diff_cells(cell, i, j, disp=disp)
        acell1, acell2 = _finite_diff_cells(auxcell, i, j, disp=disp)
        e1 = eval_jk(cell1, acell1)
        e2 = eval_jk(cell2, acell2)
        assert abs(sigma[i, j] - (e1-e2)/2/disp) < 5e-7

def test_uhf_ejk_strain_deriv_gamma_point_without_long_range():
    cell, auxcell = create_cell_auxcell()
    np.random.seed(8)
    nao = cell.nao
    nocc = 4
    mo_coeff = np.random.rand(2, nao, nao) - .5
    mo_occ = np.zeros((2, nao))
    mo_occ[:,:nocc] = 1
    dm = contract('spi,sqi->spq', mo_coeff*mo_occ[:,None], mo_coeff)
    omega = -0.3
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega).build()
    ek, sigma = uhf_stress._get_ejk_strain_deriv(
        opt, dm, hermi=1, j_factor=1, k_factor=1, omega=omega)
    assert abs(ek.sum(axis=0)).max() < 1e-11

    dm_sf = dm[0] + dm[1]
    disp = 1e-4
    def eval_jk(c, ac):
        cderi = rsdf_builder.build_cderi(c, ac, omega=omega)[0][0,0]
        cderi = cderi.transpose(1,2,0)
        ref = .5 * cp.einsum('ijp,klp,ji,lk->', cderi, cderi, dm_sf, dm_sf, optimize=True)
        ref -= .5 * cp.einsum('ijp,klp,sjk,sli->', cderi, cderi, dm, dm, optimize=True)
        return float(ref.get())

    _check_gradient(ek, cell, auxcell, eval_jk)

    for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
        cell1, cell2 = _finite_diff_cells(cell, i, j, disp=disp)
        acell1, acell2 = _finite_diff_cells(auxcell, i, j, disp=disp)
        e1 = eval_jk(cell1, acell1)
        e2 = eval_jk(cell2, acell2)
        assert abs(sigma[i, j] - (e1-e2)/2/disp) < 5e-7

def test_uhf_ejk_strain_deriv_gamma_point_with_long_range():
    cell, auxcell = create_cell_auxcell()
    np.random.seed(8)
    nao = cell.nao
    nocc = 4
    mo_coeff = np.random.rand(2, nao, nao) - .5
    mo_coeff[1]=mo_coeff[0]
    mo_occ = np.zeros((2, nao))
    mo_occ[:,:nocc] = 1
    dm = contract('spi,sqi->spq', mo_coeff*mo_occ[:,None], mo_coeff)
    omega = -0.5
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega).build()
    hermi = 1
    j_factor = 1
    k_factor = 1
    ek, sigma = uhf_stress._get_ejk_strain_deriv(
        opt, dm, hermi=hermi, j_factor=j_factor, k_factor=k_factor)
    assert abs(ek.sum(axis=0)).max() < 1e-11

    dm_sf = dm[0] + dm[1]
    disp = 1e-4
    def eval_jk(c, ac):
        cderi = rsdf_builder.build_cderi(c, ac)[0][0,0]
        cderi = cderi.transpose(1,2,0)
        ref = .5 * cp.einsum('ijp,klp,ji,lk->', cderi, cderi, dm_sf, dm_sf, optimize=True)
        ref -= .5 * cp.einsum('ijp,klp,sjk,sli->', cderi, cderi, dm, dm, optimize=True)
        return float(ref.get())

    _check_gradient(ek, cell, auxcell, eval_jk, tol=3e-6)

    for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
        cell1, cell2 = _finite_diff_cells(cell, i, j, disp=disp)
        acell1, acell2 = _finite_diff_cells(auxcell, i, j, disp=disp)
        e1 = eval_jk(cell1, acell1)
        e2 = eval_jk(cell2, acell2)
        assert abs(sigma[i, j] - (e1-e2)/2/disp) < 5e-7

def test_uhf_ejk_strain_deriv_kpts_without_long_range():
    cell, auxcell = create_cell_auxcell()
    nao = cell.nao
    kmesh = [3,1,2]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    mo_coeff = cp.empty((2,nkpts,nao,nao), dtype=np.complex128)
    s = np.array(cell.pbc_intor('int1e_ovlp', kpts=kpts))
    mo_coeff[0] = cp.asarray(np.linalg.eigh(s)[1])
    mo_coeff[1] = mo_coeff[0,:,:,::-1]
    nocc = 9
    mo_occ = cp.zeros((2, nkpts, nao))
    mo_occ[:,:,:nocc] = 1
    dm = cp.einsum('skpi,ski,skqi->skpq', mo_coeff, mo_occ, mo_coeff.conj())
    omega = -0.3
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega, kmesh).build()
    j_factor = 1
    k_factor = 1
    ejk0, sigma0 = kuhf_stress._get_ejk_strain_deriv(
        opt, dm, kpts, hermi=1, j_factor=j_factor, k_factor=k_factor, omega=omega)
    assert abs(ejk0.sum(axis=0)).max() < 2e-11

    dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    ejk, sigma = kuhf_stress._get_ejk_strain_deriv(
        opt, dm, kpts, hermi=1, j_factor=j_factor, k_factor=k_factor, omega=omega)
    assert abs(ejk0 - ejk).max() < 1e-9
    assert abs(sigma0 - sigma).max() < 1e-9

    dm_sf = dm[0] + dm[1]
    disp = 1e-4
    def eval_jk(c, ac):
        kpts = c.make_kpts(kmesh)
        cderi = rsdf_builder.build_cderi(c, ac, kpts=kpts, kmesh=kmesh, omega=omega)[0]
        jaux = 0
        for ki in range(nkpts):
            jaux += cp.einsum('pij,ji->p', cderi[ki,ki], dm_sf[ki])
        ref = j_factor * .5/nkpts**2 * jaux.dot(jaux).real.get()
        ek = 0
        for ki in range(nkpts):
            for kj in range(nkpts):
                if (ki, kj) in cderi:
                    cderi_ij = cderi[ki,kj]
                else:
                    cderi_ij = cderi[kj,ki].transpose(0,2,1).conj()
                if (kj, ki) in cderi:
                    cderi_ji = cderi[kj,ki]
                else:
                    cderi_ji = cderi[ki,kj].transpose(0,2,1).conj()
                ek += cp.einsum('pij,sjk,sli,pkl->', cderi_ij, dm[:,kj],
                                dm[:,ki], cderi_ji, optimize=True)
        ek = float(ek.real.get())
        ref -= ek * .5 / nkpts**2 * k_factor
        return ref

    _check_gradient(ejk, cell, auxcell, eval_jk, tol=3e-6)

    for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
        cell1, cell2 = _finite_diff_cells(cell, i, j, disp=disp)
        acell1, acell2 = _finite_diff_cells(auxcell, i, j, disp=disp)
        e1 = eval_jk(cell1, acell1)
        e2 = eval_jk(cell2, acell2)
        assert abs(sigma[i, j] - (e1-e2)/2/disp) < 5e-7

def test_uhf_ejk_strain_deriv_kpts_with_long_range():
    cell, auxcell = create_cell_auxcell()
    nao = cell.nao
    kmesh = [3,1,2]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    mo_coeff = cp.empty((2,nkpts,nao,nao), dtype=np.complex128)
    s = np.array(cell.pbc_intor('int1e_ovlp', kpts=kpts))
    mo_coeff[0] = cp.asarray(np.linalg.eigh(s)[1])
    mo_coeff[1] = mo_coeff[0,:,:,::-1]
    nocc = 7
    mo_occ = cp.zeros((2, nkpts, nao))
    mo_occ[:,:,:nocc] = 1
    dm = cp.einsum('skpi,ski,skqi->skpq', mo_coeff, mo_occ, mo_coeff.conj())
    omega = -0.3
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega, kmesh).build()
    j_factor = 1
    k_factor = 1
    dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    ejk, sigma = kuhf_stress._get_ejk_strain_deriv(
        opt, dm, kpts, hermi=1, j_factor=j_factor, k_factor=k_factor)
    assert abs(ejk.sum(axis=0)).max() < 3e-9

    dm_sf = dm[0] + dm[1]
    disp = 1e-4
    def eval_jk(c, ac):
        kpts = c.make_kpts(kmesh)
        cderi = rsdf_builder.build_cderi(c, ac, kpts=kpts, kmesh=kmesh)[0]
        jaux = 0
        for ki in range(nkpts):
            jaux += cp.einsum('pij,ji->p', cderi[ki,ki], dm_sf[ki])
        ref = j_factor * .5/nkpts**2 * jaux.dot(jaux).real.get()
        ek = 0
        for ki in range(nkpts):
            for kj in range(nkpts):
                if (ki, kj) in cderi:
                    cderi_ij = cderi[ki,kj]
                else:
                    cderi_ij = cderi[kj,ki].transpose(0,2,1).conj()
                if (kj, ki) in cderi:
                    cderi_ji = cderi[kj,ki]
                else:
                    cderi_ji = cderi[ki,kj].transpose(0,2,1).conj()
                ek += cp.einsum('pij,sjk,sli,pkl->', cderi_ij, dm[:,kj],
                                dm[:,ki], cderi_ji, optimize=True)
        ek = float(ek.real.get())
        ref -= ek * .5 / nkpts**2 * k_factor
        return ref

    _check_gradient(ejk, cell, auxcell, eval_jk, tol=1e-6)

    for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
        cell1, cell2 = _finite_diff_cells(cell, i, j, disp=disp)
        acell1, acell2 = _finite_diff_cells(auxcell, i, j, disp=disp)
        e1 = eval_jk(cell1, acell1)
        e2 = eval_jk(cell2, acell2)
        assert abs(sigma[i, j] - (e1-e2)/2/disp) < 5e-7

def test_uhf_ejk_strain_deriv_kpts_with_long_range1():
    cell = gto.Cell()
    cell.atom= [['H1', [0.0, 0.0, 0.0]], ['H2', [1.685,1.685,1.6]]]
    cell.a = '''
    0.00, 3.37, 3.37
    3.37, 0.00, 4.
    2.  , 3.37, 0.00'''
    cell.verbose = 0
    cell.basis = [[0, [3.3, 1]], [0, [1.1, 1]], [1, [0.8, 1]]]
    cell.unit = 'bohr'
    cell.build()
    auxcell = make_auxcell(cell)

    kmesh = [1,1,3]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)

    s = cell.pbc_intor('int1e_ovlp', kpts=kpts)
    mo_coeff = cp.array([np.linalg.eigh(s)[1]]*2)
    mo_occ = cp.zeros((2, nkpts, cell.nao))
    mo_occ[:,:,:3] = 2
    omega = -0.3

    dm = cp.einsum('skpi,ski,skqi->skpq', mo_coeff, mo_occ, mo_coeff.conj())
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega, kmesh).build()
    j_factor = .5
    k_factor = 1
    dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    ejk, sigma = kuhf_stress._get_ejk_strain_deriv(
        opt, dm, kpts, hermi=1, j_factor=j_factor, k_factor=k_factor)
    assert abs(ejk.sum(axis=0)).max() < 1e-11

    dm_sf = dm[0] + dm[1]
    disp = 1e-4
    def eval_jk(c, ac):
        kpts = c.make_kpts(kmesh)
        cderi = rsdf_builder.build_cderi(c, ac, kpts=kpts, kmesh=kmesh)[0]
        jaux = 0
        for ki in range(nkpts):
            jaux += cp.einsum('pij,ji->p', cderi[ki,ki], dm_sf[ki])
        ref = j_factor * .5/nkpts**2 * jaux.dot(jaux).real.get()

        ek = 0
        for ki in range(nkpts):
            for kj in range(nkpts):
                if (ki, kj) in cderi:
                    cderi_ij = cderi[ki,kj]
                else:
                    cderi_ij = cderi[kj,ki].transpose(0,2,1).conj()
                if (kj, ki) in cderi:
                    cderi_ji = cderi[kj,ki]
                else:
                    cderi_ji = cderi[ki,kj].transpose(0,2,1).conj()
                ek += cp.einsum('pij,sjk,sli,pkl->', cderi_ij, dm[:,kj], dm[:,ki], cderi_ji)
        ek = float(ek.real.get())
        ref -= ek * .5 / nkpts**2 * k_factor
        return ref

    _check_gradient(ejk, cell, auxcell, eval_jk, tol=1e-6)

    for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
        cell1, cell2 = _finite_diff_cells(cell, i, j, disp=disp)
        acell1, acell2 = _finite_diff_cells(auxcell, i, j, disp=disp)
        e1 = eval_jk(cell1, acell1)
        e2 = eval_jk(cell2, acell2)
        assert abs(sigma[i, j] - (e1-e2)/2/disp) < 5e-7
