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
import cupy as cp
import pyscf
import pytest
from pyscf import gto
from gpu4pyscf.solvent import pcm
from gpu4pyscf import scf, dft
from packaging import version
from gpu4pyscf.solvent.hessian.pcm import analytical_grad_vmat, analytical_hess_nuc, analytical_hess_solver, analytical_hess_qv
from gpu4pyscf.lib.cupy_helper import contract

pyscf_25 = version.parse(pyscf.__version__) <= version.parse('2.5.0')

def setUpModule():
    global mol, epsilon, lebedev_order, eps, xc, tol
    mol = gto.Mole()
    mol.atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
    '''
    mol.basis = 'def2-tzvpp'
    mol.output = '/dev/null'
    mol.build(verbose=0)
    epsilon = 78.3553
    lebedev_order = 29
    eps = 1e-3
    xc = 'B3LYP'
    tol = 1e-3

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _make_mf(method='C-PCM', restricted=True, density_fit=True):
    if restricted:
        mf = dft.rks.RKS(mol, xc=xc)
    else:
        mf = dft.uks.UKS(mol, xc=xc)

    if density_fit:
        mf = mf.density_fit()
    mf = mf.PCM()
    mf.with_solvent.method = method
    mf.with_solvent.eps = epsilon
    mf.with_solvent.lebedev_order = lebedev_order
    mf.conv_tol = 1e-12
    mf.conv_tol_cpscf = 1e-7
    mf.grids.atom_grid = (99,590)
    mf.verbose = 0
    mf.kernel()
    return mf

def _check_hessian(mf, h, ix=0, iy=0):
    pmol = mf.mol.copy()
    pmol.build()

    g = mf.nuc_grad_method()
    g.auxbasis_response = True
    g.kernel()
    g_scanner = g.as_scanner()

    coords = pmol.atom_coords()
    v = np.zeros_like(coords)
    v[ix,iy] = eps
    pmol.set_geom_(coords + v, unit='Bohr')
    pmol.build()
    _, g0 = g_scanner(pmol)

    pmol.set_geom_(coords - v, unit='Bohr')
    pmol.build()
    _, g1 = g_scanner(pmol)

    h_fd = (g0 - g1)/2.0/eps

    print(f'Norm of H({ix},{iy}) diff, {np.linalg.norm(h[ix,:,iy,:] - h_fd)}')
    assert(np.linalg.norm(h[ix,:,iy,:] - h_fd) < tol)

def _fd_grad_vmat(pcmobj, dm, mo_coeff, mo_occ, atmlst=None):
    '''
    dv_solv / da
    slow version with finite difference
    '''
    pmol = pcmobj.mol.copy()
    mol = pmol.copy()
    if atmlst is None:
        atmlst = range(mol.natm)
    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]
    coords = mol.atom_coords(unit='Bohr')
    def pcm_vmat_scanner(mol):
        pcmobj.reset(mol)
        e, v = pcmobj._get_vind(dm)
        return v

    mol.verbose = 0
    vmat = cp.empty([len(atmlst), 3, nao, nocc])
    eps = 1e-5
    for i0, ia in enumerate(atmlst):
        for ix in range(3):
            dv = np.zeros_like(coords)
            dv[ia,ix] = eps
            mol.set_geom_(coords + dv, unit='Bohr')
            vmat0 = pcm_vmat_scanner(mol)

            mol.set_geom_(coords - dv, unit='Bohr')
            vmat1 = pcm_vmat_scanner(mol)

            grad_vmat = (vmat0 - vmat1)/2.0/eps
            grad_vmat = contract("ij,jq->iq", grad_vmat, mocc)
            grad_vmat = contract("iq,ip->pq", grad_vmat, mo_coeff)
            vmat[i0,ix] = grad_vmat
    pcmobj.reset(pmol)
    return vmat

def _fd_hess_contribution(pcmobj, dm, gradient_function):
    pmol = pcmobj.mol.copy()
    mol = pmol.copy()
    coords = mol.atom_coords(unit='Bohr')

    def pcm_grad_scanner(mol):
        pcmobj.reset(mol)
        e, v = pcmobj._get_vind(dm)
        pcm_grad = gradient_function(pcmobj, dm)
        # pcm_grad = grad_nuc(pcmobj, dm)
        # pcm_grad+= grad_solver(pcmobj, dm)
        # pcm_grad+= grad_qv(pcmobj, dm)
        return pcm_grad

    mol.verbose = 0
    de = np.zeros([mol.natm, mol.natm, 3, 3])
    eps = 1e-5
    for ia in range(mol.natm):
        for ix in range(3):
            dv = np.zeros_like(coords)
            dv[ia,ix] = eps
            mol.set_geom_(coords + dv, unit='Bohr')
            g0 = pcm_grad_scanner(mol)

            mol.set_geom_(coords - dv, unit='Bohr')
            g1 = pcm_grad_scanner(mol)

            de[ia,:,ix,:] = (g0 - g1)/2.0/eps
    pcmobj.reset(pmol)
    return de

@unittest.skipIf(pcm.libsolvent is None, "solvent extension not compiled")
class KnownValues(unittest.TestCase):
    def test_df_hess_cpcm(self):
        print('testing C-PCM Hessian with DF-RKS')
        mf = _make_mf(method='C-PCM')
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_hessian(mf, h, ix=0, iy=0)
        _check_hessian(mf, h, ix=0, iy=1)

    def test_hess_cpcm(self):
        print('testing C-PCM Hessian with RKS')
        mf = _make_mf(method='C-PCM', density_fit=False)
        hobj = mf.Hessian()
        h = hobj.kernel()
        _check_hessian(mf, h, ix=0, iy=0)
        _check_hessian(mf, h, ix=0, iy=1)

    def test_df_hess_iefpcm(self):
        print("testing IEF-PCM hessian with DF-RKS")
        mf = _make_mf(method='IEF-PCM')
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_hessian(mf, h, ix=0, iy=0)
        _check_hessian(mf, h, ix=0, iy=1)

    def test_hess_iefpcm(self):
        print("testing IEF-PCM hessian with RKS")
        mf = _make_mf(method='IEF-PCM', density_fit=False)
        hobj = mf.Hessian()
        h = hobj.kernel()
        _check_hessian(mf, h, ix=0, iy=0)
        _check_hessian(mf, h, ix=0, iy=1)

    def test_df_uks_hess_iefpcm(self):
        print("testing IEF-PCM hessian with DF-UKS")
        mf = _make_mf(method='IEF-PCM', restricted=False, density_fit=True)
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_hessian(mf, h, ix=0, iy=0)
        _check_hessian(mf, h, ix=0, iy=1)

    def test_uks_hess_iefpcm(self):
        print("testing IEF-PCM hessian with UHF")
        mf = _make_mf(method='IEF-PCM', restricted=False, density_fit=False)
        hobj = mf.Hessian()
        h = hobj.kernel()
        _check_hessian(mf, h, ix=0, iy=0)
        _check_hessian(mf, h, ix=0, iy=1)

    def test_grad_vmat_cpcm(self):
        print("testing C-PCM dV_solv/dx")
        mf = _make_mf(method='C-PCM')
        hobj = mf.Hessian()

        dm = mf.make_rdm1()
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ

        test_grad_vmat = analytical_grad_vmat(hobj.base.with_solvent, dm, mo_coeff, mo_occ)
        ref_grad_vmat = _fd_grad_vmat(hobj.base.with_solvent, dm, mo_coeff, mo_occ)

        cp.testing.assert_allclose(ref_grad_vmat, test_grad_vmat, atol = 1e-10)

    def test_grad_vmat_iefpcm(self):
        print("testing IEF-PCM dV_solv/dx")
        mf = _make_mf(method='IEF-PCM')
        hobj = mf.Hessian()

        dm = mf.make_rdm1()
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ

        test_grad_vmat = analytical_grad_vmat(hobj.base.with_solvent, dm, mo_coeff, mo_occ)
        ref_grad_vmat = _fd_grad_vmat(hobj.base.with_solvent, dm, mo_coeff, mo_occ)

        cp.testing.assert_allclose(ref_grad_vmat, test_grad_vmat, atol = 1e-10)

    def test_grad_vmat_ssvpe(self):
        print("testing SS(V)PE dV_solv/dx")
        mf = _make_mf(method='SS(V)PE')
        hobj = mf.Hessian()

        dm = mf.make_rdm1()
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ

        test_grad_vmat = analytical_grad_vmat(hobj.base.with_solvent, dm, mo_coeff, mo_occ)
        ref_grad_vmat = _fd_grad_vmat(hobj.base.with_solvent, dm, mo_coeff, mo_occ)

        cp.testing.assert_allclose(ref_grad_vmat, test_grad_vmat, atol = 1e-10)

    def test_hess_nuc_iefpcm(self):
        print("testing IEF-PCM d2E_nuc/dx2")
        mf = _make_mf(method='IEF-PCM')
        hobj = mf.Hessian()
        dm = mf.make_rdm1()

        test_grad_vmat = analytical_hess_nuc(hobj.base.with_solvent, dm)
        from gpu4pyscf.solvent.grad.pcm import grad_nuc
        ref_grad_vmat = _fd_hess_contribution(hobj.base.with_solvent, dm, grad_nuc)

        cp.testing.assert_allclose(ref_grad_vmat, test_grad_vmat, atol = 1e-10)

    def test_hess_qv_iefpcm(self):
        print("testing IEF-PCM d2E_elec/dx2")
        mf = _make_mf(method='IEF-PCM')
        hobj = mf.Hessian()
        dm = mf.make_rdm1()

        test_grad_vmat = analytical_hess_qv(hobj.base.with_solvent, dm)
        from gpu4pyscf.solvent.grad.pcm import grad_qv
        ref_grad_vmat = _fd_hess_contribution(hobj.base.with_solvent, dm, grad_qv)

        cp.testing.assert_allclose(ref_grad_vmat, test_grad_vmat, atol = 1e-10)

    def test_hess_solver_cpcm(self):
        print("testing C-PCM d2E_KR/dx2")
        mf = _make_mf(method='C-PCM')
        hobj = mf.Hessian()
        dm = mf.make_rdm1()

        test_grad_vmat = analytical_hess_solver(hobj.base.with_solvent, dm)
        from gpu4pyscf.solvent.grad.pcm import grad_solver
        ref_grad_vmat = _fd_hess_contribution(hobj.base.with_solvent, dm, grad_solver)

        cp.testing.assert_allclose(ref_grad_vmat, test_grad_vmat, atol = 1e-10)

    def test_hess_solver_iefpcm(self):
        print("testing IEF-PCM d2E_KR/dx2")
        mf = _make_mf(method='IEF-PCM')
        hobj = mf.Hessian()
        dm = mf.make_rdm1()

        test_grad_vmat = analytical_hess_solver(hobj.base.with_solvent, dm)
        from gpu4pyscf.solvent.grad.pcm import grad_solver
        ref_grad_vmat = _fd_hess_contribution(hobj.base.with_solvent, dm, grad_solver)

        cp.testing.assert_allclose(ref_grad_vmat, test_grad_vmat, atol = 1e-10)

    def test_hess_solver_ssvpe(self):
        print("testing SS(V)PE d2E_KR/dx2")
        mf = _make_mf(method='SS(V)PE')
        hobj = mf.Hessian()
        dm = mf.make_rdm1()

        test_grad_vmat = analytical_hess_solver(hobj.base.with_solvent, dm)
        from gpu4pyscf.solvent.grad.pcm import grad_solver
        ref_grad_vmat = _fd_hess_contribution(hobj.base.with_solvent, dm, grad_solver)

        cp.testing.assert_allclose(ref_grad_vmat, test_grad_vmat, atol = 1e-10)

    @pytest.mark.skipif(pyscf_25, reason='requires pyscf 2.6 or higher')
    def test_to_gpu(self):
        import pyscf
        mol = gto.Mole()
        mol.atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
    '''
        mol.basis = 'sto-3g'
        mol.output = '/dev/null'
        mol.build(verbose=0)
        mf = pyscf.dft.RKS(mol, xc='b3lyp').PCM()
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-7
        mf.grids.atom_grid = (50,194)
        mf.kernel()
        hessobj = mf.Hessian()
        hess_cpu = hessobj.kernel()
        hessobj = hessobj.to_gpu()
        hess_gpu = hessobj.kernel()
        assert np.linalg.norm(hess_cpu - hess_gpu) < 1e-5

        mf = pyscf.dft.RKS(mol, xc='b3lyp').density_fit().PCM()
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-7
        mf.grids.atom_grid = (50,194)
        mf.kernel()
        hessobj = mf.Hessian()
        hess_cpu = hessobj.kernel()
        hessobj = hessobj.to_gpu()
        hess_gpu = hessobj.kernel()
        assert np.linalg.norm(hess_cpu - hess_gpu) < 1e-5

    @pytest.mark.skipif(pyscf_25, reason='requires pyscf 2.6 or higher')
    def test_to_cpu(self):
        mol = gto.Mole()
        mol.atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
    '''
        mol.basis = 'sto-3g'
        mol.output = '/dev/null'
        mol.build(verbose=0)

        mf = dft.RKS(mol, xc='b3lyp').PCM()
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-7
        mf.grids.atom_grid = (50,194)
        mf.kernel()
        hessobj = mf.Hessian()
        hess_gpu = hessobj.kernel()
        hessobj = hessobj.to_cpu()
        hess_cpu = hessobj.kernel()
        assert np.linalg.norm(hess_cpu - hess_gpu) < 1e-5

        mf = dft.RKS(mol, xc='b3lyp').density_fit().PCM()
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-7
        mf.grids.atom_grid = (50,194)
        mf.kernel()
        hessobj = mf.Hessian()
        hess_gpu = hessobj.kernel()
        hessobj = hessobj.to_cpu()
        hess_cpu = hessobj.kernel()
        assert np.linalg.norm(hess_cpu - hess_gpu) < 1e-5

if __name__ == "__main__":
    print("Full Tests for Hessian of PCMs")
    unittest.main()
