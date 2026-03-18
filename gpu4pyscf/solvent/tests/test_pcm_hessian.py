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
from gpu4pyscf.solvent.hessian.pcm import analytical_grad_vmat, analytical_hess_nuc, analytical_hess_solver, analytical_hess_qv
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.lib.multi_gpu import num_devices
from pyscf.hessian import thermo

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
        #_check_hessian(mf, h, ix=0, iy=0)
        _check_hessian(mf, h, ix=0, iy=1)

    @pytest.mark.slow
    def test_hess_cpcm(self):
        print('testing C-PCM Hessian with RKS')
        mf = _make_mf(method='C-PCM', density_fit=False)
        hobj = mf.Hessian()
        h = hobj.kernel()
        _check_hessian(mf, h, ix=0, iy=0)
        #_check_hessian(mf, h, ix=0, iy=1)

    @unittest.skipIf(num_devices > 1, '')
    def test_df_hess_iefpcm(self):
        print("testing IEF-PCM hessian with DF-RKS")
        mf = _make_mf(method='IEF-PCM')
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_hessian(mf, h, ix=0, iy=0)
        #_check_hessian(mf, h, ix=0, iy=1)

    @pytest.mark.slow
    def test_hess_iefpcm(self):
        print("testing IEF-PCM hessian with RKS")
        mf = _make_mf(method='IEF-PCM', density_fit=False)
        hobj = mf.Hessian()
        h = hobj.kernel()
        #_check_hessian(mf, h, ix=0, iy=0)
        _check_hessian(mf, h, ix=0, iy=1)

    @unittest.skipIf(num_devices > 1, '')
    def test_df_uks_hess_iefpcm(self):
        print("testing IEF-PCM hessian with DF-UKS")
        mf = _make_mf(method='IEF-PCM', restricted=False, density_fit=True)
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        #_check_hessian(mf, h, ix=0, iy=0)
        _check_hessian(mf, h, ix=0, iy=1)

    @pytest.mark.slow
    def test_uks_hess_iefpcm(self):
        print("testing IEF-PCM hessian with UHF")
        mf = _make_mf(method='IEF-PCM', restricted=False, density_fit=False)
        hobj = mf.Hessian()
        h = hobj.kernel()
        _check_hessian(mf, h, ix=0, iy=0)
        #_check_hessian(mf, h, ix=0, iy=1)

    def test_grad_vmat_cpcm(self):
        print("testing C-PCM dV_solv/dx")
        mf = _make_mf(method='C-PCM')
        hobj = mf.Hessian()

        dm = mf.make_rdm1()
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ

        test_grad_vmat = analytical_grad_vmat(hobj.base.with_solvent, dm, mo_coeff, mo_occ)
        ref_grad_vmat = _fd_grad_vmat(hobj.base.with_solvent, dm, mo_coeff, mo_occ)

        assert abs(ref_grad_vmat - test_grad_vmat).max() < 1e-9

    def test_grad_vmat_iefpcm(self):
        print("testing IEF-PCM dV_solv/dx")
        mf = _make_mf(method='IEF-PCM')
        hobj = mf.Hessian()

        dm = mf.make_rdm1()
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ

        test_grad_vmat = analytical_grad_vmat(hobj.base.with_solvent, dm, mo_coeff, mo_occ)
        ref_grad_vmat = _fd_grad_vmat(hobj.base.with_solvent, dm, mo_coeff, mo_occ)

        assert abs(ref_grad_vmat - test_grad_vmat).max() < 1e-9

    @unittest.skipIf(num_devices > 1, '')
    def test_grad_vmat_ssvpe(self):
        print("testing SS(V)PE dV_solv/dx")
        mf = _make_mf(method='SS(V)PE')
        hobj = mf.Hessian()

        dm = mf.make_rdm1()
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ

        test_grad_vmat = analytical_grad_vmat(hobj.base.with_solvent, dm, mo_coeff, mo_occ)
        ref_grad_vmat = _fd_grad_vmat(hobj.base.with_solvent, dm, mo_coeff, mo_occ)

        assert abs(ref_grad_vmat - test_grad_vmat).max() < 1e-9

    def test_hess_nuc_iefpcm(self):
        print("testing IEF-PCM d2E_nuc/dx2")
        mf = _make_mf(method='IEF-PCM')
        hobj = mf.Hessian()
        dm = mf.make_rdm1()

        test_grad_vmat = analytical_hess_nuc(hobj.base.with_solvent, dm)
        from gpu4pyscf.solvent.grad.pcm import grad_nuc
        ref_grad_vmat = _fd_hess_contribution(hobj.base.with_solvent, dm, grad_nuc)

        assert abs(ref_grad_vmat - test_grad_vmat).max() < 1e-9

    def test_hess_qv_iefpcm(self):
        print("testing IEF-PCM d2E_elec/dx2")
        mf = _make_mf(method='IEF-PCM')
        hobj = mf.Hessian()
        dm = mf.make_rdm1()

        test_grad_vmat = analytical_hess_qv(hobj.base.with_solvent, dm)
        from gpu4pyscf.solvent.grad.pcm import grad_qv
        ref_grad_vmat = _fd_hess_contribution(hobj.base.with_solvent, dm, grad_qv)

        assert abs(ref_grad_vmat - test_grad_vmat).max() < 1e-9

    def test_hess_solver_cpcm(self):
        print("testing C-PCM d2E_KR/dx2")
        mf = _make_mf(method='C-PCM')
        hobj = mf.Hessian()
        dm = mf.make_rdm1()

        test_grad_vmat = analytical_hess_solver(hobj.base.with_solvent, dm)
        from gpu4pyscf.solvent.grad.pcm import grad_solver
        ref_grad_vmat = _fd_hess_contribution(hobj.base.with_solvent, dm, grad_solver)

        assert abs(ref_grad_vmat - test_grad_vmat).max() < 1e-9

    @unittest.skipIf(num_devices > 1, '')
    def test_hess_solver_iefpcm(self):
        print("testing IEF-PCM d2E_KR/dx2")
        mf = _make_mf(method='IEF-PCM')
        hobj = mf.Hessian()
        dm = mf.make_rdm1()

        test_grad_vmat = analytical_hess_solver(hobj.base.with_solvent, dm)
        from gpu4pyscf.solvent.grad.pcm import grad_solver
        ref_grad_vmat = _fd_hess_contribution(hobj.base.with_solvent, dm, grad_solver)

        assert abs(ref_grad_vmat - test_grad_vmat).max() < 1e-9

    @unittest.skipIf(num_devices > 1, '')
    def test_hess_solver_ssvpe(self):
        print("testing SS(V)PE d2E_KR/dx2")
        mf = _make_mf(method='SS(V)PE')
        hobj = mf.Hessian()
        dm = mf.make_rdm1()

        test_grad_vmat = analytical_hess_solver(hobj.base.with_solvent, dm)
        from gpu4pyscf.solvent.grad.pcm import grad_solver
        ref_grad_vmat = _fd_hess_contribution(hobj.base.with_solvent, dm, grad_solver)

        assert abs(ref_grad_vmat - test_grad_vmat).max() < 1e-9

    @unittest.skipIf(num_devices > 1, '')
    def test_hess_atom_with_zero_grid(self):
        mol = pyscf.M( # neopentane
            atom = """
                C      1.042440    0.085610   -0.011740
                C      2.570330    0.085610   -0.011740
                C      3.079630   -0.875920   -1.084370
                C      3.079630    1.495300   -0.308120
                C      3.079630   -0.362560    1.357290
                H      0.649570    0.403960   -0.984220
                H      0.649560    0.768620    0.750200
                H      0.649560   -0.915760    0.198800
                H      2.728230   -0.577130   -2.078680
                H      2.728220   -1.896850   -0.895670
                H      4.175330   -0.895490   -1.106200
                H      2.728220    1.842340   -1.286640
                H      4.175330    1.523990   -0.314160
                H      2.728220    2.207010    0.447780
                H      2.728210   -1.373050    1.595690
                H      2.728210    0.311330    2.147090
                H      4.175320   -0.371680    1.385160
            """,
            basis = "sto-3g",
            verbose = 4,
            output = '/dev/null',
        )

        mf = scf.hf.RHF(mol)
        mf = mf.density_fit("def2-universal-jkfit")
        mf = mf.PCM()
        mf.with_solvent.method = "C-PCM"
        mf.with_solvent.lebedev_order = 19
        mf.with_solvent.radii_table = ["X", 2.49443848 + 1, "He", "Li", "Be", "B", 3.85504129] # necessary to make the center C obtain zero PCM grid points

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        test_hessian = hobj.kernel()
        results = thermo.harmonic_analysis(mol, test_hessian, imaginary_freq = False)
        test_frequency = results['freq_wavenumber']

        ref_frequency = np.array([ 382.49725957,  382.55698286,  402.21775416,  438.46239691,
                                   438.61190266,  438.67469078,  486.15477748,  486.18423614,
                                   486.23674085,  883.94112582, 1170.73675205, 1170.74635162,
                                  1170.75577403, 1203.11641837, 1203.12973367, 1203.1497858 ,
                                  1330.45267458, 1330.47708115, 1584.14514575, 1584.15960707,
                                  1584.17081246, 1776.48576588, 1776.48720879, 1776.50091526,
                                  1795.71959309, 1847.4286232 , 1847.43856619, 1847.48887814,
                                  1848.23596672, 1848.2902649 , 1864.08818299, 1864.16657949,
                                  1864.21844381, 3479.1571568 , 3479.172108  , 3479.18603545,
                                  3480.18611824, 3653.60263703, 3653.63468217, 3653.64689936,
                                  3655.43694451, 3655.48652076, 3656.86682523, 3656.87311329,
                                  3656.90944133]) # Obtained via finite difference with dx = 1e-4

        assert np.max(np.abs(test_frequency - ref_frequency)) < 0.03

    def test_to_gpu_to_cpu(self):
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
        assert np.linalg.norm(hess_cpu - hess_gpu) < 2e-6
        hessobj = hessobj.to_gpu()
        hess_gpu = hessobj.kernel()
        assert np.linalg.norm(hess_cpu - hess_gpu) < 2e-6

        mf = dft.RKS(mol, xc='b3lyp').density_fit().PCM()
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-7
        mf.grids.atom_grid = (50,194)
        mf.kernel()
        hessobj = mf.Hessian()
        hessobj.auxbasis_response = 1
        hess_gpu = hessobj.kernel()
        hessobj = hessobj.to_cpu()
        hess_cpu = hessobj.kernel()
        assert np.linalg.norm(hess_cpu - hess_gpu) < 2e-6
        hessobj = hessobj.to_gpu()
        hess_gpu = hessobj.kernel()
        assert np.linalg.norm(hess_cpu - hess_gpu) < 2e-6

    def test_iswig_hessian(self):
        mol = gto.M(
            atom = """
                O      0.000000    0.000000    0.000000
                H      0.957200    0.000000    0.000000
                H     -0.239987    0.926627    0.000000
                Ne 0 0 3.44686045
            """, # The Ne position overlaps with a grid on O
            basis = "6-31g",
            verbose = 0,
        )

        mf = mol.RHF().PCM().to_gpu()
        mf.with_solvent.method = 'C-PCM'
        mf.with_solvent.lebedev_order = 7
        mf.with_solvent.eps = 78
        mf.with_solvent.surface_discretization_method = "iswig"
        mf.conv_tol = 1e-12
        mf.kernel()
        assert mf.converged

        test_hessian = mf.Hessian().kernel()

        # ref_hessian = np.empty([mol.natm, mol.natm, 3, 3])

        # def get_g(mol):
        #     mf = mol.RHF().PCM().to_gpu()
        #     mf.with_solvent.method = 'C-PCM'
        #     mf.with_solvent.lebedev_order = 7
        #     mf.with_solvent.eps = 78
        #     mf.with_solvent.surface_discretization_method = "iswig"
        #     mf.conv_tol = 1e-12
        #     e = mf.kernel()
        #     assert mf.converged
        #     g = mf.Gradients().kernel()
        #     return g

        # dx = 4e-5
        # mol_copy = mol.copy()
        # for i_atom in range(mol.natm):
        #     for i_xyz in range(3):
        #         xyz_p = mol.atom_coords()
        #         xyz_p[i_atom, i_xyz] += dx
        #         mol_copy.set_geom_(xyz_p, unit='Bohr')
        #         mol_copy.build()
        #         g_p = get_g(mol_copy)

        #         xyz_m = mol.atom_coords()
        #         xyz_m[i_atom, i_xyz] -= dx
        #         mol_copy.set_geom_(xyz_m, unit='Bohr')
        #         mol_copy.build()
        #         g_m = get_g(mol_copy)

        #         ref_hessian[i_atom, :, i_xyz, :] = (g_p - g_m) / (2 * dx)
        # print(repr(ref_hessian))

        ref_hessian = np.array([[[[ 6.31115200e-01, -1.17566250e-01,  1.91754771e-04],
         [-1.17566251e-01,  5.70127669e-01,  2.57166495e-04],
         [ 1.91755160e-04,  2.57166999e-04,  1.81787622e-02]],

        [[-5.46041234e-01, -3.59681081e-02, -3.48818833e-04],
         [ 3.53707527e-02, -5.46108845e-02,  3.56372086e-05],
         [-8.41683714e-05, -1.09277276e-04, -8.76326728e-03]],

        [[-8.51844011e-02,  1.53528705e-01,  1.20821572e-04],
         [ 8.21903971e-02, -5.15631136e-01, -3.31092011e-04],
         [-8.62214279e-05, -1.12079578e-04, -8.99866326e-03]],

        [[ 1.10434481e-04,  5.65285388e-06,  3.62423339e-05],
         [ 5.10138819e-06,  1.14351516e-04,  3.82883716e-05],
         [-2.13653817e-05, -3.58102673e-05, -4.16831560e-04]]],


       [[[-5.46041237e-01,  3.53707551e-02, -8.41679388e-05],
         [-3.59681122e-02, -5.46108811e-02, -1.09277165e-04],
         [-3.48819845e-04,  3.56377248e-05, -8.76326726e-03]],

        [[ 5.66051453e-01, -2.70109978e-02,  3.08118095e-04],
         [-2.70109946e-02,  6.07255111e-02,  3.90212250e-05],
         [ 3.08119103e-04,  3.90212716e-05,  2.46714666e-03]],

        [[-2.00063831e-02, -8.36778874e-03, -6.70749815e-05],
         [ 6.29832285e-02, -6.12049344e-03,  3.94188027e-05],
         [ 5.60931869e-05, -5.23499278e-05,  5.94897842e-03]],

        [[-3.83253503e-06,  8.03135440e-06, -1.56875055e-04],
         [-4.12177857e-06,  5.86347021e-06,  3.08371602e-05],
         [-1.53922159e-05, -2.23091492e-05,  3.47142120e-04]]],


       [[[-8.51843994e-02,  8.21903951e-02, -8.62214658e-05],
         [ 1.53528711e-01, -5.15631140e-01, -1.12079026e-04],
         [ 1.20822330e-04, -3.31092872e-04, -8.99866296e-03]],

        [[-2.00063851e-02,  6.29832276e-02,  5.60929521e-05],
         [-8.36778946e-03, -6.12049006e-03, -5.23492795e-05],
         [-6.70755607e-05,  3.94188827e-05,  5.94897827e-03]],

        [[ 1.05186211e-01, -1.45170028e-01, -3.36609791e-05],
         [-1.45170031e-01,  5.21754950e-01,  3.10191954e-04],
         [-3.36611709e-05,  3.10192843e-04,  2.69671256e-03]],

        [[ 4.57351717e-06, -3.59421325e-06,  6.37894739e-05],
         [ 9.11038775e-06, -3.32063587e-06, -1.45763634e-04],
         [-2.00856123e-05, -1.85187621e-05,  3.52972116e-04]]],


       [[[ 1.10434523e-04,  5.10128545e-06, -2.13654574e-05],
         [ 5.65303989e-06,  1.14351600e-04, -3.58101911e-05],
         [ 3.62426169e-05,  3.82881307e-05, -4.16831754e-04]],

        [[-3.83255632e-06, -4.12178028e-06, -1.53922083e-05],
         [ 8.03133983e-06,  5.86349576e-06, -2.23091354e-05],
         [-1.56875266e-04,  3.08371088e-05,  3.47142336e-04]],

        [[ 4.57352168e-06,  9.11042830e-06, -2.00856162e-05],
         [-3.59417171e-06, -3.32064506e-06, -1.85187611e-05],
         [ 6.37894324e-05, -1.45763506e-04,  3.52972256e-04]],

        [[-1.11175464e-04, -1.00899976e-05,  5.68433467e-05],
         [-1.00899961e-05, -1.16894352e-04,  7.66381300e-05],
         [ 5.68432106e-05,  7.66381792e-05, -2.83308805e-04]]]])

        assert np.max(np.abs(test_hessian - ref_hessian)) < 2e-6

if __name__ == "__main__":
    print("Full Tests for Hessian of PCMs")
    unittest.main()
