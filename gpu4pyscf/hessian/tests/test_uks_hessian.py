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
import numpy
import cupy as cp
from pyscf import gto, scf, lib, dft
from pyscf import grad, hessian
from pyscf.hessian import uks as uks_cpu
from gpu4pyscf.hessian import uks as uks_gpu
from gpu4pyscf.dft import UKS
from gpu4pyscf.hessian.uks import _get_exc_deriv2, _get_vxc_deriv1
from gpu4pyscf.hessian.tests.test_rks_hessian import _get_exc_deriv2_numerical, _get_vxc_deriv1_numerical
from gpu4pyscf.hessian.rks import get_d2mu_dr2, get_d3mu_dr3, take_last2d
from gpu4pyscf.hessian.rks import get_drho_dA_sparse as rks_get_drho_dA_sparse
from gpu4pyscf.hessian.rks import contract_d2rho_dAdB_sparse as rks_contract_d2rho_dAdB_sparse
from gpu4pyscf.hessian.uks import get_drho_dA_sparse as uks_get_drho_dA_sparse
from gpu4pyscf.hessian.uks import contract_d2rho_dAdB_sparse as uks_contract_d2rho_dAdB_sparse


try:
    from gpu4pyscf.dispersion import dftd3, dftd4
except (ImportError, OSError):
    dftd3 = dftd4 = None

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.verbose = 6
    mol.output = '/dev/null'
    mol.atom.extend([
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ])
    mol.basis = 'sto3g'
    mol.spin = 1
    mol.charge = 1
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _check_vxc(method, xc='LDA', grid_level=1, mf=None):
    reconverge = mf is None
    if mf is None:
        mf = dft.uks.UKS(mol, xc=xc)
        mf.conv_tol = 1e-14
        mf.grids.level = grid_level
        mf.kernel()
    dm = mf.make_rdm1()
    hobj = mf.Hessian()
    fn = getattr(uks_cpu, method)
    vxc_cpu = fn(hobj, mf.mo_coeff, mf.mo_occ, 12000)

    mf = mf.to_gpu()
    if reconverge:
        mf.kernel()
    hobj = mf.Hessian()
    fn = getattr(uks_gpu, method)
    vxc_gpu = fn(hobj, mf.mo_coeff, mf.mo_occ, 12000)

    # outputs of _get_vxc_deriv2 and _get_vxc_deriv1 in GPU4PYSCF are different from PySCF for memory efficiency
    if method == '_get_vxc_deriv2':
        vxc0 = numpy.einsum('nxyij,ij->nxyi', vxc_cpu[0], dm[0])
        vxc1 = numpy.einsum('nxyij,ij->nxyi', vxc_cpu[1], dm[1])
        vxc_cpu = (vxc0, vxc1)

    if method == '_get_vxc_deriv1':
        mo_occ = mf.mo_occ.get()
        mo_coeff = mf.mo_coeff.get()
        mocca = mo_coeff[0][:,mo_occ[0]>0]
        moccb = mo_coeff[1][:,mo_occ[1]>0]
        vxc0 = numpy.einsum('nxij,jq->nxiq', vxc_cpu[0], mocca)
        vxc0 = numpy.einsum('nxiq,ip->nxpq', vxc0, mo_coeff[0])
        vxc1 = numpy.einsum('nxij,jq->nxiq', vxc_cpu[1], moccb)
        vxc1 = numpy.einsum('nxiq,ip->nxpq', vxc1, mo_coeff[1])
        vxc_cpu = (vxc0, vxc1)

    #print(f'testing {method} for {xc}')
    #print(numpy.linalg.norm(vxc_cpu[0] - vxc_gpu[0].get()))
    #print(numpy.linalg.norm(vxc_cpu[1] - vxc_gpu[1].get()))
    assert numpy.linalg.norm(vxc_cpu[0] - vxc_gpu[0].get()) < 1e-6
    assert numpy.linalg.norm(vxc_cpu[1] - vxc_gpu[1].get()) < 1e-6

def _vs_cpu(mf, tol=1e-7):
    mf.conv_tol_cpscf = 1e-8
    ref = mf.Hessian().kernel()
    hessobj = mf.Hessian().to_gpu()
    hessobj.base.cphf_grids = hessobj.base.grids
    e2_gpu = hessobj.kernel()
    assert abs(ref - e2_gpu).max() < tol

class KnownValues(unittest.TestCase):
    def test_vxc_diag_high_cost(self):
        _check_vxc('_get_vxc_diag', xc='LDA')
        _check_vxc('_get_vxc_diag', xc='PBE')
        _check_vxc('_get_vxc_diag', xc='TPSS')

    def test_vxc_deriv1_high_cost(self):
        _check_vxc('_get_vxc_deriv1', xc='LDA')
        _check_vxc('_get_vxc_deriv1', xc='PBE')
        _check_vxc('_get_vxc_deriv1', xc='TPSS')

    def test_vxc_deriv2_high_cost(self):
        _check_vxc('_get_vxc_deriv2', xc='LDA')
        _check_vxc('_get_vxc_deriv2', xc='PBE')
        _check_vxc('_get_vxc_deriv2', xc='TPSS')

    def test_hessian_lda(self, disp=None):
        print('-----testing LDA Hessian----')
        mf = mol.UKS(xc='LDA').run()
        _vs_cpu(mf)

    def test_hessian_gga(self):
        print('-----testing PBE Hessian----')
        mf = mol.UKS(xc='PBE').run()
        _vs_cpu(mf, tol=1e-6)

    def test_hessian_hybrid(self):
        print('-----testing B3LYP Hessian----')
        mf = mol.UKS(xc='b3lyp').run()
        _vs_cpu(mf, tol=2e-6)

    def test_hessian_mgga_high_cost(self):
        print('-----testing M06 Hessian----')
        mf = mol.UKS(xc='m06').run()
        _vs_cpu(mf)

    def test_hessian_rsh(self):
        print('-----testing wb97 Hessian----')
        mf = mol.UKS(xc='wb97').run()
        _vs_cpu(mf)
    
    def test_hessian_hf(self):
        print('-----testing hf Hessian----')
        mf = mol.UKS(xc='hf').run()
        _vs_cpu(mf)

    @unittest.skipIf(dftd3 is None, "requires the dftd3 library")
    def test_hessian_pbe_D3_high_cost(self):
        print('-----testing PBE Hessian with D3BJ----')
        mf = mol.UKS(xc='PBE')
        mf.conv_tol = 1e-14
        mf.disp = 'd3bj'
        mf.run()
        _vs_cpu(mf, tol=1e-6)

    @unittest.skipIf(dftd4 is None, "requires the dftd4 library")
    def test_hessian_pbe_D4(self):
        print('-----testing PBE Hessian with D4----')
        mf = mol.UKS(xc='PBE')
        mf.conv_tol = 1e-14
        mf.disp = 'd4'
        mf.run()
        _vs_cpu(mf, tol=1e-6)

    @unittest.skipIf(dftd3 is None, "requires the dftd3 library")
    def test_consistency_b3lyp_d3_hess_high_cost(self):
        print('----- testing B3LYP-D3 consistency ------')
        mf1 = mol.UKS()
        mf1.conv_tol = 1e-14
        mf1.xc = 'b3lyp'
        mf1.disp = 'd3bj'
        mf1.kernel()
        hess1 = mf1.Hessian().kernel()

        mf2 = mol.UKS()
        mf2.conv_tol = 1e-14
        mf2.xc = 'b3lyp-d3bj'
        mf2.kernel()
        hess2 = mf2.Hessian().kernel()
        assert abs(hess1 - hess2).max() < 1e-8


class SmallGrid(unittest.TestCase):
    # Keep the full basis/grid regressions above; use a minimal basis and coarse
    # quadrature here to compare CPU/GPU derivatives in the routine test suite.
    @classmethod
    def setUpClass(cls):
        # The derivative comparison uses identical CPU orbitals on both devices.
        # Reuse them across the three derivative orders and retain O p shells.
        small_mol = mol.copy()
        small_mol.basis = 'sto3g'
        small_mol.build(False, False)
        cls.vxc_mfs = {}
        for xc in ('LDA', 'PBE', 'TPSS'):
            mf = small_mol.UKS(xc=xc)
            mf.grids.level = 0
            mf.run(conv_tol=1e-14)
            assert mf.converged
            cls.vxc_mfs[xc] = mf

    def make_mf(self, xc, disp=None):
        small_mol = gto.M(atom='Li 0 0 0; H .2 .3 1.6', basis='sto3g',
                          charge=1, spin=1, verbose=0)
        mf = small_mol.UKS(xc=xc)
        mf.grids.level = 0
        mf.conv_tol = 1e-14
        mf.disp = disp
        mf.run()
        assert mf.converged
        return mf

    def test_hessian_mgga(self):
        _vs_cpu(self.make_mf('m06'), tol=1e-7)

    @unittest.skipIf(dftd3 is None, "requires the dftd3 library")
    def test_consistency_b3lyp_d3_hess(self):
        mf1 = self.make_mf('b3lyp', disp='d3bj')
        mf2 = self.make_mf('b3lyp-d3bj')
        hess1 = mf1.Hessian().kernel()
        hess2 = mf2.Hessian().kernel()
        assert abs(hess1 - hess2).max() < 1e-8

    def test_vxc_diag(self):
        for xc in ('LDA', 'PBE', 'TPSS'):
            with self.subTest(xc=xc):
                _check_vxc('_get_vxc_diag', mf=self.vxc_mfs[xc])

    def test_vxc_deriv1(self):
        for xc in ('LDA', 'PBE', 'TPSS'):
            with self.subTest(xc=xc):
                _check_vxc('_get_vxc_deriv1', mf=self.vxc_mfs[xc])

    def test_vxc_deriv2(self):
        for xc in ('LDA', 'PBE', 'TPSS'):
            with self.subTest(xc=xc):
                _check_vxc('_get_vxc_deriv2', mf=self.vxc_mfs[xc])

    @unittest.skipIf(dftd3 is None, "requires the dftd3 library")
    def test_hessian_pbe_d3bj(self):
        _vs_cpu(self.make_mf('PBE', disp='d3bj'), tol=1e-6)


class GridResponse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom = '''
                C     0.000000    0.000000    0.000000
                H     1.090000    0.000000    0.000000
                H    -0.845000    0.943190    0.000000
                H    -0.545000   -0.943190    0.000000
            ''',
            basis = 'def2-svp',
            charge = 0,
            spin = 1,
            output='/dev/null',
            verbose = 0,
        )

        cls.mol2 = gto.M(
            atom = """
                C      0.000000    0.000000    0.000000
                Br     0.000000    0.000000    1.940000
                H      1.027662    0.000000   -0.363333
                H     -0.513831    0.889981   -0.363333
                H     -0.513831   -0.889981   -0.363333
            """,
            basis = "def2-svp",
            charge = 1,
            spin = 1,
            verbose = 0,
            output='/dev/null',
        )

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()
        cls.mol2.stdout.close()
        del cls.mol, cls.mol2

    def test_hessian_unrestricted_grid_response_vv10_small(self):
        small_mol = gto.M(atom='Li 0 0 0; H .2 .3 1.6', basis='sto3g',
                            charge=1, spin=1, verbose=0)
        mf = UKS(small_mol, xc='wB97M-V')
        mf.grids.atom_grid = (20, 86)
        mf.nlcgrids.atom_grid = (15, 50)
        mf.conv_tol_cpscf = 1e-10
        mf.cphf_grids = mf.grids
        mf.run(conv_tol=1e-12)
        assert mf.converged
        h = mf.Hessian().set(grid_response=True).kernel()
        reference = numpy.array([[[[-0.00842873772 ,  0.001565385106,  0.007030742146],
           [ 0.001565385106, -0.007081051523,  0.010922079183],
           [ 0.007030742146,  0.010922079183,  0.046991171761]],

          [[ 0.00842873772 , -0.001565385106, -0.007030742146],
           [-0.001565385106,  0.007081051523, -0.010922079183],
           [-0.007030742146, -0.010922079183, -0.046991171761]]],

         [[[ 0.00842873772 , -0.001565385106, -0.007030742146],
           [-0.001565385106,  0.007081051523, -0.010922079183],
           [-0.007030742146, -0.010922079183, -0.046991171761]],

          [[-0.00842873772 ,  0.001565385106,  0.007030742146],
           [ 0.001565385106, -0.007081051523,  0.010922079183],
           [ 0.007030742146,  0.010922079183,  0.046991171761]]]])
        assert numpy.max(numpy.abs(h - reference)) < 1e-4
        assert numpy.max(numpy.abs(numpy.sum(h, axis=0))) < 1e-8

    # All reference results from the same calculation with mf.level_shift = 0

    def test_hessian_unrestricted_grid_response_d2edAdB_lda(self):
        mf = UKS(self.mol, xc = 'LDA')
        mf.grids.atom_grid = (10,14)
        mf.conv_tol = 1e-8
        mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_de2 = _get_exc_deriv2(hobj, mf.mo_coeff, mf.mo_occ, mf.make_rdm1(), max_memory = None)
        reference_de2 = _get_exc_deriv2_numerical(hobj, mf.mo_coeff, mf.mo_occ, max_memory = None)

        assert abs(test_de2 - reference_de2).max() < 1e-8

    def test_hessian_unrestricted_grid_response_d2edAdB_gga(self):
        mf = UKS(self.mol, xc = 'PBE0')
        mf.grids.atom_grid = (99,590)
        mf.conv_tol = 1e-8
        mf.small_rho_cutoff = 1e-30
        # mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        # hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_de2 = _get_exc_deriv2(hobj, mf.mo_coeff, mf.mo_occ, mf.make_rdm1(), max_memory = None)
        reference_de2 = _get_exc_deriv2_numerical(hobj, mf.mo_coeff, mf.mo_occ, max_memory = None)

        assert abs(test_de2 - reference_de2).max() < 1e-8

    def test_hessian_unrestricted_grid_response_d2edAdB_mgga(self):
        mf = UKS(self.mol, xc = 'wB97M-d3bj')
        mf.grids.atom_grid = (10,14)
        mf.conv_tol = 1e-8
        mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        # hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_de2 = _get_exc_deriv2(hobj, mf.mo_coeff, mf.mo_occ, mf.make_rdm1(), max_memory = None)
        reference_de2 = _get_exc_deriv2_numerical(hobj, mf.mo_coeff, mf.mo_occ, max_memory = None)

        assert abs(test_de2 - reference_de2).max() < 1e-8

    def test_hessian_unrestricted_grid_response_dFdA_lda(self):
        mf = UKS(self.mol, xc = 'LDA')
        mf.grids.atom_grid = (10,14)
        mf.conv_tol = 1e-8
        # mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        # hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_dF = _get_vxc_deriv1(hobj, mf.mo_coeff, mf.mo_occ, max_memory = 16000)
        reference_dF = _get_vxc_deriv1_numerical(hobj, mf.mo_coeff, mf.mo_occ, max_memory = None)

        assert abs(test_dF[0] - reference_dF[0]).max() < 1e-8
        assert abs(test_dF[1] - reference_dF[1]).max() < 1e-8

    def test_hessian_unrestricted_grid_response_dFdA_gga(self):
        mf = UKS(self.mol, xc = 'wB97X-V')
        mf.grids.atom_grid = (10,14)
        mf.conv_tol = 1e-8
        mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_dF = _get_vxc_deriv1(hobj, mf.mo_coeff, mf.mo_occ, max_memory = 16000)
        reference_dF = _get_vxc_deriv1_numerical(hobj, mf.mo_coeff, mf.mo_occ, max_memory = None)

        assert abs(test_dF[0] - reference_dF[0]).max() < 1e-8
        assert abs(test_dF[1] - reference_dF[1]).max() < 1e-8

    def test_hessian_unrestricted_grid_response_dFdA_mgga(self):
        mf = UKS(self.mol, xc = 'r2SCAN')
        mf.grids.atom_grid = (10,14)
        mf.conv_tol = 1e-8
        # mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        # hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_dF = _get_vxc_deriv1(hobj, mf.mo_coeff, mf.mo_occ, max_memory = 16000)
        reference_dF = _get_vxc_deriv1_numerical(hobj, mf.mo_coeff, mf.mo_occ, max_memory = None)

        assert abs(test_dF[0] - reference_dF[0]).max() < 1e-8
        assert abs(test_dF[1] - reference_dF[1]).max() < 1e-8

    def test_hessian_unrestricted_grid_response_omega_override(self):
        mf = UKS(self.mol, xc = 'LC_WPBE')
        mf.omega = 0.175
        mf.grids.atom_grid = (10,14)
        mf.conv_tol = 1e-8
        mf.small_rho_cutoff = 1e-30

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hobj.grid_response = True

        test_de2 = _get_exc_deriv2(hobj, mf.mo_coeff, mf.mo_occ, mf.make_rdm1(), max_memory = None)
        reference_de2 = _get_exc_deriv2_numerical(hobj, mf.mo_coeff, mf.mo_occ, max_memory = None)
        assert abs(test_de2 - reference_de2).max() < 1e-8

        test_dF = _get_vxc_deriv1(hobj, mf.mo_coeff, mf.mo_occ, max_memory = 16000)
        reference_dF = _get_vxc_deriv1_numerical(hobj, mf.mo_coeff, mf.mo_occ, max_memory = None)
        assert abs(test_dF[0] - reference_dF[0]).max() < 1e-8
        assert abs(test_dF[1] - reference_dF[1]).max() < 1e-8

    def test_hessian_unrestricted_grid_response_lda(self):
        mf = UKS(self.mol, xc = 'LDA')
        mf.grids.atom_grid = (10,14)
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-10
        mf.cphf_grids.atom_grid = mf.grids.atom_grid
        mf.cphf_grids.prune = mf.grids.prune
        mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_hessian = hobj.kernel()

        # reference_hessian = numerical_d2e_dft(mf, dx = 1e-3)
        reference_hessian = numpy.array([[[[ 0.5707195503009271,  0.1626560836001656, -0.0000000000004854],
         [ 0.1626615322662117,  0.4853317595055473, -0.0000000000001728],
         [-0.0000001587051324, -0.0000001219045698,  0.0048691754683673]],

        [[-0.3532369661635171, -0.0052009331668029, -0.0000000000003677],
         [-0.0125615622100783, -0.0093934226903713,  0.0000000000004777],
         [-0.0000002155066126,  0.0000002409634775,  0.0439479128668272]],

        [[-0.0924573270768558,  0.0309152475101993,  0.0000000000002951],
         [ 0.0319184449935617, -0.1169533122628019, -0.000000000000296 ],
         [ 0.0000002735031646, -0.000000296033309 , -0.0512557183869001]],

        [[-0.1250252570553778, -0.1883703979411733,  0.0000000000005382],
         [-0.1820184150488346, -0.3589850245495629, -0.0000000000000129],
         [ 0.0000001007051664,  0.0000001769766556,  0.0024386300524689]]],


       [[[-0.3532357514234655, -0.0125567460823595,  0.0000006820567955],
         [-0.0052050403485798, -0.009400931090614 ,  0.0000000000005765],
         [ 0.0000003278307487,  0.0000017176862988,  0.043946506016138 ]],

        [[ 0.3599218367590362,  0.010786392552288 , -0.0000002039267369],
         [ 0.0107881389995734, -0.0032869418718834, -0.0000000000005331],
         [-0.0000004748009763, -0.0000000642698186, -0.0507968627638495]],

        [[-0.001064249077809 , -0.0054911513606704, -0.0000005335874237],
         [ 0.0138796422120557,  0.0059157059226767, -0.0000000000000646],
         [ 0.0000006791437412, -0.0000007598125462,  0.0007307982716216]],

        [[-0.0056218362612381,  0.0072615048928304,  0.0000000554573633],
         [-0.0194627408593995,  0.0067721670380383,  0.0000000000000108],
         [-0.0000005321721952, -0.0000008936063001,  0.0061195584755978]]],


       [[[-0.0924630973193757,  0.031913522024779 , -0.0000007991760181],
         [ 0.0309169735785292, -0.116954701027322 ,  0.0000000000015305],
         [ 0.0000008437904472, -0.0000007215570363, -0.0512609106361728]],

        [[-0.0010633642338886,  0.0138807869993901, -0.0000003697764762],
         [-0.0054907303282947,  0.0059150908939768, -0.0000000000000812],
         [-0.0000066804597409,  0.0000008310433348,  0.0007339719452605]],

        [[ 0.0896025704321679, -0.0398631082857426,  0.0000005697865625],
         [-0.0398643466341664,  0.1196100047638482, -0.0000000000007319],
         [ 0.0000039312059164, -0.0000024887438776,  0.0447557834842077]],

        [[ 0.0039238911214157, -0.0059312007391066,  0.0000005991659451],
         [ 0.0144381033900243, -0.0085703946297588, -0.000000000000711 ],
         [ 0.0000019054630651,  0.0000023792552373,  0.0057711552068257]]],


       [[[-0.1250289911055719, -0.1820149912407087, -0.0000025929740717],
         [-0.1883796397237972, -0.3589829712742632, -0.0000000000011423],
         [-0.0000018194572307, -0.0000010889313062,  0.0024286967658768]],

        [[-0.0056186204128172, -0.0194655648873892,  0.000001416357078 ],
         [ 0.0072672095504656,  0.0067675269597852,  0.0000000000005616],
         [-0.0000049267481206,  0.0000008158922227,  0.0061253213060923]],

        [[ 0.0039219763677267,  0.0144372085543742,  0.0000002320348407],
         [-0.0059325125877363, -0.0085734247835534,  0.0000000000002568],
         [ 0.0000043812808936, -0.0000037051108293,  0.005771708528144 ]],

        [[ 0.1267256351498713,  0.187043347574356 ,  0.0000009445821455],
         [ 0.1870449427608945,  0.3607888690962957,  0.0000000000003318],
         [ 0.0000023649247005,  0.0000039781511418, -0.0143257266005054]]]])

        assert numpy.max(numpy.abs(test_hessian - reference_hessian)) < 1e-4
        # Translation invariance
        assert numpy.max(numpy.abs(numpy.sum(test_hessian, axis = 0))) < 1e-10

    def test_hessian_unrestricted_grid_response_gga(self):
        mf = UKS(self.mol, xc = "revPBE")
        mf.grids.atom_grid = (20,26)
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-10
        mf.cphf_grids.atom_grid = mf.grids.atom_grid
        mf.cphf_grids.prune = mf.grids.prune
        # mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_hessian = hobj.kernel()

        # reference_hessian = numerical_d2e_dft(mf, dx = 4e-3)
        reference_hessian = numpy.array([[[[ 0.5881463970112706,  0.1211386971501638, -0.0000000000000529],
         [ 0.1211350605861625,  0.4506283201038339, -0.0000000000000372],
         [-0.0000006512391555,  0.0000005700882752,  0.0538879584241884]],

        [[-0.3790887888731054, -0.0032609156549784, -0.0000000000003552],
         [-0.0115750661662806, -0.0584626620944129,  0.0000000000000026],
         [ 0.0000003391386338, -0.0000003840012678, -0.0115219722890125]],

        [[-0.0944036315072005,  0.0271286206489085,  0.0000000000001547],
         [ 0.0268783983182891, -0.1173670579734665,  0.0000000000000105],
         [ 0.0000000130308264, -0.0000002605340527, -0.0326636660844471]],

        [[-0.1146539766274934, -0.1450064021440989,  0.000000000000255 ],
         [-0.1364383927401902, -0.2747986000324609,  0.0000000000000251],
         [ 0.0000002990698522,  0.0000000744473094, -0.0097023200499114]]],


       [[[-0.3790855533722243, -0.0115720371956507, -0.000000024615438 ],
         [-0.0032611921789265, -0.0584615851970116,  0.000000000001371 ],
         [ 0.0000003922406479,  0.000000088927761 , -0.0115223791193567]],

        [[ 0.3870872408233417,  0.0106814730127843, -0.0000000313542464],
         [ 0.0106817628311162,  0.0499685389765993, -0.0000000000007978],
         [-0.0000003251968694,  0.0000000245130923,  0.0068502448045267]],

        [[-0.0009306057945496, -0.0038459041618033, -0.0000012297407368],
         [ 0.0160267171118583,  0.0046151625494317, -0.0000000000023973],
         [ 0.0000000054513477, -0.0000000309846593, -0.001118484828455 ]],

        [[-0.007071081662105 ,  0.0047364683462836,  0.0000012857104234],
         [-0.0234472877671116,  0.0038778836698172,  0.0000000000018252],
         [-0.0000000724935945, -0.0000000824565971,  0.0057906191428346]]],


       [[[-0.0944091521876377,  0.0268717395024085, -0.0000002566278707],
         [ 0.0271320414713836, -0.1173639820351985, -0.000000000001018 ],
         [ 0.00000165866452  ,  0.0000015454350299, -0.0326623590818173]],

        [[-0.0009295308301127,  0.0160265092487449, -0.000000078084366 ],
         [-0.0038456821990218,  0.0046156984272487,  0.0000000000004568],
         [-0.0000011299395586, -0.0000000368010368, -0.0011186451500806]],

        [[ 0.0897687143511378, -0.0382275354718908,  0.0000007725356049],
         [-0.0382279724130824,  0.1211080808877357,  0.0000000000012136],
         [ 0.0000000820235407, -0.000000091535321 ,  0.0283197706865471]],

        [[ 0.0055699686635352, -0.0046707132755064, -0.0000004378233685],
         [ 0.0149416131413477, -0.0083597972812266, -0.0000000000006533],
         [-0.0000006107469669, -0.0000014170972729,  0.005461233545463 ]]],


       [[[-0.1146544710585035, -0.1364327413227243,  0.0000000209608242],
         [-0.14501515941     , -0.2747939657836151, -0.00000000000013  ],
         [-0.0000001200790301,  0.0000000018890514, -0.0097027237846366]],

        [[-0.0070690462636303, -0.0234472506656974,  0.0000001803752727],
         [ 0.0047426336208523,  0.0038774778584225, -0.0000000000001537],
         [-0.0000000176728354,  0.000000037716901 ,  0.0057909651482515]],

        [[ 0.0055691291103999,  0.0149416264684232, -0.0000000234109521],
         [-0.0046708890939079, -0.0083608672789282,  0.0000000000001711],
         [ 0.000000118694693 , -0.0000000402801265,  0.0054611151282856]],

        [[ 0.1161543882039345,  0.1449383655194592, -0.0000001779251459],
         [ 0.1449434148808881,  0.2792773552015571,  0.0000000000001174],
         [ 0.0000000190541471,  0.0000000006782908, -0.0015493564923762]]]])

        assert numpy.max(numpy.abs(test_hessian - reference_hessian)) < 1e-4
        # Translation invariance
        assert numpy.max(numpy.abs(numpy.sum(test_hessian, axis = 0))) < 1e-10

    def test_hessian_unrestricted_grid_response_mgga(self):
        mf = UKS(self.mol, xc = 'r2SCAN')
        mf.grids.atom_grid = (20,194)
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-10
        mf.cphf_grids.atom_grid = mf.grids.atom_grid
        mf.cphf_grids.prune = mf.grids.prune
        # mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_hessian = hobj.kernel()

        # reference_hessian = numerical_d2e_dft(mf, dx = 4e-3)
        reference_hessian = numpy.array([[[[ 0.5820155221143387,  0.1166859509692322,  0.0000000000000068],
         [ 0.1166914308990807,  0.4215371295934883, -0.000000000000019 ],
         [-0.0000031473303009,  0.0000008876429591,  0.0422379005416061]],

        [[-0.3671822912818257,  0.0001641790860455, -0.0000000000001163],
         [-0.0084188365954696, -0.045232076075573 ,  0.0000000000000735],
         [ 0.0000018059421736, -0.000000696544257 ,  0.0007909441897174]],

        [[-0.0887119779785228,  0.0210857724144836,  0.0000000000000498],
         [ 0.0257029513710816, -0.0977462026366532, -0.0000000000000094],
         [ 0.0000006044114792, -0.0000007370008631, -0.0335608035483526]],

        [[-0.1261212528438083, -0.1379359024675231,  0.0000000000000598],
         [-0.1339755456715441, -0.2785588508753545, -0.0000000000000455],
         [ 0.0000007369764382,  0.0000005459006858, -0.0094680411821944]]],


       [[[-0.36718439518541  , -0.0084145827370652, -0.0000010937446911],
         [ 0.0001613660267318, -0.0452296250851095,  0.0000000000047075],
         [ 0.0000011557333458, -0.0000003566965959,  0.0007895705733541]],

        [[ 0.3720461948922005,  0.0076307092713296,  0.0000006044353657],
         [ 0.0076326721058506,  0.0370290106022019, -0.0000000000016288],
         [-0.0000007294261029,  0.0000000682071752, -0.0003285793846421]],

        [[-0.0003574875750562, -0.0046145009915538, -0.0000017249352664],
         [ 0.0141134126344089,  0.0040211384784578, -0.0000000000083034],
         [-0.0000003721849218,  0.0000003096229245, -0.0029342109973904]],

        [[-0.0045043121333632,  0.0053983744558728,  0.0000022142445921],
         [-0.021907450764691 ,  0.0041794760064773,  0.0000000000052253],
         [-0.0000000541214712, -0.0000000211295703,  0.0024732198082593]]],


       [[[-0.0887194194503661,  0.0256980264625242,  0.000000148508423 ],
         [ 0.0210853288650416, -0.0977477213169667,  0.0000000000005413],
         [ 0.0000032405625336,  0.000001767831076 , -0.0335579759419456]],

        [[-0.0003552163563569,  0.0141117532326946, -0.0000005147952575],
         [-0.0046118792429717,  0.0040227438962131,  0.0000000000001137],
         [-0.0000015905651241, -0.000000030696399 , -0.0029348152091413]],

        [[ 0.0824356948214006, -0.0330085155503534,  0.0000008643203307],
         [-0.0330096046065181,  0.1090590747974546, -0.00000000000069  ],
         [-0.0000003967634138,  0.0000002395047077,  0.0313808111003626]],

        [[ 0.0066389409815876, -0.0068012641426252, -0.0000004980334961],
         [ 0.016536154988242 , -0.0153340973802762,  0.000000000000036 ],
         [-0.0000012532340304, -0.0000019766386028,  0.0051119800508289]]],


       [[[-0.1261171771924068, -0.1339717620026659, -0.0000010840465282],
         [-0.1379348300095638, -0.2785603603433773, -0.0000000000087387],
         [ 0.0000000260601141,  0.0000011230939095, -0.0094691791798755]],

        [[-0.004504017141832 , -0.021907299380222 ,  0.0000011358784016],
         [ 0.0053978468682048,  0.0041799409876129,  0.000000000009209 ],
         [-0.0000002916932251,  0.0000000006658938,  0.0024740823052023]],

        [[ 0.0066327879852252,  0.0165384077064068,  0.000000298204179 ],
         [-0.0068089746262101, -0.0153332385975646,  0.0000000000081774],
         [ 0.0000005948822129, -0.0000005310908796,  0.0051111633828136]],

        [[ 0.123988406347697 ,  0.1393406536718544, -0.0000003500360516],
         [ 0.1393459577633538,  0.2897136579541049, -0.0000000000086471],
         [-0.000000329248323 , -0.0000005926717728,  0.0018839334914163]]]])

        assert numpy.max(numpy.abs(test_hessian - reference_hessian)) < 1e-4
        # Translation invariance
        assert numpy.max(numpy.abs(numpy.sum(test_hessian, axis = 0))) < 1e-10

    def test_hessian_unrestricted_grid_response_vv10_density_fitting(self):
        mf = UKS(self.mol, xc = 'wB97M-V')
        mf.grids.atom_grid = (10,14)
        mf.nlcgrids.atom_grid = (15,14)
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-10
        mf.cphf_grids.atom_grid = mf.grids.atom_grid
        mf.cphf_grids.prune = mf.grids.prune
        mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_hessian = hobj.kernel()

        # reference_hessian = numerical_d2e_dft(mf, dx = 4e-3)
        reference_hessian = numpy.array([[[[ 0.5829877959604642,  0.1478164925926492, -0.0000000000003468],
         [ 0.1478233698241123,  0.4519738358275474, -0.0000000000001349],
         [ 0.0000019011781698,  0.0000008791155928,  0.0277606662341917]],

        [[-0.3500657840262433, -0.0018402965775821,  0.0000000000003717],
         [-0.0100608683339942, -0.0159935492065745,  0.0000000000002656],
         [-0.0000014486573607, -0.0000000925134134,  0.0405408522813359]],

        [[-0.0940749813341246,  0.0205430024346476, -0.0000000000001694],
         [ 0.0232696936310756, -0.107787475119081 , -0.0000000000003403],
         [-0.0000001515344794, -0.0000000555053919, -0.0531778580414026]],

        [[-0.1388470305954959, -0.1665191984495662,  0.0000000000001476],
         [-0.1610321951196203, -0.3281928114979571,  0.0000000000002089],
         [-0.0000003009858057, -0.0000007310955175, -0.0151236604733161]]],


       [[[-0.35006642676093  , -0.0100617768753475,  0.0000001051682122],
         [-0.001846228657126 , -0.015990236480308 ,  0.0000000000799501],
         [-0.0000000548366127,  0.0000005119373606,  0.0405370040959886]],

        [[ 0.3571960398480456,  0.0087097512115691, -0.000000917450043 ],
         [ 0.0087121506802679,  0.0029491523678897, -0.000000000047569 ],
         [ 0.0000000534895461, -0.0000001789786375, -0.0430235023676496]],

        [[-0.0013751632899628, -0.0047636112482097, -0.0000034107404229],
         [ 0.014367996452172 ,  0.0060151146088622, -0.000000000140614 ],
         [ 0.0000002783175218,  0.0000000318952503, -0.0018898495944482]],

        [[-0.0057544498020534,  0.0061156369118298,  0.0000042230222539],
         [-0.0212339184751109,  0.0070259695036379,  0.000000000108235 ],
         [-0.0000002769700302, -0.0000003648538138,  0.0043763478656142]]],


       [[[-0.0940725766045425,  0.0232719756537297, -0.0000017198026964],
         [ 0.0205458465222345, -0.1077899085383041, -0.0000000000943822],
         [-0.0000010060938176, -0.000001449778915 , -0.0531697938112808]],

        [[-0.0013750310098315,  0.0143654110627575,  0.0000000161943172],
         [-0.0047656682238828,  0.0060185839715389,  0.0000000000605793],
         [ 0.0000002140815303,  0.0000000238238288, -0.0018979751082114]],

        [[ 0.0892461216849411, -0.0312481630046074,  0.0000035545062487],
         [-0.0312498668705308,  0.1142595461788715,  0.0000000001356289],
         [ 0.0000001978757297,  0.0000000786988114,  0.0496662327136618]],

        [[ 0.0062014859267961, -0.0063892237101293, -0.0000018508978681],
         [ 0.0154696885744932, -0.0124882216134825, -0.0000000001018295],
         [ 0.0000005941346581,  0.0000013472554461,  0.0054015362059369]]],


       [[[-0.1388477658196187, -0.1610273604757614,  0.0000005465797245],
         [-0.1665257481245682, -0.3281885078034938, -0.0000000000613658],
         [ 0.000001263466172 , -0.0000018101805332, -0.0151262089102309]],

        [[-0.0057558164847926, -0.0212344581561098,  0.0000017356110402],
         [ 0.0061141555704469,  0.0070255345170974,  0.0000000001173134],
         [-0.0000003841274243,  0.0000000388004865,  0.0043810299511624]],

        [[ 0.0062030586779199,  0.0154694402396266, -0.000000204212136 ],
         [-0.0063845989000666, -0.0124911509442605,  0.0000000000921417],
         [-0.0000009250951394,  0.0000012327283361,  0.0053991567485613]],

        [[ 0.1384005236274716,  0.1667923783920666, -0.0000020779786295],
         [ 0.166796191453919 ,  0.333654124226368 , -0.0000000001480894],
         [ 0.0000000457569538,  0.0000005386519564,  0.0053460222100712]]]])

        assert numpy.max(numpy.abs(test_hessian - reference_hessian)) < 1e-4
        # Translation invariance
        assert numpy.max(numpy.abs(numpy.sum(test_hessian, axis = 0))) < 1e-8

    def test_hessian_unrestricted_grid_response_vv10_direct_high_cost(self):
        mf = UKS(self.mol, xc = 'wB97M-V')
        mf.grids.atom_grid = (10,14)
        mf.nlcgrids.atom_grid = (15,14)
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-10
        mf.cphf_grids.atom_grid = mf.grids.atom_grid
        mf.cphf_grids.prune = mf.grids.prune
        # mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_hessian = hobj.kernel()

        # reference_hessian = numerical_d2e_dft(mf, dx = 4e-3)
        reference_hessian = numpy.array([[[[ 0.5831949235897381,  0.1478628805322846,  0.0000000000000006],
         [ 0.1478697344845913,  0.4521684956274169,  0.0000000000000138],
         [ 0.0000018742662335,  0.0000008833360848,  0.0278471293271266]],

        [[-0.3501925633656866, -0.0018426491819692,  0.0000000000000009],
         [-0.0100643052318161, -0.0160016720308494, -0.0000000000000287],
         [-0.0000014606701959, -0.000000095303991 ,  0.0405141810840608]],

        [[-0.0941144036622488,  0.0205657667966958,  0.0000000000000038],
         [ 0.0232871085837222, -0.1078323056764724, -0.0000000000000053],
         [-0.0000001298055829, -0.0000000681327078, -0.0532169035501965]],

        [[-0.1388879565571155, -0.1665859981463402, -0.0000000000000063],
         [-0.1610925378388306, -0.328334517917972 ,  0.0000000000000236],
         [-0.0000002837877433, -0.0000007199004448, -0.0151444068601844]]],


       [[[-0.3501932038282255, -0.0100652053057135,  0.0000001051164696],
         [-0.0018485610635485, -0.0159983474572467,  0.0000000000798121],
         [-0.0000000113172006,  0.0000004890112759,  0.0405103287987409]],

        [[ 0.3573044306625917,  0.0087168001650371, -0.0000009175593225],
         [ 0.0087193303356103,  0.0029514241217631, -0.0000000000471742],
         [ 0.0000000395102562, -0.0000001659725881, -0.0430090399990274]],

        [[-0.0013628927016673, -0.0047369354269911, -0.0000034090530315],
         [ 0.0143572809215842,  0.006029554815809 , -0.0000000001408067],
         [ 0.0000002736661731,  0.0000000082615859, -0.0018814884594825]],

        [[-0.0057483341368986,  0.0060853405697003,  0.0000042214958829],
         [-0.0212280501912659,  0.0070173685192132,  0.0000000001081683],
         [-0.0000003018603284, -0.0000003313014307,  0.0043801996592778]]],


       [[[-0.0941120281958939,  0.0232893113693566, -0.0000017201402778],
         [ 0.0205685873705813, -0.1078347298306276, -0.0000000000942096],
         [-0.0000010382487439, -0.0000014400014237, -0.0532088325196612]],

        [[-0.0013629335133825,  0.0143547986652887,  0.0000000161564526],
         [-0.0047391576738776,  0.0060329604104324,  0.0000000000605895],
         [ 0.0000002063908211,  0.0000000190087796, -0.0018896222220578]],

        [[ 0.0892692828731306, -0.0312693206076103,  0.0000035539828331],
         [-0.031270806810188 ,  0.1143001302188473,  0.0000000001352783],
         [ 0.000000217506041 ,  0.0000000830025354,  0.0496969080334779]],

        [[ 0.0062056788336651, -0.0063747894243071, -0.0000018499990087],
         [ 0.0154413771145251, -0.0124983607983076, -0.0000000001016551],
         [ 0.0000006143526937,  0.0000013379875541,  0.005401546708344 ]]],


       [[[-0.1388886707893392, -0.161087739306745 ,  0.0000005484856271],
         [-0.1665925569428808, -0.3283302618124415, -0.000000000061283 ],
         [ 0.0000012511061718, -0.0000018159475729, -0.0151469553728437]],

        [[-0.0057498109296006, -0.0212285024685731,  0.0000017346342518],
         [ 0.0060837710830208,  0.0070170327138969,  0.0000000001171361],
         [-0.0000003966942053,  0.0000000379701661,  0.0043848802387699]],

        [[ 0.0062072301629384,  0.0154409840764508, -0.0000002042449851],
         [-0.0063702509775221, -0.0125014258696154,  0.0000000000920771],
         [-0.0000009127730932,  0.0000012193228788,  0.0053991650164636]],

        [[ 0.1384312515547714,  0.1668752576965415, -0.0000020788748949],
         [ 0.1668790368366674,  0.3338146549645982, -0.0000000001479301],
         [ 0.0000000583610521,  0.0000005586547613,  0.0053629101171804]]]])

        assert numpy.max(numpy.abs(test_hessian - reference_hessian)) < 1e-4
        # Translation invariance
        assert numpy.max(numpy.abs(numpy.sum(test_hessian, axis = 0))) < 1e-8

    def test_hessian_grid_response_one_atom(self):
        mol = gto.M(
            atom = "F 0 -1 -2",
            basis = "6-31g",
            charge = 0,
            spin = 1,
            verbose = 0,
        )
        mf = mol.UKS(xc = "wB97M-V").density_fit(auxbasis = "def2-universal-jkfit").to_gpu()
        mf.grids.atom_grid = (10,14)
        mf.nlcgrids.atom_grid = (10,14)
        mf.conv_tol = 5e-12
        mf.max_cycle = 70
        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        mf.conv_tol_cpscf = 1e-10
        mf.cphf_grids.atom_grid = mf.grids.atom_grid
        mf.cphf_grids.prune = mf.grids.prune
        hobj.grid_response = True
        test_hessian = hobj.kernel()

        ref_hessian = numpy.zeros((1,1,3,3))

        assert numpy.max(numpy.abs(test_hessian - ref_hessian)) < 1e-9

    def test_uks_d2rho_lda(self):
        mol = self.mol2
        cp.random.seed(200)

        mf = UKS(mol, xc = "BN05")
        mf.grids.atom_grid = (20,26)

        dm0 = cp.random.rand(2, mol.nao, mol.nao) * 2 - 1

        grids = mf.grids
        grids.build(sort_grids_of_each_atom = True)
        ngrids = grids.coords.shape[0]

        weight_depsilon_drho = cp.random.rand(2, ngrids) * 2 - 1

        ni = mf._numint
        xctype = ni._xc_type(mf.xc)

        ni.gdftopt = None
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

        _sorted_mol = opt._sorted_mol
        nao = _sorted_mol.nao
        dm0_sorted = opt.sort_orbitals(dm0, axis=[1,2])
        dm_mask_buf = cp.empty(2 * nao * nao)

        ao_loc_sorted = _sorted_mol.ao_loc
        ao_expand = ao_loc_sorted[1:] - ao_loc_sorted[:-1]
        from pyscf.gto.mole import ATOM_OF
        i_atom_of_aos = numpy.repeat(_sorted_mol._bas[:,ATOM_OF], ao_expand)
        i_atom_of_aos = cp.asarray(i_atom_of_aos, dtype = cp.int32)

        natm = _sorted_mol.natm

        ref_drho_dA_full_response = cp.zeros((natm, 3, 2, ngrids))
        ref_drho_dA_orbital_response = cp.zeros((natm, 3, 2, ngrids))
        ref_d2E_dAdB_full_response = cp.zeros((natm, natm, 3, 3))
        ref_d2E_dAdB_orbital_response = cp.zeros((natm, natm, 3, 3))
        test_drho_dA_full_response = cp.zeros((natm, 3, 2, ngrids))
        test_drho_dA_orbital_response = cp.zeros((natm, 3, 2, ngrids))
        test_d2E_dAdB_full_response = cp.zeros((natm, natm, 3, 3))
        test_d2E_dAdB_orbital_response = cp.zeros((natm, natm, 3, 3))

        g0 = 0
        for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, deriv = 2):
            g1 = g0 + weight.shape[0]

            mu = ao[0]
            dmu_dr = ao[1:4]
            d2mu_dr2 = get_d2mu_dr2(ao)

            dm0_masked = take_last2d(dm0_sorted, idx, out = dm_mask_buf)

            i_atom_of_grids = int(grids.atm_idx[g0])
            assert cp.max(cp.abs(grids.atm_idx[g0:g1] - i_atom_of_grids)) == 0 # Guaranteed by grids.build(sort_grids_of_each_atom = True)

            masked_i_atom_of_aos = i_atom_of_aos[idx]

            drho_dA_orbital_response, drho_dA_grid_response = \
                rks_get_drho_dA_sparse(dm0_masked[0], xctype, natm, masked_i_atom_of_aos, i_atom_of_grids, mu, dmu_dr)
            ref_drho_dA_full_response[:,:,0,g0:g1] = drho_dA_orbital_response + drho_dA_grid_response
            ref_drho_dA_orbital_response[:,:,0,g0:g1] = drho_dA_orbital_response
            drho_dA_orbital_response, drho_dA_grid_response = \
                rks_get_drho_dA_sparse(dm0_masked[1], xctype, natm, masked_i_atom_of_aos, i_atom_of_grids, mu, dmu_dr)
            ref_drho_dA_full_response[:,:,1,g0:g1] = drho_dA_orbital_response + drho_dA_grid_response
            ref_drho_dA_orbital_response[:,:,1,g0:g1] = drho_dA_orbital_response

            drho_dA_orbital_response, drho_dA_grid_response = \
                uks_get_drho_dA_sparse(dm0_masked, xctype, natm, masked_i_atom_of_aos, i_atom_of_grids, mu, dmu_dr)
            test_drho_dA_full_response[:,:,:,g0:g1] = drho_dA_orbital_response + drho_dA_grid_response
            test_drho_dA_orbital_response[:,:,:,g0:g1] = drho_dA_orbital_response

            ref_d2E_dAdB_full_response += rks_contract_d2rho_dAdB_sparse(
                dm0_masked[0], xctype, natm, masked_i_atom_of_aos, i_atom_of_grids,
                mu, dmu_dr, d2mu_dr2, None,
                weight_depsilon_drho[0, g0:g1],
            ) + rks_contract_d2rho_dAdB_sparse(
                dm0_masked[1], xctype, natm, masked_i_atom_of_aos, i_atom_of_grids,
                mu, dmu_dr, d2mu_dr2, None,
                weight_depsilon_drho[1, g0:g1],
            )
            test_d2E_dAdB_full_response += uks_contract_d2rho_dAdB_sparse(
                dm0_masked, xctype, natm, masked_i_atom_of_aos, i_atom_of_grids,
                mu, dmu_dr, d2mu_dr2, None,
                weight_depsilon_drho[:, g0:g1],
            )

            ref_d2E_dAdB_orbital_response += rks_contract_d2rho_dAdB_sparse(
                dm0_masked[0], xctype, natm, masked_i_atom_of_aos, i_atom_of_grids,
                mu, dmu_dr, d2mu_dr2, None,
                weight_depsilon_drho[0, g0:g1],
                with_grid_response = False,
            ) + rks_contract_d2rho_dAdB_sparse(
                dm0_masked[1], xctype, natm, masked_i_atom_of_aos, i_atom_of_grids,
                mu, dmu_dr, d2mu_dr2, None,
                weight_depsilon_drho[1, g0:g1],
                with_grid_response = False,
            )
            test_d2E_dAdB_orbital_response += uks_contract_d2rho_dAdB_sparse(
                dm0_masked, xctype, natm, masked_i_atom_of_aos, i_atom_of_grids,
                mu, dmu_dr, d2mu_dr2, None,
                weight_depsilon_drho[:, g0:g1],
                with_grid_response = False,
            )

            g0 = g1
        assert g1 == ngrids

        diff_drhodA_full = cp.max(cp.abs(test_drho_dA_full_response - ref_drho_dA_full_response))
        diff_drhodA_orbital = cp.max(cp.abs(test_drho_dA_orbital_response - ref_drho_dA_orbital_response))
        max_drhodA_full = cp.max(cp.abs(ref_drho_dA_full_response))

        assert diff_drhodA_full < 1e-9 and diff_drhodA_full / max_drhodA_full < 1e-12
        assert diff_drhodA_orbital < 1e-9 and diff_drhodA_orbital / max_drhodA_full < 1e-12

        diff_d2EdAdB_full = cp.max(cp.abs(test_d2E_dAdB_full_response - ref_d2E_dAdB_full_response))
        diff_d2EdAdB_orbital = cp.max(cp.abs(test_d2E_dAdB_orbital_response - ref_d2E_dAdB_orbital_response))
        max_d2EdAdB_full = cp.max(cp.abs(ref_d2E_dAdB_full_response))

        assert diff_d2EdAdB_full < 1e-5 and diff_d2EdAdB_full / max_d2EdAdB_full < 1e-9
        assert diff_d2EdAdB_orbital < 1e-5 and diff_d2EdAdB_orbital / max_d2EdAdB_full < 1e-9

    def test_uks_d2rho_gga(self):
        mol = self.mol2
        cp.random.seed(201)

        mf = UKS(mol, xc = "B3PW91")
        mf.grids.atom_grid = (20,26)

        dm0 = cp.random.rand(2, mol.nao, mol.nao) * 2 - 1

        grids = mf.grids
        grids.build(sort_grids_of_each_atom = True)
        ngrids = grids.coords.shape[0]

        weight_depsilon_drho = cp.random.rand(2, ngrids) * 2 - 1
        weight_depsilon_dnablarho = cp.random.rand(2, 3, ngrids) * 2 - 1

        ni = mf._numint
        xctype = ni._xc_type(mf.xc)

        ni.gdftopt = None
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

        _sorted_mol = opt._sorted_mol
        nao = _sorted_mol.nao
        dm0_sorted = opt.sort_orbitals(dm0, axis=[1,2])
        dm_mask_buf = cp.empty(2 * nao * nao)

        ao_loc_sorted = _sorted_mol.ao_loc
        ao_expand = ao_loc_sorted[1:] - ao_loc_sorted[:-1]
        from pyscf.gto.mole import ATOM_OF
        i_atom_of_aos = numpy.repeat(_sorted_mol._bas[:,ATOM_OF], ao_expand)
        i_atom_of_aos = cp.asarray(i_atom_of_aos, dtype = cp.int32)

        natm = _sorted_mol.natm

        ref_drho_dA_full_response = cp.zeros((natm, 3, 2, 4, ngrids))
        ref_drho_dA_orbital_response = cp.zeros((natm, 3, 2, 4, ngrids))
        ref_d2E_dAdB_full_response = cp.zeros((natm, natm, 3, 3))
        ref_d2E_dAdB_orbital_response = cp.zeros((natm, natm, 3, 3))
        test_drho_dA_full_response = cp.zeros((natm, 3, 2, 4, ngrids))
        test_drho_dA_orbital_response = cp.zeros((natm, 3, 2, 4, ngrids))
        test_d2E_dAdB_full_response = cp.zeros((natm, natm, 3, 3))
        test_d2E_dAdB_orbital_response = cp.zeros((natm, natm, 3, 3))

        g0 = 0
        for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, deriv = 3):
            g1 = g0 + weight.shape[0]

            mu = ao[0]
            dmu_dr = ao[1:4]
            d2mu_dr2 = get_d2mu_dr2(ao)
            d3mu_dr3 = get_d3mu_dr3(ao)

            dm0_masked = take_last2d(dm0_sorted, idx, out = dm_mask_buf)

            i_atom_of_grids = int(grids.atm_idx[g0])
            assert cp.max(cp.abs(grids.atm_idx[g0:g1] - i_atom_of_grids)) == 0 # Guaranteed by grids.build(sort_grids_of_each_atom = True)

            masked_i_atom_of_aos = i_atom_of_aos[idx]

            drho_dA_orbital_response, drho_dA_grid_response = \
                rks_get_drho_dA_sparse(dm0_masked[0], xctype, natm, masked_i_atom_of_aos, i_atom_of_grids, mu, dmu_dr, d2mu_dr2)
            ref_drho_dA_full_response[:,:,0,:,g0:g1] = drho_dA_orbital_response + drho_dA_grid_response
            ref_drho_dA_orbital_response[:,:,0,:,g0:g1] = drho_dA_orbital_response
            drho_dA_orbital_response, drho_dA_grid_response = \
                rks_get_drho_dA_sparse(dm0_masked[1], xctype, natm, masked_i_atom_of_aos, i_atom_of_grids, mu, dmu_dr, d2mu_dr2)
            ref_drho_dA_full_response[:,:,1,:,g0:g1] = drho_dA_orbital_response + drho_dA_grid_response
            ref_drho_dA_orbital_response[:,:,1,:,g0:g1] = drho_dA_orbital_response

            drho_dA_orbital_response, drho_dA_grid_response = \
                uks_get_drho_dA_sparse(dm0_masked, xctype, natm, masked_i_atom_of_aos, i_atom_of_grids, mu, dmu_dr, d2mu_dr2)
            test_drho_dA_full_response[:,:,:,:,g0:g1] = drho_dA_orbital_response + drho_dA_grid_response
            test_drho_dA_orbital_response[:,:,:,:,g0:g1] = drho_dA_orbital_response

            ref_d2E_dAdB_full_response += rks_contract_d2rho_dAdB_sparse(
                dm0_masked[0], xctype, natm, masked_i_atom_of_aos, i_atom_of_grids,
                mu, dmu_dr, d2mu_dr2, d3mu_dr3,
                weight_depsilon_drho[0, g0:g1], weight_depsilon_dnablarho[0, :, g0:g1],
            ) + rks_contract_d2rho_dAdB_sparse(
                dm0_masked[1], xctype, natm, masked_i_atom_of_aos, i_atom_of_grids,
                mu, dmu_dr, d2mu_dr2, d3mu_dr3,
                weight_depsilon_drho[1, g0:g1], weight_depsilon_dnablarho[1, :, g0:g1],
            )
            test_d2E_dAdB_full_response += uks_contract_d2rho_dAdB_sparse(
                dm0_masked, xctype, natm, masked_i_atom_of_aos, i_atom_of_grids,
                mu, dmu_dr, d2mu_dr2, d3mu_dr3,
                weight_depsilon_drho[:, g0:g1], weight_depsilon_dnablarho[:, :, g0:g1],
            )

            ref_d2E_dAdB_orbital_response += rks_contract_d2rho_dAdB_sparse(
                dm0_masked[0], xctype, natm, masked_i_atom_of_aos, i_atom_of_grids,
                mu, dmu_dr, d2mu_dr2, d3mu_dr3,
                weight_depsilon_drho[0, g0:g1], weight_depsilon_dnablarho[0, :, g0:g1],
                with_grid_response = False,
            ) + rks_contract_d2rho_dAdB_sparse(
                dm0_masked[1], xctype, natm, masked_i_atom_of_aos, i_atom_of_grids,
                mu, dmu_dr, d2mu_dr2, d3mu_dr3,
                weight_depsilon_drho[1, g0:g1], weight_depsilon_dnablarho[1, :, g0:g1],
                with_grid_response = False,
            )
            test_d2E_dAdB_orbital_response += uks_contract_d2rho_dAdB_sparse(
                dm0_masked, xctype, natm, masked_i_atom_of_aos, i_atom_of_grids,
                mu, dmu_dr, d2mu_dr2, d3mu_dr3,
                weight_depsilon_drho[:, g0:g1], weight_depsilon_dnablarho[:, :, g0:g1],
                with_grid_response = False,
            )

            g0 = g1
        assert g1 == ngrids

        diff_drhodA_full = cp.max(cp.abs(test_drho_dA_full_response - ref_drho_dA_full_response))
        diff_drhodA_orbital = cp.max(cp.abs(test_drho_dA_orbital_response - ref_drho_dA_orbital_response))
        max_drhodA_full = cp.max(cp.abs(ref_drho_dA_full_response))

        assert diff_drhodA_full < 1e-6 and diff_drhodA_full / max_drhodA_full < 1e-9
        assert diff_drhodA_orbital < 1e-6 and diff_drhodA_orbital / max_drhodA_full < 1e-9

        diff_d2EdAdB_full = cp.max(cp.abs(test_d2E_dAdB_full_response - ref_d2E_dAdB_full_response))
        diff_d2EdAdB_orbital = cp.max(cp.abs(test_d2E_dAdB_orbital_response - ref_d2E_dAdB_orbital_response))
        max_d2EdAdB_full = cp.max(cp.abs(ref_d2E_dAdB_full_response))

        assert diff_d2EdAdB_full < 1e-2 and diff_d2EdAdB_full / max_d2EdAdB_full < 1e-8
        assert diff_d2EdAdB_orbital < 1e-2 and diff_d2EdAdB_orbital / max_d2EdAdB_full < 1e-8

    def test_uks_d2rho_mgga(self):
        mol = self.mol2
        cp.random.seed(202)

        mf = UKS(mol, xc = "TPSSH")
        mf.grids.atom_grid = (20,26)

        dm0 = cp.random.rand(2, mol.nao, mol.nao) * 2 - 1

        grids = mf.grids
        grids.build(sort_grids_of_each_atom = True)
        ngrids = grids.coords.shape[0]

        weight_depsilon_drho = cp.random.rand(2, ngrids) * 2 - 1
        weight_depsilon_dnablarho = cp.random.rand(2, 3, ngrids) * 2 - 1
        weight_depsilon_dtau = cp.random.rand(2, ngrids) * 2 - 1

        ni = mf._numint
        xctype = ni._xc_type(mf.xc)

        ni.gdftopt = None
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

        _sorted_mol = opt._sorted_mol
        nao = _sorted_mol.nao
        dm0_sorted = opt.sort_orbitals(dm0, axis=[1,2])
        dm_mask_buf = cp.empty(2 * nao * nao)

        ao_loc_sorted = _sorted_mol.ao_loc
        ao_expand = ao_loc_sorted[1:] - ao_loc_sorted[:-1]
        from pyscf.gto.mole import ATOM_OF
        i_atom_of_aos = numpy.repeat(_sorted_mol._bas[:,ATOM_OF], ao_expand)
        i_atom_of_aos = cp.asarray(i_atom_of_aos, dtype = cp.int32)

        natm = _sorted_mol.natm

        ref_drho_dA_full_response = cp.zeros((natm, 3, 2, 5, ngrids))
        ref_drho_dA_orbital_response = cp.zeros((natm, 3, 2, 5, ngrids))
        ref_d2E_dAdB_full_response = cp.zeros((natm, natm, 3, 3))
        ref_d2E_dAdB_orbital_response = cp.zeros((natm, natm, 3, 3))
        test_drho_dA_full_response = cp.zeros((natm, 3, 2, 5, ngrids))
        test_drho_dA_orbital_response = cp.zeros((natm, 3, 2, 5, ngrids))
        test_d2E_dAdB_full_response = cp.zeros((natm, natm, 3, 3))
        test_d2E_dAdB_orbital_response = cp.zeros((natm, natm, 3, 3))

        g0 = 0
        for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, deriv = 3):
            g1 = g0 + weight.shape[0]

            mu = ao[0]
            dmu_dr = ao[1:4]
            d2mu_dr2 = get_d2mu_dr2(ao)
            d3mu_dr3 = get_d3mu_dr3(ao)

            dm0_masked = take_last2d(dm0_sorted, idx, out = dm_mask_buf)

            i_atom_of_grids = int(grids.atm_idx[g0])
            assert cp.max(cp.abs(grids.atm_idx[g0:g1] - i_atom_of_grids)) == 0 # Guaranteed by grids.build(sort_grids_of_each_atom = True)

            masked_i_atom_of_aos = i_atom_of_aos[idx]

            drho_dA_orbital_response, drho_dA_grid_response = \
                rks_get_drho_dA_sparse(dm0_masked[0], xctype, natm, masked_i_atom_of_aos, i_atom_of_grids, mu, dmu_dr, d2mu_dr2)
            ref_drho_dA_full_response[:,:,0,:,g0:g1] = drho_dA_orbital_response + drho_dA_grid_response
            ref_drho_dA_orbital_response[:,:,0,:,g0:g1] = drho_dA_orbital_response
            drho_dA_orbital_response, drho_dA_grid_response = \
                rks_get_drho_dA_sparse(dm0_masked[1], xctype, natm, masked_i_atom_of_aos, i_atom_of_grids, mu, dmu_dr, d2mu_dr2)
            ref_drho_dA_full_response[:,:,1,:,g0:g1] = drho_dA_orbital_response + drho_dA_grid_response
            ref_drho_dA_orbital_response[:,:,1,:,g0:g1] = drho_dA_orbital_response

            drho_dA_orbital_response, drho_dA_grid_response = \
                uks_get_drho_dA_sparse(dm0_masked, xctype, natm, masked_i_atom_of_aos, i_atom_of_grids, mu, dmu_dr, d2mu_dr2)
            test_drho_dA_full_response[:,:,:,:,g0:g1] = drho_dA_orbital_response + drho_dA_grid_response
            test_drho_dA_orbital_response[:,:,:,:,g0:g1] = drho_dA_orbital_response

            ref_d2E_dAdB_full_response += rks_contract_d2rho_dAdB_sparse(
                dm0_masked[0], xctype, natm, masked_i_atom_of_aos, i_atom_of_grids,
                mu, dmu_dr, d2mu_dr2, d3mu_dr3,
                weight_depsilon_drho[0, g0:g1], weight_depsilon_dnablarho[0, :, g0:g1], weight_depsilon_dtau[0, g0:g1],
            ) + rks_contract_d2rho_dAdB_sparse(
                dm0_masked[1], xctype, natm, masked_i_atom_of_aos, i_atom_of_grids,
                mu, dmu_dr, d2mu_dr2, d3mu_dr3,
                weight_depsilon_drho[1, g0:g1], weight_depsilon_dnablarho[1, :, g0:g1], weight_depsilon_dtau[1, g0:g1],
            )
            test_d2E_dAdB_full_response += uks_contract_d2rho_dAdB_sparse(
                dm0_masked, xctype, natm, masked_i_atom_of_aos, i_atom_of_grids,
                mu, dmu_dr, d2mu_dr2, d3mu_dr3,
                weight_depsilon_drho[:, g0:g1], weight_depsilon_dnablarho[:, :, g0:g1], weight_depsilon_dtau[:, g0:g1],
            )

            ref_d2E_dAdB_orbital_response += rks_contract_d2rho_dAdB_sparse(
                dm0_masked[0], xctype, natm, masked_i_atom_of_aos, i_atom_of_grids,
                mu, dmu_dr, d2mu_dr2, d3mu_dr3,
                weight_depsilon_drho[0, g0:g1], weight_depsilon_dnablarho[0, :, g0:g1], weight_depsilon_dtau[0, g0:g1],
                with_grid_response = False,
            ) + rks_contract_d2rho_dAdB_sparse(
                dm0_masked[1], xctype, natm, masked_i_atom_of_aos, i_atom_of_grids,
                mu, dmu_dr, d2mu_dr2, d3mu_dr3,
                weight_depsilon_drho[1, g0:g1], weight_depsilon_dnablarho[1, :, g0:g1], weight_depsilon_dtau[1, g0:g1],
                with_grid_response = False,
            )
            test_d2E_dAdB_orbital_response += uks_contract_d2rho_dAdB_sparse(
                dm0_masked, xctype, natm, masked_i_atom_of_aos, i_atom_of_grids,
                mu, dmu_dr, d2mu_dr2, d3mu_dr3,
                weight_depsilon_drho[:, g0:g1], weight_depsilon_dnablarho[:, :, g0:g1], weight_depsilon_dtau[:, g0:g1],
                with_grid_response = False,
            )

            g0 = g1
        assert g1 == ngrids

        diff_drhodA_full = cp.max(cp.abs(test_drho_dA_full_response - ref_drho_dA_full_response))
        diff_drhodA_orbital = cp.max(cp.abs(test_drho_dA_orbital_response - ref_drho_dA_orbital_response))
        max_drhodA_full = cp.max(cp.abs(ref_drho_dA_full_response))

        assert diff_drhodA_full < 1e-5 and diff_drhodA_full / max_drhodA_full < 1e-9
        assert diff_drhodA_orbital < 1e-5 and diff_drhodA_orbital / max_drhodA_full < 1e-9

        diff_d2EdAdB_full = cp.max(cp.abs(test_d2E_dAdB_full_response - ref_d2E_dAdB_full_response))
        diff_d2EdAdB_orbital = cp.max(cp.abs(test_d2E_dAdB_orbital_response - ref_d2E_dAdB_orbital_response))
        max_d2EdAdB_full = cp.max(cp.abs(ref_d2E_dAdB_full_response))

        assert diff_d2EdAdB_full < 1e-1 and diff_d2EdAdB_full / max_d2EdAdB_full < 1e-7
        assert diff_d2EdAdB_orbital < 1e-1 and diff_d2EdAdB_orbital / max_d2EdAdB_full < 1e-7


if __name__ == "__main__":
    print("Full Tests for UKS Hessian")
    unittest.main()
