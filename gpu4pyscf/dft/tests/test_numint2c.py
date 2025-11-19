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
import pyscf
import cupy
from pyscf import lib, scf
from pyscf.dft.numint2c import NumInt2C as pyscf_numint2c
from pyscf.dft.numint import NumInt as pyscf_numint
from gpu4pyscf.dft import Grids
from gpu4pyscf.dft import numint2c
from gpu4pyscf.dft import numint
from pyscf.dft import numint2c as pyscf_numint2c_file
from gpu4pyscf.dft.numint2c import NumInt2C
from gpu4pyscf.dft.numint import NumInt
from gpu4pyscf import dft
from gpu4pyscf.dft import gen_grid
try:
    import mcfun
except ImportError:
    mcfun = None

def setUpModule():
    global mol, grids_cpu, grids_gpu, dm, dm0, dm1, mo_occ, mo_coeff
    mol = pyscf.M(
        atom = '''
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161''',
        basis = 'ccpvdz',
        charge = 1,
        spin = 1,  # = 2S = spin_up - spin_down
        output = '/dev/null'
        )

    np.random.seed(2)
    mf = scf.GHF(mol)
    mf.kernel()
    dm1 = mf.make_rdm1().copy()
    dm = dm1
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    dm0 = (mo_coeff*mo_occ).dot(mo_coeff.T)

    grids_gpu = Grids(mol)
    grids_gpu.level = 1
    grids_gpu.build()

    grids_cpu = grids_gpu.to_cpu()
    grids_cpu.weights = cupy.asnumpy(grids_gpu.weights)
    grids_cpu.coords = cupy.asnumpy(grids_gpu.coords)

def tearDownModule():
    global mol, grids_cpu, grids_gpu
    mol.stdout.close()
    del mol, grids_cpu, grids_gpu

LDA = 'LDA_C_VWN'
GGA_PBE = 'GGA_C_PBE'
MGGA_M06 = 'MGGA_C_M06'

class KnownValues(unittest.TestCase):

    def test_eval_rho(self):
        np.random.seed(1)
        dm = np.random.random(dm0.shape) + np.random.random(dm0.shape)*1.0j
        dm = dm + dm.conj().T
        ni_gpu = NumInt2C()
        ni_cpu = pyscf_numint2c()
        for xctype in ('LDA', 'GGA', 'MGGA'):
            deriv = 1
            if xctype == 'LDA':
                deriv = 0
            ao_gpu = ni_gpu.eval_ao(mol, grids_gpu.coords, deriv=deriv, transpose=False)
            ao_cpu = ni_cpu.eval_ao(mol, grids_cpu.coords, deriv=deriv)
            
            rho = ni_gpu.eval_rho(mol, ao_gpu, dm, xctype=xctype, hermi=0, with_lapl=False)
            ref = ni_cpu.eval_rho(mol, ao_cpu, dm, xctype=xctype, hermi=0, with_lapl=False)
            self.assertAlmostEqual(abs(rho[...,:grids_cpu.size].get() - ref).max(), 0, 10)

            rho = ni_gpu.eval_rho(mol, ao_gpu, dm0, xctype=xctype, hermi=1, with_lapl=False)
            ref = ni_cpu.eval_rho(mol, ao_cpu, dm0, xctype=xctype, hermi=1, with_lapl=False)
            self.assertAlmostEqual(abs(rho[...,:grids_cpu.size].get() - ref).max(), 0, 10)

    def test_eval_rho2(self):
        np.random.seed(1)
        mo_coeff_test = np.random.random(mo_coeff.shape) + np.random.random(mo_coeff.shape)*1.0j
        ni_gpu = NumInt2C()
        ni_gpu.collinear='m'
        ni_cpu = pyscf_numint2c()
        ni_cpu.collinear='m'
        for xctype in ('LDA', 'GGA', 'MGGA'):
            deriv = 1
            if xctype == 'LDA':
                deriv = 0
            ao_gpu = ni_gpu.eval_ao(mol, grids_gpu.coords, deriv=deriv, transpose=False)
            ao_cpu = ni_cpu.eval_ao(mol, grids_cpu.coords, deriv=deriv)
            
            rho = ni_gpu.eval_rho2(mol, ao_gpu, mo_coeff_test, mo_occ, xctype=xctype, with_lapl=False)
            ref = ni_cpu.eval_rho2(mol, ao_cpu, mo_coeff_test, mo_occ, xctype=xctype, with_lapl=False)
            self.assertAlmostEqual(abs(rho[...,:grids_cpu.size].get() - ref).max(), 0, 10)

    def test_get_rho(self):
        ni_gpu = NumInt2C()
        ni_gpu.collinear='m'

        np.random.seed(1)
        ni_gpu_1c = NumInt()
        dm_test = np.random.random(dm0.shape) + np.random.random(dm0.shape)*1.0j
        dm_test = dm_test + dm_test.T.conj()

        n2c = dm_test.shape[0]
        nao = n2c//2
        dm_1c_test = dm_test[:nao,:nao] + dm_test[nao:,nao:]
        rho_gpu = ni_gpu.get_rho(mol, dm_test, grids_gpu)
        rho_1c_gpu = ni_gpu_1c.get_rho(mol, dm_1c_test.real, grids_gpu)
        self.assertAlmostEqual(abs(rho_gpu.get() - rho_1c_gpu.get()).max(), 0, 10)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_eval_xc_eff(self):
        ni_gpu = NumInt2C()
        ni_gpu.collinear='m'
        ni_cpu = pyscf_numint2c()
        ni_cpu.collinear='m'
        np.random.seed(1)
        dm = dm0*1.0 + dm0 * 0.1j
        dm = dm + dm.T.conj()
        for xc_code in (LDA, GGA_PBE, MGGA_M06):
            n_gpu, exc_gpu, vmat_gpu = ni_gpu.nr_vxc(mol, grids_gpu, xc_code, dm)
            n_cpu, exc_cpu, vmat_cpu = ni_cpu.nr_vxc(mol, grids_cpu, xc_code, dm)
            self.assertAlmostEqual(abs(n_gpu.get() - n_cpu).max(), 0, 10)
            self.assertAlmostEqual(abs(exc_gpu.get() - exc_cpu).max(), 0, 10)
            self.assertAlmostEqual(abs(vmat_gpu.get() - vmat_cpu).max(), 0, 10)

    def test_eval_xc_eff_fp(self):
        ni_gpu = NumInt2C()
        ni_gpu.collinear='m'
        np.random.seed(1)
        dm = dm0*1.0 + dm0 * 0.1j
        dm = dm + dm.T.conj()

        n_gpu, exc_gpu, vmat_gpu = ni_gpu.nr_vxc(mol, grids_gpu, LDA, dm)
        self.assertAlmostEqual(abs(n_gpu.get() - 17.9999262659497).max(), 0, 10)
        self.assertAlmostEqual(abs(exc_gpu.get() - -1.310501342423071).max(), 0, 10)
        self.assertAlmostEqual(abs(lib.fp(vmat_gpu.get()) 
            - (-0.20448306536588537-6.75460139752253e-21j)).max(), 0, 10)

        n_gpu, exc_gpu, vmat_gpu = ni_gpu.nr_vxc(mol, grids_gpu, GGA_PBE, dm)
        self.assertAlmostEqual(abs(n_gpu.get() - 17.9999262659497).max(), 0, 10)
        self.assertAlmostEqual(abs(exc_gpu.get() - -0.7237150857425112).max(), 0, 10)
        self.assertAlmostEqual(abs(lib.fp(vmat_gpu.get()) 
            - (-0.05446425800187435-4.486282070082083e-21j)).max(), 0, 10)

        n_gpu, exc_gpu, vmat_gpu = ni_gpu.nr_vxc(mol, grids_gpu, MGGA_M06, dm)
        self.assertAlmostEqual(abs(n_gpu.get() - 17.9999262659497).max(), 0, 10)
        self.assertAlmostEqual(abs(exc_gpu.get() - -0.7703982586705045).max(), 0, 10)
        self.assertAlmostEqual(abs(lib.fp(vmat_gpu.get()) 
            - (-0.18688247306409317+7.50400133342109e-20j)).max(), 0, 10)

    def test_mcol_lda_vxc_mat(self):
        xc_code = 'lda,'

        nao = mol.nao
        n2c = nao * 2
        ao_loc = mol.ao_loc
        np.random.seed(12)
        dm = np.random.rand(n2c, n2c) * .001 + np.random.rand(n2c, n2c) * .0001j
        dm += np.eye(n2c)
        dm = dm + dm.T.conj()
        ngrids = 8
        coords = np.random.rand(ngrids,3)
        weight = np.random.rand(ngrids)

        ao = numint.eval_ao(mol, coords, deriv=0, transpose=False)
        rho = numint2c.eval_rho(mol, ao, dm, xctype='LDA', hermi=1, with_lapl=False)
        
        # --- GPU Calculations (Always Run) ---
        ni_GPU = NumInt2C()
        ni_GPU.collinear = 'mcol'
        eval_xc_GPU = ni_GPU.mcfun_eval_xc_adapter(xc_code)
        vxc_gpu = eval_xc_GPU(xc_code, rho, deriv=1, xctype='LDA')[1]
        mask_gpu = cupy.arange(mol.nbas)
        shls_slice = (0, mol.nbas)
        
        v0_gpu = numint2c._mcol_lda_vxc_mat(mol, ao, cupy.asarray(weight), rho, vxc_gpu.copy(), 
            mask_gpu, shls_slice, ao_loc, 0, assemble_spin_components=True)
        v1_gpu = numint2c._mcol_lda_vxc_mat(mol, ao, cupy.asarray(weight), rho, vxc_gpu.copy(), 
            mask_gpu, shls_slice, ao_loc, 1, assemble_spin_components=True)
        v1_gpu = v1_gpu + v1_gpu.conj().T

        # --- Assertions (Always Run) ---
        # Fingerprint checks (from _fp test)
        self.assertAlmostEqual(abs(lib.fp(v0_gpu.get()) - 
            (-9.596802359283691+2.969010922568367e-05j)).max(), 0, 13)
        self.assertAlmostEqual(abs(lib.fp(v1_gpu.get()) - 
            (-9.596802359283691+2.969010922568367e-05j)).max(), 0, 13)
        # Internal consistency check
        self.assertAlmostEqual(abs(v0_gpu.get() - v1_gpu.get()).max(), 0, 13)

        # --- CPU Comparison (Conditional) ---
        if mcfun is not None:
            ni_CPU = pyscf_numint2c()
            ni_CPU.collinear = 'mcol'
            eval_xc_cpu = ni_CPU.mcfun_eval_xc_adapter(xc_code)
            vxc_cpu = eval_xc_cpu(xc_code, rho.get(), deriv=1)[1]
            mask = np.ones((8, mol.nbas), dtype=np.uint8)
            
            v0_cpu = pyscf_numint2c_file._mcol_lda_vxc_mat(mol, ao.transpose(1,0).get(), weight, 
                rho.get(), vxc_cpu.copy(), mask, shls_slice, ao_loc, 0)
            
            # CPU vs GPU check
            self.assertAlmostEqual(abs(v0_gpu.get() - v0_cpu).max(), 0, 13)

    def test_mcol_gga_vxc_mat(self):
        xc_code = 'pbe,'

        nao = mol.nao
        n2c = nao * 2
        ao_loc = mol.ao_loc
        np.random.seed(12)
        dm = np.random.rand(n2c, n2c) * .001 + np.random.rand(n2c, n2c) * .001j
        dm += np.eye(n2c)
        dm = dm + dm.T.conj()
        ngrids = 8
        coords = np.random.rand(ngrids,3)
        weight = np.random.rand(ngrids)

        ao = numint.eval_ao(mol, coords, deriv=1, transpose=False)
        rho = numint2c.eval_rho(mol, ao, dm, xctype='GGA', hermi=1, with_lapl=False)

        # --- GPU Calculations (Always Run) ---
        ni_GPU = NumInt2C()
        ni_GPU.collinear = 'mcol'
        eval_xc_GPU = ni_GPU.mcfun_eval_xc_adapter(xc_code)
        vxc_gpu = eval_xc_GPU(xc_code, rho, deriv=1, xctype='GGA')[1]
        mask_gpu = cupy.arange(mol.nbas)
        shls_slice = (0, mol.nbas)
        
        v0_gpu = numint2c._mcol_gga_vxc_mat(mol, ao, cupy.asarray(weight), rho, vxc_gpu.copy(), 
            mask_gpu, shls_slice, ao_loc, 0, assemble_spin_components=True)
        v1_gpu = numint2c._mcol_gga_vxc_mat(mol, ao, cupy.asarray(weight), rho, vxc_gpu.copy(), 
            mask_gpu, shls_slice, ao_loc, 1, assemble_spin_components=True)
        v1_gpu = v1_gpu + v1_gpu.conj().T

        # --- Assertions (Always Run) ---
        # Fingerprint checks (from _fp test)
        self.assertAlmostEqual(abs(lib.fp(v1_gpu.get()) - 
            (-9.624260408900755+0.0003122100947141664j)).max(), 0, 13)
        self.assertAlmostEqual(abs(lib.fp(v0_gpu.get()) - 
            (-9.624260408900755+0.0003122100947141664j)).max(), 0, 13)
        # Internal consistency check
        self.assertAlmostEqual(abs(v0_gpu.get() - v1_gpu.get()).max(), 0, 13)

        # --- CPU Comparison (Conditional) ---
        if mcfun is not None:
            ni_CPU = pyscf_numint2c()
            ni_CPU.collinear = 'mcol'
            eval_xc_cpu = ni_CPU.mcfun_eval_xc_adapter(xc_code)
            vxc_cpu = eval_xc_cpu(xc_code, rho.get(), deriv=1)[1]
            mask = np.ones((8, mol.nbas), dtype=np.uint8)

            v0_cpu = pyscf_numint2c_file._mcol_gga_vxc_mat(mol, ao.transpose(0,2,1).get(), weight, 
                rho.get(), vxc_cpu.copy(), mask, shls_slice, ao_loc, 0)
            
            # CPU vs GPU check
            self.assertAlmostEqual(abs(v0_gpu.get() - v0_cpu).max(), 0, 13)

    def test_mcol_mgga_vxc_mat(self):
        xc_code = 'tpss'

        nao = mol.nao
        n2c = nao * 2
        ao_loc = mol.ao_loc
        np.random.seed(12)
        dm = np.random.rand(n2c, n2c) * .001
        dm += np.eye(n2c)
        dm = dm + dm.T.conj()
        ngrids = 8
        coords = np.random.rand(ngrids,3)
        weight = np.random.rand(ngrids)

        ao = numint.eval_ao(mol, coords, deriv=1, transpose=False)
        rho = numint2c.eval_rho(mol, ao, dm, xctype='MGGA', hermi=1, with_lapl=False)

        # --- GPU Calculations (Always Run) ---
        ni_GPU = NumInt2C()
        ni_GPU.collinear = 'mcol'
        eval_xc_GPU = ni_GPU.mcfun_eval_xc_adapter(xc_code)
        vxc_gpu = eval_xc_GPU(xc_code, rho, deriv=1, xctype='MGGA')[1]
        mask_gpu = cupy.arange(mol.nbas)
        shls_slice = (0, mol.nbas)
        
        v0_gpu = numint2c._mcol_mgga_vxc_mat(mol, ao, cupy.asarray(weight), rho, vxc_gpu.copy(), 
            mask_gpu, shls_slice, ao_loc, 0, assemble_spin_components=True)
        v1_gpu = numint2c._mcol_mgga_vxc_mat(mol, ao, cupy.asarray(weight), rho, vxc_gpu.copy(), 
            mask_gpu, shls_slice, ao_loc, 1, assemble_spin_components=True)
        v1_gpu = v1_gpu + v1_gpu.conj().T

        # --- Assertions (Always Run) ---
        # Fingerprint checks (from _fp test) <-- UPDATED
        self.assertAlmostEqual(abs(lib.fp(v1_gpu.get()) 
            - (-11.359687631112195+6.13828986953566e-22j)).max(), 0, 13)
        self.assertAlmostEqual(abs(lib.fp(v0_gpu.get()) 
            - (-11.359687631112195+6.13828986953566e-22j)).max(), 0, 13)
        
        # Internal consistency check
        self.assertAlmostEqual(abs(v0_gpu.get() - v1_gpu.get()).max(), 0, 13)
        
        # --- CPU Comparison (Conditional) ---
        if mcfun is not None:
            ni_CPU = pyscf_numint2c()
            ni_CPU.collinear = 'mcol'
            eval_xc_cpu = ni_CPU.mcfun_eval_xc_adapter(xc_code)
            vxc_cpu = eval_xc_cpu(xc_code, rho.get(), deriv=1)[1]
            mask = np.ones((8, mol.nbas), dtype=np.uint8)

            v0_cpu = pyscf_numint2c_file._mcol_mgga_vxc_mat(mol, ao.transpose(0,2,1).get(), weight, 
                rho.get(), vxc_cpu.copy(), mask, shls_slice, ao_loc, 0)
            
            # CPU vs GPU check
            self.assertAlmostEqual(abs(v0_gpu.get() - v0_cpu).max(), 0, 13)


if __name__ == "__main__":
    print("Full Tests for dft numint2c")
    unittest.main()

