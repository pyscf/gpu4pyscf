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
#

import unittest
import numpy
import cupy
from pyscf import gto, scf
from pyscf.df.hessian import rhf as df_rhf_cpu
from pyscf.hessian import rhf as rhf_cpu
from gpu4pyscf.df.hessian import rhf as df_rhf_gpu
from gpu4pyscf.hessian import rhf as rhf_gpu

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.verbose = 1
    mol.output = '/dev/null'
    mol.atom.extend([
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ])
    mol.basis = 'sto3g'
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    def test_gen_vind(self):
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-10
        mf.conv_tol_cpscf = 1e-8
        mf.kernel()
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ

        nao, nmo = mo_coeff.shape
        mocc = mo_coeff[:,mo_occ>0]
        nocc = mocc.shape[1]

        fx_cpu = rhf_cpu.gen_vind(mf, mo_coeff, mo_occ)
        mo1 = numpy.random.rand(100, nmo*nocc)
        v1vo_cpu = fx_cpu(mo1).reshape(-1,nmo*nocc)

        mf = mf.to_gpu()
        hessobj = mf.Hessian()
        fx_gpu = hessobj.gen_vind(mo_coeff, mo_occ)
        mo1 = cupy.asarray(mo1)
        v1vo_gpu = fx_gpu(mo1)
        assert numpy.linalg.norm(v1vo_cpu - v1vo_gpu.get()) < 1e-8

    def test_partial_hess_elec(self):
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-10
        mf.conv_tol_cpscf = 1e-8
        mf.kernel()
        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        e1_cpu, ej_cpu, ek_cpu = df_rhf_cpu._partial_hess_ejk(hobj)

        mf = mf.to_gpu()
        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        e1_gpu, ej_gpu, ek_gpu = df_rhf_gpu._partial_hess_ejk(hobj)
        assert numpy.linalg.norm(e1_cpu - e1_gpu.get()) < 1e-5
        assert numpy.linalg.norm(ej_cpu - ej_gpu.get()) < 1e-5
        assert numpy.linalg.norm(ek_cpu - ek_gpu.get()) < 1e-5

    def test_make_h1(self):
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-10
        mf.conv_tol_cpscf = 1e-8
        mf.kernel()
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        mocc = mo_coeff[:,mo_occ>0]
        hobj = mf.Hessian()
        hobj.auxbasis_response = 1
        h1_cpu = df_rhf_cpu.make_h1(hobj, mo_coeff, mo_occ)
        mo1_cpu, mo_e1_cpu = hobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1_cpu, verbose=1)
        h1_cpu = numpy.asarray(h1_cpu)
        h1_cpu = numpy.einsum('xypq,pi,qj->xyij', h1_cpu, mo_coeff, mocc)

        mf = mf.to_gpu()
        mf.conv_tol = 1e-10
        mf.conv_tol_cpscf = 1e-8
        hobj = mf.Hessian()
        hobj.auxbasis_response = 1
        mo_occ = cupy.asarray(mo_occ)
        h1_gpu = df_rhf_gpu.make_h1(hobj, mo_coeff, mo_occ)
        h1_gpu = cupy.asarray(h1_gpu)
        mo_energy = cupy.asarray(mo_energy)
        mo_coeff = cupy.asarray(mo_coeff)
        fx = hobj.gen_vind(mo_coeff, mo_occ)
        mo1_gpu, mo_e1_gpu = hobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1_gpu, fx, verbose=1)
        assert numpy.linalg.norm(h1_cpu - h1_gpu.get()) < 1e-5
        assert numpy.linalg.norm((mo_e1_cpu - mo_e1_gpu)) < 1e-4

    def test_df_rhf_hess_elec(self):
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-10
        mf.conv_tol_cpscf = 1e-8
        mf.kernel()
        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hess_cpu = hobj.hess_elec()

        mf = mf.to_gpu()
        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hess_gpu = hobj.hess_elec()
        assert numpy.linalg.norm(hess_cpu - hess_gpu.get()) < 1e-5

    def test_df_rhf_hessian(self):
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-10
        mf.conv_tol_cpscf = 1e-8
        mf.kernel()
        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hess_cpu = hobj.kernel()
        mf = mf.to_gpu()
        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hess_gpu = hobj.kernel()
        assert numpy.linalg.norm(hess_cpu - hess_gpu) < 1e-5

if __name__ == "__main__":
    print("Full Tests for DF RHF Hessian")
    unittest.main()
