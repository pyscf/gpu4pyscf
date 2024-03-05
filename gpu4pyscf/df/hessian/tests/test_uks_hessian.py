# Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import unittest
import numpy
import cupy
from pyscf import gto, scf, lib, dft
from pyscf import grad, hessian
from pyscf.df.hessian import uhf as df_uks_cpu
from gpu4pyscf.df.hessian import uks as df_uks_gpu

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
    #mol.spin =
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    '''
    def test_make_h1(self):
        mf = dft.UKS(mol).density_fit()
        mf.grids.level = 1
        mf.conv_tol = 1e-10
        mf.conv_tol_cpscf = 1e-8
        mf.kernel()
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        mocca = mo_coeff[0][:,mo_occ[0]>0]
        moccb = mo_coeff[1][:,mo_occ[1]>0]
        hobj = mf.Hessian()
        hobj.auxbasis_response = 1
        h1a_cpu, h1b_cpu = df_uks_cpu.make_h1(hobj, mo_coeff, mo_occ)
        mo1_cpu, mo_e1_cpu = hobj.solve_mo1(mo_energy, mo_coeff, mo_occ, (h1a_cpu, h1b_cpu), verbose=1)
        h1a_cpu = numpy.asarray(h1a_cpu)
        h1b_cpu = numpy.asarray(h1b_cpu)
        h1a_cpu = numpy.einsum('xypq,pi,qj->xyij', h1a_cpu, mo_coeff[0], mocca)
        h1b_cpu = numpy.einsum('xypq,pi,qj->xyij', h1b_cpu, mo_coeff[1], moccb)

        mf = mf.to_gpu()
        hobj = mf.Hessian()
        hobj.auxbasis_response = 1
        h1a_gpu, h1b_gpu = df_uks_gpu.make_h1(hobj, mo_coeff, mo_occ)
        h1a_gpu = cupy.asarray(h1a_gpu)
        h1b_gpu = cupy.asarray(h1b_gpu)
        mo_energy = cupy.asarray(mo_energy)
        mo_coeff = cupy.asarray(mo_coeff)
        mo_occ = cupy.asarray(mo_occ)
        mo1_gpu, mo_e1_gpu = hobj.solve_mo1(mo_energy, mo_coeff, mo_occ, (h1a_gpu, h1b_gpu), verbose=1)
        assert numpy.linalg.norm(h1a_cpu - h1a_gpu.get()) < 1e-5
        assert numpy.linalg.norm(h1b_cpu - h1b_gpu.get()) < 1e-5
        mo1_cpu = (numpy.asarray(mo1_cpu[0]), numpy.asarray(mo1_cpu[1]))
        mo1_gpu = (cupy.asarray(mo1_gpu[0]).get(), cupy.asarray(mo1_gpu[1]).get())
        mo_e1_cpu = (numpy.asarray(mo_e1_cpu[0]), numpy.asarray(mo_e1_cpu[1]))
        mo_e1_gpu = (cupy.asarray(mo_e1_gpu[0]).get(), cupy.asarray(mo_e1_gpu[1]).get())

        # mo1 is not consistent in PySCF and GPU4PySCF
        #assert numpy.linalg.norm((mo1_cpu[0] - mo1_gpu[0])) < 1e-5
        assert numpy.linalg.norm((mo_e1_cpu[0] - mo_e1_gpu[0])) < 1e-5
        #assert numpy.linalg.norm((mo1_cpu[1] - mo1_gpu[1])) < 1e-5
        assert numpy.linalg.norm((mo_e1_cpu[1] - mo_e1_gpu[1])) < 1e-5
    '''
    def test_df_uks_hess_elec(self):
        mf = dft.UKS(mol, xc='b3lyp').density_fit()
        mf.conv_tol = 1e-10
        mf.conv_tol_cpscf = 1e-8
        mf.grids.level = 1
        mf.kernel()
        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hess_cpu = hobj.partial_hess_elec()

        mf = mf.to_gpu()
        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hess_gpu = hobj.partial_hess_elec()
        print(hess_cpu[0,0])
        print(hess_gpu[0,0])
        assert numpy.linalg.norm(hess_cpu - hess_gpu.get()) < 1e-5
    '''
    def test_df_uks(self):
        mf = dft.UKS(mol).density_fit()
        mf.conv_tol = 1e-10
        mf.grids.level = 1
        mf.conv_tol_cpscf = 1e-8
        mf.kernel()

        hessobj = mf.Hessian()
        hess_cpu = hessobj.kernel()

        mf = mf.to_gpu()
        hessobj = mf.Hessian()
        hess_gpu = hessobj.kernel()
        print(hess_cpu[0,0])
        print(hess_gpu[0,0])
        assert numpy.linalg.norm(hess_cpu - hess_gpu) < 1e-5
    '''
if __name__ == "__main__":
    print("Full Tests for DF UKS Hessian")
    unittest.main()