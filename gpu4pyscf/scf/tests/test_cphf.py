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

from functools import reduce
import unittest
import numpy
import cupy
from pyscf import gto, scf, lib
from pyscf import grad, hessian
from pyscf.hessian.uhf import gen_vind as gen_vind_cpu
from pyscf.hessian.rhf import gen_vind as gen_vind_cpu_rhf
from pyscf.scf import ucphf as ucphf_cpu
from pyscf.scf import cphf as cphf_cpu
from gpu4pyscf.scf import ucphf as ucphf_gpu
from gpu4pyscf.scf import cphf as cphf_gpu

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

def _ao2mo(mat, mo_coeff, mocc):
    return numpy.asarray([reduce(numpy.dot, (mo_coeff.T, x, mocc)) for x in mat])

def gen_vind_gpu(mf, mo_coeff, mo_occ):
    v1vo = gen_vind_cpu(mf, mo_coeff, mo_occ)
    return cupy.asarray(v1vo)

class KnownValues(unittest.TestCase):
    def test_cphf(self):
        mf = scf.RHF(mol)
        mf.kernel()
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        fx = gen_vind_cpu_rhf(mf, mo_coeff, mo_occ)
        hessobj = mf.Hessian()
        h1ao = hessobj.make_h1(mo_coeff, mo_occ)
        s1a = -mol.intor('int1e_ipovlp', comp=3)

        mocc = mo_coeff[:, mo_occ > 0]

        h1vo_list = []
        s1vo_list = []
        for i in range(mol.natm):
            h1vo_list.append(_ao2mo(h1ao[i], mo_coeff, mocc))
            s1vo_list.append(_ao2mo(s1a, mo_coeff, mocc))
        h1vo = numpy.vstack(h1vo_list)
        s1vo = numpy.vstack(s1vo_list)
        mo1_cpu, e1_cpu = cphf_cpu.solve(fx, mo_energy, mo_occ, h1vo, s1vo, tol=1e-9)

        def fx_gpu(mo1):
            v1vo = fx(mo1.get())
            return cupy.asarray(v1vo)
            
        mo_energy = cupy.asarray(mo_energy)
        mo_occ = cupy.asarray(mo_occ)
        h1vo = cupy.asarray(h1vo)
        s1vo = cupy.asarray(s1vo)
        mo1_gpu, e1_gpu = cphf_gpu.solve(fx_gpu, mo_energy, mo_occ, h1vo, s1vo, tol=1e-9)

        assert cupy.linalg.norm(mo1_cpu - mo1_gpu.get()) < 1e-6
        assert cupy.linalg.norm(e1_cpu - e1_gpu.get()) < 1e-6

    def test_cphf_with_guess(self):
        # Test GPU CPHF solver with an initial guess (mo10) against CPU default
        mf = scf.RHF(mol)
        mf.kernel()
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        fx = gen_vind_cpu_rhf(mf, mo_coeff, mo_occ)
        hessobj = mf.Hessian()
        h1ao = hessobj.make_h1(mo_coeff, mo_occ)
        s1a = -mol.intor('int1e_ipovlp', comp=3)

        mocc = mo_coeff[:, mo_occ > 0]

        h1vo_list = []
        s1vo_list = []
        for i in range(mol.natm):
            h1vo_list.append(_ao2mo(h1ao[i], mo_coeff, mocc))
            s1vo_list.append(_ao2mo(s1a, mo_coeff, mocc))
        h1vo = numpy.vstack(h1vo_list)
        s1vo = numpy.vstack(s1vo_list)

        # CPU uses default behavior (no mo10)
        mo1_cpu, e1_cpu = cphf_cpu.solve(fx, mo_energy, mo_occ, h1vo, s1vo, tol=1e-9)

        def fx_gpu(mo1):
            v1vo = fx(mo1.get())
            return cupy.asarray(v1vo)
            
        mo_energy = cupy.asarray(mo_energy)
        mo_occ = cupy.asarray(mo_occ)
        h1vo_gpu = cupy.asarray(h1vo)
        s1vo_gpu = cupy.asarray(s1vo)
        
        # Generate a small random initial guess for the GPU solver
        numpy.random.seed(1)
        mo10 = numpy.random.random(h1vo.shape) * 0.1
        mo10_gpu = cupy.asarray(mo10)
        
        mo1_gpu, e1_gpu = cphf_gpu.solve(fx_gpu, mo_energy, mo_occ, h1vo_gpu, s1vo_gpu, tol=1e-9, mo10=mo10_gpu)

        assert cupy.linalg.norm(mo1_cpu - mo1_gpu.get()) < 1e-6
        assert cupy.linalg.norm(e1_cpu - e1_gpu.get()) < 1e-6

    def test_ucphf(self):
        mf = scf.UHF(mol)
        mf.kernel()
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        fx = gen_vind_cpu(mf, mf.mo_coeff, mf.mo_occ)
        hessobj = mf.Hessian()
        h1ao = hessobj.make_h1(mo_coeff, mo_occ)
        s1a = -mol.intor('int1e_ipovlp', comp=3)

        mocca = mo_coeff[0][:,mo_occ[0] > 0]
        moccb = mo_coeff[1][:,mo_occ[1] > 0]

        h1voa = []
        h1vob = []
        s1voa = []
        s1vob = []
        for i in range(mol.natm):
            h1voa.append(_ao2mo(h1ao[0][i], mo_coeff[0], mocca))
            h1vob.append(_ao2mo(h1ao[1][i], mo_coeff[1], moccb))
            s1voa.append(_ao2mo(s1a, mo_coeff[0], mocca))
            s1vob.append(_ao2mo(s1a, mo_coeff[1], moccb))
        h1vo = (numpy.vstack(h1voa), numpy.vstack(h1vob))
        s1vo = (numpy.vstack(s1voa), numpy.vstack(s1vob))
        mo1_cpu, e1_cpu = ucphf_cpu.solve(fx, mo_energy, mo_occ, h1vo, s1vo, tol=1e-9)

        def fx_gpu(mo1):
            v1vo = fx(mo1.get())
            return cupy.asarray(v1vo)
        mo_energy = cupy.asarray(mo_energy)
        mo_occ = cupy.asarray(mo_occ)
        h1vo = cupy.asarray(h1vo)
        s1vo = cupy.asarray(s1vo)
        mo1_gpu, e1_gpu = ucphf_gpu.solve(fx_gpu, mo_energy, mo_occ, h1vo, s1vo, tol=1e-9)

        assert cupy.linalg.norm(mo1_cpu[0] - mo1_gpu[0].get()) < 1e-6
        assert cupy.linalg.norm(mo1_cpu[1] - mo1_gpu[1].get()) < 1e-6
        assert cupy.linalg.norm(e1_cpu[0] - e1_gpu[0].get()) < 1e-6
        assert cupy.linalg.norm(e1_cpu[1] - e1_gpu[1].get()) < 1e-6

if __name__ == "__main__":
    print("Full Tests for CPHF/UCPHF")
    unittest.main()
