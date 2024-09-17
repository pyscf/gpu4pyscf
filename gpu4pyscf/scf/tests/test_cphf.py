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

from functools import reduce
import unittest
import numpy
import cupy
from pyscf import gto, scf, lib
from pyscf import grad, hessian
from pyscf.hessian.uhf import gen_vind as gen_vind_cpu
from pyscf.scf import ucphf as ucphf_cpu
from gpu4pyscf.scf import ucphf as ucphf_gpu

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
    print("Full Tests for Unrestricted CPHF")
    unittest.main()