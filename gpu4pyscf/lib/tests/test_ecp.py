# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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
import scipy
import cupy
import ctypes
from pyscf import gto
from pyscf.dft import radi
from gpu4pyscf.lib.cupy_helper import load_library

libecp = load_library('libecp')
libecp_cpu = gto.moleintor.libcgto

def ang_nuc_part(l, rij):
    omega_xyz = numpy.empty((l+1)*(l+2)//2)
    k = 0
    for i1 in reversed(range(l+1)):
        for j1 in reversed(range(l-i1+1)):
            k1 = l - i1 - j1
            omega_xyz[k] = rij[0]**i1 * rij[1]**j1 * rij[2]**k1
            k += 1
    if l == 0:
        return omega_xyz * 0.282094791773878143
    elif l == 1:
        return omega_xyz * 0.488602511902919921
    else:
        omega = numpy.empty((2*l+1))
        fc2s = libecp_cpu.CINTc2s_ket_sph
        fc2s(omega.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(1),
             omega_xyz.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(l))
        return omega

class KnownValues(unittest.TestCase):

    def test_bessel(self):
        n = 5
        rs = radi.gauss_chebyshev(n)[0]
        bessel1 = cupy.empty([n, 8])
        order = numpy.arange(8)
        rs = rs.reshape(n,-1)
        bessel0 = scipy.special.spherical_in(order, rs) * numpy.exp(-rs)
        rs = cupy.asarray(rs)
        libecp.ECPsph_ine(ctypes.cast(bessel1.data.ptr, ctypes.c_void_p),
                          ctypes.c_int(7),
                          ctypes.cast(rs.data.ptr, ctypes.c_void_p),
                          ctypes.c_int(n))
        self.assertTrue(numpy.allclose(bessel0, bessel1))

    def test_ang_nuc_part(self):
        n = 10
        x = numpy.random.rand(n,3)
        x_gpu = cupy.asarray(x)
        for l in range(4):
            omega_gpu = cupy.empty([n, 2*l+1])
            libecp.ECPang_nuc_part(
                ctypes.cast(omega_gpu.data.ptr, ctypes.c_void_p),
                ctypes.cast(x_gpu.data.ptr, ctypes.c_void_p),
                ctypes.c_int(n),
                ctypes.c_int(l))
            omega_cpu = numpy.empty([n, 2*l+1])
            for i in range(n):
                omega_cpu[i] = ang_nuc_part(l, x[i])
            assert numpy.linalg.norm(omega_cpu - omega_gpu.get())

if __name__ == "__main__":
    print("Full tests for ECP module")
    unittest.main()
