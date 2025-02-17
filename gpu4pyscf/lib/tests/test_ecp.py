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
from pyscf import gto, lib
from pyscf.dft import radi
from gpu4pyscf.lib.cupy_helper import load_library

libecp = load_library('libgecp')
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

def type1_rad_part(lmax, k, aij, ur, rs):
    rad_all = numpy.empty((lmax+1,lmax+1))

    order = numpy.arange(lmax+1).reshape(lmax+1,-1)
    bessel_val = scipy.special.spherical_in(order, k*rs) * numpy.exp(-k*rs)

    ur_base = numpy.exp(k**2/(4*aij)) * ur * numpy.exp(-aij*(rs-k/(2*aij))**2)
    idx = abs(ur_base) > 1e-80
    for lab in range(lmax+1):
        val = ur_base[idx] * rs[idx]**lab
        for l in range(lmax+1):
            if (lab+l) % 2 == 0:
                val1 = val * bessel_val[l,idx]
                rad_all[lab,l] = val1.sum()
            else:
                rad_all[lab,l] = 0
    return rad_all

mol = gto.M(atom='''
            Na 0.5 0.5 0.
            H  0.  1.  1.
            ''',
            basis={'Na':'lanl2dz',
                   'H':[[0,[1.21,1.],[.521,1.]],
                        [1,[3.12,1.],[.512,1.]],
                        [2,[2.54,1.],[.554,1.]],
                        [3,[0.98,1.],[.598,1.]],
                        [4,[0.79,1.],[.579,1.]]]},
            ecp = {'Na': gto.basis.parse_ecp('''
Na nelec 10
Na ul
0      2.0000000              6.0000000
1    175.5502590            -10.0000000
2      2.3365719             -6.0637782
2      0.7799867             -0.7299393
Na S
0    243.3605846              3.0000000
#1     41.5764759             36.2847626
#2     13.2649167             72.9304880
#2      0.9764209              6.0123861
#Na P
#0   1257.2650682              5.0000000
#1    189.6248810            117.4495683
#2     54.5247759            423.3986704
#2      0.9461106              7.1241813
''')})


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
    '''
    def test_ang_nuc_part(self):
        n = 10
        x = numpy.random.rand(n,3)
        x_gpu = cupy.asarray(x)
        fn = libecp.ECPang_nuc_part
        for l in range(4):
            omega_gpu = cupy.empty([n, 2*l+1])
            fn(ctypes.cast(omega_gpu.data.ptr, ctypes.c_void_p),
               ctypes.cast(x_gpu.data.ptr, ctypes.c_void_p),
               ctypes.c_int(n),
               ctypes.c_int(l))
            omega_cpu = numpy.empty([n, 2*l+1])
            for i in range(n):
                omega_cpu[i] = ang_nuc_part(l, x[i])
            print(omega_cpu.shape)
            print(omega_gpu.shape)
            assert numpy.linalg.norm(omega_cpu - omega_gpu.get()) < 1e-10
    '''
    def test_rad_part(self):
        rs, ws = radi.gauss_chebyshev(128)
        cache = numpy.empty(100000)

        for ish in range(len(mol._ecpbas)):
            ur1 = numpy.zeros_like(rs)

            libecp_cpu.ECPrad_part(
                ur1.ctypes.data_as(ctypes.c_void_p),
                rs.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(0), ctypes.c_int(len(rs)), ctypes.c_int(1),
                (ctypes.c_int*2)(ish, ish+1),
                mol._ecpbas.ctypes.data_as(ctypes.c_void_p),
                mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
                mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
                mol._env.ctypes.data_as(ctypes.c_void_p),
                lib.c_null_ptr(), cache.ctypes.data_as(ctypes.c_void_p))
            ur1 *= ws
            ecpbas = cupy.asarray(mol._ecpbas)
            env = cupy.asarray(mol._env)
            ur0 = cupy.zeros_like(rs)
            rs0 = cupy.asarray(rs)
            ws0 = cupy.asarray(ws)
            libecp.ECPrad_part(
                ctypes.c_int(ish),
                ctypes.cast(ecpbas.data.ptr, ctypes.c_void_p),
                ctypes.cast(env.data.ptr, ctypes.c_void_p),
                ctypes.cast(rs0.data.ptr, ctypes.c_void_p),
                ctypes.cast(ws0.data.ptr, ctypes.c_void_p),
                ctypes.cast(ur0.data.ptr, ctypes.c_void_p),
                ctypes.c_int(128),
            )
            assert numpy.linalg.norm(ur0.get() - ur1) < 1e-10

    def test_type1_rad_part(self):
        k = 1.621
        aij = .792
        rs, ws = radi.gauss_chebyshev(128)
        ur = numpy.random.rand(128)#rad_part(mol, mol._ecpbas, rs) * ws
        cache = numpy.empty(100000)
        for l in range(4):
            rad_all1 = numpy.zeros([l+1,l+1])
            libecp_cpu.type1_rad_part(
                rad_all1.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(l),
                ctypes.c_double(k), ctypes.c_double(aij),
                ur.ctypes.data_as(ctypes.c_void_p),
                rs.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(len(rs)), ctypes.c_int(1),
                cache.ctypes.data_as(ctypes.c_void_p))

            ur_gpu = cupy.asarray(ur)
            rad_all0 = cupy.zeros([l+1,l+1])
            libecp.ECPtype1_rad_part(
                ctypes.cast(rad_all0.data.ptr, ctypes.c_void_p),
                ctypes.c_int(l),
                ctypes.c_double(k),
                ctypes.c_double(aij),
                ctypes.cast(ur_gpu.data.ptr, ctypes.c_void_p),
                ctypes.c_int(1))
            assert numpy.linalg.norm(rad_all0.get() - rad_all1) < 1e-10

    def test_type1_rad_ang(self):
        numpy.random.seed(4)
        n = 1
        ri = numpy.random.random(3) - .5
        for l in range(8):
            rad_all = numpy.random.random((l+1,l+1))
            rad_ang1 = numpy.zeros((l+1,l+1,l+1))
            libecp_cpu.type1_rad_ang(
                rad_ang1.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(l),
                ri.ctypes.data_as(ctypes.c_void_p),
                rad_all.ctypes.data_as(ctypes.c_void_p))

            ri_gpu = cupy.asarray(ri)
            rad_all = cupy.asarray(rad_all)
            rad_ang0 = cupy.zeros_like(rad_ang1)
            libecp.ECPtype1_rad_ang(
                ctypes.cast(rad_ang0.data.ptr, ctypes.c_void_p),
                ctypes.c_int(l),
                ctypes.c_int(n),
                ctypes.cast(ri_gpu.data.ptr, ctypes.c_void_p),
                ctypes.c_double(1.0),
                ctypes.cast(rad_all.data.ptr, ctypes.c_void_p))
            assert numpy.linalg.norm(rad_ang1 - rad_ang0.get()) < 1e-7

    def test_type1_cart(self):
        for ish in range(mol.nbas):
            for jsh in range(mol.nbas):
                li = mol.bas_angular(ish)
                lj = mol.bas_angular(jsh)
                if li > lj:
                    continue
                di = (li+1) * (li+2) // 2 * mol.bas_nctr(ish)
                dj = (lj+1) * (lj+2) // 2 * mol.bas_nctr(jsh)
                mat0 = numpy.zeros((di,dj))
                cache = numpy.empty(100000)
                ecpbas = mol._ecpbas.copy()
                shls = (ish, jsh)
                shl_ptr = (ctypes.c_int*2)(*shls)
                null_ptr = lib.c_null_ptr()

                libecp_cpu.ECPtype1_cart(
                    mat0.ctypes.data_as(ctypes.c_void_p),
                    shl_ptr,
                    ecpbas.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(len(mol._ecpbas)),
                    mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
                    mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
                    mol._env.ctypes.data_as(ctypes.c_void_p),
                    null_ptr, cache.ctypes.data_as(ctypes.c_void_p))

                mat1 = cupy.zeros_like(mat0)
                ecpbas0 = ecpbas[ecpbas[:,gto.ANG_OF] < 0]
                ecpbas = cupy.asarray(ecpbas0)
                atm = cupy.asarray(mol._atm)
                bas = cupy.asarray(mol._bas)
                env = cupy.asarray(mol._env)
                ecploc = cupy.asarray([0,len(ecpbas)], dtype=numpy.int32)
                tasks = cupy.asarray([ish,jsh,0], dtype=numpy.int32)
                li = mol.bas_angular(ish)
                lj = mol.bas_angular(jsh)

                libecp.ECPtype1_cart(
                    ctypes.cast(mat1.data.ptr, ctypes.c_void_p),
                    ctypes.cast(tasks.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(1),
                    ctypes.cast(ecpbas.data.ptr, ctypes.c_void_p),
                    ctypes.cast(ecploc.data.ptr, ctypes.c_void_p),
                    ctypes.cast(atm.data.ptr, ctypes.c_void_p),
                    ctypes.cast(bas.data.ptr, ctypes.c_void_p),
                    ctypes.cast(env.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(li),
                    ctypes.c_int(lj))
                print(li,lj)
                print(mat0[:3,:3])
                print(mat1[:3,:3])
                assert numpy.linalg.norm(mat1.get() - mat0) < 1e-10
    '''
    def test_type2_rad_part(self):
        rc = .8712
        nrs = 128
        rs, _ = radi.gauss_chebyshev(nrs)
        cache = numpy.empty(100000)
        for ish in range(mol.nbas):
            npi = mol.bas_nprim(ish)
            l = mol.bas_angular(ish)
            ai = mol.bas_exp(ish)
            ci = mol._libcint_ctr_coeff(ish)
            for lc in range(5):
                facs1 = numpy.zeros((nrs, l+lc+1))
                libecp_cpu.type2_facs_rad(
                    facs1.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(ish), ctypes.c_int(lc),
                    ctypes.c_double(rc),
                    rs.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(len(rs)), ctypes.c_int(1),
                    mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
                    mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
                    mol._env.ctypes.data_as(ctypes.c_void_p),
                    cache.ctypes.data_as(ctypes.c_void_p))

                facs0 = cupy.zeros_like(facs1)
                ai = cupy.asarray(ai)
                ci = cupy.asarray(ci)
                libecp.ECPtype2_facs_rad(
                    ctypes.cast(facs0.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(l),
                    ctypes.c_int(lc),
                    ctypes.c_int(npi),
                    ctypes.c_double(rc),
                    ctypes.cast(ci.data.ptr, ctypes.c_void_p),
                    ctypes.cast(ai.data.ptr, ctypes.c_void_p))
                print(l, lc, numpy.linalg.norm(facs1 - facs0.get()))
                #assert numpy.linalg.norm(facs1 - facs0.get()) < 1e-10
    '''
    def test_type2_ang_part(self):
        numpy.random.seed(4)
        rca = numpy.random.random(3)
        cache = numpy.empty(100000)
        for li in range(4):
            for lc in range(4):
                facs1 = numpy.zeros((li+1,(li+1)*(li+2)//2,lc*2+1,li+lc+1))
                libecp_cpu.type2_facs_ang(
                    facs1.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(li), ctypes.c_int(lc),
                    rca.ctypes.data_as(ctypes.c_void_p),
                    cache.ctypes.data_as(ctypes.c_void_p))
                facs0 = cupy.zeros_like(facs1)
                rca_gpu = cupy.asarray(rca)
                libecp.ECPtype2_facs_ang(
                    ctypes.cast(facs0.data.ptr, ctypes.c_void_p),
                    ctypes.cast(rca_gpu.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(li),
                    ctypes.c_int(lc))
                assert numpy.linalg.norm(facs0.get() - facs1) < 1e-10

    def test_type2_cart(self):
        for ish in range(mol.nbas):
            for jsh in range(mol.nbas):
                li = mol.bas_angular(ish)
                lj = mol.bas_angular(jsh)
                if li > lj:
                    continue
                di = (li+1) * (li+2) // 2 * mol.bas_nctr(ish)
                dj = (lj+1) * (lj+2) // 2 * mol.bas_nctr(jsh)
                mat0 = numpy.zeros((di,dj))
                cache = numpy.empty(100000)
                ecpbas = mol._ecpbas.copy()
                shls = (ish, jsh)
                shl_ptr = (ctypes.c_int*2)(*shls)
                null_ptr = lib.c_null_ptr()

                libecp_cpu.ECPtype2_cart(
                    mat0.ctypes.data_as(ctypes.c_void_p),
                    shl_ptr,
                    ecpbas.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(len(mol._ecpbas)),
                    mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
                    mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
                    mol._env.ctypes.data_as(ctypes.c_void_p),
                    null_ptr, cache.ctypes.data_as(ctypes.c_void_p))

                mat1 = cupy.zeros_like(mat0)
                #print(ecpbas)
                ecpbas0 = ecpbas[ecpbas[:,gto.ANG_OF] >= 0]
                ecpbas = cupy.asarray(ecpbas0)
                atm = cupy.asarray(mol._atm)
                bas = cupy.asarray(mol._bas)
                env = cupy.asarray(mol._env)
                ecploc = cupy.asarray([0,len(ecpbas)], dtype=numpy.int32)
                tasks = cupy.asarray([ish,jsh,0], dtype=numpy.int32)
                ntasks = 1
                libecp.ECPtype2_cart(
                    ctypes.cast(mat1.data.ptr, ctypes.c_void_p),
                    ctypes.cast(tasks.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(ntasks),
                    ctypes.cast(ecpbas.data.ptr, ctypes.c_void_p),
                    ctypes.cast(ecploc.data.ptr, ctypes.c_void_p),
                    ctypes.cast(atm.data.ptr, ctypes.c_void_p),
                    ctypes.cast(bas.data.ptr, ctypes.c_void_p),
                    ctypes.cast(env.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(li),
                    ctypes.c_int(lj))
                assert numpy.linalg.norm(mat1.get() - mat0) < 1e-10

if __name__ == "__main__":
    print("Full tests for ECP module")
    unittest.main()
