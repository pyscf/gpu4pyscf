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

import pyscf
from pyscf import df
from pyscf.gto.moleintor import getints, make_cintopt
from pyscf.df.grad.rhf import _int3c_wrapper
import numpy as np
import unittest
from gpu4pyscf.df import int3c2e
from gpu4pyscf.lib.cupy_helper import load_library

libgint = load_library('libgint')

'''
check int3c2e consistency between pyscf and gpu4pyscf
'''

def setUpModule():
    global mol, auxmol
    atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''
    mol = pyscf.M(atom=atom,
                  basis= 'def2-tzvpp',
                  verbose=1,
                  output='/dev/null')
    auxmol = df.addons.make_auxmol(mol, auxbasis='def2-tzvpp-jkfit')
    auxmol.output = '/dev/null'

def tearDownModule():
    global mol, auxmol
    mol.stdout.close()
    auxmol.stdout.close()
    del mol, auxmol

omega = 0.2

def check_int3c2e_derivatives(ip_type):
    nbas = mol.nbas
    pmol = mol + auxmol
    intor = mol._add_suffix('int3c2e_'+ip_type)
    opt = make_cintopt(mol._atm, mol._bas, mol._env, intor)
    shls_slice = (0, nbas, 0, nbas, nbas, pmol.nbas)
    int3c_pyscf = getints(intor, pmol._atm, pmol._bas, pmol._env, shls_slice, aosym='s1', cintopt=opt)
    int3c_gpu = int3c2e.get_int3c2e_general(mol, auxmol, ip_type=ip_type).get()
    assert np.linalg.norm(int3c_pyscf - int3c_gpu) < 1e-8

    with mol.with_range_coulomb(omega):
        nbas = mol.nbas
        pmol = mol + auxmol
        intor = mol._add_suffix('int3c2e_'+ip_type)
        opt = make_cintopt(mol._atm, mol._bas, mol._env, intor)
        shls_slice = (0, nbas, 0, nbas, nbas, pmol.nbas)
        int3c_pyscf = getints(intor, pmol._atm, pmol._bas, pmol._env, shls_slice, aosym='s1', cintopt=opt)
        int3c_gpu = int3c2e.get_int3c2e_general(mol, auxmol, ip_type=ip_type, omega=omega).get()
        assert np.linalg.norm(int3c_pyscf - int3c_gpu) < 1e-8

class KnownValues(unittest.TestCase):
    def test_int3c2e(self):
        get_int3c = _int3c_wrapper(mol, auxmol, 'int3c2e', 's1')
        int3c_pyscf = get_int3c((0, mol.nbas, 0, mol.nbas, 0, auxmol.nbas))
        int3c_gpu = int3c2e.get_int3c2e(mol, auxmol, aosym='s1').get()
        assert np.linalg.norm(int3c_gpu - int3c_pyscf) < 1e-8

    def test_int3c2e_omega(self):
        omega = 0.2
        with mol.with_range_coulomb(omega):
            get_int3c = _int3c_wrapper(mol, auxmol, 'int3c2e', 's1')
            int3c_pyscf = get_int3c((0, mol.nbas, 0, mol.nbas, 0, auxmol.nbas))
            int3c_gpu = int3c2e.get_int3c2e(mol, auxmol, aosym='s1', omega=omega).get()
        assert np.linalg.norm(int3c_gpu[0,0,:] - int3c_pyscf[0,0,:]) < 1e-8

    def test_int3c2e_ip1(self):
        check_int3c2e_derivatives('ip1')

    def test_int3c2e_ip2(self):
        check_int3c2e_derivatives('ip2')

    def test_int3c2e_ipip1(self):
        check_int3c2e_derivatives('ipip1')

    def test_int3c2e_ipip2(self):
        check_int3c2e_derivatives('ipip2')

    def test_int3c2e_ip1ip2(self):
        check_int3c2e_derivatives('ip1ip2')

    def test_int3c2e_ipvip1(self):
        check_int3c2e_derivatives('ipvip1')

    def test_int1e_iprinv(self):
        from pyscf import gto
        coords = mol.atom_coords()
        charges = mol.atom_charges()

        fakemol = gto.fakemol_for_charges(coords)
        int3c = int3c2e.get_int3c2e_general(mol, fakemol, ip_type='ip1').get()

        for i,q in enumerate(charges):
            mol.set_rinv_origin(coords[i])
            h1ao = mol.intor('int1e_iprinv', comp=3) # <\nabla|1/r|>
            assert np.linalg.norm(int3c[:,:,:,i] - h1ao) < 1e-7

    def test_int1e_ipiprinv(self):
        from pyscf import gto
        coords = mol.atom_coords()
        charges = mol.atom_charges()

        fakemol = gto.fakemol_for_charges(coords)
        fakemol.output = '/dev/null'
        int3c = int3c2e.get_int3c2e_general(mol, fakemol, ip_type='ipip1').get()

        for i,q in enumerate(charges):
            mol.set_rinv_origin(coords[i])
            h1ao = mol.intor('int1e_ipiprinv', comp=9) # <\nabla|1/r|>
            assert np.linalg.norm(int3c[:,:,:,i] - h1ao) < 1e-7

    def test_int1e_iprinvip(self):
        from pyscf import gto
        coords = mol.atom_coords()
        charges = mol.atom_charges()

        fakemol = gto.fakemol_for_charges(coords)
        fakemol.output = '/dev/null'
        int3c = int3c2e.get_int3c2e_general(mol, fakemol, ip_type='ipvip1').get()

        for i,q in enumerate(charges):
            mol.set_rinv_origin(coords[i])
            h1ao = mol.intor('int1e_iprinvip', comp=9) # <\nabla|1/r|>
            assert np.linalg.norm(int3c[:,:,:,i] - h1ao) < 1e-7

if __name__ == "__main__":
    print("Full Tests for int3c")
    unittest.main()
