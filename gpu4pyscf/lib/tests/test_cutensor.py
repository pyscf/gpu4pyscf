# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
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
from gpu4pyscf.lib.cupy_helper import *

class KnownValues(unittest.TestCase):
    def test_contract(self):
        a = cupy.random.rand(10,9,11)
        b = cupy.random.rand(11,7,13)
        c_einsum = cupy.einsum('ijk,ikl->jl', a[3:9,:,4:10], b[3:9,:6, 7:13])
        c_contract = contract('ijk,ikl->jl', a[3:9,:,4:10], b[3:9,:6, 7:13])
        assert cupy.linalg.norm(c_einsum - c_contract) < 1e-10

        a = cupy.random.rand(10,10,10,10)
        b = cupy.random.rand(20,20)
        c_einsum = cupy.einsum('lkji,jl->ik', a, b[10:20,10:20])
        c_contract = contract('lkji,jl->ik', a, b[10:20,10:20])
        assert cupy.linalg.norm(c_einsum - c_contract) < 1e-10

        a = cupy.random.rand(10,10,10,10)
        b = cupy.random.rand(20,20)
        c_einsum = cupy.einsum('lkji,jk->il', a, b[10:20,10:20])
        c_contract = contract('lkji,jk->il', a, b[10:20,10:20])
        assert cupy.linalg.norm(c_einsum - c_contract) < 1e-10

    def test_cache(self):
        a = cupy.random.rand(20,20,20,20)
        b = cupy.random.rand(20,20)

        c = contract('ijkl,ik->jl', a, b)
        c_einsum = cupy.einsum('ijkl,ik->jl', a, b)
        assert cupy.linalg.norm(c - c_einsum) < 1e-10

        c = contract('ijkl,jl->ik', a, b)
        c_einsum = cupy.einsum('ijkl,jl->ik', a, b)
        assert cupy.linalg.norm(c - c_einsum) < 1e-10
    
if __name__ == "__main__":
    print("Full tests for cutensor module")
    unittest.main()