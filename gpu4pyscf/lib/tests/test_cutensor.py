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
import cupy
from gpu4pyscf.lib.cupy_helper import contract, contract_trinary
from gpu4pyscf.lib import cutensor

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

    def test_complex_valued(self):
        a = cupy.random.rand(10,9,11) + cupy.random.rand(10,9,11)*1j
        b = cupy.random.rand(11,7,13) + cupy.random.rand(11,7,13)*1j
        c_einsum = cupy.einsum('ijk,ikl->jl', a[3:9,:,4:10], b[3:9,:6, 7:13])
        c_contract = contract('ijk,ikl->jl', a[3:9,:,4:10], b[3:9,:6, 7:13])
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

    @unittest.skipIf('', cutensor.__version__ is None or cutensor.__version__ < 20301)
    def test_trinary_contraction(self):
        a = cupy.random.rand(48,11)
        b = cupy.random.rand(48,13)
        c = cupy.random.rand(48,4)
        ref = numpy.einsum('pi,pj,pk->kij', a.get(), b.get(), c.get(), optimize=True)
        out = contract_trinary('pi,pj,pk->kij', a, b, c)
        assert abs(out.get() - ref).max() < 1e-10

        a = cupy.random.rand(48,11)
        b = cupy.random.rand(48,13)
        c = cupy.random.rand(48,4)
        ref = numpy.einsum('pi,pj,pk->ikj', a.get(), b.get(), c.get(), optimize=True)
        out = contract_trinary('pi,pj,pk->ikj', a, b, c)
        assert abs(out.get() - ref).max() < 1e-10

        a = cupy.random.rand(48,11)
        b = cupy.random.rand(48,13)
        c = cupy.random.rand(48,4)
        ref = numpy.einsum('pi,pj,pk->ij', a.get(), b.get(), c.get(), optimize=True)
        out = contract_trinary('pi,pj,pk->ij', a, b, c)
        assert abs(out.get() - ref).max() < 1e-10

        a = cupy.random.rand(48,11)
        b = cupy.random.rand(3,48,13)
        c = cupy.random.rand(48,4)
        ref = numpy.einsum('pi,xpj,pk->xkij', a.get(), b.get(), c.get(), optimize=True)
        out = contract_trinary('pi,xpj,pk->xkij', a, b, c)
        assert abs(out.get() - ref).max() < 1e-10

        a = cupy.random.rand(48,11)
        b = cupy.random.rand(3,48,13)
        c = cupy.random.rand(48,4)
        ref = numpy.einsum('xpj,pi,pk->xkij', b.get(), a.get(), c.get(), optimize=True)
        out = contract_trinary('xpj,pi,pk->xkij', b, a, c)
        assert abs(out.get() - ref).max() < 1e-10

        a = cupy.random.rand(4,48,11)
        b = cupy.random.rand(48,13)
        c = cupy.random.rand(4,48)
        ref = numpy.einsum('xpi,pj,xp->ij', a.get(), b.get(), c.get(), optimize=True)
        out = contract_trinary('xpi,pj,xp->ij', a, b, c)
        assert abs(out.get() - ref).max() < 1e-10

        a = cupy.random.rand(4,48,11)
        b = cupy.random.rand(48,13)
        c = cupy.random.rand(4,48)
        ref = numpy.einsum('xpi,xp,pj->ij', a.get(), c.get(), b.get(), optimize=True)
        out = contract_trinary('xpi,xp,pj->ij', a, c, b)
        assert abs(out.get() - ref).max() < 1e-10

        a = cupy.random.rand(20,4096)
        b = cupy.random.rand(6,4096)
        c = cupy.empty((6,20,20))
        c = contract_trinary('ig,ng,jg->nij', a, b, a, out=c)
        ref = numpy.einsum('ig,ng,jg->nij', a.get(), b.get(), a.get(), optimize=True)
        assert abs(c.get() - ref).max() < 1e-10

if __name__ == "__main__":
    print("Full tests for cutensor module")
    unittest.main()
