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
import numpy as np
import cupy
import pyscf
from gpu4pyscf.df import cderi
import ctypes

class KnownValues(unittest.TestCase):
    def test_cderi(self):
        naux = 5
        nao = 3
        nblocks = 2
        eri = cupy.random.rand(naux, nao, nao)
        eri = eri + eri.transpose([0,2,1])

        print('initializing ... ')
        cderi_obj = cderi.CDERI(nao, naux, nblocks)
        row, col = cupy.tril_indices(nao)

        print('adding blocks ... ')
        row1 = row[:len(row)//2]
        col1 = col[:len(col)//2]
        data = eri[:, row1, col1]
        cderi_obj.add_block(data, row1, col1)
        
        print('adding blocks ...')
        row2 = row[len(row)//2:]
        col2 = col[len(col)//2:]
        data = eri[:, row2, col2]
        cderi_obj.add_block(data, row2, col2)
        cupy.cuda.runtime.deviceSynchronize()

        print('unpacking ...')
        p0 = 2; p1 = 4
        buf = cupy.zeros([p1-p0, nao, nao])
        cderi_obj.unpack(p0, p1, out=buf)

        assert cupy.linalg.norm(buf - eri[p0:p1,:,:]) < 1e-10
if __name__ == "__main__":
    print("Full Tests for CDERI")
    unittest.main()