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

import numpy as np
import cupy
import ctypes
from gpu4pyscf.lib.cupy_helper import load_library

libcupy_helper = load_library('libcupy_helper')

class CDERI_POINTER(ctypes.Structure):
    pass

class CDERI():
    """
    CDERI object with sparsity in ij direction
    """
    def __init__(self, nao, naux, nblocks) -> None:
        cderi_ptr = ctypes.POINTER(CDERI_POINTER)()
        self.handle = cderi_ptr
        self.nao = nao
        self.naux = naux
        self.row = []
        self.col = []
        self.data = []
        libcupy_helper.init_cderi(
            ctypes.byref(cderi_ptr),
            ctypes.c_int(nblocks),
            ctypes.c_int(nao))
        return

    def __del__(self):
        self.row = []
        self.col = []
        self.data = []
        libcupy_helper.delete_cderi(ctypes.byref(self.handle))

    def add_block(self, data, rows, cols):
        self.row.append(rows)
        self.col.append(cols)
        self.data.append(data)

        rows = cupy.asarray(rows, dtype=cupy.int64)
        cols = cupy.asarray(cols, dtype=cupy.int64)
        assert rows.dtype == cupy.int64 and cols.dtype == cupy.int64
        nij = len(rows)
        err = libcupy_helper.add_block(
            ctypes.byref(self.handle),
            ctypes.c_int(nij),
            ctypes.c_int(self.naux),
            ctypes.cast(rows.data.ptr, ctypes.c_void_p),
            ctypes.cast(cols.data.ptr, ctypes.c_void_p),
            ctypes.cast(data.data.ptr, ctypes.c_void_p))
        if err != 0:
            raise RuntimeError('failed to add the block')
        return

    def unpack(self, p0, p1, out=None):
        if out is None: out = cupy.zeros([p1-p0, self.nao, self.nao])

        libcupy_helper.unpack(
            ctypes.byref(self.handle),
            ctypes.c_int(p0),
            ctypes.c_int(p1),
            ctypes.cast(out.data.ptr, ctypes.c_void_p)
        )
        return out

