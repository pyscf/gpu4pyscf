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

