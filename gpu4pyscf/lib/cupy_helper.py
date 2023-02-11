# gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
#
# Copyright (C) 2022 Qiming Sun
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

import ctypes
import numpy as np
import cupy
from pyscf import lib

libcupy_helper = lib.load_library('libcupy_helper')

# for i > j of 2d mat, mat[j,i] = mat[i,j]
def hermi_triu(mat, hermi=1, inplace=True):
    '''Use the elements of the lower triangular part to fill the upper triangular part.
    See also pyscf.lib.hermi_triu
    '''
    if not inplace:
        mat = mat.copy('C')
    assert mat.flags.c_contiguous

    if mat.ndim == 2:
        n = mat.shape[0]
        counts = 1
    elif mat.ndim == 3:
        counts, n = mat.shape[:2]
    else:
        raise ValueError(f'dimension not supported {mat.ndim}')

    libcupy_helper.CPdsymm_triu(
        ctypes.cast(mat.data.ptr, ctypes.c_void_p),
        ctypes.c_int(n), ctypes.c_int(counts))
    return mat

@lib.with_doc(lib.unpack_tril)
def unpack_tril(tril, filltriu=lib.HERMITIAN, axis=-1, out=None):
    assert tril.flags.c_contiguous
    assert tril.dtype == np.double
    if tril.ndim == 1:
        count, nd = 1, tril.size
        nd = int((nd*2)**.5)
        shape = (nd, nd)
    elif tril.ndim == 2:
        if axis == 0:
            nd, count = tril.shape
        else:
            count, nd = tril.shape
        nd = int((nd*2)**.5)
        shape = (count, nd, nd)
    else:
        raise NotImplementedError('unpack_tril for high-dimension arrays')

    if out is None:
        out = cupy.empty(shape)
    else:
        out.size >= np.prod(shape)
    libcupy_helper.CPdunpack_tril_2d(
        ctypes.c_int(count), ctypes.c_int(nd),
        ctypes.cast(tril.data.ptr, ctypes.c_void_p),
        ctypes.cast(out.data.ptr, ctypes.c_void_p), ctypes.c_int(filltriu))
    return out
