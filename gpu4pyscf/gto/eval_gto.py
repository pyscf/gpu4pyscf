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

import ctypes
import numpy as np
import cupy

from gpu4pyscf.lib.cupy_helper import load_library

CUTOFF = 1e-10
BLKSIZE = 256
libgdft = load_library('libgdft')

def make_screen_index(sorted_mol, coords, shls_slice=None, cutoff=CUTOFF, blksize=BLKSIZE):
    ngrids = coords.shape[0]
    nbas = sorted_mol.nbas
    ao_loc = sorted_mol.ao_loc_nr()
    nblocks = (ngrids + blksize - 1) // blksize
    s_index = cupy.zeros([nblocks, len(ao_loc)-1], dtype=np.int32)
    atm_coords = cupy.asarray(sorted_mol.atom_coords(), order='F')
    stream = cupy.cuda.get_current_stream()
    env = cupy.asarray(sorted_mol._env)
    if not hasattr(sorted_mol, 'l_ctr_offsets'):
        raise RuntimeError('Mol object is not sorted. Use PySCF routine instead.')
    ctr_offsets = sorted_mol.l_ctr_offsets
    nctr = ctr_offsets.size - 1
    err = libgdft.GDFTscreen_index(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(s_index.data.ptr, ctypes.c_void_p),
        ctypes.c_double(cutoff),
        ctypes.cast(coords.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(blksize),
        sorted_mol._bas.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nbas),
        ctypes.cast(atm_coords.data.ptr, ctypes.c_void_p),
        ctypes.c_int(sorted_mol.natm),
        ctypes.cast(env.data.ptr, ctypes.c_void_p),
        ctr_offsets.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nctr))
    
    if err != 0:
        raise RuntimeError('Failed in GDFTscreen_index kernel')
    
    return s_index
