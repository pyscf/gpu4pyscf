#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
import cupy as cp
from gpu4pyscf.gto.mole import Cell, SortedCell

def get_Gv_base(cell, mesh=None):
    if mesh is None:
        mesh = cell.mesh

    # Default, the 3D uniform grids
    rx = cp.fft.fftfreq(mesh[0], 1./mesh[0])
    ry = cp.fft.fftfreq(mesh[1], 1./mesh[1])
    rz = cp.fft.fftfreq(mesh[2], 1./mesh[2])
    return rx, ry, rz

def get_Gv_weights(cell, mesh=None):
    '''Calculate G-vectors and weights.

    Returns:
        Gv : (ngris, 3) ndarray of floats
            The array of G-vectors.
    '''
    if cell.dimension <= 2 and cell.low_dim_ft_type == 'inf_vacuum':
        raise NotImplementedError

    rx, ry, rz = Gvbase = get_Gv_base(cell, mesh)
    b = cell.reciprocal_vectors()
    weights = abs(np.linalg.det(b)) / (2*np.pi)**3

    #:Gv = lib.cartesian_prod(Gvbase).dot(b)
    b = cp.asarray(b)
    brx = b[0,:,None] * rx
    bry = b[1,:,None] * ry
    brz = b[2,:,None] * rz
    Gv = brx[:,:,None,None] + bry[:,None,:,None] + brz[:,None,None,:]
    Gv = Gv.reshape(3,-1).T
    return Gv, Gvbase, weights

def get_Gv(cell, mesh=None):
    return get_Gv_weights(cell, mesh)[0]
