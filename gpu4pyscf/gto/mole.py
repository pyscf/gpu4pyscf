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


import os
import numpy as np
import cupy
import functools
from pyscf import gto

@functools.lru_cache(20)
def get_cart2sph(lmax=12):
    cart2sph = []
    for l in range(lmax):
        c2s = gto.mole.cart2sph(l, normalized='sp')
        cart2sph.append(cupy.asarray(c2s, order='C'))
    return cart2sph

