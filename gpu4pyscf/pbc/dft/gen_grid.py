#!/usr/bin/env python
#
# Copyright 2024 The GPU4PySCF Developers. All Rights Reserved.
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
import cupy as cp
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.dft import gen_grid as gen_grid_cpu
from pyscf.pbc.gto.cell import get_uniform_grids
from gpu4pyscf.lib import utils

class UniformGrids(lib.StreamObject):
    '''Uniform Grid class.'''

    def __init__(self, cell):
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.mesh = cell.mesh
        self.non0tab = None
        self._coords = None
        self._weights = None

    @property
    def coords(self):
        if self._coords is not None:
            return self._coords
        else:
            return cp.asarray(get_uniform_grids(self.cell, self.mesh))
    @coords.setter
    def coords(self, x):
        self._coords = x

    @property
    def weights(self):
        if self._weights is not None:
            return self._weights
        else:
            ngrids = np.prod(self.mesh)
            weights = cp.empty(ngrids)
            weights[:] = self.cell.vol / ngrids
            return weights
    @weights.setter
    def weights(self, x):
        self._weights = x

    @property
    def size(self):
        return np.prod(self.mesh)

    reset = gen_grid_cpu.UniformGrids.reset
    build = gen_grid_cpu.UniformGrids.build
    dump_flags = gen_grid_cpu.UniformGrids.dump_flags
    kernel = gen_grid_cpu.UniformGrids.kernel

    to_gpu = utils.to_gpu
    device = utils.device

    def to_cpu(self):
        obj = utils.to_cpu(self)
        return obj.reset()

class BeckeGrids:
    pass
