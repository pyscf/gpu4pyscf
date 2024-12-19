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
    to_cpu = utils.to_cpu

class BeckeGrids:
    pass
