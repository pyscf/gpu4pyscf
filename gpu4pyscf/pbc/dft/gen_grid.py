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
from pyscf.pbc.dft import gen_grid as gen_grid_cpu
from pyscf.pbc.gto.cell import get_uniform_grids
from gpu4pyscf.dft import Grids
from gpu4pyscf.lib import utils, logger

__all__ = [
    'UniformGrids', 'BeckeGrids', 'AtomicGrids'
]

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

    def reset(self, cell=None):
        if cell is not None:
            self.cell = cell
        self.non0tab = None
        self._coords = None
        self._weights = None
        return self

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

    def argsort(self, tile=8):
        '''Return the indices that would group the grids in space.
        '''
        mx, my, mz = self.mesh
        nx = (mx + tile-1) // tile
        ny = (my + tile-1) // tile
        nz = (mz + tile-1) // tile

        _idx = np.arange(tile)
        idx_in_tile = _idx[:,None,None] * (my*mz) + _idx[:,None] * mz + _idx

        zigzag_xy = np.arange(nx*ny).reshape(nx, ny)
        zigzag_xy[1::2] = zigzag_xy[1::2,::-1]
        zigzag_xyz = nx*ny * np.arange(nz)[:,None] + zigzag_xy.ravel()
        zigzag_xyz[1::2] = zigzag_xyz[1::2,::-1]

        xs, ys, zs = np.unravel_index(zigzag_xyz.ravel(), (nx, ny, nz))
        xs *= tile
        ys *= tile
        zs *= tile
        idx = []
        for xi, yi, zi in zip(xs, ys, zs):
            offset = (xi * my + yi) * mz + zi
            idx.append(offset + idx_in_tile[:mx-xi,:my-yi,:mz-zi].ravel())
        return np.hstack(idx)

    build = gen_grid_cpu.UniformGrids.build
    dump_flags = gen_grid_cpu.UniformGrids.dump_flags
    kernel = gen_grid_cpu.UniformGrids.kernel

    to_gpu = utils.to_gpu
    to_cpu = utils.to_cpu


class BeckeGrids(Grids):
    '''Atomic grids for all-electron calculation.'''
    def __init__(self, cell):
        self.cell = cell
        Grids.__init__(self, cell)

    def build(self, cell=None, with_non0tab=False):
        if cell is None: cell = self.cell
        coords, weights = gen_grid_cpu.get_becke_grids(
            self.cell, self.atom_grid, radi_method=self.radi_method,
            level=self.level, prune=self.prune)
        self.coords = cp.asarray(coords)
        self.weights = cp.asarray(weights)
        if with_non0tab:
            raise NotImplementedError
        self.non0tab = None
        logger.info(self, 'tot grids = %d', len(self.weights))
        logger.info(self, 'cell vol = %.9g  sum(weights) = %.9g',
                    cell.vol, self.weights.sum())
        return self

    to_gpu = utils.to_gpu
    to_cpu = utils.to_cpu

AtomicGrids = BeckeGrids
