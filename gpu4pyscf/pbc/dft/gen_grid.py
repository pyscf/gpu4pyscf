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
from pyscf.pbc.gto import eval_gto as pbc_eval_gto
from pyscf.dft.gen_grid import gen_atomic_grids
import gpu4pyscf
from gpu4pyscf.dft import Grids
from gpu4pyscf.lib import utils, logger
from gpu4pyscf.dft import radi
from gpu4pyscf.lib.cupy_helper import load_library

libgdft = load_library('libgdft')

__all__ = [
    'UniformGrids', 'BeckeGrids', 'AtomicGrids'
]

# modified from pyscf.dft.gen_grid.gen_partition
def get_becke_grids(cell, atom_grid={}, radi_method=gpu4pyscf.dft.radi.gauss_chebyshev,
                    level=3, prune=gpu4pyscf.dft.gen_grid.nwchem_prune,
                    radii_adjust=None, atomic_radii=radi.BRAGG_RADII):
    '''real-space grids using Becke scheme

    Args:
        cell : instance of :class:`Cell`

    Returns:
        coords : (N, 3) ndarray
            The real-space grid point coordinates.
        weights : (N) ndarray
    '''
    assert cell.dimension == 3
    dimension = cell.dimension

    rcut = pbc_eval_gto._estimate_rcut(cell).max()
    Ls = pbc_eval_gto.get_lattice_Ls(cell, rcut=rcut)
    logger.debug(cell, f'Becke grid rcut = {rcut}')

    supatm_coords = Ls.reshape(-1,1,3) + cell.atom_coords()
    atom_grids_tab = gen_atomic_grids(cell, atom_grid, radi_method, level, prune)
    coords_all = []
    weights_all = []
    b = cell.reciprocal_vectors(norm_to=1)
    supatm_idx = []
    atm_idx = []
    k = 0
    tol = 1e-15
    for iL, L in enumerate(Ls):
        for ia in range(cell.natm):
            coords, vol = atom_grids_tab[cell.atom_symbol(ia)]
            coords = coords + supatm_coords[iL,ia]
            # search for grids in unit cell
            c = b.dot(coords.T)

            mask = np.ones(c.shape[1], dtype=bool)
            if dimension >= 1:
                mask &= (c[0]>-.5-tol) & (c[0]<.5+tol)
            if dimension >= 2:
                mask &= (c[1]>-.5-tol) & (c[1]<.5+tol)
            if dimension == 3:
                mask &= (c[2]>-.5-tol) & (c[2]<.5+tol)

            vol = vol[mask]
            if vol.size > 8: # The number 8 is an arbitrary number that makes the calculation much faster.
                c = c[:,mask]
                if dimension >= 1:
                    vol[abs(c[0]+.5) < tol] *= .5
                    vol[abs(c[0]-.5) < tol] *= .5
                if dimension >= 2:
                    vol[abs(c[1]+.5) < tol] *= .5
                    vol[abs(c[1]-.5) < tol] *= .5
                if dimension == 3:
                    vol[abs(c[2]+.5) < tol] *= .5
                    vol[abs(c[2]-.5) < tol] *= .5
                coords = coords[mask]
                coords_all.append(coords)
                weights_all.append(vol)
                supatm_idx.append(k)
                atm_idx.append(ia)
            k += 1

    supatm_coords = np.asarray(supatm_coords.reshape(-1,3)[supatm_idx], order='C')
    sup_natm = len(supatm_coords)

    supatm_idx = np.hstack([np.full(weights_all[i].size, i) for i in range(len(weights_all))])

    coords_all = np.vstack(coords_all)
    weights_all = np.hstack(weights_all)

    ngrids = weights_all.size
    assert coords_all.shape == (ngrids, 3)
    assert supatm_idx.shape == (ngrids,)

    weights_all = cp.asarray(weights_all, dtype = cp.float64)
    coords_all = cp.asarray(coords_all, dtype = cp.float64, order = "F")
    supatm_coords = cp.asarray(supatm_coords, dtype = cp.float64, order = "F")
    supatm_idx = cp.asarray(supatm_idx, dtype = cp.int32)

    quadrature_weights_all = weights_all.copy()
    supatm_to_atm_idx = cp.asarray(atm_idx, dtype = cp.int32)

    if radii_adjust is None:
        # a_factor = cp.zeros((sup_natm, sup_natm), dtype = cp.float64)
        a_factor_ptr = lib.c_null_ptr()
    else:
        fake_supcell = type('FakeSuperCell', (object,), {})()
        fake_supcell.elements = [ cell.elements[i] for i in atm_idx ]

        assert radii_adjust == radi.treutler_atomic_radii_adjust
        a_factor = -radi.get_treutler_fac(fake_supcell, atomic_radii)
        assert a_factor.shape == (sup_natm, sup_natm)
        a_factor_ptr = ctypes.cast(a_factor.data.ptr, ctypes.c_void_p)

    err = libgdft.GDFTbecke_partition_weights(
        ctypes.cast(weights_all.data.ptr, ctypes.c_void_p),
        ctypes.cast(coords_all.data.ptr, ctypes.c_void_p),
        ctypes.cast(supatm_coords.data.ptr, ctypes.c_void_p),
        a_factor_ptr,
        ctypes.cast(supatm_idx.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(sup_natm),
    )
    if err != 0:
        raise RuntimeError('GDFTbecke_partition_weights kernel failed')

    return coords_all, weights_all, quadrature_weights_all, supatm_idx, supatm_coords, supatm_to_atm_idx

def get_becke_weight_derivative(grids, natm):
    assert type(grids) is BeckeGrids
    ngrids = grids.coords.shape[0]
    assert grids.supatm_idx.shape[0] == ngrids
    assert grids.quadrature_weights.shape[0] == ngrids
    sup_natm = grids.supatm_coords.shape[0]
    assert grids.supatm_to_atm_idx.shape[0] == sup_natm

    # a_factor = cp.zeros((sup_natm, sup_natm), dtype = cp.float64)
    a_factor_ptr = lib.c_null_ptr()

    grids_coords = cp.asarray(grids.coords, order = "F")
    grids_quadrature_weights = cp.asarray(grids.quadrature_weights)
    grids_supatm_idx = cp.asarray(grids.supatm_idx)
    grids_supatm_coords = cp.asarray(grids.supatm_coords, order = "F")
    grids_supatm_to_atm_idx = cp.asarray(grids.supatm_to_atm_idx)

    P_B = cp.zeros([sup_natm, ngrids], order = "C")
    libgdft.GDFTbecke_eval_PB(
        ctypes.cast(P_B.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_supatm_coords.data.ptr, ctypes.c_void_p),
        a_factor_ptr,
        ctypes.c_int(ngrids),
        ctypes.c_int(sup_natm),
    )
    sum_P_B = cp.sum(P_B, axis = 0)
    inv_sum_P_B = cp.zeros(ngrids)
    nonzero_sum_P_B_location = (sum_P_B > 1e-14)
    inv_sum_P_B[nonzero_sum_P_B_location] = 1.0 / sum_P_B[nonzero_sum_P_B_location]
    nonzero_sum_P_B_location = None
    sum_P_B = None

    dweight_dA_supercell = cp.zeros([sup_natm, 3, ngrids], order = "C")
    libgdft.GDFTbecke_partition_weight_derivative(
        ctypes.cast(dweight_dA_supercell.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_quadrature_weights.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_supatm_coords.data.ptr, ctypes.c_void_p),
        a_factor_ptr,
        ctypes.cast(grids_supatm_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(P_B.data.ptr, ctypes.c_void_p),
        ctypes.cast(inv_sum_P_B.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(sup_natm),
    )
    P_B = None
    inv_sum_P_B = None

    dweight_dA_supercell[grids_supatm_idx, 0, cp.arange(ngrids)] = -cp.sum(dweight_dA_supercell[:, 0, :], axis=[0])
    dweight_dA_supercell[grids_supatm_idx, 1, cp.arange(ngrids)] = -cp.sum(dweight_dA_supercell[:, 1, :], axis=[0])
    dweight_dA_supercell[grids_supatm_idx, 2, cp.arange(ngrids)] = -cp.sum(dweight_dA_supercell[:, 2, :], axis=[0])

    dweight_dA_unitcell = cp.zeros([natm, 3, ngrids])
    cp.add.at(dweight_dA_unitcell, grids_supatm_to_atm_idx, dweight_dA_supercell)

    return dweight_dA_unitcell

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
        assert cell is self.cell

        log = logger.new_logger(cell)
        t0 = log.init_timer()

        coords, weights, quadrature_weights, supatm_idx, supatm_coords, supatm_to_atm_idx = get_becke_grids(
            self.cell, self.atom_grid, radi_method=self.radi_method,
            level=self.level, prune=self.prune)
        self.coords = cp.asarray(coords)
        self.weights = cp.asarray(weights)
        self.quadrature_weights = cp.asarray(quadrature_weights)
        self.supatm_idx = cp.asarray(supatm_idx)
        self.supatm_coords = cp.asarray(supatm_coords)
        self.supatm_to_atm_idx = supatm_to_atm_idx
        if with_non0tab:
            raise NotImplementedError
        self.non0tab = None
        logger.info(self, 'tot grids = %d', len(self.weights))
        logger.info(self, 'cell vol = %.9g  sum(weights) = %.9g',
                    cell.vol, self.weights.sum())

        log.timer_debug1('PBC grid Becke weight calculation', *t0)
        return self

    def reset(self, cell=None):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        if cell is not None:
            self.cell = cell
        self.coords = None
        self.weights = None
        self.atm_idx = None
        self.quadrature_weights = None
        self.supatm_idx = None
        self.supatm_coords = None
        self.supatm_to_atm_idx = None
        self.non0tab = None
        return self

    to_gpu = utils.to_gpu
    to_cpu = utils.to_cpu

AtomicGrids = BeckeGrids
