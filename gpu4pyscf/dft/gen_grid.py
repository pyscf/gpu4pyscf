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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
# Modified by Xiaojie Wu <wxj6000@gmail.com>

'''
Generate DFT grids and weights, based on the code provided by Gerald Knizia <>

Reference for Lebedev-Laikov grid:
  V. I. Lebedev, and D. N. Laikov "A quadrature formula for the sphere of the
  131st algebraic order of accuracy", Doklady Mathematics, 59, 477-481 (1999)
'''


import sys
import ctypes
import numpy
import numpy as np
import cupy
import cupy as cp
from pyscf import lib
from pyscf import gto
from pyscf.dft import gen_grid as gen_grid_cpu
from gpu4pyscf.lib import utils
from pyscf.gto.eval_gto import BLKSIZE, NBINS, CUTOFF, make_screen_index
from pyscf import __config__
from gpu4pyscf.lib import logger
from gpu4pyscf.dft import radi
from gpu4pyscf.lib.cupy_helper import load_library, asarray
from gpu4pyscf import __config__ as __gpu4pyscf_config__

libdft = lib.load_library('libdft')
libgdft = load_library('libgdft')
libgdft.GDFTbecke_partition_weights.result_type = ctypes.c_int

from pyscf.dft.gen_grid import GROUP_BOUNDARY_PENALTY, NELEC_ERROR_TOL, LEBEDEV_ORDER, LEBEDEV_NGRID

GROUP_BOX_SIZE = 3.0
ALIGNMENT_UNIT = getattr(__gpu4pyscf_config__, 'grid_aligned', 128)


def sg1_prune(nuc, rads, n_ang, radii=radi.SG1RADII):
    '''SG1, CPL, 209, 506

    Args:
        nuc : int
            Nuclear charge.

        rads : 1D array
            Grid coordinates on radical axis.

        n_ang : int
            Max number of grids over angular part.

    Kwargs:
        radii : 1D array
            radii (in Bohr) for atoms in periodic table

    Returns:
        A list has the same length as rads. The list element is the number of
        grids over angular part for each radial grid.
    '''
# In SG1 the ang grids for the five regions
#            6  38 86  194 86
    if nuc >= 19:
        return 194 * numpy.ones_like(rads, dtype=numpy.int64)

    leb_ngrid = numpy.array([6, 38, 86, 194, 86], dtype=numpy.int64)
    alphas = numpy.array((
        (0.25  , 0.5, 1.0, 4.5),
        (0.1667, 0.5, 0.9, 3.5),
        (0.1   , 0.4, 0.8, 2.5)))

    r_atom = radii[nuc] + 1e-200
    rads = numpy.asarray(rads)
    if nuc <= 2:  # H, He
        place = ((rads/r_atom).reshape(-1,1) > alphas[0]).sum(axis=1)
    elif nuc <= 10:  # Li - Ne
        place = ((rads/r_atom).reshape(-1,1) > alphas[1]).sum(axis=1)
    else:
        place = ((rads/r_atom).reshape(-1,1) > alphas[2]).sum(axis=1)
    return leb_ngrid[place]

def nwchem_prune(nuc, rads, n_ang, radii=radi.BRAGG_RADII):
    '''NWChem

    Args:
        nuc : int
            Nuclear charge.

        rads : 1D array
            Grid coordinates on radical axis.

        n_ang : int
            Max number of grids over angular part.

    Kwargs:
        radii : 1D array
            radii (in Bohr) for atoms in periodic table

    Returns:
        A list has the same length as rads. The list element is the number of
        grids over angular part for each radial grid.
    '''
    alphas = numpy.array((
        (0.25  , 0.5, 1.0, 4.5),
        (0.1667, 0.5, 0.9, 3.5),
        (0.1   , 0.4, 0.8, 2.5)))
    leb_ngrid = LEBEDEV_NGRID[4:]  # [38, 50, 74, 86, ...]
    if n_ang < 50:
        return numpy.repeat(n_ang, len(rads))
    elif n_ang == 50:
        leb_l = numpy.array([1, 2, 2, 2, 1])
    else:
        idx = numpy.where(leb_ngrid==n_ang)[0][0]
        leb_l = numpy.array([1, 3, idx-1, idx, idx-1])
    r_atom = radii[nuc] + 1e-200
    if nuc <= 2:  # H, He
        place = ((rads/r_atom).reshape(-1,1) > alphas[0]).sum(axis=1)
    elif nuc <= 10:  # Li - Ne
        place = ((rads/r_atom).reshape(-1,1) > alphas[1]).sum(axis=1)
    else:
        place = ((rads/r_atom).reshape(-1,1) > alphas[2]).sum(axis=1)
    angs = leb_l[place]
    angs = leb_ngrid[angs]
    return angs

# Prune scheme JCP 102, 346 (1995); DOI:10.1063/1.469408
def treutler_prune(nuc, rads, n_ang, radii=None):
    '''Treutler-Ahlrichs

    Args:
        nuc : int
            Nuclear charge.

        rads : 1D array
            Grid coordinates on radical axis.

        n_ang : int
            Max number of grids over angular part.

    Returns:
        A list has the same length as rads. The list element is the number of
        grids over angular part for each radial grid.
    '''
    nr = len(rads)
    leb_ngrid = numpy.empty(nr, dtype=int)
    leb_ngrid[:nr//3] = 14 # l=5
    leb_ngrid[nr//3:nr//2] = 50 # l=11
    leb_ngrid[nr//2:] = n_ang
    return leb_ngrid



###########################################################
# Becke partitioning

# Stratmann, Scuseria, Frisch. CPL, 257, 213 (1996), eq.11
def stratmann(g):
    '''Stratmann, Scuseria, Frisch. CPL, 257, 213 (1996); DOI:10.1016/0009-2614(96)00600-8'''
    a = .64  # for eq. 14
    g = numpy.asarray(g)
    ma = g/a
    ma2 = ma * ma
    g1 = numpy.asarray((1/16.)*(ma*(35 + ma2*(-35 + ma2*(21 - 5 *ma2)))))
    g1[g<=-a] = -1
    g1[g>= a] =  1
    return g1

def original_becke(g):
    '''Becke, JCP 88, 2547 (1988); DOI:10.1063/1.454033'''
#    This funciton has been optimized in the C code VXCgen_grid
#    g = (3 - g**2) * g * .5
#    g = (3 - g**2) * g * .5
#    g = (3 - g**2) * g * .5
#    return g
    pass

def gen_atomic_grids(mol, atom_grid={}, radi_method=radi.gauss_chebyshev,
                     level=3, prune=nwchem_prune, **kwargs):
    '''Generate number of radial grids and angular grids for the given molecule.

    Returns:
        A dict, with the atom symbol for the dict key.  For each atom type,
        the dict value has two items: one is the meshgrid coordinates wrt the
        atom center; the second is the volume of that grid.
    '''
    if isinstance(atom_grid, (list, tuple)):
        atom_grid = dict([(mol.atom_symbol(ia), atom_grid)
                          for ia in range(mol.natm)])
    atom_grids_tab = {}
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)

        if symb not in atom_grids_tab:
            chg = gto.charge(symb)
            if symb in atom_grid:
                n_rad, n_ang = atom_grid[symb]
                if n_ang not in LEBEDEV_NGRID:
                    if n_ang in LEBEDEV_ORDER:
                        logger.warn(mol, 'n_ang %d for atom %d %s is not '
                                    'the supported Lebedev angular grids. '
                                    'Set n_ang to %d', n_ang, ia, symb,
                                    LEBEDEV_ORDER[n_ang])
                        n_ang = LEBEDEV_ORDER[n_ang]
                    else:
                        raise ValueError('Unsupported angular grids %d' % n_ang)
            else:
                n_rad = _default_rad(chg, level)
                n_ang = _default_ang(chg, level)
            rad, dr = radi_method(n_rad, chg, ia, **kwargs)

            rad_weight = 4*numpy.pi * rad**2 * dr

            if callable(prune):
                angs = prune(chg, rad, n_ang)
            else:
                angs = [n_ang] * n_rad
            logger.debug(mol, 'atom %s rad-grids = %d, ang-grids = %s',
                         symb, n_rad, angs)
            if isinstance(angs, cupy.ndarray): angs = angs.get()
            angs = numpy.array(angs)
            coords = []
            vol = []
            for n in sorted(set(angs)):
                grid = numpy.empty((n,4))
                libdft.MakeAngularGrid(grid.ctypes.data_as(ctypes.c_void_p),
                                       ctypes.c_int(n))
                idx = numpy.where(angs==n)[0]
                for i0, i1 in lib.prange(0, len(idx), 12):  # 12 radi-grids as a group
                    coords.append(numpy.einsum('i,jk->jik',rad[idx[i0:i1]],
                                               grid[:,:3]).reshape(-1,3))
                    vol.append(numpy.einsum('i,j->ji', rad_weight[idx[i0:i1]],
                                            grid[:,3]).ravel())
                #coords.append(cupy.einsum('i,jk->jik', rad[idx], grid[:,:3]).reshape(-1,3))
                #vol.append(cupy.einsum('i,j->ji', rad_weight[idx], grid[:,3]).ravel())

            atom_grids_tab[symb] = (cupy.vstack(coords), cupy.hstack(vol))

    return atom_grids_tab

def get_partition(mol, atom_grids_tab,
                  radii_adjust=None, atomic_radii=radi.BRAGG_RADII,
                  becke_scheme=original_becke, concat=True):
    '''Generate the mesh grid coordinates and weights for DFT numerical integration.
    We can change radii_adjust, becke_scheme functions to generate different meshgrid.

    Kwargs:
        concat: bool
            Whether to concatenate grids and weights in return

    Returns:
        grid_coord and grid_weight arrays.  grid_coord array has shape (N,3);
        weight 1D array has N elements.
    '''
    assert becke_scheme is original_becke
    atm_coords = cupy.asarray(mol.atom_coords() , order='F')
    atm_ngrids = numpy.array([atom_grids_tab[mol.atom_symbol(ia)][1].size
                              for ia in range(mol.natm)])
    ngrids = atm_ngrids.sum()
    coords = cupy.empty((ngrids, 3), order='F')
    weights = cupy.empty(ngrids)
    atm_idx = cupy.empty(ngrids, dtype=numpy.int32)
    p0 = p1 = 0
    for ia in range(mol.natm):
        r, vol = atom_grids_tab[mol.atom_symbol(ia)]
        p0, p1 = p1, p1 + vol.size
        coords[p0:p1] = r
        coords[p0:p1] += atm_coords[ia]
        weights[p0:p1] = vol
        atm_idx[p0:p1] = ia

    # support atomic_radii_adjust = None
    assert radii_adjust == radi.treutler_atomic_radii_adjust
    a = -radi.get_treutler_fac(mol, atomic_radii)
    #a = -radi.get_becke_fac(mol, atomic_radii)
    err = libgdft.GDFTbecke_partition_weights(
        ctypes.cast(weights.data.ptr, ctypes.c_void_p),
        ctypes.cast(coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(atm_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(a.data.ptr, ctypes.c_void_p),
        ctypes.cast(atm_idx.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(mol.natm)
    )
    if err != 0:
        raise RuntimeError('GDFTbecke_partition_weights kernel failed')
    if not concat:
        offsets = numpy.cumsum(atm_ngrids)
        coords = cupy.split(coords, offsets[:-1])
        weights = cupy.split(weights, offsets[:-1])
    return coords, weights
gen_partition = get_partition

def make_mask(mol, coords, relativity=0, shls_slice=None, cutoff=CUTOFF,
              verbose=None):
    '''Mask to indicate whether a shell is ignorable on grids. See also the
    function gto.eval_gto.make_screen_index

    Args:
        mol : an instance of :class:`Mole`

        coords : 2D array, shape (N,3)
            The coordinates of grids.

    Kwargs:
        relativity : bool
            No effects.
        shls_slice : 2-element list
            (shl_start, shl_end).
            If given, only part of AOs (shl_start <= shell_id < shl_end) are
            evaluated.  By default, all shells defined in mol will be evaluated.
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        2D mask array of shape (N,nbas), where N is the number of grids, nbas
        is the number of shells.
    '''
    if isinstance(coords, cupy.ndarray):
        coords = coords.get()
    return make_screen_index(mol, coords, shls_slice, cutoff)

def argsort_group(group_ids, ngroup):
    '''Sort the grids based on the group_ids.
    '''
    groups = []
    for i in range(ngroup):
        groups.append(cupy.argwhere(group_ids==i)[0])
    return cupy.hstack(groups)

def atomic_group_grids(mol, coords):
    '''
    partition the entire space based on atomic position
    '''
    from scipy.spatial import distance_matrix
    natm = mol.natm
    ngrids = coords.shape[0]
    atom_coords = mol.atom_coords()
    dist = distance_matrix(atom_coords, atom_coords)
    visited = numpy.zeros(natm, dtype=bool)
    current_node = numpy.argmin(atom_coords[:,0])
    # greedy traverse atoms
    path = [current_node]
    while len(path) < natm:
        visited[current_node] = True
        # Set distances to visited nodes as infinity so they won't be chosen
        distances_to_unvisited = numpy.where(visited, numpy.inf, dist[current_node])
        next_node = numpy.argmin(distances_to_unvisited)
        path.append(next_node)
        current_node = next_node
    atom_coords = cupy.asarray(atom_coords[path])

    coords = cupy.asarray(coords, order='F')
    atom_coords = cupy.asarray(atom_coords, order='F')
    group_ids = cupy.empty([ngrids], dtype=numpy.int32)
    stream = cupy.cuda.get_current_stream()
    err = libgdft.GDFTgroup_grids(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(group_ids.data.ptr, ctypes.c_void_p),
        ctypes.cast(atom_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(coords.data.ptr, ctypes.c_void_p),
        ctypes.c_int(natm),
        ctypes.c_int(ngrids)
    )
    if err != 0:
        raise RuntimeError('CUDA Error')
    idx = group_ids.argsort()
    return idx

def arg_group_grids(mol, coords, box_size=GROUP_BOX_SIZE):
    '''
    Parition the entire space into small boxes according to the input box_size.
    Group the grids against these boxes.
    '''
    atom_coords = mol.atom_coords()
    boundary = [atom_coords.min(axis=0) - GROUP_BOUNDARY_PENALTY,
                atom_coords.max(axis=0) + GROUP_BOUNDARY_PENALTY]
    # how many boxes inside the boundary
    boxes = ((boundary[1] - boundary[0]) * (1./box_size)).round().astype(int)
    tot_boxes = numpy.prod(boxes + 2)
    logger.debug(mol, 'tot_boxes %d, boxes in each direction %s', tot_boxes, boxes)
    # box_size is the length of each edge of the box
    box_size = cupy.asarray((boundary[1] - boundary[0]) / boxes)
    frac_coords = (coords - cupy.asarray(boundary[0])) * (1./box_size)
    box_ids = cupy.floor(frac_coords).astype(int)
    box_ids[box_ids<-1] = -1
    box_ids[box_ids[:,0] > boxes[0], 0] = boxes[0]
    box_ids[box_ids[:,1] > boxes[1], 1] = boxes[1]
    box_ids[box_ids[:,2] > boxes[2], 2] = boxes[2]

    boxes *= 2 # for safety
    box_id = box_ids[:,0] + box_ids[:,1] * boxes[0] + box_ids[:,2] * boxes[0] * boxes[1]
    #rev_idx = numpy.unique(box_ids.get(), axis=0, return_inverse=True)[1]
    rev_idx = cupy.unique(box_id, return_inverse=True)[1]
    return rev_idx.argsort()

def _load_conf(mod, name, default):
    var = getattr(__config__, name, None)
    if var is None:
        var = default
    elif isinstance(var):
        if mod is None:
            mod = sys.modules[__name__]
        var = getattr(mod, var)

    if callable(var):
        return staticmethod(var)
    else:
        return var

class Grids(lib.StreamObject):

    from gpu4pyscf.lib.utils import to_gpu, device

    atomic_radii = _load_conf(radi, 'dft_gen_grid_Grids_atomic_radii',
                                   radi.BRAGG_RADII)
    radii_adjust = _load_conf(radi, 'dft_gen_grid_Grids_radii_adjust',
                                   radi.treutler_atomic_radii_adjust)
    radi_method = _load_conf(radi, 'dft_gen_grid_Grids_radi_method',
                                  radi.treutler)
    becke_scheme = _load_conf(None, 'dft_gen_grid_Grids_becke_scheme',
                              original_becke)
    prune = _load_conf(None, 'dft_gen_grid_Grids_prune', nwchem_prune)
    level = getattr(__config__, 'dft_gen_grid_Grids_level', 3)
    alignment    = ALIGNMENT_UNIT
    cutoff       = CUTOFF
    _keys        = gen_grid_cpu.Grids._keys.union({
        'grid_sorting_index', 'atm_idx', 'padding'
    })

    __init__   = gen_grid_cpu.Grids.__init__
    dump_flags = gen_grid_cpu.Grids.dump_flags

    def __setattr__(self, key, val):
        if key in ('atom_grid', 'atomic_radii', 'radii_adjust', 'radi_method',
                   'becke_scheme', 'prune', 'level'):
            self.reset()
        super().__setattr__(key, val)

    @property
    def size(self):
        return getattr(self.weights, 'size', 0)

    def build(self, mol=None, with_non0tab=False, sort_grids=True, **kwargs):
        if mol is None: mol = self.mol
        if self.verbose >= logger.WARN:
            self.check_sanity()
        log = logger.new_logger(self)
        t0 = log.init_timer()
        atom_grids_tab = self.gen_atomic_grids(
            mol, self.atom_grid, self.radi_method, self.level, self.prune, **kwargs)
        self.coords, self.weights = self.get_partition(
            mol, atom_grids_tab, self.radii_adjust, self.atomic_radii, self.becke_scheme)

        atm_idx = cupy.empty(self.coords.shape[0], dtype=numpy.int32)
        quadrature_weights = cupy.empty(self.coords.shape[0])
        p0 = p1 = 0
        for ia in range(mol.natm):
            r, vol = atom_grids_tab[mol.atom_symbol(ia)]
            p0, p1 = p1, p1 + vol.size
            atm_idx[p0:p1] = ia
            quadrature_weights[p0:p1] = vol
        self.atm_idx = atm_idx
        self.quadrature_weights = quadrature_weights

        t0 = log.timer_debug1('generating atomic grids', *t0)
        if self.alignment > 1:
            padding = _padding_size(self.size, self.alignment)
            log.debug('Padding %d grids', padding)
            if padding > 0:
                # cupy.vstack and cupy.hstack convert numpy array into cupy array first
                self.coords = cupy.vstack(
                    [self.coords, cupy.full((padding, 3), 1e-4)])
                self.weights = cupy.hstack([self.weights, cupy.zeros(padding)])
                self.quadrature_weights = cupy.hstack([self.quadrature_weights, cupy.zeros(padding)])
                self.atm_idx = cupy.hstack([self.atm_idx, cupy.full(padding, -1, dtype=numpy.int32)])

        if sort_grids:
            #idx = arg_group_grids(mol, self.coords)
            idx = atomic_group_grids(mol, self.coords)
            self.coords = self.coords[idx]
            self.weights = self.weights[idx]
            self.quadrature_weights = self.quadrature_weights[idx]
            self.atm_idx = self.atm_idx[idx]
            t0 = log.timer_debug1('sorting grids', *t0)

        if with_non0tab:
            self.non0tab = self.make_mask(mol, self.coords)
            self.screen_index = self.non0tab
            t0 = log.timer_debug1('generating grids mask', *t0)
        else:
            self.screen_index = self.non0tab = None
        log.info('tot grids = %d', len(self.weights))

        # (idx, non0shl_idx, ctr_offsets_slice, ao_loc_slice)
        self._non0ao_idx = None
        return self

    def kernel(self, mol=None, with_non0tab=False):
        self.dump_flags()
        return self.build(mol, with_non0tab=with_non0tab)

    def reset(self, mol=None):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        if mol is not None:
            self.mol = mol
        self.coords = None
        self.weights = None
        self.non0tab = None
        self.screen_index = None
        self._non0ao_idx = None
        return self

    gen_atomic_grids = lib.module_method(
        gen_atomic_grids, ['atom_grid', 'radi_method', 'level', 'prune'])

    @lib.with_doc(get_partition.__doc__)
    def get_partition(self, mol, atom_grids_tab=None,
                      radii_adjust=None, atomic_radii=radi.BRAGG_RADII,
                      becke_scheme=original_becke, concat=True):
        if atom_grids_tab is None:
            atom_grids_tab = self.gen_atomic_grids(mol)
        return get_partition(mol, atom_grids_tab, radii_adjust, atomic_radii,
                             becke_scheme, concat=concat)

    gen_partition = get_partition

    make_mask = lib.module_method(make_mask, absences=['cutoff'])

    def prune_by_density_(self, rho, threshold=0):
        '''Prune grids if the electron density on the grid is small'''
        if threshold == 0:
            return self

        mol = self.mol
        n = cupy.dot(rho, self.weights)
        if abs(n-mol.nelectron) < NELEC_ERROR_TOL*n:
            rho *= self.weights
            idx = abs(rho) > threshold / self.weights.size
            self.coords  = cupy.asarray(self.coords [idx], order='C')
            self.weights = cupy.asarray(self.weights[idx], order='C')
            self.atm_idx = cupy.asarray(self.atm_idx[idx], order='C')
            self.quadrature_weights = cupy.asarray(self.quadrature_weights[idx], order='C')
            logger.debug(self, 'Drop grids %d', rho.size - self.weights.size)
            if self.alignment > 1:
                padding = _padding_size(self.size, self.alignment)
                logger.debug(self, 'prune_by_density_: %d padding grids', padding)
                if padding > 0:
                    self.coords = cupy.vstack(
                        [self.coords, cupy.full((padding, 3), 1e-4)])
                    self.weights = cupy.hstack([self.weights, cupy.zeros(padding)])
                    self.quadrature_weights = cupy.hstack([self.quadrature_weights, cupy.zeros(padding)])
                    self.atm_idx = cupy.hstack([self.atm_idx, cupy.full(padding, -1, dtype=numpy.int32)])
            if self.non0tab is not None:
                # with_non0tab is enalbed when initialling the grids. Update the
                # screen_index for the pruned grids
                self.non0tab = self.make_mask(mol, self.coords)
                self.screen_index = self.non0tab
        else:
            logger.debug(self, 'Electron density is not accurate enough. '
                         'Grids are not pruned.')

        # The existing cache stores the indices for old grids, should be cleared.
        self._non0ao_idx = None
        return self

    def _build_non0ao_idx_cache(self, opt=None):
        '''cache ao indices'''
        from gpu4pyscf.dft import numint
        if opt is None:
            opt = numint._GDFTOpt.from_mol(self.mol)
        mol = opt._sorted_mol
        log = logger.new_logger(mol, mol.verbose)
        t1 = log.init_timer()
        stream = cp.cuda.get_current_stream()

        coords = cp.asarray(self.coords.T, order='C')
        _sorted_mol = opt._sorted_mol
        ao_loc = _sorted_mol.ao_loc_nr()
        nao = ao_loc[-1]
        nbas = len(ao_loc) - 1
        ngrids = self.size
        cutoff = numint.AO_THRESHOLD
        block_size = numint.MIN_BLK_SIZE
        nblocks = (ngrids + block_size - 1) // block_size
        non0shl_mask = cp.zeros((nblocks, nbas), dtype=np.int8)
        coords = cp.asarray(self.coords, order='F')
        _atm_gpu = cp.asarray(_sorted_mol._atm, dtype=np.int32)
        _bas_gpu = cp.asarray(_sorted_mol._bas, dtype=np.int32)
        _env_gpu = cp.asarray(_sorted_mol._env, dtype=np.float64)

        libgdft.GDFTscreen_index(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            ctypes.cast(non0shl_mask.data.ptr, ctypes.c_void_p),
            ctypes.c_double(np.log(cutoff)),
            ctypes.cast(coords.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ngrids), ctypes.c_int(block_size),
            ctypes.cast(_atm_gpu.data.ptr, ctypes.c_void_p),
            ctypes.c_int(len(_atm_gpu)),
            ctypes.cast(_bas_gpu.data.ptr, ctypes.c_void_p),
            ctypes.c_int(len(_bas_gpu)),
            ctypes.cast(_env_gpu.data.ptr, ctypes.c_void_p))

        # offset of contraction pattern, used in eval_ao
        l_ctr_offsets = opt.l_ctr_offsets
        non0shl_counts = cp.zeros(nblocks, dtype=np.int32)
        ctr_offsets_slice = [non0shl_counts]
        for i, (p0, p1) in enumerate(zip(l_ctr_offsets[:-1], l_ctr_offsets[1:])):
            non0shl_counts = non0shl_counts + cp.count_nonzero(non0shl_mask[:,p0:p1], axis=1)
            ctr_offsets_slice.append(non0shl_counts)

        non0shl_idx_sections = non0shl_counts.cumsum()[:-1].get()
        non0shl_mask = non0shl_mask.view(bool)
        non0shl_idx = cp.where(non0shl_mask)[1].astype(np.int32).get()

        ao_dims = ao_loc[1:] - ao_loc[:-1]
        ao_seg_idx = np.split(np.arange(nao, dtype=np.int32), ao_loc[1:-1])
        idx = []
        ao_loc_slice = []
        for _non0shl_idx in np.split(non0shl_idx, non0shl_idx_sections):
            if len(_non0shl_idx) == 0:
                idx.append(np.empty(0, dtype=np.int32))
                ao_loc_slice.append(np.zeros(1, dtype=np.int32))
                continue
            idx_in_block = [ao_seg_idx[x] for x in _non0shl_idx]
            idx.append(np.hstack(idx_in_block))
            _offsets = np.append(np.int32(0), ao_dims[_non0shl_idx]).cumsum(dtype=np.int32)
            ao_loc_slice.append(_offsets)

        idx_sections = np.cumsum([len(x) for x in idx])[:-1]
        ao_loc_slice_sections = np.cumsum([len(x) for x in ao_loc_slice])[:-1]
        idx = np.asarray(np.hstack(idx), dtype=np.int32)
        ao_loc_slice = np.asarray(np.hstack(ao_loc_slice), dtype=np.int32)
        ctr_offsets_slice = np.asarray(
            cp.stack(ctr_offsets_slice).T.get(order='C'), dtype=np.int32)

        non0ao_idx = ((idx, idx_sections),
                      (non0shl_idx, non0shl_idx_sections),
                      ctr_offsets_slice,
                      (ao_loc_slice, ao_loc_slice_sections))
        t1 = log.timer_debug2('init ao sparsity', *t1)
        return non0ao_idx

    def get_non0ao_idx(self, opt=None):
        if self._non0ao_idx is None:
            self._non0ao_idx = self._build_non0ao_idx_cache(opt)

        ((idx, idx_sections),
         (non0shl_idx, non0shl_idx_sections),
         ctr_offsets_slice,
         (ao_loc_slice, ao_loc_slice_sections)) = self._non0ao_idx
        idx = cp.split(asarray(idx, dtype=np.int32), idx_sections)
        non0shl_idx = cp.split(asarray(non0shl_idx, dtype=np.int32), non0shl_idx_sections)
        ao_loc_slice = cp.split(asarray(ao_loc_slice, dtype=np.int32), ao_loc_slice_sections)
        paddings = [0] * len(idx)
        return list(zip(paddings, idx, non0shl_idx, ctr_offsets_slice, ao_loc_slice))

    def to_cpu(self):
        grids = gen_grid_cpu.Grids(self.mol)
        utils.to_cpu(self, out=grids)
        return grids

_default_rad = gen_grid_cpu._default_rad
RAD_GRIDS = gen_grid_cpu.RAD_GRIDS
_default_ang = gen_grid_cpu._default_ang
ANG_ORDER = gen_grid_cpu.ANG_ORDER
_padding_size = gen_grid_cpu._padding_size
