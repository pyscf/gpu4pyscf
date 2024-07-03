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
import copy
from pyscf import gto

@functools.lru_cache(20)
def get_cart2sph(lmax=12):
    cart2sph = []
    for l in range(lmax):
        c2s = gto.mole.cart2sph(l, normalized='sp')
        cart2sph.append(np.asarray(c2s, order='C'))
    return cart2sph

def basis_seg_contraction(mol, allow_replica=False):
    '''transform generally contracted basis to segment contracted basis
    Kwargs:
        allow_replica:
            transform the generally contracted basis to replicated
            segment-contracted basis
    '''
    bas_templates = {}
    _bas = []
    _env = mol._env.copy()

    aoslices = mol.aoslice_by_atom()
    for ia, (ib0, ib1) in enumerate(aoslices[:,:2]):
        key = tuple(mol._bas[ib0:ib1,gto.PTR_EXP])
        if key in bas_templates:
            bas_of_ia = bas_templates[key]
            bas_of_ia = bas_of_ia.copy()
            bas_of_ia[:,gto.ATOM_OF] = ia
        else:
            # Generate the template for decontracted basis
            bas_of_ia = []
            for shell in mol._bas[ib0:ib1]:
                l = shell[gto.ANG_OF]
                nctr = shell[gto.NCTR_OF]
                if nctr == 1:
                    bas_of_ia.append(shell)
                    continue

                # Only basis with nctr > 1 needs to be decontracted
                nprim = shell[gto.NPRIM_OF]
                pcoeff = shell[gto.PTR_COEFF]
                if allow_replica:
                    bs = np.repeat(shell[np.newaxis], nctr, axis=0)
                    bs[:,gto.NCTR_OF] = 1
                    bs[:,gto.PTR_COEFF] = np.arange(pcoeff, pcoeff+nprim*nctr, nprim)
                    bas_of_ia.append(bs)
                else:
                    pexp = shell[gto.PTR_EXP]
                    exps = _env[pexp:pexp+nprim]
                    norm = gto.gto_norm(l, exps)
                    # remove normalization from contraction coefficients
                    _env[pcoeff:pcoeff+nprim] = norm
                    bs = np.repeat(shell[np.newaxis], nprim, axis=0)
                    bs[:,gto.NPRIM_OF] = 1
                    bs[:,gto.NCTR_OF] = 1
                    bs[:,gto.PTR_EXP] = np.arange(pexp, pexp+nprim)
                    bs[:,gto.PTR_COEFF] = np.arange(pcoeff, pcoeff+nprim)
                    bas_of_ia.append(bs)

            bas_of_ia = np.vstack(bas_of_ia)
            bas_templates[key] = bas_of_ia
        _bas.append(bas_of_ia)

    pmol = copy.copy(mol)
    pmol.output = mol.output
    pmol.verbose = mol.verbose
    pmol.stdout = mol.stdout
    pmol.cart = True
    pmol._bas = np.asarray(np.vstack(_bas), dtype=np.int32)
    pmol._env = _env
    return pmol

def sort_atoms(mol):
    """
    Sort atoms in a molecule based on their distance to the first atom.

    Parameters:
    mol (Mole): A molecule object

    Returns:
    list: A list of atom indices sorted in ascending order based on their distance.

    """
    from scipy.spatial import distance_matrix
    atom_coords = mol.atom_coords()
    charges = mol.atom_charges()
    heavy_atoms = np.argwhere(charges != 1).ravel()
    
    visited = np.zeros(len(heavy_atoms), dtype=bool)
    heavy_coords = atom_coords[heavy_atoms,:]
    current_node = np.argmin(heavy_coords[:,0])
    dist = distance_matrix(atom_coords[heavy_atoms], atom_coords[heavy_atoms])

    # greedy traverse heavy atoms
    path = [current_node]
    while len(path) < len(heavy_atoms):
        visited[current_node] = True
        # Set distances to visited nodes as infinity so they won't be chosen
        distances_to_unvisited = np.where(visited, np.inf, dist[current_node]).ravel()
        next_node = np.argmin(distances_to_unvisited)
        path.append(next_node)
        current_node = next_node
    
    # Assign Hydrogen atoms to heavy atoms
    full_path = [[heavy_atoms[idx]] for idx in path]
    hydrogen_atoms = np.argwhere(charges == 1).ravel()
    dist = distance_matrix(atom_coords[hydrogen_atoms], atom_coords[heavy_atoms])
    for i, d in enumerate(dist):
        heavy_idx = np.argmin(d)
        full_path[heavy_idx].append(hydrogen_atoms[i])
    
    return [x for heavy_list in full_path for x in heavy_list]
