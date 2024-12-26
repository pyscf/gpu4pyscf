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


import functools
import numpy as np
import scipy.linalg
from pyscf import gto
from pyscf.gto import (ANG_OF, ATOM_OF, NPRIM_OF, NCTR_OF, PTR_COORD, PTR_COEFF,
                       PTR_EXP)
from gpu4pyscf.lib import logger

PTR_BAS_COORD = 7

@functools.lru_cache(20)
def get_cart2sph(lmax=12):
    cart2sph = []
    for l in range(lmax):
        c2s = gto.mole.cart2sph(l, normalized='sp')
        cart2sph.append(np.asarray(c2s, order='C'))
    return cart2sph

def basis_seg_contraction(mol, allow_replica=1):
    '''transform generally contracted basis to segment contracted basis
    Kwargs:
        allow_replica:
            when angular momentum lower than (or equal to) this value, transform
            the generally contracted basis to replicated segment-contracted basis.
            By default, high angular momentum functions (d, f shells) are fully
            uncontracted.
    '''
    # Ensure backward compatibility. When allow_replica is True, decontraction
    # to primitive functions is disabled. When allow_replica is False, all
    # general contraction are decontracted.
    if allow_replica is True:
        allow_replica = 8
    elif allow_replica is False:
        allow_replica = -1

    bas_templates = {}
    _bas = []
    _env = mol._env.copy()
    contr_coeff = []
    aoslices = mol.aoslice_by_atom()
    for ia, (ib0, ib1) in enumerate(aoslices[:,:2]):
        key = tuple(mol._bas[ib0:ib1,PTR_COEFF])
        if key in bas_templates:
            bas_of_ia, coeff = bas_templates[key]
            bas_of_ia = bas_of_ia.copy()
            bas_of_ia[:,ATOM_OF] = ia
        else:
            # Generate the template for decontracted basis
            coeff = []
            bas_of_ia = []
            for shell in mol._bas[ib0:ib1]:
                l = shell[ANG_OF]
                nf = (l + 1) * (l + 2) // 2
                nctr = shell[NCTR_OF]
                if nctr == 1:
                    bas_of_ia.append(shell)
                    coeff.append(np.eye(nf))
                    continue
                # Only basis with nctr > 1 needs to be decontracted
                nprim = shell[NPRIM_OF]
                pcoeff = shell[PTR_COEFF]
                if l <= allow_replica:
                    coeff.extend([np.eye(nf)] * nctr)
                    bs = np.repeat(shell[np.newaxis], nctr, axis=0)
                    bs[:,NCTR_OF] = 1
                    bs[:,PTR_COEFF] = np.arange(pcoeff, pcoeff+nprim*nctr, nprim)
                    bas_of_ia.append(bs)
                else: # To avoid recomputation, decontract to primitive functions
                    pexp = shell[PTR_EXP]
                    exps = _env[pexp:pexp+nprim]
                    norm = gto.gto_norm(l, exps)
                    # remove normalization from contraction coefficients
                    c = _env[pcoeff:pcoeff+nprim*nctr].reshape(nctr,nprim)
                    c = np.einsum('ip,p,ef->iepf', c, 1/norm, np.eye(nf))
                    coeff.append(c.reshape(nf*nctr, nf*nprim).T)

                    _env[pcoeff:pcoeff+nprim] = norm
                    bs = np.repeat(shell[np.newaxis], nprim, axis=0)
                    bs[:,NPRIM_OF] = 1
                    bs[:,NCTR_OF] = 1
                    bs[:,PTR_EXP] = np.arange(pexp, pexp+nprim)
                    bs[:,PTR_COEFF] = np.arange(pcoeff, pcoeff+nprim)
                    bas_of_ia.append(bs)

            if len(bas_of_ia) > 0:
                bas_of_ia = np.vstack(bas_of_ia)
                bas_templates[key] = (bas_of_ia, coeff)
            else:
                continue

        _bas.append(bas_of_ia)
        contr_coeff.extend(coeff)

    pmol = mol.copy()
    pmol.cart = True
    pmol._bas = np.asarray(np.vstack(_bas), dtype=np.int32)
    pmol._env = _env
    contr_coeff = scipy.linalg.block_diag(*contr_coeff)

    if not mol.cart:
        contr_coeff = contr_coeff.dot(mol.cart2sph_coeff())
    return pmol, contr_coeff

def sort_atoms(mol):
    """
    Sort atoms in a molecule based on their distance to the first atom.

    Parameters:
    mol (Mole): A molecule object

    Returns:
    list: A list of atom indices sorted in ascending order based on their distance.

    """
    from scipy.spatial import distance_matrix
    charges = mol.atom_charges()
    heavy_atoms = np.argwhere(charges != 1).ravel()
    if heavy_atoms.size == 0:
        return range(mol.natm)

    atom_coords = mol.atom_coords()
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
    if hydrogen_atoms.size > 0:
        dist = distance_matrix(atom_coords[hydrogen_atoms], atom_coords[heavy_atoms])
        for i, d in enumerate(dist):
            heavy_idx = np.argmin(d)
            full_path[heavy_idx].append(hydrogen_atoms[i])

    return [x for heavy_list in full_path for x in heavy_list]

def group_basis(mol, tile=1, group_size=None):
    '''Group basis functions according to their [l, nprim] patterns'''
    mol, coeff = basis_seg_contraction(mol)
    # Sort basis according to angular momentum and contraction patterns so
    # as to group the basis functions to blocks in GPU kernel.
    l_ctrs = mol._bas[:,[ANG_OF, NPRIM_OF]]
    # Ensure the more contracted Gaussians being accessed first
    l_ctrs_descend = l_ctrs.copy()
    l_ctrs_descend[:,1] = -l_ctrs[:,1]
    uniq_l_ctr, where, inv_idx, l_ctr_counts = np.unique(
        l_ctrs_descend, return_index=True, return_inverse=True, return_counts=True, axis=0)
    uniq_l_ctr[:,1] = -uniq_l_ctr[:,1]

    nao_orig = coeff.shape[1]
    ao_loc = mol.ao_loc
    coeff = np.split(coeff, ao_loc[1:-1], axis=0)

    pad_bas = []
    if tile > 1:
        l_ctr_counts_orig = l_ctr_counts.copy()
        pad_inv_idx = []
        env_ptr = mol._env.size
        # for each pattern, padding basis to the end of mol._bas, ensure alignment to tile
        for n, (l_ctr, m, counts) in enumerate(zip(uniq_l_ctr, where, l_ctr_counts)):
            if counts % tile == 0: continue
            n_alined = (counts+tile-1) & (0x100000-tile)
            padding = n_alined - counts
            l_ctr_counts[n] = n_alined

            bas = mol._bas[m].copy()
            bas[PTR_COEFF] = env_ptr
            pad_bas.extend([bas] * padding)
            pad_inv_idx.extend([n] * padding)

            l = l_ctr[0]
            nf = (l + 1) * (l + 2) // 2
            coeff.extend([np.zeros((nf, nao_orig))] * padding)

        inv_idx = np.hstack([inv_idx.ravel(), pad_inv_idx])

    sorted_idx = np.argsort(inv_idx.ravel(), kind='stable').astype(np.int32)
    coeff = np.vstack([coeff[i] for i in sorted_idx])
    assert coeff.shape[0] < 32768

    max_nprims = uniq_l_ctr[:,1].max()
    mol._env = np.append(mol._env, np.zeros(max_nprims))
    if pad_bas:
        mol._bas = np.vstack([mol._bas, pad_bas])[sorted_idx]
    else:
        mol._bas = mol._bas[sorted_idx]
    assert mol._bas.dtype == np.int32

    ## Limit the number of AOs in each group
    if group_size is not None:
        uniq_l_ctr, l_ctr_counts = _split_l_ctr_groups(
            uniq_l_ctr, l_ctr_counts, group_size, tile)

    if mol.verbose >= logger.DEBUG1:
        logger.debug1(mol, 'Number of shells for each [l, nprim] group')
        if tile > 1:
            for l_ctr, n, n8 in zip(uniq_l_ctr, l_ctr_counts_orig, l_ctr_counts):
                logger.debug1(mol, '    %s : %s -> %s', l_ctr, n, n8)
        else:
            for l_ctr, n in zip(uniq_l_ctr, l_ctr_counts):
                logger.debug1(mol, '    %s : %s', l_ctr, n)

    # PTR_BAS_COORD is required by various CUDA kernels
    mol._bas[:,PTR_BAS_COORD] = mol._atm[mol._bas[:,ATOM_OF],PTR_COORD]
    return mol, coeff, uniq_l_ctr, l_ctr_counts

def _split_l_ctr_groups(uniq_l_ctr, l_ctr_counts, group_size, align=1):
    '''Splits l_ctr patterns into small groups with group_size the maximum
    number of AOs in each group
    '''
    l = uniq_l_ctr[:,0]
    nf = l * (l + 1) // 2
    _l_ctrs = []
    _l_ctr_counts = []
    for l_ctr, counts in zip(uniq_l_ctr, l_ctr_counts):
        l = l_ctr[0]
        nf = (l + 1) * (l + 2) // 2
        max_shells = max(group_size//nf-align+1, align, 2)
        max_shells = (max_shells + align - 1) & (0x100000-align)
        if counts <= max_shells:
            _l_ctrs.append(l_ctr)
            _l_ctr_counts.append(counts)
            continue

        nsubs, remaining = counts.__divmod__(max_shells)
        _l_ctrs.extend([l_ctr] * nsubs)
        _l_ctr_counts.extend([max_shells] * nsubs)
        if remaining > 0:
            _l_ctrs.append(l_ctr)
            _l_ctr_counts.append(remaining)
    uniq_l_ctr = np.vstack(_l_ctrs)
    l_ctr_counts = np.hstack(_l_ctr_counts)
    return uniq_l_ctr, l_ctr_counts
