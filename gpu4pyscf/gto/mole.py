# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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
import cupy as cp
import scipy.linalg
from pyscf import gto
from pyscf.gto import (ANG_OF, ATOM_OF, NPRIM_OF, NCTR_OF, PTR_COORD, PTR_COEFF,
                       PTR_EXP)

PTR_BAS_COORD = 7

_c2s = {}

def cart2sph_by_l(l, normalized='sp'):
    device_id = cp.cuda.Device().id
    if (l, device_id, normalized) not in _c2s:
        c2s = gto.mole.cart2sph(l, normalized=normalized)
        _c2s[l,device_id,normalized] = cp.asarray(c2s, order='C')
    return _c2s[l,device_id,normalized]

def basis_seg_contraction(mol, allow_replica=1, sparse_coeff=False):
    '''transform generally contracted basis to segment contracted basis.
    Note return_mol.cart is set to True.

    Kwargs:
        allow_replica:
            when angular momentum lower than (or equal to) this value, transform
            the generally contracted basis to replicated segment-contracted basis.
            By default, high angular momentum functions (d, f shells) are fully
            uncontracted.
    '''
    from gpu4pyscf.lib.cupy_helper import block_diag, asarray
    # Ensure backward compatibility. When allow_replica is True, decontraction
    # to primitive functions is disabled. When allow_replica is False, all
    # general contraction are decontracted.
    if allow_replica is True:
        allow_replica = 8
    elif allow_replica is False:
        allow_replica = -1

    # Preallocate a buffer in cupy memory pool for small arrays held in bas_templates
    workspace = cp.empty(30**2*100)
    workspace = None # noqa: F841
    bas_templates = {}
    lmax = mol._bas[:,ANG_OF].max()
    if mol.cart:
        c2s = [np.eye((l+1)*(l+2)//2) for l in range(lmax+1)]
    else:
        c2s = [gto.mole.cart2sph(l, normalized='sp') for l in range(lmax+1)]
    c2s_gpu = [asarray(c, order='C') for c in c2s]
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
                    coeff.append(c2s_gpu[l])
                    continue
                # Only basis with nctr > 1 needs to be decontracted
                nprim = shell[NPRIM_OF]
                pcoeff = shell[PTR_COEFF]
                if l <= allow_replica:
                    coeff.extend([c2s_gpu[l]] * nctr)
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
                    c = np.einsum('ip,p,fe->pfie', c, 1/norm, c2s[l])
                    coeff.append(asarray(c.reshape(nf*nprim,-1), order='C'))

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
        if not sparse_coeff:
            contr_coeff.extend(coeff)

    pmol = mol.copy()
    pmol.cart = True
    pmol._bas = np.asarray(np.vstack(_bas), dtype=np.int32)
    pmol._env = _env

    if not sparse_coeff:
        contr_coeff = block_diag(contr_coeff)
        return pmol, contr_coeff
    else:
        return pmol, None

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

def group_basis(mol, tile=1, group_size=None, return_bas_mapping=False, sparse_coeff=False):
    '''Group basis functions according to their [l, nprim] patterns.

    bas_mapping is the index that transforms _bas from sorted_mol to mol:
    mol._bas = sorted_mol._bas[bas_mapping]
    '''
    from gpu4pyscf.lib import logger
    original_mol = mol

    mol, coeff = basis_seg_contraction(mol, sparse_coeff = sparse_coeff)

    # Sort basis according to angular momentum and contraction patterns so
    # as to group the basis functions to blocks in GPU kernel.
    l_ctrs = mol._bas[:,[ANG_OF, NPRIM_OF]]
    # Ensure the more contracted Gaussians being accessed first
    l_ctrs_descend = l_ctrs.copy()
    l_ctrs_descend[:,1] = -l_ctrs[:,1]
    uniq_l_ctr, where, inv_idx, l_ctr_counts = np.unique(
        l_ctrs_descend, return_index=True, return_inverse=True, return_counts=True, axis=0)
    uniq_l_ctr[:,1] = -uniq_l_ctr[:,1]

    if not sparse_coeff:
        nao_orig = coeff.shape[1]
        ao_loc = mol.ao_loc
        coeff = cp.split(coeff, ao_loc[1:-1], axis=0)
    else:
        ao_loc = mol.ao_loc_nr(cart=original_mol.cart)
        ao_idx = np.array_split(np.arange(original_mol.nao), ao_loc[1:-1])
        sorted_idx = np.argsort(inv_idx.ravel(), kind='stable').astype(np.int32)
        ao_idx = np.hstack([ao_idx[i] for i in sorted_idx])

    pad_bas = []
    if tile > 1:
        assert not return_bas_mapping, 'bas_mapping requires tile=1'
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
            if not sparse_coeff:
                coeff.extend([cp.zeros((nf, nao_orig))] * padding)

        inv_idx = np.hstack([inv_idx.ravel(), pad_inv_idx])

    sorted_idx = np.argsort(inv_idx.ravel(), kind='stable').astype(np.int32)

    if_pad_bas = np.array([False] * mol.nbas + [True] * len(pad_bas))[sorted_idx]

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

    if not sparse_coeff:
        coeff = cp.vstack([coeff[i] for i in sorted_idx])
        assert coeff.shape[0] < 32768
        if return_bas_mapping:
            return mol, coeff, uniq_l_ctr, l_ctr_counts, sorted_idx.argsort()
        else:
            return mol, coeff, uniq_l_ctr, l_ctr_counts
    else:
        n_cartesian = sum([(l+1)*(l+2)//2 for l in mol._bas[:,ANG_OF]])
        assert n_cartesian < 32768
        l_ctr_offsets = np.cumsum(l_ctr_counts)[:-1]
        if_pad_bas_per_l_ctr = np.split(if_pad_bas, l_ctr_offsets)
        l_ctr_pad_counts = np.array([np.sum(if_pad) for if_pad in if_pad_bas_per_l_ctr])
        if return_bas_mapping:
            return mol, ao_idx, l_ctr_pad_counts, uniq_l_ctr, l_ctr_counts, sorted_idx.argsort()
        else:
            return mol, ao_idx, l_ctr_pad_counts, uniq_l_ctr, l_ctr_counts

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

def extract_pgto_params(mol, op='diffused'):
    '''A helper function to extract exponents and contraction coefficients of
    the most diffused or compact primitive GTOs for each shell. These exponents
    and coefficients are typically used in estimating rcut and Ecut for PBC
    methods.
    '''
    if op != 'diffused' and op != 'compact':
        raise RuntimeError(f'Unsupported operation {op}')

    e = np.hstack(mol.bas_exps())
    c = np.hstack([abs(mol._libcint_ctr_coeff(i)).max(axis=1)
                   for i in range(mol.nbas)])
    l = np.repeat(mol._bas[:,ANG_OF], mol._bas[:,NPRIM_OF])
    basis_id = np.repeat(np.arange(mol.nbas), mol._bas[:,NPRIM_OF])
    precision = 1e-8
    if op == 'diffused':
        # A quick estimation for the radius that each primitive GTO decays to the
        # value smaller than the required precision
        r2 = np.log(c**2/precision * 10**l + 1e-200) / e
        # groupby.argmin()
        r2_order = np.argsort(-r2)
        _, idx = np.unique(basis_id[r2_order], return_index=True)
        idx = r2_order[idx]
    else:
        # A quick estimation for the resolution of planewaves that each
        # primitive GTO requires
        ke = np.log(c**2 / precision * 50**l + 1e-200) * e
        # groupby.argmax()
        ke_order = np.argsort(-ke)
        _, idx = np.unique(basis_id[ke_order], return_index=True)
        idx = ke_order[idx]
    return e[idx], c[idx]
