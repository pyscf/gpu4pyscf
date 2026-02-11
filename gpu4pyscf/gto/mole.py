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


import ctypes
import numpy as np
import cupy as cp
import scipy.linalg
from pyscf import gto
from pyscf.pbc import gto as pbcgto
from pyscf.gto import (ANG_OF, ATOM_OF, NPRIM_OF, NCTR_OF, PTR_COORD, PTR_COEFF,
                       PTR_EXP)
from gpu4pyscf.lib.utils import load_library
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import block_diag, asarray

__all__ = [
    'cart2sph_by_l', 'basis_seg_contraction', 'group_basis',
    'extract_pgto_params', 'groupby', 'Mole', 'Cell', 'SortedCell',
    'SortedMole', 'RysIntEnvVars',
]

libvhf_rys = load_library('libgvhf_rys')

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

def group_basis(mol, tile=1, group_size=None, return_bas_mapping=False,
                sparse_coeff=False):
    '''Group and sort basis functions according to their [l, nprim] patterns.

    Kwargs:
        tile (int):
            Align the number of basis shells in each group to a multiple of tile.
            Basis functions with zero contraction coefficients may be padded to
            preserve alignment. Default is 1.
        group_size (int):
            Maximum number of basis shells within each group. Be default, no
            limit is applied.
        return_bas_mapping (bool):
            bas_mapping is an index array that can transform _bas from
            sorted_mol to mol: mol._bas = sorted_mol._bas[bas_mapping]
        sparse_coeff (bool):
            One-to-one mapping between the sorted_mol and mol is assumed.
            The array of mapping indices instead of a single transformation
            matrix is returned if this option is specified.
    '''
    from gpu4pyscf.lib import logger
    original_mol = mol

    # When sparse_coeff is enabled, an array of AO mapping indices will be
    # returned which can facilitate the transformation of the integral matrix
    # between sorted_mol and mol using fancy-indexing, without applying the
    # expensive C.T.dot(mat).dot(C). This fast transformation assumes one-one
    # mapping between the basis shells of the two types of mol instatnce,
    # ignoring general contraction. Enabling `allow_replica` will produce
    # replicated segment-contracted shells for general contracted shells.
    if sparse_coeff:
        mol, coeff = basis_seg_contraction(
            mol, allow_replica=True, sparse_coeff=sparse_coeff)
    else:
        mol, coeff = basis_seg_contraction(mol, sparse_coeff=sparse_coeff)

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
        l_ctr_offsets = np.cumsum(l_ctr_counts)[:-1]
        if_pad_bas_per_l_ctr = np.split(if_pad_bas, l_ctr_offsets)
        l_ctr_pad_counts = np.array([np.sum(if_pad) for if_pad in if_pad_bas_per_l_ctr])
        l_ctr_pad_counts = np.asarray(l_ctr_pad_counts, dtype=np.int32)
        if return_bas_mapping:
            return mol, ao_idx, l_ctr_pad_counts, uniq_l_ctr, l_ctr_counts, sorted_idx.argsort()
        else:
            return mol, ao_idx, l_ctr_pad_counts, uniq_l_ctr, l_ctr_counts

def _split_l_ctr_groups(uniq_l_ctr, l_ctr_counts, group_size, align=1):
    '''Splits l_ctr patterns into small groups with group_size the maximum
    number of AOs in each group
    '''
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

def extract_pgto_params(mol, op='diffuse'):
    '''A helper function to extract exponents and contraction coefficients of
    the most diffuse or compact primitive GTOs for each shell. These exponents
    and coefficients are typically used in estimating rcut and Ecut for PBC
    methods.
    '''
    op = op[:7]
    if op != 'diffuse' and op != 'compact':
        raise RuntimeError(f'Unsupported operation {op}')

    e = np.hstack(mol.bas_exps())
    c = np.hstack([abs(mol._libcint_ctr_coeff(i)).max(axis=1)
                   for i in range(mol.nbas)])
    l = np.repeat(mol._bas[:,ANG_OF], mol._bas[:,NPRIM_OF])
    basis_id = np.repeat(np.arange(mol.nbas), mol._bas[:,NPRIM_OF])
    precision = 1e-8
    if op == 'diffuse':
        # A quick estimation for the radius that each primitive GTO decays to the
        # value smaller than the required precision
        r2 = np.log(c**2/precision * 10**l + 1e-200) / e
        idx = groupby(basis_id, r2, 'argmax')
    else:
        # A quick estimation for the resolution of planewaves that each
        # primitive GTO requires
        ke = np.log(c**2 / precision * 50**l + 1e-200) * e
        idx = groupby(basis_id, ke, 'argmax')
    return e[idx], c[idx]

def groupby(labels, a, op='argmin'):
    '''Perform groupby(labels, a).op(). For example,
    groupby(['A', 'A', 'B'], [1, 2, 3], 'min') => [1, 3]
    '''
    if 'min' in op:
        a_order = a.argsort()
        _, idx = np.unique(labels[a_order], return_index=True)
        idx = a_order[idx]
    elif 'max' in op:
        a_order = a.argsort()[::-1]
        _, idx = np.unique(labels[a_order], return_index=True)
        idx = a_order[idx]
    elif op == 'sum':
        labels, inv = np.unique(labels, return_inverse=True)
        if a.ndim == 1:
            summed = np.bincount(inv, weights=a)
        else:
            summed = np.zeros((len(labels), *a.shape[1:]), dtype=a.dtype)
            np.add.at(summed, inv, a)
        return summed
    else:
        raise NotImplementedError

    if 'arg' in op:
        return idx
    else:
        return a[idx]

class Mole(gto.Mole):
    def __getattr__(self, key):
        '''To support accessing methods (mol.HF, mol.KS, mol.CCSD, mol.CASSCF, ...)
        from Mole object.
        '''
        if key[0] == '_':  # Skip private attributes and Python builtins
            return object.__getattribute__(self, key)

        from gpu4pyscf import scf, dft

        attr_name = key
        mf_xc = None
        for mod in (dft, scf):
            mf_method = getattr(mod, key, None)
            if callable(mf_method):
                key = None
                break
        else:
            if 'TD' in key[:3]:
                if 'TDA' in key:
                    if key == 'dTDA':
                        mf_method = dft.KS
                    else:
                        mf_method = 'SCF_TO_BE_DETERMINED'
                elif 'TDHF' in key:
                    mf_method = scf.HF
                elif 'TDDFT' in key:
                    mf_method = dft.KS
                else:
                    raise AttributeError(f'method {key} not supported')
            elif 'CI' in key or 'CC' in key or 'CAS' in key or 'MP' in key:
                mf_method = scf.HF
                raise NotImplementedError
            else:
                return object.__getattribute__(self, key)

        post_mf_key = key
        SCF_KW = {'xc', 'U_idx', 'U_val', 'C_ao_lo', 'minao_ref'}

        def fn(*args, **kwargs):
            if mf_xc is not None:
                assert 'xc' not in kwargs
                kwargs['xc'] = mf_xc

            mf_kw = {}
            remaining_kw = {}
            for k, v in kwargs.items():
                if k in SCF_KW:
                    mf_kw[k] = v
                else:
                    remaining_kw[k] = v
            if mf_method == 'SCF_TO_BE_DETERMINED':
                if 'xc' in mf_kw:
                    mf = dft.KS(self, **mf_kw)
                else:
                    mf = scf.HF(self, **mf_kw)
            else:
                mf = mf_method(self, **mf_kw)

            if post_mf_key is None:
                if args:
                    raise RuntimeError(
                        f'mol.{attr_name} function does not support positional arguments')
                return mf.set(**remaining_kw)

            post_mf = getattr(mf, post_mf_key)
            # Initialize SCF object for post-SCF methods if applicable
            if self.nelectron != 0:
                mf.run()
            return post_mf(*args, **remaining_kw)
        return gto.Mole._MoleLazyCallAdapter(fn, attr_name)

    def to_cpu(self):
        return self.view(gto.Mole)

    @classmethod
    def from_cpu(cls, mol):
        return mol.view(cls)

class Cell(pbcgto.cell.Cell):
    def __getattr__(self, key):
        '''To support accessing methods (cell.HF, cell.KKS, cell.KUCCSD, ...)
        from Cell object.
        '''
        if key[0] == '_':  # Skip private attributes and Python builtins
            return object.__getattribute__(self, key)

        from gpu4pyscf.pbc import scf, dft

        attr_name = key
        mf_xc = None
        for mod in (dft, scf):
            mf_method = getattr(mod, key, None)
            if callable(mf_method):
                key = None
                break
        else:
            if key[0] == 'K':  # with k-point sampling
                raise NotImplementedError
            else:
                if 'TD' in key[:3]:
                    if 'TDA' in key:
                        mf_method = 'SCF_TO_BE_DETERMINED'
                    elif 'TDHF' in key:
                        mf_method = scf.HF
                    elif 'TDDFT' in key:
                        mf_method = dft.KS
                    else:
                        raise AttributeError(f'method {key} not supported')
                elif 'CI' in key or 'CC' in key or 'MP' in key:
                    mf_method = scf.HF
                    raise NotImplementedError
                else:
                    return object.__getattribute__(self, key)

        post_mf_key = key
        SCF_KW = {'kpt', 'kpts', 'xc', 'exxdiv',
                  'U_idx', 'U_val', 'C_ao_lo', 'minao_ref'}

        def fn(*args, **kwargs):
            if mf_xc is not None:
                assert 'xc' not in kwargs
                kwargs['xc'] = mf_xc

            mf_kw = {}
            remaining_kw = {}
            for k, v in kwargs.items():
                if k in SCF_KW:
                    mf_kw[k] = v
                else:
                    remaining_kw[k] = v

            if mf_method == 'SCF_TO_BE_DETERMINED':
                if 'xc' in mf_kw:
                    mf = dft.KS(self, **mf_kw)
                else:
                    mf = scf.HF(self, **mf_kw)
            elif mf_method == 'KSCF_TO_BE_DETERMINED':
                if 'xc' in mf_kw:
                    mf = dft.KKS(self, **mf_kw)
                else:
                    mf = scf.KHF(self, **mf_kw)
            else:
                mf = mf_method(self, **mf_kw)

            if post_mf_key is None:
                if args:
                    raise RuntimeError(
                        f'cell.{attr_name} function does not support positional arguments')
                return mf.set(**remaining_kw)

            post_mf = getattr(mf, post_mf_key)
            if self.nelectron != 0:
                mf.run()
            return post_mf(*args, **remaining_kw)
        return gto.mole._MoleLazyCallAdapter(fn, attr_name)

    def to_cpu(self):
        return self.view(pbcgto.cell.Cell)

    @classmethod
    def from_cpu(cls, cell):
        return cell.view(cls)

class SortedGTO:
    @classmethod
    def from_mol(cls, mol, group_size=None,
                 allow_replica=True, allow_split_seg_contraction=False):
        if isinstance(mol, SortedGTO):
            return mol
        elif not isinstance(mol, (pbcgto.Cell, gto.Mole)):
            raise RuntimeError(f'SortedMole cannot be constructed from {mol}')

        self, recontract_bas, recontract_coef, pbas_idx = _recontract_basis(
            mol, allow_replica, allow_split_seg_contraction)
        if isinstance(mol, pbcgto.Cell):
            self = self.view(SortedCell)
        else:
            self = self.view(SortedMole)
        self.mol = self.cell = mol
        self.recontract_bas = cp.asarray(recontract_bas, dtype=np.int32)
        self.recontract_coef = cp.asarray(recontract_coef)

        # Sort basis according to angular momentum and contraction patterns so
        # as to group the basis functions to blocks in GPU kernel.
        l_ctrs = self._bas[:,[ANG_OF, NPRIM_OF]]
        # Ensure the more contracted Gaussians being accessed first
        l_ctrs_descend = l_ctrs.copy()
        l_ctrs_descend[:,1] = -l_ctrs[:,1]
        uniq_l_ctr, where, inv_idx, l_ctr_counts = np.unique(
            l_ctrs_descend, return_index=True, return_inverse=True, return_counts=True, axis=0)
        uniq_l_ctr[:,1] = -uniq_l_ctr[:,1]

        # Limit the number of AOs in each group
        if group_size is not None:
            uniq_l_ctr, l_ctr_counts = _split_l_ctr_groups(
                uniq_l_ctr, l_ctr_counts, group_size)

        if mol.verbose >= logger.DEBUG1:
            logger.debug1(mol, 'Number of shells for each [l, nprim] group')
            for l_ctr, n in zip(uniq_l_ctr, l_ctr_counts):
                logger.debug1(mol, '    %s : %s', l_ctr, n)

        sorted_idx = np.argsort(inv_idx.ravel(), kind='stable')
        self._bas = np.asarray(self._bas[sorted_idx], dtype=np.int32)

        # PTR_BAS_COORD is required by various CUDA kernels
        self._bas[:,PTR_BAS_COORD] = self._atm[self._bas[:,ATOM_OF],PTR_COORD]

        self.uniq_l_ctr = uniq_l_ctr
        self.l_ctr_counts = l_ctr_counts
        self.sorted_idx = sorted_idx
        inv_sorted = cp.empty(len(self._bas), dtype=np.int32)
        inv_sorted[sorted_idx] = cp.arange(len(self._bas))
        # recontraction_idx stores the indices of primitive shells (self._bas)
        # for each original contracted shell (self.mol._bas). The offset of each
        # contracted shell for recontraction_idx is provided by the
        # recontract_bas[:,PTR_BAS_IDX]
        self.recontraction_idx = inv_sorted[pbas_idx]
        self.p_ao_loc = self.ao_loc_nr(cart=True)
        return self

    from_cell = from_mol

    @property
    def c_ao_loc(self):
        l = self.recontract_bas[:,ANG_OF]
        if self.mol.cart:
            dims = (l+1)*(l+2)//2 * self.recontract_bas[:,NCTR_OF]
        else:
            dims = (l*2+1) * self.recontract_bas[:,NCTR_OF]
        return cp.append(cp.asarray([0], dtype=np.int32), dims.cumsum(dtype=np.int32))

    def CT_dot_mat(self, mat):
        '''ctr_coeff.T.dot(mat)
        '''
        mat = cp.asarray(mat, dtype=np.float64, order='C')
        mat_ndim = mat.ndim
        if mat_ndim == 1:
            return self.mat_dot_C(mat)
        elif mat_ndim == 2:
            mat = mat[None]

        if self.mol.cart:
            kern = libvhf_rys.bra_sorted2cart
        else:
            kern = libvhf_rys.bra_sorted2sph
        nao = self.mol.nao
        counts, nao_sorted, ncol = mat.shape
        assert nao_sorted == self.p_ao_loc[-1]
        if mat.dtype == np.complex128:
            ncol *= 2
        out = cp.zeros((counts, nao, ncol))
        if out.size > 0:
            c_ao_loc = cp.asarray(self.c_ao_loc, dtype=np.int32)
            p_ao_loc = cp.asarray(self.p_ao_loc, dtype=np.int32)
            err = kern(
                ctypes.cast(out.data.ptr, ctypes.c_void_p),
                ctypes.cast(mat.data.ptr, ctypes.c_void_p),
                ctypes.cast(self.recontract_coef.data.ptr, ctypes.c_void_p),
                ctypes.cast(self.recontract_bas.data.ptr, ctypes.c_void_p),
                ctypes.cast(self.recontraction_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(c_ao_loc.data.ptr, ctypes.c_void_p),
                ctypes.cast(p_ao_loc.data.ptr, ctypes.c_void_p),
                ctypes.c_int(len(self.recontract_bas)), ctypes.c_int(self.nbas),
                ctypes.c_int(ncol), ctypes.c_int(counts))
            assert err == 0

        if mat.dtype == np.complex128:
            out = out.view(np.complex128)
        if mat_ndim == 2:
            out = out[0]
        return out

    def C_dot_mat(self, mat):
        '''ctr_coeff.dot(mat)'''
        mat = cp.asarray(mat, dtype=np.float64, order='C')
        mat_ndim = mat.ndim
        if mat_ndim == 1:
            return self.mat_dot_CT(mat)
        elif mat_ndim == 2:
            mat = mat[None]

        if self.mol.cart:
            kern = libvhf_rys.bra_cart2sorted
        else:
            kern = libvhf_rys.bra_sph2sorted
        nao_sorted = self.p_ao_loc[-1]
        counts, nao, ncol = mat.shape
        assert nao == self.mol.nao
        if mat.dtype == np.complex128:
            ncol *= 2
        out = cp.zeros((counts, nao_sorted, ncol))
        if out.size > 0:
            c_ao_loc = cp.asarray(self.c_ao_loc, dtype=np.int32)
            p_ao_loc = cp.asarray(self.p_ao_loc, dtype=np.int32)
            err = kern(
                ctypes.cast(out.data.ptr, ctypes.c_void_p),
                ctypes.cast(mat.data.ptr, ctypes.c_void_p),
                ctypes.cast(self.recontract_coef.data.ptr, ctypes.c_void_p),
                ctypes.cast(self.recontract_bas.data.ptr, ctypes.c_void_p),
                ctypes.cast(self.recontraction_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(c_ao_loc.data.ptr, ctypes.c_void_p),
                ctypes.cast(p_ao_loc.data.ptr, ctypes.c_void_p),
                ctypes.c_int(len(self.recontract_bas)), ctypes.c_int(self.nbas),
                ctypes.c_int(ncol), ctypes.c_int(counts))
            assert err == 0

        if mat.dtype == np.complex128:
            out = out.view(np.complex128)
        if mat_ndim == 2:
            out = out[0]
        return out

    def mat_dot_C(self, mat):
        '''mat.dot(ctr_coeff)'''
        mat_ndim = mat.ndim
        mat_dtype = mat.dtype
        if mat_ndim == 1:
            mat = mat[None,None]
        elif mat_ndim == 2:
            mat = mat[None]

        if self.mol.cart:
            kern = libvhf_rys.ket_sorted2cart
        else:
            kern = libvhf_rys.ket_sorted2sph
        nao = self.mol.nao
        counts, nrow, nao_sorted = mat.shape
        assert nao_sorted == self.p_ao_loc[-1]
        if mat_dtype == np.complex128:
            mat = cp.asarray(mat.view(np.float64).transpose(0,1,3,2), order='C')
        else:
            mat = cp.asarray(mat, dtype=np.float64, order='C')
        out = cp.zeros((counts, nrow, nao))
        if out.size > 0:
            c_ao_loc = cp.asarray(self.c_ao_loc, dtype=np.int32)
            p_ao_loc = cp.asarray(self.p_ao_loc, dtype=np.int32)
            err = kern(
                ctypes.cast(out.data.ptr, ctypes.c_void_p),
                ctypes.cast(mat.data.ptr, ctypes.c_void_p),
                ctypes.cast(self.recontract_coef.data.ptr, ctypes.c_void_p),
                ctypes.cast(self.recontract_bas.data.ptr, ctypes.c_void_p),
                ctypes.cast(self.recontraction_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(c_ao_loc.data.ptr, ctypes.c_void_p),
                ctypes.cast(p_ao_loc.data.ptr, ctypes.c_void_p),
                ctypes.c_int(len(self.recontract_bas)), ctypes.c_int(self.nbas),
                ctypes.c_int(nrow*counts))
            assert err == 0

        if mat_dtype == np.complex128:
            mat = None
            out, tmp = cp.empty((counts, nrow, nao), dtype=np.complex128), out
            out.real = tmp[:,:nrow]
            out.imag = tmp[:,nrow:]
        if mat_ndim == 1:
            out = out[0,0]
        elif mat_ndim == 2:
            out = out[0]
        return out

    def mat_dot_CT(self, mat):
        '''mat.dot(ctr_coeff.T)'''
        mat_ndim = mat.ndim
        mat_dtype = mat.dtype
        if mat_ndim == 1:
            mat = mat[None,None]
        elif mat_ndim == 2:
            mat = mat[None]

        if self.mol.cart:
            kern = libvhf_rys.ket_cart2sorted
        else:
            kern = libvhf_rys.ket_sph2sorted
        nao_sorted = self.p_ao_loc[-1]
        counts, nrow, nao = mat.shape
        assert nao == self.mol.nao
        if mat_dtype == np.complex128:
            mat = cp.asarray(mat.view(np.float64).transpose(0,1,3,2), order='C')
        else:
            mat = cp.asarray(mat, dtype=np.float64, order='C')
        out = cp.zeros((counts, nrow, nao_sorted))
        if out.size > 0:
            c_ao_loc = cp.asarray(self.c_ao_loc, dtype=np.int32)
            p_ao_loc = cp.asarray(self.p_ao_loc, dtype=np.int32)
            err = kern(
                ctypes.cast(out.data.ptr, ctypes.c_void_p),
                ctypes.cast(mat.data.ptr, ctypes.c_void_p),
                ctypes.cast(self.recontract_coef.data.ptr, ctypes.c_void_p),
                ctypes.cast(self.recontract_bas.data.ptr, ctypes.c_void_p),
                ctypes.cast(self.recontraction_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(c_ao_loc.data.ptr, ctypes.c_void_p),
                ctypes.cast(p_ao_loc.data.ptr, ctypes.c_void_p),
                ctypes.c_int(len(self.recontract_bas)), ctypes.c_int(self.nbas),
                ctypes.c_int(nrow*counts))
            assert err == 0

        if mat_dtype == np.complex128:
            mat = None
            out, tmp = cp.empty((counts, nrow, nao_sorted), dtype=np.complex128), out
            out.real = tmp[:,:nrow]
            out.imag = tmp[:,nrow:]
        if mat_ndim == 1:
            out = out[0,0]
        elif mat_ndim == 2:
            out = out[0]
        return out

    def apply_CT_dot(self, mat, axis=0):
        '''C.T.dot(tensor)'''
        assert axis < mat.ndim
        axis = axis % mat.ndim
        dtype = mat.dtype
        assert dtype in (np.float64, np.complex128)
        if mat.ndim == axis+1: # last axis
            if mat.dtype == np.float64:
                return self.mat_dot_C(mat)
            out = cp.empty(mat.shape[:-1] + (self.mol.nao,), dtype=np.complex128)
            out.real = self.mat_dot_C(mat.real)
            out.imag = self.mat_dot_C(mat.imag)
            return out

        out_shape = list(mat.shape)
        out_shape[axis] = -1
        counts = np.prod(mat.shape[:axis], dtype=int)
        if dtype == np.complex128:
            mat = mat.view(np.float64)
        out = self.CT_dot_mat(mat.reshape(counts, mat.shape[axis], -1))
        if dtype == np.complex128:
            out = out.view(np.complex128)
        return out.reshape(out_shape)

    def apply_C_dot(self, mat, axis=0):
        '''C.dot(tensor)'''
        assert axis < mat.ndim
        axis = axis % mat.ndim
        dtype = mat.dtype
        assert dtype in (np.float64, np.complex128)
        if mat.ndim == axis+1: # last axis
            if dtype == np.float64:
                return self.mat_dot_CT(mat)
            out = cp.empty(mat.shape[:-1] + (self.nao,), dtype=np.complex128)
            out.real = self.mat_dot_CT(mat.real)
            out.imag = self.mat_dot_CT(mat.imag)
            return out

        out_shape = list(mat.shape)
        out_shape[axis] = -1
        counts = np.prod(mat.shape[:axis], dtype=int)
        if dtype == np.complex128:
            mat = mat.view(np.float64)
        out = self.C_dot_mat(mat.reshape(counts, mat.shape[axis], -1))
        if dtype == np.complex128:
            out = out.view(np.complex128)
        return out.reshape(out_shape)

    def apply_C_mat_CT(self, mat):
        assert 1 < mat.ndim <= 3
        dtype = mat.dtype
        if dtype == np.float64:
            mat = self.mat_dot_CT(mat)
            return self.C_dot_mat(mat)

        assert dtype == np.complex128
        out_shape = list(mat.shape)
        out_shape[-1] = self.nao
        out = cp.empty(out_shape, dtype=np.complex128)
        out.real = self.mat_dot_CT(mat.real)
        out.imag = self.mat_dot_CT(mat.imag)
        out_shape[-1] *= 2
        out = self.C_dot_mat(out.view(np.float64).reshape(out_shape))
        return out.view(np.complex128)

    def apply_CT_mat_C(self, mat):
        assert 1 < mat.ndim <= 3
        dtype = mat.dtype
        if dtype == np.float64:
            mat = self.CT_dot_mat(mat)
            return self.mat_dot_C(mat)

        assert dtype == np.complex128
        out_shape = list(mat.shape)
        out_shape[-1] = self.cell.nao
        out = cp.empty(out_shape, dtype=np.complex128)
        out.real = self.mat_dot_C(mat.real)
        out.imag = self.mat_dot_C(mat.imag)
        out_shape[-1] *= 2
        out = self.CT_dot_mat(out.view(np.float64).reshape(out_shape))
        return out.view(np.complex128)

    @property
    def ctr_coeff(self):
        mat = cp.eye(self.mol.nao)
        return self.C_dot_mat(mat)

    def rys_envs(self):
        raise NotImplementedError

class SortedMole(Mole, SortedGTO):
    def rys_envs(self):
        _env = _scale_sp_ctr_coeff(self)
        return RysIntEnvVars.new(
            self.natm, self.nbas, self._atm, self._bas, _env, self.p_ao_loc)

    def shell_overlap_mask(self, hermi=1, precision=1e-14):
        '''absmax(<i|j>) > precision for each shell pair'''
        from gpu4pyscf.pbc.gto.int1e import _shell_overlap_mask
        return _shell_overlap_mask(self, hermi, precision)

    def generate_shl_pairs(self, hermi=1, mask=None, gout_stride_lookup=None):
        if mask is None:
            mask = self.shell_overlap_mask(hermi)
        # The effective shell pair = ish*nbas+jsh
        bas_ij_cache = {}
        l_ctr_offsets = np.append(0, np.cumsum(self.l_ctr_counts))
        nbas = self.nbas
        groups = len(self.uniq_l_ctr)
        if hermi == 1:
            ij_tasks = [(i, j) for i in range(groups) for j in range(i+1)]
        else:
            ij_tasks = [(i, j) for i in range(groups) for j in range(groups)]
        for i, j in ij_tasks:
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
            t_ij = (cp.arange(ish0, ish1, dtype=np.int32)[:,None] * nbas +
                    cp.arange(jsh0, jsh1, dtype=np.int32))
            if hermi == 1 and i == j:
                sub_mask = mask[ish0:ish1,jsh0:jsh1].copy()
                sub_mask = cp.tril(sub_mask)
            else:
                sub_mask = mask[ish0:ish1,jsh0:jsh1]
            bas_ij_cache[i,j] = t_ij[sub_mask]
        return bas_ij_cache

    def aggregate_shl_pairs(self, bas_ij_cache=None, nsp_per_block=512):
        if bas_ij_cache is None:
            bas_ij_cache = self.generate_shl_pairs()
        bas_ij_idx = []
        shl_pair_offsets = []
        sp0 = sp1 = 0
        l = self.uniq_l_ctr[:,0]
        for (i, j), bas_ij in bas_ij_cache.items():
            bas_ij_idx.append(cp.asarray(bas_ij))
            sp0, sp1 = sp1, sp1 + len(bas_ij)
            if isinstance(nsp_per_block, (int, np.integer)):
                batch_size = nsp_per_block
            else:
                batch_size = nsp_per_block[l[i], l[j]]
            shl_pair_offsets.append(cp.arange(
                sp0, sp1, batch_size, dtype=np.int32))
        bas_ij_idx = cp.asarray(cp.hstack(bas_ij_idx), dtype=np.int32)
        shl_pair_offsets.append(np.int32(sp1))
        shl_pair_offsets = cp.asarray(cp.hstack(shl_pair_offsets), dtype=np.int32)
        return bas_ij_idx, shl_pair_offsets

class SortedCell(Cell, SortedGTO):
    def rys_envs(self):
        _env = _scale_sp_ctr_coeff(self)
        Ls = asarray(self.get_lattice_Ls(rcut=self.rcut))
        Ls = Ls[cp.linalg.norm(Ls-.1, axis=1).argsort()]
        nimgs = len(Ls)
        return PBCIntEnvVars.new(
            self.natm, self.nbas, 1, nimgs, self._atm, self._bas, _env, self.p_ao_loc, Ls)

    def shell_overlap_mask(self, hermi=1, precision=1e-14):
        '''absmax(<i|j>) > precision for each shell pair'''
        from gpu4pyscf.pbc.gto.int1e import _shell_overlap_mask
        Ls = asarray(self.cell.get_lattice_Ls())
        Ls = Ls[cp.linalg.norm(Ls-.1, axis=1).argsort()]
        return _shell_overlap_mask(self, hermi, precision, Ls)

    generate_shl_pairs = SortedMole.generate_shl_pairs
    aggregate_shl_pairs = SortedMole.aggregate_shl_pairs

class RysIntEnvVars(ctypes.Structure):
    _fields_ = [
        ('natm', ctypes.c_int),
        ('nbas', ctypes.c_int),
        ('atm', ctypes.c_void_p),
        ('bas', ctypes.c_void_p),
        ('env', ctypes.c_void_p),
        ('ao_loc', ctypes.c_void_p),
    ]

    @classmethod
    def new(cls, natm, nbas, atm, bas, env, ao_loc):
        atm = cp.asarray(atm)
        bas = cp.asarray(bas)
        env = cp.asarray(env)
        ao_loc = cp.asarray(ao_loc)
        obj = RysIntEnvVars(natm, nbas, atm.data.ptr, bas.data.ptr,
                            env.data.ptr, ao_loc.data.ptr)
        # Keep a reference to these arrays, prevent releasing them upon returning
        obj._env_ref_holder = (atm, bas, env, ao_loc)
        return obj

    def copy(self):
        atm, bas, env, ao_loc = self._env_ref_holder
        return RysIntEnvVars.new(self.natm, self.nbas, atm, bas, env, ao_loc)

    @property
    def device(self):
        return self._env_ref_holder[2].device

class PBCIntEnvVars(ctypes.Structure):
    _fields_ = [
        ('natm', ctypes.c_int),
        ('nbas', ctypes.c_int),
        ('atm', ctypes.c_void_p),
        ('bas', ctypes.c_void_p),
        ('env', ctypes.c_void_p),
        ('ao_loc', ctypes.c_void_p),
        ('bvk_ncells', ctypes.c_int),
        ('nimgs', ctypes.c_int),
        ('img_coords', ctypes.c_void_p),
    ]

    @classmethod
    def new(cls, natm, nbas, ncells, nimgs, atm, bas, env, ao_loc, Ls):
        atm = cp.asarray(atm)
        bas = cp.asarray(bas)
        env = cp.asarray(env)
        ao_loc = cp.asarray(ao_loc)
        Ls = cp.asarray(Ls)
        obj = PBCIntEnvVars(natm, nbas, atm.data.ptr, bas.data.ptr, env.data.ptr,
                            ao_loc.data.ptr, ncells, nimgs, Ls.data.ptr)
        # Keep a reference to these arrays, prevent releasing them upon returning
        obj._env_ref_holder = (atm, bas, env, ao_loc, Ls)
        return obj

    def copy(self):
        atm, bas, env, ao_loc, Ls = self._env_ref_holder
        return PBCIntEnvVars.new(
            self.natm, self.nbas, self.bvk_ncells, self.nimgs,
            atm, bas, env, ao_loc, Ls)

    @property
    def device(self):
        return self._env_ref_holder[2].device

def _scale_sp_ctr_coeff(mol):
    # Match normalization factors of s, p functions in libcint
    _env = mol._env.copy()
    ls = mol._bas[:,ANG_OF]
    ptr, idx = np.unique(mol._bas[:,PTR_COEFF], return_index=True)
    ptr = ptr[ls[idx] < 2]
    idx = idx[ls[idx] < 2]
    fac = ((ls[idx]*2+1) / (4*np.pi)) ** .5
    nprim = mol._bas[idx,NPRIM_OF]
    nctr = mol._bas[idx,NCTR_OF]
    for p, n, f in zip(ptr, nprim*nctr, fac):
        _env[p:p+n] *= f
    return _env

def _recontract_basis(mol, allow_replica=None, allow_split_seg_contraction=True):
    '''transform generally contracted basis to segment contracted basis.
    Note return_mol.cart is set to True.

    Kwargs:
        allow_replica:
            when angular momentum lower than (or equal to) this value, transform
            the generally contracted basis to replicated segment-contracted basis.
            By default, high angular momentum functions (d, f shells) are fully
            uncontracted.
        allow_split_seg_contraction:
            Allows the segmented contracted basis to be divided into small
            segments to improve load balance between deifferent shells.
    '''
    if allow_replica is True:
        allow_replica = 8
    elif allow_replica is False or allow_replica is None:
        allow_replica = -1

    PTR_PBAS_IDX = 4
    def split_shell_plain(shell):
        nctr = shell[NCTR_OF]
        shells = np.repeat(shell[np.newaxis], nctr, axis=0)
        shells[:,NCTR_OF] = 1
        shells[:,PTR_COEFF] += np.arange(nctr) * shell[NPRIM_OF]
        p2c_bas = shells.copy()
        p2c_bas[:,NPRIM_OF] = 1
        p2c_bas[:,PTR_COEFF] = np.arange(nctr)
        p2c_bas[:,PTR_PBAS_IDX] = np.arange(nctr)
        return shells, p2c_bas, np.ones(nctr), np.arange(nctr, dtype=np.int32)

    if not allow_split_seg_contraction:
        split_shell = split_shell_plain
    else:
        partial_decontraction_plan = {}
        nctr = mol._bas[:,NCTR_OF]
        nprim = mol._bas[:,NPRIM_OF]
        ls = mol._bas[:,ANG_OF]
        mask = (nctr == 1) | (ls <= allow_replica) #| (nprim >= 3*nctr)
        prim_pattern = mol._bas[:,[ANG_OF,NPRIM_OF]][mask]
        uniq_l_ctr, counts = np.unique(prim_pattern, return_counts=True, axis=0)
        if len(uniq_l_ctr) > 0:
            lmax = uniq_l_ctr[:,0].max()
            uniq_l = uniq_l_ctr[:,0]
            for l in range(lmax+1):
                l_counts = counts[uniq_l == l]
                if len(l_counts) <= 2 or l_counts.min() > 5:
                    continue
                l_nprim = uniq_l_ctr[uniq_l == l, 1]
                if l_nprim[0] != 1:
                    continue
                primary_base = l_nprim[1]
                secondary_base = l_nprim[0]
                for nprim, count in zip(l_nprim[2:], l_counts[2:]):
                    if count > 5:
                        primary_base = nprim
                        secondary_base = l_nprim[1]
                        continue
                    rep1, rem = divmod(nprim, primary_base)
                    rep2, rem = divmod(rem, secondary_base)
                    plan = [primary_base] * rep1 + [secondary_base] * rep2 + [1] * rem
                    partial_decontraction_plan[l, nprim] = np.array(plan)

        logger.debug1(mol, 'partial decontraction plan = %s', partial_decontraction_plan)

        def split_shell(shell):
            nprim = shell[NPRIM_OF]
            if nprim == 1:
                return split_shell_plain(shell)

            l = shell[ANG_OF]
            splits = partial_decontraction_plan.get((l, nprim))
            if splits is None or len(splits) == 1:
                return split_shell_plain(shell)

            nctr = shell[NCTR_OF]
            if nctr == 1:
                nsub_shl = len(splits)
                shells = np.repeat(shell[np.newaxis], nsub_shl, axis=0)
                offsets = np.cumsum(splits[:-1])
                shells[:,NPRIM_OF] = splits
                shells[1:,PTR_EXP] += offsets
                shells[1:,PTR_COEFF] += offsets
                p2c_bas = shell.copy()
                p2c_bas[NPRIM_OF] = nsub_shl
                p2c_bas[NCTR_OF] = 1
                p2c_bas[PTR_COEFF] = 0
                p2c_bas[PTR_PBAS_IDX] = 0
                return (shells, p2c_bas[np.newaxis],
                        np.ones(nsub_shl), # sum-over nsub_shl
                        np.arange(nsub_shl, dtype=np.int32))
            '''
            # split the [np x nc] coeffcients into
            # [[sub_np_1],[sub_np_2], ...] * nc shells
            # PTR_COEFF points to the address of each sub shell at
            # overall_offset + [0, x, 2x, ..., nprim, nprim+x, nprim+2x, ...]
            # Note, this mixed contraction scheme requires atomicAdd in
            # C_dot_mat and mat_dot_CT transfromation.
            if splits is None or len(splits) == 1:
                nprim_to_split = nprim
                splits = np.array([nprim])
            else:
                splits = splits[splits > nctr]
                nprim_to_split = splits.sum()

            # The contracted shell is split into nseg_shl
            # small-segment shells and (nprim-nprim_to_split) primitive shells
            nseg_shl = len(splits)
            nprim_remaining = nprim - nprim_to_split
            nsub_shl = nseg_shl + nprim_remaining
            pshell_idx = np.empty((nctr, nsub_shl), dtype=np.int32)
            c1 = np.empty((nctr, nsub_shl))
            if nprim_to_split > 0:
                shells = shell[np.newaxis]
                shells = np.repeat(shells, nseg_shl, axis=0)
                offsets = np.cumsum(splits[:-1])
                shells[:,NPRIM_OF] = splits
                shells[:,NCTR_OF] = 1
                shells[1:,PTR_EXP] += offsets
                shells[1:,PTR_COEFF] += offsets
                shells = np.repeat(shells[np.newaxis], nctr, axis=0)
                shells[:,:,PTR_COEFF] += np.arange(nctr)[:,None] * nprim
                shells = shells.reshape(-1, 8)
                c1[:,:nseg_shl] = 1.
                pshell_idx[:,:nseg_shl] = np.arange(len(shells)).reshape(nctr, nseg_shl)
            else:
                shells = np.zeros((0, len(shell)), dtype=np.int32)

            if nprim_remaining > 0:
                pcoeff = shell[PTR_COEFF]
                c = _env[pcoeff:pcoeff+nprim*nctr].reshape(nctr,nprim)
                shell_remaining = shell.copy()
                shell_remaining[NPRIM_OF] = nprim_remaining
                shell_remaining[PTR_EXP] += nprim_to_split
                shell_remaining[PTR_COEFF] += nprim_to_split
                shell_remaining, c2 = fully_uncontract(shell_remaining, c[:,nprim_to_split:])
                shells = np.vstack([shells, shell_remaining])
                c1[:,nseg_shl:] = c2
                pshell_idx[:,nseg_shl:] = np.arange(nctr*nseg_shl, len(shells))

            p2c_bas = np.repeat(shell[np.newaxis], nctr, axis=0)
            p2c_bas[:,NPRIM_OF] = nsub_shl
            p2c_bas[:,NCTR_OF] = 1
            p2c_bas[:,PTR_COEFF] = np.arange(nctr) * nsub_shl
            p2c_bas[:,PTR_PBAS_IDX] = np.arange(nctr) * nsub_shl
            return shells, p2c_bas, c1.ravel(), pshell_idx.ravel()
            '''
            return split_shell_plain(shell)

    def fully_uncontract(shell, c):
        l = shell[ANG_OF]
        nprim = shell[NPRIM_OF]
        pexp = shell[PTR_EXP]
        pcoeff = shell[PTR_COEFF]
        exps = _env[pexp:pexp+nprim]
        norm = gto.gto_norm(l, exps)
        # remove normalization from contraction coefficients
        c = c / norm
        # Overwrite the existing contraction coefficients. must make
        # a copy of _env to avoid overwritting mol._env
        _env[pcoeff:pcoeff+nprim] = norm
        shells = np.repeat(shell[np.newaxis], nprim, axis=0)
        shells[:,NPRIM_OF] = 1
        shells[:,NCTR_OF] = 1
        shells[:,PTR_EXP] += np.arange(nprim)
        shells[:,PTR_COEFF] += np.arange(nprim)
        return shells, c

    bas_templates = {}
    _env = mol._env.copy()
    _bas = []
    ctr_coef = []
    recontract_bas = []
    pbas_idx_recontraction = []
    pbas_idx_size = 0
    pbas = 0
    ptr_coef = 0
    aoslices = mol.aoslice_by_atom()
    for ia, (ib0, ib1) in enumerate(aoslices[:,:2]):
        if ib0 == ib1:
            continue
        key = tuple(mol._bas[ib0:ib1,PTR_COEFF])
        if key not in bas_templates:
            bas_of_ia = []
            recontract = []
            pbas_idx = []
            pidx_offset = 0
            pbas_local = 0
            for shell in mol._bas[ib0:ib1]:
                l = shell[ANG_OF]
                nprim = shell[NPRIM_OF]
                nctr = shell[NCTR_OF]
                if nctr == 1 or l <= allow_replica or nprim >= 3*nctr:
                    shells, p2c_bas, c, idx = split_shell(shell)
                    bas_of_ia.append(shells)
                    p2c_bas[:,PTR_COEFF] += ptr_coef
                    p2c_bas[:,PTR_PBAS_IDX] += pidx_offset
                    recontract.append(p2c_bas)
                    pbas_idx.append(idx + pbas_local)
                    ctr_coef.append(c)
                    pbas_local += len(shells)
                    pidx_offset += len(idx)
                    ptr_coef += c.size

                else: # To avoid recomputation, decontract to primitive functions
                    pcoeff = shell[PTR_COEFF]
                    c = _env[pcoeff:pcoeff+nprim*nctr].reshape(nctr,nprim)
                    shell, c = fully_uncontract(shell, c)
                    bas_of_ia.append(shell)
                    recontract.append(
                        np.array([ia, l, nprim, nctr, pidx_offset, 0, ptr_coef, 0], dtype=np.int32))
                    pbas_idx.append(np.arange(nprim, dtype=np.int32) + pbas_local)
                    ctr_coef.append(c.ravel())
                    pbas_local += nprim
                    pidx_offset += nprim
                    ptr_coef += c.size

            bas_templates[key] = (np.vstack(bas_of_ia), np.vstack(recontract), np.hstack(pbas_idx))

        bas_of_ia, recontract, pbas_idx = bas_templates[key]
        bas_of_ia = bas_of_ia.copy()
        bas_of_ia[:,ATOM_OF] = ia
        _bas.append(bas_of_ia)

        recontract = recontract.copy()
        recontract[:,ATOM_OF] = ia
        recontract[:,PTR_PBAS_IDX] += pbas_idx_size
        recontract_bas.append(recontract)
        pbas_idx_recontraction.append(pbas_idx + pbas)
        pbas_idx_size += len(pbas_idx)
        pbas += len(bas_of_ia)

    pmol = mol.copy(deep=False)
    pmol.cart = True
    if _bas:
        pmol._bas = np.asarray(np.vstack(_bas), dtype=np.int32)
    pmol._env = _env

    recontract_bas = np.vstack(recontract_bas)
    recontract_coef = np.hstack(ctr_coef)
    pbas_idx_recontraction = np.hstack(pbas_idx_recontraction)
    return pmol, recontract_bas, recontract_coef, pbas_idx_recontraction
