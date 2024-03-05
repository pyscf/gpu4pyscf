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

import pyscf
import time
import cupy
import numpy as np
import scipy
import ctypes
from pyscf import lib, gto
from pyscf.scf import _vhf
from gpu4pyscf.df import int3c2e
from gpu4pyscf.scf.hf import BasisProdCache
from gpu4pyscf.lib.cupy_helper import load_library, block_c2s_diag
libgint = load_library('libgint')
libgvhf = load_library('libgvhf')
lib.num_threads(8)


def get_j_int3c2e_pass1(intopt, dm0):
    '''
    get rhoj pass1 for int3c2e
    '''
    n_dm = 1

    naux = intopt.naux
    rhoj = cupy.zeros([naux])
    coeff = intopt.coeff
    if dm0.ndim == 3:
        dm0 = dm0[0] + dm0[1]
    dm_cart = cupy.einsum('pi,ij,qj->pq', coeff, dm0, coeff)

    num_cp_ij = [len(log_qs) for log_qs in intopt.log_qs]
    num_cp_kl = [len(log_qs) for log_qs in intopt.aux_log_qs]

    bins_locs_ij = np.append(0, np.cumsum(num_cp_ij)).astype(np.int32)
    bins_locs_kl = np.append(0, np.cumsum(num_cp_kl)).astype(np.int32)

    ncp_ij = len(intopt.log_qs)
    ncp_kl = len(intopt.aux_log_qs)
    norb = dm_cart.shape[0]
    err = libgvhf.GINTbuild_j_int3c2e_pass1(
        intopt.bpcache,
        ctypes.cast(dm_cart.data.ptr, ctypes.c_void_p),
        ctypes.cast(rhoj.data.ptr, ctypes.c_void_p),
        ctypes.c_int(norb),
        ctypes.c_int(naux),
        ctypes.c_int(n_dm),
        bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
        bins_locs_kl.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(ncp_ij),
        ctypes.c_int(ncp_kl))
    if err != 0:
        raise RuntimeError('CUDA error in get_j_pass1')
    return rhoj


class VHFOpt(_vhf.VHFOpt):
    def __init__(self, mol, auxmol, intor, prescreen='CVHFnoscreen',
                 qcondname='CVHFsetnr_direct_scf', dmcondname=None):
        # use local basis_seg_contraction for efficiency
        self.mol = int3c2e.basis_seg_contraction(mol, allow_replica=True)
        self.auxmol = int3c2e.basis_seg_contraction(auxmol, allow_replica=True)
        '''
        # Note mol._bas will be sorted in .build() method. VHFOpt should be
        # initialized after mol._bas updated.
        '''
        self.nao = self.mol.nao
        self.naux = self.auxmol.nao

        self._intor = intor
        self._prescreen = prescreen
        self._qcondname = qcondname
        self._dmcondname = dmcondname

        self.bpcache = None

        self.cart_ao_idx = None
        self.sph_ao_idx = None
        self.cart_aux_idx = None
        self.sph_aux_idx = None

        self.cart_ao_loc = []
        self.cart_aux_loc = []
        self.sph_ao_loc = []
        self.sph_aux_loc = []

        self.cart2sph = None
        self.aux_cart2sph = None

        self.angular = None
        self.aux_angular = None

        self.cp_idx = None
        self.cp_jdx = None

        self.log_qs = None
        self.aux_log_qs = None

    def clear(self):
        _vhf.VHFOpt.__del__(self)
        libgvhf.GINTdel_basis_prod(ctypes.byref(self.bpcache))
        return self

    def __del__(self):
        try:
            self.clear()
        except AttributeError:
            pass

    def build(self, cutoff=1e-14, group_size=None,
              group_size_aux=None, diag_block_with_triu=False, aosym=False):
        '''
        int3c2e is based on int2e with (ao,ao|aux,1)
        a tot_mol is created with concatenating [mol, fake_mol, aux_mol]
        we will pair (ao,ao) and (aux,1) separately.
        '''
        sorted_mol, sorted_idx, uniq_l_ctr, l_ctr_counts = int3c2e.sort_mol(
            self.mol)
        if group_size is not None:
            uniq_l_ctr, l_ctr_counts = int3c2e._split_l_ctr_groups(
                uniq_l_ctr, l_ctr_counts, group_size)

        # sort fake mol
        fake_mol = int3c2e.make_fake_mol()
        _, _, fake_uniq_l_ctr, fake_l_ctr_counts = int3c2e.sort_mol(fake_mol)

        # sort auxiliary mol
        sorted_auxmol, sorted_aux_idx, aux_uniq_l_ctr, aux_l_ctr_counts = int3c2e.sort_mol(
            self.auxmol)
        if group_size_aux is not None:
            aux_uniq_l_ctr, aux_l_ctr_counts = int3c2e._split_l_ctr_groups(
                aux_uniq_l_ctr, aux_l_ctr_counts, group_size_aux)

        tmp_mol = gto.mole.conc_mol(fake_mol, sorted_auxmol)
        tot_mol = gto.mole.conc_mol(sorted_mol, tmp_mol)

        # Initialize vhfopt after reordering mol._bas
        _vhf.VHFOpt.__init__(self, sorted_mol, self._intor, self._prescreen,
                             self._qcondname, self._dmcondname)
        self.direct_scf_tol = cutoff

        # TODO: is it more accurate to filter with overlap_cond (or exp_cond)?
        q_cond = self.get_q_cond()
        l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
        log_qs, pair2bra, pair2ket = int3c2e.get_pairing(
            l_ctr_offsets, l_ctr_offsets, q_cond,
            diag_block_with_triu=diag_block_with_triu, aosym=aosym)
        self.log_qs = log_qs.copy()

        # contraction coefficient for ao basis
        cart_ao_loc = sorted_mol.ao_loc_nr(cart=True)
        sph_ao_loc = sorted_mol.ao_loc_nr(cart=False)
        self.cart_ao_loc = [cart_ao_loc[cp] for cp in l_ctr_offsets]
        self.sph_ao_loc = [sph_ao_loc[cp] for cp in l_ctr_offsets]
        self.angular = [l[0] for l in uniq_l_ctr]

        cart_ao_loc = self.mol.ao_loc_nr(cart=True)
        sph_ao_loc = self.mol.ao_loc_nr(cart=False)
        nao = sph_ao_loc[-1]
        ao_idx = np.array_split(np.arange(nao), sph_ao_loc[1:-1])
        self.sph_ao_idx = np.hstack([ao_idx[i] for i in sorted_idx])

        # cartesian ao index
        nao = cart_ao_loc[-1]
        ao_idx = np.array_split(np.arange(nao), cart_ao_loc[1:-1])
        self.cart_ao_idx = np.hstack([ao_idx[i] for i in sorted_idx])
        ncart = cart_ao_loc[-1]
        nsph = sph_ao_loc[-1]
        self.cart2sph = block_c2s_diag(ncart, nsph, self.angular, l_ctr_counts)
        inv_idx = np.argsort(self.sph_ao_idx, kind='stable').astype(np.int32)
        self.coeff = self.cart2sph[:, inv_idx]

        # pairing auxiliary basis with fake basis set
        fake_l_ctr_offsets = np.append(0, np.cumsum(fake_l_ctr_counts))
        fake_l_ctr_offsets += l_ctr_offsets[-1]

        aux_l_ctr_offsets = np.append(0, np.cumsum(aux_l_ctr_counts))

        # contraction coefficient for auxiliary basis
        cart_aux_loc = sorted_auxmol.ao_loc_nr(cart=True)
        sph_aux_loc = sorted_auxmol.ao_loc_nr(cart=False)
        self.cart_aux_loc = [cart_aux_loc[cp] for cp in aux_l_ctr_offsets]
        self.sph_aux_loc = [sph_aux_loc[cp] for cp in aux_l_ctr_offsets]
        self.aux_angular = [l[0] for l in aux_uniq_l_ctr]

        cart_aux_loc = self.auxmol.ao_loc_nr(cart=True)
        sph_aux_loc = self.auxmol.ao_loc_nr(cart=False)
        ncart = cart_aux_loc[-1]
        nsph = sph_aux_loc[-1]
        # inv_idx = np.argsort(self.sph_aux_idx, kind='stable').astype(np.int32)
        aux_l_ctr_offsets += fake_l_ctr_offsets[-1]

        # hardcoded for grids
        aux_pair2bra = [np.arange(aux_l_ctr_offsets[0], aux_l_ctr_offsets[-1])]
        aux_pair2ket = [np.ones(ncart) * fake_l_ctr_offsets[0]]
        aux_log_qs = [np.ones(ncart)]

        self.aux_log_qs = aux_log_qs.copy()
        pair2bra += aux_pair2bra
        pair2ket += aux_pair2ket

        uniq_l_ctr = np.concatenate(
            [uniq_l_ctr, fake_uniq_l_ctr, aux_uniq_l_ctr])
        l_ctr_offsets = np.concatenate([
            l_ctr_offsets,
            fake_l_ctr_offsets[1:],
            aux_l_ctr_offsets[1:]])

        bas_pair2shls = np.hstack(
            pair2bra + pair2ket).astype(np.int32).reshape(2, -1)
        bas_pairs_locs = np.append(0, np.cumsum(
            [x.size for x in pair2bra])).astype(np.int32)
        log_qs = log_qs + aux_log_qs
        ao_loc = tot_mol.ao_loc_nr(cart=True)
        ncptype = len(log_qs)

        self.bpcache = ctypes.POINTER(BasisProdCache)()
        scale_shellpair_diag = 1.
        libgint.GINTinit_basis_prod(
            ctypes.byref(self.bpcache), ctypes.c_double(scale_shellpair_diag),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            bas_pair2shls.ctypes.data_as(ctypes.c_void_p),
            bas_pairs_locs.ctypes.data_as(
                ctypes.c_void_p), ctypes.c_int(ncptype),
            tot_mol._atm.ctypes.data_as(
                ctypes.c_void_p), ctypes.c_int(tot_mol.natm),
            tot_mol._bas.ctypes.data_as(
                ctypes.c_void_p), ctypes.c_int(tot_mol.nbas),
            tot_mol._env.ctypes.data_as(ctypes.c_void_p))
        self.bas_pairs_locs = bas_pairs_locs
        ncptype = len(self.log_qs)
        if aosym:
            self.cp_idx, self.cp_jdx = np.tril_indices(ncptype)
        else:
            nl = int(round(np.sqrt(ncptype)))
            self.cp_idx, self.cp_jdx = np.unravel_index(
                np.arange(ncptype), (nl, nl))


def eval_chelpg_layer_gpu(mf, deltaR=0.3, Rhead=2.8, ifqchem=True):
    """Cal chelpg charge

    Args:
        mf: mean field object in pyscf
        deltaR (float, optional): the intervel in the cube. Defaults to 0.3.
        Rhead (float, optional): the head length. Defaults to 3.0.
        ifqchem (bool, optional): whether use the modification in qchem. Defaults to True.

    Returns:
        numpy.array: charges
    """
    t0 = time.process_time()
    t0w = time.time()
    BOHR = 0.52917721092  # Angstroms
    atomcoords = mf.mol.atom_coords(unit='B')
    dm = cupy.array(mf.make_rdm1())
    RVDW_bondi = {1: 1.1/BOHR, 2: 1.40/BOHR,
                  3: 1.82/BOHR, 6: 1.70/BOHR, 7: 1.55/BOHR, 8: 1.52/BOHR, 9: 1.47/BOHR, 10: 1.54/BOHR,
                  11: 2.27/BOHR, 12: 1.73/BOHR, 14: 2.10/BOHR, 15: 1.80/BOHR, 16: 1.80/BOHR, 17: 1.75/BOHR, 18: 1.88/BOHR,
                  19: 2.75/BOHR, 35: 1.85/BOHR}

    Roff = Rhead/BOHR
    Deltar = 0.1

    # smoothing function
    def tau_f(R, Rcut, Roff):
        return (R - Rcut)**2 * (3*Roff - Rcut - 2*R) / (Roff - Rcut)**3

    Rshort = np.array([RVDW_bondi[iatom] for iatom in mf.mol._atm[:, 0]])
    idxxmin = np.argmin(atomcoords[:, 0] - Rshort)
    idxxmax = np.argmax(atomcoords[:, 0] + Rshort)
    idxymin = np.argmin(atomcoords[:, 1] - Rshort)
    idxymax = np.argmax(atomcoords[:, 1] + Rshort)
    idxzmin = np.argmin(atomcoords[:, 2] - Rshort)
    idxzmax = np.argmax(atomcoords[:, 2] + Rshort)
    atomtypes = np.array(mf.mol._atm[:, 0])
    # Generate the grids in the cube
    xmin = atomcoords[:, 0].min() - Rhead/BOHR - RVDW_bondi[atomtypes[idxxmin]]
    xmax = atomcoords[:, 0].max() + Rhead/BOHR + RVDW_bondi[atomtypes[idxxmax]]
    ymin = atomcoords[:, 1].min() - Rhead/BOHR - RVDW_bondi[atomtypes[idxymin]]
    ymax = atomcoords[:, 1].max() + Rhead/BOHR + RVDW_bondi[atomtypes[idxymax]]
    zmin = atomcoords[:, 2].min() - Rhead/BOHR - RVDW_bondi[atomtypes[idxzmin]]
    zmax = atomcoords[:, 2].max() + Rhead/BOHR + RVDW_bondi[atomtypes[idxzmax]]
    x = np.arange(xmin, xmax, deltaR/BOHR)
    y = np.arange(ymin, ymax, deltaR/BOHR)
    z = np.arange(zmin, zmax, deltaR/BOHR)
    gridcoords = np.meshgrid(x, y, z)
    gridcoords = np.vstack(list(map(np.ravel, gridcoords))).T

    # [natom, ngrids] distance between an atom and a grid
    r_pX = scipy.spatial.distance.cdist(atomcoords, gridcoords)
    # delete the grids in the vdw surface and out the Rhead surface.
    # the minimum distance to any atom
    Rkmin = (r_pX - np.expand_dims(Rshort, axis=1)).min(axis=0)
    Ron = Rshort + Deltar
    Rlong = Roff - Deltar
    AJk = np.ones(r_pX.shape)  # the short-range weight
    idx = r_pX < np.expand_dims(Rshort, axis=1)
    AJk[idx] = 0
    if ifqchem:
        idx2 = (r_pX < np.expand_dims(Ron, axis=1)) * \
            (r_pX >= np.expand_dims(Rshort, axis=1))
        AJk[idx2] = tau_f(r_pX, np.expand_dims(Rshort, axis=1),
                          np.expand_dims(Ron, axis=1))[idx2]
        wLR = 1 - tau_f(Rkmin, Rlong, Roff)  # the long-range weight
        idx1 = Rkmin < Rlong
        idx2 = Rkmin > Roff
        wLR[idx1] = 1
        wLR[idx2] = 0
    else:
        wLR = np.ones(r_pX.shape[-1])  # the long-range weight
        idx = Rkmin > Roff
        wLR[idx] = 0
    w = wLR*np.prod(AJk, axis=0)  # weight for a specific poing
    idx = w <= 1.0E-14
    w = np.delete(w, idx)
    r_pX = np.delete(r_pX, idx, axis=1)
    gridcoords = np.delete(gridcoords, idx, axis=0)

    ngrids = gridcoords.shape[0]
    r_pX = cupy.array(r_pX)
    r_pX_potential = 1/r_pX
    potential_real = cupy.dot(cupy.array(
        mf.mol.atom_charges()), r_pX_potential)
    nbatch = 256*256

    # assert nbatch < ngrids
    fmol = pyscf.gto.fakemol_for_charges(gridcoords[:nbatch])
    intopt = VHFOpt(mf.mol, fmol, 'int2e')
    for ibatch in range(0, ngrids, nbatch):
        max_grid = min(ibatch+nbatch, ngrids)
        num_grids = max_grid - ibatch
        ptr = intopt.auxmol._atm[:num_grids, gto.PTR_COORD]
        intopt.auxmol._env[np.vstack(
            (ptr, ptr+1, ptr+2)).T] = gridcoords[ibatch:max_grid]
        intopt.build(1e-14, diag_block_with_triu=False, aosym=True)
        potential_real[ibatch:max_grid] -= 2.0 * \
            get_j_int3c2e_pass1(intopt, dm)[:num_grids]

    w = cupy.array(w)
    r_pX_potential_omega = r_pX_potential*w
    GXA = r_pX_potential_omega@r_pX_potential.T
    eX = r_pX_potential_omega@potential_real
    GXA_inv = cupy.linalg.inv(GXA)
    g = GXA_inv@eX
    alpha = (g.sum() - mf.mol.charge)/(GXA_inv.sum())
    q = g - alpha*GXA_inv@cupy.ones((mf.mol.natm))
    t6 = time.process_time()
    t6w = time.time()
    print("Total cpu time: ", t6 - t0)
    print("Total wall time: ", t6w - t0w)
    return q

