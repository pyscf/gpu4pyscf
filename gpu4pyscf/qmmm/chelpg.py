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
from gpu4pyscf.scf.int4c2e import BasisProdCache, libgint, libgvhf
from gpu4pyscf.lib.cupy_helper import load_library, block_c2s_diag
from gpu4pyscf.lib import logger

from pyscf.data import radii
modified_Bondi = radii.VDW.copy()
modified_Bondi[1] = 1.1/radii.BOHR      # modified version

# TODO: replace int3c2e.get_j_int3c2e_pass1 with int1e_grids
def _build_VHFOpt(intopt, cutoff=1e-14, group_size=None,
            group_size_aux=None, diag_block_with_triu=False, aosym=False):
    '''
    Implement the similar functionality as VHFOpt.build, 
    but without transformation for auxiliary basis.
    '''
    sorted_mol, sorted_idx, uniq_l_ctr, l_ctr_counts = int3c2e.sort_mol(
        intopt.mol)
    if group_size is not None:
        uniq_l_ctr, l_ctr_counts = int3c2e._split_l_ctr_groups(
            uniq_l_ctr, l_ctr_counts, group_size)

    # sort fake mol
    fake_mol = int3c2e.make_fake_mol()
    _, _, fake_uniq_l_ctr, fake_l_ctr_counts = int3c2e.sort_mol(fake_mol)

    # sort auxiliary mol
    sorted_auxmol, _, aux_uniq_l_ctr, aux_l_ctr_counts = int3c2e.sort_mol(
        intopt.auxmol)
    if group_size_aux is not None:
        aux_uniq_l_ctr, aux_l_ctr_counts = int3c2e._split_l_ctr_groups(
            aux_uniq_l_ctr, aux_l_ctr_counts, group_size_aux)

    tmp_mol = gto.mole.conc_mol(fake_mol, sorted_auxmol)
    tot_mol = gto.mole.conc_mol(sorted_mol, tmp_mol)

    # Initialize vhfopt after reordering mol._bas
    _vhf.VHFOpt.__init__(intopt, sorted_mol, intopt._intor, intopt._prescreen,
                            intopt._qcondname, intopt._dmcondname)
    intopt.direct_scf_tol = cutoff

    # TODO: is it more accurate to filter with overlap_cond (or exp_cond)?
    q_cond = intopt.get_q_cond()
    l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
    log_qs, pair2bra, pair2ket = int3c2e.get_pairing(
        l_ctr_offsets, l_ctr_offsets, q_cond,
        diag_block_with_triu=diag_block_with_triu, aosym=aosym)
    intopt.log_qs = log_qs.copy()

    # contraction coefficient for ao basis
    cart_ao_loc = sorted_mol.ao_loc_nr(cart=True)
    sph_ao_loc = sorted_mol.ao_loc_nr(cart=False)
    intopt.cart_ao_loc = [cart_ao_loc[cp] for cp in l_ctr_offsets]
    intopt.sph_ao_loc = [sph_ao_loc[cp] for cp in l_ctr_offsets]
    intopt.angular = [l[0] for l in uniq_l_ctr]

    cart_ao_loc = intopt.mol.ao_loc_nr(cart=True)
    sph_ao_loc = intopt.mol.ao_loc_nr(cart=False)
    nao = sph_ao_loc[-1]
    ao_idx = np.array_split(np.arange(nao), sph_ao_loc[1:-1])
    intopt.sph_ao_idx = np.hstack([ao_idx[i] for i in sorted_idx])

    # cartesian ao index
    nao = cart_ao_loc[-1]
    ao_idx = np.array_split(np.arange(nao), cart_ao_loc[1:-1])
    intopt.cart_ao_idx = np.hstack([ao_idx[i] for i in sorted_idx])
    ncart = cart_ao_loc[-1]
    intopt.cart2sph = block_c2s_diag(intopt.angular, l_ctr_counts)

    # pairing auxiliary basis with fake basis set
    fake_l_ctr_offsets = np.append(0, np.cumsum(fake_l_ctr_counts))
    fake_l_ctr_offsets += l_ctr_offsets[-1]

    aux_l_ctr_offsets = np.append(0, np.cumsum(aux_l_ctr_counts))

    # contraction coefficient for auxiliary basis
    cart_aux_loc = sorted_auxmol.ao_loc_nr(cart=True)
    sph_aux_loc = sorted_auxmol.ao_loc_nr(cart=False)
    intopt.cart_aux_loc = [cart_aux_loc[cp] for cp in aux_l_ctr_offsets]
    intopt.sph_aux_loc = [sph_aux_loc[cp] for cp in aux_l_ctr_offsets]
    intopt.aux_angular = [l[0] for l in aux_uniq_l_ctr]

    cart_aux_loc = intopt.auxmol.ao_loc_nr(cart=True)
    sph_aux_loc = intopt.auxmol.ao_loc_nr(cart=False)
    ncart = cart_aux_loc[-1]
    # inv_idx = np.argsort(intopt.sph_aux_idx, kind='stable').astype(np.int32)
    aux_l_ctr_offsets += fake_l_ctr_offsets[-1]

    # hardcoded for grids
    aux_pair2bra = [np.arange(aux_l_ctr_offsets[0], aux_l_ctr_offsets[-1])]
    aux_pair2ket = [np.ones(ncart) * fake_l_ctr_offsets[0]]
    aux_log_qs = [np.ones(ncart)]

    intopt.aux_log_qs = aux_log_qs.copy()
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

    intopt.bpcache = ctypes.POINTER(BasisProdCache)()
    scale_shellpair_diag = 1.
    libgint.GINTinit_basis_prod(
        ctypes.byref(intopt.bpcache), ctypes.c_double(scale_shellpair_diag),
        ao_loc.ctypes.data_as(ctypes.c_void_p),
        bas_pair2shls.ctypes.data_as(ctypes.c_void_p),
        bas_pairs_locs.ctypes.data_as(
            ctypes.c_void_p), ctypes.c_int(ncptype),
        tot_mol._atm.ctypes.data_as(
            ctypes.c_void_p), ctypes.c_int(tot_mol.natm),
        tot_mol._bas.ctypes.data_as(
            ctypes.c_void_p), ctypes.c_int(tot_mol.nbas),
        tot_mol._env.ctypes.data_as(ctypes.c_void_p))
    intopt.bas_pairs_locs = bas_pairs_locs
    ncptype = len(intopt.log_qs)
    if aosym:
        intopt.cp_idx, intopt.cp_jdx = np.tril_indices(ncptype)
    else:
        nl = int(round(np.sqrt(ncptype)))
        intopt.cp_idx, intopt.cp_jdx = np.unravel_index(
            np.arange(ncptype), (nl, nl))

    intopt._sorted_mol = sorted_mol
    intopt._sorted_auxmol = sorted_auxmol
    if intopt.mol.cart:
        intopt._ao_idx = intopt.cart_ao_idx
    else:
        intopt._ao_idx = intopt.sph_ao_idx

def eval_chelpg_layer_gpu(mf, deltaR=0.3, Rhead=2.8, ifqchem=True, Rvdw=modified_Bondi, verbose=None):
    """Cal chelpg charge

    Args:
        mf: mean field object in pyscf
        deltaR (float, optional): the intervel in the cube. Defaults to 0.3.
        Rhead (float, optional): the head length. Defaults to 3.0.
        ifqchem (bool, optional): whether use the modification in qchem. Defaults to True.
        Rvdw (dict, optional): vdw radius. Defaults to modified Bondi radii.
    Returns:
        numpy.array: charges
    """
    log = logger.new_logger(mf, verbose)
    t1 = log.init_timer()

    atomcoords = mf.mol.atom_coords(unit='B')
    dm = cupy.array(mf.make_rdm1())

    Roff = Rhead/radii.BOHR
    Deltar = 0.1

    # smoothing function
    def tau_f(R, Rcut, Roff):
        return (R - Rcut)**2 * (3*Roff - Rcut - 2*R) / (Roff - Rcut)**3

    Rshort = np.array([Rvdw[iatom] for iatom in mf.mol._atm[:, 0]])
    idxxmin = np.argmin(atomcoords[:, 0] - Rshort)
    idxxmax = np.argmax(atomcoords[:, 0] + Rshort)
    idxymin = np.argmin(atomcoords[:, 1] - Rshort)
    idxymax = np.argmax(atomcoords[:, 1] + Rshort)
    idxzmin = np.argmin(atomcoords[:, 2] - Rshort)
    idxzmax = np.argmax(atomcoords[:, 2] + Rshort)
    atomtypes = np.array(mf.mol._atm[:, 0])
    # Generate the grids in the cube
    xmin = atomcoords[:, 0].min() - Rhead/radii.BOHR - Rvdw[atomtypes[idxxmin]]
    xmax = atomcoords[:, 0].max() + Rhead/radii.BOHR + Rvdw[atomtypes[idxxmax]]
    ymin = atomcoords[:, 1].min() - Rhead/radii.BOHR - Rvdw[atomtypes[idxymin]]
    ymax = atomcoords[:, 1].max() + Rhead/radii.BOHR + Rvdw[atomtypes[idxymax]]
    zmin = atomcoords[:, 2].min() - Rhead/radii.BOHR - Rvdw[atomtypes[idxzmin]]
    zmax = atomcoords[:, 2].max() + Rhead/radii.BOHR + Rvdw[atomtypes[idxzmax]]
    x = np.arange(xmin, xmax, deltaR/radii.BOHR)
    y = np.arange(ymin, ymax, deltaR/radii.BOHR)
    z = np.arange(zmin, zmax, deltaR/radii.BOHR)
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
    intopt = int3c2e.VHFOpt(mf.mol, fmol, 'int2e')
    for ibatch in range(0, ngrids, nbatch):
        max_grid = min(ibatch+nbatch, ngrids)
        num_grids = max_grid - ibatch
        ptr = intopt.auxmol._atm[:num_grids, gto.PTR_COORD]
        intopt.auxmol._env[np.vstack(
            (ptr, ptr+1, ptr+2)).T] = gridcoords[ibatch:max_grid]
        _build_VHFOpt(intopt, 1e-14, diag_block_with_triu=False, aosym=True)
        potential_real[ibatch:max_grid] -= 2.0 * \
            int3c2e.get_j_int3c2e_pass1(intopt, dm, sort_j=False)[:num_grids]

    w = cupy.array(w)
    r_pX_potential_omega = r_pX_potential*w
    GXA = r_pX_potential_omega@r_pX_potential.T
    eX = r_pX_potential_omega@potential_real
    GXA_inv = cupy.linalg.inv(GXA)
    g = GXA_inv@eX
    alpha = (g.sum() - mf.mol.charge)/(GXA_inv.sum())
    q = g - alpha*GXA_inv@cupy.ones((mf.mol.natm))
    t1 = log.timer_debug1('compute ChElPG charge', *t1)
    return q

