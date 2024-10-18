# Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
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
#
# Author: Chenghan Li <lch004218@gmail.com>

import cupy as cp
import numpy as np

from pyscf import gto, lib
from pyscf.gto.mole import is_au
from pyscf.lib import param, logger

from gpu4pyscf.lib import cupy_helper

from cupyx.scipy.special import erfc, erf

contract = cupy_helper.contract

def get_multipole_tensors_pp(Rij, multipole_orders, rij=None):
    ''' Compute the interaction tensors between point multipoles.

    Args:
        Rij : 3D array, shape (Ni,Nj,3)
            Distance vectors pointing from particle j to i
        multipole_orders : list of int
            orders of multipole interactions to compute
            0: Tij, 1: Tija, 2: Tijab, ...

    Kwargs:
        rij : 2D array, shape (Ni,Nj)
            Distance between particles i and j

    Returns:
        A list of multipole tensors requested in ascending order
    '''
    if max(multipole_orders) > 3:
        raise NotImplementedError('Tijabcd or higher not implemented')
    if rij is None:
        rij = cp.linalg.norm(Rij, axis=-1)

    res = list()

    Tij = 1 / rij
    if 0 in multipole_orders:
        res.append(Tij)
    if 1 in multipole_orders:
        Tija = -Rij * Tij[:,:,None]**3
        res.append(Tija)
    if 2 in multipole_orders:
        Tijab  = 3 * Rij[:,:,:,None] * Rij[:,:,None,:] * Tij[:,:,None,None]**5
        Tijab -= Tij[:,:,None,None]**3 * cp.eye(3)[None,None,:,:]
        res.append(Tijab)
    if 3 in multipole_orders:
        Tijabc = -15 * Rij[:,:,:,None,None] * Rij[:,:,None,:,None] * Rij[:,:,None,None,:]\
            * Tij[:,:,None,None,None]**7
        Rij = Rij * Tij[:,:,None]**5
        Tijabc += 3 * Rij[:,:,:,None,None] * cp.eye(3)[None,None,None,:,:]
        Tijabc += 3 * Rij[:,:,None,:,None] * cp.eye(3)[None,None,:,None,:]
        Tijabc += 3 * Rij[:,:,None,None,:] * cp.eye(3)[None,None,:,:,None]
        res.append(Tijabc)
    return res

def get_multipole_tensors_pg(Rij, eta, multipole_orders, rij=None):
    ''' Compute the interaction tensors between point and gaussian multipoles.

    Args:
        Rij : 3D array, shape (Ni,Nj,3)
            Distance vectors pointing from particle j to i
        eta : float or 1D array, shape (Nj,)
            Exponents of gaussians
        multipole_orders : list of int
            orders of multipole interactions to compute
            0: Tij, 1: Tija, 2: Tijab, ...

    Kwargs:
        rij : 2D array, shape (Ni,Nj)
            Distance between particles i and j

    Returns:
        A list of multipole tensors requested in ascending order
    '''
    if max(multipole_orders) > 3:
        raise NotImplementedError('Tijabcd or higher not implemented')
    if rij is None:
        rij = cp.linalg.norm(Rij, axis=-1)

    res_dict = dict()

    if isinstance(eta, float):
        Tij = erfc(eta * rij) / rij
        res_dict[0] = Tij
        if max(multipole_orders) > 0:
            ekR = cp.exp(-eta**2 * rij**2)
            invr3 = (Tij + 2*eta/np.sqrt(np.pi) * ekR) / rij**2
            Tija = -Rij * invr3[:,:,None]
            res_dict[1] = Tija
        if max(multipole_orders) > 1:
            Tijab  = Rij /rij[:,:,None]**2
            Tijab  = 3 * Rij[:,:,:,None] * Tijab[:,:,None,:]
            Tijab -= cp.ones_like(rij)[:,:,None,None] * cp.eye(3)[None,None,:,:]
            invr5 = invr3 + 4/3*eta**3/np.sqrt(np.pi) * ekR # NOTE this is invr5 * r**2
            Tijab = Tijab * invr5[:,:,None,None]
            # NOTE the below is present in Eq 8 but missing in Eq 12
            # J. Chem. Phys. 119, 7471–7483 (2003)
            Tijab += 4/3*eta**3/np.sqrt(np.pi)* ekR[:,:,None,None] * cp.eye(3)[None,None,:,:]
            res_dict[2] = Tijab
        if max(multipole_orders) > 2:
            invr7 = invr5 / rij**2 + 8/15 / np.sqrt(np.pi) * eta**5 * ekR  # NOTE this is invr7 * r**2
            Tijabc  = -15 * Rij[:,:,:,None,None] * Rij[:,:,None,:,None] * Rij[:,:,None,None,:]\
                / rij[:,:,None,None,None]**2
            Tijabc += 3 * Rij[:,:,:,None,None] * cp.eye(3)[None,None,None,:,:]
            Tijabc += 3 * Rij[:,:,None,:,None] * cp.eye(3)[None,None,:,None,:]
            Tijabc += 3 * Rij[:,:,None,None,:] * cp.eye(3)[None,None,:,:,None]
            Tijabc = Tijabc * invr7[:,:,None,None,None]
            ekRR = ekR[:,:,None] * Rij
            Tijabc -= 8/5 / np.sqrt(np.pi) * eta**5 * ekRR[:,:,:,None,None] * cp.eye(3)[None,None,None,:,:]
            Tijabc -= 8/5 / np.sqrt(np.pi) * eta**5 * ekRR[:,:,None,:,None] * cp.eye(3)[None,None,:,None,:]
            Tijabc -= 8/5 / np.sqrt(np.pi) * eta**5 * ekRR[:,:,None,None,:] * cp.eye(3)[None,None,:,:,None]
            res_dict[3] = Tijabc
    else:
        Tij = erfc(eta * rij) / rij
        res_dict[0] = Tij
        if max(multipole_orders) > 0:
            ekR = cp.exp(-eta**2 * rij**2)
            invr3 = (Tij + 2*eta/np.sqrt(np.pi) * ekR) / rij**2
            Tija = -Rij * invr3[:,:,None]
            res_dict[1] = Tija
        if max(multipole_orders) > 1:
            Tijab  = Rij /rij[:,:,None]**2
            Tijab  = 3 * Rij[:,:,:,None] * Tijab[:,:,None,:]
            Tijab -= cp.ones_like(rij)[:,:,None,None] * cp.eye(3)[None,None,:,:]
            invr5 = invr3 + 4/3*eta**3/np.sqrt(np.pi) * ekR # NOTE this is invr5 * r**2
            Tijab = Tijab * invr5[:,:,None,None]
            eekR = eta[None,:]**3 * ekR
            Tijab += 4/3/np.sqrt(np.pi)* eekR[:,:,None,None] * cp.eye(3)[None,None,:,:]
            res_dict[2] = Tijab
        if max(multipole_orders) > 2:
            invr7 = invr5 / rij**2 + 8/15 / np.sqrt(np.pi) * eta**5 * ekR  # NOTE this is invr7 * r**2
            Tijabc  = -15 * Rij[:,:,:,None,None] * Rij[:,:,None,:,None] * Rij[:,:,None,None,:]\
                / rij[:,:,None,None,None]**2
            Tijabc += 3 * Rij[:,:,:,None,None] * cp.eye(3)[None,None,None,:,:]
            Tijabc += 3 * Rij[:,:,None,:,None] * cp.eye(3)[None,None,:,None,:]
            Tijabc += 3 * Rij[:,:,None,None,:] * cp.eye(3)[None,None,:,:,None]
            Tijabc = Tijabc * invr7[:,:,None,None,None]
            eekRR = eta[None,:,None]**2 * eekR[:,:,None] * Rij
            Tijabc -= 8/5 / np.sqrt(np.pi) * eekRR[:,:,:,None,None] * cp.eye(3)[None,None,None,:,:]
            Tijabc -= 8/5 / np.sqrt(np.pi) * eekRR[:,:,None,:,None] * cp.eye(3)[None,None,:,None,:]
            Tijabc -= 8/5 / np.sqrt(np.pi) * eekRR[:,:,None,None,:] * cp.eye(3)[None,None,:,:,None]
            res_dict[3] = Tijabc

    res = list()
    for i in np.unique(multipole_orders):
        res.append(res_dict[i])
    return res

def get_qm_octupoles(mol, dm):
    dm = cp.asarray(dm)
    nao = mol.nao
    bas_atom = mol._bas[:,gto.ATOM_OF]
    aoslices = mol.aoslice_by_atom()
    qm_octupoles = list()
    for i in range(mol.natm):
        b0, b1 = np.where(bas_atom == i)[0][[0,-1]]
        shls_slice = (0, mol.nbas, b0, b1+1)
        with mol.with_common_orig(mol.atom_coord(i)):
            s1rrr = mol.intor('int1e_rrr', shls_slice=shls_slice)
            s1rrr = s1rrr.reshape((3,3,3,nao,-1))
        p0, p1 = aoslices[i, 2:]
        qm_octupoles.append(
            -contract('uv,xyzvu->xyz', dm[p0:p1], cp.asarray(s1rrr)))
    qm_octupoles = cp.asarray(qm_octupoles)
    return qm_octupoles

def energy_octupole(coords1, coords2, octupoles, charges):
    mem_avail = cupy_helper.get_avail_mem()
    blksize = int(mem_avail/64/3/(1+len(coords2)))
    if blksize == 0:
        raise RuntimeError(f"Not enough GPU memory, mem_avail = {mem_avail}, blkszie = {blksize}")
    ene = 0
    for i0, i1 in lib.prange(0, len(coords1), blksize):
        Rij = coords1[i0:i1,None,:] - coords2[None,:,:]
        Tijabc = get_multipole_tensors_pp(Rij, [3])[0]
        vj = contract('ijabc,iabc->j', Tijabc, octupoles[i0:i1])
        ene += cp.dot(vj, charges) / 6
    return ene.get()

def loop_icell(i, a):
    '''loop over cell images in i-th layer around the center cell
    '''
    if i == 0:
        yield cp.zeros(3)
    else:
        for nx in [-i,i]:
            for ny in range(-i,i+1):
                for nz in range(-i,i+1):
                    yield cp.dot(cp.asarray([nx, ny, nz]), a)
        for nx in range(-i+1,i):
            for ny in [-i,i]:
                for nz in range(-i,i+1):
                    yield cp.dot(cp.asarray([nx, ny, nz]), a)
        for nx in range(-i+1,i):
            for ny in range(-i+1,i):
                for nz in [-i, i]:
                    yield cp.dot(cp.asarray([nx, ny, nz]), a)

def estimate_error(mol, mm_coords, a, mm_charges, rcut_hcore, dm, precision=1e-8, unit='angstrom'):
    qm_octupoles = get_qm_octupoles(mol, dm)

    a = cp.asarray(a)
    mm_coords = cp.asarray(mm_coords)
    mm_charges = cp.asarray(mm_charges)
    if not is_au(unit):
        mm_coords = mm_coords / param.BOHR
        a = a / param.BOHR
        rcut_hcore = rcut_hcore / param.BOHR

    qm_coords = cp.asarray(mol.atom_coords())
    qm_cen = cp.mean(qm_coords, axis=0)

    err_tot = 0
    icell = 0
    while True:
        err_icell = 0
        for shift in loop_icell(icell, a):
            coords2 = mm_coords + shift
            dist2 = coords2 - qm_cen
            dist2 = contract('ix,ix->i', dist2, dist2)
            mask = dist2 > rcut_hcore**2
            coords2 = coords2[mask]
            if coords2.size != 0:
                err_icell += energy_octupole(qm_coords, coords2, qm_octupoles, mm_charges[mask])
        err_tot += err_icell
        if abs(err_icell) < precision and icell > 0:
            break
        icell += 1
    return err_tot

def determine_hcore_cutoff(mol, mm_coords, a, mm_charges, rcut_min, dm, rcut_step=1.0, precision=1e-4, rcut_max=1e4, unit='angstrom'):

    qm_octupoles = get_qm_octupoles(mol, dm)

    a = cp.asarray(a)
    mm_coords = cp.asarray(mm_coords)
    mm_charges = cp.asarray(mm_charges)
    if not is_au(unit):
        mm_coords = mm_coords / param.BOHR
        a = a / param.BOHR
        rcut_min = rcut_min / param.BOHR
        rcut_step = rcut_step / param.BOHR
        rcut_max = rcut_max / param.BOHR

    qm_coords = cp.asarray(mol.atom_coords())
    qm_cen = cp.mean(qm_coords, axis=0)
    rs_precision = .01 * precision

    err_tot = 0
    icell = 0
    while True:
        err_icell = 0
        for shift in loop_icell(icell, a):
            coords2 = mm_coords + shift
            dist2 = coords2 - qm_cen
            dist2 = contract('ix,ix->i', dist2, dist2)
            mask = dist2 > rcut_min**2
            coords2 = coords2[mask]
            if coords2.size != 0:
                err_icell += energy_octupole(qm_coords, coords2, qm_octupoles, mm_charges[mask])
        err_tot += err_icell
        if abs(err_icell) < rs_precision:
            break
        icell += 1

    max_icell = icell
    rcut = rcut_min
    trust_level = 0
    for rcut in np.arange(rcut_min, rcut_max, rcut_step):
        err_rcut = err_tot
        for icell in range(max_icell+1):
            for shift in loop_icell(icell, a):
                coords2 = mm_coords + shift
                dist2 = coords2 - qm_cen
                dist2 = contract('ix,ix->i', dist2, dist2)
                mask = (dist2 > rcut_min**2) & (dist2 <= rcut**2)
                coords2 = coords2[mask]
                if coords2.size != 0:
                    err_rcut -= energy_octupole(qm_coords, coords2, qm_octupoles, mm_charges[mask])
        if abs(err_rcut) < precision:
            trust_level += 1
        else:
            trust_level = 0
        if trust_level > 1:
            break
    if not is_au(unit):
        rcut = rcut * param.BOHR
    return rcut, err_rcut
