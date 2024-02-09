import cupy as cp
import numpy as np

from pyscf import gto, lib
from pyscf.gto.mole import is_au
from pyscf.lib import param, logger

from gpu4pyscf.lib import cupy_helper

def get_qm_octupoles(mol, dm):
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
            -lib.einsum('uv,xyzvu->xyz', dm[p0:p1], s1rrr))
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
        rij = cp.linalg.norm(Rij, axis=-1)
        Tij = 1 / rij
        Tijabc  = -15 * cp.einsum('ija,ijb,ijc->ijabc', Rij, Rij, Rij)
        Tijabc  = cp.einsum('ijabc,ij->ijabc', Tijabc, Tij**7)
        Tijabc += 3 * cp.einsum('ija,bc,ij->ijabc', Rij, np.eye(3), Tij**5)
        Tijabc += 3 * cp.einsum('ijb,ac,ij->ijabc', Rij, np.eye(3), Tij**5)
        Tijabc += 3 * cp.einsum('ijc,ab,ij->ijabc', Rij, np.eye(3), Tij**5)
        vj = cp.einsum('ijabc,iabc->j', Tijabc, octupoles[i0:i1])
        ene += vj @ charges / 6
    return ene.get()

def loop_icell(i, a):
    ''' loop over cell images in i-th layer around the center cell
    '''
    if i == 0:
        yield cp.zeros(3)
    else:
        for nx in [-i,i]:
            for ny in range(-i,i+1):
                for nz in range(-i,i+1):
                    yield cp.asarray([nx, ny, nz]) @ a
        for nx in range(-i+1,i):
            for ny in [-i,i]:
                for nz in range(-i,i+1):
                    yield cp.asarray([nx, ny, nz]) @ a
        for nx in range(-i+1,i):
            for ny in range(-i+1,i):
                for nz in [-i, i]:
                    yield cp.asarray([nx, ny, nz]) @ a

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
            dist2 = cp.einsum('ix,ix->i', dist2, dist2)
            mask = dist2 > rcut_hcore**2
            coords2 = coords2[mask]
            err_icell += energy_octupole(qm_coords, coords2, qm_octupoles, mm_charges[mask])
        err_tot += err_icell
        if abs(err_icell) < precision:
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
            dist2 = cp.einsum('ix,ix->i', dist2, dist2)
            mask = dist2 > rcut_min**2
            coords2 = coords2[mask]
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
                dist2 = cp.einsum('ix,ix->i', dist2, dist2)
                mask = (dist2 > rcut_min**2) & (dist2 <= rcut**2)
                coords2 = coords2[mask]
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
