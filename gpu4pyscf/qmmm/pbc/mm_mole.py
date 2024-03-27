import numpy
from pyscf import gto
from pyscf import pbc
from pyscf import qmmm
from pyscf.gto.mole import is_au
from pyscf.data.elements import charge
from pyscf.lib import param, logger
from scipy.special import lambertw
from cupyx.scipy.special import erf, erfc
from pyscf import lib

import cupy as cp
from gpu4pyscf.df.df import ALIGNED, MIN_BLK_SIZE
from gpu4pyscf.lib import cupy_helper

class Cell(qmmm.mm_mole.Mole, pbc.gto.Cell):
    '''Cell class for MM particles.

    Args:
        atoms : geometry of MM particles (unit Bohr).

            | [[atom1, (x, y, z)],
            |  [atom2, (x, y, z)],
            |  ...
            |  [atomN, (x, y, z)]]

    Kwargs:
        charges : 1D array
            fractional charges of MM particles
        zeta : 1D array
            Gaussian charge distribution parameter.
            rho(r) = charge * Norm * exp(-zeta * r^2)

    '''
    def __init__(self, atoms, a, 
            rcut_ewald=None, rcut_hcore=None,
            charges=None, zeta=None):
        pbc.gto.Cell.__init__(self)
        self.atom = self._atom = atoms
        self.unit = 'Bohr'
        self.charge_model = 'point'
        assert numpy.linalg.norm(a - numpy.diag(numpy.diag(a))) < 1e-12
        self.a = a
        if rcut_ewald is None:
            rcut_ewald = min(numpy.diag(a)) * 0.5
            logger.warn(self, "Setting rcut_ewald to be half box size")
        if rcut_hcore is None:
            rcut_hcore = numpy.linalg.norm(numpy.diag(a)) / 2
            logger.warn(self, "Setting rcut_hcore to be half box diagonal")
        # rcut_ewald has to be < box size cuz my get_lattice_Ls only considers nearest cell
        assert rcut_ewald < min(numpy.diag(a)), "Only rcut_ewald < box size implemented"
        self.rcut_ewald = rcut_ewald
        self.rcut_hcore = rcut_hcore

        # Initialize ._atm and ._env to save the coordinates and charges and
        # other info of MM particles
        natm = len(atoms)
        _atm = numpy.zeros((natm,6), dtype=numpy.int32)
        _atm[:,gto.CHARGE_OF] = [charge(a[0]) for a in atoms]
        coords = numpy.asarray([a[1] for a in atoms], dtype=numpy.double)
        if charges is None:
            _atm[:,gto.NUC_MOD_OF] = gto.NUC_POINT
            charges = _atm[:,gto.CHARGE_OF:gto.CHARGE_OF+1]
        else:
            _atm[:,gto.NUC_MOD_OF] = gto.NUC_FRAC_CHARGE
            charges = numpy.asarray(charges)[:,numpy.newaxis]

        self._env = numpy.append(numpy.zeros(gto.PTR_ENV_START),
                                 numpy.hstack((coords, charges)).ravel())
        _atm[:,gto.PTR_COORD] = gto.PTR_ENV_START + numpy.arange(natm) * 4
        _atm[:,gto.PTR_FRAC_CHARGE] = gto.PTR_ENV_START + numpy.arange(natm) * 4 + 3

        if zeta is not None:
            self.charge_model = 'gaussian'
            zeta = numpy.asarray(zeta, dtype=float).ravel()
            self._env = numpy.append(self._env, zeta)
            _atm[:,gto.PTR_ZETA] = gto.PTR_ENV_START + natm*4 + numpy.arange(natm)

        self._atm = _atm

        eta, _ = self.get_ewald_params()
        e = self.precision
        Q = numpy.sum(self.atom_charges()**2)
        L = self.vol**(1/3)
        kmax = numpy.sqrt(3)*eta/2/numpy.pi * numpy.sqrt(lambertw( 4*Q**(2/3)/3/numpy.pi**(2/3)/L**2/eta**(2/3) / e**(4/3) ).real)
        self.mesh = numpy.ceil(numpy.diag(self.lattice_vectors()) * kmax).astype(int) * 2 + 1

        self._built = True

    def get_lattice_Ls(self):
        Ts = lib.cartesian_prod((numpy.arange(-1, 2),
                                 numpy.arange(-1, 2),
                                 numpy.arange(-1, 2)))
        Lall = numpy.dot(Ts, self.lattice_vectors())
        return Lall

    def get_ewald_params(self, precision=None, rcut=None):
        if rcut is None:
            ew_cut = self.rcut_ewald
        else:
            ew_cut = rcut
        if precision is None:
            precision = self.precision
        e = precision
        Q = numpy.sum(self.atom_charges()**2)
        ew_eta = 1 / ew_cut * numpy.sqrt(lambertw(1/e*numpy.sqrt(Q/2/self.vol)).real)
        return ew_eta, ew_cut

    def get_ewald_pot(self, coords1, coords2=None, charges2=None, zetas2=None):
        assert self.dimension == 3
        assert (coords2 is None and charges2 is None) or \
            (coords2 is not None and charges2 is not None and zetas2 is not None)
        coords1 = cp.asarray(coords1)
        if coords2 is not None:
            coords2 = cp.asarray(coords2)

        if charges2 is not None:
            assert len(charges2) == len(coords2)
        else:
            coords2 = coords1

        ew_eta, ew_cut = self.get_ewald_params()
        mesh = self.mesh
        
        logger.debug(self, f"Ewald exponent {ew_eta}")

        # TODO Lall should respect ew_rcut
        Lall = cp.asarray(self.get_lattice_Ls())

        all_coords2 = (coords2[None,:,:] - Lall[:,None,:]).reshape(-1,3)
        all_coords2 = cp.asarray(all_coords2)
        if charges2 is not None:
            all_charges2 = cp.hstack([charges2] * len(Lall))
        else:
            all_charges2 = None
        dist2 = all_coords2 - cp.mean(coords1, axis=0)[None]
        dist2 = cp.einsum('jx,jx->j', dist2, dist2)

        if all_charges2 is not None:
            ewovrl0 = cp.zeros(len(coords1))
            ewovrl1 = cp.zeros((len(coords1), 3))
            ewovrl2 = cp.zeros((len(coords1), 3, 3))
        else:
            ewovrl00 = cp.zeros((len(coords1), len(coords1))) 
            ewovrl01 = cp.zeros((len(coords1), len(coords1), 3)) 
            ewovrl11 = cp.zeros((len(coords1), len(coords1), 3, 3)) 
            ewovrl02 = cp.zeros((len(coords1), len(coords1), 3, 3)) 
            ewself00 = cp.zeros((len(coords1), len(coords1))) 
            ewself01 = cp.zeros((len(coords1), len(coords1), 3)) 
            ewself11 = cp.zeros((len(coords1), len(coords1), 3, 3)) 
            ewself02 = cp.zeros((len(coords1), len(coords1), 3, 3)) 

        mem_avail = cupy_helper.get_avail_mem()
        blksize = int(mem_avail/64/3/len(all_coords2))
        if blksize == 0:
            raise RuntimeError(f"Not enough GPU memory, mem_avail = {mem_avail}, blkszie = {blksize}")
        for i0, i1 in lib.prange(0, len(coords1), blksize):
            R = coords1[i0:i1,None,:] - all_coords2[None,:,:]
            r = cp.linalg.norm(R, axis=-1)
            r[r<1e-16] = 1e100
            rmax_qm = max(cp.linalg.norm(coords1 - cp.mean(coords1, axis=0), axis=-1))
    
            # substract the real-space Coulomb within rcut_hcore
            mask = dist2 <= self.rcut_hcore**2
            Tij = 1 / r[:,mask]
            Rij = R[:,mask]
            Tija = -cp.einsum('ijx,ij->ijx', Rij, Tij**3)
            Tijab  = 3 * cp.einsum('ija,ijb->ijab', Rij, Rij) 
            Tijab  = cp.einsum('ijab,ij->ijab', Tijab, Tij**5)
            Tijab -= cp.einsum('ij,ab->ijab', Tij**3, cp.eye(3))
            if all_charges2 is not None:
                charges = all_charges2[mask]
                # ew0 = -d^2 E / dQi dqj qj
                # ew1 = -d^2 E / dDia dqj qj
                # ew2 = -d^2 E / dOiab dqj qj
                # qm pc - mm pc
                ewovrl0[i0:i1] += -cp.einsum('ij,j->i', Tij, charges)
                # qm dip - mm pc
                ewovrl1[i0:i1] += -cp.einsum('j,ija->ia', charges, Tija)
                # qm quad - mm pc
                ewovrl2[i0:i1] += -cp.einsum('j,ijab->iab', charges, Tijab) / 3
            else:
                # NOTE a too small rcut_hcore truncates QM atoms, while this correction
                # should be applied to all QM pairs regardless of rcut_hcore
                # NOTE this is now checked in get_hcore
                #assert r[:,mask].shape[0] == r[:,mask].shape[1]   # real-space should not see qm images
                # ew00 = -d^2 E / dQi dQj
                # ew01 = -d^2 E / dQi dDja
                # ew11 = -d^2 E / dDia dDjb
                # ew02 = -d^2 E / dQi dOjab
                ewovrl00[i0:i1] += -Tij
                ewovrl01[i0:i1] +=  Tija
                ewovrl11[i0:i1] +=  Tijab
                ewovrl02[i0:i1] += -Tijab / 3
    
            # difference between MM gaussain charges and MM point charges
            if all_charges2 is not None and self.charge_model == 'gaussian':
                zetas = cp.asarray(zetas2)
                mask = dist2 > self.rcut_hcore**2
                min_expnt = cp.min(zetas)
                max_ewrcut = pbc.gto.cell._estimate_rcut(min_expnt, 0, 1., self.precision)
                cut2 = (max_ewrcut + rmax_qm)**2
                mask = mask & (dist2 <= cut2)
                expnts = cp.hstack([cp.sqrt(zetas)] * len(Lall))[mask]
                r_ = r[:,mask]
                R_ = R[:,mask]
                ekR = cp.exp(-cp.einsum('j,ij->ij', expnts**2, r_**2))
                Tij = erfc(cp.einsum('j,ij->ij', expnts, r_)) / r_
                invr3 = (Tij + cp.einsum('j,ij->ij', expnts, 2/cp.sqrt(cp.pi)*ekR)) / r_**2
                Tija = -cp.einsum('ijx,ij->ijx', R_, invr3)
                Tijab  = 3 * cp.einsum('ija,ijb,ij->ijab', R_, R_, 1/r_**2)
                Tijab -= cp.einsum('ij,ab->ijab', cp.ones_like(r_), cp.eye(3))
                invr5 = invr3 + cp.einsum('j,ij->ij', expnts**3, 4/3/cp.sqrt(cp.pi) * ekR)
                Tijab = cp.einsum('ijab,ij->ijab', Tijab, invr5)
                Tijab += cp.einsum('j,ij,ab->ijab', expnts**3, 4/3/cp.sqrt(cp.pi)*ekR, cp.eye(3))
                ewovrl0[i0:i1] -= cp.einsum('ij,j->i', Tij, all_charges2[mask])
                ewovrl1[i0:i1] -= cp.einsum('j,ija->ia', all_charges2[mask], Tija)
                ewovrl2[i0:i1] -= cp.einsum('j,ijab->iab', all_charges2[mask], Tijab) / 3
    
            # ewald real-space sum
            if all_charges2 is not None:
                cut2 = (ew_cut + rmax_qm)**2
                mask = dist2 <= cut2
                r_ = r[:,mask]
                R_ = R[:,mask]
                all_charges2_ = all_charges2[mask]
            else:
                # ewald sum will run over all qm images regardless of ew_cut
                # this is to ensure r and R will always have the shape of (i1-i0, L*num_qm)
                r_ = r
                R_ = R
            ekR = cp.exp(-ew_eta**2 * r_**2)
            # Tij = \hat{1/r} = f0 / r = erfc(r) / r
            Tij = erfc(ew_eta * r_) / r_
            # Tija = -Rija \hat{1/r^3} = -Rija / r^2 ( \hat{1/r} + 2 eta/sqrt(pi) exp(-eta^2 r^2) )
            invr3 = (Tij + 2*ew_eta/cp.sqrt(cp.pi) * ekR) / r_**2
            Tija = -cp.einsum('ijx,ij->ijx', R_, invr3)
            # Tijab = (3 RijaRijb - Rij^2 delta_ab) \hat{1/r^5}
            Tijab  = 3 * cp.einsum('ija,ijb,ij->ijab', R_, R_, 1/r_**2)
            Tijab -= cp.einsum('ij,ab->ijab', cp.ones_like(r_), cp.eye(3))
            invr5 = invr3 + 4/3*ew_eta**3/cp.sqrt(cp.pi) * ekR # NOTE this is invr5 * r**2
            Tijab = cp.einsum('ijab,ij->ijab', Tijab, invr5)
            # NOTE the below is present in Eq 8 but missing in Eq 12
            Tijab += 4/3*ew_eta**3/cp.sqrt(cp.pi)*cp.einsum('ij,ab->ijab', ekR, cp.eye(3))
    
            if all_charges2 is not None:
                ewovrl0[i0:i1] += cp.einsum('ij,j->i', Tij, all_charges2_)
                ewovrl1[i0:i1] += cp.einsum('j,ija->ia', all_charges2_, Tija)
                ewovrl2[i0:i1] += cp.einsum('j,ijab->iab', all_charges2_, Tijab) / 3
            else:
                Tij = cp.sum(Tij.reshape(i1-i0, len(Lall), len(coords1)), axis=1)
                Tija = cp.sum(Tija.reshape(i1-i0, len(Lall), len(coords1), 3), axis=1)
                Tijab = cp.sum(Tijab.reshape(i1-i0, len(Lall), len(coords1), 3, 3), axis=1)
                ewovrl00[i0:i1] += Tij
                ewovrl01[i0:i1] -= Tija
                ewovrl11[i0:i1] -= Tijab
                ewovrl02[i0:i1] += Tijab / 3
            ekR = Tij = invr3 = Tijab = invr5 = None
    
            if all_charges2 is not None:
                pass
            else:
                ewself01[i0:i1] += 0
                ewself02[i0:i1] += 0
                # -d^2 Eself / dQi dQj
                ewself00[i0:i1] += -cp.eye(len(coords1))[i0:i1] * 2 * ew_eta / cp.sqrt(cp.pi)
                # -d^2 Eself / dDia dDjb
                ewself11[i0:i1] += -cp.einsum('ij,ab->ijab', cp.eye(len(coords1))[i0:i1], cp.eye(3)) \
                        * 4 * ew_eta**3 / 3 / cp.sqrt(cp.pi)

            r_ = R_ = all_charges2_ = None
    
        R = r = dist2 = all_charges2 = mask = None

        # g-space sum (using g grid)
        logger.debug(self, f"Ewald mesh {mesh}")

        Gv, Gvbase, weights = self.get_Gv_weights(mesh)
        absG2 = cp.einsum('gx,gx->g', Gv, Gv)
        absG2[absG2==0] = 1e200

        coulG = 4*cp.pi / absG2
        coulG *= weights
        # NOTE Gpref is actually Gpref*2
        Gpref = cp.exp(-absG2/(4*ew_eta**2)) * coulG

        GvR2 = cp.einsum('gx,ix->ig', Gv, coords2)
        cosGvR2 = cp.cos(GvR2)
        sinGvR2 = cp.sin(GvR2)

        if charges2 is not None:
            GvR1 = cp.einsum('gx,ix->ig', Gv, coords1)
            cosGvR1 = cp.cos(GvR1)
            sinGvR1 = cp.sin(GvR1)
            zcosGvR2 = cp.einsum("i,ig->g", charges2, cosGvR2)
            zsinGvR2 = cp.einsum("i,ig->g", charges2, sinGvR2)
            # qm pc - mm pc
            ewg0  = cp.einsum('ig,g,g->i', cosGvR1, zcosGvR2, Gpref)
            ewg0 += cp.einsum('ig,g,g->i', sinGvR1, zsinGvR2, Gpref)
            # qm dip - mm pc
            p = ['einsum_path', (2, 3), (0, 2), (0, 1)]
            ewg1  = cp.einsum('gx,ig,g,g->ix', Gv, cosGvR1, zsinGvR2, Gpref, optimize=p)
            ewg1 -= cp.einsum('gx,ig,g,g->ix', Gv, sinGvR1, zcosGvR2, Gpref, optimize=p)
            # qm quad - mm pc
            p = ['einsum_path', (3, 4), (0, 3), (0, 2), (0, 1)]
            ewg2  = -cp.einsum('gx,gy,ig,g,g->ixy', Gv, Gv, cosGvR1, zcosGvR2, Gpref, optimize=p)
            ewg2 += -cp.einsum('gx,gy,ig,g,g->ixy', Gv, Gv, sinGvR1, zsinGvR2, Gpref, optimize=p)
            ewg2 /= 3
        else:
            # qm pc - qm pc
            ewg00  = cp.einsum('ig,jg,g->ij', cosGvR2, cosGvR2, Gpref)
            ewg00 += cp.einsum('ig,jg,g->ij', sinGvR2, sinGvR2, Gpref)
            # qm pc - qm dip
            ewg01  = cp.einsum('gx,ig,jg,g->ijx', Gv, sinGvR2, cosGvR2, Gpref)
            ewg01 -= cp.einsum('gx,ig,jg,g->ijx', Gv, cosGvR2, sinGvR2, Gpref)
            # qm dip - qm dip
            ewg11  = cp.einsum('gx,gy,ig,jg,g->ijxy', Gv, Gv, cosGvR2, cosGvR2, Gpref)
            ewg11 += cp.einsum('gx,gy,ig,jg,g->ijxy', Gv, Gv, sinGvR2, sinGvR2, Gpref)
            # qm pc - qm quad
            ewg02  = -cp.einsum('gx,gy,ig,jg,g->ijxy', Gv, Gv, cosGvR2, cosGvR2, Gpref)
            ewg02 += -cp.einsum('gx,gy,ig,jg,g->ijxy', Gv, Gv, sinGvR2, sinGvR2, Gpref)
            ewg02 /= 3

        if charges2 is not None:
            return ewovrl0 + ewg0, ewovrl1 + ewg1, ewovrl2 + ewg2
        else:
            return ewovrl00 + ewself00 + ewg00, \
                   ewovrl01 + ewself01 + ewg01, \
                   ewovrl11 + ewself11 + ewg11, \
                   ewovrl02 + ewself02 + ewg02

def create_mm_mol(atoms_or_coords, a, charges=None, radii=None, 
        rcut_ewald=None, rcut_hcore=None, unit='Angstrom'):
    '''Create an MM object based on the given coordinates and charges of MM
    particles.

    Args:
        atoms_or_coords : array-like
            Cartesian coordinates of MM atoms, in the form of a 2D array:
            [(x1, y1, z1), (x2, y2, z2), ...]
        a : (3,3) ndarray
            Lattice primitive vectors. Each row represents a lattice vector
            Reciprocal lattice vectors are given by  b1,b2,b3 = 2 pi inv(a).T

    Kwargs:
        charges : 1D array
            The charges of MM atoms.
        radii : 1D array
            The Gaussian charge distribuction radii of MM atoms.
        unit : string
            The unit of the input. Default is 'Angstrom'.
    '''
    if isinstance(atoms_or_coords, numpy.ndarray):
        # atoms_or_coords == np.array([(xx, xx, xx)])
        # Patch ghost atoms
        atoms = [(0, c) for c in atoms_or_coords]
    elif (isinstance(atoms_or_coords, (list, tuple)) and
          atoms_or_coords and
          isinstance(atoms_or_coords[0][1], (int, float))):
        # atoms_or_coords == [(xx, xx, xx)]
        # Patch ghost atoms
        atoms = [(0, c) for c in atoms_or_coords]
    else:
        atoms = atoms_or_coords
    atoms = gto.format_atom(atoms, unit=unit)

    if radii is None:
        zeta = None
    else:
        radii = numpy.asarray(radii, dtype=float).ravel()
        if not is_au(unit):
            radii = radii / param.BOHR
        zeta = 1 / radii**2

    kwargs = {'charges': charges, 'zeta': zeta}

    if not is_au(unit):
        a = a / param.BOHR
        if rcut_ewald is not None:
            rcut_ewald = rcut_ewald / param.BOHR
        if rcut_hcore is not None:
            rcut_hcore = rcut_hcore / param.BOHR

    if rcut_ewald is not None:
        kwargs['rcut_ewald'] = rcut_ewald
    if rcut_hcore is not None:
        kwargs['rcut_hcore'] = rcut_hcore

    return Cell(atoms, a, **kwargs)

create_mm_cell = create_mm_mol
