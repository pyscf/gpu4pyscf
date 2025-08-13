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

import numpy as np
import cupy
import h5py
import itertools
from functools import reduce
from pyscf import gto
from pyscf import lib as pyscf_lib
from pyscf.scf import hf as hf_cpu
from pyscf.scf import chkfile
from gpu4pyscf.gto.ecp import get_ecp
from gpu4pyscf import lib
from gpu4pyscf.lib import utils
from gpu4pyscf.lib.cupy_helper import (
    eigh, tag_array, return_cupy_array, cond, asarray, get_avail_mem,
    block_diag, sandwich_dot)
from gpu4pyscf.scf import diis, jk, j_engine
from gpu4pyscf.lib import logger

__all__ = [
    'get_jk', 'get_occ', 'get_grad', 'damping', 'level_shift', 'get_fock',
    'energy_elec', 'RHF', 'SCF'
]

def get_jk(mol, dm, hermi=1, vhfopt=None, with_j=True, with_k=True, omega=None,
           verbose=None):
    '''Compute J, K matrices with CPU-GPU hybrid algorithm
    '''
    with mol.with_range_coulomb(omega):
        vj, vk = jk.get_jk(mol, dm, hermi, vhfopt, with_j, with_k, verbose)
    if not isinstance(dm, cupy.ndarray):
        if with_j: vj = vj.get()
        if with_k: vk = vk.get()
    return vj, vk

def _get_jk(mf, mol, dm=None, hermi=1, with_j=True, with_k=True,
            omega=None):
    if omega is None:
        omega = mol.omega
    vhfopt = mf._opt_gpu.get(omega)
    if vhfopt is None:
        with mol.with_range_coulomb(omega):
            vhfopt = mf._opt_gpu[omega] = jk._VHFOpt(mol, mf.direct_scf_tol).build()

    vj, vk = get_jk(mol, dm, hermi, vhfopt, with_j, with_k, omega)
    return vj, vk

def make_rdm1(mo_coeff, mo_occ):
    mo_coeff = cupy.asarray(mo_coeff)
    mo_occ = cupy.asarray(mo_occ)
    is_occ = mo_occ > 0
    mocc = mo_coeff[:, is_occ]
    dm = cupy.dot(mocc*mo_occ[is_occ], mocc.conj().T)
    occ_coeff = mo_coeff[:, is_occ]
    return tag_array(dm, occ_coeff=occ_coeff, mo_occ=mo_occ, mo_coeff=mo_coeff)

def get_occ(mf, mo_energy=None, mo_coeff=None):
    if mo_energy is None: mo_energy = mf.mo_energy
    e_idx = cupy.argsort(mo_energy)
    nmo = mo_energy.size
    mo_occ = cupy.zeros(nmo)
    nocc = mf.mol.nelectron // 2
    mo_occ[e_idx[:nocc]] = 2
    if mf.verbose >= logger.INFO and nocc < nmo:
        homo = float(mo_energy[e_idx[nocc-1]])
        lumo = float(mo_energy[e_idx[nocc]])
        if homo+1e-3 > lumo:
            logger.warn(mf, 'HOMO %.15g == LUMO %.15g', homo, lumo)
        else:
            logger.info(mf, '  HOMO = %.15g  LUMO = %.15g', homo, lumo)
    return mo_occ

def get_veff(mf, mol=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
    if dm is None: dm = mf.make_rdm1()
    if dm_last is not None and mf.direct_scf:
        dm = asarray(dm) - asarray(dm_last)
    vj = mf.get_j(mol, dm, hermi)
    vhf = mf.get_k(mol, dm, hermi)
    vhf *= -.5
    vhf += vj
    if vhf_last is not None:
        vhf += asarray(vhf_last)
    return vhf

def get_grad(mo_coeff, mo_occ, fock_ao):
    occidx = mo_occ > 0
    viridx = ~occidx
    g = reduce(cupy.dot, (mo_coeff[:,viridx].conj().T, fock_ao,
                           mo_coeff[:,occidx])) * 2
    return g.ravel()

def damping(s, d, f, factor):
    dm_vir = cupy.eye(s.shape[0]) - cupy.dot(s, d)
    f0 = reduce(cupy.dot, (dm_vir, f, d, s))
    f0 = (f0+f0.conj().T) * (factor/(factor+1.))
    return f - f0

def level_shift(s, d, f, factor):
    dm_vir = s - reduce(cupy.dot, (s, d, s))
    return f + dm_vir * factor

def get_hcore(mol):
    if mol._pseudo:
        # Although mol._pseudo for GTH PP is only available in Cell, GTH PP
        # may exist if mol is converted from cell object.
        from pyscf.gto import pp_int
        h = mol.intor_symmetric('int1e_kin')
        h += pp_int.get_gth_pp(mol)
        h = asarray(h)
    else:
        assert not mol.nucmod
        from gpu4pyscf.gto.int3c1e import int1e_grids
        #:h = mol.intor_symmetric('int1e_nuc')
        h = int1e_grids(mol, mol.atom_coords(), charges=-mol.atom_charges())
        h += asarray(mol.intor_symmetric('int1e_kin'))
    if len(mol._ecpbas) > 0:
        h += get_ecp(mol)
    return h

def get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
    if h1e is None: h1e = mf.get_hcore()
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    h1e = cupy.asarray(h1e)
    vhf = cupy.asarray(vhf)
    f = h1e + vhf
    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if s1e is None: s1e = mf.get_ovlp()
    if dm is None: dm = mf.make_rdm1()
    s1e = cupy.asarray(s1e)
    dm = cupy.asarray(dm)
    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp

    if 0 <= cycle < diis_start_cycle-1 and abs(damp_factor) > 1e-4:
        f = damping(s1e, dm*.5, f, damp_factor)
    if diis is not None and cycle >= diis_start_cycle:
        f = diis.update(s1e, dm, f, mf, h1e, vhf)

    if abs(level_shift_factor) > 1e-4:
        f = level_shift(s1e, dm*.5, f, level_shift_factor)
    return f

def energy_elec(self, dm=None, h1e=None, vhf=None):
    '''
    electronic energy
    '''
    if dm is None: dm = self.make_rdm1()
    if h1e is None: h1e = self.get_hcore()
    if vhf is None: vhf = self.get_veff(self.mol, dm)
    e1 = cupy.einsum('ij,ji->', h1e, dm).real
    e_coul = cupy.einsum('ij,ji->', vhf, dm).real * .5
    e1 = e1.get()[()]
    e_coul = e_coul.get()[()]
    self.scf_summary['e1'] = e1
    self.scf_summary['e2'] = e_coul
    logger.debug(self, 'E1 = %s  E_coul = %s', e1, e_coul)
    return e1+e_coul, e_coul

def _kernel(mf, conv_tol=1e-10, conv_tol_grad=None,
           dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
    conv_tol = mf.conv_tol
    mol = mf.mol
    verbose = mf.verbose
    log = logger.new_logger(mf, verbose)
    t0 = t1 = log.init_timer()
    if(conv_tol_grad is None):
        conv_tol_grad = conv_tol**.5
        log.info('Set gradient conv threshold to %g', conv_tol_grad)

    if dm0 is None:
        dm0 = mf.get_init_guess(mol, mf.init_guess)
        t1 = log.timer_debug1('generating initial guess', *t1)

    if hasattr(dm0, 'mo_coeff') and hasattr(dm0, 'mo_occ'):
        if dm0.ndim == 2:
            mo_coeff = cupy.asarray(dm0.mo_coeff[:,dm0.mo_occ>0])
            mo_occ = cupy.asarray(dm0.mo_occ[dm0.mo_occ>0])
            dm0 = tag_array(dm0, mo_occ=mo_occ, mo_coeff=mo_coeff)
        else:
            # Drop attributes like mo_coeff, mo_occ for UHF and other methods.
            dm0 = asarray(dm0, order='C')

    h1e = cupy.asarray(mf.get_hcore(mol))
    s1e = cupy.asarray(mf.get_ovlp(mol))
    t1 = log.timer_debug1('hcore', *t1)

    dm, dm0 = asarray(dm0, order='C'), None
    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    log.info('init E= %.15g', e_tot)
    t1 = log.timer_debug1('total prep', *t0)
    scf_conv = False

    # Skip SCF iterations. Compute only the total energy of the initial density
    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback
    else:
        mf_diis = None

    dump_chk = dump_chk and mf.chkfile is not None
    if dump_chk:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        chkfile.save_mol(mol, mf.chkfile)

    for cycle in range(mf.max_cycle):
        t0 = log.init_timer()
        mo_coeff = mo_occ = mo_energy = fock = None
        dm_last = dm
        last_hf_e = e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, mf_diis)
        t1 = log.timer_debug1('DIIS', *t0)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        fock = None
        t1 = log.timer_debug1('eig', *t1)

        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        dm = asarray(dm) # Remove the attached attributes
        t1 = log.timer_debug1('veff', *t1)

        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        norm_gorb = cupy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        e_tot = mf.energy_tot(dm, h1e, vhf)

        norm_ddm = cupy.linalg.norm(dm-dm_last)
        t1 = log.timer_debug1('total', *t0)
        log.info('cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                 cycle+1, e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)

        if dump_chk:
            mf.dump_chk(locals())

        e_diff = abs(e_tot-last_hf_e)
        if(e_diff < conv_tol and norm_gorb < conv_tol_grad):
            scf_conv = True
            break
    else:
        log.warn("SCF failed to converge")

    if scf_conv and abs(mf.level_shift) > 0:
        # An extra diagonalization, to remove level shift
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm)
        norm_gorb = cupy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        norm_ddm = cupy.linalg.norm(dm-dm_last)

        conv_tol = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if abs(e_tot-last_hf_e) < conv_tol or norm_gorb < conv_tol_grad:
            scf_conv = True
        else:
            log.warn("Level-shifted SCF extra cycle failed to converge")
            scf_conv = False
        logger.info(mf, 'Extra cycle  E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                    e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)
        if dump_chk:
            mf.dump_chk(locals())

    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ


def energy_tot(mf, dm=None, h1e=None, vhf=None):
    r'''Total Hartree-Fock energy, electronic part plus nuclear repulstion
    See :func:`scf.hf.energy_elec` for the electron part

    Note this function has side effects which cause mf.scf_summary updated.

    '''
    nuc = mf.energy_nuc()
    e_tot = mf.energy_elec(dm, h1e, vhf)[0] + nuc
    if mf.do_disp():
        if 'dispersion' in mf.scf_summary:
            e_tot += mf.scf_summary['dispersion']
        else:
            e_disp = mf.get_dispersion()
            mf.scf_summary['dispersion'] = e_disp
            e_tot += e_disp
    mf.scf_summary['nuc'] = nuc.real
    if isinstance(e_tot, cupy.ndarray):
        e_tot = e_tot.get()
    return e_tot

def scf(mf, dm0=None, **kwargs):
    cput0 = logger.init_timer(mf)

    mf.dump_flags()
    mf.build(mf.mol)

    if dm0 is None and mf.mo_coeff is not None and mf.mo_occ is not None:
        # Initial guess from existing wavefunction
        dm0 = mf.make_rdm1()

    if mf.max_cycle > 0 or mf.mo_coeff is None:
        mf.converged, mf.e_tot, \
                mf.mo_energy, mf.mo_coeff, mf.mo_occ = \
                _kernel(mf, mf.conv_tol, mf.conv_tol_grad,
                        dm0=dm0, callback=mf.callback,
                        conv_check=mf.conv_check, **kwargs)
    else:
        # Avoid to update SCF orbitals in the non-SCF initialization
        # (issue #495).  But run regular SCF for initial guess if SCF was
        # not initialized.
        mf.e_tot = _kernel(mf, mf.conv_tol, mf.conv_tol_grad,
                            dm0=dm0, callback=mf.callback,
                            conv_check=mf.conv_check, **kwargs)[1]

    logger.timer(mf, 'SCF', *cput0)
    mf._finalize()
    return mf.e_tot

def canonicalize(mf, mo_coeff, mo_occ, fock=None):
    '''Canonicalization diagonalizes the Fock matrix within occupied, open,
    virtual subspaces separatedly (without change occupancy).
    '''
    if fock is None:
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        fock = mf.get_fock(dm=dm)
    coreidx = mo_occ == 2
    viridx = mo_occ == 0
    openidx = ~(coreidx | viridx)
    mo = cupy.empty_like(mo_coeff)
    mo_e = cupy.empty(mo_occ.size)
    for idx in (coreidx, openidx, viridx):
        if cupy.any(idx) > 0:
            orb = mo_coeff[:,idx]
            f1 = orb.conj().T.dot(fock).dot(orb)
            e, c = cupy.linalg.eigh(f1)
            mo[:,idx] = orb.dot(c)
            mo_e[idx] = e
    return mo_e, mo

def init_guess_by_minao(mol):
    '''Generate initial guess density matrix based on ANO basis, then project
    the density matrix to the basis set defined by ``mol``

    Note: this function is inconsistent with the latest PySCF (v2.9) and eariler versions.
    This function returns block diagonal density matrix associated with each atom.
    While the function in PySCF projects the density matrix into the full space of atomic basis

    Returns:
        Density matrix, 2D ndarray
    '''
    from pyscf.scf import atom_hf
    from pyscf.scf import addons

    def minao_basis(symb, nelec_ecp):
        occ = []
        basis_ano = []
        if gto.is_ghost_atom(symb):
            return occ, basis_ano

        stdsymb = gto.mole._std_symbol(symb)
        basis_add = gto.basis.load('ano', stdsymb)
# coreshl defines the core shells to be removed in the initial guess
        coreshl = gto.ecp.core_configuration(nelec_ecp, atom_symbol=stdsymb)
        # coreshl = (0,0,0,0)  # it keeps all core electrons in the initial guess
        for l in range(4):
            ndocc, frac = atom_hf.frac_occ(stdsymb, l)
            if ndocc >= coreshl[l]:
                degen = l * 2 + 1
                occ_l = [2, ]*(ndocc-coreshl[l]) + [frac, ]
                occ.append(np.repeat(occ_l, degen))
                basis_ano.append([l] + [b[:1] + b[1+coreshl[l]:ndocc+2]
                                        for b in basis_add[l][1:]])
            else:
                logger.debug(mol, '*** ECP incorporates partially occupied '
                             'shell of l = %d for atom %s ***', l, symb)
        occ = np.hstack(occ)

        if nelec_ecp > 0:
            if symb in mol._basis:
                input_basis = mol._basis[symb]
            elif stdsymb in mol._basis:
                input_basis = mol._basis[stdsymb]
            else:
                raise KeyError(symb)

            basis4ecp = [[] for i in range(4)]
            for bas in input_basis:
                l = bas[0]
                if l < 4:
                    basis4ecp[l].append(bas)

            occ4ecp = []
            for l in range(4):
                nbas_l = sum((len(bas[1]) - 1) for bas in basis4ecp[l])
                ndocc, frac = atom_hf.frac_occ(stdsymb, l)
                ndocc -= coreshl[l]
                assert ndocc <= nbas_l

                if nbas_l > 0:
                    occ_l = np.zeros(nbas_l)
                    occ_l[:ndocc] = 2
                    if frac > 0:
                        occ_l[ndocc] = frac
                    occ4ecp.append(np.repeat(occ_l, l * 2 + 1))

            occ4ecp = np.hstack(occ4ecp)
            basis4ecp = list(itertools.chain.from_iterable(basis4ecp))

# Compared to ANO valence basis, to check whether the ECP basis set has
# reasonable AO-character contraction.  The ANO valence AO should have
# significant overlap to ECP basis if the ECP basis has AO-character.
            atm1 = gto.Mole()
            atm2 = gto.Mole()
            atom = [[symb, (0.,0.,0.)]]
            atm1._atm, atm1._bas, atm1._env = atm1.make_env(atom, {symb:basis4ecp}, [])
            atm2._atm, atm2._bas, atm2._env = atm2.make_env(atom, {symb:basis_ano}, [])
            atm1._built = True
            atm2._built = True
            s12 = gto.intor_cross('int1e_ovlp', atm1, atm2)
            if abs(np.linalg.det(s12[occ4ecp>0][:,occ>0])) > .1:
                occ, basis_ano = occ4ecp, basis4ecp
            else:
                logger.debug(mol, 'Density of valence part of ANO basis '
                             'will be used as initial guess for %s', symb)
        return occ, basis_ano

    # Issue 548
    if any(gto.charge(mol.atom_symbol(ia)) > 96 for ia in range(mol.natm)):
        from pyscf.scf.hf import init_guess_by_atom
        logger.info(mol, 'MINAO initial guess is not available for super-heavy '
                    'elements. "atom" initial guess is used.')
        return init_guess_by_atom(mol)

    nelec_ecp_dic = {mol.atom_symbol(ia): mol.atom_nelec_core(ia)
                          for ia in range(mol.natm)}

    basis = {}
    occdic = {}
    for symb, nelec_ecp in nelec_ecp_dic.items():
        occ_add, basis_add = minao_basis(symb, nelec_ecp)
        occdic[symb] = occ_add
        basis[symb] = basis_add

    mol1 = gto.Mole()
    mol1._built = True
    mol2 = mol.copy()

    aoslice = mol.aoslice_by_atom()
    nao = aoslice[-1,3]
    dm = cupy.zeros((nao, nao))
    # Preallocate a buffer in cupy memory pool for small arrays held in atm_conf
    workspace = cupy.empty(50**2*12)
    workspace = None # noqa: F841
    atm_conf = {}
    mo_coeff = []
    mo_occ = []
    for ia, (p0, p1) in enumerate(aoslice[:,2:]):
        symb = mol.atom_symbol(ia)
        if gto.is_ghost_atom(symb):
            n = p1 - p0
            mo_coeff.append(cupy.zeros((n, 0)))
            mo_occ.append(cupy.zeros(0))
            continue

        if symb not in atm_conf:
            nelec_ecp = mol.atom_nelec_core(ia)
            occ, basis = minao_basis(symb, nelec_ecp)
            mol1._atm, mol1._bas, mol1._env = mol1.make_env(
                [mol._atom[ia]], {symb: basis}, [])
            i0, i1 = aoslice[ia,:2]
            mol2._bas = mol._bas[i0:i1]
            s22 = mol2.intor_symmetric('int1e_ovlp')
            s21 = gto.mole.intor_cross('int1e_ovlp', mol2, mol1)
            c = pyscf_lib.cho_solve(s22, s21, strict_sym_pos=False)
            c = cupy.asarray(c[:,occ>0], order='C')
            occ = cupy.asarray(occ[occ>0], order='C')
            atm_conf[symb] = occ, c

        occ, c = atm_conf[symb]
        dm[p0:p1,p0:p1] = (c*occ).dot(c.conj().T)
        mo_coeff.append(c)
        mo_occ.append(occ)

    mo_coeff = block_diag(mo_coeff)
    mo_occ = cupy.hstack(mo_occ)
    return tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)

def as_scanner(mf):
    if isinstance(mf, pyscf_lib.SinglePointScanner):
        return mf

    logger.info(mf, 'Create scanner for %s', mf.__class__)
    name = mf.__class__.__name__ + SCF_Scanner.__name_mixin__
    return pyscf_lib.set_class(SCF_Scanner(mf), (SCF_Scanner, mf.__class__), name)

class SCF_Scanner(pyscf_lib.SinglePointScanner):
    def __init__(self, mf_obj):
        self.__dict__.update(mf_obj.__dict__)
        self._last_mol_fp = mf_obj.mol.ao_loc

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        # Cleanup intermediates associated to the previous mol object
        self.reset(mol)

        if 'dm0' in kwargs:
            dm0 = kwargs.pop('dm0')
        elif self.mo_coeff is None:
            dm0 = None
        else:
            dm0 = None
            if np.array_equal(self._last_mol_fp, mol.ao_loc):
                dm0 = self.make_rdm1()
            elif self.chkfile and h5py.is_hdf5(self.chkfile):
                dm0 = self.from_chk(self.chkfile)
        self.mo_coeff = None  # To avoid last mo_coeff being used by SOSCF
        e_tot = self.kernel(dm0=dm0, **kwargs)
        self._last_mol_fp = mol.ao_loc
        return e_tot

class SCF(pyscf_lib.StreamObject):

    # attributes
    conv_tol            = hf_cpu.SCF.conv_tol
    conv_tol_grad       = hf_cpu.SCF.conv_tol_grad
    max_cycle           = hf_cpu.SCF.max_cycle
    init_guess          = hf_cpu.SCF.init_guess
    conv_tol_cpscf      = 1e-6   # TODO: reuse the default value in PySCF

    disp                = None
    DIIS                = diis.SCF_DIIS
    diis                = hf_cpu.SCF.diis
    diis_space          = hf_cpu.SCF.diis_space
    diis_damp           = hf_cpu.SCF.diis_damp
    diis_start_cycle    = hf_cpu.SCF.diis_start_cycle
    diis_file           = hf_cpu.SCF.diis_file
    diis_space_rollback = hf_cpu.SCF.diis_space_rollback
    damp                = hf_cpu.SCF.damp
    level_shift         = hf_cpu.SCF.level_shift
    direct_scf          = hf_cpu.SCF.direct_scf
    direct_scf_tol      = hf_cpu.SCF.direct_scf_tol
    conv_check          = hf_cpu.SCF.conv_check
    callback            = hf_cpu.SCF.callback
    _keys               = hf_cpu.SCF._keys

    # methods
    def __init__(self, mol):
        if not mol._built:
            mol.build()
        self.mol = mol
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory
        self.stdout = mol.stdout

        # The chkfile part is different from pyscf, we turn off chkfile by default.
        self.chkfile = None

##################################################
# don't modify the following attributes, they are not input options
        self.mo_energy = None
        self.mo_coeff = None
        self.mo_occ = None
        self.e_tot = 0
        self.converged = False
        self.scf_summary = {}

        self._opt_gpu = {None: None}
        self._opt_jengine = {None: None}
        self._eri = None # Note: self._eri requires large amount of memory

    __getstate__, __setstate__ = pyscf_lib.generate_pickle_methods(
        excludes=('_opt_gpu', '_eri', '_numint'))

    def check_sanity(self):
        s1e = self.get_ovlp()
        if isinstance(s1e, cupy.ndarray) and s1e.ndim == 2:
            c = cond(s1e, sympos=True)
        else:
            c = cupy.asarray([cond(xi, sympos=True) for xi in s1e])
        logger.debug(self, 'cond(S) = %s', c)
        if cupy.max(c)*1e-17 > self.conv_tol:
            logger.warn(self, 'Singularity detected in overlap matrix (condition number = %4.3g). '
                        'SCF may be inaccurate and hard to converge.', cupy.max(c))
        return super().check_sanity()

    build                    = hf_cpu.SCF.build
    opt                      = NotImplemented
    dump_flags               = hf_cpu.SCF.dump_flags
    get_ovlp                 = return_cupy_array(hf_cpu.SCF.get_ovlp)
    get_fock                 = get_fock
    get_occ                  = get_occ
    get_grad                 = staticmethod(get_grad)
    init_guess_by_atom       = hf_cpu.SCF.init_guess_by_atom
    init_guess_by_huckel     = hf_cpu.SCF.init_guess_by_huckel
    init_guess_by_mod_huckel = hf_cpu.SCF.init_guess_by_mod_huckel
    init_guess_by_1e         = hf_cpu.SCF.init_guess_by_1e
    init_guess_by_chkfile    = hf_cpu.SCF.init_guess_by_chkfile
    from_chk                 = hf_cpu.SCF.from_chk
    get_init_guess           = return_cupy_array(hf_cpu.SCF.get_init_guess)
    make_rdm2                = NotImplemented
    energy_elec              = energy_elec
    energy_tot               = energy_tot
    energy_nuc               = hf_cpu.SCF.energy_nuc
    check_convergence        = None
    _eigh                    = staticmethod(eigh)
    eig                      = hf_cpu.SCF.eig
    do_disp                  = hf_cpu.SCF.do_disp
    get_dispersion           = hf_cpu.SCF.get_dispersion
    kernel = scf             = scf
    as_scanner               = hf_cpu.SCF.as_scanner
    _finalize                = hf_cpu.SCF._finalize
    init_direct_scf          = NotImplemented
    get_jk                   = _get_jk
    get_veff                 = NotImplemented
    mulliken_meta            = hf_cpu.SCF.mulliken_meta
    pop                      = hf_cpu.SCF.pop
    _is_mem_enough           = NotImplemented
    density_fit              = NotImplemented
    newton                   = NotImplemented
    x2c = x2c1e = sfx2c1e    = NotImplemented
    stability                = NotImplemented
    nuc_grad_method          = NotImplemented
    update_                  = NotImplemented
    istype                   = hf_cpu.SCF.istype
    to_rhf                   = NotImplemented
    to_uhf                   = NotImplemented
    to_ghf                   = NotImplemented
    to_rks                   = NotImplemented
    to_uks                   = NotImplemented
    to_gks                   = NotImplemented
    to_ks                    = NotImplemented
    canonicalize             = NotImplemented
    mulliken_pop             = NotImplemented
    mulliken_meta            = NotImplemented

    def init_guess_by_minao(self, mol=None):
        if mol is None: mol = self.mol
        return init_guess_by_minao(mol)

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        return get_hcore(mol)

    def make_rdm1(self, mo_coeff=None, mo_occ=None, **kwargs):
        if mo_occ is None: mo_occ = self.mo_occ
        if mo_coeff is None: mo_coeff = self.mo_coeff
        return make_rdm1(mo_coeff, mo_occ)

    def dip_moment(self, mol=None, dm=None, unit='Debye', origin=None,
                   verbose=logger.NOTE):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        return hf_cpu.dip_moment(mol, dm.get(), unit, origin, verbose)

    def quad_moment(self, mol=None, dm=None, unit='DebyeAngstrom', origin=None,
                    verbose=logger.NOTE):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        return hf_cpu.quad_moment(mol, dm.get(), unit, origin, verbose)

    def remove_soscf(self):
        lib.logger.warn('remove_soscf has no effect in current version')
        return self

    def analyze(self, *args, **kwargs):
        return self.to_cpu().analyze()

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._opt_gpu = {None: None}
        self._opt_jengine = {None: None}
        self.scf_summary = {}
        return self

    def dump_chk(self, envs):
        assert isinstance(envs, dict)
        if self.chkfile:
            chkfile.dump_scf(
                self.mol, self.chkfile, envs['e_tot'],
                cupy.asnumpy(envs['mo_energy']), cupy.asnumpy(envs['mo_coeff']),
                cupy.asnumpy(envs['mo_occ']), overwrite_mol=False)

    def get_j(self, mol, dm, hermi=1, omega=None):
        if omega is None:
            omega = mol.omega
        if omega not in self._opt_jengine:
            jopt = j_engine._VHFOpt(mol, self.direct_scf_tol).build()
            self._opt_jengine[omega] = jopt
        jopt = self._opt_jengine[omega]
        vj = j_engine.get_j(mol, dm, hermi, jopt)
        if not isinstance(dm, cupy.ndarray):
            vj = vj.get()
        return vj

    def get_k(self, mol=None, dm=None, hermi=1, omega=None):
        return self.get_jk(mol, dm, hermi, with_j=False, omega=omega)[1]

class KohnShamDFT:
    '''
    A mock DFT base class, to be compatible with PySCF
    '''

class RHF(SCF):

    to_gpu = utils.to_gpu
    device = utils.device

    _keys = {'e_disp', 'h1e', 's1e', 'e_mf', 'conv_tol_cpscf', 'disp_with_3body'}

    get_veff = get_veff
    canonicalize = canonicalize

    def check_sanity(self):
        mol = self.mol
        if mol.nelectron != 1 and mol.spin != 0:
            logger.warn(self, 'Invalid number of electrons %d for RHF method.',
                        mol.nelectron)
        mem = get_avail_mem()
        nao = mol.nao
        if nao**2*20*8 > mem:
            logger.warn(self, 'GPU memory may be insufficient for SCF of this system. '
                        'It is recommended to use the scf.LRHF or dft.LRKS class for this system.')
        return SCF.check_sanity(self)

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        '''
        electronic energy
        '''
        if dm is None: dm = self.make_rdm1()
        if h1e is None: h1e = self.get_hcore()
        if vhf is None: vhf = self.get_veff(self.mol, dm)
        assert dm.dtype == np.float64
        e1 = float(h1e.ravel().dot(dm.ravel()))
        e_coul = float(vhf.ravel().dot(dm.ravel())) * .5
        self.scf_summary['e1'] = e1
        self.scf_summary['e2'] = e_coul
        logger.debug(self, 'E1 = %s  E_coul = %s', e1, e_coul)
        return e1+e_coul, e_coul

    def nuc_grad_method(self):
        from gpu4pyscf.grad import rhf
        return rhf.Gradients(self)

    def density_fit(self, auxbasis=None, with_df=None, only_dfj=False):
        import gpu4pyscf.df.df_jk
        if self.istype('_Solvation'):
            raise RuntimeError(
                'It is recommended to call density_fit() before applying a solvent model. '
                'Calling density_fit() after the solvent model may result in '
                'incorrect nuclear gradients, TDDFT, and other methods.')
        return gpu4pyscf.df.df_jk.density_fit(self, auxbasis, with_df, only_dfj)

    def newton(self):
        from gpu4pyscf.scf.soscf import newton
        return newton(self)

    def to_cpu(self):
        mf = hf_cpu.RHF(self.mol)
        utils.to_cpu(self, out=mf)
        return mf
