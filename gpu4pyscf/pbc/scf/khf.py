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

'''
Hartree-Fock for periodic systems with k-point sampling
'''

__all__ = [
    'KRHF'
]

import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.pbc.scf import khf as khf_cpu
from pyscf.pbc import tools
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import (
    return_cupy_array, contract, tag_array, sandwich_dot, eigh)
from gpu4pyscf.scf import hf as mol_hf
from gpu4pyscf.pbc.scf import hf as pbchf
from gpu4pyscf.pbc import df
from gpu4pyscf.pbc.gto import int1e

def get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
             fock_last=None):
    h1e_kpts, s_kpts, vhf_kpts, dm_kpts = h1e, s1e, vhf, dm
    if h1e_kpts is None: h1e_kpts = mf.get_hcore()
    if vhf_kpts is None: vhf_kpts = mf.get_veff(mf.cell, dm_kpts)
    f_kpts = h1e_kpts + vhf_kpts
    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f_kpts

    if s_kpts is None: s_kpts = mf.get_ovlp()
    if dm_kpts is None: dm_kpts = mf.make_rdm1()

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if damp_factor is None:
        damp_factor = mf.damp
    if damp_factor is not None and 0 <= cycle < diis_start_cycle-1 and fock_last is not None:
        f_kpts = cp.asarray([pbchf.damping(f, f_prev, damp_factor)
                             for f,f_prev in zip(f_kpts,fock_last)])
    if diis and cycle >= diis_start_cycle:
        f_kpts = diis.update(s_kpts, dm_kpts, f_kpts, mf, h1e_kpts, vhf_kpts, f_prev=fock_last)

    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if level_shift_factor is not None:
        f_kpts = [pbchf.level_shift(s, dm_kpts[k], f_kpts[k], level_shift_factor)
                  for k, s in enumerate(s_kpts)]
    return cp.asarray(f_kpts)

def get_fermi(mf, mo_energy_kpts=None, mo_occ_kpts=None):
    '''Fermi level
    '''
    if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy
    if mo_occ_kpts is None: mo_occ_kpts = mf.mo_occ
    assert isinstance(mo_energy_kpts, cp.ndarray) and mo_energy_kpts.ndim == 3
    assert isinstance(mo_occ_kpts, cp.ndarray) and mo_occ_kpts.ndim == 3

    # mo_energy_kpts and mo_occ_kpts are k-point RHF quantities
    assert (mo_energy_kpts[0].ndim == 1)
    assert (mo_occ_kpts[0].ndim == 1)

    nocc = mo_occ_kpts.sum() / 2
    # nocc may not be perfect integer when smearing is enabled
    nocc = int(nocc.round(3))
    fermi = cp.partition(mo_energy_kpts.ravel(), nocc-1)[nocc-1]

    if mf.verbose >= logger.DEBUG:
        for k, mo_e in enumerate(mo_energy_kpts):
            mo_occ = mo_occ_kpts[k]
            if mo_occ[mo_e > fermi].sum() > 1.:
                logger.warn(mf, 'Occupied band above Fermi level: \n'
                            'k=%d, mo_e=%s, mo_occ=%s', k, mo_e, mo_occ)
    return fermi

def get_occ(mf, mo_energy_kpts=None, mo_coeff_kpts=None):
    '''Label the occupancies for each orbital for sampled k-points.

    This is a k-point version of scf.hf.SCF.get_occ
    '''
    if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy

    nkpts = len(mo_energy_kpts)
    nocc = mf.cell.tot_electrons(nkpts) // 2

    if isinstance(mo_energy_kpts, cp.ndarray):
        mo_energy = cp.sort(mo_energy_kpts.ravel())
        fermi = mo_energy[nocc-1]
        mo_occ_kpts = (mo_energy_kpts <= fermi).astype(np.float64) * 2
    else:
        mo_energy = cp.sort(cp.hstack(mo_energy_kpts))
        fermi = mo_energy[nocc-1]
        mo_occ_kpts = []
        for mo_e in mo_energy_kpts:
            mo_occ_kpts.append((mo_e <= fermi).astype(np.float64) * 2)

    if mf.verbose >= logger.DEBUG:
        if nocc < mo_energy.size:
            logger.info(mf, 'HOMO = %.12g  LUMO = %.12g',
                        mo_energy[nocc-1], mo_energy[nocc])
            if mo_energy[nocc-1]+1e-3 > mo_energy[nocc]:
                logger.warn(mf, 'HOMO %.12g == LUMO %.12g',
                            mo_energy[nocc-1], mo_energy[nocc])
        else:
            logger.info(mf, 'HOMO = %.12g', mo_energy[nocc-1])
    return mo_occ_kpts

def get_grad(mo_coeff_kpts, mo_occ_kpts, fock):
    '''
    returns 1D array of gradients, like non K-pt version
    note that occ and virt indices of different k pts now occur
    in sequential patches of the 1D array
    '''
    grad_kpts = [pbchf.get_grad(c, o, f).ravel()
                 for c, o, f in zip(mo_coeff_kpts, mo_occ_kpts, fock)]
    return cp.hstack(grad_kpts)

def make_rdm1(mo_coeff_kpts, mo_occ_kpts, **kwargs):
    '''One particle density matrices for all k-points.

    Returns:
        dm_kpts : (nkpts, nao, nao) ndarray
    '''
    if isinstance(mo_occ_kpts, cp.ndarray):
        c = mo_coeff_kpts * mo_occ_kpts[:,None,:]
        dm = contract('kpi,kqi->kpq', mo_coeff_kpts, c.conj())
    else:
        nao = mo_coeff_kpts[0].shape[0]
        nkpts = len(mo_coeff_kpts)
        dtype = np.result_type(*mo_coeff_kpts)
        dm = cp.empty((nkpts, nao, nao), dtype=dtype)
        for k in range(nkpts):
            dm[k] = pbchf.make_rdm1(mo_coeff_kpts[k], mo_occ_kpts[k])
    return tag_array(dm, mo_coeff=mo_coeff_kpts, mo_occ=mo_occ_kpts)

def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
    '''Following pyscf.scf.hf.energy_elec()
    '''
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if h1e_kpts is None: h1e_kpts = mf.get_hcore()
    if vhf_kpts is None: vhf_kpts = mf.get_veff(mf.cell, dm_kpts)

    nkpts = len(dm_kpts)
    e1 = 1./nkpts * cp.einsum('kij,kji->', dm_kpts, h1e_kpts).get()
    e_coul = 1./nkpts * cp.einsum('kij,kji->', dm_kpts, vhf_kpts).get() * 0.5
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['e2'] = e_coul.real
    logger.debug(mf, 'E1 = %s  E_coul = %s', e1, e_coul)
    if abs(e_coul.imag) > mf.cell.precision*10:
        logger.warn(mf, "Coulomb energy has imaginary part %s. "
                    "Coulomb integrals (e-e, e-N) may not converge !",
                    e_coul.imag)
    return (e1+e_coul).real, e_coul.real

def canonicalize(mf, mo_coeff_kpts, mo_occ_kpts, fock=None):
    if hasattr(mf, 'overlap_canonical_decomposed_x') and mf.overlap_canonical_decomposed_x is not None:
        raise NotImplementedError("Overlap matrix canonical decomposition (removing linear dependency for diffused orbitals) "
                                  "not supported for canonicalize() function with k-point sampling")
    if fock is None:
        dm = mf.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
        fock = mf.get_fock(dm=dm)
    fock = sandwich_dot(fock, mo_coeff_kpts)
    occidx = mo_occ_kpts == 2
    viridx = ~occidx
    mo_coeff = cp.empty_like(mo_coeff_kpts)
    mo_energy = cp.empty(mo_occ_kpts.shape, dtype=np.float64)
    nkpts = len(mo_coeff_kpts)
    for k in range(nkpts):
        for idx in (occidx, viridx):
            if cp.count_nonzero(idx) > 0:
                e, c = cp.linalg.eigh(fock[k,idx[:,None],idx])
                mo_coeff[k,:,idx] = mo_coeff_kpts[k,:,idx].dot(c)
                mo_energy[k,idx] = e
    return mo_energy, mo_coeff

def _cast_mol_init_guess(fn):
    def fn_init_guess(mf, cell=None, kpts=None):
        if cell is None: cell = mf.cell
        if kpts is None: kpts = mf.kpts
        dm = fn(mf, cell)
        assert dm.ndim == 2
        nkpts = len(kpts)
        dm = cp.repeat(dm[None], nkpts, axis=0)
        if hasattr(dm, 'mo_coeff'):
            mo_coeff = cp.repeat(dm.mo_coeff[None], nkpts, axis=0)
            mo_occ = cp.repeat(dm.mo_occ[None], nkpts, axis=0)
            dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
        return dm
    fn_init_guess.__name__ = fn.__name__
    fn_init_guess.__doc__ = fn.__doc__
    return fn_init_guess

def get_rho(mf, dm=None, grids=None, kpts=None):
    '''Compute density in real space
    '''
    from gpu4pyscf.pbc.dft import gen_grid
    from gpu4pyscf.pbc.dft import numint
    if dm is None:
        dm = mf.make_rdm1()
    if getattr(dm[0], 'ndim', None) != 2:  # KUHF
        dm = dm[0] + dm[1]
    if grids is None:
        grids = gen_grid.UniformGrids(mf.cell)
    if kpts is None:
        kpts = mf.kpts
    ni = numint.KNumInt()
    return ni.get_rho(mf.cell, dm, grids, kpts, mf.max_memory)


class KSCF(pbchf.SCF):
    '''SCF base class with k-point sampling.

    Compared to molecular SCF, some members such as mo_coeff, mo_occ
    now have an additional first dimension for the k-points,
    e.g. mo_coeff is (nkpts, nao, nao) ndarray

    Attributes:
        kpts : (nks,3) ndarray
            The sampling k-points in Cartesian coordinates, in units of 1/Bohr.
    '''
    conv_tol_grad = khf_cpu.KSCF.conv_tol_grad

    # Range separation JK builder
    rsjk = None
    j_engine = None

    _keys = {'cell', 'exx_built', 'exxdiv', 'with_df', 'rsjk', 'j_engine', 'kpts'}

    def __init__(self, cell, kpts=None, exxdiv='ewald'):
        mol_hf.SCF.__init__(self, cell)
        self.with_df = df.FFTDF(cell)
        self.exxdiv = exxdiv
        if kpts is not None:
            self.kpts = kpts
        self.conv_tol = max(cell.precision * 10, 1e-8)
        self.exx_built = False

    def dump_flags(self, verbose=None):
        mol_hf.SCF.dump_flags(self, verbose)
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** PBC SCF flags ********')
        log.info('N kpts = %d', len(self.kpts))
        log.debug('kpts = %s', self.kpts)
        log.info('Exchange divergence treatment (exxdiv) = %s', self.exxdiv)
        cell = self.cell
        if ((cell.dimension >= 2 and cell.low_dim_ft_type != 'inf_vacuum') and
            isinstance(self.exxdiv, str) and self.exxdiv.lower() == 'ewald'):
            madelung = tools.pbc.madelung(cell, self.kpts)
            log.info('    madelung (= occupied orbital energy shift) = %s', madelung)
            nkpts = len(self.kpts)
            # FIXME: consider the fractional num_electron or not? This maybe
            # relates to the charged system.
            nelectron = float(self.cell.tot_electrons(nkpts)) / nkpts
            log.info('    Total energy shift due to Ewald probe charge'
                     ' = -1/2 * Nelec*madelung = %.12g',
                     madelung*nelectron * -.5)
        if getattr(self, 'smearing_method', None) is not None:
            log.info('Smearing method = %s', self.smearing_method)
        log.info('DF object = %s', self.with_df)
        if not getattr(self.with_df, 'build', None):
            # .dump_flags() is called in pbc.df.build function
            self.with_df.dump_flags(verbose)
        return self

    kpts = khf_cpu.KSCF.kpts
    mol = pbchf.SCF.mol
    mo_energy_kpts = khf_cpu.KSCF.mo_energy_kpts
    mo_coeff_kpts = khf_cpu.KSCF.mo_coeff_kpts
    mo_occ_kpts = khf_cpu.KSCF.mo_occ_kpts

    check_sanity = pbchf.SCF.check_sanity
    reset = pbchf.SCF.reset

    def build(self, cell=None):
        # To handle the attribute kpt or kpts loaded from chkfile
        if 'kpts' in self.__dict__:
            self.kpts = self.__dict__.pop('kpts')

        kpts = self.kpts
        with_df = self.with_df
        if len(kpts) > 1 and getattr(with_df, '_j_only', False):
            logger.warn(self, 'df.j_only cannot be used with k-point HF')
            with_df._j_only = False
            with_df.reset()

        if self.verbose >= logger.WARN:
            self.check_sanity()
        return self

    def get_ovlp(self, cell=None, kpts=None):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        return int1e.int1e_ovlp(cell, kpts)

    def get_hcore(self, cell=None, kpts=None):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        if cell.pseudo:
            nuc = self.with_df.get_pp(kpts)
        else:
            nuc = self.with_df.get_nuc(kpts)
        if len(cell._ecpbas) > 0:
            raise NotImplementedError('ECP in PBC SCF')
        t = int1e.int1e_kin(cell, kpts)
        return nuc + t

    def get_j(self, cell, dm_kpts, hermi=1, kpts=None, kpts_band=None,
              omega=None):
        if self.j_engine:
            from gpu4pyscf.pbc.scf.j_engine import get_j
            vj = get_j(cell, dm_kpts, hermi, kpts, kpts_band, self.j_engine)
        else:
            vj = self.with_df.get_jk(dm_kpts, hermi, kpts, kpts_band, with_k=False)[0]
        return vj

    def get_k(self, cell, dm_kpts, hermi=1, kpts=None, kpts_band=None,
              omega=None):
        if self.rsjk:
            from gpu4pyscf.pbc.scf.rsjk import get_k
            sr_factor = lr_factor = None
            if omega is not None:
                if omega > 0:
                    sr_factor, lr_factor = 0, 1
                elif omega < 0:
                    omega = -omega
                    sr_factor, lr_factor = 1, 0
            vk = get_k(cell, dm_kpts, hermi, kpts, kpts_band, omega, self.rsjk,
                       sr_factor, lr_factor, exxdiv=self.exxdiv)
        else:
            vk = self.with_df.get_jk(dm_kpts, hermi, kpts, kpts_band, with_j=False,
                                     omega=omega, exxdiv=self.exxdiv)[1]
        return vk

    def get_jk(self, cell=None, dm_kpts=None, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, **kwargs):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        cpu0 = logger.init_timer(self)
        if self.rsjk or self.j_engine:
            vj = vk = None
            if with_j:
                vj = self.get_j(cell, dm_kpts, hermi, kpts, kpts_band)
            if with_k:
                vk = self.get_k(cell, dm_kpts, hermi, kpts, kpts_band, omega)
        else:
            vj, vk = self.with_df.get_jk(
                dm_kpts, hermi, kpts, kpts_band, with_j, with_k,
                omega=omega, exxdiv=self.exxdiv)
        logger.timer(self, 'vj and vk', *cpu0)
        return vj, vk

    def get_veff(self, cell=None, dm_kpts=None, dm_last=None, vhf_last=None,
                 hermi=1, kpts=None, kpts_band=None):
        '''Hartree-Fock potential matrix for the given density matrix.
        See :func:`scf.hf.get_veff` and :func:`scf.hf.RHF.get_veff`
        '''
        if dm_kpts is None:
            dm_kpts = self.make_rdm1()
        vj, vk = self.get_jk(cell, dm_kpts, hermi, kpts, kpts_band)
        vhf = vj - vk * .5
        return vhf

    def get_grad(self, mo_coeff_kpts, mo_occ_kpts, fock=None):
        '''
        returns 1D array of gradients, like non K-pt version
        note that occ and virt indices of different k pts now occur
        in sequential patches of the 1D array
        '''
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
            fock = self.get_hcore(self.cell, self.kpts) + self.get_veff(self.cell, dm1)
        return get_grad(mo_coeff_kpts, mo_occ_kpts, fock)

    def eig(self, h_kpts, s_kpts, overwrite=False):
        nkpts, nao = h_kpts.shape[:2]
        eig_kpts = cp.empty((nkpts, nao))
        mo_coeff_kpts = cp.empty((nkpts, nao, nao), dtype=h_kpts.dtype)

        x_kpts = None
        if hasattr(self, 'overlap_canonical_decomposed_x') and self.overlap_canonical_decomposed_x is not None:
            x_kpts = [cp.asarray(x) for x in self.overlap_canonical_decomposed_x]

        if x_kpts is None:
            for k in range(nkpts):
                e, c = eigh(h_kpts[k], s_kpts[k], overwrite)
                eig_kpts[k] = e
                mo_coeff_kpts[k] = c
        else:
            for k in range(nkpts):
                xk = x_kpts[k]
                ek, ck = cp.linalg.eigh(xk.T.conj() @ h_kpts[k] @ xk)
                ck = xk @ ck
                _, nmo_k = xk.shape
                eig_kpts[k, :nmo_k] = ek
                eig_kpts[k, nmo_k:] = float(cp.max(cp.abs(ek))) * 2 + 1e5
                mo_coeff_kpts[k, :, :nmo_k] = ck
                mo_coeff_kpts[k, :, nmo_k:] = 0
        return eig_kpts, mo_coeff_kpts

    def make_rdm1(self, mo_coeff_kpts=None, mo_occ_kpts=None, **kwargs):
        if mo_coeff_kpts is None:
            mo_coeff_kpts = self.mo_coeff
        if mo_occ_kpts is None:
            mo_occ_kpts = self.mo_occ
        return make_rdm1(mo_coeff_kpts, mo_occ_kpts, **kwargs)

    make_rdm2 = NotImplemented

    init_direct_scf = NotImplemented
    get_fock = get_fock
    get_fermi = get_fermi
    get_occ = get_occ
    energy_elec = energy_elec
    energy_nuc = pbchf.SCF.energy_nuc
    get_rho = get_rho
    _finalize = pbchf.SCF._finalize
    canonicalize = canonicalize

    get_bands = khf_cpu.KSCF.get_bands

    get_init_guess = NotImplemented
    init_guess_by_minao = _cast_mol_init_guess(pbchf.SCF.init_guess_by_minao)
    init_guess_by_atom = _cast_mol_init_guess(pbchf.SCF.init_guess_by_atom)
    init_guess_by_1e = return_cupy_array(khf_cpu.KSCF.init_guess_by_1e)
    init_guess_by_chkfile = return_cupy_array(khf_cpu.KSCF.init_guess_by_chkfile)
    from_chk = return_cupy_array(khf_cpu.KSCF.from_chk)

    analyze = NotImplemented
    mulliken_pop = NotImplemented
    mulliken_meta = NotImplemented
    density_fit = NotImplemented
    rs_density_fit = NotImplemented
    newton = NotImplemented
    x2c = x2c1e = sfx2c1e = NotImplemented
    spin_square = NotImplemented
    dip_moment = NotImplemented
    stability = NotImplemented
    to_rhf = NotImplemented
    to_uhf = NotImplemented
    to_ghf = NotImplemented
    to_kscf = NotImplemented
    to_khf = NotImplemented
    to_ks = NotImplemented
    convert_from_ = NotImplemented

    smearing = pbchf.SCF.smearing

    def dump_chk(self, envs):
        mol_hf.SCF.dump_chk(self, envs)
        if self.chkfile:
            with lib.H5FileWrap(self.chkfile, 'a') as fh5:
                fh5['scf/kpts'] = cp.asnumpy(self.kpts)
        return self

class KRHF(KSCF):

    check_sanity = pbchf.SCF.check_sanity

    def get_init_guess(self, cell=None, key='minao', s1e=None):
        kpts = self.kpts
        if s1e is None:
            s1e = self.get_ovlp(cell, kpts)
        dm = mol_hf.SCF.get_init_guess(self, cell, key)
        nkpts = len(kpts)
        if dm.ndim == 2:
            # dm[nao,nao] at gamma point -> dm_kpts[nkpts,nao,nao]
            dm = cp.repeat(dm[None,:,:], nkpts, axis=0)
        dm_kpts = dm

        ne = cp.einsum('kij,kji->', dm_kpts, s1e).real
        # FIXME: consider the fractional num_electron or not? This maybe
        # relate to the charged system.
        nelectron = float(self.cell.tot_electrons(nkpts))
        if abs(ne - nelectron) > 0.01*nkpts:
            logger.debug(self, 'Big error detected in the electron number '
                         'of initial guess density matrix (Ne/cell = %g)!\n'
                         '  This can cause huge error in Fock matrix and '
                         'lead to instability in SCF for low-dimensional '
                         'systems.\n  DM is normalized wrt the number '
                         'of electrons %s', ne/nkpts, nelectron/nkpts)
            dm_kpts *= (nelectron / ne).reshape(-1,1,1)
        return dm_kpts

    def density_fit(self, auxbasis=None, with_df=None):
        from gpu4pyscf.pbc.df.df_jk import density_fit
        return density_fit(self, auxbasis, with_df)

    def Gradients(self):
        from gpu4pyscf.pbc.grad.krhf import Gradients
        return Gradients(self)

    def to_cpu(self):
        mf = khf_cpu.KRHF(self.cell)
        utils.to_cpu(self, out=mf)
        return mf

    def analyze(self, verbose=None, **kwargs):
        '''Analyze the given SCF object:  print orbital energies, occupancies;
        print orbital coefficients; Mulliken population analysis; Dipole moment
        '''
        from pyscf.pbc.scf.khf import mulliken_meta
        if verbose is None:
            verbose = self.verbose
        log = logger.new_logger(self, verbose)
        mo_energy = self.mo_energy.get()
        mo_occ = self.mo_occ.get()
        cell = self.cell
        kpts = self.kpts
        if log.verbose >= logger.NOTE:
            self.dump_scf_summary(log)
            log.note('**** MO energy ****')
            log.note('k-point                    nocc    HOMO/AU         LUMO/AU')
            for k, kpt in enumerate(cell.get_scaled_kpts(kpts)):
                nocc = np.count_nonzero(mo_occ[k])
                homo = mo_energy[k,nocc-1]
                lumo = mo_energy[k,nocc  ]
                log.note('%2d (%6.3f %6.3f %6.3f) %2d   %15.9f %15.9f',
                         k, kpt[0], kpt[1], kpt[2], nocc, homo, lumo)

        log.note('**** Population analysis for atoms in the reference cell ****')
        s = self.get_ovlp(kpts=kpts).get()
        dm = self.make_rdm1().get()
        pop, chg = mulliken_meta(cell, dm, kpts=kpts, s=s, verbose=verbose)
        dip = None
        return (pop, chg), dip
