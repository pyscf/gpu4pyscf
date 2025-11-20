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
Hartree-Fock for periodic systems at a single k-point
'''

__all__ = [
    'RHF'
]

import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.pbc.scf import hf as hf_cpu
from pyscf.pbc import tools
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import return_cupy_array, contract
from gpu4pyscf.scf import hf as mol_hf
from gpu4pyscf.pbc import df
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.scf.smearing import smearing

def get_bands(mf, kpts_band, cell=None, dm=None, kpt=None):
    '''Get energy bands at the given (arbitrary) 'band' k-points.

    Returns:
        mo_energy : (nmo,) ndarray or a list of (nmo,) ndarray
            Bands energies E_n(k)
        mo_coeff : (nao, nmo) ndarray or a list of (nao,nmo) ndarray
            Band orbitals psi_n(k)
    '''
    if cell is None: cell = mf.cell
    if dm is None: dm = mf.make_rdm1()
    if kpt is None: kpt = mf.kpt

    kpts_band = np.asarray(kpts_band)
    single_kpt_band = (getattr(kpts_band, 'ndim', None) == 1)
    kpts_band = kpts_band.reshape(-1,3)

    fock = mf.get_veff(cell, dm, kpt=kpt, kpts_band=kpts_band)
    fock += mf.get_hcore(cell, kpts_band)
    s1e = mf.get_ovlp(cell, kpts_band)
    nkpts, nao = fock.shape[:2]
    mo_energy = cp.empty((nkpts, nao))
    mo_coeff = cp.empty((nkpts, nao, nao), dtype=fock.dtype)
    for k in range(nkpts):
        e, c = mf.eig(fock[k], s1e[k])
        mo_energy[k] = e
        mo_coeff[k] = c

    if single_kpt_band:
        mo_energy = mo_energy[0]
        mo_coeff = mo_coeff[0]
    return mo_energy, mo_coeff

damping = mol_hf.damping
level_shift = mol_hf.level_shift
get_fock = mol_hf.get_fock
get_occ = mol_hf.get_occ
get_grad = mol_hf.get_grad
make_rdm1 = mol_hf.make_rdm1
energy_elec = mol_hf.energy_elec

def get_rho(mf, dm=None, grids=None, kpt=None):
    '''Compute density in real space
    '''
    from gpu4pyscf.pbc.dft import gen_grid
    from gpu4pyscf.pbc.dft import numint
    if dm is None:
        dm = mf.make_rdm1()
    if getattr(dm, 'ndim', None) != 2:  # UHF
        dm = dm[0] + dm[1]
    if grids is None:
        grids = gen_grid.UniformGrids(mf.cell)
    if kpt is None:
        kpt = mf.kpt
    ni = numint.NumInt()
    return ni.get_rho(mf.cell, dm, grids, kpt, mf.max_memory)

class SCF(mol_hf.SCF):
    '''SCF base class adapted for PBCs.

    Attributes:
        kpt : (3,) ndarray
            The AO k-point in Cartesian coordinates, in units of 1/Bohr.

        exxdiv : str
            Exchange divergence treatment, can be one of

            | None : ignore G=0 contribution in exchange
            | 'ewald' : Ewald probe charge correction [JCP 122, 234102 (2005); DOI:10.1063/1.1926272]

        with_df : density fitting object
            Default is the instance of FFTDF class (GPW method).
    '''

    # Range separation JK builder
    rsjk = None
    j_engine = None

    _keys = {'cell', 'exxdiv', 'with_df', 'rsjk', 'j_engine', 'kpt'}

    def __init__(self, cell, kpt=None, exxdiv='ewald'):
        mol_hf.SCF.__init__(self, cell)
        self.with_df = df.FFTDF(cell)
        self.exxdiv = exxdiv
        if kpt is not None:
            self.kpt = kpt
        self.conv_tol = max(cell.precision * 10, 1e-8)

    def check_sanity(self):
        if (isinstance(self.exxdiv, str) and self.exxdiv.lower() != 'ewald' and
            isinstance(self.with_df, df.DF)):
            logger.warn(self, 'exxdiv %s is not supported in DF', self.exxdiv)

        mol_hf.SCF.check_sanity(self)
        return self

    @property
    def kpt(self):
        if 'kpt' in self.__dict__:
            # To handle the attribute kpt loaded from chkfile
            self.kpt = self.__dict__.pop('kpt')
        return self.with_df.kpts.reshape(3)
    @kpt.setter
    def kpt(self, x):
        kpts = np.reshape(x, (1, 3))
        if np.any(kpts != 0):
            raise NotImplementedError('single kpt SCF not available')
        self.with_df.kpts = kpts
        if self.rsjk:
            self.rsjk.kpts = kpts

    def reset(self, cell=None):
        '''Reset cell and relevant attributes associated to the old cell object'''
        mol_hf.SCF.reset(self, cell)
        if cell is not None:
            self.cell = cell
        self.with_df.reset(cell)
        if self.rsjk is not None:
            self.rsjk.reset(cell)
        if self.j_engine is not None:
            self.j_engine.reset(cell)
        return self

    def dump_flags(self, verbose=None):
        mol_hf.SCF.dump_flags(self, verbose)
        log = logger.new_logger(self, verbose)
        log.info('******** PBC SCF flags ********')
        log.info('kpt = %s', self.kpt)
        log.info('Exchange divergence treatment (exxdiv) = %s', self.exxdiv)
        cell = self.cell
        if ((cell.dimension >= 2 and cell.low_dim_ft_type != 'inf_vacuum') and
            isinstance(self.exxdiv, str) and self.exxdiv.lower() == 'ewald'):
            madelung = tools.pbc.madelung(cell, self.kpt[None])
            log.info('    madelung (= occupied orbital energy shift) = %s', madelung)
            log.info('    Total energy shift due to Ewald probe charge'
                     ' = -1/2 * Nelec*madelung = %.12g',
                     madelung*cell.nelectron * -.5)
        if getattr(self, 'smearing_method', None) is not None:
            log.info('Smearing method = %s', self.smearing_method)
        log.info('DF object = %s', self.with_df)
        if not getattr(self.with_df, 'build', None):
            # .dump_flags() is called in pbc.df.build function
            self.with_df.dump_flags(verbose)

    def build(self, cell=None):
        # To handle the attribute kpt or kpts loaded from chkfile
        if 'kpt' in self.__dict__:
            self.kpt = self.__dict__.pop('kpt')

        if self.verbose >= logger.WARN:
            self.check_sanity()
        return self

    kpts = hf_cpu.SCF.kpts
    mol = hf_cpu.SCF.mol # required by the hf.kernel

    get_bands = get_bands
    get_rho = get_rho

    def get_ovlp(self, cell=None, kpt=None):
        if kpt is None: kpt = self.kpt
        if cell is None: cell = self.cell
        return int1e.int1e_ovlp(cell, kpt)

    def get_hcore(self, cell=None, kpt=None):
        if kpt is None: kpt = self.kpt
        if cell is None: cell = self.cell
        if cell.pseudo:
            nuc = self.with_df.get_pp(kpt)
        else:
            nuc = self.with_df.get_nuc(kpt)
        if len(cell._ecpbas) > 0:
            raise NotImplementedError('ECP in PBC SCF')
        t = int1e.int1e_kin(cell, kpt)
        return nuc + t

    def get_jk(self, cell=None, dm=None, hermi=1, kpt=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, **kwargs):
        r'''Get Coulomb (J) and exchange (K) following :func:`scf.hf.RHF.get_jk_`.
        for particular k-point (kpt).

        When kpts_band is given, the J, K matrices on kpts_band are evaluated.

            J_{pq} = \sum_{rs} (pq|rs) dm[s,r]
            K_{pq} = \sum_{rs} (pr|sq) dm[r,s]

        where r,s are orbitals on kpt. p and q are orbitals on kpts_band
        if kpts_band is given otherwise p and q are orbitals on kpt.
        '''
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        cpu0 = logger.init_timer(self)
        if kpt is None:
            kpt = self.kpt
        if self.rsjk or self.j_engine:
            vj = vk = None
            if with_j:
                vj = self.get_j(cell, dm, hermi, kpt, kpts_band)
            if with_k:
                vk = self.get_k(cell, dm, hermi, kpt, kpts_band, omega)
        else:
            vj, vk = self.with_df.get_jk(dm, hermi, kpt, kpts_band, with_j,
                                         with_k, omega, exxdiv=self.exxdiv)
        logger.timer(self, 'vj and vk', *cpu0)
        return vj, vk

    def get_j(self, cell, dm, hermi=1, kpt=None, kpts_band=None, omega=None):
        r'''Compute J matrix for the given density matrix and k-point (kpt).
        When kpts_band is given, the J matrices on kpts_band are evaluated.

            J_{pq} = \sum_{rs} (pq|rs) dm[s,r]

        where r,s are orbitals on kpt. p and q are orbitals on kpts_band
        if kpts_band is given otherwise p and q are orbitals on kpt.
        '''
        if kpt is None:
            kpt = self.kpt
        if self.j_engine:
            from gpu4pyscf.pbc.scf.j_engine import get_j
            vj = get_j(cell, dm, hermi, kpt, kpts_band, self.j_engine)
        else:
            vj = self.with_df.get_jk(dm, hermi, kpt, kpts_band, with_k=False)[0]
        return vj

    def get_k(self, cell, dm, hermi=1, kpt=None, kpts_band=None, omega=None):
        '''Compute K matrix for the given density matrix.
        '''
        if kpt is None:
            kpt = self.kpt
        if self.rsjk:
            from gpu4pyscf.pbc.scf.rsjk import get_k
            sr_factor = lr_factor = None
            if omega is not None:
                if omega > 0:
                    sr_factor, lr_factor = 0, 1
                elif omega < 0:
                    omega = -omega
                    sr_factor, lr_factor = 1, 0
            vk = get_k(cell, dm, hermi, kpt, kpts_band, omega, self.rsjk,
                       sr_factor, lr_factor, exxdiv=self.exxdiv)
        else:
            vk = self.with_df.get_jk(dm, hermi, kpt, kpts_band, with_j=False,
                                     omega=omega, exxdiv=self.exxdiv)[1]
        return vk

    def get_veff(self, cell=None, dm=None, dm_last=None, vhf_last=None,
                 hermi=1, kpt=None, kpts_band=None):
        '''Hartree-Fock potential matrix for the given density matrix.
        See :func:`scf.hf.get_veff` and :func:`scf.hf.RHF.get_veff`
        '''
        if dm is None:
            dm = self.make_rdm1()
        vj, vk = self.get_jk(cell, dm, hermi, kpt, kpts_band)
        vhf = vj - vk * .5
        return vhf

    def energy_nuc(self):
        cell = self.cell
        if cell.dimension == 0:
            raise NotImplementedError
        return cell.enuc

    def get_init_guess(self, cell=None, key='minao', s1e=None):
        if cell is None: cell = self.cell
        dm = mol_hf.SCF.get_init_guess(self, cell, key)
        dm = normalize_dm_(self, dm, s1e)
        return dm

    _finalize = hf_cpu.SCF._finalize

    init_guess_by_1e = hf_cpu.SCF.init_guess_by_1e
    init_guess_by_chkfile = hf_cpu.SCF.init_guess_by_chkfile
    from_chk = hf_cpu.SCF.from_chk
    analyze = NotImplemented
    mulliken_pop = NotImplemented
    density_fit = NotImplemented
    rs_density_fit = NotImplemented
    x2c = x2c1e = sfx2c1e = NotImplemented
    spin_square = NotImplemented
    dip_moment = NotImplemented
    Gradients = NotImplemented
    smearing = smearing

    def nuc_grad_method(self):
        return self.Gradients()

    def multigrid_numint(self, mesh=None):
        '''Apply the MultiGrid algorithm for XC numerical integartion'''
        raise NotImplementedError

    def dump_chk(self, envs):
        mol_hf.SCF.dump_chk(self, envs)
        if self.chkfile:
            with lib.H5FileWrap(self.chkfile, 'a') as fh5:
                fh5['scf/kpt'] = self.kpt
        return self

    to_gpu = utils.to_gpu
    device = utils.device
    to_cpu = NotImplemented


class KohnShamDFT:
    '''A mock DFT base class

    The base class is defined in the pbc.dft.rks module. This class can
    be used to verify if an SCF object is an pbc-Hartree-Fock method or an
    pbc-DFT method. It should be overwritten by the actual KohnShamDFT class
    when loading dft module.
    '''


class RHF(SCF):

    energy_elec = mol_hf.RHF.energy_elec

    def density_fit(self, auxbasis=None, with_df=None):
        from gpu4pyscf.pbc.df.df_jk import density_fit
        mf = density_fit(self, auxbasis, with_df)
        mf.with_df.is_gamma_point = (mf.kpt == 0).all()
        return mf

    def Gradients(self):
        from gpu4pyscf.pbc.grad.rhf import Gradients
        return Gradients(self)

    def to_cpu(self):
        mf = hf_cpu.RHF(self.cell)
        utils.to_cpu(self, out=mf)
        return mf

    def analyze(self, verbose=logger.DEBUG, with_meta_lowdin=True, **kwargs):
        '''Analyze the given SCF object:  print orbital energies, occupancies;
        print orbital coefficients; Mulliken population analysis; Diople moment.
        '''
        from pyscf.scf.hf import mulliken_meta, mulliken_pop, MO_BASE
        log = logger.new_logger(self, verbose)
        cell = self.cell
        mo_energy = self.mo_energy.get()
        mo_occ = self.mo_occ.get()

        if log.verbose >= logger.NOTE:
            self.dump_scf_summary(log)
            log.note('**** MO energy ****')
            for i, c in enumerate(mo_occ):
                log.note('MO #%-3d energy= %-18.15g occ= %g', i+MO_BASE, mo_energy[i], c)

        s = self.get_ovlp().get()
        dm = self.make_rdm1().get()
        if with_meta_lowdin:
            pop = mulliken_meta(cell, dm, s=s, verbose=log)
        else:
            pop = mulliken_pop(cell, dm, s=s, verbose=log)
        dip = None
        return pop, dip

def normalize_dm_(mf, dm, s1e=None):
    '''
    Force density matrices integrated to the correct number of electrons.
    '''
    cell = mf.cell
    if s1e is None:
        s1e = mf.get_ovlp(cell)
    ne = contract('ij,ji->', dm, s1e).real
    if abs(ne - cell.nelectron) > 0.01:
        logger.debug(mf, 'Big errors in the electron number of initial guess '
                     'density matrix (Ne/cell = %g)!', ne)
        dm *= cell.nelectron / ne
    return dm
