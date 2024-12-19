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
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import return_cupy_array, contract
from gpu4pyscf.scf import hf as mol_hf
from gpu4pyscf.pbc import df

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

    _keys = hf_cpu.SCF._keys

    def __init__(self, cell, kpt=np.zeros(3), exxdiv='ewald'):
        mol_hf.SCF.__init__(self, cell)
        self.with_df = df.FFTDF(cell)
        # Range separation JK builder
        self.rsjk = None
        self.exxdiv = exxdiv
        self.kpt = kpt
        self.conv_tol = max(cell.precision * 10, 1e-8)

    def check_sanity(self):
        if (isinstance(self.exxdiv, str) and self.exxdiv.lower() != 'ewald' and
            isinstance(self.with_df, df.DF)):
            logger.warn(self, 'exxdiv %s is not supported in DF', self.exxdiv)

        if self.verbose >= logger.DEBUG:
            mol_hf.SCF.check_sanity(self)
        return self

    kpt = hf_cpu.SCF.kpt
    kpts = hf_cpu.SCF.kpts
    mol = hf_cpu.SCF.mol # required by the hf.kernel

    reset = hf_cpu.SCF.reset
    build = hf_cpu.SCF.build
    dump_flags = hf_cpu.SCF.dump_flags

    get_bands = get_bands
    get_rho = get_rho

    get_ovlp = return_cupy_array(hf_cpu.SCF.get_ovlp)

    def get_hcore(self, cell=None, kpt=None):
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt
        if cell.pseudo:
            nuc = self.with_df.get_pp(kpt)
        else:
            nuc = self.with_df.get_nuc(kpt)
        if len(cell._ecpbas) > 0:
            raise NotImplementedError('ECP in PBC SCF')
        return nuc + cp.asarray(cell.pbc_intor('int1e_kin', 1, 1, kpt))

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
        if kpt is None: kpt = self.kpt

        cpu0 = logger.init_timer(self)
        dm = cp.asarray(dm)
        nao = dm.shape[-1]
        vj, vk = self.with_df.get_jk(dm.reshape(-1,nao,nao), hermi, kpt, kpts_band,
                                     with_j, with_k, omega, exxdiv=self.exxdiv)
        if with_j:
            vj = _format_jks(vj, dm, kpts_band)
        if with_k:
            vk = _format_jks(vk, dm, kpts_band)
        logger.timer(self, 'vj and vk', *cpu0)
        return vj, vk

    def get_j(self, cell=None, dm=None, hermi=1, kpt=None, kpts_band=None,
              omega=None):
        r'''Compute J matrix for the given density matrix and k-point (kpt).
        When kpts_band is given, the J matrices on kpts_band are evaluated.

            J_{pq} = \sum_{rs} (pq|rs) dm[s,r]

        where r,s are orbitals on kpt. p and q are orbitals on kpts_band
        if kpts_band is given otherwise p and q are orbitals on kpt.
        '''
        return self.get_jk(cell, dm, hermi, kpt, kpts_band, with_k=False,
                           omega=omega)[0]

    def get_k(self, cell=None, dm=None, hermi=1, kpt=None, kpts_band=None,
              omega=None):
        '''Compute K matrix for the given density matrix.
        '''
        return self.get_jk(cell, dm, hermi, kpt, kpts_band, with_j=False,
                           omega=omega)[1]

    get_veff = hf_cpu.SCF.get_veff
    energy_nuc = hf_cpu.SCF.energy_nuc
    _finalize = hf_cpu.SCF._finalize

    def get_init_guess(self, cell=None, key='minao', s1e=None):
        if cell is None: cell = self.cell
        dm = mol_hf.SCF.get_init_guess(self, cell, key)
        dm = normalize_dm_(self, dm, s1e)
        return dm

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

    def dump_chk(self, envs):
        mol_hf.SCF.dump_chk(self, envs)
        if self.chkfile:
            with lib.H5FileWrap(self.chkfile, 'a') as fh5:
                fh5['scf/kpt'] = self.kpt
        return self


class KohnShamDFT:
    '''A mock DFT base class

    The base class is defined in the pbc.dft.rks module. This class can
    be used to verify if an SCF object is an pbc-Hartree-Fock method or an
    pbc-DFT method. It should be overwritten by the actual KohnShamDFT class
    when loading dft module.
    '''


class RHF(SCF):

    to_gpu = utils.to_gpu
    device = utils.device

    def to_cpu(self):
        mf = hf_cpu.RHF(self.cell)
        utils.to_cpu(self, out=mf)
        return mf

def _format_jks(vj, dm, kpts_band):
    if kpts_band is None:
        vj = vj.reshape(dm.shape)
    elif kpts_band.ndim == 1:  # a single k-point on bands
        vj = vj.reshape(dm.shape)
    elif getattr(dm, "ndim", 0) == 2:
        vj = vj[0]
    return vj

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
