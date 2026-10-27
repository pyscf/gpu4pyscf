# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
#
# Author: Tianyu Zhu <zhutianyu1991@gmail.com>
# Author: Christopher Hillenbrand <chillenbrand15@gmail.com>
# Author: Chaoqun Zhang <cq_zhang@outlook.com>
# Author: Jincheng Yu <pimetamon@gmail.com>
# Author: Jiachen Li <lijiachen.duke@gmail.com>
#

"""
Spin-restricted random phase approximation (direct RPA) with N^4 scaling.

References:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    New J. Phys. 14, 053020 (2012)
"""

import cupy as cp
import numpy as np
import time

from pyscf import lib as lib_cpu
from pyscf.mp.mp2 import get_frozen_mask

from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots

from gpu4pyscf.df import DF, make_auxbasis
from gpu4pyscf.mp.mp2 import get_nocc, get_nmo
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import hermi_triu
from gpu4pyscf.lib.cutensor import contract
from gpu4pyscf.gw.gw_ac import _mo_without_core, _mo_energy_without_core, get_rho_response, get_idx_metal

# fuse functions for GPU
@cp.fuse()
def _Pia(Lia, eia, omega):
    return Lia * (4.0 * eia / (omega * omega + eia * eia))


@cp.fuse()
def _Xia(Lia, eia, omega):
    # Pi = -X X^T with this X; needs eia <= 0 for every pair (occupied orbitals below virtual ones)
    return Lia * cp.sqrt(-4.0 * eia / (omega * omega + eia * eia))


@cp.fuse()
def _Pia_fia(Lia, eia, fia, omega):
    return Lia * (4.0 * eia * fia / (omega * omega + eia * eia))


@cp.fuse()
def _Xia_fia(Lia, eia, fia, omega):
    # Pi = -X X^T with this X; needs eia * fia <= 0 for every pair (occupations non-increasing with orbital energy)
    return Lia * cp.sqrt(-4.0 * eia * fia / (omega * omega + eia * eia))


@cp.fuse()
def _Pia_frac(Lia, eia, fia, omega):
    # factor 2.0 instead of 4.0: double counting between fractional orbitals
    return Lia * (2.0 * eia * fia / (omega * omega + eia * eia))


@cp.fuse()
def _Xia_frac(Lia, eia, fia, omega):
    # X X^T = -(Pia Lia^T) of _Pia_frac; needs eia * fia <= 0 for every pair
    return Lia * cp.sqrt(-2.0 * eia * fia / (omega * omega + eia * eia))


def kernel(rpa, mo_energy, mo_coeff, Lpq=None):
    """RPA correlation and total energy.

    Parameters
    ----------
    rpa : RPA
        rpa object
    mo_energy : double 1d array
        molecular orbital energies
    mo_coeff : double 2d array
        molecular orbital coefficients
    Lpq : double 3d array, optional
        density fitting 3-center integral in MO basis, by default None

    Returns
    -------
    e_tot : float
        RPA total energy
    e_hf : float
        HF energy (exact exchange for given mo_coeff)
    e_corr : float
        RPA correlation energy
    """
    # only support frozen core
    if rpa.frozen is not None:
        assert isinstance(rpa.frozen, int)

    if Lpq is None:
        rpa.initialize_df(auxbasis=rpa.auxbasis)
        if rpa.outcore is False:
            with lib_cpu.temporary_env(rpa.with_df, verbose=0), lib_cpu.temporary_env(rpa.mol, verbose=0):
                Lpq = rpa.ao2mo()

    # Grids for integration on imaginary axis
    freqs, wts = rpa.get_grids(nw=rpa.nw, mo_energy=mo_energy)

    # Compute HF exchange energy (EXX)
    e_hf = rpa.get_ehf()

    # Compute RPA correlation energy
    if rpa.outcore:
        e_corr = get_rpa_ecorr_outcore(rpa, mo_coeff, freqs, wts)
    else:
        e_corr = get_rpa_ecorr(rpa, Lpq, freqs, wts)

    # Compute total energy
    e_hf = float(e_hf)
    e_corr = float(e_corr)
    e_tot = e_hf + e_corr

    logger.debug(rpa, '  RPA total energy = %s', e_tot)
    logger.debug(rpa, '  EXX energy = %s, RPA corr energy = %s', e_hf, e_corr)

    return e_tot, e_hf, e_corr


def get_rpa_ecorr(rpa, Lpq, freqs, wts):
    """Compute RPA correlation energy.

    Parameters
    ----------
    rpa : RPA
        rpa object
    Lpq : double 3d array
        density fitting 3-center integral in MO basis
    freqs : double 1d array
        frequency grid
    wts : double 1d array
        weight of grids

    Returns
    -------
    e_corr : double
        RPA correlation energy
    """
    mo_energy = _mo_energy_without_core(rpa, rpa.mo_energy)
    mo_occ = _mo_energy_without_core(rpa, rpa.mo_occ)
    nw = len(freqs)
    naux = Lpq.shape[0]

    is_metal = hasattr(rpa._scf, 'sigma')

    e_corr = 0.0
    Pi = None
    for w in range(nw):
        if is_metal:
            Pi = get_rho_response_metal(freqs[w], mo_energy, Lpq, mo_occ, out=Pi)
        else:
            Pi = get_rho_response(freqs[w], mo_energy, Lpq, out=Pi)
        ec_w = cp.linalg.slogdet((cp.eye(naux) - Pi))[1]
        ec_w += cp.trace(Pi)
        e_corr += 1.0 / (2.0 * cp.pi) * ec_w * wts[w]

    return e_corr


def get_rpa_ecorr_outcore(rpa, mo_coeff, freqs, wts):
    """Low-memory routine to compute RPA correlation energy.

    Parameters
    ----------
    rpa : RPA
        rpa object
    mo_coeff : double 2d array
        orbital coefficient
    freqs : double 1d array
        frequency grid
    wts : double 1d array
        weight of grids

    Returns
    -------
    e_corr : double
        RPA correlation energy
    """
    mo_energy = _mo_energy_without_core(rpa, rpa.mo_energy)
    mo_occ = _mo_energy_without_core(rpa, rpa.mo_occ)
    nocc = rpa.nocc
    nw = len(freqs)
    nmo = rpa.nmo
    naux = rpa._naux

    is_metal = False
    mo_occ_1d = cp.array(mo_occ).reshape(-1)
    if cp.linalg.norm(cp.abs(mo_occ_1d - 1.0) - 1.0) > 1e-5:
        is_metal = True

    e_corr = 0.0
    Pi = cp.zeros(shape=[nw, naux, naux], dtype=cp.double)
    if is_metal:
        idx_occ, idx_frac, idx_vir = get_idx_metal(mo_occ_1d)

        # Pi from frac to all orbs
        orb_start = idx_frac[0]
        orb_end = idx_frac[-1] + 1
        ijslice = (orb_start, orb_end, 0, nmo)
        Lpq = rpa.ao2mo(ijslice=ijslice)
        for w in range(nw):
            eia = mo_energy[idx_frac][:, None] - mo_energy[None, :]
            fia = (mo_occ[idx_frac][:, None] - mo_occ[None, :]) / 2.0
            # plain version
            # eia = 4.0 * eia * fia / (freqs[w]**2 + eia**2)
            # Pia = Lpq * eia
            # fused version
            # Pia = _Pia_fia(Lpq, eia, fia, freqs[w])
            # This line computes Pi[:,:,w] += einsum('Pia, Qia -> PQ', Pia, Lpq)
            # Pi[w] += Pia.reshape(naux, -1) @ Lpq.reshape(naux, -1).T
            # SYRK version: Pi[w] -= X X^T, accumulated in the lower triangle
            Xia = _Xia_fia(Lpq, eia, fia, freqs[w])
            cp.cublas.syrk('N', Xia.reshape(naux, -1), out=Pi[w], alpha=-1.0, beta=1.0, lower=True)
            Xia = None

        # Pi double counting from frac
        orb_start = idx_frac[0]
        orb_end = idx_frac[-1] + 1
        ijslice = (orb_start, orb_end, orb_start, orb_end)
        Lpq = rpa.ao2mo(ijslice=ijslice)
        for w in range(nw):
            eia = mo_energy[idx_frac][:, None] - mo_energy[idx_frac][None, :]
            fia = (mo_occ[idx_frac][:, None] - mo_occ[idx_frac][None, :]) / 2.0
            # plain version
            # eia = 2.0 * eia * fia / (freqs[w]**2 + eia**2)
            # Pia = Lpq * eia
            # fused version
            # Pia = _Pia_frac(Lpq, eia, fia, freqs[w])
            # This line computes Pi[:,:,w] -= einsum('Pia, Qia -> PQ', Pia, Lpq)
            # Pi[w] -= Pia.reshape(naux, -1) @ Lpq.reshape(naux, -1).T
            # SYRK version: Pi[w] += X X^T (the subtracted term is -X X^T), accumulated in the lower triangle
            Xia = _Xia_frac(Lpq, eia, fia, freqs[w])
            cp.cublas.syrk('N', Xia.reshape(naux, -1), out=Pi[w], alpha=1.0, beta=1.0, lower=True)
            Xia = None

        # Pi from occ to vir orbs
        nocc_metal = len(idx_occ)
        nseg = nocc_metal // rpa.segsize + 1
        for i in range(nseg):
            orb_start = i * rpa.segsize
            orb_end = min((i + 1) * rpa.segsize, nocc_metal)
            if orb_end == orb_start:
                continue
            ijslice = (orb_start, orb_end, idx_vir[0], nmo)
            Lpq = rpa.ao2mo(ijslice=ijslice)
            for w in range(nw):
                eia = mo_energy[orb_start:orb_end, None] - mo_energy[None, idx_vir[0] :]
                # plain version
                # eia = 4.0 * eia / (freqs[w]**2 + eia**2)
                # Pia = Lpq * eia
                # fused version
                # Pia = _Pia(Lpq, eia, freqs[w])
                # This line computes Pi[:,:,w] += einsum('Pia, Qia -> PQ', Pia, Lpq)
                # Pi[w] += Pia.reshape(naux, -1) @ Lpq.reshape(naux, -1).T
                # SYRK version: Pi[w] -= X X^T, accumulated in the lower triangle
                Xia = _Xia(Lpq, eia, freqs[w])
                cp.cublas.syrk('N', Xia.reshape(naux, -1), out=Pi[w], alpha=-1.0, beta=1.0, lower=True)
                Xia = None
    else:
        nseg = nocc // rpa.segsize + 1
        for i in range(nseg):
            orb_start = i * rpa.segsize
            orb_end = min((i + 1) * rpa.segsize, nocc)
            if orb_end == orb_start:
                continue
            ijslice = (orb_start, orb_end, nocc, nmo)
            Lpq = rpa.ao2mo(ijslice=ijslice)
            for w in range(nw):
                eia = mo_energy[orb_start:orb_end, None] - mo_energy[None, nocc:]
                fia = (mo_occ[orb_start:orb_end, None] - mo_occ[None, nocc:]) / 2.0
                # plain version
                # eia = 4.0 * eia * fia / (freqs[w]**2 + eia**2)
                # Pia = Lpq * eia
                # fused version
                # Pia = _Pia_fia(Lpq, eia, fia, freqs[w])
                # This line computes Pi[:,:,w] += einsum('Pia, Qia -> PQ', Pia, Lpq)
                # Pi[w] += Pia.reshape(naux, -1) @ Lpq.reshape(naux, -1).T
                # SYRK version: Pi[w] -= X X^T, accumulated in the lower triangle
                Xia = _Xia_fia(Lpq, eia, fia, freqs[w])
                cp.cublas.syrk('N', Xia.reshape(naux, -1), out=Pi[w], alpha=-1.0, beta=1.0, lower=True)
                Xia = None

    for w in range(nw):
        # get full Pi from lower triangle
        hermi_triu(Pi[w])
        ec_w = cp.linalg.slogdet((cp.eye(naux) - Pi[w]))[1]
        ec_w += cp.trace(Pi[w])
        e_corr += 1.0 / (2.0 * cp.pi) * ec_w * wts[w]

    return e_corr


def get_rho_response_metal(omega, mo_energy, Lpq, mo_occ, out=None):
    """Compute density response function in auxiliary basis at freq iw.

    Parameters
    ----------
    omega : double
        imaginary part of a frequency point
    mo_energy : double 1d array
        orbital energy
    Lpq : double 3d array
        three-center density-fitting matrix in MO
    mo_occ : double 1d array
        occupation number
    out : double ndarray, optional
        a location into which the result is stored, by default None

    Returns
    -------
    Pi : double 2d array
        density-density response function in auxiliary basis at freq iw
    """
    idx_occ, idx_frac, _ = get_idx_metal(mo_occ)
    nocc = len(idx_occ)
    nfrac = len(idx_frac)
    naux = Lpq.shape[0]
    eia = mo_energy[: nocc + nfrac, None] - mo_energy[None, nocc:]
    fia = (mo_occ[: nocc + nfrac, None] - mo_occ[None, nocc:]) / 2.0
    # Add a factor of 0.5 for double counting
    fia[nocc:, :nfrac] *= 0.5
    # plain version
    # eia = 4.0 * eia * fia / (omega**2 + eia**2)
    # Pia = Lpq * eia
    # fused version
    # Pia = _Pia_fia(Lpq, eia, fia, omega)
    # This line computes Pi = einsum('Pia, Qia -> PQ', Pia, Lpq)
    # Pi = cp.matmul(Pia.reshape(naux, -1), Lpq.reshape(naux, -1).T, out=out)
    # SYRK version: Pi = -X X^T, lower triangle only
    Xia = _Xia_fia(Lpq, eia, fia, omega)
    if out is None:
        out = cp.empty((naux, naux), dtype=cp.double)
    cp.cublas.syrk('N', Xia.reshape(naux, -1), out=out, alpha=-1.0, beta=0.0, lower=True)
    Xia = None
    # get full Pi from lower triangle
    return hermi_triu(out)


def get_rpa_exx(rpa, acfd=False, correction_only=False):
    """Calculate RPA exchange energy.
    For gapped systems, Hartree-Fock and adiabatic connection fluctuation dissipation exchange energies are the same.
    For metallic systems, they are different.
    The ACFD exchange energy is given by equation 12 in doi.org/10.1103/PhysRevB.81.115126

    Parameters
    ----------
    rpa : RPA
        rpa object
    acfd : bool, optional
        calculate ACFD exchange energy, by default False
    correction_only : bool, optional
        only calculate the correction term, by default False

    Returns
    -------
    ex : double
        exchange energy
    """
    mo_coeff = cp.array(rpa.mo_coeff)
    mo_occ = cp.array(rpa.mo_occ)

    if not hasattr(rpa._scf, 'sigma'):
        if correction_only:
            return 0.0

        dm = rpa._scf.make_rdm1()
        with lib_cpu.temporary_env(rpa.mol, verbose=0):
            vk = rpa._scf.get_k(rpa._scf.mol, dm)
        return -0.25 * contract('ij,ji->', dm, vk)

    if getattr(rpa, 'with_df', None) is None:
        rpa.initialize_df(auxbasis=rpa.auxbasis)

    ex = 0.0
    if hasattr(rpa._scf, 'sigma'):
        idx_occ, idx_frac, _ = get_idx_metal(mo_occ)
        nocc = len(idx_occ) + len(idx_frac)
        Lij = rpa.ao2mo(mo_coeff=mo_coeff, mo_occ=mo_occ, ijslice=(0, nocc, 0, nocc))

        if acfd is True:
            mo_occ_ij = cp.minimum(mo_occ[:nocc, None], mo_occ[None, :nocc]) / 2.0
            if correction_only is True:
                mo_occ_ij -= mo_occ[:nocc, None] * mo_occ[None, :nocc] / 4.0
        else:
            mo_occ_ij = mo_occ[:nocc, None] * mo_occ[None, :nocc] / 4.0
        Lij_occ = Lij * mo_occ_ij[None]
        # ex -= np.einsum('Lij,Lij->', Lij_occ.reshape(-1, nocc, nocc), Lij.reshape(-1, nocc, nocc))
        ex -= cp.cublas.dot(Lij_occ.ravel(), Lij.ravel())
    else:
        nocc = cp.count_nonzero(mo_occ)
        Lij = rpa.ao2mo(ijslice=(0, nocc, 0, nocc))
        ex -= cp.cublas.dotc(Lij.ravel(), Lij.ravel())
    return ex


class RPA(lib_cpu.StreamObject):
    def __init__(self, mf, frozen=None, auxbasis=None):
        self.mol = mf.mol  # mol object
        self._scf = mf  # mean-field object
        self.verbose = self.mol.verbose  # verbose level
        self.stdout = self.mol.stdout  # standard output
        self.max_memory = mf.max_memory  # max memory in MB
        self.auxbasis = auxbasis  # auxiliary basis for density fitting

        # options
        self.frozen = frozen  # frozen orbital options
        self.grids_alg = 'legendre'  # algorithm to generate grids
        self.nw = 40  # number of grids
        self.acfd_exx = False  # calculate ACFD exchange energy
        self.outcore = False  # low-memory routine
        self.segsize = 60  # number of orbitals in one segment for outcore

        # don't modify the following attributes, they are not input options
        self._nocc = None  # number of occupied orbitals
        self._nmo = None  # number of orbitals (exclude frozen orbitals)
        self._naux = None  # number of auxiliary basis functions
        self.mo_energy = cp.array(mf.mo_energy, copy=True)  # orbital energy
        self.mo_coeff = cp.array(mf.mo_coeff, copy=True)  # orbital coefficient
        self.mo_occ = cp.array(mf.mo_occ, copy=True)  # occupation number
        self.e_corr = None  # correlation energy
        self.e_hf = None  # Hartree-Fock energy
        self.e_tot = None  # total energy

        return

    def dump_flags(self, verbose=None):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('RPA nocc = %d, nvir = %d', nocc, nvir)
        if self.frozen is not None:
            log.info('frozen orbitals = %s', str(self.frozen))
        log.info('grid type = %s', self.grids_alg)
        log.info('number of grids = %d', self.nw)
        log.info('ACFD exchange energy = %s', self.acfd_exx)
        log.info('outcore mode = %s', self.outcore)
        if self.outcore is True:
            log.info('outcore segment size = %d', self.segsize)
        log.info('')
        return

    @property
    def nocc(self):
        return self.get_nocc()

    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return self.get_nmo()

    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def kernel(self, mo_energy=None, mo_coeff=None, Lpq=None):
        """RPA correlation and total energy

        Calculated total energy, HF energy and RPA correlation energy
        are stored in self.e_tot, self.e_hf, self.e_corr

        Parameters
        ----------
        mo_energy : double 1d array
            molecular orbital energies
        mo_coeff : double 2d ndarray
            molecular orbital coefficients
        Lpq : double 3d array, optional
            density fitting 3-center integral in MO basis, by default None
        nw : int, optional
            number of frequency point on imaginary axis, by default None

        Returns
        -------
        self.e_corr : float
            RPA correlation energy
        """
        if mo_coeff is None:
            mo_coeff = _mo_without_core(self, self.mo_coeff)
        if mo_energy is None:
            mo_energy = _mo_energy_without_core(self, self.mo_energy)

        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        self.e_tot, self.e_hf, self.e_corr = kernel(self, mo_energy, mo_coeff, Lpq=Lpq)

        logger.timer(self, 'RPA', *cput0)
        return self.e_corr

    def initialize_df(self, auxbasis=None):
        """Initialize density fitting.

        Parameters
        ----------
        auxbasis : str, optional
            name of auxiliary basis set, by default None
        """
        if getattr(self._scf, 'with_df', None):
            self.with_df = self._scf.with_df
        else:
            self.with_df = DF(self._scf.mol)
            if auxbasis is not None:
                self.with_df.auxbasis = auxbasis
            else:
                try:
                    self.with_df.auxbasis = make_auxbasis(self._scf.mol, mp2fit=True)
                except RuntimeError:
                    self.with_df.auxbasis = make_auxbasis(self._scf.mol, mp2fit=False)
        if self.with_df._cderi is None:
            self.with_df.build()
        self._naux = self.with_df.naux
        self._keys.update(['with_df'])
        return

    def ao2mo(self, mo_coeff=None, mo_occ=None, ijslice=None):
        """Transform density-fitting integral from AO to MO.

        Parameters
        ----------
        mo_coeff : double 2d array, optional
            coefficient from AO to MO, by default None
        mo_occ : double 1d array, optional
            occupation number, by default None

        Returns
        -------
        Lpq : double 3d array
            three-center density-fitting matrix in MO
        """
        if mo_coeff is None:
            mo_coeff = _mo_without_core(self, self.mo_coeff)
        if mo_occ is None:
            mo_occ = _mo_energy_without_core(self, self.mo_occ)

        mo = cp.asarray(mo_coeff)
        naux = self.with_df.naux
        if ijslice is None:
            nmo = mo_coeff.shape[1]
            idx_nocc, idx_frac, idx_vir = get_idx_metal(mo_occ)
            nocc = len(idx_nocc) + len(idx_frac)
            nvir = len(idx_vir) + len(idx_frac)
            ijslice = (0, nocc, nmo - nvir, nmo)

        mo_coeff_i = cp.ascontiguousarray(mo[:, ijslice[0]:ijslice[1]])
        mo_coeff_j = cp.ascontiguousarray(mo[:, ijslice[2]:ijslice[3]])
        nao, ni = mo_coeff_i.shape

        Lov = cp.empty((naux, ijslice[1] - ijslice[0], ijslice[3] - ijslice[2]), dtype=cp.double)
        buf = None
        p0 = 0
        for Lao, _ in self.with_df.loop():
            p1 = p0 + Lao.shape[0]
            if buf is None:
                # first block is the largest; later blocks reuse it
                buf = cp.empty(Lao.shape[0] * nao * ni)
            tmp = buf[:(p1 - p0) * nao * ni].reshape(p1 - p0, nao, ni)
            contract('Pmn,ni->Pmi', Lao, mo_coeff_i, out=tmp)
            contract('Pmi,mj->Pij', tmp, mo_coeff_j, out=Lov[p0:p1])
            p0 = p1
        assert p0 == naux, 'cderi loop produced fewer rows than expected'
        return Lov

    def get_grids(self, alg=None, nw=None, mo_energy=None):
        """Generate grids for integration.

        Parameters
        ----------
        alg : str, optional
            algorithm for generating grids, by default None
        nw : int, optional
            number of grids, by default None
        mo_energy : double 1d array, optional
            orbital energy, by default None

        Returns
        -------
        freqs : double 1d array
            frequency grid
        wts : double 1d array
            weight of grids
        """
        if alg is None:
            alg = self.grids_alg
        if mo_energy is None:
            mo_energy = _mo_energy_without_core(self, self.mo_energy)
        if alg == 'legendre':
            nw = 40 if nw is None else nw
            freqs, wts = _get_scaled_legendre_roots(nw)
        else:
            raise NotImplementedError('Grids algorithm not implemented!')

        freqs = cp.asarray(freqs)
        wts = cp.asarray(wts)
        return freqs, wts

    def get_ehf(self):
        """Get Hartree-Fock energy.

        Returns
        -------
        e_hf : double
            Hartree-Fock energy
        """
        with lib_cpu.temporary_env(self.with_df, verbose=0), lib_cpu.temporary_env(self.mol, verbose=0):
            with lib_cpu.temporary_env(self.with_df, verbose=0), lib_cpu.temporary_env(self._scf.mol, verbose=0):
                dm = self._scf.make_rdm1()
                e_1e = contract('ij,ji->', dm, self._scf.get_hcore())
                e_j = 0.5 * contract('ij,ji->', dm, self._scf.get_j(self._scf.mol, dm))
                e_x = self.get_rpa_exx(acfd=self.acfd_exx, correction_only=False)
                e_nuc = self._scf.energy_nuc()
                e_hf = e_1e + e_j + e_x + e_nuc
        return e_hf

    def get_rpa_exx(self, acfd=False, correction_only=False):
        """Calculate RPA exchange energy.

        Parameters
        ----------
        acfd : bool
            whether to include ACFD correction, by default False
        correction_only : bool
            only return the correction term, by default False

        Returns
        -------
        ex : double
            exchange energy
        """
        ex = get_rpa_exx(self, acfd=acfd, correction_only=correction_only)
        return ex

    to_cpu = utils.to_cpu
    to_gpu = utils.to_gpu
    device = utils.device
