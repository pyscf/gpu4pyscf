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
# Author: Jiachen Li <lijiachen.duke@gmail.com>
#

"""
Periodic spin-restricted G0W0 method based on the analytic continuation scheme, on one GPU.
This implementation has N^4 scaling,
and is faster than GW-CD (N^4~N^5) and fully analytic GW (N^6) methods.
GW-AC is recommended for valence states only, and is inaccurate for core states.

References:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    New J. Phys. 14 053020 (2012)
"""

import cupy as cp
import h5py
import numpy as np
import scipy
import scipy.optimize
import time

from pyscf import lib
from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots, PadeAC, TwoPoleAC
from pyscf.lib import temporary_env
from pyscf.pbc import df, dft
from pyscf.pbc.gw.krgw_ac import get_ef, get_g0_k
from pyscf.pbc.gw.krpa import get_kconserv_ria_efficient
from pyscf.pbc.mp.kmp2 import get_frozen_mask

from gpu4pyscf.df import make_auxbasis
from gpu4pyscf.lib import cupy_helper, logger
from gpu4pyscf.lib.cutensor import contract
from gpu4pyscf.pbc.df import GDF
from gpu4pyscf.pbc.dft import numint
from gpu4pyscf.pbc.lib.kpts_helper import kk_adapted_iter
from gpu4pyscf.gw.gw_ac import get_idx_metal
from gpu4pyscf.gw.utils.gw_cp_helper import mkslice, array_scale

# fuse functions for GPU
@cp.fuse()
def _Pia(Lia, eia, omega):
    return Lia * (eia / (omega**2 + eia**2))


@cp.fuse()
def _Pia_metal(Lia, eia, fia, omega):
    return Lia * (eia * fia / (omega**2 + eia**2))


@cp.fuse()
def _eia(eia, omega):
    return eia / (omega**2 + eia**2)


@cp.fuse()
def _eia_q(eia, qij, omega):
    return eia / (omega**2 + eia**2) * cp.conj(qij)


@cp.fuse()
def _g0(emo, wts, freq):
    return wts * emo / (emo**2 + freq**2)


def kernel(gw):
    mf = gw._scf
    nocc = gw.nocc
    nmo = gw.nmo
    nkpts = gw.nkpts

    # set frozen orbitals
    gw.set_frozen_orbs()
    orbs = gw.orbs
    orbs_frz = gw.orbs_frz
    kptlist = gw.kptlist
    if kptlist is None:
        gw.kptlist = kptlist = range(gw.nkpts)
    mo_energy_frz = _mo_energy_frozen(gw, gw.mo_energy)
    mo_coeff_frz = _mo_frozen(gw, gw.mo_coeff)

    # v_xc
    with temporary_env(mf, verbose=0), temporary_env(mf.mol, verbose=0), temporary_env(mf.with_df, verbose=0):
        dm = mf.make_rdm1()
        # gpu4pyscf KRHF.get_j does not default kpts to mf.kpts
        v_mf_ao = mf.get_veff(mf.mol, dm) - mf.get_j(mf.mol, dm, kpts=mf.kpts)
    v_mf = cp.zeros(shape=[nkpts, nmo, nmo], dtype=cp.complex128)
    for k in range(nkpts):
        v_mf[k] = mo_coeff_frz[k].T.conj() @ v_mf_ao[k] @ mo_coeff_frz[k]
    gw.vxc = v_mf

    # v_hf from DFT/HF density
    vk = gw.get_sigma_exchange()

    # finite size correction for exchange self-energy
    if gw.fc:
        vk_corr = -2.0 / np.pi * (6.0 * np.pi**2 / gw.mol.vol / nkpts) ** (1.0 / 3.0)
        for k in range(nkpts):
            for i in range(nocc):
                vk[k][i, i] = vk[k][i, i] + vk_corr
    gw.vk = vk

    # set up Fermi level
    gw.ef = ef = get_ef(kmf=mf, mo_energy=cp.asnumpy(mf.mo_energy))

    # grids for integration on imaginary axis
    gw.freqs, gw.wts = freqs, wts = _get_scaled_legendre_roots(gw.nw)

    # calculate self-energy on imaginary axis
    if gw.outcore:
        sigmaI, omega = get_sigma_outcore(
            gw, freqs, wts, ef=ef, mo_energy=mo_energy_frz, orbs=orbs_frz, kptlist=kptlist, iw_cutoff=gw.ac_iw_cutoff,
            fullsigma=gw.fullsigma,
        )
    else:
        sigmaI, omega = get_sigma(
            gw, freqs, wts, ef=ef, mo_energy=mo_energy_frz, orbs=orbs_frz, kptlist=kptlist, iw_cutoff=gw.ac_iw_cutoff,
            fullsigma=gw.fullsigma,
        )

    # begin working on the host
    omega = cp.asnumpy(omega)
    gw.vk = vk = cp.asnumpy(vk)
    gw.vxc = v_mf = cp.asnumpy(v_mf)
    # note that sigmaI and gw.sigmaI can be different
    sigmaI = cp.asnumpy(sigmaI)
    if gw.sigmaI is not None:
        gw.sigmaI = cp.asnumpy(gw.sigmaI)
    gw.mo_energy = cp.asnumpy(gw.mo_energy)
    gw.mo_coeff = cp.asnumpy(gw.mo_coeff)
    gw.mo_occ = cp.asnumpy(gw.mo_occ)

    # analytic continuation
    if gw.ac == 'twopole':
        acobj = TwoPoleAC(list(range(nmo)), nocc)
    elif gw.ac == 'pade':
        acobj = PadeAC(npts=gw.ac_pade_npts, step_ratio=gw.ac_pade_step_ratio)
    elif gw.ac == 'pes':
        raise NotImplementedError
    else:
        raise ValueError('Unknown GW-AC type %s' % (str(gw.ac)))

    acobj.ac_fit(sigmaI, omega, axis=-1)

    if gw.fullsigma:
        diag_acobj = acobj.diagonal(axis1=1, axis2=2)
    else:
        diag_acobj = acobj

    mo_energy_mf = cp.asnumpy(mf.mo_energy)
    mo_energy = np.zeros_like(mo_energy_mf)
    for ik, k in enumerate(kptlist):
        for ip, p in enumerate(orbs_frz):
            if gw.qpe_linearized:
                # linearized G0W0
                de = 1e-6
                ep = mo_energy_mf[k][orbs[ip]]
                sigmaR = diag_acobj[ik, ip].ac_eval(ep).real
                dsigma = diag_acobj[ik, ip].ac_eval(ep + de).real - sigmaR.real
                zn = 1.0 / (1.0 - dsigma / de)
                if gw.qpe_linearized_range is not None:
                    zn = 1.0 if zn < gw.qpe_linearized_range[0] or zn > gw.qpe_linearized_range[1] else zn
                mo_energy[k, orbs[ip]] = ep + zn * (sigmaR + vk[k, p, p] - v_mf[k, p, p]).real
            else:
                # self-consistently solve QP equation
                def quasiparticle(omega):
                    sigmaR = diag_acobj[ik, ip].ac_eval(omega)
                    return omega - mo_energy_mf[k][orbs[ip]] - (sigmaR + vk[k, p, p] - v_mf[k, p, p]).real

                try:
                    mo_energy[k, orbs[ip]] = scipy.optimize.newton(
                        quasiparticle, mo_energy_mf[k][orbs[ip]], tol=gw.qpe_tol, maxiter=gw.qpe_max_iter
                    )
                except RuntimeError:
                    logger.warn(gw, 'QPE for k=%d orbital=%d not converged!', k, orbs[ip])

    # save GW results
    gw.mo_energy = mo_energy
    gw.acobj = acobj

    if gw.verbose >= logger.DEBUG:
        with np.printoptions(threshold=len(mo_energy_mf[0])):
            for k in range(nkpts):
                logger.debug(gw, '  GW mo_energy @ k%d =\n%s', k, mo_energy[k])
        logger.warn(gw, 'GW QP energies may not be sorted from min to max')

    if gw.writefile > 0:
        with h5py.File('vxc.h5', 'w') as feri:
            feri['vk'] = np.asarray(vk)
            feri['v_mf'] = np.asarray(v_mf)

        with h5py.File('sigma_imag.h5', 'w') as feri:
            feri['sigmaI'] = np.asarray(sigmaI)
            feri['omega'] = np.asarray(omega)
            if gw.sigmaI is not None:
                feri['sigmaI_full'] = np.asarray(gw.sigmaI)

        acobj.save('ac_coeff.h5')

    return


def get_rho_response(omega, mo_energy, Lia, kidx):
    """Get Pi=PV.
    P is density-density response function.
    V is two-electron integral.
    See equation 24 in 10.1021/acs.jctc.0c00704

    Parameters
    ----------
    omega : double
        real position of imaginary frequency
    mo_energy : double 2d array
        orbital energy
    Lia : complex 4d ndarray
        occupied-virtual block of three-center density-fitting matrix in MO, aux-major (naux, nkpts, nocc, nvir)
    kidx : list
        momentum-conserved k-point list kj=kidx[ki]

    Returns
    -------
    Pi : complex ndarray
        Pi in auxiliary basis at freq iw
    """
    naux, nkpts, nocc, nvir = Lia.shape
    # Find ka that conserves with ki and kL (-ki+ka+kL=G)
    eia = mo_energy[:, :nocc, None] - mo_energy[np.asarray(kidx)][:, None, nocc:]
    # plain version
    # eia = eia / (omega**2 + eia**2)
    # Pia = Lia * eia
    # fused version
    Pia = _Pia(Lia, eia, omega)

    # Response from both spin-up and spin-down density
    # Pi = (4./nkpts) * einsum('Pkia,Qkia->PQ', Pia, Lia.conj())
    # One GEMM over the combined (k, i, a) index of the aux-major arrays, instead of one GEMM per k-point
    Pi = cp.empty(shape=[naux, naux], dtype=cp.complex128)
    cp.cublas.gemm(
        'H',
        'N',
        Lia.reshape(naux, nkpts * nocc * nvir).T,
        Pia.reshape(naux, nkpts * nocc * nvir).T,
        out=Pi.T,
        alpha=4.0 / nkpts,
        beta=0.0,
    )
    return Pi


def get_metal_windows(mo_energy, mo_occ, i, a, threshold=1.0e-8):
    """Occupation windows of the metallic response of the pair (ki, ka), see get_rho_response_metal.
    Only pairs with f_p != f_q contribute, so the occupied-occupied and virtual-virtual pairs are left out:
        window 1: rows = occupied + fractional orbitals at ki, columns = fractional + virtual orbitals at ka
        window 2: rows = fractional + virtual orbitals at ki, columns = occupied + fractional orbitals at ka
    Both orders of every pair are kept, as in the full nmo x nmo block, so no time-reversal symmetry is assumed.
    The fractional-fractional pairs are in both windows; their weight is set to zero in window 2.

    Parameters
    ----------
    mo_energy : double 2d array
        orbital energy
    mo_occ : double 2d array
        occupation number (0 to 2)
    i : int
        k-point index of the rows (ki)
    a : int
        k-point index of the columns (ka = kidx[ki])
    threshold : double, optional
        occupation threshold of get_idx_metal, by default 1e-8 (as in the molecular GWAC)

    Returns
    -------
    windows : list of (slice, slice, double 2d array, double 2d array)
        row slice (at ki), column slice (at ka), eia and fia = (f_p - f_q) / 2 of the two windows
    """
    nmo = mo_energy.shape[-1]
    idx_occ_i, idx_frac_i, _ = get_idx_metal(mo_occ[i], threshold=threshold)
    idx_occ_a, idx_frac_a, _ = get_idx_metal(mo_occ[a], threshold=threshold)
    nocc_i, nfrac_i = len(idx_occ_i), len(idx_frac_i)
    nocc_a, nfrac_a = len(idx_occ_a), len(idx_frac_a)
    windows = []
    for rows, cols in ((slice(0, nocc_i + nfrac_i), slice(nocc_a, nmo)),
                       (slice(nocc_i, nmo), slice(0, nocc_a + nfrac_a))):
        eia = mo_energy[i][rows, None] - mo_energy[a][None, cols]
        fia = (mo_occ[i][rows, None] - mo_occ[a][None, cols]) / 2.0
        windows.append((rows, cols, eia, fia))
    # the fractional-fractional pairs are counted in window 1
    windows[1][3][:nfrac_i, nocc_a:nocc_a + nfrac_a] = 0.0
    return windows


def get_rho_response_metal(omega, Lia, eia, fia, nkpts):
    """Get Pi=PV for metallic systems, from the occupation windows of get_metal_windows.
    P is density-density response function.
    V is two-electron integral.
    See equation 24 in 10.1021/acs.jctc.0c00704

    Parameters
    ----------
    omega : double 1d array
        real positions of imaginary frequencies
    Lia : complex 2d ndarray
        three-center density-fitting matrix in MO, the window elements of every pair (ki, ka=kidx[ki]) gathered into
        one aux-major index (naux, nia)
    eia : double 1d array
        orbital energy differences of the window elements
    fia : double 1d array
        (f_p - f_q) / 2 of the window elements
    nkpts : int
        number of k-points

    Returns
    -------
    Pi : complex ndarray
        Pi in auxiliary basis at every freq iw, (nw, naux, naux)
    """
    naux, nia = Lia.shape
    nw = len(omega)
    omega = cp.asarray(omega)

    # Compute Pi for kL
    Pi = cp.empty(shape=[nw, naux, naux], dtype=cp.complex128)
    # a block of frequencies of Pia uses at most 40% of the free GPU memory
    blksize = max(1, min(nw, int(0.4 * cupy_helper.get_avail_mem() / (16 * naux * nia))))
    for w0, w1 in lib.prange(0, nw, blksize):
        # plain version
        # eia = eia * fia / (omega**2 + eia**2)
        # Pia = Lia * eia
        # fused version, (w, naux, nia) for the block of frequencies
        Pia = _Pia_metal(Lia, eia, fia, omega[w0:w1, None, None])

        # Response from both spin-up and spin-down density
        # both ia and ai are included (window 2 holds the ai pairs), this gives a factor of 2.0
        # Pi = (2./nkpts) * einsum('Pia,Qia->PQ', Pia, Lia.conj())
        # One GEMM over the window elements of every ki, the frequencies of the block are stacked in the rows (w, P)
        cp.cublas.gemm(
            'H',
            'N',
            Lia.T,
            Pia.reshape((w1 - w0) * naux, nia).T,
            out=Pi[w0:w1].reshape((w1 - w0) * naux, naux).T,
            alpha=2.0 / nkpts,
            beta=0.0,
        )
        Pia = None

    return Pi


def get_rho_response_head(omega, mo_energy, qij):
    """Compute head (G=0, G'=0) density response function in auxiliary basis at freq iw.
    equation 48 in 10.1021/acs.jctc.0c00704

    Parameters
    ----------
    omega : double
        frequency point
    mo_energy : double ndarray
        orbital energy
    qij : complex ndarray
        pair density matrix defined as equation 51 in 10.1021/acs.jctc.0c00704

    Returns
    -------
    Pi_00 : complex
        head response function
    """
    nkpts, nocc = qij.shape[:2]
    Pi_00 = 0j
    for k in range(nkpts):
        eia = mo_energy[k, :nocc, None] - mo_energy[k, None, nocc:]
        # plain version
        # eia = eia / (omega**2 + eia**2)
        # fused version
        eia = _eia(eia, omega)
        Pi_00 += 4.0 / nkpts * cp.einsum('ia,ia->', eia, qij[k].conj() * qij[k])
    return Pi_00


def get_rho_response_wing(omega, mo_energy, Lia, qij):
    """Compute wing (G=P, G'=0) density response function in auxiliary basis at freq iw.
    equation 48 in 10.1021/acs.jctc.0c00704

    Parameters
    ----------
    omega : double
        frequency point
    mo_energy : double 2d array
        orbital energy
    Lia : complex 4d array
        occupied-virtual block of three-center density fitting matrix in MO at kL = 0, aux-major
        (naux, nkpts, nocc, nvir)
    qij : complex ndarray
        pair density matrix defined as equation 51 in 10.1021/acs.jctc.0c00704

    Returns
    -------
    Pi : complex ndarray
        wing response function
    """
    naux, nkpts, nocc, nvir = Lia.shape
    eia = mo_energy[:, :nocc, None] - mo_energy[:, None, nocc:]
    # plain version
    # eia = eia / (omega**2 + eia**2)
    # eia_q = eia * qij.conj()
    # fused version
    eia_q = _eia_q(eia, qij, omega)
    # Pi = 4.0 / nkpts * sum_k Lia[:, k] @ eia_q[k], one matrix-vector product over (k, i, a)
    Pi = 4.0 / nkpts * cp.matmul(Lia.reshape(naux, nkpts * nocc * nvir), eia_q.reshape(nkpts * nocc * nvir))
    return Pi


def get_qij_components(gw, mo_energy, mo_coeff, uniform_grids=False):
    """Cartesian components of the long-wavelength pair density: qij(q) = sum_x q_x qij_x.
    qij is linear in q, and the integration grid, AO values and AO gradients do not depend on q, so loops over
    q points only need these components once. AOs are evaluated in blocks of grid points to bound the GPU memory
    of numint.eval_ao, which allocates (4, n_bvk_cells, nao, ngrids) doubles.
    See get_qij.

    Parameters
    ----------
    gw : KRGWAC
        gw object, provides attributes: nocc, nmo, kpts, mol
    mo_energy : double ndarray
        orbital energy
    mo_coeff : complex ndarray
        coefficient from AO to MO
    uniform_grids : bool, optional
        use uniform grids, by default False

    Returns
    -------
    qij_x : complex ndarray
        (3, nkpts, nocc, nvir)
    """
    nocc = gw.nocc
    nmo = gw.nmo
    nvir = nmo - nocc
    kpts = gw.kpts
    nkpts = len(kpts)
    cell = gw.mol
    nao = cell.nao_nr()

    if uniform_grids:
        with temporary_env(cell, verbose=0):
            mydf = df.FFTDF(cell, kpts=kpts)
            coords = cell.gen_uniform_grids(mydf.mesh)
    else:
        with temporary_env(cell, verbose=0):
            coords, weights = dft.gen_grid.get_becke_grids(cell, level=4)
        weights = cp.asarray(weights)
    ngrid = len(coords)
    # numint.eval_ao holds (4, n_bvk_cells, nao, blksize) doubles, keep it near 2 GB
    blksize = max(1024, int(2e9 / (8 * 4 * nkpts * nao)))

    qij_x = cp.zeros(shape=[3, nkpts, nocc, nvir], dtype=cp.complex128)
    for i, kpti in enumerate(kpts):
        ao_ao_grad = cp.zeros(shape=[3, nao, nao], dtype=cp.complex128)
        for p0 in range(0, ngrid, blksize):
            p1 = min(p0 + blksize, ngrid)
            ao_p = numint.eval_ao(cell, coords[p0:p1], kpt=kpti, deriv=1)
            ao = ao_p[0]
            ao_grad = ao_p[1:4]
            if uniform_grids:
                ao_ao_grad += cp.einsum('mg,xgn->xmn', ao.T.conj(), ao_grad) * cell.vol / ngrid
            else:
                # ao_ao_grad = cp.einsum('g,mg,xgn->xmn', weights, ao.T.conj(), ao_grad)
                ao_ao_grad += contract('mg,xgn->xmn', ao.T.conj() * weights[p0:p1], ao_grad)
            ao_p = ao = ao_grad = None
        enm = 1.0 / (mo_energy[i][nocc:, None] - mo_energy[i][None, :nocc])
        for x in range(3):
            # q_ao_ao_grad = -1j * einsum('x,xmn->mn', q, ao_ao_grad); everything below is linear in q
            q_mo_mo_grad = mo_coeff[i][:, :nocc].T.conj() @ (-1j * ao_ao_grad[x]) @ mo_coeff[i][:, nocc:]
            qij_x[x, i] = enm.T * q_mo_mo_grad / np.sqrt(cell.vol)

    return qij_x


def get_qij(gw, q, mo_energy, mo_coeff, uniform_grids=False):
    """Compute pair density matrix in the long-wavelength limit through kp perturbation theory
    qij = 1/Omega * |< psi_{ik} | e^{iqr} | psi_{ak-q} >|^2
    equation 51 in 10.1021/acs.jctc.0c00704
    Ref: Phys. Rev. B 83, 245122 (2011)

    For several q points, call get_qij_components once and contract it with each q instead.

    Parameters
    ----------
    gw : KRGWAC
        gw object, provides attributes: nocc, nmo, kpts, mol
    q : double
        q grid
    mo_energy : double ndarray
        orbital energy
    mo_coeff : complex ndarray
        coefficient from AO to MO
    uniform_grids : bool, optional
        use uniform grids, by default False

    Returns
    -------
    qij : complex ndarray
        pair density matrix in the long-wavelength limit
    """
    qij_x = get_qij_components(gw, mo_energy, mo_coeff, uniform_grids=uniform_grids)
    return cp.einsum('x,xkia->kia', cp.asarray(q, dtype=cp.double).astype(cp.complex128), qij_x)


def get_sigma(
    gw, freqs, wts, ef, mo_energy, orbs=None, kptlist=None, mo_coeff=None, mo_occ=None, iw_cutoff=None, fullsigma=False
):
    """Get GW self-energy.
    See equation 27 in 10.1021/acs.jctc.0c00704

    Parameters
    ----------
    gw : KRGWAC
        GW objects, provides attributes: _scf, mol, frozen, nmo, nocc, kpts, nkpts, mo_coeff, mo_occ, fc, fc_grid, with_df
    freqs : double array
        position of imaginary frequency
    wts : double array
        weight of frequency points
    ef : double
        Fermi level
    mo_energy : double ndarray
        non-frozen orbital energy
    orbs : list, optional
        orbital index in non-frozen nmo to calculate self-energy, by default None
    kptlist : list, optional
        k-point index to calculate self-energy, by default None
    mo_coeff : complex ndarray, optional
        coefficient from AO to non-frozen MO, by default None
    mo_occ : double ndarray, optional
        non-frozen occupation number, by default None
    iw_cutoff : complex, optional
        imaginary grid cutoff for fitting, by default None
    fullsigma : bool, optional
        calculate off-diagonal elements, by default False

    Returns
    -------
    sigma: complex ndarray
        self-energy on the imaginary axis
    omega: complex ndarray
        imaginary frequency grids of self-energy
    """
    nocc = gw.nocc
    nmo = gw.nmo
    nkpts = gw.nkpts
    kpts = gw.kpts
    mydf = gw.with_df

    if orbs is None:
        orbs = list(range(nmo))
    if kptlist is None:
        kptlist = list(range(nkpts))
    norbs = len(orbs)
    nklist = len(kptlist)
    nw = len(freqs)

    if mo_coeff is None:
        mo_coeff = _mo_frozen(gw, gw.mo_coeff)
    if mo_occ is None:
        mo_occ = _mo_occ_frozen(gw, gw.mo_occ)
    nao = mo_coeff[0].shape[0]

    # possible kpts shift center
    kscaled = gw.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]
    # momentum conservation of every kL at once: kconserv_table[kL, ki] = kj with -ki+kj+kL=G
    kconserv_table = get_kconserv_ria_efficient(gw.mol, kpts - kpts[0])

    # Integration on numerical grids
    if iw_cutoff is not None and gw.rdm is False:
        nw_sigma = sum(iw < iw_cutoff for iw in freqs) + 1
    else:
        nw_sigma = nw + 1

    omega = np.zeros(shape=[nw_sigma], dtype=np.complex128)
    omega[1:] = 1j * freqs[: (nw_sigma - 1)] + ef
    emo = cp.asarray(omega)[None, None, :] - mo_energy[:, :, None]
    # all frequencies are handled at once for every kL, see below
    freqs_gpu = cp.asarray(freqs)
    wts_gpu = cp.asarray(wts)
    if hasattr(gw._scf, 'sigma'):
        # get_metal_windows works on host copies, so that it does not launch kernels for every ki
        mo_energy_host = cp.asnumpy(mo_energy)
        mo_occ_host = cp.asnumpy(mo_occ)

    if fullsigma is False:
        sigma = cp.zeros(shape=[nklist, norbs, nw_sigma], dtype=cp.complex128)
    else:
        sigma = cp.zeros(shape=[nklist, norbs, norbs, nw_sigma], dtype=cp.complex128)
    if gw.fc:
        # Set up q mesh for q->0 finite size correction
        if not gw.fc_grid:
            q_pts = np.array([1e-3, 0, 0], dtype=np.double).reshape(1, 3)
        else:
            Nq = 3
            q_pts = np.zeros(shape=[Nq**3 - 1, 3], dtype=np.double)
            for i in range(Nq):
                for j in range(Nq):
                    for k in range(Nq):
                        if i == 0 and j == 0 and k == 0:
                            continue
                        else:
                            q_pts[i * Nq**2 + j * Nq + k - 1, 0] = k * 5e-4
                            q_pts[i * Nq**2 + j * Nq + k - 1, 1] = j * 5e-4
                            q_pts[i * Nq**2 + j * Nq + k - 1, 2] = i * 5e-4
        nq_pts = len(q_pts)
        q_abs = gw.mol.get_abs_kpts(q_pts)

        # Get qij = 1/sqrt(Omega) * < psi_{ik} | e^{iqr} | psi_{ak-q} > at q: (nkpts, nocc, nvir)
        qij = cp.zeros(shape=[nq_pts, nkpts, nocc, nmo - nocc], dtype=cp.complex128)

        for k in range(nq_pts):
            qij[k] = get_qij(gw, q_abs[k], mo_energy, mo_coeff)

    for kL in range(nkpts):
        # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
        # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
        kidx = kconserv_table[kL]
        kidx_r = np.argsort(kidx)
        # every (ki, kidx[ki]) conserves momentum with kL
        dk = -kscaled + kscaled[kidx] + kscaled[kL]
        assert np.linalg.norm(np.round(dk) - dk, axis=1).max() < 1e-12

        # Read (L|pq) for every ki and kj that conserves with kL and ki (-ki+kj+kL=G).
        # GDF only stores one of the two conjugated momentum transfers kL and -kL,
        # (L|ki kj) is either Lpq_kL[ki] or Lpq_kL[kj].conj().transpose(0, 2, 1).
        for kp, _, ki_idx, kj_idx in kk_adapted_iter(mydf.kmesh):
            is_conj = (kidx[ki_idx] != kj_idx).any()
            if not is_conj or (kidx[kj_idx] == ki_idx).all():
                break
        else:
            raise RuntimeError(f'Momentum transfer of kL = {kL} is not found in GDF')
        naux = mydf._cderi[kp].shape[0]

        # Lij: (ki, L, i, j) for looping every kL
        Lij = cp.zeros(shape=[nkpts, naux, nmo, nmo], dtype=cp.complex128)
        # a block uses at most 40% of the free GPU memory, with the two buffers and the ao2mo intermediates
        blksize = max(1, min(naux, int(0.4 * cupy_helper.get_avail_mem() / (64 * nkpts * nao**2))))
        aux_iter = [(kp, b0, b1) for b0, b1 in lib.prange(0, naux, blksize)]
        # one output and one unpacking buffer for all blocks, so each block overwrites the previous one
        out_buf = cp.empty(nkpts * blksize * nao**2, dtype=cp.complex128)
        unpack_buf = cp.empty(nkpts * blksize * nao**2, dtype=cp.complex128)
        p0 = 0
        # slice over naux
        for _, Lpq_kL, sign in mydf.loop(blksize, kpts=kpts, aux_iter=aux_iter, buf=unpack_buf, out=out_buf):
            if sign < 0:
                continue  # 2D cells (dimension=2): the negative part of the cderi is not used, as before
            p1 = p0 + Lpq_kL.shape[1]
            logger.debug(gw, 'Read Lpq (kL: %s / %s, aux: %s-%s)', kL + 1, nkpts, p0, p1)
            # ao2mo transform (L|pq) to (L|ij) of every ki and kj = kidx[ki] at once
            if is_conj:
                Lpq = Lpq_kL[kidx].conj().transpose(0, 1, 3, 2)
            else:
                Lpq = Lpq_kL
            Lpq = Lpq.astype(cp.complex128, copy=False)
            # This computes Lij[:, p0:p1] = einsum('kmi,kPmn,knj->kPij', mo_coeff.conj(), Lpq, mo_coeff[kidx])
            tmp = contract('kPmn,kmi->kPin', Lpq, mo_coeff.conj())
            contract('kPin,knj->kPij', tmp, mo_coeff[kidx], out=Lij[:, p0:p1])
            # free this block before the next one is read
            tmp = Lpq = None
            p0 = p1
        Lpq_kL = Lpq = out_buf = unpack_buf = None

        if hasattr(gw._scf, 'sigma') is False:
            # aux-major occupied-virtual block (naux, nkpts, nocc, nvir) for get_rho_response
            Lia = cp.ascontiguousarray(Lij[:, :, :nocc, nocc:].transpose(1, 0, 2, 3))
        else:
            # occupation windows of every ki for get_rho_response_metal,
            # gathered once into an aux-major block (naux, nia), as they do not depend on the frequency
            idx_k, idx_pq, eia, fia = [], [], [], []
            for i in range(nkpts):
                for rows, cols, eia_i, fia_i in get_metal_windows(mo_energy_host, mo_occ_host, i, kidx[i]):
                    pq = np.arange(nmo * nmo).reshape(nmo, nmo)[rows, cols].ravel()
                    idx_k.append(np.full(pq.size, i))
                    idx_pq.append(pq)
                    eia.append(eia_i.ravel())
                    fia.append(fia_i.ravel())
            idx_k = cp.asarray(np.concatenate(idx_k))
            idx_pq = cp.asarray(np.concatenate(idx_pq))
            Lia = cp.ascontiguousarray(Lij.reshape(nkpts, naux, nmo * nmo)[idx_k, :, idx_pq].T)
            eia = cp.asarray(np.concatenate(eia))
            fia = cp.asarray(np.concatenate(fia))
            idx_k = idx_pq = None

        # frequencies are handled in blocks: Pi, its inverse and the self-energy intermediates of a block
        # use at most 40% of the free GPU memory (every frequency is in one block for small systems)
        nW = nmo * norbs if fullsigma is False else nmo * norbs * norbs
        mem_w = 16 * (4 * naux**2 + nmo * norbs * naux + 2 * nW + nmo * nw_sigma)
        wblksize = max(1, min(nw, int(0.4 * cupy_helper.get_avail_mem() / mem_w)))
        naux_ones = cp.ones(shape=[1, naux], dtype=cp.complex128)
        if fullsigma:
            l_slice_mQn = cp.zeros(shape=[nmo, naux, norbs], dtype=cp.complex128)

        for w0, w1 in lib.prange(0, nw, wblksize):
            nwb = w1 - w0
            # Pi at every frequency of the block, (nwb, naux, naux)
            if hasattr(gw._scf, 'sigma'):
                Pi = get_rho_response_metal(freqs[w0:w1], Lia, eia, fia, nkpts)
            else:
                Pi = cp.empty(shape=[nwb, naux, naux], dtype=cp.complex128)
                for w in range(nwb):
                    Pi[w] = get_rho_response(freqs[w0 + w], mo_energy, Lia, kidx)
            Pi_inv = cp.linalg.inv(cp.eye(naux) - Pi)
            Pi = None

            if gw.fc and kL == 0:
                Del_00 = cp.zeros(shape=[nwb], dtype=cp.complex128)
                eps_inv_P0 = cp.zeros(shape=[nwb, naux], dtype=cp.complex128)
                for w in range(nwb):
                    eps_inv_00 = 0j
                    for iq in range(nq_pts):
                        # head dielectric matrix eps_00, equation 47 in 10.1021/acs.jctc.0c00704
                        Pi_00 = get_rho_response_head(freqs[w0 + w], mo_energy, qij[iq])
                        eps_00 = 1.0 - 4.0 * np.pi / np.linalg.norm(q_abs[iq]) ** 2.0 * Pi_00

                        # wings dielectric matrix eps_P0, equation 48 in 10.1021/acs.jctc.0c00704
                        Pi_P0 = get_rho_response_wing(freqs[w0 + w], mo_energy, Lia, qij[iq])
                        eps_P0 = -np.sqrt(4.0 * np.pi) / np.linalg.norm(q_abs[iq]) * Pi_P0

                        # inverse dielectric matrix
                        # equation 53 in 10.1021/acs.jctc.0c00704
                        eps_inv_00 += 1.0 / nq_pts * 1.0 / (eps_00 - eps_P0.conj() @ Pi_inv[w] @ eps_P0)
                        # equation 54 in 10.1021/acs.jctc.0c00704
                        eps_inv_P0[w] += 1.0 / nq_pts * (-eps_inv_00) * cp.matmul(Pi_inv[w], eps_P0)

                    # head correction, equation 43 in 10.1021/acs.jctc.0c00704
                    Del_00[w] = 2.0 / np.pi * (6.0 * np.pi**2 / gw.mol.vol / nkpts) ** (1.0 / 3.0) * (eps_inv_00 - 1.0)

            Pi_inv -= cp.eye(naux)
            # (naux, nwb, naux), so that one GEMM contracts Lij with Pi_inv of every frequency of the block
            Pi_inv = cp.ascontiguousarray(Pi_inv.transpose(1, 0, 2))

            # allocate intermediates of the block
            mnQ = cp.empty(shape=[nmo * norbs, nwb * naux], dtype=cp.complex128)
            if fullsigma is False:
                Wmn = cp.empty(shape=[nmo, norbs, nwb], dtype=cp.complex128)
            else:
                Wmn = cp.empty(shape=[nmo, norbs * nwb, norbs], dtype=cp.complex128)

            for k in range(nklist):
                kn = kptlist[k]
                # Find km that conserves with kn and kL (-km+kn+kL=G)
                km = kidx_r[kn]

                if len(orbs) == nmo:
                    l_slice = cp.ascontiguousarray(Lij[km].reshape(naux, -1))
                    if fullsigma:
                        l_slice_mQn[:] = Lij[km].transpose(1, 0, 2)
                else:
                    l_slice = cp.ascontiguousarray(Lij[km, :, :, mkslice(orbs)].reshape(naux, -1))
                    if fullsigma:
                        l_slice_mQn[:] = Lij[km, :, :, mkslice(orbs)].transpose(1, 0, 2)

                # plain version
                # g0 = wts[w] * emo[km] / (emo[km]**2 + freqs[w] ** 2)
                # fused version, (nmo, nwb, nw_sigma) for every frequency w of the block
                g0 = _g0(emo[km][:, None, :], wts_gpu[w0:w1, None], freqs_gpu[w0:w1, None])

                # Qmn = einsum('Pmn,PwQ->mnwQ', Lij[km][:, :, orbs].conj(), Pi_inv)
                cp.cublas.gemm('N', 'H', Pi_inv.reshape(naux, nwb * naux).T, l_slice.T, out=mnQ.T)

                if fullsigma is False:
                    # Wmn = einsum('mnwQ,Qmn->mnw',Qmn,Lij[km][:,:,orbs])
                    Qmn = mnQ.reshape(nmo * norbs, nwb, naux)
                    Qmn *= l_slice.T[:, None, :]
                    cp.matmul(mnQ.reshape(nmo * norbs * nwb, naux), naux_ones.T, out=Wmn.reshape(nmo * norbs * nwb, 1))
                    # (nmo, nwb, norbs), so that (m, w) is one index
                    Wmn_t = cp.ascontiguousarray(Wmn.transpose(0, 2, 1))

                    # sigma[k] += -1./nkpts * einsum('mwn,mwv->nv',Wmn,g0) / np.pi, one GEMM over (m, w)
                    cp.cublas.gemm(
                        'T',
                        'N',
                        g0.reshape(nmo * nwb, nw_sigma),
                        Wmn_t.reshape(nmo * nwb, norbs),
                        out=sigma[k].T,
                        alpha=-1.0 / nkpts / np.pi,
                        beta=1.0,
                    )
                else:
                    # for orbm in range(nmo):
                    #     Wmn[orbm] = np.dot(Qmn[orbm].reshape(norbs * nwb, naux), Lij[km][:,orbm,orbs])
                    cp.matmul(mnQ.reshape(nmo, norbs * nwb, naux), l_slice_mQn, out=Wmn)
                    # (nmo, nwb, norbs, norbs), so that (m, w) is one index
                    Wmn_t = cp.ascontiguousarray(Wmn.reshape(nmo, norbs, nwb, norbs).transpose(0, 2, 1, 3))

                    # sigma[k] += -1./nkpts * einsum('mwnl,mwv->nlv',Wmn,g0) / np.pi, one GEMM over (m, w)
                    cp.cublas.gemm(
                        'T',
                        'N',
                        g0.reshape(nmo * nwb, nw_sigma),
                        Wmn_t.reshape(nmo * nwb, norbs * norbs),
                        out=sigma[k].reshape(norbs * norbs, nw_sigma).T,
                        alpha=-1.0 / nkpts / np.pi,
                        beta=1.0,
                    )
                Qmn = Wmn_t = None

                if gw.fc and kL == 0:
                    # Find km that conserves with kn and kL (-km+kn+kL=G)
                    assert kn == km
                    # head correction
                    tmp = -cp.einsum('w,nwv->nv', Del_00, g0[mkslice(orbs)]) / np.pi

                    # wing correction
                    Wn_P0 = cp.einsum('Pnn,wP->wn', Lij[kn], eps_inv_P0)
                    Wn_P0 = Wn_P0[:, mkslice(orbs)].real * 2.0
                    Del_P0 = np.sqrt(gw.mol.vol/4/np.pi**3) * (6*np.pi**2/gw.mol.vol/nkpts) ** (2/3) * Wn_P0
                    tmp += -cp.einsum('wn,nwv->nv', Del_P0, g0[mkslice(orbs)]) / np.pi
                    if fullsigma is False:
                        sigma[k] += tmp
                    else:
                        sigma[k, cp.arange(norbs), cp.arange(norbs), :] += tmp
            # free the intermediates of the block before the next block
            mnQ = Wmn = Pi_inv = g0 = None

        del Lij, Lia, l_slice
        if hasattr(gw._scf, 'sigma'):
            del eia, fia
        if fullsigma:
            del l_slice_mQn

    if gw.rdm:
        gw.sigmaI = sigma

    return sigma, omega


def get_sigma_outcore(
    gw, freqs, wts, ef, mo_energy, orbs=None, kptlist=None, mo_coeff=None, mo_occ=None, iw_cutoff=None, fullsigma=False
):
    """Low-memory routine to get GW self-energy.
    See equation 27 in 10.1021/acs.jctc.0c00704

    Parameters
    ----------
    gw : KRGWAC
        GW objects, provides attributes: _scf, mol, frozen, nmo, nocc, kpts, nkpts, mo_coeff, mo_occ, fc, fc_grid, with_df
    freqs : double array
        position of imaginary frequency
    wts : double array
        weight of frequency points
    ef : double
        Fermi level
    mo_energy : double ndarray
        non-frozen orbital energy
    orbs : list, optional
        orbital index in non-frozen nmo to calculate self-energy, by default None
    kptlist : list, optional
        k-point index to calculate self-energy, by default None
    mo_coeff : complex ndarray, optional
        coefficient from AO to non-frozen MO, by default None
    mo_occ : double ndarray, optional
        non-frozen occupation number, by default None
    iw_cutoff : complex, optional
        imaginary grid cutoff for fitting, by default None
    fullsigma : bool, optional
        calculate off-diagonal elements, by default False

    Returns
    -------
    sigma: complex ndarray
        self-energy on the imaginary axis
    omega: complex ndarray
        imaginary frequency grids of self-energy
    """
    assert gw.fc is False, "finite-size correction is not implemented in get_sigma_outcore"
    nocc = gw.nocc
    nmo = gw.nmo
    nkpts = gw.nkpts
    kpts = gw.kpts
    mydf = gw.with_df

    if orbs is None:
        orbs = list(range(nmo))
    if kptlist is None:
        kptlist = list(range(nkpts))
    norbs = len(orbs)
    nklist = len(kptlist)
    nw = len(freqs)

    if mo_coeff is None:
        mo_coeff = _mo_frozen(gw, gw.mo_coeff)
    if mo_occ is None:
        mo_occ = _mo_occ_frozen(gw, gw.mo_occ)

    # possible kpts shift center
    kscaled = gw.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]
    # momentum conservation of every kL at once: kconserv_table[kL, ki] = kj with -ki+kj+kL=G
    kconserv_table = get_kconserv_ria_efficient(gw.mol, kpts - kpts[0])

    # Integration on numerical grids
    if iw_cutoff is not None and gw.rdm is False:
        nw_sigma = sum(iw < iw_cutoff for iw in freqs) + 1
    else:
        nw_sigma = nw + 1

    omega = np.zeros(shape=[nw_sigma], dtype=np.complex128)
    omega[1:] = 1j * freqs[: (nw_sigma - 1)] + ef
    emo = cp.asarray(omega)[None, None, :] - mo_energy[:, :, None]

    if fullsigma is False:
        sigma = cp.zeros(shape=[nklist, norbs, nw_sigma], dtype=cp.complex128)
    else:
        sigma = cp.zeros(shape=[nklist, norbs, norbs, nw_sigma], dtype=cp.complex128)

    cput0 = (time.process_time(), time.perf_counter())
    for kL in range(nkpts):
        cput3 = (time.process_time(), time.perf_counter())
        # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
        # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
        kidx = kconserv_table[kL]
        kidx_r = np.argsort(kidx)
        # every (ki, kidx[ki]) conserves momentum with kL
        dk = -kscaled + kscaled[kidx] + kscaled[kL]
        assert np.linalg.norm(np.round(dk) - dk, axis=1).max() < 1e-12

        # Read (L|pq) for every ki and kj that conserves with kL and ki (-ki+kj+kL=G).
        # GDF only stores one of the two conjugated momentum transfers kL and -kL,
        # (L|ki kj) is either Lpq_kL[ki] or Lpq_kL[kj].conj().transpose(0, 2, 1).
        for kp, _, ki_idx, kj_idx in kk_adapted_iter(mydf.kmesh):
            is_conj = (kidx[ki_idx] != kj_idx).any()
            if not is_conj or (kidx[kj_idx] == ki_idx).all():
                break
        else:
            raise RuntimeError(f'Momentum transfer of kL = {kL} is not found in GDF')
        naux = mydf._cderi[kp].shape[0]
        Lpq_kL = next(mydf.loop(naux, kpts=kpts, aux_iter=[(kp, 0, naux)]))[1]

        Pi = cp.zeros(shape=[nw, naux, naux], dtype=cp.complex128)
        cput1 = (time.process_time(), time.perf_counter())
        for i in range(nkpts):
            a = kidx[i]
            logger.debug(gw, 'Pi (kL: %s / %s, ki: %s, kj: %s)' % (kL + 1, nkpts, a, kidx_r[a]))
            if is_conj:
                Lpq_ao = Lpq_kL[a].conj().transpose(0, 2, 1)
            else:
                Lpq_ao = Lpq_kL[i]
            Lpq_ao = Lpq_ao.astype(cp.complex128, copy=False)

            # This computes Lpq = einsum('mi,Pmn,nj->Pij', mo_coeff[i].conj(), Lpq_ao, mo_coeff[a])
            tmp = contract('Pmn,mi->Pin', Lpq_ao, mo_coeff[i].conj())
            Lpq = contract('Pin,nj->Pij', tmp, mo_coeff[a])
            tmp = Lpq_ao = None

            if hasattr(gw._scf, 'sigma'):
                # eia = mo_energy[i, :, None] - mo_energy[a, None, :]
                # fia = (mo_occ[i][:, None] - mo_occ[a][None, :]) / 2.0
                # Lia = Lpq
                # occupation windows of (ki, ka), see get_rho_response_metal
                for rows, cols, eia, fia in get_metal_windows(mo_energy, mo_occ, i, a):
                    ni, na = eia.shape
                    Lia = cp.ascontiguousarray(Lpq[:, rows, cols])
                    if ni * na == 0:
                        continue
                    for w in range(nw):
                        # plain version
                        # freqs_w = freqs[w]
                        # eia_w = eia * fia / (freqs_w**2 + eia**2)
                        # Pia = Lia * eia_w
                        # fused version
                        Pia = _Pia_metal(Lia, eia, fia, freqs[w])

                        # Response from both spin-up and spin-down density
                        # both ia and ai are included (window 2 holds the ai pairs), this gives a factor of 2.0
                        # Pi += (2./nkpts) * einsum('Pia,Qia->PQ', Pia, Lpq_i.conj())
                        cp.cublas.gemm(
                            'H',
                            'N',
                            Lia.reshape(naux, ni * na).T,
                            Pia.reshape(naux, ni * na).T,
                            out=Pi[w].T,
                            alpha=2.0 / nkpts,
                            beta=1.0,
                        )
                        del Pia
                del eia, fia
            else:
                eia = mo_energy[i, :nocc, None] - mo_energy[a, None, nocc:]
                Lia = cp.ascontiguousarray(Lpq[:, :nocc, nocc:])
                nvir = Lia.shape[-1]
                for w in range(nw):
                    # plain version
                    # freqs_w = freqs[w]
                    # eia_w = eia / (freqs_w**2 + eia**2)
                    # Pia = Lia * eia_w
                    # fused version
                    Pia = _Pia(Lia, eia, freqs[w])

                    # Response from both spin-up and spin-down density
                    # Pi += (4./nkpts) * einsum('Pia,Qia->PQ', Pia, Lov.conj())
                    cp.cublas.gemm(
                        'H',
                        'N',
                        Lia.reshape(naux, nocc * nvir).T,
                        Pia.reshape(naux, nocc * nvir).T,
                        out=Pi[w].T,
                        alpha=4.0 / nkpts,
                        beta=1.0,
                    )
                    del Pia
                del eia
            del Lpq, Lia

        logger.timer(gw, 'Calculate Pi for kL: %s / %s' % (kL + 1, nkpts), *cput1)

        for w in range(nw):
            Pi[w] = cp.linalg.inv(cp.eye(naux) - Pi[w])
            Pi[w] -= cp.eye(naux)
        Pi_inv = Pi

        # allocate intermediates
        naux_ones = cp.ones(shape=[1, naux], dtype=cp.complex128)
        mnQ = cp.zeros(shape=[nmo * norbs, naux], dtype=cp.complex128)
        if fullsigma is False:
            Wmn = cp.zeros(shape=[nmo, norbs], dtype=cp.complex128)
        else:
            Wmn = cp.zeros(shape=[nmo, norbs, norbs], dtype=cp.complex128)

        for k in range(nklist):
            kn = kptlist[k]
            # Find km that conserves with kn and kL (-km+kn+kL=G)
            km = kidx_r[kn]

            cput2 = (time.process_time(), time.perf_counter())
            logger.debug(gw, 'sigma (kL: %s / %s, ki: %s, kj: %s)' % (kL + 1, nkpts, km, kn))
            if is_conj:
                Lpq_ao = Lpq_kL[kn].conj().transpose(0, 2, 1)
            else:
                Lpq_ao = Lpq_kL[km]
            Lpq_ao = Lpq_ao.astype(cp.complex128, copy=False)

            # This computes Lpq = einsum('mi,Pmn,nj->Pij', mo_coeff[km].conj(), Lpq_ao, mo_coeff[kn])
            tmp = contract('Pmn,mi->Pin', Lpq_ao, mo_coeff[km].conj())
            Lpq = contract('Pin,nj->Pij', tmp, mo_coeff[kn])
            tmp = Lpq_ao = None

            if len(orbs) == nmo:
                l_slice = cp.ascontiguousarray(Lpq.reshape(naux, -1))
                if fullsigma:
                    l_slice_mQn = cp.ascontiguousarray(Lpq.transpose(1, 0, 2))
            else:
                l_slice = cp.ascontiguousarray(Lpq[:, :, mkslice(orbs)].reshape(naux, -1))
                if fullsigma:
                    l_slice_mQn = cp.ascontiguousarray(Lpq[:, :, mkslice(orbs)].transpose(1, 0, 2))

            for w in range(nw):
                # plain version
                # g0 = wts[w] * emo[km] / (emo[km]**2 + freqs[w] ** 2)
                # fused version
                g0 = _g0(emo[km], wts[w], freqs[w])

                # Qmn = einsum('Pmn,PQ->Qmn', Lij[km][:, :, orbs].conj(), Pi_inv)
                cp.cublas.gemm('N', 'H', Pi_inv[w].T, l_slice.T, out=mnQ.T)

                if fullsigma is False:
                    # Wmn = 1./nkpts * einsum('Qmn,Qmn->mn',Qmn,Lij[km][:,:,orbs])
                    mnQ *= l_slice.T
                    cp.matmul(mnQ, naux_ones.T, out=Wmn.reshape(nmo * norbs, 1))
                    array_scale(Wmn, 1.0 / nkpts / np.pi)

                    # sigma[k] += -einsum('mn,mw->nw',Wmn,g0[km]) / np.pi
                    # 1 / np.pi is included in Wmn above
                    sigma[k] -= cp.matmul(Wmn.reshape(nmo, norbs).T, g0)
                else:
                    # for orbm in range(nmo):
                    #     Wmn[orbm] = 1./nkpts * np.dot(Qmn[:,orbm,:].transpose(),Lij[km][:,orbm,orbs])
                    cp.matmul(mnQ.reshape(nmo, norbs, naux), l_slice_mQn, out=Wmn)
                    array_scale(Wmn, 1.0 / nkpts / np.pi)

                    # sigma[k] += -einsum('mnl,mw->nlw',Wmn,g0[km])/np.pi
                    # 1 / np.pi is included in Wmn above
                    sigma[k] -= cp.matmul(Wmn.reshape(nmo, norbs * norbs).T, g0).reshape(norbs, norbs, nw_sigma)

            del Lpq, l_slice
            if fullsigma:
                del l_slice_mQn
            logger.timer(gw, 'GW correlation self-energy for kL: %s / %s kn: %d' % (kL + 1, nkpts, kn), *cput2)

        del Pi, Pi_inv, mnQ, Wmn
        Lpq_kL = None
        logger.timer(gw, 'GW correlation self-energy for kL: %s / %s' % (kL + 1, nkpts), *cput3)

    if gw.rdm:
        gw.sigmaI = sigma

    logger.timer(gw, 'GW correlation self-energy', *cput0)

    return sigma, omega


def get_sigma_exchange(gw, mo_coeff_full=None, mo_occ_full=None):
    """Get exchange self-energy (EXX).

    Parameters
    ----------
    gw : KRGWAC
        gw object
    mo_coeff : complex ndarray, optional
        orbital coefficient, by default None
    mo_occ : double ndarray, optional
        occupation number, by default None

    Returns
    -------
    vk : complex ndarray
        exchange self-energy
    """
    nmo = gw.nmo
    nkpts = gw.nkpts
    kpts = gw.kpts
    mydf = gw.with_df

    if mo_coeff_full is None:
        mo_coeff_full = gw.mo_coeff
    if mo_occ_full is None:
        mo_occ_full = gw.mo_occ
    mo_coeff_full = cp.asarray(mo_coeff_full)
    mo_occ_full = cp.asarray(mo_occ_full)
    nao = mo_coeff_full[0].shape[0]
    nmo_full = nao
    nocc_full = int(np.sum(gw._scf.mo_occ[0])) // 2

    # possible kpts shift center
    kscaled = gw.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]
    # momentum conservation of every kL at once: kconserv_table[kL, ki] = kj with -ki+kj+kL=G
    kconserv_table = get_kconserv_ria_efficient(gw.mol, kpts - kpts[0])

    vk = cp.zeros(shape=[nkpts, nmo_full, nmo_full], dtype=cp.complex128)

    for kL in range(nkpts):
        # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
        # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
        kidx = kconserv_table[kL]
        kidx_r = np.argsort(kidx)
        # every (ki, kidx[ki]) conserves momentum with kL
        dk = -kscaled + kscaled[kidx] + kscaled[kL]
        assert np.linalg.norm(np.round(dk) - dk, axis=1).max() < 1e-12

        # Read (L|pq) for every ki and kj that conserves with kL and ki (-ki+kj+kL=G).
        # GDF only stores one of the two conjugated momentum transfers kL and -kL,
        # (L|ki kj) is either Lpq_kL[ki] or Lpq_kL[kj].conj().transpose(0, 2, 1).
        for kp, _, ki_idx, kj_idx in kk_adapted_iter(mydf.kmesh):
            is_conj = (kidx[ki_idx] != kj_idx).any()
            if not is_conj or (kidx[kj_idx] == ki_idx).all():
                break
        else:
            raise RuntimeError(f'Momentum transfer of kL = {kL} is not found in GDF')
        naux = mydf._cderi[kp].shape[0]
        # a block uses at most 40% of the free GPU memory, with the two buffers and the ao2mo intermediates
        blksize = max(1, min(naux, int(0.4 * cupy_helper.get_avail_mem() / (112 * nkpts * nao**2))))
        aux_iter = [(kp, b0, b1) for b0, b1 in lib.prange(0, naux, blksize)]
        # one output and one unpacking buffer for all blocks, so each block overwrites the previous one
        out_buf = cp.empty(nkpts * blksize * nao**2, dtype=cp.complex128)
        unpack_buf = cp.empty(nkpts * blksize * nao**2, dtype=cp.complex128)
        # slice over naux
        for _, Lpq_kL, sign in mydf.loop(blksize, kpts=kpts, aux_iter=aux_iter, buf=unpack_buf, out=out_buf):
            if sign < 0:
                continue  # 2D cells (dimension=2): the negative part of the cderi is not used, as before

            # kn is i, km = kidx_r[kn] is j that conserves with kn and kL (-km+kn+kL=G), every kn at once
            # ao2mo transform (L|pq) to (L|ij)
            if is_conj:
                Lpq_ao = Lpq_kL.conj().transpose(0, 1, 3, 2)
            else:
                Lpq_ao = Lpq_kL[kidx_r]
            Lpq_ao = Lpq_ao.astype(cp.complex128, copy=False)

            if hasattr(gw._scf, 'sigma'):
                # This computes Lij = einsum('kmi,kPmn,knj->kPij', mo_coeff_full[kidx_r].conj(), Lpq_ao, mo_coeff_full)
                tmp = contract('kPmn,kmi->kPin', Lpq_ao, mo_coeff_full[kidx_r].conj())
                Lij = contract('kPin,knj->kPij', tmp, mo_coeff_full)
                tmp = None

                # vk[k] -= 1.0 / nkpts * einsum('i,Lip,Liq->pq', mo_occ[km], Lij[km].conj(), Lij[km]) * 0.5
                Lij_occ = Lij * mo_occ_full[kidx_r][:, None, :, None]
                contract('kPip,kPiq->kpq', Lij.conj(), Lij_occ, alpha=-0.5 / nkpts, beta=1.0, out=vk)
                Lij_occ = None
            else:
                # This computes Lij = einsum('kmi,kPmn,knj->kPij', mo_coeff_full[kidx_r][:, :, :nocc_full].conj(), Lpq_ao,
                # mo_coeff_full)
                tmp = contract('kPmn,kmi->kPin', Lpq_ao, mo_coeff_full[kidx_r][:, :, :nocc_full].conj())
                Lij = contract('kPin,knj->kPij', tmp, mo_coeff_full)
                tmp = None

                # vk[k] -= 1.0 / nkpts * einsum('Lip,Liq->pq', Lij[km].conj(), Lij[km])
                contract('kPip,kPiq->kpq', Lij.conj(), Lij, alpha=-1.0 / nkpts, beta=1.0, out=vk)
            Lij = None
        Lpq_kL = Lpq_ao = out_buf = unpack_buf = None

    if nmo != nmo_full:
        frozen_mask = get_frozen_mask(gw)
        identity = cp.eye(nmo_full, dtype=cp.complex128)
        vk_frz = cp.zeros(shape=[nkpts, nmo, nmo], dtype=cp.complex128)
        for k in range(nkpts):
            mask = cp.asarray(frozen_mask[k])
            vk_frz[k] = identity[mask, :] @ vk[k] @ identity[:, mask]
        vk = vk_frz

    return vk


def make_gf(gw, omega, eta):
    """Get dynamical Green's function and self-energy.

    Parameters
    ----------
    gw : KRGWAC
        GW object, provides attributes: orbs, kptlist, ef, ac_coeff, omega_fit, vk, vxc, _scf.mo_energy
    omega : double or complex array
        frequency grids
    eta : double
        broadening parameter

    Returns
    -------
    gf : complex ndarray
        GW Green's function
    gf0 : complex ndarray
        mean-field Green's function
    sigma : complex ndarray
        GW correlation self-energy
    """
    assert gw.frozen is None or gw.frozen == 0

    if eta is None:
        eta = gw.eta

    nomega = len(omega)
    sigma = np.zeros(shape=[gw.nkpts, gw.nmo, gw.nmo, nomega], dtype=np.complex128)
    if gw.fullsigma:
        for ik, k in enumerate(gw.kptlist):
            for ip, p in enumerate(gw.orbs_frz):
                for iq, q in enumerate(gw.orbs_frz):
                    sigma[k, p, q] = gw.acobj[ik, ip, iq].ac_eval(omega + 1j * eta) + gw.vk[k, p, q] - gw.vxc[k, p, q]
    else:
        for ik, k in enumerate(gw.kptlist):
            for ip, p in enumerate(gw.orbs_frz):
                sigma[k, p, p] = gw.acobj[ik, ip].ac_eval(omega + 1j * eta) + gw.vk[k, p, p] - gw.vxc[k, p, p]

    gf0 = get_g0_k(omega, cp.asnumpy(gw._scf.mo_energy), eta)
    gf = np.zeros_like(gf0)
    for k in range(gw.nkpts):
        for iw in range(nomega):
            gf[k, :, :, iw] = np.linalg.inv(np.linalg.inv(gf0[k, :, :, iw]) - sigma[k, :, :, iw])

    return gf, gf0, sigma


def make_rdm1_linear(gw, ao_repr=False):
    """Get GW density matrix from Green's function G(it=0).
    G is from linear Dyson equation, which conserves particle number
    G = G0 + G0 Sigma G0
    See equation 16 in 10.1021/acs.jctc.0c01264

    Parameters
    ----------
    gw : KRGWAC
        GW object, provides attributes: sigmaI, mol, _scf, freqs, wts, frozen, orbs, fc
    ao_repr : bool, optional
        return density matrix in AO, by default False

    Returns
    -------
    rdm1 : double ndarray
        density matrix
    """
    assert gw.sigmaI is not None
    assert gw.rdm is True and gw.fullsigma is True
    assert gw.frozen is None or gw.frozen == 0
    sigmaI = gw.sigmaI[:, :, :, 1:]
    freqs = 1j * gw.freqs
    wts = gw.wts
    nmo = gw.nmo
    nkpts = gw.nkpts
    if len(gw.orbs) != nmo:
        sigma = np.zeros(shape=[nkpts, nmo, nmo, len(freqs)], dtype=sigmaI.dtype)
        for k in range(nkpts):
            for ia, a in enumerate(gw.orbs):
                for ib, b in enumerate(gw.orbs):
                    sigma[k, a, b, :] = sigmaI[k, ia, ib, :]
    else:
        # copy, so that adding vk - vxc below does not change gw.sigmaI
        sigma = np.array(sigmaI, copy=True)

    for iw in range(len(freqs)):
        sigma[:, :, :, iw] += gw.vk - gw.vxc
    gf0 = get_g0_k(freqs, cp.asnumpy(gw._scf.mo_energy) - gw.ef, eta=0)
    gf = np.array(gf0, copy=True)
    for k in range(nkpts):
        for iw in range(len(freqs)):
            # G = G0 + G0 Sigma G0
            gf[k, :, :, iw] += gf0[k, :, :, iw] @ sigma[k, :, :, iw] @ gf0[k, :, :, iw]

    # GW density matrix
    rdm1 = np.zeros(shape=[nkpts, nmo, nmo], dtype=np.double)
    for k in range(nkpts):
        rdm1[k] = 2.0 / np.pi * np.einsum('ijw,w->ij', gf[k], wts).real + np.eye(nmo)
        logger.info(gw, 'GW particle number @ k%d = %s', k, np.trace(rdm1[k]))

    # Symmetrize density matrix
    for k in range(nkpts):
        rdm1[k] = 0.5 * (rdm1[k] + rdm1[k].T)

    if ao_repr is True:
        ovlp = cp.asnumpy(gw._scf.get_ovlp())
        mo_coeff = cp.asnumpy(gw._scf.mo_coeff)
        for k in range(nkpts):
            CS = ovlp[k] @ mo_coeff[k]
            rdm1[k] = CS @ rdm1[k] @ CS.conj().T

    return rdm1


def _mo_energy_frozen(gw, mo_energy):
    """Get non-frozen orbital energy.

    Parameters
    ----------
    gw : KRGWAC
        GW object, provides attributes: frozen, nmo, nkpt
    mo_energy : double ndarray
        full orbital energy

    Returns
    -------
    mo_energy_frozen : double ndarray
        non-frozen orbital energy
    """
    frozen_mask = get_frozen_mask(gw)
    nmo = gw.nmo
    nkpts = gw.nkpts
    mo_energy_frozen = cp.zeros(shape=[nkpts, nmo], dtype=cp.double)
    for k in range(nkpts):
        mo_energy_frozen[k] = cp.asarray(mo_energy[k])[frozen_mask[k]]
    return mo_energy_frozen


def _mo_frozen(gw, mo):
    """Get non-frozen orbital coefficient.

    Parameters
    ----------
    gw : KRGWAC
        GW object, provides attributes: frozen, nmo, nkpt
    mo : complex ndarray
        full orbital coefficient

    Returns
    -------
    mo_frozen : complex ndarray
        non-frozen orbital coefficient
    """
    frozen_mask = get_frozen_mask(gw)
    nmo = gw.nmo
    nkpts = gw.nkpts
    nao = mo[0].shape[0]
    mo_frozen = cp.zeros(shape=[nkpts, nao, nmo], dtype=cp.complex128)
    for k in range(nkpts):
        mo_frozen[k] = cp.asarray(mo[k])[:, frozen_mask[k]]
    return mo_frozen


def _mo_occ_frozen(gw, mo_occ):
    """Get non-frozen occupation number.

    Parameters
    ----------
    gw : KRGWAC
        GW object, provides attributes: frozen, nmo, nkpt
    mo_occ : double ndarray
        full occupation number

    Returns
    -------
    mo_occ_frozen : double ndarray
        non-frozen occupation number
    """
    frozen_mask = get_frozen_mask(gw)
    nmo = gw.nmo
    nkpts = gw.nkpts
    mo_occ_frozen = cp.zeros(shape=[nkpts, nmo], dtype=cp.double)
    for k in range(nkpts):
        mo_occ_frozen[k] = cp.asarray(mo_occ[k])[frozen_mask[k]]
    return mo_occ_frozen


def set_frozen_orbs(gw):
    """Set .frozen attribute from frozen mask.

    Parameters
    ----------
    gw : KRGWAC
        unrestricted GW object
    """
    if gw.frozen is not None:
        if gw.orbs is not None:
            if isinstance(gw.frozen, (int, np.int64)):
                # frozen core
                gw.orbs_frz = [x - gw.frozen for x in gw.orbs]
            else:
                # frozen list
                assert isinstance(gw.frozen[0], (int, np.int64))
                gw.orbs_frz = []
                for orbi in gw.orbs:
                    count = len([p for p in gw.frozen if p <= orbi])
                    gw.orbs_frz.append(orbi - count)
            if any(np.array(gw.orbs_frz) < 0):
                raise RuntimeError('GW orbs must be larger than frozen core!')
        else:
            gw.orbs_frz = range(gw.nmo)
            gw.orbs = range(len(gw._scf.mo_energy[0]))
            if isinstance(gw.frozen, (int, np.int64)):
                gw.orbs = list(set(gw.orbs) - set(range(gw.frozen)))
            else:
                assert isinstance(gw.frozen[0], (int, np.int64))
                gw.orbs = list(set(gw.orbs) - set(gw.frozen))
    else:
        if gw.orbs is None:
            gw.orbs = range(len(gw._scf.mo_energy[0]))
        gw.orbs_frz = gw.orbs
    return


class KRGWAC(lib.StreamObject):
    def __init__(self, mf, frozen=None, auxbasis=None):
        self.mol = mf.mol  # mol object
        self._scf = mf  # mean-field object
        self.verbose = self.mol.verbose  # verbose level
        self.stdout = self.mol.stdout  # standard output
        self.max_memory = mf.max_memory  # max memory in MB

        # options
        self.frozen = frozen  # frozen orbital option
        self.auxbasis = auxbasis  # auxiliary basis for DF, used if mf has no GDF
        self.orbs = None  # list of orbital index in full nmo
        self.orbs_frz = None  # list of orbital index in non-frozen nmo
        self.kptlist = None  # list of k-points to evaluate
        self.fullsigma = False  # calculate off-diagonal self-energy
        self.rdm = False  # calculate GW density matrix
        self.vhf_df = False  # use density-fitting for exchange self-energy
        self.fc = True  # finite-size correction to self-energy
        self.fc_grid = False  # grids for finite-size correction to self-energy
        self.outcore = False  # low-memory routine to calculate self-energy
        self.eta = 5.0e-3  # broadening parameter
        self.nw = 100  # number of grids for integration
        self.ac = 'pade'  # analytical continuation method
        self.ac_iw_cutoff = 5.0  # imaginary frequency cutting for fitting self-energy
        self.ac_pade_npts = 18  # number of selected points for Pade approximation
        self.ac_pade_step_ratio = 2.0 / 3.0  # final/initial step size for Pade approximation
        self.qpe_max_iter = 100  # max iteration in iteratively solving quasiparticle equation
        self.qpe_tol = 1.0e-6  # tolerance in Newton method for iteratively quasiparticle equation
        self.qpe_linearized = False  # use linearized quasiparticle equation
        self.qpe_linearized_range = [0.5, 1.5]  # Z-shot factor range, if not in this range, z=1
        self.writefile = 0  # write file level

        # DF-KGW must use GDF integrals, see initialize_df
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            raise NotImplementedError
        self._keys.update(['with_df'])

        ##################################################
        # don't modify the following attributes, they are not input options
        self._nocc = None  # number of NON-FROZEN occupied orbitals
        self._nmo = None  # number of NON-FROZEN orbitals
        self.kpts = mf.kpts  # k-point list
        self.nkpts = len(self.kpts)  # number of k-points
        self.mo_energy = None  # orbital energy
        self.mo_coeff = None  # orbital coefficient
        self.mo_occ = None  # occupiation number

        # results
        self.vk = None  # exchange matrix in MO
        self.vxc = None  # mean-field exchange-correlation matrix in MO
        self.freqs = None  # frequency grids
        self.wts = None  # weights of frequency grids
        self.ef = None  # Fermi level
        self.acobj = None  # analytical continuation object
        self.ac_coeff = None  # Pade fitting coefficient, old interface, to be deprecated
        self.omega_fit = None  # AC fitting frequency, old interface, to be deprecated
        self.sigmaI = None  # self-energy in the imaginary axis

        return

    def dump_flags(self, verbose=None):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        log.info('GW nocc = %d, nvir = %d, nkpts = %d', nocc, nvir, nkpts)
        if self.frozen is not None:
            log.info('frozen orbitals = %s', str(self.frozen))
        if self.kptlist is not None:
            log.info('k-point list = %s', str(self.kptlist))
        if self.orbs is not None:
            log.info('orbital list = %s', str(self.orbs))
        log.info('off-diagonal self-energy = %s', self.fullsigma)
        log.info('GW density matrix = %s', self.rdm)
        log.info('density-fitting for exchange = %s', self.vhf_df)
        log.info('outcore for self-energy= %s', self.outcore)
        log.info('finite size corrections = %s', self.fc)
        if self.fc_grid is not None:
            log.info('grids for finite size corrections = %s', self.fc_grid)
        log.info('broadening parameter = %.3e', self.eta)
        log.info('number of grids = %d', self.nw)
        log.info('analytic continuation method = %s', self.ac)
        log.info('imaginary frequency cutoff = %s', str(self.ac_iw_cutoff))
        if self.ac == 'pade':
            log.info('Pade points = %d', self.ac_pade_npts)
            log.info('Pade step ratio = %.3f', self.ac_pade_step_ratio)
        log.info('use perturbative linearized QP eqn = %s', self.qpe_linearized)
        if self.qpe_linearized is True:
            log.info('linearized factor range = %s', self.qpe_linearized_range)
        else:
            log.info('QPE max iter = %d', self.qpe_max_iter)
            log.info('QPE tolerance = %.1e', self.qpe_tol)
        log.info('')
        return

    @property
    def nocc(self):
        frozen_mask = get_frozen_mask(self)
        nkpts = len(self._scf.mo_energy)
        nelec = 0.0
        for k in range(nkpts):
            nelec += np.sum(self._scf.mo_occ[k][frozen_mask[k]])
        nelec = int(nelec / nkpts)
        return nelec // 2

    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        frozen_mask = get_frozen_mask(self)
        return len(self._scf.mo_energy[0][frozen_mask[0]])

    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    def kernel(self, orbs=None, kptlist=None):
        """Run a G0W0 calculation.

        Parameters
        ----------
        orbs : list, optional
            orbital list to calculate self-energy, by default None
        kptlist : list, optional
            k-point list to calculate self-energy, by default None
        """
        # host copies, pyscf.pbc.mp.kmp2.get_frozen_mask reads them with np.asarray
        if self.mo_energy is None:
            self.mo_energy = np.array(cp.asnumpy(self._scf.mo_energy), copy=True)
        if self.mo_coeff is None:
            self.mo_coeff = np.array(cp.asnumpy(self._scf.mo_coeff), copy=True)
        if self.mo_occ is None:
            self.mo_occ = np.array(cp.asnumpy(self._scf.mo_occ), copy=True)

        self.orbs = orbs
        self.kptlist = kptlist

        if hasattr(self._scf, "sigma"):
            self.nw = max(400, self.nw)
            self.ac_pade_npts = 18
            self.ac_pade_step_ratio = 5.0 / 6.0
            self.fc = False

        self.initialize_df(auxbasis=self.auxbasis)

        nmo = self.nmo
        naux = self.with_df.get_naoaux()
        nkpts = self.nkpts
        if self.outcore is False:
            mem_incore = (2 * nkpts * nmo**2 * naux) * 16 / 1e6
            mem_avail = cupy_helper.get_avail_mem() / 1e6
            if mem_incore > 0.99 * mem_avail:
                logger.warn(self, 'GPU memory may not be enough for incore, try outcore=True!')
                raise NotImplementedError

        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        kernel(self)
        logger.timer(self, 'KRGWAC', *cput0)
        return

    def initialize_df(self, auxbasis=None):
        """Initialize density fitting.

        Parameters
        ----------
        auxbasis : str, optional
            name of auxiliary basis set, by default None
        """
        # KRGWAC must use GDF integrals. A mean field without density fitting carries the FFTDF that
        # gpu4pyscf KRHF/KRKS create by default, build a k-point GDF in that case.
        if isinstance(getattr(self._scf, 'with_df', None), GDF):
            self.with_df = self._scf.with_df
        else:
            self.with_df = GDF(self._scf.mol, kpts=self.kpts)
            if auxbasis is not None:
                self.with_df.auxbasis = auxbasis
            else:
                try:
                    self.with_df.auxbasis = make_auxbasis(self._scf.mol, mp2fit=True)
                except RuntimeError:
                    self.with_df.auxbasis = make_auxbasis(self._scf.mol, mp2fit=False)
        if getattr(self.with_df, "_j_only", False):
            self.with_df._j_only = False
            self.with_df.reset()
        if self.with_df._cderi is None:
            self.with_df.build(j_only=False)
        self._keys.update(['with_df'])
        return

    set_frozen_orbs = set_frozen_orbs
    make_rdm1 = make_rdm1_linear
    make_gf = make_gf
    get_sigma_exchange = get_sigma_exchange
