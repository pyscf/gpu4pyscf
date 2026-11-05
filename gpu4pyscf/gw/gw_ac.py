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
Spin-restricted G0W0 method based on the analytic continuation scheme, on one GPU.
This implementation has N^4 scaling.
GW-AC is recommended for valence states only, and is inaccurate for core states.

References:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    New J. Phys. 14 053020 (2012)
"""


import cupy as cp
import h5py
import numpy as np
from scipy.optimize import newton

from pyscf import lib as lib_cpu
from pyscf.gw.gw_ac import get_g0
from pyscf.gw.utils.ac_grid import PadeAC, TwoPoleAC, _get_scaled_legendre_roots
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask
from pyscf.scf.addons import _fermi_smearing_occ, _gaussian_smearing_occ, _smearing_optimize
from pyscf.scf.hf import KohnShamDFT

from gpu4pyscf import scf
from gpu4pyscf.df import DF, make_auxbasis
from gpu4pyscf.lib import cupy_helper, logger, utils
from gpu4pyscf.lib.cutensor import contract
from gpu4pyscf.scf import hf as gpu_hf
from gpu4pyscf.gw.utils.gw_cp_helper import mkslice, addto_diagonal, get_id_minus_pi_inv

# fuse functions for GPU
@cp.fuse()
def _Pia(Lia, eia, omega):
    return Lia * (4.0 * eia / (omega * omega + eia * eia))


@cp.fuse()
def _Xia(Lia, eia, omega):
    # Pi = -X X^T with X = Lia * sqrt(-4 eia / (omega^2 + eia^2)); eia < 0 for occupied-virtual pairs
    return Lia * cp.sqrt(-4.0 * eia / (omega * omega + eia * eia))


@cp.fuse()
def _Pia_metal(Lpq, eia, fia, omega):
    return Lpq * (eia * fia / (omega * omega + eia * eia))


@cp.fuse()
def _Xia_fia(Lia, eia, fia, omega):
    # Pi = -X X^T with this X; needs eia * fia <= 0 for every pair (occupations non-increasing with orbital energy)
    return Lia * cp.sqrt(-4.0 * eia * fia / (omega * omega + eia * eia))


@cp.fuse()
def _g0(emo, wts, freq):
    return wts * emo / (emo * emo + freq * freq)


@cp.fuse()
def _Qmn(Qmn, l_slice):
    return Qmn * l_slice


def kernel(gw):
    # local variables for convenience
    mf = gw._scf
    nocc = gw.nocc

    # set frozen orbitals
    set_frozen_orbs(gw)
    orbs = gw.orbs
    orbs_frz = gw.orbs_frz

    # get non-frozen quantities
    mo_energy_frz = _mo_energy_without_core(gw, gw.mo_energy)
    mo_coeff_frz = _mo_without_core(gw, gw.mo_coeff)

    if gw.Lpq is None and (not gw.outcore):
        with lib_cpu.temporary_env(gw.with_df, verbose=0), lib_cpu.temporary_env(gw.mol, verbose=0):
            gw.Lpq = gw.ao2mo(mo_coeff_frz)
    Lpq = gw.Lpq

    # mean-field exchange-correlation
    with lib_cpu.temporary_env(mf, verbose=0), lib_cpu.temporary_env(mf.mol, verbose=0):
        dm = mf.make_rdm1()
        v_mf = mo_coeff_frz.T @ (mf.get_veff(mf.mol, dm) - mf.get_j(mf.mol, dm)) @ mo_coeff_frz
    gw.vxc = v_mf

    # exchange self-energy
    if gw.vhf_df and gw.outcore is False:
        # TODO: support smearing
        vk = -contract('Lpi,Liq->pq', Lpq[:, :, :nocc], Lpq[:, :nocc, :])
    else:
        vk = gw.get_sigma_exchange(mo_coeff=mo_coeff_frz)
    gw.vk = vk

    # set up Fermi level
    gw.ef = ef = gw.get_ef(mo_energy=mf.mo_energy)

    # grids for integration on imaginary axis
    quad_freqs, quad_wts = _get_scaled_legendre_roots(gw.nw)
    eval_freqs_with_zero = gw.setup_evaluation_grid(fallback_freqs=quad_freqs, fallback_wts=quad_wts)
    quad_freqs = cp.asarray(quad_freqs)
    quad_wts = cp.asarray(quad_wts)
    eval_freqs_with_zero = cp.asarray(eval_freqs_with_zero)

    # compute self-energy on imaginary axis
    if gw.outcore:
        sigmaI, omega = get_sigma_outcore(
            gw, orbs_frz, quad_freqs, quad_wts, ef, mo_energy_frz, mo_coeff_frz, iw_cutoff=gw.ac_iw_cutoff,
            eval_freqs=eval_freqs_with_zero, fullsigma=gw.fullsigma
        )
    else:
        sigmaI, omega = get_sigma(
            gw, orbs_frz, Lpq, quad_freqs, quad_wts, ef, mo_energy_frz, iw_cutoff=gw.ac_iw_cutoff,
            eval_freqs=eval_freqs_with_zero, fullsigma=gw.fullsigma
        )

    # begin working on the host
    omega = cp.asnumpy(omega)
    gw.vk = vk = cp.asnumpy(vk)
    gw.vxc = v_mf = cp.asnumpy(v_mf)
    # note that sigmaI and gw.sigmaI can be different
    sigmaI = cp.asnumpy(sigmaI)
    gw.sigmaI = cp.asnumpy(gw.sigmaI)
    gw.mo_energy = cp.asnumpy(gw.mo_energy)
    gw.mo_coeff = cp.asnumpy(gw.mo_coeff)
    gw.mo_occ = cp.asnumpy(gw.mo_occ)

    # analytic continuation
    if gw.ac == 'twopole':
        acobj = TwoPoleAC(orbs_frz, nocc)
    elif gw.ac == 'pade':
        acobj = PadeAC(npts=gw.ac_pade_npts, step_ratio=gw.ac_pade_step_ratio)
        if gw.ac_idx is not None:
            acobj.idx = gw.ac_idx
    else:
        raise ValueError('Unknown GW-AC type %s' % gw.ac)
    acobj.ac_fit(sigmaI, omega, axis=-1)

    # get GW quasiparticle energy
    if gw.fullsigma:
        diag_acobj = acobj.diagonal(axis1=0, axis2=1)
    else:
        diag_acobj = acobj

    mo_energy_mf = cp.asnumpy(mf.mo_energy)
    mo_energy = np.zeros_like(mo_energy_mf)
    for ip, (p_in_frz, p_in_all) in enumerate(zip(orbs_frz, orbs)):
        if gw.qpe_linearized is True:
            # linearized G0W0
            de = 1e-6
            sigmaR = diag_acobj[ip].ac_eval(mo_energy_mf[p_in_all]).real
            dsigma = diag_acobj[ip].ac_eval(mo_energy_mf[p_in_all] + de).real - sigmaR.real
            zn = 1.0 / (1.0 - dsigma / de)
            if gw.qpe_linearized_range is not None:
                zn = 1.0 if zn < gw.qpe_linearized_range[0] or zn > gw.qpe_linearized_range[1] else zn
            mo_energy[p_in_all] = mo_energy_mf[p_in_all] + \
                    zn * (sigmaR.real + vk[p_in_frz, p_in_frz] - v_mf[p_in_frz, p_in_frz])
        else:
            # self-consistently solve QP equation
            def quasiparticle(omega):
                sigmaR = diag_acobj[ip].ac_eval(omega).real
                return omega - mo_energy_mf[p_in_all] - (sigmaR + vk[p_in_frz, p_in_frz] - v_mf[p_in_frz, p_in_frz])

            try:
                mo_energy[p_in_all] = newton(
                    quasiparticle, mo_energy_mf[p_in_all], tol=gw.qpe_tol, maxiter=gw.qpe_max_iter
                )
            except RuntimeError:
                logger.warn(gw, 'QPE for orbital=%d not converged!', p_in_all)

    # save GW results
    gw.acobj = acobj
    gw.mo_energy = mo_energy
    with np.printoptions(threshold=len(mo_energy)):
        logger.debug(gw, '  GW mo_energy =\n%s', mo_energy)
    logger.warn(gw, 'GW QP energies may not be sorted from min to max')

    if gw.writefile > 0:
        with h5py.File('vxc.h5', 'w') as feri:
            feri['vk'] = np.asarray(vk)
            feri['v_mf'] = np.asarray(v_mf)

        with h5py.File('sigma_imag.h5', 'w') as feri:
            feri['sigmaI'] = np.asarray(sigmaI)
            feri['omega'] = np.asarray(omega)

        acobj.save('ac_coeff.h5')

    return


def get_rho_response(omega, mo_energy, Lia, out=None):
    """Compute density-density response function in auxiliary basis at freq iw.
    See equation 58 in 10.1088/1367-2630/14/5/053020,
    and equation 24 in doi.org/10.1021/acs.jctc.0c00704.

    Parameters
    ----------
    omega : double
        imaginary part of a frequency point
    mo_energy : double 1d array
        orbital energy
    Lia : double 3d ndarray
        occ-vir block of three-center density-fitting matrix
    out : double 2d array, optional
        a location into which the result is stored, by default None

    Returns
    -------
    Pi : double 2d array
        density-density response function in auxiliary basis at freq iw
    """
    naux, nocc, nvir = Lia.shape
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]

    # factor 4.0 comes from (1) spin-up and spin-down (2) conjugated term
    # plain version
    # eia = 4.0 * eia / (omega**2 + eia**2)
    # Pia = Lia * eia
    # SYRK version: since eia < 0, Pi = -X X^T with X = Lia * sqrt(-4 eia / (omega^2 + eia^2)).
    # SYRK needs half the flops of the GEMM; it writes the lower triangle, which is then mirrored.
    Xia = _Xia(Lia, eia, omega)
    if out is None:
        out = cp.empty((naux, naux), dtype=cp.double)
    cp.cublas.syrk('N', Xia.reshape(naux, nocc * nvir), out=out, alpha=-1.0, beta=0.0, lower=True)
    Xia = None
    return cupy_helper.hermi_triu(out)


def get_idx_metal(mo_occ, threshold=1.0e-6):
    """Get occupied, fractional, and virtual orbital indexes for metals.

    With Fermi smearing, all partially-filled bands can fall just below 2,
    leaving no strictly occupied orbitals. Treat the leading non-virtual
    block as fractional in that case.

    Parameters
    ----------
    mo_occ : double 1d array
        occupation number
    threshold : double, optional
        threshold to determine fractionally occupied orbitals, by default 1.0e-6

    Returns
    -------
    idx_occ : list
        list of occupied orbital indexes
    idx_frac : list
        list of fractionally occupied orbital indexes
    idx_vir : list
        list of virtual orbital indexes
    """
    mo_occ = cp.asnumpy(mo_occ) if isinstance(mo_occ, cp.ndarray) else np.asarray(mo_occ)
    idx_occ = np.where(mo_occ > 2.0 - threshold)[0]
    idx_vir = np.where(mo_occ < threshold)[0]
    frac_start = idx_occ[-1] + 1 if idx_occ.size else 0
    frac_stop = idx_vir[0] if idx_vir.size else len(mo_occ)
    idx_frac = np.arange(frac_start, frac_stop)

    return idx_occ, idx_frac, idx_vir


def get_rho_response_metal(omega, mo_energy, mo_occ, Lpq, out=None):
    """Get response function in auxiliary basis for metallic systems.

    Parameters
    ----------
    omega : double
        imaginary part of a frequency point
    mo_energy : double 1d array
        orbital energy
    mo_occ : double 1d array
        occupation number with a factor of 2
    Lpq : double 3d array
        three-center density-fitting matrix in MO, only its occupied-and-fractional x fractional-and-virtual block
        is used
    out : double ndarray, optional
        a location into which the result is stored, by default None

    Returns
    -------
    Pi : double 2d array
        density-density response function in auxiliary basis at freq iw
    """
    naux = Lpq.shape[0]
    # eia = mo_energy[:, None] - mo_energy[None, :]
    # fia = mo_occ[:, None] - mo_occ[None, :]
    #
    # # factor 4.0 comes from (1) spin-up and spin-down (2) conjugated term
    # # both ia and ai are included, this gives a factor of 2.0
    # # restricted mo_occ gives another factor of 2.0
    # # plain version
    # # eia = eia * fia / (omega**2 + eia**2)
    # # Pia = Lpq * eia

    # Only pairs with different occupations contribute
    # rows i = occupied and fractional orbitals, columns a = fractional and virtual orbitals
    idx_occ, idx_frac, _ = get_idx_metal(mo_occ, threshold=1e-8)
    nocc = len(idx_occ)
    nfrac = len(idx_frac)
    eia = mo_energy[: nocc + nfrac, None] - mo_energy[None, nocc:]
    fia = (mo_occ[: nocc + nfrac, None] - mo_occ[None, nocc:]) / 2.0
    # Add a factor of 0.5 for double counting
    fia[nocc:, :nfrac] *= 0.5
    # SYRK version: Pi = -X X^T, lower triangle only
    Xia = _Xia_fia(Lpq[:, : nocc + nfrac, nocc:], eia, fia, omega)
    if out is None:
        out = cp.empty((naux, naux), dtype=cp.double)
    cp.cublas.syrk('N', Xia.reshape(naux, -1), out=out, alpha=-1.0, beta=0.0, lower=True)
    Xia = None
    # get full Pi from lower triangle
    return cupy_helper.hermi_triu(out)


def get_sigma(
        gw, orbs, Lpq, quad_freqs, quad_wts, ef, mo_energy, mo_coeff=None, mo_occ=None, iw_cutoff=None, eval_freqs=None,
        mo_energy_w=None, fullsigma=False):
    """Compute GW correlation self-energy on imaginary axis.
    See equation 62 and 62 in 10.1088/1367-2630/14/5/053020,
    and equation 27 in doi.org/10.1021/acs.jctc.0c00704.

    Parameters
    ----------
    gw : GWAC
        GW object
    orbs : list
        list of orbital indexes
    Lpq : double 3d array
        three-center density-fitting matrix in MO space
    quad_freqs : double 1d array
        position of imaginary frequency grids used for integration
    quad_wts : double 1d array
        weight of imaginary frequency grids
    ef : double
        Fermi level
    mo_energy : double 1d array
        orbital energy in G
    mo_coeff : double 2d array
        coefficient from AO to MO
    mo_occ : double 1d array, optional
        occupation number, by default None
    iw_cutoff : double, optional
        imaginary grid cutoff for fitting, by default None
    eval_freqs : double 1d array, optional
        imaginary frequency points at which to evaluate self energy.
        Self energy is evaluated at ef + 1j * eval_freqs.
    mo_energy_w : double 1d array, optional
        orbital energy in W, by default None
    fullsigma : bool, optional
        calculate off-diagonal elements, by default False

    Returns
    -------
    sigma : complex 2d or 3d ndarray
        self-energy on the imaginary axis
    omega : complex 1d array
        imaginary frequency grids of self-energy
    """
    if eval_freqs is None:
        eval_freqs = quad_freqs

    nocc = gw.nocc
    nquadfreqs = len(quad_freqs)
    nevalfreqs = len(eval_freqs)
    naux, nmo, _ = Lpq.shape
    norbs = len(orbs)

    mo_energy_g = mo_energy
    if mo_energy_w is None:
        mo_energy_w = mo_energy

    if mo_coeff is None:
        mo_coeff = _mo_without_core(gw, gw.mo_coeff)

    if mo_occ is None:
        mo_occ = _mo_energy_without_core(gw, gw.mo_occ)

    # integration on numerical grids
    if iw_cutoff is not None and gw.rdm is False:
        nw_sigma = int(sum(eval_freqs < iw_cutoff))
    else:
        nw_sigma = int(nevalfreqs)
    nw_cutoff = nw_sigma if iw_cutoff is None else int(sum(eval_freqs < iw_cutoff))

    omega = ef + 1j * eval_freqs[:nw_sigma]
    emo = omega[None] - mo_energy_g[:, None]

    if fullsigma is False:
        sigma_real = cp.zeros(shape=[nw_sigma, norbs], dtype=cp.double)
        sigma_imag = cp.zeros(shape=[nw_sigma, norbs], dtype=cp.double)
    else:
        sigma_real = cp.zeros(shape=[nw_sigma, norbs, norbs], dtype=cp.double)
        sigma_imag = cp.zeros(shape=[nw_sigma, norbs, norbs], dtype=cp.double)

    # make density-fitting matrix for contractions
    if hasattr(gw._scf, 'sigma') is False:
        Lia = cp.ascontiguousarray(Lpq[:, :nocc, nocc:])
    # assume Lpq = Lpq, so we don't generate Lpq[:, mkslice(orbs), :]
    l_slice = Lpq[:, :, mkslice(orbs)].reshape(naux, -1)

    # self-energy is calculated as equation 27 in doi.org/10.1021/acs.jctc.0c00704
    logger.info(gw, 'Starting get_sigma_diag main loop with %d frequency points.', nquadfreqs)
    Pi = None
    if fullsigma is False:
        Qmn = None
        Wmn = None
        naux_ones = cp.ones(shape=[1, naux], dtype=cp.double)
    else:
        Qmn = cp.zeros(shape=[naux, norbs], dtype=cp.double)
        Wmn = cp.zeros(shape=[nmo, norbs, norbs], dtype=cp.double)

    for w in range(nquadfreqs):
        if gw.verbose >= 4:
            gw.stdout.write(f'{w:4d} ')
            if w % 12 == 11:
                gw.stdout.write('\n')
            gw.stdout.flush()

        # Pi_inv = (I - Pi)^-1 - I.
        if hasattr(gw._scf, 'sigma'):
            Pi = get_rho_response_metal(quad_freqs[w], mo_energy_w, mo_occ, Lpq, out=Pi)
        else:
            Pi = get_rho_response(quad_freqs[w], mo_energy_w, Lia)
        # Pi_inv contains (I - Pi)^-1.
        Pi_inv = get_id_minus_pi_inv(Pi, overwrite_input=True)

        # second line in equation 27
        # Pi_inv now contains (I - Pi)^-1 - I.
        addto_diagonal(Pi_inv, -1.0)

        # plain version
        # g0 = quad_wts[w] * emo / (emo**2 + quad_freqs[w] ** 2)
        # fused version
        g0 = _g0(emo, quad_wts[w], quad_freqs[w])

        # split g0 into real and imag parts to avoid costly type conversions
        g0r = cp.ascontiguousarray(g0.real)
        g0i = cp.ascontiguousarray(g0.imag)

        if fullsigma is False:
            # n is the index of orbitals in orbs, m is the index of orbitals of nmo
            # last line of equation 27, contraction from left to right
            # Qmn = \sum_P V^{nm}_P (Pi_inv)_{PQ} = \sum_P V^{mn}_P (Pi_inv)_{PQ}
            Qmn = cp.matmul(Pi_inv, l_slice, out=Qmn)

            # Qmn = Qmn v^{mn}_Q
            # plain version
            # Qmn *= l_slice
            # fused version
            Qmn = _Qmn(Qmn, l_slice)

            # Wmn = \sum_Q Qmn
            Wmn = cp.matmul(naux_ones, Qmn, out=Wmn)

            # sigma -= einsum('mn, mw -> nw', Wmn, g0) / np.pi
            # CPU version using scipy BLAS
            # sla.blas.dgemm(
            #     alpha=-1.0 / np.pi,
            #     a=Wmn.reshape(nmo, norbs).T,
            #     b=g0r.T,
            #     c=sigma_real.T,
            #     trans_a=0,
            #     trans_b=1,
            #     beta=1.0,
            #     overwrite_c=True,
            # )
            cp.cublas.gemm('N', 'T', Wmn.reshape(nmo, norbs).T, g0r.T, out=sigma_real.T, alpha=-1.0 / np.pi, beta=1.0)
            # CPU version using scipy BLAS
            # sla.blas.dgemm(
            #     alpha=-1.0 / np.pi,
            #     a=Wmn.reshape(nmo, norbs).T,
            #     b=g0i.T,
            #     c=sigma_imag.T,
            #     trans_a=0,
            #     trans_b=1,
            #     beta=1.0,
            #     overwrite_c=True,
            # )
            cp.cublas.gemm('N', 'T', Wmn.reshape(nmo, norbs).T, g0i.T, out=sigma_imag.T, alpha=-1.0 / np.pi, beta=1.0)
        else:
            # n and n' are the index of orbitals in orbs, m is the index of orbitals of nmo
            # last line of equation 27, contraction from left to right
            # Qmn = \sum_P V^{nm}_P (PiV)_{PQ} = \sum_P V^{mn}_P (PiV)_{PQ}
            for orbm in range(nmo):
                cp.matmul(Pi_inv, l_slice.reshape(naux, nmo, norbs)[:, orbm, :], out=Qmn)

                # Wmn is actually Wmnn'
                # Wmnn' = \sum_Q Qmn v^{mn'}_Q
                cp.matmul(Qmn.T, l_slice.reshape(naux, nmo, norbs)[:, orbm, :], out=Wmn[orbm])

            # sigma -= einsum('mnl,mw->wnl', Wmn, g0)/np.pi
            # CPU version using scipy BLAS
            # sla.blas.dgemm(
            #     alpha=-1.0 / np.pi,
            #     a=Wmn.reshape(nmo, norbs * norbs).T,
            #     b=g0r.T,
            #     c=sigma_real.reshape(nw_sigma, norbs * norbs).T,
            #     trans_a=0,
            #     trans_b=1,
            #     beta=1.0,
            #     overwrite_c=True,
            # )
            cp.cublas.gemm(
                'N',
                'T',
                Wmn.reshape(nmo, norbs * norbs).T,
                g0r.T,
                out=sigma_real.reshape(nw_sigma, norbs * norbs).T,
                alpha=-1.0 / np.pi,
                beta=1.0,
            )
            # CPU version using scipy BLAS
            # sla.blas.dgemm(
            #     alpha=-1.0 / np.pi,
            #     a=Wmn.reshape(nmo, norbs * norbs).T,
            #     b=g0i.T,
            #     c=sigma_imag.reshape(nw_sigma, norbs * norbs).T,
            #     trans_a=0,
            #     trans_b=1,
            #     beta=1.0,
            #     overwrite_c=True,
            # )
            cp.cublas.gemm(
                'N',
                'T',
                Wmn.reshape(nmo, norbs * norbs).T,
                g0i.T,
                out=sigma_imag.reshape(nw_sigma, norbs * norbs).T,
                alpha=-1.0 / np.pi,
                beta=1.0,
            )

    sigma = sigma_real + 1.0j * sigma_imag
    if fullsigma is False:
        sigma = cp.ascontiguousarray(sigma.transpose(1, 0))
    else:
        sigma = cp.ascontiguousarray(sigma.transpose(1, 2, 0))

    logger.info(gw, '\nFinished get_sigma_diag main loop.')

    if gw.rdm is True:
        gw.sigmaI = sigma

    return sigma[..., :nw_cutoff], omega[:nw_cutoff]


def get_sigma_outcore(gw, orbs, quad_freqs, quad_wts, ef, mo_energy, mo_coeff, mo_occ=None,
                      iw_cutoff=None, eval_freqs=None, mo_energy_w=None, fullsigma=False):
    """Low-memory routine to compute GW correlation self-energy on imaginary axis.
    See equation 62 and 62 in 10.1088/1367-2630/14/5/053020,
    and equation 27 in doi.org/10.1021/acs.jctc.0c00704.

    Parameters
    ----------
    gw : GWAC
        GW object
    orbs : list
        list of orbital indexes
    quad_freqs : double 1d array
        position of imaginary frequency grids used for integration
    quad_wts : double 1d array
        weight of imaginary frequency grids
    ef : double
        Fermi level
    mo_energy : double 1d array
        orbital energy in G
    mo_coeff : double 2d array
        coefficient from AO to MO
    mo_occ : double 2d array, optional
        occupation number, by default None
    iw_cutoff : double, optional
        imaginary grid cutoff for fitting, by default None
    eval_freqs : double 1d array, optional
        imaginary frequency points at which to evaluate self energy.
        Self energy is evaluated at ef + 1j * eval_freqs.
    mo_energy_w : double array, optional
        orbital energy in W, by default None
    fullsigma : bool, optional
        calculate off-diagonal elements, by default False

    Returns
    -------
    sigma : complex 2d or 3d array
        self-energy on the imaginary axis
    omega : complex 1d array
        imaginary frequency grids of self-energy
    """
    if eval_freqs is None:
        eval_freqs = quad_freqs

    nocc = gw.nocc
    nmo = gw.nmo
    nw = len(quad_freqs)
    nw2 = len(eval_freqs)
    norbs = len(orbs)
    naux = gw._naux

    mo_energy_g = mo_energy
    if mo_energy_w is None:
        mo_energy_w = mo_energy

    if mo_occ is None:
        mo_occ = _mo_energy_without_core(gw, gw.mo_occ)

    # integration on numerical grids
    if iw_cutoff is not None and gw.rdm is False:
        nw_sigma = int(sum(eval_freqs < iw_cutoff))
    else:
        nw_sigma = nw2
    nw_cutoff = nw_sigma if iw_cutoff is None else int(sum(eval_freqs < iw_cutoff))
    omega = ef + 1j * eval_freqs[:nw_sigma]

    if hasattr(gw._scf, 'sigma'):
        # Only pairs with different occupations contribute, as in get_rho_response_metal:
        # rows i = occupied and fractional orbitals, columns a = fractional and virtual orbitals
        idx_occ, idx_frac, _ = get_idx_metal(mo_occ, threshold=1e-8)
        nocc_metal = len(idx_occ)
        nfrac = len(idx_frac)

    # nseg for response function is determined by the occupied and fractional orbitals (metal) or nocc
    if hasattr(gw._scf, 'sigma'):
        # nseg = int((nmo + gw.segsize - 1) // gw.segsize)
        nseg = int((nocc_metal + nfrac + gw.segsize - 1) // gw.segsize)
    else:
        nseg = int((nocc + gw.segsize - 1) // gw.segsize)

    Pi = cp.zeros(shape=[nw, naux, naux], dtype=cp.double)
    for i in range(nseg):
        if hasattr(gw._scf, 'sigma'):
            orb_start = i * gw.segsize
            # orb_end = min((i + 1) * gw.segsize, nmo)
            # ijslice = (orb_start, orb_end, 0, nmo)
            orb_end = min((i + 1) * gw.segsize, nocc_metal + nfrac)
            ijslice = (orb_start, orb_end, nocc_metal, nmo)
        else:
            orb_start = i * gw.segsize
            orb_end = min((i + 1) * gw.segsize, nocc)
            ijslice = (orb_start, orb_end, nocc, nmo)
        Lia = gw.ao2mo(mo_coeff=mo_coeff, ijslice=ijslice)

        if hasattr(gw._scf, 'sigma'):
            eia = mo_energy_w[orb_start:orb_end, None] - mo_energy_w[None, nocc_metal:]
            fia = (mo_occ[orb_start:orb_end, None] - mo_occ[None, nocc_metal:]) / 2.0
            # Add a factor of 0.5 for double counting: rows and columns that are both fractional
            if orb_end > nocc_metal:
                fia[max(nocc_metal - orb_start, 0):, :nfrac] *= 0.5

        for w in range(nw):
            if hasattr(gw._scf, 'sigma'):
                # eia = mo_energy_w[orb_start:orb_end, None] - mo_energy_w[None, :]
                # fia = mo_occ[orb_start:orb_end, None] - mo_occ[None, :]
                # # plain version
                # # eia = eia * fia / (quad_freqs[w] ** 2 + eia**2)
                # # Pia = Lia * eia
                # # fused version
                # Pia = _Pia_metal(Lia, eia, fia, quad_freqs[w])
                # SYRK version: Pi[w] -= X X^T, accumulated in the lower triangle
                Xia = _Xia_fia(Lia, eia, fia, quad_freqs[w])
                cp.cublas.syrk('N', Xia.reshape(naux, -1), out=Pi[w], alpha=-1.0, beta=1.0, lower=True)
                del Xia
            else:
                eia = mo_energy_w[orb_start:orb_end, None] - mo_energy_w[None, nocc:]
                # plain version
                # eia = 4.0 * eia / (quad_freqs[w] ** 2 + eia**2)
                # Pia = Lia * eia
                # fused version
                Pia = _Pia(Lia, eia, quad_freqs[w])
                #Pi[w] += np.matmul(Pia.reshape(naux, -1), Lia.reshape(naux, -1).T)
                # CPU version using scipy BLAS
                # sla.blas.dgemm(
                #     alpha=1.0,
                #     a=Pia.reshape(naux, -1).T,
                #     b=Lia.reshape(naux, -1).T,
                #     c=Pi[w].T, # Pi is symmetric
                #     beta=1.0,
                #     trans_a=1,
                #     trans_b=0,
                #     overwrite_c=True,
                # )
                cp.cublas.gemm(
                    'T', 'N', Pia.reshape(naux, -1).T, Lia.reshape(naux, -1).T, out=Pi[w].T, alpha=1.0, beta=1.0
                )
                del Pia
        del Lia

    #for w in range(nw):
    #    Pi[w] = np.linalg.inv(np.eye(naux) - Pi[w]) - np.eye(naux)
    # identity = cp.eye(naux)
    # Pi[:] = cp.linalg.inv(identity - Pi) - identity
    # del identity
    # Pi_inv = (I - Pi)^-1 - I, one frequency at a time and in place: a 3-D cupy.linalg.inv holds several copies of
    # the whole (nw, naux, naux) stack
    for w in range(nw):
        if hasattr(gw._scf, 'sigma'):
            # get full Pi from lower triangle
            cupy_helper.hermi_triu(Pi[w])
        get_id_minus_pi_inv(Pi[w], overwrite_input=True)
        addto_diagonal(Pi[w], -1.0)
    Pi_inv = Pi

    # nseg for self-energy is determined by nmo
    nseg = int((nmo + gw.segsize - 1) // gw.segsize)

    if fullsigma is False:
        sigma_real = cp.zeros(shape=[nw_sigma, norbs], dtype=cp.double)
        sigma_imag = cp.zeros(shape=[nw_sigma, norbs], dtype=cp.double)
    else:
        sigma_real = cp.zeros(shape=[nw_sigma, norbs, norbs], dtype=cp.double)
        sigma_imag = cp.zeros(shape=[nw_sigma, norbs, norbs], dtype=cp.double)

    if fullsigma is False:
        Qmn = None
        Wmn = None
        naux_ones = cp.ones((1, naux))

    emo = omega[None] - mo_energy_g[:, None]
    for i in range(nseg):
        if gw.verbose >= 4:
            gw.stdout.write(f'{i:4d} ')
            if i % 12 == 11:
                gw.stdout.write('\n')
            gw.stdout.flush()

        orb_start = i * gw.segsize
        orb_end = min((i + 1) * gw.segsize, nmo)
        ijslice = (orb_start, orb_end, 0, nmo)
        Lpq = gw.ao2mo(mo_coeff=mo_coeff, ijslice=ijslice)
        l_slice = cp.ascontiguousarray(Lpq[:, :, mkslice(orbs)].reshape(naux, -1))
        del Lpq

        for w in range(nw):
            g0 = quad_wts[w] * emo[orb_start:orb_end] / (emo[orb_start:orb_end] ** 2 + quad_freqs[w] ** 2)
            # split g0 into real and imag parts to avoid costly type conversions
            g0r = cp.ascontiguousarray(g0.real)
            g0i = cp.ascontiguousarray(g0.imag)

            if fullsigma is False:
                Qmn = cp.matmul(Pi_inv[w].T, l_slice)
                # plain version
                # Qmn *= l_slice
                # fused version
                Qmn = _Qmn(Qmn, l_slice)

                Wmn = cp.matmul(naux_ones, Qmn)

                # sigma -= einsum('mn,mw->wn', Wmn, g0) / np.pi
                # CPU version using BLAS
                # sla.blas.dgemm(
                #     alpha=-1.0 / np.pi,
                #     a=Wmn.reshape(orb_end - orb_start, norbs).T,
                #     b=g0r.T,
                #     c=sigma_real.T,
                #     trans_a=0,
                #     trans_b=1,
                #     beta=1.0,
                #     overwrite_c=True,
                # )
                cp.cublas.gemm(
                    'N',
                    'T',
                    Wmn.reshape(orb_end - orb_start, norbs).T,
                    g0r.T,
                    out=sigma_real.T,
                    alpha=-1.0 / cp.pi,
                    beta=1.0,
                )
                # CPU version using BLAS
                # sla.blas.dgemm(
                #     alpha=-1.0 / np.pi,
                #     a=Wmn.reshape(orb_end - orb_start, norbs).T,
                #     b=g0i.T,
                #     c=sigma_imag.T,
                #     trans_a=0,
                #     trans_b=1,
                #     beta=1.0,
                #     overwrite_c=True,
                # )
                cp.cublas.gemm(
                    'N',
                    'T',
                    Wmn.reshape(orb_end - orb_start, norbs).T,
                    g0i.T,
                    out=sigma_imag.T,
                    alpha=-1.0 / cp.pi,
                    beta=1.0,
                )
            else:
                Wmn = cp.zeros(shape=[orb_end - orb_start, norbs, norbs], dtype=cp.double)
                for orbm in range(orb_end - orb_start):
                    Qmn = cp.matmul(Pi_inv[w], l_slice.reshape(naux, orb_start - orb_end, norbs)[:, orbm, :])
                    cp.matmul(Qmn.T, l_slice.reshape(naux, orb_start - orb_end, norbs)[:, orbm, :], out=Wmn[orbm])

                # sigma -= einsum('mnl,mw->wnl', Wmn, g0)/np.pi
                # CPU version using BLAS
                # sla.blas.dgemm(
                #     alpha=-1.0 / np.pi,
                #     a=Wmn.reshape(orb_end - orb_start, norbs * norbs).T,
                #     b=g0r.T,
                #     c=sigma_real.reshape(nw_sigma, norbs * norbs).T,
                #     trans_a=0,
                #     trans_b=1,
                #     beta=1.0,
                #     overwrite_c=True,
                # )
                cp.cublas.gemm(
                    'N',
                    'T',
                    Wmn.reshape(orb_end - orb_start, norbs * norbs).T,
                    g0r.T,
                    out=sigma_real.reshape(nw_sigma, norbs * norbs).T,
                    alpha=-1.0 / cp.pi,
                    beta=1.0,
                )
                # CPU version using BLAS
                # sla.blas.dgemm(
                #     alpha=-1.0 / np.pi,
                #     a=Wmn.reshape(orb_end - orb_start, norbs * norbs).T,
                #     b=g0i.T,
                #     c=sigma_imag.reshape(nw_sigma, norbs * norbs).T,
                #     trans_a=0,
                #     trans_b=1,
                #     beta=1.0,
                #     overwrite_c=True,
                # )
                cp.cublas.gemm(
                    'N',
                    'T',
                    Wmn.reshape(orb_end - orb_start, norbs * norbs).T,
                    g0i.T,
                    out=sigma_imag.reshape(nw_sigma, norbs * norbs).T,
                    alpha=-1.0 / cp.pi,
                    beta=1.0,
                )

    sigma = sigma_real + 1.0j * sigma_imag
    if fullsigma is False:
        sigma = cp.ascontiguousarray(sigma.transpose(1, 0))
    else:
        sigma = cp.ascontiguousarray(sigma.transpose(1, 2, 0))

    logger.info(gw, '\nFinished get_sigma_diag_outcore main loop.')

    if gw.rdm is True:
        gw.sigmaI = sigma

    return sigma[..., :nw_cutoff], omega[:nw_cutoff]


def _mo_energy_without_core(gw, mo_energy):
    """Get non-frozen orbital energy.

    Parameters
    ----------
    gw : GWAC
        GW object, provides attributes: frozen, mo_occ, _nmo
    mo_energy : double 1d array
        full orbital energy

    Returns
    -------
    mo_energy : double 1d array
        non-frozen orbital energy
    """
    return mo_energy[get_frozen_mask(gw)]


def _mo_without_core(gw, mo):
    """Get non-frozen orbital coefficient.

    Parameters
    ----------
    gw : GWAC
        GW object, provides attributes: frozen, mo_occ, _nmo
    mo : double 2d array
        full orbital coefficient

    Returns
    -------
    mo : double 2d array
        non-frozen orbital coefficient
    """
    return mo[:, get_frozen_mask(gw)]


def set_frozen_orbs(gw):
    """Set orbs and orbs_frz attributes from frozen mask.
    orbs: list of orbital index in all orbitals
    orbs_frz: list of orbital index in non-frozen orbitals

    Parameters
    ----------
    gw : GWAC
        restricted GW object
    """
    if gw.frozen is not None:
        if gw.orbs is not None:
            if isinstance(gw.frozen, (int, np.integer)):
                # frozen core
                gw.orbs_frz = [x - gw.frozen for x in gw.orbs]
            else:
                # frozen list
                assert isinstance(gw.frozen[0], (int, np.integer))
                gw.orbs_frz = []
                for orbi in gw.orbs:
                    count = len([p for p in gw.frozen if p <= orbi])
                    gw.orbs_frz.append(orbi - count)
            if any(np.array(gw.orbs_frz) < 0):
                raise RuntimeError('GW orbs must be larger than frozen core!')
        else:
            gw.orbs_frz = range(gw.nmo)
            gw.orbs = range(len(gw._scf.mo_energy))
            if isinstance(gw.frozen, (int, np.integer)):
                gw.orbs = list(set(gw.orbs) - set(range(gw.frozen)))
            else:
                assert isinstance(gw.frozen[0], (int, np.integer))
                gw.orbs = list(set(gw.orbs) - set(gw.frozen))
    else:
        if gw.orbs is None:
            gw.orbs = gw.orbs_frz = range(len(gw._scf.mo_energy))
        else:
            gw.orbs_frz = gw.orbs
    return


class GWAC(lib_cpu.StreamObject):
    def __init__(self, mf, frozen=None, auxbasis=None):
        self.mol = mf.mol  # mol object
        self._scf = mf  # mean-field object
        self.verbose = self.mol.verbose  # verbose level
        self.stdout = self.mol.stdout  # standard output
        self.max_memory = mf.max_memory  # max memory in MB
        self.auxbasis = auxbasis  # auxiliary basis set for density fitting

        # options
        self.frozen = frozen  # frozen orbital options
        self.orbs = None  # list of orbital index in full nmo
        self.fullsigma = False  # calculate off-diagonal self-energy
        self.rdm = False  # calculate GW density matrix
        self.vhf_df = False  # use density-fitting for exchange self-energy
        self.outcore = False  # low-memory routine to calculate self-energy
        self.segsize = 100  # number of orbitals in one segment for outcore
        self.eta = 5.0e-3  # broadening parameter
        self.nw = 100  # number of quadrature points for integration
        self.nw2 = None  # number of points at which to evaluate self-energy
        self.ac = 'pade'  # analytical continuation method
        self.ac_iw_cutoff = 5.0  # imaginary frequency cutting for fitting self-energy
        self.ac_pade_npts = 18  # number of selected points for Pade approximation
        self.ac_pade_step_ratio = 2.0 / 3.0  # final/initial step size for Pade approximation
        self.ac_pes_mmax = 6
        self.ac_pes_maxiter = 200  # max iteration in PES fitting
        self.ac_pes_disp = False
        self.ac_idx = None  # indices of iw to use for AC fitting
        self.qpe_max_iter = 100  # max iteration in iteratively solving quasiparticle equation
        self.qpe_tol = 1.0e-6  # tolerance in Newton method for iteratively quasiparticle equation
        self.qpe_linearized = False  # use linearized quasiparticle equation
        self.qpe_linearized_range = [0.5, 1.5]  # Z-shot factor range, if not in this range, z=1
        self.writefile = 0  # write file level

        # don't modify the following attributes, they are not input options
        self._nocc = None  # number of occupied orbitals
        self._nmo = None  # number of orbitals (exclude frozen orbitals)
        self._naux = None  # number of auxiliary basis functions
        self.orbs_frz = None  # list of orbital index in non-frozen orbitals
        self.mo_energy = cp.array(mf.mo_energy, copy=True)  # quasiparticle energy
        self.mo_coeff = cp.array(mf.mo_coeff, copy=True)  # quasiparticle orbtial coefficient
        self.mo_occ = cp.array(mf.mo_occ, copy=True)  # quasiparticle orbital occupation
        self.Lpq = None  # three-center density-fitting matrix in MO

        # results
        self.vk = None  # exchange self-energy matrix
        self.vxc = None  # mean-field vxc matrix
        self.freqs = None  # frequency grids, size=nw2
        self.wts = None  # weights of frequency grids, size=nw2
        self.ef = None  # Fermi level
        self.sigmaI = None  # self-energy in the imaginary axis
        self.acobj = None  # analytical continuation object

        return

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('GW nocc = %d, nvir = %d', nocc, nvir)
        log.info('frozen orbitals = %s', self.frozen)
        log.info('off-diagonal self-energy = %s', self.fullsigma)
        log.info('GW density matrix = %s', self.rdm)
        log.info('density-fitting for exchange = %s', self.vhf_df)
        log.info('outcore for self-energy= %s', self.outcore)
        if self.outcore is True:
            log.info('outcore segment size = %d', self.segsize)
        log.info('broadening parameter = %.3e', self.eta)
        if self.nw2 is None:
            log.info('number of grids = %d', self.nw)
        else:
            log.info('grid size for W is %d', self.nw)
            log.info('grid size for self-energy is %d', self.nw2)
        log.info('analytic continuation method = %s', self.ac)
        log.info('imaginary frequency cutoff = %.1f', self.ac_iw_cutoff)
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
        return int(self.get_nocc())

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

    def kernel(self):
        """Do one-shot GW calculation using analytical continuation."""
        # smeared GW needs denser grids to be accurate
        if hasattr(self._scf, 'sigma'):
            assert self.frozen == 0 or self.frozen is None
            self.nw = max(400, self.nw)
            self.ac_pade_npts = 18
            self.ac_pade_step_ratio = 5.0 / 6.0

        # outcore rebuilds MO integrals from the AO cderi, so it needs with_df even when Lpq is given
        if self.Lpq is None or self.outcore:
            self.initialize_df(auxbasis=self.auxbasis)

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        kernel(self)
        logger.timer(self, 'GW', *cput0)
        return

    def initialize_df(self, auxbasis=None):
        """Initialize density fitting and build the cderi.

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
        # naux and nao are only set by build(), and loop() would only build lazily
        if self.with_df._cderi is None:
            self.with_df.build()
        self._naux = self.with_df.naux
        self._keys.update(['with_df'])
        return

    def ao2mo(self, mo_coeff=None, ijslice=None):
        """Transform density-fitting integral from AO to MO.

        GPU equivalent of nr_e2(with_df._cderi, mo, ijslice, aosym='s2').

        Parameters
        ----------
        mo_coeff : double 2d array, optional
            coefficient from AO to MO, by default the non-frozen orbitals
        ijslice : tuple, optional
            (1st idx start, 1st idx end, 2nd idx start, 2nd idx end), by default all orbitals

        Returns
        -------
        Lpq : double 3d array
            three-center density-fitting matrix in MO, (naux, ni, nj)
        """
        if mo_coeff is None:
            mo_coeff = _mo_without_core(self, self.mo_coeff)
        mo = cp.asarray(mo_coeff)
        nmo = mo.shape[1]
        if ijslice is None:
            ijslice = (0, nmo, 0, nmo)
        i0, i1, j0, j1 = ijslice
        ni, nj = int(i1 - i0), int(j1 - j0)
        naux = self.with_df.naux

        need = 8 * naux * ni * nj / 1e6
        available = cupy_helper.get_avail_mem() / 1e6
        if need > available:
            logger.warn(self, 'GPU memory is not enough for ao2mo, need %.1f G, available %.1f G. ' % (need, available))

        mo_i = cp.ascontiguousarray(mo[:, i0:i1])
        mo_j = cp.ascontiguousarray(mo[:, j0:j1])
        nao, ni = mo_i.shape

        Lpq = cp.empty((naux, ni, nj), dtype=cp.double)
        buf = None
        p0 = 0
        for Lao, _ in self.with_df.loop():
            p1 = p0 + Lao.shape[0]
            if buf is None:
                # first block is the largest; later blocks reuse it
                buf = cp.empty(Lao.shape[0] * nao * ni)
            tmp = buf[:(p1 - p0) * nao * ni].reshape(p1 - p0, nao, ni)
            contract('Pmn,ni->Pmi', Lao, mo_i, out=tmp)
            contract('Pmi,mj->Pij', tmp, mo_j, out=Lpq[p0:p1])
            p0 = p1
        assert p0 == naux, 'cderi loop produced fewer rows than expected'
        return Lpq


    def get_ef(self, mo_energy=None):
        """Get Fermi level.
        For gapped systems, Fermi level is computed as the average between HOMO and LUMO.
        For metallic systems, Fermi level is optmized according to mo_energy.

        Parameters
        ----------
        mo_energy : double 1d array, optional
            orbital energy, by default None

        Returns
        -------
        ef : double
            Fermi level
        """
        if mo_energy is None:
            mo_energy = self.mo_energy
        mo_energy = cp.asnumpy(mo_energy)

        if hasattr(self._scf, 'sigma'):
            f_occ = _fermi_smearing_occ if self._scf.smearing_method.lower() == 'fermi' else _gaussian_smearing_occ
            ef = _smearing_optimize(f_occ, mo_energy, self._scf.mol.nelectron / 2, self._scf.sigma)[0]
        else:
            # working with full space mo_energy and nocc here
            nocc = self._scf.mol.nelectron // 2
            if (mo_energy[nocc] - mo_energy[nocc - 1]) < 1e-3:
                logger.warn(self, 'GW not well-defined for degeneracy!')
            ef = (mo_energy[nocc - 1] + mo_energy[nocc]) * 0.5
        return float(ef)

    def energy_tot(self):
        """Compute GW total energy according to Galitskii-Migdal formula.
        See equation 6 in doi.org/10.1103/PhysRevB.86.081102,
        and equation 11 in doi.org/10.1021/acs.jctc.0c01264.
        NOTE: self-energy evaluted by numerical integration at large frequency is difficult to be accurate,
        so this function is numerically unstable.

        Returns
        -------
        E_tot : double
            GW total energy
        E_hf : double
            Hartree-Fock total energy
        Ec : double
            GW correlation energy
        """
        assert self.sigmaI is not None
        assert self.rdm and self.fullsigma
        sigmaI = self.sigmaI[:, :, 1:]
        freqs = cp.asnumpy(1j * self.freqs)
        wts = self.wts
        nmo = self.nmo

        if len(self.orbs) != nmo:
            sigma = np.zeros((nmo, nmo, len(freqs)), dtype=sigmaI.dtype)
            for ia, a in enumerate(self.orbs):
                for ib, b in enumerate(self.orbs):
                    sigma[a, b, :] = sigmaI[ia, ib, :]
        else:
            sigma = sigmaI

        # Compute mean-field Green's function on imag freq
        gf0 = get_g0(freqs, cp.asnumpy(self._scf.mo_energy) - self.ef, eta=0)

        # Compute GW correlation energy
        g_sigma = 1.0 / 2.0 / np.pi * np.einsum('ijw,ijw,w->ij', gf0, sigma, wts)
        # factor 2.0 from integration on negative imaginary axis
        Ec = 2.0 * np.trace(g_sigma).real

        # Compute HF energy using DFT density matrix
        # NOTE: this definitation can be wrong
        # TODO: update HF-energy with MRGW functions
        dm = self._scf.make_rdm1()
        # NOTE: should I use GDF or FFTDF?
        rhf = scf.RHF(self.mol)
        rhf.verbose = 0
        E_hf = float(rhf.energy_elec(dm=dm)[0]) + float(self._scf.energy_nuc())

        E_tot = E_hf + Ec

        return E_tot, E_hf, Ec

    def make_rdm1(self, ao_repr=False, mode='linear'):
        r"""Get GW density matrix from G(it=0).
        G(it=0) = \int G(iw) dw
        As shown in doi.org/10.1021/acs.jctc.0c01264, calculate G0W0 Green's function using Dyson equation is not
        particle number conserving.
        The linear mode G = G0 + G0 Sigma G0 is particle number conserving.

        Parameters
        ----------
        ao_repr : bool, optional
            return dm in AO space instead of MO space, by default False
        mode : str, optional
            mode for Dyson equation, 'linear' or 'dyson', by default 'linear'

        Returns
        -------
        rdm1 : double 2d array
            one-particle density matrix
        """

        assert self.sigmaI is not None
        assert self.rdm and self.fullsigma
        assert mode in ['dyson', 'linear']
        sigmaI = self.sigmaI[:, :, 1:]
        freqs = 1j * self.freqs
        wts = self.wts
        nmo = self.nmo
        if len(self.orbs) != nmo:
            sigma = np.zeros((nmo, nmo, len(freqs)), dtype=sigmaI.dtype)
            for ia, a in enumerate(self.orbs):
                for ib, b in enumerate(self.orbs):
                    sigma[a, b, :] = sigmaI[ia, ib, :]
        else:
            sigma = sigmaI

        # Compute GW Green's function on imag freq
        gf0 = get_g0(freqs, cp.asnumpy(self._scf.mo_energy) - self.ef, eta=0)
        gf = np.zeros_like(gf0)
        if mode == 'linear':
            for iw in range(len(freqs)):
                gf[:, :, iw] = gf0[:, :, iw] + (gf0[:, :, iw] @ (sigma[:, :, iw] + self.vk - self.vxc) @ gf0[:, :, iw])
        elif mode == 'dyson':
            for iw in range(len(freqs)):
                gf[:, :, iw] = np.linalg.inv(np.linalg.inv(gf0[:, :, iw]) - sigma[:, :, iw] - self.vk + self.vxc)

        # GW density matrix
        rdm1 = 2.0 / np.pi * np.einsum('ijw,w->ij', gf, wts).real + np.eye(nmo)
        # Symmetrize density matrix
        rdm1 = 0.5 * (rdm1 + rdm1.T)
        logger.info(self, 'GW particle number = %s', np.trace(rdm1))

        if ao_repr is True:
            mo_coeff = cp.asnumpy(self._scf.mo_coeff)
            rdm1 = mo_coeff @ rdm1 @ mo_coeff.T

        return rdm1

    def make_gf(self, omega, eta=0.0, mode='dyson'):
        """Get G0W0 Green's function by AC fitting.

        Parameters
        ----------
        omega : complex 1d array
            frequency on which to evaluate the Green's function
        eta : double, optional
            broadening parameter. Defaults to 0.
        mode : str, optional
            mode for Dyson equation, 'linear' or 'dyson', by default 'dyson'

        Returns
        -------
        gf : complex 3d array
            GW Green's function
        gf0 : complex 3d array
            non-interacting Green's function
        sigma : complex 3d array
            self-energy
        """
        mo_energy = cp.asnumpy(self._scf.mo_energy)
        gf0 = get_g0(omega, mo_energy, eta)

        gf = np.zeros_like(gf0)
        if self.fullsigma is True:
            sigma = self.acobj.ac_eval(omega + 1j * eta)
            sigma_diff = np.array(sigma, copy=True)
            for iw in range(len(omega)):
                sigma_diff[:, :, iw] += self.vk - self.vxc
        else:
            sigma = np.zeros_like(gf0)
            sigma_diff = np.zeros_like(gf0)
            for iw in range(len(omega)):
                for i in range(len(mo_energy)):
                    sigma[i, i, iw] = self.acobj[i].ac_eval(omega + 1j * eta)
                    sigma_diff[i, i, iw] = sigma[i, i, iw] + self.vk[i, i] - self.vxc[i, i]

        for iw in range(len(omega)):
            if mode == 'linear':
                gf[:, :, iw] = gf0[:, :, iw] + (gf0[:, :, iw] @ sigma_diff[:, :, iw] @ gf0[:, :, iw])
            elif mode == 'dyson':
                gf[:, :, iw] = np.linalg.inv(np.linalg.inv(gf0[:, :, iw]) - sigma_diff[:, :, iw])

        return gf, gf0, sigma

    def get_sigma_exchange(self, mo_coeff):
        """Get exchange self-energy (EXX).

        Parameters
        ----------
        mo_coeff : double 2d array
            orbital coefficient

        Returns
        -------
        vk : double 2d array
            exchange self-energy
        """
        mf = self._scf
        with lib_cpu.temporary_env(mf, verbose=0), lib_cpu.temporary_env(self.mol, verbose=0):
            dm = mf.make_rdm1()
            # keep this condition for embedding calculations
            if isinstance(mf, gpu_hf.RHF) and not isinstance(mf, KohnShamDFT):
                vk_ao = cp.asarray(mf.get_veff(mf.mol, dm)) - cp.asarray(mf.get_j(mf.mol, dm))
            else:
                # For a restricted density matrix, RHF.get_veff - RHF.get_j is exactly -K/2; smearing only
                vk_ao = -0.5 * cp.asarray(mf.get_k(mf.mol, dm))
        mo_coeff = cp.asarray(mo_coeff)
        vk = mo_coeff.T @ vk_ao @ mo_coeff
        return vk

    def setup_evaluation_grid(self, fallback_freqs=None, fallback_wts=None):
        """Set up self-energy grid, aka freqs2, aka gw.freqs.

        Parameters
        ----------
        fallback_freqs : double 1d array
            These are used as last resort if neither gw.nw2 nor gw.freqs is set.
        fallback_wts : double 1d array
            weights corresponding to fallback_freqs.

        Returns
        -------
        eval_freqs_with_zero : double 1d array
            self.freqs prepended with 0 for convenience.
        """
        if self.freqs is not None:
            pass
        elif self.nw2 is not None:
            assert self.freqs is None and self.wts is None, 'freqs and wts must not be set if nw2 is specified'
            self.freqs, self.wts = _get_scaled_legendre_roots(self.nw2)
        else:
            assert fallback_freqs is not None and fallback_wts is not None, 'freqs and wts must be set'
            self.freqs = fallback_freqs.copy()
            self.wts = fallback_wts.copy()
        assert self.freqs.ndim == 1, 'freqs must be 1D array'
        assert self.wts.shape == self.freqs.shape, 'freqs and wts must have the same shape'

        eval_freqs_with_zero = np.concatenate(([0.0], self.freqs))
        return eval_freqs_with_zero

    to_cpu = utils.to_cpu
    to_gpu = utils.to_gpu
    device = utils.device
