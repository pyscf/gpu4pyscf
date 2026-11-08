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
Periodic spin-restricted random phase approximation (direct RPA) with N^4 scaling, on one GPU.

References:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    New J. Phys. 14, 053020 (2012)
"""

import cupy as cp
import numpy as np
import time

from pyscf import lib as lib_cpu
from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots
from pyscf.pbc import tools
from pyscf.pbc.gw.krpa import get_kconserv_ria_efficient
from pyscf.pbc.mp.kmp2 import get_nocc, get_nmo, get_frozen_mask

from gpu4pyscf.df import make_auxbasis
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import get_avail_mem
from gpu4pyscf.lib.cutensor import contract
from gpu4pyscf.pbc import df
from gpu4pyscf.pbc.lib.kpts_helper import kk_adapted_iter
from gpu4pyscf.gw.gw_ac import get_idx_metal
from gpu4pyscf.pbc.gw.krgw_ac import _mo_occ_frozen, _mo_energy_frozen, _mo_frozen, get_rho_response, \
    get_rho_response_head, get_rho_response_wing, get_qij_components

# fuse functions for GPU
@cp.fuse()
def _eia(eia, omega):
    return eia / (omega**2 + eia**2)


@cp.fuse()
def _eia_fia(eia, fia, omega):
    return eia * fia / (omega**2 + eia**2)


@cp.fuse()
def _eia_q(eia, qov, omega):
    return eia * cp.conj(qov) / (omega**2 + eia**2)


def kernel(rpa, mo_energy, with_e_hf=None):
    """RPA correlation and total energy

    Parameters
    ----------
    rpa : KRPA
        rpa object
    mo_energy : double array
        molecular orbital energies
    with_e_hf : float, optional
        extra input HF energy, by default None

    Returns
    -------
    e_tot : float
        RPA total energy
    e_hf : float
        HF energy (exact exchange for given mo_coeff)
    e_corr : float
        RPA correlation energy
    """
    # Compute HF exchange energy (EXX)
    if with_e_hf is None:
        with lib_cpu.temporary_env(rpa.with_df, verbose=0), lib_cpu.temporary_env(rpa._scf.mol, verbose=0):
            dm = rpa._scf.make_rdm1()
            e_1e = 1.0 / len(rpa.kpts) * cp.einsum('kij,kji', dm, rpa._scf.get_hcore()).real
            # gpu4pyscf KRHF.get_j does not default kpts to rhf.kpts
            e_j = 0.5 / len(rpa.kpts) * cp.einsum('kij,kji', dm, rpa._scf.get_j(rpa._scf.cell, dm, kpts=rpa.kpts)).real
            e_x = get_rpa_exx(rpa, acfd=rpa.acfd_exx, correction_only=False)
            e_nuc = rpa._scf.energy_nuc()
            e_hf = e_1e + e_j + e_x + e_nuc
    else:
        e_hf = with_e_hf
        if rpa.verbose >= logger.DEBUG:
            logger.debug(rpa, f'  Setting EXX energy explicitly to {e_hf}')

    is_metal = hasattr(rpa._scf, 'sigma')

    # Turn off FC for metals
    if is_metal and rpa.fc:
        logger.warn(rpa, 'FC not available for metals - setting rpa.fc to False')
        rpa.fc = False

    # Grids for integration on imaginary axis
    freqs, wts = rpa.get_grids(nw=rpa.nw, mo_energy=mo_energy)

    # Compute RPA correlation energy
    if rpa.outcore:
        if is_metal:
            e_corr = get_rpa_ecorr_outcore_metal(rpa, freqs, wts)
        else:
            e_corr = get_rpa_ecorr_outcore(rpa, freqs, wts)
    else:
        e_corr = get_rpa_ecorr(rpa, freqs, wts)

    # Compute total energy
    e_hf = float(e_hf)
    e_corr = float(e_corr)
    e_tot = e_hf + e_corr

    if rpa.verbose >= logger.DEBUG:
        logger.debug(rpa, f'  RPA total energy = {e_tot}')
        logger.debug(rpa, f'  EXX energy = {e_hf}, RPA corr energy = {e_corr}')

    return e_tot, e_hf, e_corr


def get_metal_slice(mo_energy, mo_occ, kidx):
    """Occupation windows of the metallic response for the pairs (ki, ka=kidx[ki]).
    Only pairs with f_i != f_a contribute: rows i = occupied and fractional orbitals at ki,
    columns a = fractional and virtual orbitals at ka. The a -> i pairs, which the full nmo x nmo block also
    contains, give the same contribution, hence the factor 4 (not 2) in get_rho_response_metal.
    Fractional-fractional pairs belong to both sets and are halved.

    Parameters
    ----------
    mo_energy : double 2d array
        orbital energy
    mo_occ : double 2d array
        occupation number (0 to 2)
    kidx : list
        momentum-conserved k-point list ka=kidx[ki]

    Returns
    -------
    slices : list of (slice, slice)
        row (at ki) and column (at ka) window for each ki
    eia : double 3d array
        orbital energy differences, (nkpts, nocc_max, nvir_max), 1 in the padding
    fia : double 3d array
        occupation differences / 2, (nkpts, nocc_max, nvir_max), 0 in the padding
    """
    nkpts = len(kidx)
    # slices: row and column window of each ki; counts: (occupied orbitals at ki, fractional orbitals at ka)
    slices, counts = [], []
    for i in range(nkpts):
        a = kidx[i]
        idx_occ_i, _, idx_vir_i = get_idx_metal(mo_occ[i])
        idx_occ_a, idx_frac_a, idx_vir_a = get_idx_metal(mo_occ[a])
        i_start = idx_occ_i[0] if idx_occ_i.size else 0
        i_stop = idx_vir_i[0] if idx_vir_i.size else len(mo_occ[i])
        a_start = idx_occ_a[-1] + 1 if idx_occ_a.size else 0
        a_stop = idx_vir_a[-1] + 1 if idx_vir_a.size else len(mo_occ[a])
        slices.append((slice(int(i_start), int(i_stop)), slice(int(a_start), int(a_stop))))
        counts.append((len(idx_occ_i), len(idx_frac_a)))
    ni = max(si.stop - si.start for si, _ in slices)
    na = max(sa.stop - sa.start for _, sa in slices)
    eia = cp.ones(shape=[nkpts, ni, na], dtype=cp.double)
    fia = cp.zeros(shape=[nkpts, ni, na], dtype=cp.double)
    for i, ((si, sa), (nocc_i, nfrac_a)) in enumerate(zip(slices, counts)):
        a = kidx[i]
        e = mo_energy[i, si, None] - mo_energy[a, None, sa]
        f = (mo_occ[i][si, None] - mo_occ[a][None, sa]) / 2.0
        # factor of 0.5 is for double counting
        f[nocc_i:, :nfrac_a] *= 0.5
        eia[i, :e.shape[0], :e.shape[1]] = e
        fia[i, :f.shape[0], :f.shape[1]] = f
    return slices, eia, fia


def get_rho_response_metal(omega, Lia, eia, fia):
    """Get Pi=PV for metallic systems, from the occupation windows of get_metal_slice.
    P is density-density response function.
    V is two-electron integral.
    See equation 24 in doi.org/10.1021/acs.jctc.0c00704.

    NOTE: this function is different from the one in krgw_ac.py, which uses the full nmo x nmo block.
    They should be merged in the future.

    Parameters
    ----------
    omega : double
        real position of imaginary frequency
    Lia : complex 4d ndarray
        three-center density-fitting matrix in MO for the windows, aux-major (naux, nkpts, nocc_max, nvir_max),
        zero in the padding
    eia : double 3d array
        orbital energy differences of the windows
    fia : double 3d array
        occupation differences / 2 of the windows, zero in the padding

    Returns
    -------
    Pi : complex ndarray
        Pi in auxiliary basis at freq iw
    """
    naux, nkpts = Lia.shape[:2]
    # plain version
    # eia = eia * fia / (omega**2 + eia**2)
    # fused version
    eia = _eia_fia(eia, fia, omega)
    Pia = Lia * eia

    # Response from both spin-up and spin-down density, i -> a and a -> i (see get_metal_slice)
    # Pi = (4./nkpts) * einsum('Pkia,Qkia->PQ', Pia, Lia.conj())
    Pi = cp.empty(shape=[naux, naux], dtype=cp.complex128)
    cp.cublas.gemm('H', 'N', Lia.reshape(naux, -1).T, Pia.reshape(naux, -1).T, out=Pi.T, alpha=4.0 / nkpts, beta=0.0)
    return Pi


def rho_accum_inner(Pi, eia, omega, Lov, alpha=0.0, fia=None):
    """Get contribution to response function from current occupied-virtual block.

    Parameters
    ----------
    Pi : complex 2d array
        density-density response function, will be overwritten
    eia : double 2d array
        occupied-virtual orbital energy difference
    omega : double
        real position of imaginary frequency
    Lov : complex 3d array
        occupied-virtual block of three-center density-fitting matrix in MO
    alpha : float, optional
        prefactor, by default 0.0
    fia : double 2d array, optional
        occupied-virtual occupation number difference, by default None
    """
    naux, nocc, nvir = Lov.shape
    if fia is None:
        # plain version
        # eia = eia / (omega**2 + eia**2)
        # fused version
        eia = _eia(eia, omega)
    else:
        # plain version
        # eia = eia * fia / (omega**2 + eia**2)
        # fused version
        eia = _eia_fia(eia, fia, omega)
    Pia = (Lov * eia).reshape(naux, nocc * nvir)

    # C <- alpha * op(A) @ op(B) + beta * C
    cp.cublas.gemm('H', 'N', Lov.reshape(naux, nocc * nvir).T, Pia.T, out=Pi.T, alpha=alpha, beta=1.0)

    return


def rho_wing_accum_inner(Pi_P0, eia, omega, Lov, qov, alpha=0.0):
    """Accumulate the finite-size-correction wing response for one OV slice.

    Parameters
    ----------
    Pi_P0 : complex 1d array
        finite-size correction to density-density response function, will be overwritten
    eia : double 2d array
        occupied-virtual orbital energy difference
    omega : double
        frequency
    Lov : complex 3d array
        occupied-virtual block of three-center density-fitting matrix in MO
    qov : complex 3d array
        virtual-occupied correction
    alpha : float, optional
        prefactor, by default 0.0
    """
    naux, nocc, nvir = Lov.shape
    # plain version
    # eia_q = eia * qov.conj() / (omega**2 + eia**2)
    # fused version
    eia_q = _eia_q(eia, qov, omega)

    Pi_P0 += alpha * cp.matmul(Lov.reshape(naux, nocc * nvir), eia_q.reshape(nocc * nvir))

    return


def get_rpa_ecorr(rpa, freqs, wts):
    """Compute RPA correlation energy.

    Parameters
    ----------
    rpa : KRPA
        rpa object
    freqs : double 1d array
        frequency grid
    wts : double 1d array
        weight of grids

    Returns
    -------
    e_corr : double
        correlation energy
    """
    mo_coeff = cp.asarray(_mo_frozen(rpa, rpa._scf.mo_coeff))
    mo_energy = cp.asarray(_mo_energy_frozen(rpa, rpa._scf.mo_energy))
    mo_occ = cp.asarray(_mo_occ_frozen(rpa, rpa._scf.mo_occ))

    nocc = rpa.nocc
    nmo = rpa.nmo
    nvir = nmo - nocc
    nkpts = rpa.nkpts
    kpts = rpa.kpts
    nw = len(freqs)
    mydf = rpa.with_df

    # possible kpts shift center
    kscaled = rpa.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    is_metal = hasattr(rpa._scf, 'sigma')

    if rpa.fc:
        qij, q_abs, nq_pts = rpa.get_q_mesh(mo_energy, mo_coeff)

    e_corr = 0j

    # Precompute k-conservation table
    # Given k-point indices (kL, i), kconserv_table[kshift,i] contains
    # the index j that satisfies momentum conservation,
    # (k(i) - k(j) - k(kL)) \dot a = 2n\pi
    # i.e.
    # - ki + kj + kL = G
    kconserv_table = get_kconserv_ria_efficient(rpa.mol, kpts - kpts[0])

    for kL in range(nkpts):
        kidx = kconserv_table[kL]
        if is_metal:
            # occupation windows of each (ki, kj), see get_metal_slice
            slice_ia, eia_w, fia_w = get_metal_slice(mo_energy, mo_occ, kidx)
        # Lij: aux-major (L, ki, i, j) for looping every kL
        Lij = None
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
        for i in range(nkpts):
            j = kidx[i]
            kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
            assert np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
            logger.debug(rpa, f'Read Lpq (kL: {kL+1} / {nkpts}, ki: {i}, kj: {j})')
            # ao2mo transform (L|pq) to (L|ij)
            # support unequal naux on different k points
            if is_conj:
                Lpq = Lpq_kL[j].conj().transpose(0, 2, 1)
            else:
                Lpq = Lpq_kL[i]
            Lpq = Lpq.astype(cp.complex128, copy=False)

            naux = Lpq.shape[0]
            if not is_metal:
                if Lij is None:
                    Lij = cp.zeros((naux, nkpts, nocc, nvir), dtype=cp.complex128)
                # This computes Lij[:, i] = einsum('mi,Pmn,na->Pia', mo_coeff[i][:, :nocc].conj(), Lpq, mo_coeff[j][:, nocc:])
                tmp = contract('Pmn,mi->Pin', Lpq, mo_coeff[i][:, :nocc].conj())
                Lij[:, i] = contract('Pin,na->Pia', tmp, mo_coeff[j][:, nocc:])
                tmp = None
            else:
                # Only the window block of Lpq is needed (occupied and fractional rows at ki, fractional and
                # virtual columns at kj), stored with zero padding
                si, sa = slice_ia[i]
                if Lij is None:
                    Lij = cp.zeros((naux, nkpts, eia_w.shape[1], eia_w.shape[2]), dtype=cp.complex128)
                tmp = contract('Pmn,mi->Pin', Lpq, mo_coeff[i][:, si].conj())
                Lij[:, i, :si.stop - si.start, :sa.stop - sa.start] = contract('Pin,na->Pia', tmp, mo_coeff[j][:, sa])
                tmp = None
        Lpq_kL = Lpq = None

        for w in range(nw):
            if is_metal:
                Pi = get_rho_response_metal(freqs[w], Lij, eia_w, fia_w)
            else:
                Pi = get_rho_response(freqs[w], mo_energy, Lij, kidx)
            if kL == 0 and rpa.fc:
                for iq in range(nq_pts):
                    # head Pi_00
                    Pi_00 = get_rho_response_head(freqs[w], mo_energy, qij[iq])
                    Pi_00 = 4.0 * np.pi / np.linalg.norm(q_abs[iq]) ** 2 * Pi_00
                    # wings Pi_P0
                    Pi_P0 = get_rho_response_wing(freqs[w], mo_energy, Lij, qij[iq])
                    Pi_P0 = np.sqrt(4.0 * np.pi) / np.linalg.norm(q_abs[iq]) * Pi_P0

                    # assemble Pi
                    Pi_fc = cp.zeros((naux + 1, naux + 1), dtype=Pi.dtype)
                    Pi_fc[0, 0] = Pi_00
                    Pi_fc[0, 1:] = Pi_P0.conj()
                    Pi_fc[1:, 0] = Pi_P0
                    Pi_fc[1:, 1:] = Pi

                    # average over the q points of the finite-size correction
                    e_corr += get_rpa_ecorr_w(Pi_fc, wts[w]) / nq_pts
            else:
                e_corr += get_rpa_ecorr_w(Pi, wts[w])

    e_corr = e_corr.real
    e_corr *= 1.0 / (2.0 * np.pi) / nkpts
    return e_corr


def get_rpa_ecorr_outcore(rpa, freqs, wts):
    """Low-memory routine to compute RPA correlation energy.

    Parameters
    ----------
    rpa : KRPA
        rpa object
    freqs : double 1d array
        frequency grid
    wts : double 1d array
        weight of grids

    Returns
    -------
    e_corr : double
        correlation energy
    """
    mo_coeff = cp.asarray(_mo_frozen(rpa, rpa._scf.mo_coeff))
    mo_energy = cp.asarray(_mo_energy_frozen(rpa, rpa._scf.mo_energy))

    nocc = rpa.nocc
    nmo = rpa.nmo
    nao = mo_coeff.shape[1]
    nkpts = rpa.nkpts
    kpts = rpa.kpts
    nw = len(freqs)
    mydf = rpa.with_df

    # possible kpts shift center
    kscaled = rpa.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    if rpa.fc:
        qij, q_abs, nq_pts = rpa.get_q_mesh(mo_energy, mo_coeff)

    e_corr = 0j

    # Precompute k-conservation table
    # Given k-point indices (kL, i), kconserv_table[kshift,i] contains
    # the index j that satisfies momentum conservation,
    # (k(i) - k(j) - k(kL)) \dot a = 2n\pi
    # i.e.
    # - ki + kj + kL = G
    kconserv_table = get_kconserv_ria_efficient(rpa.mol, kpts - kpts[0])

    for kL in range(nkpts):
        Pi = None
        Pi_P0 = None
        # Read (L|pq) for every ki and kj that conserves with kL and ki (-ki+kj+kL=G).
        # GDF only stores one of the two conjugated momentum transfers kL and -kL,
        # (L|ki kj) is either Lpq_kL[ki] or Lpq_kL[kj].conj().transpose(0, 2, 1).
        for kp, _, ki_idx, kj_idx in kk_adapted_iter(mydf.kmesh):
            is_conj = (kconserv_table[kL, ki_idx] != kj_idx).any()
            if not is_conj or (kconserv_table[kL, kj_idx] == ki_idx).all():
                break
        else:
            raise RuntimeError(f'Momentum transfer of kL = {kL} is not found in GDF')
        naux = mydf._cderi[kp].shape[0]
        for i in range(nkpts):
            # Find (ki,kj) that satisfies momentum conservation with kL
            kconserv = -kscaled[i] + kscaled[kconserv_table[kL, i]] + kscaled[kL]
            assert np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
        Pi = cp.zeros((nw, naux, naux), dtype=cp.complex128)
        if kL == 0 and rpa.fc:
            Pi_P0 = cp.zeros((nq_pts, nw, naux), dtype=cp.complex128)

        # outcore routine slices over nocc
        nseg = nocc // rpa.segsize + 1
        for iseg in range(nseg):
            orb_start = iseg * rpa.segsize
            orb_end = min((iseg + 1) * rpa.segsize, nocc)
            if orb_end == orb_start:
                continue
            norb_this_iter = orb_end - orb_start

            # Lij_seg: (ki, L, i, j) of this segment for every ki
            Lij_seg = cp.empty((nkpts, naux, norb_this_iter, nmo - nocc), dtype=cp.complex128)
            # a block uses at most 40% of the free GPU memory and at most 10 GB
            blksize = max(1, min(naux, int(min(0.4 * get_avail_mem(), 10e9) / (32 * nkpts * nao**2))))
            aux_iter = [(kp, b0, b1) for b0, b1 in lib_cpu.prange(0, naux, blksize)]
            # one output and one unpacking buffer for all blocks, so each block overwrites the previous one
            out_buf = cp.empty(nkpts * blksize * nao**2, dtype=cp.complex128)
            unpack_buf = cp.empty(nkpts * blksize * nao**2, dtype=cp.complex128)
            p0 = 0
            # slice over naux
            for _, Lpq_kL, sign in mydf.loop(blksize, kpts=kpts, aux_iter=aux_iter, buf=unpack_buf, out=out_buf):
                if sign < 0:
                    continue  # 2D cells (dimension=2): the negative part of the cderi is not used, as before
                p1 = p0 + Lpq_kL.shape[1]
                logger.debug(rpa, f'Read Lpq (kL: {kL+1} / {nkpts}, segment: {iseg+1} / {nseg}, aux: {p0}-{p1})')
                for i in range(nkpts):
                    j = kconserv_table[kL, i]
                    # ao2mo transform (L|pq) to (L|ij)
                    if is_conj:
                        Lpq = Lpq_kL[j].conj().transpose(0, 2, 1)
                    else:
                        Lpq = Lpq_kL[i]
                    Lpq = Lpq.astype(cp.complex128, copy=False)

                    # This computes Lij_seg[i, p0:p1] = einsum('mi,Pmn,na->Pia',
                    # mo_coeff[i][:, orb_start:orb_end].conj(), Lpq, mo_coeff[j][:, nocc:])
                    tmp = contract('Pmn,mi->Pin', Lpq, mo_coeff[i][:, orb_start:orb_end].conj())
                    Lij_seg[i, p0:p1] = contract('Pin,na->Pia', tmp, mo_coeff[j][:, nocc:])
                    tmp = None
                p0 = p1
            Lpq_kL = Lpq = out_buf = unpack_buf = None

            for i in range(nkpts):
                Lij_slice = Lij_seg[i]
                # Find ka that conserves with ki and kL (-ki+ka+kL=G)
                a_inner = kconserv_table[kL, i]
                eia = mo_energy[i][orb_start:orb_end, None] - mo_energy[a_inner][None, nocc:]
                for w in range(nw):
                    rho_accum_inner(Pi[w], eia, freqs[w], Lij_slice, alpha=4.0 / nkpts)
                    if kL == 0 and rpa.fc:
                        for iq in range(nq_pts):
                            rho_wing_accum_inner(
                                Pi_P0[iq, w], eia, freqs[w], Lij_slice, qij[iq, i, orb_start:orb_end], alpha=4.0 / nkpts
                            )
            Lij_seg = Lij_slice = None

        for w in range(nw):
            if kL == 0 and rpa.fc:
                for iq in range(nq_pts):
                    Pi_00 = get_rho_response_head(freqs[w], mo_energy, qij[iq])
                    Pi_00 = 4.0 * np.pi / np.linalg.norm(q_abs[iq]) ** 2 * Pi_00
                    Pi_P0_iq = np.sqrt(4.0 * np.pi) / np.linalg.norm(q_abs[iq]) * Pi_P0[iq, w]

                    Pi_fc = cp.zeros((naux + 1, naux + 1), dtype=Pi.dtype)
                    Pi_fc[0, 0] = Pi_00
                    Pi_fc[0, 1:] = Pi_P0_iq.conj()
                    Pi_fc[1:, 0] = Pi_P0_iq
                    Pi_fc[1:, 1:] = Pi[w]

                    # average over the q points of the finite-size correction
                    e_corr += get_rpa_ecorr_w(Pi_fc, wts[w]) / nq_pts
            else:
                e_corr += get_rpa_ecorr_w(Pi[w], wts[w])

    e_corr = e_corr.real
    e_corr *= 1.0 / (2.0 * np.pi) / nkpts
    return e_corr


def get_rpa_ecorr_outcore_metal(rpa, freqs, wts):
    """Low-memory routine to compute RPA correlation energy for metals.

    Parameters
    ----------
    rpa : KRPA
        rpa object
    freqs : double 1d array
        frequency grid
    wts : double 1d array
        weight of grids

    Returns
    -------
    e_corr : double
        correlation energy
    """
    mo_coeff = cp.asarray(_mo_frozen(rpa, rpa._scf.mo_coeff))
    mo_energy = cp.asarray(_mo_energy_frozen(rpa, rpa._scf.mo_energy))
    mo_occ = cp.asarray(_mo_occ_frozen(rpa, rpa._scf.mo_occ))

    nmo = rpa.nmo
    nao = mo_coeff.shape[1]
    nkpts = rpa.nkpts
    kpts = rpa.kpts
    nw = len(freqs)
    mydf = rpa.with_df

    # possible kpts shift center
    kscaled = rpa.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    e_corr = 0j

    # Precompute k-conservation table
    # Given k-point indices (kL, i), kconserv_table[kshift,i] contains
    # the index j that satisfies momentum conservation,
    # (k(i) - k(j) - k(kL)) \dot a = 2n\pi
    # i.e.
    # - ki + kj + kL = G
    kconserv_table = get_kconserv_ria_efficient(rpa.mol, kpts - kpts[0])

    for kL in range(nkpts):
        Pi = None
        # Read (L|pq) for every ki and kj that conserves with kL and ki (-ki+kj+kL=G).
        # GDF only stores one of the two conjugated momentum transfers kL and -kL,
        # (L|ki kj) is either Lpq_kL[ki] or Lpq_kL[kj].conj().transpose(0, 2, 1).
        for kp, _, ki_idx, kj_idx in kk_adapted_iter(mydf.kmesh):
            is_conj = (kconserv_table[kL, ki_idx] != kj_idx).any()
            if not is_conj or (kconserv_table[kL, kj_idx] == ki_idx).all():
                break
        else:
            raise RuntimeError(f'Momentum transfer of kL = {kL} is not found in GDF')
        naux = mydf._cderi[kp].shape[0]
        # numbers of occupied and fractional orbitals at ki and at kj, for every ki
        nocc_i, nfrac_i, nocc_j, nfrac_j = [], [], [], []
        for i in range(nkpts):
            j = kconserv_table[kL, i]
            # Find (ki,kj) that satisfies momentum conservation with kL
            kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
            assert np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12

            idx_occ_i, idx_frac_i, _ = get_idx_metal(mo_occ[i])
            idx_occ_j, idx_frac_j, _ = get_idx_metal(mo_occ[j])

            nocc_i.append(len(idx_occ_i))
            nfrac_i.append(len(idx_frac_i))
            nocc_j.append(len(idx_occ_j))
            nfrac_j.append(len(idx_frac_j))
        Pi = cp.zeros((nw, naux, naux), dtype=cp.complex128)

        # outcore routines for metal slices over nocc+nfrac
        nseg = max(nocc_i[i] + nfrac_i[i] for i in range(nkpts)) // rpa.segsize + 1
        for iseg in range(nseg):
            orb_start = iseg * rpa.segsize
            orb_end = [min((iseg + 1) * rpa.segsize, nocc_i[i] + nfrac_i[i]) for i in range(nkpts)]
            kseg = [i for i in range(nkpts) if orb_end[i] > orb_start]
            if not kseg:
                break

            # Lij_seg[i]: (L, i, j) of this segment at ki
            Lij_seg = {}
            for i in kseg:
                Lij_seg[i] = cp.empty((naux, orb_end[i] - orb_start, nmo - nocc_j[i]), dtype=cp.complex128)
            # a block uses at most 40% of the free GPU memory and at most 10 GB
            blksize = max(1, min(naux, int(min(0.4 * get_avail_mem(), 10e9) / (32 * nkpts * nao**2))))
            aux_iter = [(kp, b0, b1) for b0, b1 in lib_cpu.prange(0, naux, blksize)]
            # one output and one unpacking buffer for all blocks, so each block overwrites the previous one
            out_buf = cp.empty(nkpts * blksize * nao**2, dtype=cp.complex128)
            unpack_buf = cp.empty(nkpts * blksize * nao**2, dtype=cp.complex128)
            p0 = 0
            # slice over naux
            for _, Lpq_kL, sign in mydf.loop(blksize, kpts=kpts, aux_iter=aux_iter, buf=unpack_buf, out=out_buf):
                if sign < 0:
                    continue  # 2D cells (dimension=2): the negative part of the cderi is not used, as before
                p1 = p0 + Lpq_kL.shape[1]
                logger.debug(rpa, f'Read Lpq (kL: {kL+1} / {nkpts}, segment: {iseg+1} / {nseg}, aux: {p0}-{p1})')
                for i in kseg:
                    j = kconserv_table[kL, i]
                    # ao2mo transform (L|pq) to (L|ij)
                    if is_conj:
                        Lpq = Lpq_kL[j].conj().transpose(0, 2, 1)
                    else:
                        Lpq = Lpq_kL[i]
                    Lpq = Lpq.astype(cp.complex128, copy=False)

                    # This computes Lij_seg[i][p0:p1] = einsum('mi,Pmn,na->Pia',
                    # mo_coeff[i][:, orb_start:orb_end[i]].conj(), Lpq, mo_coeff[j][:, nocc_j[i]:])
                    tmp = contract('Pmn,mi->Pin', Lpq, mo_coeff[i][:, orb_start:orb_end[i]].conj())
                    Lij_seg[i][p0:p1] = contract('Pin,na->Pia', tmp, mo_coeff[j][:, nocc_j[i]:])
                    tmp = None
                p0 = p1
            Lpq_kL = Lpq = out_buf = unpack_buf = None

            for i in kseg:
                # Find ka that conserves with ki and kL (-ki+ka+kL=G)
                j = kconserv_table[kL, i]
                eia = mo_energy[i][orb_start:orb_end[i], None] - mo_energy[j][None, nocc_j[i]:]
                fia = (mo_occ[i][orb_start:orb_end[i], None] - mo_occ[j][None, nocc_j[i]:]) / 2.0
                # The overall fia[nocc_i:, :nfrac_j] *= 0.5 for double counting
                if orb_start >= nocc_i[i]:
                    fia[:, :nfrac_j[i]] *= 0.5
                elif orb_end[i] > nocc_i[i]:
                    offset = nocc_i[i] - orb_start
                    fia[offset:, :nfrac_j[i]] *= 0.5
                for w in range(nw):
                    rho_accum_inner(Pi[w], eia, freqs[w], Lij_seg[i], alpha=4.0 / nkpts, fia=fia)
            Lij_seg = None

        for w in range(nw):
            e_corr += get_rpa_ecorr_w(Pi[w], wts[w])

    e_corr = e_corr.real
    e_corr *= 1.0 / (2.0 * np.pi) / nkpts
    return e_corr


def get_rpa_ecorr_w(Pi_w, wts_w):
    """Get contribution to RPA correlation energy from a single frequency.

    Parameters
    ----------
    Pi_w : complex 2d array
        density-density response function at a single frequency
    wts_w : double
        weights of the frequency

    Returns
    -------
    e_corr : double
        correlation energy
    """
    # First, compute ec_w = Tr(Pi_w) + |log(det(I-Pi_w))|
    ec_w = cp.trace(Pi_w)
    # The following two lines are equivalent to
    # Pi_w = cp.eye(naux) - Pi_w
    Pi_w *= -1.0
    cp.fill_diagonal(Pi_w, cp.diagonal(Pi_w) + 1.0)

    ec_w += cp.linalg.slogdet(Pi_w)[1]
    #e_corr = 1.0 / (2.0 * np.pi) / nkpts * ec_w * wts_w
    e_corr = ec_w * wts_w

    return e_corr


def get_rpa_exx(rpa, acfd=False, correction_only=False):
    """Calculate RPA exchange energy.
    For gapped systems, Hartree-Fock and adiabatic connection fluctuation dissipation exchange energies are the same.
    For metallic systems, they are different.
    The ACFD exchange energy is given by equation 12 in doi.org/10.1103/PhysRevB.81.115126

    Parameters
    ----------
    rpa : KRPA
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
    mo_coeff = cp.asarray(_mo_frozen(rpa, rpa._scf.mo_coeff))
    mo_occ = cp.asarray(_mo_occ_frozen(rpa, rpa._scf.mo_occ))

    nocc = rpa.nocc
    nao = mo_coeff.shape[1]
    nkpts = rpa.nkpts
    kpts = rpa.kpts
    mydf = rpa.with_df

    # Precompute k-conservation table, kconserv_table[kL, i] = j with - ki + kj + kL = G,
    # for kpts shifted by kpts[0] (possible kpts shift center)
    kconserv_table = get_kconserv_ria_efficient(rpa.mol, kpts - kpts[0])

    ex = 0j
    for kL in range(nkpts):
        # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
        # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
        kidx = kconserv_table[kL]
        kidx_r = np.argsort(kidx)

        # Read (L|pq) for every km and kn that conserves with kL and km (-km+kn+kL=G).
        # GDF only stores one of the two conjugated momentum transfers kL and -kL,
        # (L|km kn) is either Lpq_kL[km] or Lpq_kL[kn].conj().transpose(0, 2, 1).
        for kp, _, ki_idx, kj_idx in kk_adapted_iter(mydf.kmesh):
            is_conj = (kidx[ki_idx] != kj_idx).any()
            if not is_conj or (kidx[kj_idx] == ki_idx).all():
                break
        else:
            raise RuntimeError(f'Momentum transfer of kL = {kL} is not found in GDF')
        naux = mydf._cderi[kp].shape[0]
        # a block uses at most 40% of the free GPU memory and at most 10 GB
        blksize = max(1, min(naux, int(min(0.4 * get_avail_mem(), 10e9) / (32 * nkpts * nao**2))))
        aux_iter = [(kp, b0, b1) for b0, b1 in lib_cpu.prange(0, naux, blksize)]
        # one output and one unpacking buffer for all blocks, so each block overwrites the previous one
        out_buf = cp.empty(nkpts * blksize * nao**2, dtype=cp.complex128)
        unpack_buf = cp.empty(nkpts * blksize * nao**2, dtype=cp.complex128)
        for _, Lpq_kL, sign in mydf.loop(blksize, kpts=kpts, aux_iter=aux_iter, buf=unpack_buf, out=out_buf):
            if sign < 0:
                continue  # 2D cells (dimension=2): the negative part of the cderi is not used, as before

            for kn in range(nkpts):
                # Find km that conserves with kn and kL (-km+kn+kL=G)
                km = kidx_r[kn]

                # logger.debug(gw, 'Read Lpq (kL: %s / %s, ki: %s, kj: %s)' % (kL + 1, nkpts, i, j))
                # ao2mo transform (L|pq) to (L|ij)
                # support unequal naux on different k points
                if is_conj:
                    Lpq_ao = Lpq_kL[kn].conj().transpose(0, 2, 1)
                else:
                    Lpq_ao = Lpq_kL[km]
                Lpq_ao = Lpq_ao.astype(cp.complex128, copy=False)

                Lij = None
                if hasattr(rpa._scf, 'sigma'):
                    idx_occ_i, idx_frac_i, _ = get_idx_metal(mo_occ[km])
                    idx_occ_j, idx_frac_j, _ = get_idx_metal(mo_occ[kn])
                    nocc_i = len(idx_occ_i) + len(idx_frac_i)
                    nocc_j = len(idx_occ_j) + len(idx_frac_j)
                    # This computes Lij = einsum('mi,Pmn,nj->Pij', mo_coeff[km][:, :nocc_i].conj(), Lpq_ao,
                    # mo_coeff[kn][:, :nocc_j])
                    tmp = contract('Pmn,mi->Pin', Lpq_ao, mo_coeff[km][:, :nocc_i].conj())
                    Lij = contract('Pin,nj->Pij', tmp, mo_coeff[kn][:, :nocc_j])
                    tmp = None

                    if acfd is True:
                        if correction_only is True:
                            mo_occ_ij = cp.minimum(mo_occ[km][:nocc_i, None], mo_occ[kn][None, :nocc_j]) / 2.0
                            mo_occ_ij -= mo_occ[km][:nocc_i, None] * mo_occ[kn][None, :nocc_j] / 4.0
                        else:
                            mo_occ_ij = cp.minimum(mo_occ[km][:nocc_i, None], mo_occ[kn][None, :nocc_j]) / 2.0
                    else:
                        mo_occ_ij = mo_occ[km][:nocc_i, None] * mo_occ[kn][None, :nocc_j] / 4.0
                    Lij_occ = Lij * mo_occ_ij[None]
                    # ex -= cp.einsum('Lij,Lij->', Lij_occ.reshape(-1, nocc, nocc), Lij.reshape(-1, nocc, nocc).conj())
                    ex -= cp.cublas.dotc(Lij_occ.ravel(), Lij.ravel())
                else:
                    # This computes Lij = einsum('mi,Pmn,nj->Pij', mo_coeff[km][:, :nocc].conj(), Lpq_ao,
                    # mo_coeff[kn][:, :nocc])
                    tmp = contract('Pmn,mi->Pin', Lpq_ao, mo_coeff[km][:, :nocc].conj())
                    Lij = contract('Pin,nj->Pij', tmp, mo_coeff[kn][:, :nocc])
                    tmp = None
                    # ex -= cp.einsum('Lij,Lij->', Lij.reshape(-1, nocc, nocc), Lij.reshape(-1, nocc, nocc).conj())
                    ex -= cp.cublas.dotc(Lij.ravel(), Lij.ravel())
        Lpq_kL = Lpq_ao = out_buf = unpack_buf = None

    ex = ex.real
    ex /= nkpts**2

    if rpa._scf.exxdiv == 'ewald' and rpa._scf.cell.dimension != 0:
        madelung = tools.pbc.madelung(rpa._scf.cell, kpts)
        exxdiv_shift = madelung * cp.sum(mo_occ**2) / (4.0 * nkpts)
        if not correction_only:
            ex -= exxdiv_shift
        if acfd is True:
            for k in range(nkpts):
                idx_occ, idx_frac, _ = get_idx_metal(mo_occ[k])
                f_i = mo_occ[k][:(len(idx_occ) + len(idx_frac))] / 2.0
                ex -= madelung * cp.sum(f_i - f_i * f_i) / nkpts

    return ex


class KRPA(lib_cpu.StreamObject):
    def __init__(self, mf, frozen=None, auxbasis=None):
        self.mol = mf.mol  # mol object
        self._scf = mf  # mean-field object
        self.verbose = self.mol.verbose  # verbose level
        self.stdout = self.mol.stdout  # standard output
        self.max_memory = mf.max_memory  # max memory in MB

        # options
        self.frozen = frozen  # frozen orbital options
        self.grids_alg = 'legendre'  # algorithm to generate grids
        self.auxbasis = auxbasis  # auxiliary basis for DF
        self.nw = 40  # number of grids
        self.outcore = False  # low-memory routine
        self.segsize = 50  # number of orbitals in one segment for outcore
        self.fc = False  # finite-size correction
        self.fc_grid = False  # grids for finite-size correction
        self.acfd_exx = False  # calculate ACFD exchange energy

        # don't modify the following attributes, they are not input options
        self._nocc = None  # number of occupied orbitals
        self._nmo = None  # number of orbitals (exclude frozen orbitals)
        self.kpts = mf.kpts  # k-points
        self.nkpts = len(self.kpts)  # number of k-points
        # host copies, pyscf.pbc.mp.kmp2.get_frozen_mask reads them with np.asarray
        self.mo_energy = np.array(cp.asnumpy(mf.mo_energy))  # orbital energy
        self.mo_coeff = np.array(cp.asnumpy(mf.mo_coeff))  # orbital coefficient
        self.mo_occ = np.array(cp.asnumpy(mf.mo_occ))  # occupation number
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
        nkpts = self.nkpts
        log.info(f'RPA nocc = {nocc}, nvir = {nvir}, nkpts = {nkpts}')
        if self.frozen is not None:
            log.info(f'frozen orbitals = {str(self.frozen)}')
        log.info('grid type = %s', self.grids_alg)
        log.info('number of grids = %d', self.nw)
        log.info('outcore mode = %s', self.outcore)
        if self.outcore is True:
            log.info('outcore segment size = %d', self.segsize)
        log.info('RPA finite size corrections = %s', self.fc)
        log.info('ACFD exchange energy = %s', self.acfd_exx)
        log.info('')
        return

    @property
    def nocc(self):
        frozen_mask = get_frozen_mask(self)
        nkpts = len(self._scf.mo_energy)
        nelec = 0.0
        for k in range(nkpts):
            nelec += cp.sum(self._scf.mo_occ[k][frozen_mask[k]])
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

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def kernel(self, mo_energy=None, mo_coeff=None, with_e_hf=None):
        """RPA correlation and total energy

        Calculated total energy, HF energy and RPA correlation energy
        are stored in self.e_tot, self.e_hf, self.e_corr

        Parameters
        ----------
        mo_energy : double array
            molecular orbital energies
        mo_coeff : double ndarray
            molecular orbital coefficients
        with_e_hf : float, optional
            If given, overrides the HF energy computation.

        Returns
        -------
        e_tot : float
            RPA total energy
        e_hf : float
            HF energy (exact exchange for given mo_coeff)
        e_corr : float
            RPA correlation energy
        """
        if mo_coeff is None:
            mo_coeff = _mo_frozen(self, self._scf.mo_coeff)
        if mo_energy is None:
            mo_energy = _mo_energy_frozen(self, self._scf.mo_energy)

        cput0 = (time.process_time(), time.perf_counter())
        self.initialize_df(auxbasis=self.auxbasis)
        self.dump_flags()
        self.e_tot, self.e_hf, self.e_corr = kernel(self, mo_energy, with_e_hf=with_e_hf)

        logger.timer(self, 'RPA', *cput0)
        return self.e_tot, self.e_hf, self.e_corr

    def initialize_df(self, auxbasis=None):
        """Initialize density fitting.

        Parameters
        ----------
        auxbasis : str, optional
            name of auxiliary basis set, by default None
        """
        # KRPA must use GDF integrals. A mean field without density fitting has with_df = None or
        # the FFTDF that gpu4pyscf KRHF/KRKS create by default, build a k-point GDF in that case.
        if isinstance(getattr(self._scf, 'with_df', None), df.GDF):
            self.with_df = self._scf.with_df
        else:
            self.with_df = df.GDF(self._scf.mol, kpts=self.kpts)
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

    def get_grids(self, alg=None, nw=None, mo_energy=None):
        """Generate grids for integration.

        Parameters
        ----------
        alg : str, optional
            algorithm for generating grids, by default None
        nw : int, optional
            number of grids, by default None
        mo_energy : double 2d array, optional
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
            mo_energy = _mo_energy_frozen(self, self._scf.mo_energy)
        if alg == 'legendre':
            nw = 40 if nw is None else nw
            freqs, wts = _get_scaled_legendre_roots(nw)
        else:
            raise NotImplementedError('Grids algorithm not implemented!')

        freqs = cp.asarray(freqs)
        wts = cp.asarray(wts)
        return freqs, wts

    def get_q_mesh(self, mo_energy, mo_coeff):
        """Get q-mesh for finite size correction.
        Equation 39-42 in doi.org/10.1021/acs.jctc.0c00704

        Parameters
        ----------
        mo_energy : double 2d array
            orbital energy
        mo_coeff : double 3d array
            coefficient from AO to MO

        Returns
        -------
        qij : double 1d array
            q-mesh grids
        q_abs : double 1d array
            absolute positions of q-mesh grids
        nq_pts : init
            number of q-mesh grids
        """
        nocc = self.nocc
        nmo = self.nmo
        nkpts = self.nkpts
        # Set up q mesh for q->0 finite size correction
        if not self.fc_grid:
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
        q_abs = self.mol.get_abs_kpts(q_pts)

        # Get qij = 1/sqrt(Omega) * < psi_{ik} | e^{iqr} | psi_{ak-q} > at q: (nkpts, nocc, nvir)
        qij = cp.zeros(shape=[nq_pts, nkpts, nocc, nmo - nocc], dtype=cp.complex128)

        # qij is linear in q: the grid and AO work is done once for all q points
        qij_x = get_qij_components(self, mo_energy, mo_coeff)
        for k in range(nq_pts):
            qij[k] = cp.einsum('x,xkia->kia', cp.asarray(q_abs[k]).astype(cp.complex128), qij_x)

        return qij, q_abs, nq_pts

    def get_acfd_exx(self, correction_only=False):
        """Calculate ACFD exchange energy.

        Parameters
        ----------
        correction_only : bool
            only return the correction term

        Returns
        -------
        ex_acfd : double
            ACFD exchange energy
        """
        ex_acfd = get_rpa_exx(self, acfd=True, correction_only=correction_only)
        return ex_acfd

    to_cpu = utils.to_cpu
    to_gpu = utils.to_gpu
    device = utils.device
