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
Geometric direct minimization (GDM) SCF solver
'''

# Attention: The orbital gradient (also called kappa) in GDM implementation refers to the upper right (nocc, nvir)
# part of the exponential matrix. It is exactly the same as the V_ov matrix defined in the GDM paper.
# It is the NOT consistent with the orbital gradient definition in SOSCF, they differ by negative transpose.

import numpy as np
import cupy as cp
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import eigh
from gpu4pyscf.pbc.scf.khf import kpt_pair_reverse_map, PBC_LINDEP_MO_ENERGY_OFFSET

GDM_LBFGS_Y_DOT_S_ZERO_THRESHOLD = 1e-12

def pack_nocc_nvir_shape_vector(vector, nocc, nmo, nkpts = None):
    if nkpts is None:
        assert isinstance(nmo, int)
        if isinstance(nocc, int):
            assert nocc < nmo
            assert vector.shape == (nocc, nmo - nocc)
            return vector.flatten(order = "C")
        else:
            nocca, noccb = nocc
            assert isinstance(nocca, int) and isinstance(noccb, int)
            assert nocca < nmo and noccb < nmo
            va, vb = vector
            assert va.shape == (nocca, nmo - nocca)
            assert vb.shape == (noccb, nmo - noccb)
            return cp.concatenate([va.flatten(order = "C"), vb.flatten(order = "C")])
    else:
        assert len(nocc) > 0
        if isinstance(nocc[0], int):
            assert len(nocc) == nkpts
            assert len(nmo) == nkpts
            assert len(vector) == nkpts
            flattened_v = []
            for k in range(nkpts):
                assert isinstance(nocc[k], int) and nocc[k] < nmo[k]
                assert vector[k].shape == (nocc[k], nmo[k] - nocc[k])
                flattened_v.append(vector[k].flatten(order = "C"))
            return cp.concatenate(flattened_v)
        else:
            nocca, noccb = nocc
            nmoa, nmob = nmo
            va, vb = vector
            assert len(nocca) == nkpts and len(noccb) == nkpts
            assert len(nmoa) == nkpts and len(nmob) == nkpts
            assert len(va) == nkpts and len(vb) == nkpts
            flattened_v = []
            for k in range(nkpts):
                assert isinstance(nocca[k], int) and nocca[k] < nmoa[k]
                assert va[k].shape == (nocca[k], nmoa[k] - nocca[k])
                flattened_v.append(va[k].flatten(order = "C"))
            for k in range(nkpts):
                assert isinstance(noccb[k], int) and noccb[k] < nmob[k]
                assert vb[k].shape == (noccb[k], nmob[k] - noccb[k])
                flattened_v.append(vb[k].flatten(order = "C"))
            return cp.concatenate(flattened_v)

def unpack_nocc_nvir_shape_vector(vector, nocc, nmo, nkpts = None):
    assert vector.ndim == 1
    if nkpts is None:
        assert isinstance(nmo, int)
        if isinstance(nocc, int):
            assert nocc < nmo
            assert vector.size == nocc * (nmo - nocc)
            return vector.reshape(nocc, nmo - nocc)
        else:
            nocca, noccb = nocc
            assert isinstance(nocca, int) and isinstance(noccb, int)
            assert nocca < nmo and noccb < nmo
            assert vector.size == nocca * (nmo - nocca) + noccb * (nmo - noccb)
            va = vector[:nocca * (nmo - nocca)]
            vb = vector[nocca * (nmo - nocca):]
            return (va.reshape(nocca, nmo - nocca), vb.reshape(noccb, nmo - noccb))
    else:
        assert len(nocc) > 0
        if isinstance(nocc[0], int):
            assert len(nocc) == nkpts
            assert len(nmo) == nkpts
            vks = []
            offset_k = 0
            for k in range(nkpts):
                assert isinstance(nocc[k], int) and nocc[k] < nmo[k]
                vk = vector[offset_k : offset_k + nocc[k] * (nmo[k] - nocc[k])]
                vk = vk.reshape(nocc[k], nmo[k] - nocc[k])
                vks.append(vk)
                offset_k += nocc[k] * (nmo[k] - nocc[k])
            assert offset_k == vector.size
            return vks
        else:
            nocca, noccb = nocc
            nmoa, nmob = nmo
            assert len(nocca) == nkpts and len(noccb) == nkpts
            assert len(nmoa) == nkpts and len(nmob) == nkpts
            va_ks = []
            vb_ks = []
            offset_k = 0
            for k in range(nkpts):
                assert isinstance(nocca[k], int) and nocca[k] < nmoa[k]
                vk = vector[offset_k : offset_k + nocca[k] * (nmoa[k] - nocca[k])]
                vk = vk.reshape(nocca[k], nmoa[k] - nocca[k])
                va_ks.append(vk)
                offset_k += nocca[k] * (nmoa[k] - nocca[k])
            for k in range(nkpts):
                assert isinstance(noccb[k], int) and noccb[k] < nmob[k]
                vk = vector[offset_k : offset_k + noccb[k] * (nmob[k] - noccb[k])]
                vk = vk.reshape(noccb[k], nmob[k] - noccb[k])
                vb_ks.append(vk)
                offset_k += noccb[k] * (nmob[k] - noccb[k])
            assert offset_k == vector.size
            return va_ks, vb_ks

def apply_exp_U(mo_coeff, nocc, nmo, kappa, nkpts = None, conj_k_idx = None):
    """
        Computes C @ U where
        U = exp( 0 & kappa \\ -kappa^dagger & 0 )
    """
    new_mo_coeff = cp.zeros_like(mo_coeff)
    if nkpts is None:
        assert isinstance(nmo, int)
        if isinstance(nocc, int):
            assert nocc < nmo
            assert mo_coeff.ndim == 2
            assert kappa.shape == (nocc, nmo - nocc)
            assert mo_coeff.dtype == kappa.dtype

            mocc = mo_coeff[:, :nocc]
            mvir = mo_coeff[:, nocc:]
            U, sigma, VT = cp.linalg.svd(kappa, full_matrices=False)
            V = VT.conj().T
            del VT

            cos_sigma = cp.cos(sigma) - 1
            sin_sigma = cp.sin(sigma)
            moccU = mocc @ U
            mvirV = mvir @ V

            new_mo_coeff[:, :nocc] = moccU @ (cos_sigma * U).conj().T + mocc - mvirV @ (sin_sigma * U).conj().T
            new_mo_coeff[:, nocc:] = moccU @ (sin_sigma * V).conj().T + mvirV @ (cos_sigma * V).conj().T + mvir
            del U, V, sigma, moccU, mvirV, mocc, mvir, cos_sigma, sin_sigma
        else:
            assert len(nocc) == 2
            assert len(kappa) == 2
            assert mo_coeff.ndim == 3 and mo_coeff.shape[0] == 2
            for s in range(2):
                assert isinstance(nocc[s], int) and nocc[s] < nmo
                assert kappa[s].shape == (nocc[s], nmo - nocc[s])
                assert mo_coeff.dtype == kappa[s].dtype

                mocc = mo_coeff[s, :, :nocc[s]]
                mvir = mo_coeff[s, :, nocc[s]:]
                U, sigma, VT = cp.linalg.svd(kappa[s], full_matrices=False)
                V = VT.conj().T
                del VT
                cos_sigma = cp.cos(sigma) - 1
                sin_sigma = cp.sin(sigma)
                moccU = mocc @ U
                mvirV = mvir @ V
                new_mo_coeff[s, :, :nocc[s]] = moccU @ (cos_sigma * U).conj().T + mocc - mvirV @ (sin_sigma * U).conj().T
                new_mo_coeff[s, :, nocc[s]:] = moccU @ (sin_sigma * V).conj().T + mvirV @ (cos_sigma * V).conj().T + mvir
                del U, V, sigma, moccU, mvirV, mocc, mvir, cos_sigma, sin_sigma
    else:
        assert len(nocc) > 0
        if isinstance(nocc[0], int):
            assert len(nocc) == nkpts
            assert len(nmo) == nkpts
            assert len(kappa) == nkpts
            assert mo_coeff.ndim == 3 and mo_coeff.shape[0] == nkpts
            for k in range(nkpts):
                assert isinstance(nocc[k], int) and nocc[k] < nmo[k]
                kappa_k = kappa[k]
                assert kappa_k.shape == (nocc[k], nmo[k] - nocc[k])
                assert mo_coeff.dtype == kappa_k.dtype

                if (conj_k_idx is not None) and (conj_k_idx[k] >= 0) and (nocc[k] == nocc[conj_k_idx[k]]):
                    new_mo_coeff[k] = new_mo_coeff[conj_k_idx[k]].conj()
                    continue

                mocc = mo_coeff[k, :, :nocc[k]]
                mvir = mo_coeff[k, :, nocc[k]:nmo[k]]
                if (conj_k_idx is not None) and (k == -conj_k_idx[k] - 1) and cp.iscomplexobj(mo_coeff):
                    mocc = mocc.real
                    mvir = mvir.real
                    kappa_k = kappa_k.real
                U, sigma, VT = cp.linalg.svd(kappa_k, full_matrices=False)
                V = VT.conj().T
                del VT
                cos_sigma = cp.cos(sigma) - 1
                sin_sigma = cp.sin(sigma)
                moccU = mocc @ U
                mvirV = mvir @ V
                new_mo_coeff[k, :, :nocc[k]] = moccU @ (cos_sigma * U).conj().T + mocc - mvirV @ (sin_sigma * U).conj().T
                new_mo_coeff[k, :, nocc[k]:nmo[k]] = moccU @ (sin_sigma * V).conj().T + mvirV @ (cos_sigma * V).conj().T + mvir
                new_mo_coeff[k, :, nmo[k]:] = 0
                del U, V, sigma, moccU, mvirV, mocc, mvir, cos_sigma, sin_sigma
        else:
            assert len(nocc) == 2
            assert len(nmo) == 2
            assert len(kappa) == 2
            assert mo_coeff.ndim == 4 and mo_coeff.shape[0:2] == (2, nkpts)
            for s in range(2):
                assert len(nocc[s]) == nkpts
                assert len(kappa[s]) == nkpts
                for k in range(nkpts):
                    nocc_sk = nocc[s][k]
                    nmo_sk = nmo[s][k]
                    kappa_sk = kappa[s][k]
                    assert isinstance(nocc_sk, int) and nocc_sk < nmo_sk
                    assert kappa_sk.shape == (nocc_sk, nmo_sk - nocc_sk)
                    assert mo_coeff.dtype == kappa_sk.dtype

                    if (conj_k_idx is not None) and (conj_k_idx[k] >= 0) and (nocc_sk == nocc[s][conj_k_idx[k]]):
                        new_mo_coeff[s, k] = new_mo_coeff[s, conj_k_idx[k]].conj()
                        continue

                    mocc = mo_coeff[s, k, :, :nocc_sk]
                    mvir = mo_coeff[s, k, :, nocc_sk:nmo_sk]
                    if (conj_k_idx is not None) and (k == -conj_k_idx[k] - 1) and cp.iscomplexobj(mo_coeff):
                        mocc = mocc.real
                        mvir = mvir.real
                        kappa_sk = kappa_sk.real
                    U, sigma, VT = cp.linalg.svd(kappa_sk, full_matrices=False)
                    V = VT.conj().T
                    del VT
                    cos_sigma = cp.cos(sigma) - 1
                    sin_sigma = cp.sin(sigma)
                    moccU = mocc @ U
                    mvirV = mvir @ V
                    new_mo_coeff[s, k, :, :nocc_sk] = moccU @ (cos_sigma * U).conj().T + mocc - mvirV @ (sin_sigma * U).conj().T
                    new_mo_coeff[s, k, :, nocc_sk:nmo_sk] = moccU @ (sin_sigma * V).conj().T + mvirV @ (cos_sigma * V).conj().T + mvir
                    new_mo_coeff[s, k, :, nmo_sk:] = 0
                    del U, V, sigma, moccU, mvirV, mocc, mvir, cos_sigma, sin_sigma

    return new_mo_coeff

def packed_apply_exp_U(mo_coeff, nocc, nmo, kappa, nkpts = None, conj_k_idx = None):
    return apply_exp_U(mo_coeff, nocc, nmo, unpack_nocc_nvir_shape_vector(kappa, nocc, nmo, nkpts), nkpts, conj_k_idx)

def pseudo_canonicalize(fock, mo_coeff, nocc, nmo, kappas, nkpts = None, conj_k_idx = None):
    '''
        Computes Coo^dagger X Cvv
        where Coo and Cvv diagonalize the oo and vv block of the Fock matrix

        This is equivalent to Eq 28 e^(H^dagger) V e^H, where V = ( 0 & kappa \\ -kappa^dagger & 0 ), e^H = ( Coo & 0 \\ 0 & Cvv )
    '''
    new_mo_coeff = cp.zeros_like(mo_coeff)
    if nkpts is None:
        assert isinstance(nmo, int)
        if isinstance(nocc, int):
            assert nocc < nmo
            assert mo_coeff.ndim == 2
            assert fock.ndim == 2

            mocc = mo_coeff[:, :nocc]
            mvir = mo_coeff[:, nocc:]

            Foo = mocc.conj().T @ fock @ mocc
            eoo, Coo = eigh(Foo)
            del Foo
            Fvv = mvir.conj().T @ fock @ mvir
            evv, Cvv = eigh(Fvv)
            del Fvv

            new_mo_energy = cp.zeros(nmo)
            new_mo_energy[:nocc] = eoo
            new_mo_energy[nocc:] = evv
            del eoo, evv
            new_mo_coeff[:, :nocc] = mocc @ Coo
            new_mo_coeff[:, nocc:] = mvir @ Cvv
            del mocc, mvir

            transformed_kappas = []
            for kappa in kappas:
                assert kappa.shape == (nocc, nmo - nocc)
                assert mo_coeff.dtype == kappa.dtype
                kappa = Coo.conj().T @ kappa @ Cvv
                transformed_kappas.append(kappa)
                del kappa
            del Coo, Cvv

            return new_mo_energy, new_mo_coeff, transformed_kappas
        else:
            assert len(nocc) == 2
            assert mo_coeff.ndim == 3 and mo_coeff.shape[0] == 2
            assert fock.ndim == 3 and fock.shape[0] == 2
            new_mo_energy = cp.zeros((2, nmo), dtype = cp.float64)
            transformed_kappas = []
            for s in range(2):
                assert isinstance(nocc[s], int) and nocc[s] < nmo

                mocc = mo_coeff[s, :, :nocc[s]]
                mvir = mo_coeff[s, :, nocc[s]:]
                Foo = mocc.conj().T @ fock[s] @ mocc
                eoo, Coo = eigh(Foo)
                del Foo
                Fvv = mvir.conj().T @ fock[s] @ mvir
                evv, Cvv = eigh(Fvv)
                del Fvv

                new_mo_energy[s, :nocc[s]] = eoo
                new_mo_energy[s, nocc[s]:] = evv
                del eoo, evv
                new_mo_coeff[s, :, :nocc[s]] = mocc @ Coo
                new_mo_coeff[s, :, nocc[s]:] = mvir @ Cvv
                del mocc, mvir

                for kappa in kappas:
                    assert len(kappa) == 2
                    kappa_s = kappa[s]
                    assert kappa_s.shape == (nocc[s], nmo - nocc[s])
                    assert mo_coeff.dtype == kappa_s.dtype
                    kappa_s = Coo.conj().T @ kappa_s @ Cvv
                    transformed_kappas.append(kappa_s)
                    del kappa_s
                del Coo, Cvv

            n_kappa = len(kappas)
            assert len(transformed_kappas) == 2 * n_kappa
            transformed_kappas = [ (transformed_kappas[i], transformed_kappas[i + n_kappa]) for i in range(n_kappa) ]
            return new_mo_energy, new_mo_coeff, transformed_kappas
    else:
        assert len(nocc) > 0
        if isinstance(nocc[0], int):
            assert len(nocc) == nkpts
            assert len(nmo) == nkpts
            assert mo_coeff.ndim == 3 and mo_coeff.shape[0] == nkpts
            assert fock.ndim == 3 and fock.shape[0] == nkpts
            new_mo_energy = cp.zeros((nkpts, mo_coeff.shape[-1]), dtype = cp.float64)
            n_kappa = len(kappas)
            transformed_kappas = [ [None] * nkpts for _ in range(n_kappa) ]
            for k in range(nkpts):
                assert isinstance(nocc[k], int) and nocc[k] < nmo[k]

                if (conj_k_idx is not None) and (conj_k_idx[k] >= 0) and (nocc[k] == nocc[conj_k_idx[k]]):
                    new_mo_energy[k] = new_mo_energy[conj_k_idx[k]]
                    new_mo_coeff[k] = new_mo_coeff[conj_k_idx[k]].conj()
                    Coo = Cvv = None
                else:
                    mocc = mo_coeff[k, :, :nocc[k]]
                    mvir = mo_coeff[k, :, nocc[k]:nmo[k]]
                    fock_k = fock[k]
                    if (conj_k_idx is not None) and (k == -conj_k_idx[k] - 1) and cp.iscomplexobj(mo_coeff):
                        mocc = mocc.real
                        mvir = mvir.real
                        fock_k = fock_k.real
                    Foo = mocc.conj().T @ fock_k @ mocc
                    eoo, Coo = eigh(Foo)
                    del Foo
                    Fvv = mvir.conj().T @ fock_k @ mvir
                    evv, Cvv = eigh(Fvv)
                    del Fvv

                    new_mo_energy[k, :nocc[k]] = eoo
                    new_mo_energy[k, nocc[k]:nmo[k]] = evv
                    new_mo_energy[k, nmo[k]:] = abs(evv.max()) * 2 + PBC_LINDEP_MO_ENERGY_OFFSET
                    del eoo, evv
                    new_mo_coeff[k, :, :nocc[k]] = mocc @ Coo
                    new_mo_coeff[k, :, nocc[k]:nmo[k]] = mvir @ Cvv
                    new_mo_coeff[k, :, nmo[k]:] = 0
                    del mocc, mvir

                for i_kappa in range(n_kappa):
                    kappa = kappas[i_kappa]
                    assert len(kappa) == nkpts
                    kappa_k = kappa[k]
                    assert kappa_k.shape == (nocc[k], nmo[k] - nocc[k])
                    assert mo_coeff.dtype == kappa_k.dtype
                    if (conj_k_idx is not None) and (conj_k_idx[k] >= 0) and (nocc[k] == nocc[conj_k_idx[k]]):
                        kappa_k = transformed_kappas[i_kappa][conj_k_idx[k]].conj()
                        assert kappa_k is not None
                    else:
                        kappa_k = Coo.conj().T @ kappa_k @ Cvv

                    if (conj_k_idx is not None) and (k == -conj_k_idx[k] - 1) and cp.iscomplexobj(kappa_k):
                        kappa_k.imag = 0
                    transformed_kappas[i_kappa][k] = kappa_k
                    del kappa_k
                del Coo, Cvv

            return new_mo_energy, new_mo_coeff, transformed_kappas
        else:
            assert len(nocc) == 2
            assert len(nmo) == 2
            assert mo_coeff.ndim == 4 and mo_coeff.shape[0:2] == (2, nkpts)
            assert fock.ndim == 4 and fock.shape[0:2] == (2, nkpts)
            new_mo_energy = cp.zeros((2, nkpts, mo_coeff.shape[-1]), dtype = cp.float64)
            n_kappa = len(kappas)
            transformed_kappas = [ ([None] * nkpts, [None] * nkpts) for _ in range(n_kappa) ]
            for s in range(2):
                assert len(nocc[s]) == nkpts
                assert len(nmo[s]) == nkpts
                for k in range(nkpts):
                    nocc_sk = nocc[s][k]
                    nmo_sk = nmo[s][k]
                    assert isinstance(nocc_sk, int) and nocc_sk < nmo_sk

                    if (conj_k_idx is not None) and (conj_k_idx[k] >= 0) and (nocc_sk == nocc[s][conj_k_idx[k]]):
                        new_mo_energy[s, k] = new_mo_energy[s, conj_k_idx[k]]
                        new_mo_coeff[s, k] = new_mo_coeff[s, conj_k_idx[k]].conj()
                        Coo = Cvv = None
                    else:
                        mocc = mo_coeff[s, k, :, :nocc_sk]
                        mvir = mo_coeff[s, k, :, nocc_sk:nmo_sk]
                        fock_sk = fock[s, k]
                        if (conj_k_idx is not None) and (k == -conj_k_idx[k] - 1) and cp.iscomplexobj(mo_coeff):
                            mocc = mocc.real
                            mvir = mvir.real
                            fock_sk = fock_sk.real
                        Foo = mocc.conj().T @ fock_sk @ mocc
                        eoo, Coo = eigh(Foo)
                        del Foo
                        Fvv = mvir.conj().T @ fock_sk @ mvir
                        evv, Cvv = eigh(Fvv)
                        del Fvv

                        new_mo_energy[s, k, :nocc_sk] = eoo
                        new_mo_energy[s, k, nocc_sk:nmo_sk] = evv
                        new_mo_energy[s, k, nmo_sk:] = abs(evv.max()) * 2 + PBC_LINDEP_MO_ENERGY_OFFSET
                        del eoo, evv
                        new_mo_coeff[s, k, :, :nocc_sk] = mocc @ Coo
                        new_mo_coeff[s, k, :, nocc_sk:nmo_sk] = mvir @ Cvv
                        new_mo_coeff[s, k, :, nmo_sk:] = 0
                        del mocc, mvir

                    for i_kappa in range(n_kappa):
                        kappa = kappas[i_kappa]
                        assert len(kappa) == 2
                        kappa_s = kappa[s]
                        assert len(kappa_s) == nkpts
                        kappa_sk = kappa_s[k]
                        assert kappa_sk.shape == (nocc_sk, nmo_sk - nocc_sk)
                        assert mo_coeff.dtype == kappa_sk.dtype
                        if (conj_k_idx is not None) and (conj_k_idx[k] >= 0) and (nocc_sk == nocc[s][conj_k_idx[k]]):
                            kappa_sk = transformed_kappas[i_kappa][s][conj_k_idx[k]].conj()
                            assert kappa_sk is not None
                        else:
                            kappa_sk = Coo.conj().T @ kappa_sk @ Cvv

                        if (conj_k_idx is not None) and (k == -conj_k_idx[k] - 1) and cp.iscomplexobj(kappa_sk):
                            kappa_sk.imag = 0
                        transformed_kappas[i_kappa][s][k] = kappa_sk
                        del kappa_s, kappa_sk
                    del Coo, Cvv

            return new_mo_energy, new_mo_coeff, transformed_kappas

def packed_pseudo_canonicalize(fock, mo_coeff, nocc, nmo, packed_kappas, nkpts = None, conj_k_idx = None):
    assert not any([kappa is None for kappa in packed_kappas])
    kappas = [unpack_nocc_nvir_shape_vector(kappa, nocc, nmo, nkpts) for kappa in packed_kappas]
    new_mo_energy, new_mo_coeff, new_kappas = pseudo_canonicalize(fock, mo_coeff, nocc, nmo, kappas, nkpts, conj_k_idx)
    packed_new_kappas = [pack_nocc_nvir_shape_vector(kappa, nocc, nmo, nkpts) for kappa in new_kappas]
    return new_mo_energy, new_mo_coeff, packed_new_kappas

def approximate_optimal_diagonal_hessian(mo_energy, nocc, nmo, delta_E, minimal_diagonal_hessian, nkpts = None):
    if nkpts is None:
        assert isinstance(nmo, int)
        if isinstance(nocc, int):
            assert nocc < nmo
            assert mo_energy.shape == (nmo,)

            mo_energy_gap = mo_energy[None, nocc:] - mo_energy[:nocc, None] # (nocc, nvir)
            # Factor of two explained in kappa_gradient() function
            mo_energy_gap = 2.0 * cp.maximum((mo_energy_gap + delta_E), minimal_diagonal_hessian)
            return mo_energy_gap
        else:
            nocca, noccb = nocc
            assert isinstance(nocca, int) and isinstance(noccb, int)
            assert nocca < nmo and noccb < nmo
            assert mo_energy.shape == (2, nmo)

            mo_energy_gapa = mo_energy[0, nocca:][None, :] - mo_energy[0, :nocca][:, None]
            mo_energy_gapa = 2.0 * cp.maximum((mo_energy_gapa + delta_E), minimal_diagonal_hessian)
            mo_energy_gapb = mo_energy[1, noccb:][None, :] - mo_energy[1, :noccb][:, None]
            mo_energy_gapb = 2.0 * cp.maximum((mo_energy_gapb + delta_E), minimal_diagonal_hessian)
            return mo_energy_gapa, mo_energy_gapb
    else:
        assert len(nocc) > 0
        if isinstance(nocc[0], int):
            assert len(nocc) == nkpts
            assert len(nmo) == nkpts
            assert mo_energy.shape[0] == nkpts
            mo_energy_gap = []
            for k in range(nkpts):
                assert isinstance(nocc[k], int) and nocc[k] < nmo[k]
                mo_energy_gap_k = mo_energy[k, nocc[k]:nmo[k]][None, :] - mo_energy[k, :nocc[k]][:, None]
                mo_energy_gap_k = 2.0 / nkpts * cp.maximum((mo_energy_gap_k + delta_E), minimal_diagonal_hessian)
                mo_energy_gap.append(mo_energy_gap_k)
            return mo_energy_gap
        else:
            nocca, noccb = nocc
            nmoa, nmob = nmo
            assert len(nocca) == nkpts and len(noccb) == nkpts
            assert len(nmoa) == nkpts and len(nmob) == nkpts
            assert mo_energy.shape[0:2] == (2, nkpts)
            mo_energy_gapa = []
            mo_energy_gapb = []
            for k in range(nkpts):
                assert isinstance(nocca[k], int) and nocca[k] < nmoa[k]
                mo_energy_gap_k = mo_energy[0, k, nocca[k]:nmoa[k]][None, :] - mo_energy[0, k, :nocca[k]][:, None]
                mo_energy_gap_k = 2.0 / nkpts * cp.maximum((mo_energy_gap_k + delta_E), minimal_diagonal_hessian)
                mo_energy_gapa.append(mo_energy_gap_k)
            for k in range(nkpts):
                assert isinstance(noccb[k], int) and noccb[k] < nmob[k]
                mo_energy_gap_k = mo_energy[1, k, noccb[k]:nmob[k]][None, :] - mo_energy[1, k, :noccb[k]][:, None]
                mo_energy_gap_k = 2.0 / nkpts * cp.maximum((mo_energy_gap_k + delta_E), minimal_diagonal_hessian)
                mo_energy_gapb.append(mo_energy_gap_k)
            return mo_energy_gapa, mo_energy_gapb

def packed_approximate_optimal_diagonal_hessian(mo_energy, nocc, nmo, delta_E, minimal_diagonal_hessian, nkpts = None):
    h = approximate_optimal_diagonal_hessian(mo_energy, nocc, nmo, delta_E, minimal_diagonal_hessian, nkpts)
    return pack_nocc_nvir_shape_vector(h, nocc, nmo, nkpts)

def kappa_gradient(mo_coeff, nocc, nmo, fock, nkpts = None, conj_k_idx = None):
    if nkpts is None:
        assert isinstance(nmo, int)
        if isinstance(nocc, int):
            assert nocc < nmo
            assert mo_coeff.ndim == 2
            assert fock.ndim == 2

            mocc = mo_coeff[:, :nocc]
            mvir = mo_coeff[:, nocc:]
            Fov = mocc.conj().T @ fock @ mvir
            # The following factor of -2.0 comes from:
            # dE(alpha * kappa)/dalpha = - Tr(F_{ov} kappa^dagger) - Tr(F_{vo} kappa) = -2 Re( Tr(F_{vo} kappa) )
            g = -2.0 * Fov
            return g
        else:
            nocca, noccb = nocc
            assert isinstance(nocca, int) and isinstance(noccb, int)
            assert nocca < nmo and noccb < nmo
            assert mo_coeff.ndim == 3 and mo_coeff.shape[0] == 2
            assert fock.ndim == 3 and fock.shape[0] == 2

            mocc = mo_coeff[0, :, :nocca]
            mvir = mo_coeff[0, :, nocca:]
            Fov = mocc.conj().T @ fock[0] @ mvir
            ga = -2.0 * Fov
            mocc = mo_coeff[1, :, :noccb]
            mvir = mo_coeff[1, :, noccb:]
            Fov = mocc.conj().T @ fock[1] @ mvir
            gb = -2.0 * Fov
            return ga, gb
    else:
        assert len(nocc) > 0
        if isinstance(nocc[0], int):
            assert len(nocc) == nkpts
            assert len(nmo) == nkpts
            assert mo_coeff.ndim == 3 and mo_coeff.shape[0] == nkpts
            assert fock.ndim == 3 and fock.shape[0] == nkpts
            g = [ None ] * nkpts
            for k in range(nkpts):
                assert isinstance(nocc[k], int) and nocc[k] < nmo[k]
                if (conj_k_idx is not None) and (conj_k_idx[k] >= 0) and (nocc[k] == nocc[conj_k_idx[k]]):
                    g[k] = g[conj_k_idx[k]].conj()
                    continue
                mocc = mo_coeff[k, :, :nocc[k]]
                mvir = mo_coeff[k, :, nocc[k]:nmo[k]]
                Fov = mocc.conj().T @ fock[k] @ mvir
                gk = -2.0 / nkpts * Fov
                if (conj_k_idx is not None) and (k == -conj_k_idx[k] - 1) and cp.iscomplexobj(gk):
                    gk.imag = 0
                g[k] = gk
            return g
        else:
            assert len(nocc) == 2
            assert len(nmo) == 2
            assert mo_coeff.ndim == 4 and mo_coeff.shape[0:2] == (2, nkpts)
            assert fock.ndim == 4 and fock.shape[0:2] == (2, nkpts)
            g = ([ None ] * nkpts, [ None ] * nkpts)
            for s in range(2):
                assert len(nocc[s]) == nkpts
                assert len(nmo[s]) == nkpts
                for k in range(nkpts):
                    nocc_sk = nocc[s][k]
                    nmo_sk = nmo[s][k]
                    assert isinstance(nocc_sk, int) and nocc_sk < nmo_sk
                    if (conj_k_idx is not None) and (conj_k_idx[k] >= 0) and (nocc_sk == nocc[s][conj_k_idx[k]]):
                        g[s][k] = g[s][conj_k_idx[k]].conj()
                        continue
                    mocc = mo_coeff[s, k, :, :nocc_sk]
                    mvir = mo_coeff[s, k, :, nocc_sk:nmo_sk]
                    Fov = mocc.conj().T @ fock[s, k] @ mvir
                    gk = -2.0 / nkpts * Fov
                    if (conj_k_idx is not None) and (k == -conj_k_idx[k] - 1) and cp.iscomplexobj(gk):
                        gk.imag = 0
                    g[s][k] = gk
            return g

def packed_kappa_gradient(mo_coeff, nocc, nmo, fock, nkpts = None, conj_k_idx = None):
    g = kappa_gradient(mo_coeff, nocc, nmo, fock, nkpts, conj_k_idx)
    return pack_nocc_nvir_shape_vector(g, nocc, nmo, nkpts)

def reorder_mo_coeff_by_occupation(mo_coeff, mo_energy, mo_occ):
    nmo_full = mo_coeff.shape[-1]
    assert mo_occ.shape[-1] == nmo_full
    assert mo_energy.shape == mo_occ.shape
    new_mo_coeff = mo_coeff.copy()
    new_mo_occ = cp.zeros_like(mo_occ)
    new_mo_energy = cp.zeros_like(mo_energy)
    if mo_coeff.ndim == 2:
        assert mo_occ.ndim == 1
        occidx = mo_occ > 0
        viridx = mo_occ == 0
        nocc = int(cp.sum(occidx))
        nvir = int(cp.sum(viridx))
        assert nocc > 0 and nvir > 0, "Occupied or virtual space goes to zero in RHF/RKS, likely a bug."
        assert cp.max(cp.abs(mo_occ[occidx] - cp.round(mo_occ[occidx]))) < 1e-12, "Fractional occupation not allowed."
        nocc = int(cp.sum(occidx))
        new_mo_coeff[:, :nocc] = mo_coeff[:, occidx]
        new_mo_coeff[:, nocc:] = mo_coeff[:, viridx]
        new_mo_occ[:nocc] = mo_occ[occidx]
        new_mo_energy[:nocc] = mo_energy[occidx]
        new_mo_energy[nocc:] = mo_energy[viridx]
    elif mo_coeff.ndim == 3:
        assert mo_occ.ndim == 2
        n_s_or_k = mo_coeff.shape[0]
        assert mo_occ.shape[0] == n_s_or_k
        for sk in range(n_s_or_k):
            occidx = mo_occ[sk] > 0
            viridx = mo_occ[sk] == 0
            nocc = int(cp.sum(occidx))
            nvir = int(cp.sum(viridx))
            assert nocc > 0 and nvir > 0, "Occupied or virtual space of one spin or one k point goes to zero. GDM does not allow this pattern yet."
            assert cp.max(cp.abs(mo_occ[sk][occidx] - cp.round(mo_occ[sk][occidx]))) < 1e-12, "Fractional occupation not allowed."
            new_mo_coeff[sk, :, :nocc] = mo_coeff[sk][:, occidx]
            new_mo_coeff[sk, :, nocc:] = mo_coeff[sk][:, viridx]
            new_mo_occ[sk, :nocc] = mo_occ[sk, occidx]
            new_mo_energy[sk, :nocc] = mo_energy[sk, occidx]
            new_mo_energy[sk, nocc:] = mo_energy[sk, viridx]
    elif mo_coeff.ndim == 4:
        assert mo_occ.ndim == 3
        n_s, n_k = mo_coeff.shape[0:2]
        assert mo_occ.shape[0:2] == (n_s, n_k)
        for s in range(n_s):
            for k in range(n_k):
                occidx = mo_occ[s, k] > 0
                viridx = mo_occ[s, k] == 0
                nocc = int(cp.sum(occidx))
                nvir = int(cp.sum(viridx))
                assert nocc > 0 and nvir > 0, "Occupied or virtual space of one spin or one k point goes to zero. GDM does not allow this pattern yet."
                assert cp.max(cp.abs(mo_occ[s, k][occidx] - cp.round(mo_occ[s, k][occidx]))) < 1e-12, "Fractional occupation not allowed."
                new_mo_coeff[s, k, :, :nocc] = mo_coeff[s, k][:, occidx]
                new_mo_coeff[s, k, :, nocc:] = mo_coeff[s, k][:, viridx]
                new_mo_occ[s, k, :nocc] = mo_occ[s, k, occidx]
                new_mo_energy[s, k, :nocc] = mo_energy[s, k, occidx]
                new_mo_energy[s, k, nocc:] = mo_energy[s, k, viridx]
    else:
        raise ValueError(f"mo_coeff.ndim = {mo_coeff.ndim} not supported.")
    return new_mo_coeff, new_mo_energy, new_mo_occ

def get_nocc(mo_occ, nkpts = None):
    nmo_full = mo_occ.shape[-1]
    if nkpts is None:
        if mo_occ.ndim == 1:
            assert mo_occ.shape == (nmo_full,)
            nocc = int(cp.sum(mo_occ == 2))
            nvir = int(cp.sum(mo_occ == 0))
            assert nocc + nvir == nmo_full, "Only integer occupations are supported for GDM yet."
            assert cp.all(mo_occ[:nocc] == 2), "Occupation number must be sorted."
            return nocc
        else:
            assert mo_occ.shape == (2, nmo_full)
            nocca = int(cp.sum(mo_occ[0] == 1))
            nvira = int(cp.sum(mo_occ[0] == 0))
            assert nocca + nvira == nmo_full, "Only integer occupations are supported for GDM yet."
            assert cp.all(mo_occ[0, :nocca] == 1), "Occupation number must be sorted."
            noccb = int(cp.sum(mo_occ[1] == 1))
            nvirb = int(cp.sum(mo_occ[1] == 0))
            assert noccb + nvirb == nmo_full, "Only integer occupations are supported for GDM yet."
            assert cp.all(mo_occ[1, :noccb] == 1), "Occupation number must be sorted."
            return (nocca, noccb)
    else:
        if mo_occ.ndim == 2:
            assert mo_occ.shape == (nkpts, nmo_full)
            nocc_list = []
            for k in range(nkpts):
                nocc = int(cp.sum(mo_occ[k] == 2))
                nvir = int(cp.sum(mo_occ[k] == 0))
                assert nocc + nvir == nmo_full, "Only integer occupations are supported for GDM yet."
                assert cp.all(mo_occ[k, :nocc] == 2), "Occupation number must be sorted."
                nocc_list.append(nocc)
            return nocc_list
        else:
            assert mo_occ.shape == (2, nkpts, nmo_full)
            nocca_list = []
            noccb_list = []
            for k in range(nkpts):
                nocca = int(cp.sum(mo_occ[0, k] == 1))
                nvira = int(cp.sum(mo_occ[0, k] == 0))
                assert nocca + nvira == nmo_full, "Only integer occupations are supported for GDM yet."
                assert cp.all(mo_occ[0, k, :nocca] == 1), "Occupation number must be sorted."
                nocca_list.append(nocca)
                noccb = int(cp.sum(mo_occ[1, k] == 1))
                nvirb = int(cp.sum(mo_occ[1, k] == 0))
                assert noccb + nvirb == nmo_full, "Only integer occupations are supported for GDM yet."
                assert cp.all(mo_occ[1, k, :noccb] == 1), "Occupation number must be sorted."
                noccb_list.append(noccb)
            return (nocca_list, noccb_list)

def get_nmo(mo_coeff, nkpts = None):
    nmo_full = int(mo_coeff.shape[-1])
    if nkpts is None:
        nmo = nmo_full
    else:
        if mo_coeff.ndim == 3:
            assert mo_coeff.shape[0] == nkpts
            nmo = [ nmo_full ] * nkpts
            for k in range(nkpts):
                n_lindep = 0
                for i_mo in range(nmo_full):
                    if cp.all(mo_coeff[k, :, i_mo] == 0):
                        n_lindep += 1
                nmo[k] -= n_lindep
                for i_mo in range(nmo[k]):
                    assert not cp.all(mo_coeff[k, :, i_mo] == 0)
                for i_mo in range(nmo[k], nmo_full):
                    assert cp.all(mo_coeff[k, :, i_mo] == 0)
        else:
            assert mo_coeff.ndim == 4
            assert mo_coeff.shape[0:2] == (2, nkpts)
            nmo = ([ nmo_full ] * nkpts, [ nmo_full ] * nkpts)
            for s in range(2):
                for k in range(nkpts):
                    n_lindep = 0
                    for i_mo in range(nmo_full):
                        if cp.all(mo_coeff[s, k, :, i_mo] == 0):
                            n_lindep += 1
                    nmo[s][k] -= n_lindep
                    for i_mo in range(nmo[s][k]):
                        assert not cp.all(mo_coeff[s, k, :, i_mo] == 0)
                    for i_mo in range(nmo[s][k], nmo_full):
                        assert cp.all(mo_coeff[s, k, :, i_mo] == 0)
    return nmo

def get_energy_and_fock(mf, mo_coeff, mo_occ, h1e, s1e):
    assert isinstance(mo_coeff, cp.ndarray)
    assert isinstance(mo_occ, cp.ndarray)
    assert isinstance(h1e, cp.ndarray)
    assert isinstance(s1e, cp.ndarray)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    vhf = mf.get_veff(mf.mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    fock = mf.get_fock(h1e, s1e, vhf, dm)
    return e_tot, fock

def lbfgs_get_direction(gradient, lbfgs_s_y, initial_inverse_hessian = None):
    '''
        Notations follow https://en.wikipedia.org/wiki/Limited-memory_BFGS

        lbfgs_s_y order is older first (smaller index)
    '''
    q = gradient.copy()
    assert q.ndim == 1
    alphas = []
    rhos = []
    for s, y in reversed(lbfgs_s_y):
        assert s.shape == q.shape and y.shape == q.shape
        y_dot_s = float(cp.vdot(y, s).real)
        nonzero_y_dot_s = y_dot_s > GDM_LBFGS_Y_DOT_S_ZERO_THRESHOLD * float(cp.linalg.norm(y)) * float(cp.linalg.norm(s))
        rho = 1 / y_dot_s if nonzero_y_dot_s else 0.0

        alpha = rho * float(cp.vdot(s, q).real)
        q -= alpha * y
        rhos.append(rho)
        alphas.append(alpha)
    rhos = reversed(rhos)
    alphas = reversed(alphas)
    # From now on, rhos and alphas orders are the same as lbfgs_s_y order

    if initial_inverse_hessian is None:
        z = q # initial_inverse_hessian = cp.eye(q.size)
    else:
        z = initial_inverse_hessian * q
    assert z.shape == q.shape

    for (s, y), alpha, rho in zip(lbfgs_s_y, alphas, rhos):
        beta = rho * float(cp.vdot(y, z).real)
        z += s * (alpha - beta)

    return -z

def _kernel(mf, dm0,
            minimal_diagonal_hessian,
            min_n_lbfgs_record,
            max_n_lbfgs_record,
            max_step_norm,
            line_search_c,
            line_search_tau,
            max_n_line_search_attempt,
            line_search_initial_scale,
            check_occupation_per_n_iteration):
    """
        Geometric Direct Minimization
        Implementation follows:
        Van Voorhis, T.; Head-Gordon, M. A geometric approach to direct minimization.
        Molecular Physics 2002, 100, 1713-1721. https://doi.org/10.1080/00268970110103642

        Except for the two-loop-recursion LBFGS and backtracking line search,
        which follow wikipedia pages:
        https://en.wikipedia.org/wiki/Limited-memory_BFGS
        https://en.wikipedia.org/wiki/Backtracking_line_search
    """
    verbose = mf.verbose
    log = logger.new_logger(mf, verbose)

    if dm0 is None:
        dm0 = mf.get_init_guess(mf.mol, mf.init_guess)
    dm0 = cp.asarray(dm0)

    h1e = cp.asarray(mf.get_hcore())
    s1e = cp.asarray(mf.get_ovlp())
    vhf0 = mf.get_veff(mf.mol, dm0)
    fock0 = mf.get_fock(h1e, s1e, vhf0, dm0)
    x_orth = mf.check_linear_dependency(s1e, log)
    mo_energy, mo_coeff = mf.eig(fock0, s1e, x=x_orth) # The initial density is converted to orbitals by one Roothaan diagonalization.
    del vhf0, fock0
    mo_occ = cp.asarray(mf.get_occ(mo_energy, mo_coeff))
    e_tot, fock = get_energy_and_fock(mf, mo_coeff, mo_occ, h1e, s1e)

    from gpu4pyscf.pbc.scf.hf  import RHF as PBC_RHF_NOK
    from gpu4pyscf.pbc.scf.uhf import UHF as PBC_UHF_NOK
    if isinstance(mf, PBC_RHF_NOK) or isinstance(mf, PBC_UHF_NOK):
        raise NotImplementedError("pbc.scf.RHF or pbc.scf.UHF are not supported for GDM. Please use pbc.scf.KRHF or pbc.scf.KUHF with kpts=[[0, 0, 0]] instead.")

    kpts = getattr(mf, 'kpts', None)
    if kpts is not None:
        nkpts = len(kpts)
        kpairs = mf.iter_kpt_pairs()
        conj_k_idx = kpt_pair_reverse_map(nkpts, kpairs)
    else:
        nkpts = None
        conj_k_idx = None
    nocc = get_nocc(mo_occ, nkpts)
    nmo = get_nmo(mo_coeff, nkpts)

    if mf.conv_tol_grad is None:
        mf.conv_tol_grad = mf.conv_tol**0.5
        log.info('Set gradient conv threshold to %g', mf.conv_tol_grad)

    print_spin = hasattr(mf, 'spin_square') and callable(mf.spin_square) and mf.istype('UHF') # Note: for KUHF, mf.istype('UHF') is False
    if not print_spin:
        log.info(" Cycle               Energy   Gradient Norm    Step")
    else:
        log.info(" Cycle               Energy   Gradient Norm      <S^2>    Step")

    lbfgs_s_y = []
    previous_gradient = previous_step = None
    previous_e_tot = None
    step_type = "Uninitialized error! Please report a bug if you see this line."

    converged = False
    assert mf.max_cycle > 1

    for cycle in range(mf.max_cycle):
        flatten_to_transform = [s_or_y for sy in lbfgs_s_y for s_or_y in sy]
        if previous_gradient is not None and previous_step is not None:
            flatten_to_transform.extend([previous_gradient, previous_step])
        mo_energy, mo_coeff, flatten_transformed = packed_pseudo_canonicalize(fock, mo_coeff, nocc, nmo, flatten_to_transform, nkpts, conj_k_idx)
        del flatten_to_transform

        if previous_gradient is not None and previous_step is not None:
            previous_step = flatten_transformed.pop()
            previous_gradient = flatten_transformed.pop()

        transformed_s_y = []
        for i in range(0, len(flatten_transformed), 2):
            s, y = flatten_transformed[i], flatten_transformed[i + 1]
            s_dot_y = float(cp.vdot(s, y).real)
            if not s_dot_y > GDM_LBFGS_Y_DOT_S_ZERO_THRESHOLD * float(cp.linalg.norm(s)) * float(cp.linalg.norm(y)):
                continue
            transformed_s_y.append((s, y))
        del flatten_transformed
        lbfgs_s_y = transformed_s_y
        del transformed_s_y

        gradient = packed_kappa_gradient(mo_coeff, nocc, nmo, fock, nkpts, conj_k_idx)
        gradient_norm = float(cp.linalg.norm(gradient))
        if nkpts is not None:
            gradient_norm *= cp.sqrt(nkpts) # gradient_norm = sqrt(1/nkpts * sum_k g_k^2), and g_k already includes a factor of 1/nkpts

        if nkpts is not None and cycle % check_occupation_per_n_iteration == 0:
            new_mo_occ = cp.asarray(mf.get_occ(mo_energy, mo_coeff))
            if cp.max(cp.abs(new_mo_occ - mo_occ)) > 1e-10:
                log.info("MO occupation pattern changed. GDM reset.")

                lbfgs_s_y = []
                previous_gradient = previous_step = None
                previous_e_tot = None
                step_type = "Uninitialized error! Please report a bug if you see this line."

                mo_coeff, mo_energy, mo_occ = reorder_mo_coeff_by_occupation(mo_coeff, mo_energy, new_mo_occ)
                del new_mo_occ
                nocc = get_nocc(mo_occ, nkpts)
                e_tot, fock = get_energy_and_fock(mf, mo_coeff, mo_occ, h1e, s1e)

                gradient = packed_kappa_gradient(mo_coeff, nocc, nmo, fock, nkpts, conj_k_idx)
                gradient_norm = float(cp.linalg.norm(gradient)) * cp.sqrt(nkpts)

        if not print_spin:
            spin_text = ""
        else:
            assert len(nocc) == 2
            assert mo_coeff.ndim == 3 and mo_coeff.shape[0] == 2 # KUHF not supported yet
            ss, _ = mf.spin_square((mo_coeff[0, :, :nocc[0]], mo_coeff[1, :, :nocc[1]]))
            spin_text = f" {ss:10.4f}"

        if previous_e_tot is None:
            log.info(f"{cycle + 1:6d} {e_tot:20.12f} {gradient_norm:15.3e}{spin_text}  Roothaan")
        else:
            delta_E = e_tot - previous_e_tot
            if gradient_norm < mf.conv_tol_grad and abs(delta_E) < mf.conv_tol:
                converged = True
                log.info(f"{cycle + 1:6d} {e_tot:20.12f} {gradient_norm:15.3e}{spin_text}  Converged")
                break
            else:
                log.info(f"{cycle + 1:6d} {e_tot:20.12f} {gradient_norm:15.3e}{spin_text}  {step_type}")

        if previous_gradient is not None:
            y = gradient - previous_gradient
            s = previous_step
            s_dot_y = float(cp.vdot(s, y).real)
            if s_dot_y > GDM_LBFGS_Y_DOT_S_ZERO_THRESHOLD * float(cp.linalg.norm(s)) * float(cp.linalg.norm(y)):
                lbfgs_s_y.append((s, y))
                lbfgs_s_y = lbfgs_s_y[-max_n_lbfgs_record:]

        delta_E = 0.0 if previous_e_tot is None else e_tot - previous_e_tot
        approx_hess_diag = packed_approximate_optimal_diagonal_hessian(mo_energy, nocc, nmo, delta_E, minimal_diagonal_hessian, nkpts)

        if len(lbfgs_s_y) < min_n_lbfgs_record:
            step = -gradient / approx_hess_diag
            step_type = "Descent"
        else:
            sqrt_approx_hess_diag = cp.sqrt(approx_hess_diag)
            weighted_lbfgs_s_y = [ (s * sqrt_approx_hess_diag, y / sqrt_approx_hess_diag) for s, y in lbfgs_s_y ]
            weighted_gradient = gradient / sqrt_approx_hess_diag
            weighted_step = lbfgs_get_direction(weighted_gradient, weighted_lbfgs_s_y)
            step = weighted_step / sqrt_approx_hess_diag
            step_type = "L-BFGS"
            if float(cp.vdot(gradient, step).real) >= 0.0:
                lbfgs_s_y.clear()
                step = -gradient / approx_hess_diag
                step_type = "Descent Reset"

        step_norm = float(cp.linalg.norm(step))
        if nkpts is not None:
            step_norm /= cp.sqrt(nkpts) # step_norm = sqrt(1/nkpts * sum_k step_k^2)
        if step_norm > max_step_norm:
            step *= max_step_norm / step_norm
            step_type = "Clipped " + step_type

        # Backtracking line search
        # Notation follows https://en.wikipedia.org/wiki/Backtracking_line_search
        assert 0.0 < line_search_tau and line_search_tau < 1.0
        assert 0.0 < line_search_c and line_search_c < 1.0

        line_search_succeed = False
        line_search_slope = float(cp.vdot(gradient, step).real) # It's negative!
        line_search_scale = line_search_initial_scale
        for i_line_search_attempt in range(max_n_line_search_attempt + 1):
            line_search_step = line_search_scale * step
            line_search_mo_coeff = packed_apply_exp_U(mo_coeff, nocc, nmo, line_search_step, nkpts, conj_k_idx)
            line_search_energy, line_search_fock = get_energy_and_fock(mf, line_search_mo_coeff, mo_occ, h1e, s1e)
            if e_tot - line_search_energy >= -line_search_c * line_search_scale * line_search_slope:
                line_search_succeed = True
                break
            line_search_scale *= line_search_tau

        if i_line_search_attempt > 0:
            step_type = f"Line search ({i_line_search_attempt})"

        if not line_search_succeed:
            lbfgs_s_y.clear()
            log.info("Line search failed! However we accept the last (smallest) line search step and keep going. LBFGS record cleared.")

        previous_e_tot = e_tot
        e_tot = line_search_energy
        mo_coeff = line_search_mo_coeff
        fock = line_search_fock
        previous_step = line_search_step
        previous_gradient = gradient

    mo_energy, mo_coeff, _ = pseudo_canonicalize(fock, mo_coeff, nocc, nmo, [], nkpts, conj_k_idx)
    e_tot, fock = get_energy_and_fock(mf, mo_coeff, mo_occ, h1e, s1e)

    if nkpts is not None:
        new_mo_occ = cp.asarray(mf.get_occ(mo_energy, mo_coeff))
        if cp.max(cp.abs(new_mo_occ - mo_occ)) > 1e-10:
            log.warn("MO occupation number changed after the fixed-occupation GDM converged. "
                     "Please restart from the returned orbitals.")
            converged = False

    mf.converged = converged
    mf.cycles = cycle + 1
    mf.e_tot = e_tot
    mf.mo_energy = mo_energy
    mf.mo_coeff = mo_coeff
    mf.mo_occ = mo_occ

    return e_tot

class GDM:
    _keys = {}

    def __init__(self, mf):
        self.mf = mf

        # Approximated hessian parameters
        self.minimal_diagonal_hessian = 1e-4

        # L-BFGS parameters
        self.min_n_lbfgs_record = 1
        self.max_n_lbfgs_record = 20
        self.max_step_norm = 0.5

        # Line search parameters
        self.line_search_c = 1e-4
        self.line_search_tau = 0.5
        self.max_n_line_search_attempt = 8
        self.line_search_initial_scale = 1.0

        # PBC related parameters
        self.check_occupation_per_n_iteration = 10

    to_gpu = utils.to_gpu
    device = utils.device

    def to_cpu(self):
        raise NotImplementedError("GDM not implemented in CPU mode yet")

    def build(self, mol=None):
        self.mf.build(mol)
        return self

    def reset(self, mol=None):
        self.mf.reset(mol)
        return self

    def undo_gdm(self):
        return self.mf

    def error_undo_first(self):
        raise ValueError("Please call undo_gdm() first to get the mean-field object back!")
    nuc_grad_method = error_undo_first
    Gradients = error_undo_first
    Hessian = error_undo_first

    def kernel(self, dm0=None):
        return _kernel(self.mf, dm0,
                       self.minimal_diagonal_hessian,
                       self.min_n_lbfgs_record,
                       self.max_n_lbfgs_record,
                       self.max_step_norm,
                       self.line_search_c,
                       self.line_search_tau,
                       self.max_n_line_search_attempt,
                       self.line_search_initial_scale,
                       self.check_occupation_per_n_iteration)

def gdm(mf):
    if isinstance(mf, GDM):
        return mf

    from gpu4pyscf.scf.hf import SCF
    assert isinstance(mf, SCF)

    if mf.istype('RHF'):
        pass
    elif mf.istype('ROHF'):
        raise NotImplementedError("GDM does not support ROHF yet")
    elif mf.istype('UHF'):
        pass
    elif mf.istype('GHF'):
        raise NotImplementedError("GDM does not support GHF yet")
    elif mf.istype('RDHF'):
        raise NotImplementedError("GDM does not support RDHF yet")
    elif mf.istype('DHF'):
        raise NotImplementedError("GDM does not support DHF yet")
    elif mf.istype('KRHF'):
        pass
    elif mf.istype('KUHF'):
        pass
    elif mf.istype('KGHF'):
        raise NotImplementedError("GDM does not support KGHF yet")
    else:
        raise ValueError(f"Unknown SCF type {type(mf)}")

    return GDM(mf)
