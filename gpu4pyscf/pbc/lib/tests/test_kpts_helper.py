# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
import cupy as cp
import pyscf
from pyscf import lib
from pyscf.pbc.tools import super_cell, k2gamma
from gpu4pyscf.pbc.lib import kpts_helper

def test_kk_adapted_iter():
    def kk_adapted(kmesh):
        kmesh = np.asarray(kmesh)
        nkpts = np.prod(kmesh)
        nx, ny, nz = kmesh
        kx = np.fft.fftfreq(nx, 1./nx).astype(int)
        ky = np.fft.fftfreq(ny, 1./ny).astype(int)
        kz = np.fft.fftfreq(nz, 1./nz).astype(int)

        kxyz = lib.cartesian_prod([kx, ky, kz])
        dk = (kxyz[None,:,:] - kxyz[:,None,:]).reshape(-1, 3)

        dk %= kmesh
        uniq_ks, uniq_index, uniq_inverse = np.unique(
            dk, axis=0, return_index=True, return_inverse=True)

        ks_conj = -uniq_ks
        strides = np.array((ny*nz, nz, 1))
        ks_idx = (uniq_ks % kmesh).dot(strides)
        ks_idx_conj = (ks_conj % kmesh).dot(strides)

        independent_idx = np.sort(np.where(ks_idx <= ks_idx_conj)[0])
        out = []
        for x in independent_idx:
            kp = ks_idx[x]
            kp_conj = ks_idx_conj[x]
            kpt_ij_idx = np.where(uniq_inverse == x)[0]
            assert len(kpt_ij_idx) == nkpts
            kpti_idx, kptj_idx = divmod(kpt_ij_idx, nkpts)
            out.append([kp, kp_conj, kpti_idx.tolist(), kptj_idx.tolist()])
        return out

    for kx in range(2, 5):
        for ky in range(1, 8):
            for kz in range(2, 4):
                kmesh = [kx,ky,kz]
                ref = kk_adapted(kmesh)
                dat = [[kp, kpc, ki.tolist(), kj.tolist()]
                       for kp, kpc, ki, kj in kpts_helper.kk_adapted_iter(kmesh)]
                assert dat == ref

def test_kk_adapted_iter_off_gamma():
    cell = pyscf.M(atom='He', basis=[[0, [1, 1]]],
                   a=[[5., 0., 0.], [1., 6., 0.], [0., 1., 7.]])
    mp_mesh = [4, 1, 1]
    kpts = cell.make_kpts(mp_mesh, with_gamma_point=False)
    qpts = cell.make_kpts(mp_mesh, with_gamma_point=True)
    scaled_k = cell.get_scaled_kpts(kpts)
    scaled_q = cell.get_scaled_kpts(qpts)
    differences = scaled_k[None, :, :] - scaled_k[:, None, :]
    groups = list(kpts_helper.kk_adapted_iter(mp_mesh, with_gamma_point=False))
    expected_pairs = [[0, 0], [1, 3], [2, 2]]
    # Conjugation is on the transfer mesh, not the shifted orbital mesh.
    np.testing.assert_array_equal([g[:2] for g in groups], expected_pairs)
    pairs = list(kpts_helper.kk_adapted_iter(
        mp_mesh, with_gamma_point=False, return_pair_index=False))
    np.testing.assert_array_equal(pairs, expected_pairs)
    for q, q_conj, ki, kj in groups:
        # Check the complete pair list against actual momentum differences.
        delta = differences - scaled_q[q]
        ref_i, ref_j = np.where(abs(delta - np.rint(delta)).max(axis=2) < 1e-12)
        np.testing.assert_array_equal(ki, ref_i)
        np.testing.assert_array_equal(kj, ref_j)

def test_fft_matrix():
    cell = pyscf.M(atom='He', basis=[[0, [1, 1]]],
                   a=np.eye(3)*3+np.random.rand(3,3)*.2)
    kx = 1
    for ky in range(2, 9):
        for kz in range(2, 4):
            kmesh = [kx,ky,kz]
            kpts = cell.make_kpts(kmesh)
            bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(cell, kmesh, True)
            ref = cp.exp(1j*cp.asarray(bvkmesh_Ls.dot(kpts.T)))
            dat = kpts_helper.fft_matrix(kmesh)
            assert abs(ref - dat).max() < 1e-14

def test_fft_matrix_off_gamma():
    cell = pyscf.M(atom='He', basis=[[0, [1, 1]]],
                   a=[[5., 0., 0.], [1., 6., 0.], [0., 1., 7.]])
    kpts = cell.make_kpts([2, 3, 1], with_gamma_point=False)
    Ls = k2gamma.translation_vectors_for_kmesh(cell, [4, 3, 1], True)
    ref = np.exp(1j * Ls.dot(kpts.T))
    dat = kpts_helper.fft_matrix([2, 3, 1], with_gamma_point=False).get()
    assert dat.shape == (12, 6)
    np.testing.assert_allclose(dat, ref, atol=1e-14, rtol=0)
    np.testing.assert_allclose(dat.conj().T.dot(dat), 12*np.eye(6),
                               atol=1e-14, rtol=0)

def test_conj_kpts():
    cell = pyscf.M(atom='He', basis=[[0, [1, 1]]],
                   a=[[5., 0., 0.], [1., 6., 0.], [0., 1., 7.]])
    mp_mesh = [2, 3, 1]
    kpts = cell.make_kpts(mp_mesh, with_gamma_point=True)
    mapping = kpts_helper.conj_kpts(mp_mesh, with_gamma_point=True)
    np.testing.assert_array_equal(mapping, [0, 2, 1, 3, 5, 4])
    pairs = kpts_helper.conj_kpts(mp_mesh, with_gamma_point=True, return_pair=True)
    np.testing.assert_array_equal(pairs, [[0, 0], [1, 2], [3, 3], [4, 5]])

    kpts = cell.make_kpts(mp_mesh, with_gamma_point=False)
    mapping = kpts_helper.conj_kpts(mp_mesh, with_gamma_point=False)
    np.testing.assert_array_equal(mapping, [5, 4, 3, 2, 1, 0])
    pairs = kpts_helper.conj_kpts(mp_mesh, with_gamma_point=False, return_pair=True)
    np.testing.assert_array_equal(pairs, [[0, 5], [1, 4], [2, 3]])
