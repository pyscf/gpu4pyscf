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

import unittest
from unittest import mock

import cupy as cp
import numpy as np
from pyscf.pbc import gto
from pyscf.pbc.df.df import make_auxcell

from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.pbc.df import int3c2e
from gpu4pyscf.pbc.df.grad import rhf, krhf, kuhf


def make_system(kmesh, unrestricted=False):
    cell = gto.M(atom='C 0.2 0.3 0.4; C 1.3 1.1 1.7',
                 a=np.eye(3)*5, basis='ccpvdz', precision=1e-9, verbose=0)
    auxcell = make_auxcell(cell, 'weigend')
    kpts = cell.make_kpts(kmesh)
    nkpts, nao, nocc = len(kpts), cell.nao, 5
    rng = np.random.default_rng(12)
    shape = (2, nkpts, nao, nocc) if unrestricted else (nkpts, nao, nocc)
    coeff = rng.random(shape) * .1
    if nkpts > 1:
        coeff = coeff + 1j*rng.random(shape)*.1
    occ = np.linspace(.2, 1., nocc)
    factor = cp.asarray(coeff * np.sqrt(occ))
    dm = factor @ factor.swapaxes(-1, -2).conj()
    dm = tag_array(dm, factor_l=factor, factor_r=None)
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, .3, kmesh).build()
    return opt, dm, kpts


def test_buffer_budget():
    for nao, pairs, naux, nocc, nkpts, nspin in [
        (118, 130806, 1828, 24, 18, 1),
        (236, 249000, 3656, 48, 8, 1),
        (60, 8000, 300, 12, 3, 2),
    ]:
        memory = 80*1024**3
        batch, block = krhf._get_j3c_block_sizes(
            memory, nao, pairs, naux, nocc, nkpts, nkpts, 28, nspin)
        occupied = nspin*naux*nkpts**2*nocc**2*16
        compressed = pairs*nkpts*24*batch
        dense = 3*nao**2*nkpts**2*16*block
        assert occupied + compressed + dense <= memory
        assert 1 <= block <= batch
        assert batch >= 28
    with unittest.TestCase().assertRaisesRegex(RuntimeError, 'Insufficient'):
        krhf._get_j3c_block_sizes(
            80*1024**3, 1488, 1110804, 24912, 384, 1, 1, 28, 2)


def test_real_gamma_matches_complex():
    for module, unrestricted in [(krhf, False), (kuhf, True)]:
        opt, dm, kpts = make_system([1, 1, 1], unrestricted)
        complex_dm = tag_array(dm.astype(np.complex128),
                               factor_l=dm.factor_l.astype(np.complex128),
                               factor_r=None)
        for omega in [.3, 0.]:
            for j_factor in [0, 1]:
                reference = module._get_ejk_derivatives(
                    opt, complex_dm, kpts, hermi=1, omega=omega,
                    j_factor=j_factor, exxdiv='ewald')
                result = module._get_ejk_derivatives(
                    opt, dm, kpts, hermi=1, omega=omega,
                    j_factor=j_factor, exxdiv='ewald')
                for value, expected in zip(result, reference):
                    np.testing.assert_allclose(
                        value, expected, atol=2e-9, rtol=1e-9,
                        err_msg=str((module.__name__, omega, j_factor)))


def test_auxiliary_blocks_and_tail():
    for module, unrestricted in [(krhf, False), (kuhf, True)]:
        opt, dm, kpts = make_system([2, 1, 1], unrestricted)
        reference = module._get_ejk_derivatives(
            opt, dm, kpts, hermi=1, omega=.3)
        with mock.patch.object(module, '_get_j3c_block_sizes', return_value=(31, 7)):
            result = module._get_ejk_derivatives(
                opt, dm, kpts, hermi=1, omega=.3)
        for value, expected in zip(result, reference):
            np.testing.assert_allclose(value, expected, atol=2e-9, rtol=1e-9)


def test_complex_gamma_is_not_cast_to_real():
    opt, dm, kpts = make_system([1, 1, 1])
    factor = dm.factor_l.astype(np.complex128)
    factor[:,::2] *= 1j
    complex_dm = tag_array(factor @ factor.swapaxes(-1, -2).conj(),
                           factor_l=factor, factor_r=None)
    with mock.patch.object(krhf.rhf, '_get_ejk_derivatives',
                           side_effect=AssertionError('Unexpected real dispatch')):
        result = krhf._get_ejk_derivatives(
            opt, complex_dm, kpts, hermi=1, omega=.3)
    assert all(np.isfinite(value).all() for value in result)


def test_gamma_metric_blocks_and_zero_density():
    opt, dm, kpts = make_system([1, 1, 1])
    reference = krhf._get_ejk_derivatives(
        opt, dm, kpts, hermi=1, omega=.3)
    original = rhf.get_avail_mem
    calls = [0]

    def available(exclude_memory_pool=False):
        calls[0] += 1
        # The third query sizes the metric output after SR and LR construction.
        if calls[0] == 3:
            return opt.auxcell.nao * 5 * 8 * 5
        return original(exclude_memory_pool)

    with mock.patch.object(rhf, 'get_avail_mem', side_effect=available):
        result = krhf._get_ejk_derivatives(
            opt, dm, kpts, hermi=1, omega=.3)
    for value, expected in zip(result, reference):
        np.testing.assert_allclose(value, expected, atol=2e-9, rtol=1e-9)
    zero = tag_array(cp.zeros_like(dm), factor_l=dm.factor_l[:,:,:0], factor_r=None)
    result = krhf._get_ejk_derivatives(
        opt, zero, kpts, hermi=1, omega=.3)
    for value in result:
        np.testing.assert_array_equal(value, np.zeros_like(value))


def test_fourier_blocks():
    for module, unrestricted in [(krhf, False), (kuhf, True)]:
        opt, dm, kpts = make_system([2, 1, 1], unrestricted)
        reference = module._get_ejk_derivatives(opt, dm, kpts, hermi=1)
        with mock.patch.object(module, '_get_lr_block_size', return_value=32):
            result = module._get_ejk_derivatives(opt, dm, kpts, hermi=1)
        for value, expected in zip(result, reference):
            np.testing.assert_allclose(value, expected, atol=2e-9, rtol=1e-9)
