# Copyright 2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import cupy as cp
import pytest
from pyscf.pbc import gto
from gpu4pyscf.pbc.scf import hf, uhf, ghf, khf, kuhf, kghf, rsjk
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.df import aft, fft, df
from gpu4pyscf.pbc.dft import multigrid_v3


@pytest.fixture(scope='module')
def cell():
    return gto.M(atom='He .2 .3 .4; He 1.7 1.1 .8', a=np.eye(3)*5,
                 unit='Bohr', basis=[[0, [.8, 1.]], [1, [.6, 1.]]],
                 pseudo='gth-pade', mesh=[15]*3, precision=1e-9, verbose=0)


@pytest.mark.parametrize('kind', ['none', 'single_gamma', 'batch_gamma', 'single', 'batch'])
@pytest.mark.parametrize('method,intor,comp,hermi', [
    (int1e.int1e_ovlp, 'int1e_ovlp', 1, 1),
    (int1e.int1e_kin, 'int1e_kin', 1, 1),
    (int1e.int1e_ipovlp, 'int1e_ipovlp', 3, 2)])
def test_integral_kpoint_axes(cell, kind, method, intor, comp, hermi):
    kpts = {'none': None, 'single_gamma': np.zeros(3),
            'batch_gamma': np.zeros((1, 3)),
            'single': cell.make_kpts([3, 1, 1])[1],
            'batch': cell.make_kpts([3, 1, 1])}[kind]
    result = method(cell, kpts)
    expected_shape = (cell.nao, cell.nao) if comp == 1 else (comp, cell.nao, cell.nao)
    if kpts is not None and kpts.ndim == 2:
        expected_shape = (len(kpts),) + expected_shape
    assert result.shape == expected_shape
    if kpts is None:
        assert result.dtype == np.float64
    else:
        assert result.dtype in (np.float64, np.complex128)
    ref = np.asarray(cell.pbc_intor(intor, hermi=hermi, kpts=kpts))
    assert ref.shape == expected_shape
    np.testing.assert_allclose(result.get(), ref, atol=2e-8, rtol=0)


@pytest.mark.parametrize('factory', [hf.RHF, uhf.UHF, ghf.GHF])
@pytest.mark.parametrize('pseudo', [False, True])
def test_scf_gamma_integrals(cell, factory, pseudo):
    cell = cell.copy()
    if not pseudo:
        cell.pseudo = None
        cell.build(False, False)
    mf = factory(cell)
    ni = multigrid_v3.MultiGridNumInt(cell)
    h = (ni.get_pp() if pseudo else ni.get_nuc()) + int1e.int1e_kin(cell)
    s = int1e.int1e_ovlp(cell)
    if factory is ghf.GHF:
        h, s = ghf._block_diag(h), ghf._block_diag(s)
    for method, reference in [('get_hcore', h), ('get_ovlp', s)]:
        result = getattr(mf, method)()
        assert result.shape == reference.shape
        assert result.dtype.kind in 'fc'
        cp.testing.assert_allclose(result, reference, atol=1e-11, rtol=0)
        cp.testing.assert_allclose(result, getattr(mf, method)(kpt=np.zeros(3)),
                                   atol=1e-11, rtol=0)


@pytest.mark.parametrize('factory', [hf.RHF, uhf.UHF, ghf.GHF])
def test_scf_explicit_single_and_batch_integrals(cell, factory):
    mf = factory(cell)
    kpt = cell.make_kpts([3, 1, 1])[1]
    for method in ('get_hcore', 'get_ovlp'):
        single = getattr(mf, method)(kpt=kpt)
        batch = getattr(mf, method)(kpt=kpt[None])
        n = cell.nao * (2 if factory is ghf.GHF else 1)
        assert single.shape == (n, n)
        assert batch.shape == (1, n, n)
        assert single.dtype == batch.dtype == np.complex128
        cp.testing.assert_allclose(single, batch[0], atol=1e-11, rtol=0)


@pytest.mark.parametrize('factory', [khf.KRHF, kuhf.KUHF, kghf.KGHF])
@pytest.mark.parametrize('nkpts', [1, 3])
def test_kscf_stored_sampling_and_overrides(cell, factory, nkpts):
    kpts = cell.make_kpts([nkpts, 1, 1])
    mf = factory(cell) if nkpts == 1 else factory(cell, kpts=kpts)
    ni = multigrid_v3.MultiGridNumInt(cell)
    h = ni.get_pp(kpts) + int1e.int1e_kin(cell, kpts)
    s = int1e.int1e_ovlp(cell, kpts)
    n = cell.nao
    if factory is kghf.KGHF:
        h, s = ghf._block_diag(h), ghf._block_diag(s)
        n *= 2
    for method, reference in [('get_hcore', h), ('get_ovlp', s)]:
        result = getattr(mf, method)()
        assert result.shape == (nkpts, n, n)
        assert result.dtype.kind in 'fc'
        cp.testing.assert_allclose(result, reference, atol=1e-11, rtol=0)
        explicit = getattr(mf, method)(kpts=kpts)
        cp.testing.assert_allclose(result, explicit, atol=1e-11, rtol=0)
        gamma = getattr(mf, method)(kpts=np.zeros(3))
        gamma_batch = getattr(mf, method)(kpts=np.zeros((1, 3)))
        assert gamma.shape == (n, n)
        assert gamma_batch.shape == (1, n, n)
        cp.testing.assert_allclose(gamma, gamma_batch[0], atol=1e-11, rtol=0)
        cp.testing.assert_allclose(gamma, result[0], atol=2e-8, rtol=0)


def test_ghf_stored_nonzero_single_point(cell):
    kpt = cell.make_kpts([3, 1, 1])[1]
    mf = ghf.GHF(cell, kpt=kpt)
    for method in ('get_hcore', 'get_ovlp'):
        result = getattr(mf, method)()
        assert result.shape == (2*cell.nao, 2*cell.nao)
        assert result.dtype == np.complex128
        cp.testing.assert_allclose(result, getattr(mf, method)(kpt=kpt),
                                   atol=1e-11, rtol=0)


@pytest.mark.parametrize('factory', [hf.RHF, uhf.UHF, ghf.GHF])
def test_scf_jk_gamma_density_layout(cell, factory):
    mf = factory(cell, exxdiv=None)
    dm = cp.eye(cell.nao)
    if factory is uhf.UHF:
        dm = cp.stack([dm*.6, dm*.4])
    elif factory is ghf.GHF:
        dm = ghf._block_diag(dm).astype(np.complex128)
        dm[0, cell.nao] = .1j
        dm[cell.nao, 0] = -.1j
    vj, vk = mf.get_jk(dm=dm)
    assert vj.shape == vk.shape == dm.shape
    refj, refk = mf.get_jk(dm=dm, kpt=np.zeros(3))
    cp.testing.assert_allclose(vj, refj, atol=1e-11, rtol=0)
    cp.testing.assert_allclose(vk, refk, atol=1e-11, rtol=0)
    assert (mf.get_hcore() + mf.get_veff(dm=dm)).shape == dm.shape


@pytest.mark.parametrize('factory', [khf.KRHF, kuhf.KUHF, kghf.KGHF])
@pytest.mark.parametrize('nkpts', [1, 3])
def test_kscf_jk_stored_sampling_density_layout(cell, factory, nkpts):
    kpts = cell.make_kpts([nkpts, 1, 1])
    mf = factory(cell, kpts=kpts, exxdiv=None)
    dm = int1e.int1e_ovlp(cell, kpts) * .1
    if factory is kuhf.KUHF:
        dm = cp.stack([dm*.6, dm*.4])
    elif factory is kghf.KGHF:
        dm = ghf._block_diag(dm).astype(np.complex128)
        dm[:, 0, cell.nao] = .1j
        dm[:, cell.nao, 0] = -.1j
    vj, vk = mf.get_jk(dm_kpts=dm)
    assert vj.shape == vk.shape == dm.shape
    refj, refk = mf.get_jk(dm_kpts=dm, kpts=kpts)
    cp.testing.assert_allclose(vj, refj, atol=1e-11, rtol=0)
    cp.testing.assert_allclose(vk, refk, atol=1e-11, rtol=0)
    assert (mf.get_hcore() + mf.get_veff(dm_kpts=dm)).shape == dm.shape


@pytest.mark.parametrize('method', ['_get_j_sr', '_get_k_sr'])
@pytest.mark.parametrize('spin', [False, True])
def test_rsjk_nonzero_single_and_batch(cell, method, spin):
    kpts = cell.make_kpts([3, 1, 1])
    opt = rsjk.PBCJKMatrixOpt(cell).build(kpts=kpts)
    dm = int1e.int1e_ovlp(cell, kpts[1]) * .1
    if spin:
        dm = cp.stack([dm*.6, dm*.4])
    batch_dm = cp.expand_dims(dm, axis=-3)
    single = getattr(opt, method)(dm, hermi=1, kpts=kpts[1])
    one = getattr(opt, method)(batch_dm, hermi=1, kpts=kpts[1:2])
    assert single.shape == dm.shape
    assert one.shape == batch_dm.shape
    assert single.dtype == one.dtype == np.complex128
    cp.testing.assert_allclose(single, one.reshape(dm.shape), atol=1e-10, rtol=0)
    dm_kpts = int1e.int1e_ovlp(cell, kpts) * .1
    if spin:
        dm_kpts = cp.stack([dm_kpts*.6, dm_kpts*.4])
    result = getattr(opt, method)(dm_kpts, hermi=1, kpts=kpts)
    assert result.shape == dm_kpts.shape
    assert result.dtype == np.complex128
    cp.testing.assert_allclose(result, result.swapaxes(-1, -2).conj(), atol=1e-10, rtol=0)


@pytest.mark.parametrize('backend', [aft.AFTDF, fft.FFTDF, df.GDF])
@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('pseudo', [False, True])
def test_scf_hcore_df_backend(cell, monkeypatch, backend, batched, pseudo):
    # Exercise the DF path without requiring a large multigrid mesh.
    monkeypatch.setattr(hf, 'ALLOWED_FFT_MESH_SIZE', 0)
    cell = cell.copy()
    if not pseudo:
        cell.pseudo = None
        cell.build(False, False)
    kpts = cell.make_kpts([3, 1, 1]) if batched else np.zeros(3)
    mf = khf.KRHF(cell, kpts=kpts) if batched else hf.RHF(cell)
    mf.with_df = backend(cell, kpts=kpts.reshape(-1, 3))
    v = mf.with_df.get_pp(kpts) if pseudo else mf.with_df.get_nuc(kpts)
    expected = v + int1e.int1e_kin(cell, kpts)
    result = mf.get_hcore()
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    cp.testing.assert_allclose(result, expected, atol=1e-11, rtol=0)
