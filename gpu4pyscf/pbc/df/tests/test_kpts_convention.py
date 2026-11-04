# Copyright 2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import cupy as cp
import pytest
from pyscf.pbc import gto
from pyscf.pbc.lib.kpts import KPoints
from gpu4pyscf.pbc.df import aft, fft, df, ft_ao, int2c2e, int3c2e
from gpu4pyscf.pbc.dft import multigrid, multigrid_v2, multigrid_v3, numint
from gpu4pyscf.pbc.dft.gen_grid import UniformGrids
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.gto.pseudo.pp_int import get_pp_nl_gpu


@pytest.fixture(scope='module')
def cell():
    return gto.M(atom='He .2 .3 .4', a=np.eye(3)*5, unit='Bohr',
                 basis=[[0, [1., 1.]], [0, [.4, 1.]]], pseudo='gth-pade',
                 mesh=[15]*3, precision=1e-8, verbose=0)


@pytest.mark.parametrize('factory', [aft.AFTDF, fft.FFTDF, df.GDF,
                                    multigrid.MultiGridNumInt,
                                    multigrid_v2.MultiGridNumInt,
                                    multigrid_v3.MultiGridNumInt])
@pytest.mark.parametrize('method', ['get_nuc', 'get_pp'])
def test_potential_gamma_shapes(cell, factory, method):
    if factory is multigrid_v2.MultiGridNumInt and multigrid_v2.libgpbc is None:
        pytest.skip('multigrid v2 kernels not compiled')
    obj = factory(cell)
    get_matrix = getattr(obj, method)
    default = get_matrix()
    single = get_matrix(np.zeros(3))
    batch = get_matrix(np.zeros((1, 3)))
    assert default.shape == single.shape == (cell.nao, cell.nao)
    assert batch.shape == (1, cell.nao, cell.nao)
    assert default.dtype == np.float64
    assert single.dtype in (np.float64, np.complex128)
    assert batch.dtype in (np.float64, np.complex128)
    cp.testing.assert_allclose(default, single, atol=1e-11, rtol=0)
    cp.testing.assert_allclose(default, batch[0], atol=1e-11, rtol=0)


def test_nonlocal_potential_gamma_shapes(cell):
    default = get_pp_nl_gpu(cell)
    single = get_pp_nl_gpu(cell, np.zeros(3))
    batch = get_pp_nl_gpu(cell, np.zeros((1, 3)))
    assert default.shape == single.shape == (cell.nao, cell.nao)
    assert batch.shape == (1, cell.nao, cell.nao)
    assert default.dtype == np.float64
    assert single.dtype in (np.float64, np.complex128)
    assert batch.dtype in (np.float64, np.complex128)
    cp.testing.assert_allclose(default, single, atol=1e-12, rtol=0)
    cp.testing.assert_allclose(default, batch[0], atol=1e-12, rtol=0)


@pytest.mark.parametrize('deriv', [0, 1, 2])
def test_ao_gamma_shapes(cell, deriv):
    coords = cell.get_uniform_grids()[:20]
    default = numint.eval_ao_kpts(cell, coords, deriv=deriv)
    single = numint.eval_ao_kpts(cell, coords, np.zeros(3), deriv=deriv)
    batch = numint.eval_ao_kpts(cell, coords, np.zeros((1, 3)), deriv=deriv)
    reference = numint.eval_ao(cell, coords, kpt=None, deriv=deriv)
    assert default.shape == single.shape == reference.shape
    assert batch.shape == (1,) + default.shape
    cp.testing.assert_allclose(default, reference, atol=1e-12, rtol=0)
    cp.testing.assert_allclose(default, single, atol=1e-12, rtol=0)
    cp.testing.assert_allclose(default, batch[0], atol=1e-12, rtol=0)


def test_fourier_gamma_and_cached_sampling(cell):
    Gv = cell.get_Gv([3]*3)
    default = ft_ao.ft_aopair_kpts(cell, Gv)
    single = ft_ao.ft_aopair_kpts(cell, Gv, kptjs=np.zeros(3))
    batch = ft_ao.ft_aopair_kpts(cell, Gv, kptjs=np.zeros((1, 3)))
    assert default.shape == single.shape == (len(Gv), cell.nao, cell.nao)
    assert batch.shape == (1,) + default.shape
    cp.testing.assert_allclose(default, single, atol=1e-12, rtol=0)
    cp.testing.assert_allclose(default, batch[0], atol=1e-12, rtol=0)
    cp.testing.assert_allclose(default, ft_ao.gen_ft_kernel(cell)(Gv), atol=1e-12, rtol=0)
    kpts = cell.make_kpts([2, 1, 1])
    kernel = ft_ao.FTOpt(cell, [2, 1, 1]).gen_ft_kernel(kpts=kpts)
    cached = kernel(Gv)
    assert cached.shape == (2, len(Gv), cell.nao, cell.nao)
    cp.testing.assert_allclose(cached, kernel(Gv, kpts=kpts), atol=1e-12, rtol=0)


@pytest.mark.parametrize('factory', [aft.AFTDF, fft.FFTDF, df.GDF])
@pytest.mark.parametrize('shape', ['single', 'batch', 'spin', 'spin_batch'])
def test_jk_preserves_density_shape(cell, factory, shape):
    dm = cp.eye(cell.nao)
    if shape == 'batch':
        dm = dm[None]
    elif shape in ('spin', 'spin_batch'):
        dm = cp.stack([dm*.6, dm*.4])
        if shape == 'spin_batch':
            dm = dm[:, None]
    obj = factory(cell)
    vj, vk = obj.get_jk(dm)
    assert vj.shape == vk.shape == dm.shape
    assert vj.dtype == vk.dtype == np.float64
    refj, refk = obj.get_jk(dm, kpts=np.zeros((1, 3)))
    cp.testing.assert_allclose(vj, refj, atol=1e-10, rtol=0)
    cp.testing.assert_allclose(vk, refk, atol=1e-10, rtol=0)


@pytest.mark.parametrize('spin', [False, True])
def test_xc_preserves_density_shape(cell, spin):
    grids = UniformGrids(cell).build()
    ni = numint.KNumInt()
    dm = cp.eye(cell.nao)
    if spin:
        dm = cp.stack([dm*.6, dm*.4])
        dm_batch = dm[:, None]
        nr = numint.nr_uks
    else:
        dm_batch = dm[None]
        nr = numint.nr_rks
    ne, exc, v = nr(ni, cell, grids, 'lda,vwn', dm)
    neb, excb, vb = nr(ni, cell, grids, 'lda,vwn', dm_batch)
    assert v.shape == dm.shape
    assert vb.shape == dm_batch.shape
    cp.testing.assert_allclose(v, vb.reshape(dm.shape), atol=1e-11, rtol=0)
    np.testing.assert_allclose(ne, neb, atol=1e-11, rtol=0)
    np.testing.assert_allclose(exc, excb, atol=1e-11, rtol=0)


def test_scf_uses_stored_sampling(cell):
    kpts = cell.make_kpts([2, 1, 1])
    mf = cell.KRHF(kpts=kpts).to_gpu()
    for method in ('get_hcore', 'get_ovlp'):
        get_matrix = getattr(mf, method)
        result = get_matrix()
        assert result.shape == (2, cell.nao, cell.nao)
        cp.testing.assert_allclose(result, get_matrix(kpts=kpts), atol=1e-11, rtol=0)
    assert int1e.int1e_ovlp(cell).shape == (cell.nao, cell.nao)


@pytest.mark.parametrize('engine,method', [
    ('rsjk', '_get_j_sr'), ('rsjk', '_get_k_sr'),
    ('rsjk', '_get_j_lr'), ('rsjk', '_get_k_lr'), ('j_engine', '_get_j_sr')])
@pytest.mark.parametrize('spin_batch', [False, True])
def test_direct_jk_gamma_density_shape(cell, engine, method, spin_batch):
    from gpu4pyscf.pbc.scf import rsjk, j_engine
    factory = rsjk.PBCJKMatrixOpt if engine == 'rsjk' else j_engine.PBCJMatrixOpt
    opt = factory(cell).build()
    dm = cp.eye(cell.nao)
    if spin_batch:
        dm = cp.stack([dm*.6, dm*.4])[:, None]
    get_matrix = getattr(opt, method)
    default = get_matrix(dm, hermi=1)
    single = get_matrix(dm, hermi=1, kpts=np.zeros(3))
    batch = get_matrix(dm, hermi=1, kpts=np.zeros((1, 3)))
    assert default.shape == single.shape == batch.shape == dm.shape
    assert default.dtype == single.dtype == batch.dtype == np.float64
    cp.testing.assert_allclose(default, single, atol=1e-10, rtol=0)
    cp.testing.assert_allclose(default, batch, atol=1e-10, rtol=0)
    if engine == 'rsjk':
        sampling = get_matrix(dm, hermi=1, kpts=KPoints(cell))
        cp.testing.assert_allclose(default, sampling, atol=1e-10, rtol=0)


@pytest.mark.parametrize('spin_batch', [False, True])
def test_rsjk_gamma_derivative_density_layout(cell, spin_batch):
    from gpu4pyscf.pbc.scf import rsjk
    opt = rsjk.PBCJKMatrixOpt(cell).build()
    dm = cp.eye(cell.nao)
    if spin_batch:
        dm = cp.stack([dm*.6, dm*.4])[:, None]
    default = opt._get_ejk_derivatives(dm)
    single = opt._get_ejk_derivatives(dm, kpts=np.zeros(3))
    batch = opt._get_ejk_derivatives(dm, kpts=np.zeros((1, 3)))
    assert default.shape == single.shape == batch.shape == (cell.natm+3, 3)
    np.testing.assert_allclose(default, single, atol=1e-10, rtol=0)
    np.testing.assert_allclose(default, batch, atol=1e-10, rtol=0)
    sampling = opt._get_ejk_derivatives(dm, kpts=KPoints(cell))
    np.testing.assert_allclose(default, sampling, atol=1e-10, rtol=0)


@pytest.mark.parametrize('factory', [aft.AFTDF, fft.FFTDF, df.GDF])
@pytest.mark.parametrize('method', ['get_nuc', 'get_pp'])
def test_df_potential_nonzero_single_and_batch(cell, factory, method):
    cell = cell.copy()
    cell.atom = 'He .2 .3 .4; He 1.7 1.1 .8'
    cell.build(False, False)
    kpts = cell.make_kpts([3, 1, 1])
    get_matrix = getattr(factory(cell), method)
    single = get_matrix(kpts[1])
    one = get_matrix(kpts[1:2])
    batch = get_matrix(kpts)
    assert single.shape == (cell.nao, cell.nao)
    assert one.shape == (1, cell.nao, cell.nao)
    assert batch.shape == (3, cell.nao, cell.nao)
    assert single.dtype == one.dtype == batch.dtype == np.complex128
    cp.testing.assert_allclose(single, one[0], atol=1e-10, rtol=0)
    cp.testing.assert_allclose(single, batch[1], atol=1e-10, rtol=0)
    cp.testing.assert_allclose(batch[0], get_matrix(), atol=1e-10, rtol=0)


@pytest.mark.parametrize('kind', ['none', 'single_gamma', 'batch_gamma', 'single', 'batch'])
def test_auxiliary_metric_kpoint_shapes(cell, kind):
    # The overlap-based default cutoff is too short for this diffuse metric.
    cell = cell.copy()
    cell.rcut = 30
    kpts = {'none': None, 'single_gamma': np.zeros(3),
            'batch_gamma': np.zeros((1, 3)),
            'single': cell.make_kpts([3, 1, 1])[1],
            'batch': cell.make_kpts([3, 1, 1])}[kind]
    result = int2c2e.int2c2e(cell, kpts=kpts, omega=-.4)
    shape = (cell.nao, cell.nao)
    if kpts is not None and kpts.ndim == 2:
        shape = (len(kpts),) + shape
    assert result.shape == shape
    # The CPU lattice sum needs a larger cutoff for the diffuse auxiliary metric.
    refcell = cell.copy()
    refcell.rcut = 50
    refcell.precision = 1e-12
    with refcell.with_range_coulomb(-.4):
        ref = np.asarray(refcell.pbc_intor('int2c2e', hermi=1, kpts=kpts))
    np.testing.assert_allclose(result.get(), ref, atol=1e-7, rtol=0)
    if kpts is None:
        assert result.dtype == np.float64


@pytest.mark.parametrize('method', ['contract_rhoG', 'contract_auxvec'])
def test_contracted_potential_kpoint_shapes(cell, method):
    kpts = cell.make_kpts([3, 1, 1])
    if method == 'contract_rhoG':
        opt = ft_ao.FTOpt(cell, [3, 1, 1]).build()
        Gv = cell.get_Gv([3]*3)
        rhoG = cp.asarray(np.exp(-.3*np.einsum('gi,gi->g', Gv, Gv)),
                          dtype=np.complex128)
        args = (rhoG, Gv)
    else:
        opt = int3c2e.SRInt3c2eOpt(cell, cell, omega=-.4, bvk_kmesh=[3, 1, 1]).build()
        args = (cp.ones(opt.auxcell.nao),)
    contract = getattr(opt, method)
    gamma = contract(*args)
    gamma_batch = contract(*args, kpts=np.zeros((1, 3)))
    single = contract(*args, kpts=kpts[1])
    one = contract(*args, kpts=kpts[1:2])
    batch = contract(*args, kpts=kpts)
    sampling = contract(*args, kpts=KPoints(cell, kpts))
    cp.testing.assert_allclose(sampling, batch, atol=1e-10, rtol=0)
    assert gamma.shape == single.shape == (cell.nao, cell.nao)
    assert gamma_batch.shape == one.shape == (1, cell.nao, cell.nao)
    assert batch.shape == (3, cell.nao, cell.nao)
    assert gamma.dtype == np.float64
    assert single.dtype == one.dtype == batch.dtype == np.complex128
    cp.testing.assert_allclose(gamma, gamma_batch[0], atol=1e-10, rtol=0)
    cp.testing.assert_allclose(gamma, batch[0], atol=1e-10, rtol=0)
    cp.testing.assert_allclose(single, one[0], atol=1e-10, rtol=0)
    cp.testing.assert_allclose(single, batch[1], atol=1e-10, rtol=0)


def test_fourier_nonzero_single_and_batch(cell):
    kpts = cell.make_kpts([3, 1, 1])
    Gv = cell.get_Gv([3]*3)
    single = ft_ao.ft_aopair_kpts(cell, Gv, kptjs=kpts[1])
    one = ft_ao.ft_aopair_kpts(cell, Gv, kptjs=kpts[1:2])
    batch = ft_ao.ft_aopair_kpts(cell, Gv, kptjs=kpts)
    assert single.shape == (len(Gv), cell.nao, cell.nao)
    assert one.shape == (1,) + single.shape
    assert batch.shape == (3,) + single.shape
    assert single.dtype == one.dtype == batch.dtype == np.complex128
    cp.testing.assert_allclose(single, one[0], atol=1e-10, rtol=0)
    cp.testing.assert_allclose(single, batch[1], atol=1e-10, rtol=0)


@pytest.mark.parametrize('kind', ['none', 'single', 'batch', 'kpoints'])
def test_fourier_cached_gamma_shape(cell, kind):
    kpts = {'none': None, 'single': np.zeros(3),
            'batch': np.zeros((1, 3)),
            'kpoints': KPoints(cell, np.zeros((1, 3)))}[kind]
    Gv = cell.get_Gv([3]*3)
    kernel = ft_ao.FTOpt(cell).gen_ft_kernel(kpts=kpts)
    reference = ft_ao.ft_aopair(cell, Gv)
    if kind in ('batch', 'kpoints'):
        reference = reference[None]
    result = kernel(Gv)
    assert result.shape == reference.shape
    cp.testing.assert_allclose(result, reference, atol=1e-12, rtol=0)
    assert kernel(Gv, kpts=np.zeros(3)).shape == (len(Gv), cell.nao, cell.nao)
    assert kernel(Gv, kpts=np.zeros((1, 3))).shape == (1, len(Gv), cell.nao, cell.nao)


def test_fourier_kpoints(cell):
    kpts = cell.make_kpts([3, 1, 1])
    sampling = KPoints(cell, kpts)
    Gv = cell.get_Gv([3]*3)
    reference = ft_ao.ft_aopair_kpts(cell, Gv, kptjs=kpts)
    result = ft_ao.ft_aopair_kpts(cell, Gv, kptjs=sampling)
    cp.testing.assert_allclose(result, reference, atol=1e-12, rtol=0)
    opt = ft_ao.FTOpt(cell, [3, 1, 1])
    kernel = opt.gen_ft_kernel(kpts=sampling)
    cp.testing.assert_allclose(kernel(Gv), reference, atol=1e-12, rtol=0)
    cp.testing.assert_allclose(kernel(Gv, kpts=sampling), reference, atol=1e-12, rtol=0)
    rhoG = cp.ones(len(Gv), dtype=np.complex128)
    cp.testing.assert_allclose(opt.contract_rhoG(rhoG, Gv, sampling),
                               opt.contract_rhoG(rhoG, Gv, kpts), atol=1e-12, rtol=0)


@pytest.mark.parametrize('factory', [aft.AFTDF, fft.FFTDF, df.GDF])
@pytest.mark.parametrize('nkpts', [1, 3])
def test_df_kpoints_object(cell, factory, nkpts):
    kpts = cell.make_kpts([nkpts, 1, 1])
    sampling = KPoints(cell, kpts)
    obj = factory(cell, kpts=kpts)
    for method in ('get_nuc', 'get_pp'):
        get_matrix = getattr(obj, method)
        result = get_matrix(sampling)
        reference = get_matrix(kpts)
        assert result.shape == (nkpts, cell.nao, cell.nao)
        assert result.dtype == reference.dtype
        cp.testing.assert_allclose(result, reference, atol=1e-10, rtol=0)
    dm = cp.stack([cp.eye(cell.nao)*.6, cp.eye(cell.nao)*.4])
    dm = cp.repeat(dm[:, None], nkpts, axis=1)
    for result, reference in zip(obj.get_jk(dm, kpts=sampling),
                                 obj.get_jk(dm, kpts=kpts)):
        assert result.shape == dm.shape
        assert result.dtype == reference.dtype
        cp.testing.assert_allclose(result, reference, atol=1e-10, rtol=0)


@pytest.mark.parametrize('method', ['get_ej_derivatives', 'get_ek_derivatives'])
def test_aft_gamma_derivative_sampling(cell, method):
    dm = cp.eye(cell.nao)
    derivative = getattr(aft.AFTDF(cell), method)
    reference = derivative(dm)
    assert reference.shape == (cell.natm+3, 3)
    for kpts in (np.zeros(3), np.zeros((1, 3)), KPoints(cell)):
        result = derivative(dm, kpts=kpts)
        assert result.dtype == reference.dtype == np.float64
        cp.testing.assert_allclose(result, reference, atol=1e-10, rtol=0)


@pytest.mark.parametrize('factory', [aft.AFTDF, fft.FFTDF])
def test_df_stored_sampling(cell, factory):
    kpts = cell.make_kpts([3, 1, 1])
    obj = factory(cell, kpts=kpts)
    if factory is aft.AFTDF:
        result = cp.concatenate([x for x, _, _ in obj.ft_loop(mesh=[3]*3)], axis=1)
        reference = cp.concatenate([x for x, _, _ in
                                    obj.ft_loop(mesh=[3]*3, kpts=kpts)], axis=1)
        assert result.shape == (3, 27, cell.nao, cell.nao)
    else:
        dm = cp.repeat(cp.eye(cell.nao)[None], 3, axis=0)
        result = obj.get_j_e1(dm)
        reference = obj.get_j_e1(dm, kpts=KPoints(cell, kpts))
    cp.testing.assert_allclose(result, reference, atol=1e-10, rtol=0)


@pytest.mark.parametrize('container', [list, tuple])
def test_fft_exchange_array_like_kpts(cell, container):
    from gpu4pyscf.pbc.df import fft_jk
    obj = fft.FFTDF(cell)
    dm = cp.eye(cell.nao)[None]
    reference = fft_jk.get_k_kpts(obj, dm, kpts=np.zeros((1, 3)))
    result = fft_jk.get_k_kpts(obj, dm, kpts=container([[0., 0., 0.]]))
    assert result.shape == dm.shape
    assert result.dtype == np.float64
    cp.testing.assert_allclose(result, reference, atol=1e-12, rtol=0)


@pytest.mark.parametrize('spin', [False, True])
@pytest.mark.parametrize('kind', ['none', 'single', 'batch', 'kpoints'])
def test_format_jks_original_sampling(cell, spin, kind):
    from gpu4pyscf.pbc.df.fft_jk import _format_jks
    kpts = {'none': None, 'single': np.zeros(3),
            'batch': np.zeros((1, 3)), 'kpoints': KPoints(cell)}[kind]
    shape = (cell.nao, cell.nao)
    if kind in ('batch', 'kpoints'):
        shape = (1,) + shape
    if spin:
        shape = (2,) + shape
    dm = cp.zeros(shape)
    bands = cell.make_kpts([2, 1, 1])
    values = cp.arange((2 if spin else 1)*2*cell.nao**2).reshape(-1, 2, cell.nao, cell.nao)
    result = _format_jks(values, dm, bands, kpts)
    expected_shape = (2, 2, cell.nao, cell.nao) if spin else (2, cell.nao, cell.nao)
    assert result.shape == expected_shape
    cp.testing.assert_array_equal(result, values.reshape(expected_shape))


@pytest.mark.parametrize('spin', [False, True])
def test_fft_exchange_original_sampling_with_bands(cell, spin):
    from gpu4pyscf.pbc.df import fft_jk
    obj = fft.FFTDF(cell)
    dm = cp.eye(cell.nao)
    if spin:
        dm = cp.stack([dm*.6, dm*.4])
    batch_dm = cp.expand_dims(dm, axis=-3)
    bands = cell.make_kpts([2, 1, 1])
    reference = fft_jk.get_k_kpts(obj, batch_dm, kpts=np.zeros((1, 3)), kpts_band=bands)
    for kpts in (None, np.zeros(3)):
        result = fft_jk.get_k_kpts(obj, dm, kpts=kpts, kpts_band=bands)
        assert result.shape == reference.shape
        cp.testing.assert_allclose(result, reference, atol=1e-11, rtol=0)
    # A public caller may accept a vector band, but the formatter receives 2D.
    result = fft_jk.get_k_kpts(obj, dm, kpts=None, kpts_band=bands[1])
    assert result.shape == dm.shape[:-2] + (1, cell.nao, cell.nao)
    cp.testing.assert_allclose(result[..., 0, :, :], reference[..., 1, :, :], atol=1e-11, rtol=0)


@pytest.mark.parametrize('factory', [multigrid.MultiGridNumInt, multigrid_v2.MultiGridNumInt])
def test_multigrid_j_original_sampling_with_bands(cell, factory):
    if factory is multigrid_v2.MultiGridNumInt and multigrid_v2.libgpbc is None:
        pytest.skip('multigrid v2 kernels not compiled')
    ni = factory(cell)
    dm = cp.eye(cell.nao)
    bands = np.zeros((1, 3))
    reference = ni.get_j(dm[None], kpts=np.zeros((1, 3)), kpts_band=bands)
    for kpts in (None, np.zeros(3)):
        result = ni.get_j(dm, kpts=kpts, kpts_band=bands)
        assert result.shape == reference.shape == (1, cell.nao, cell.nao)
        cp.testing.assert_allclose(result, reference, atol=1e-11, rtol=0)
