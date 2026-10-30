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

import numpy as np
import cupy as cp
import pytest
import scipy.linalg
from pyscf.pbc import gto, scf, dft
from gpu4pyscf.pbc.dft import gen_grid
from gpu4pyscf.pbc.scf import ghf, kghf
from gpu4pyscf.pbc import scf as gpu_scf


@pytest.fixture(scope='module')
def cell():
    return gto.Cell(atom='H 0 0 0; H .7 .2 .1; H .2 .8 .3',
                    a=np.eye(3)*4, basis='gth-szv', pseudo='gth-pade',
                    spin=1, mesh=[9, 9, 9], precision=1e-8, verbose=0).build()


def make_gamma_pair(cell, backend):
    gpu = ghf.GHF(cell, exxdiv=None)
    cpu = scf.GHF(cell, exxdiv=None)
    if backend == 'gdf':
        gpu = gpu.density_fit(auxbasis='weigend')
        cpu = cpu.density_fit(auxbasis='weigend')
    gpu.with_df.mesh = cpu.with_df.mesh = cell.mesh
    return gpu, cpu, np.zeros((1, 3))


def make_kpoint_pair(cell, backend):
    # GDF uses a Gamma-centered BvK mesh; shifted FFTDF meshes exercise
    # complex Bloch charge densities.
    center = [.25, 0, 0] if backend == 'fft' else [0, 0, 0]
    kpts = cell.make_kpts([2, 1, 1], scaled_center=center, wrap_around=True)
    gpu = kghf.KGHF(cell, kpts, exxdiv=None)
    cpu = scf.KGHF(cell, kpts, exxdiv=None)
    if backend == 'gdf':
        gpu = gpu.density_fit(auxbasis='weigend')
        cpu = cpu.density_fit(auxbasis='weigend')
    gpu.with_df.mesh = cpu.with_df.mesh = cell.mesh
    return gpu, cpu, kpts


def spin_rotation(nao):
    theta, phi = .41, .67
    u = np.array([[np.cos(theta), -np.exp(-1j*phi)*np.sin(theta)],
                  [np.exp(1j*phi)*np.sin(theta), np.cos(theta)]])
    return np.kron(u, np.eye(nao))


def occupied_spin_projectors(cell, kpts):
    dms, trans_dms = [], []
    rotation = spin_rotation(cell.nao)
    for overlap in cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts):
        e, v = scipy.linalg.eigh(overlap)
        c = (v * e**-.5) @ v.conj().T
        # A three-electron determinant: two alpha and one beta electron.
        da = c[:, :2] @ c[:, :2].conj().T
        db = c[:, :1] @ c[:, :1].conj().T
        dm = scipy.linalg.block_diag(da, db)
        metric = scipy.linalg.block_diag(overlap, overlap)
        np.testing.assert_allclose(np.trace(dm @ metric), 3., atol=1e-12)
        np.testing.assert_allclose(dm @ metric @ dm, dm, atol=1e-12)
        dms.append(dm)
        # An occupied-to-virtual transition of that determinant, rotated
        # with the same physical SU(2) transformation as the density.
        occupied = np.r_[c[:, 0], np.zeros(cell.nao)]
        virtual = np.r_[c[:, 2], np.zeros(cell.nao)]
        trans_dms.append(rotation @ np.outer(virtual, occupied.conj()) @ rotation.conj().T)
    return np.asarray(dms), np.asarray(trans_dms), rotation


@pytest.fixture(scope='module', params=['fft', 'gdf'])
def gamma_spin_case(cell, request):
    gpu, cpu, kpts = make_gamma_pair(cell, request.param)
    base, trans_dm, rotation = occupied_spin_projectors(cell, kpts)
    base, trans_dm = base[0], trans_dm[0]
    rotated = rotation @ base @ rotation.conj().T
    assert np.max(abs(rotated[:cell.nao, cell.nao:].imag)) > 1e-3
    return gpu, cpu, base, rotated, trans_dm, rotation


@pytest.fixture(scope='module', params=['fft', 'gdf'])
def kpoint_spin_case(cell, request):
    gpu, cpu, kpts = make_kpoint_pair(cell, request.param)
    base, trans_dm, rotation = occupied_spin_projectors(cell, kpts)
    rotated = rotation @ base @ rotation.conj().T
    assert np.max(abs(rotated[..., :cell.nao, cell.nao:].imag)) > 1e-3
    return gpu, cpu, base, rotated, trans_dm, rotation


def cpu_jk(mf, dm):
    # Assemble spin blocks from scalar CPU DF contractions: released GHF
    # adapters interchange the alpha-beta and beta-alpha density blocks.
    nao = mf.cell.nao
    kpts = mf.kpts if isinstance(mf, scf.kghf.KGHF) else mf.kpt
    charge = dm[..., :nao, :nao] + dm[..., nao:, nao:]
    j = mf.with_df.get_jk(charge, hermi=1, kpts=kpts,
                         with_k=False, exxdiv=mf.exxdiv)[0]
    vj = np.zeros_like(dm)
    vj[..., :nao, :nao] = vj[..., nao:, nao:] = j
    vk = np.empty_like(dm)
    for i in range(2):
        for k in range(2):
            block = dm[..., i*nao:(i+1)*nao, k*nao:(k+1)*nao]
            vk[..., i*nao:(i+1)*nao, k*nao:(k+1)*nao] = mf.with_df.get_jk(
                block, hermi=0, kpts=kpts, with_j=False, exxdiv=mf.exxdiv)[1]
    return vj, vk


def _check_jk_matches_cpu_and_spin_rotation(spin_case):
    gpu, cpu, base, rotated, _, rotation = spin_case
    j0, k0 = gpu.get_jk(None, cp.asarray(base), hermi=1)
    j, k = gpu.get_jk(None, cp.asarray(rotated), hermi=1)
    refj, refk = cpu_jk(cpu, rotated)
    np.testing.assert_allclose(j.get(), refj, atol=2e-8, rtol=1e-8)
    np.testing.assert_allclose(k.get(), refk, atol=2e-8, rtol=1e-8)
    np.testing.assert_allclose(j.get(), j0.get(), atol=1e-10)
    np.testing.assert_allclose(k.get(), rotation @ k0.get() @ rotation.conj().T, atol=1e-10)
    np.testing.assert_allclose(k.get(), k.get().swapaxes(-1, -2).conj(), atol=1e-10)
    nao = gpu.cell.nao
    np.testing.assert_allclose(j[..., :nao, :nao].get(), j[..., nao:, nao:].get(), atol=1e-12)
    np.testing.assert_array_equal(j[..., :nao, nao:].get(), 0)
    veff = gpu.get_veff(None, cp.asarray(rotated))
    np.testing.assert_allclose(veff.get(), j.get()-k.get(), atol=1e-10)
    expected_coulomb = .5*np.einsum('...ij,...ji->...', rotated, j.get()).real.mean()
    assert float(veff.ecoul) == pytest.approx(expected_coulomb, abs=1e-10)


def test_gamma_jk_matches_cpu_and_spin_rotation(gamma_spin_case):
    _check_jk_matches_cpu_and_spin_rotation(gamma_spin_case)


def test_kpoint_jk_matches_cpu_and_spin_rotation(kpoint_spin_case):
    _check_jk_matches_cpu_and_spin_rotation(kpoint_spin_case)


def _check_jk_seperate_calls(spin_case):
    gpu, cpu, base, rotated, _, _ = spin_case
    batch = cp.asarray(np.stack([base, rotated]))
    j, k = gpu.get_jk(None, batch, hermi=1)
    refj, refk = cpu_jk(cpu, batch.get())
    np.testing.assert_allclose(j.get(), refj, atol=2e-8, rtol=1e-8)
    np.testing.assert_allclose(k.get(), refk, atol=2e-8, rtol=1e-8)
    np.testing.assert_allclose(gpu.get_j(None, batch, hermi=1).get(), j.get(), atol=1e-10)
    np.testing.assert_allclose(gpu.get_k(None, batch, hermi=1).get(), k.get(), atol=1e-10)
    for i, dm in enumerate(batch):
        single_j, single_k = gpu.get_jk(None, dm, hermi=1)
        np.testing.assert_allclose(single_j.get(), j[i].get(), atol=1e-10)
        np.testing.assert_allclose(single_k.get(), k[i].get(), atol=1e-10)


def test_gamma_jk_seperate_calls(gamma_spin_case):
    _check_jk_seperate_calls(gamma_spin_case)


def test_kpoint_jk_seperate_calls(kpoint_spin_case):
    _check_jk_seperate_calls(kpoint_spin_case)


def _check_nonhermitian_spin_transition_density(spin_case):
    gpu, cpu, _, _, trans_dm, _ = spin_case
    adjoint = trans_dm.swapaxes(-1, -2).conj()
    assert np.linalg.norm(trans_dm-adjoint) > .1
    # CPU FFTDF needs Hermitian charge densities. Use the exact linear
    # decomposition of the transition instead of an unsupported CPU shortcut.
    jr, kr = cpu_jk(cpu, (trans_dm+adjoint)*.5)
    ji, ki = cpu_jk(cpu, (trans_dm-adjoint)/2j)
    j, k = gpu.get_jk(None, cp.asarray(trans_dm), hermi=0)
    np.testing.assert_allclose(j.get(), jr+1j*ji, atol=2e-8, rtol=1e-8)
    np.testing.assert_allclose(k.get(), kr+1j*ki, atol=2e-8, rtol=1e-8)


def test_gamma_nonhermitian_spin_transition_density(gamma_spin_case):
    _check_nonhermitian_spin_transition_density(gamma_spin_case)


def test_kpoint_nonhermitian_spin_transition_density(kpoint_spin_case):
    _check_nonhermitian_spin_transition_density(kpoint_spin_case)


def test_kpoint_gdf_j(cell):
    gpu, cpu, kpts = make_kpoint_pair(cell, 'gdf')
    base, _, rotation = occupied_spin_projectors(cell, kpts)
    dm = rotation @ base @ rotation.conj().T
    j = gpu.get_j(None, cp.asarray(dm), hermi=1)
    k = gpu.get_k(None, cp.asarray(dm), hermi=1)
    refj, refk = cpu_jk(cpu, dm)
    np.testing.assert_allclose(j.get(), refj, atol=2e-8, rtol=1e-8)
    np.testing.assert_allclose(k.get(), refk, atol=2e-8, rtol=1e-8)


def test_gamma_scf_properties(gamma_spin_case):
    gpu, cpu, _, _, _, rotation = gamma_spin_case
    # Hcore has identical alpha/beta blocks; overlap agrees with CPU GHF.
    h, overlap = gpu.get_hcore(), gpu.get_ovlp()
    nao = gpu.cell.nao
    scalar = gpu_scf.UHF(gpu.cell)
    scalar_h = scalar.get_hcore().get()
    expected_h = np.zeros_like(h.get())
    expected_h[..., :nao, :nao] = scalar_h
    expected_h[..., nao:, nao:] = scalar_h
    np.testing.assert_allclose(h.get(), expected_h, atol=1e-11)
    np.testing.assert_allclose(overlap.get(), cpu.get_ovlp(), atol=1e-11)
    # A small Zeeman term avoids an artificial spin-degenerate Fermi boundary.
    field_h = h.copy()
    field_h[..., :nao, :nao] += .07*overlap[..., :nao, :nao]
    field_h[..., nao:, nao:] -= .07*overlap[..., nao:, nao:]
    r = cp.asarray(rotation)
    field_h = r @ field_h @ r.conj().T
    energy, coeff = gpu.eig(field_h, overlap)
    energy, coeff = cp.asarray(energy), cp.asarray(coeff)
    # Occupations are 0/1 and give the total electron count over sampled k-points.
    occ = gpu.get_occ(energy, coeff)
    assert float(occ.sum()) == gpu.cell.nelectron
    np.testing.assert_array_equal(cp.unique(occ).get(), [0., 1.])
    gpu.mo_energy, gpu.mo_coeff, gpu.mo_occ = energy, coeff, occ
    # get_occ without arguments uses the stored orbital energies.
    np.testing.assert_array_equal(gpu.get_occ().get(), occ.get())
    # The occupied-orbital density satisfies D S D = D and Tr(D S) = N per cell.
    dm = gpu.make_rdm1()
    np.testing.assert_allclose((dm @ overlap @ dm).get(), dm.get(), atol=1e-10)
    electrons = cp.trace(dm @ overlap, axis1=-2, axis2=-1).real.mean()
    assert float(electrons) == pytest.approx(gpu.cell.nelectron, abs=1e-11)
    # The Fermi energy is the highest occupied eigenvalue.
    occupied_energy = energy[occ > 0]
    assert gpu.get_fermi() == pytest.approx(float(occupied_energy.max()), abs=1e-12)
    # Eigenvectors of the supplied Fock matrix have zero orbital gradient.
    np.testing.assert_allclose(gpu.get_grad(coeff, occ, field_h).get(), 0, atol=1e-10)
    # Level shifting raises only virtual-orbital energies by the requested amount.
    shifted = gpu.get_fock(h1e=field_h, s1e=overlap, vhf=cp.zeros_like(h),
                           dm=dm, cycle=1, diis=None, damp_factor=0.,
                           level_shift_factor=.2)
    delta_mo = coeff.swapaxes(-1, -2).conj() @ (shifted-field_h) @ coeff
    expected = cp.eye(2*nao)*(1-occ)[..., None, :]*.2
    np.testing.assert_allclose(delta_mo.get(), expected.get(), atol=1e-10)
    # With no supplied Fock matrix, get_grad builds it from the stored density.
    fock = h + gpu.get_veff()
    matrices = [(coeff, occ, fock)]
    expected_gradient = cp.hstack([(c[:, o == 0].conj().T @ f @ c[:, o > 0]).ravel()
                                  for c, o, f in matrices])
    np.testing.assert_allclose(gpu.get_grad(coeff, occ).get(), expected_gradient.get(), atol=1e-10)
    # The spin expectation value agrees with CPU GHF.
    s2 = gpu.spin_square()[0]
    ref_s2 = cpu.spin_square(coeff[:, occ > 0].get(), overlap.get())[0]
    assert float(s2) == pytest.approx(ref_s2, abs=1e-10)
    # Real-space charge density agrees with CPU integration of D_aa + D_bb.
    grids = gen_grid.UniformGrids(gpu.cell).build()
    rho = gpu.get_rho(dm=dm, grids=grids)
    scalar_dm = dm[..., :nao, :nao] + dm[..., nao:, nao:]
    ni = dft.numint.NumInt()
    cpu_grids = dft.gen_grid.UniformGrids(gpu.cell).build()
    # GPU get_rho follows the spatially tiled block-loop order. Compare
    # densities at those same coordinates, not two different grid layouts.
    cpu_grids.coords = cp.asnumpy(grids.coords)[cp.asnumpy(grids.argsort())]
    points = gpu.kpt
    expected_rho = ni.get_rho(gpu.cell, scalar_dm.get(), cpu_grids, points)
    np.testing.assert_allclose(cp.asnumpy(rho), expected_rho, atol=1e-10, rtol=1e-10)


def test_kpoint_scf_properties(kpoint_spin_case):
    gpu, cpu, _, _, _, rotation = kpoint_spin_case
    # Hcore has identical alpha/beta blocks; overlap agrees with CPU GHF.
    h, overlap = gpu.get_hcore(), gpu.get_ovlp()
    nao = gpu.cell.nao
    scalar = gpu_scf.KUHF(gpu.cell, gpu.kpts)
    scalar_h = scalar.get_hcore().get()
    expected_h = np.zeros_like(h.get())
    expected_h[..., :nao, :nao] = scalar_h
    expected_h[..., nao:, nao:] = scalar_h
    np.testing.assert_allclose(h.get(), expected_h, atol=1e-11)
    np.testing.assert_allclose(overlap.get(), cpu.get_ovlp(), atol=1e-11)
    # A small Zeeman term avoids an artificial spin-degenerate Fermi boundary.
    field_h = h.copy()
    field_h[..., :nao, :nao] += .07*overlap[..., :nao, :nao]
    field_h[..., nao:, nao:] -= .07*overlap[..., nao:, nao:]
    r = cp.asarray(rotation)
    field_h = r @ field_h @ r.conj().T
    energy, coeff = gpu.eig(field_h, overlap)
    energy, coeff = cp.asarray(energy), cp.asarray(coeff)
    # Occupations are 0/1 and give the total electron count over sampled k-points.
    occ = gpu.get_occ(energy, coeff)
    assert float(occ.sum()) == gpu.cell.nelectron*len(gpu.kpts)
    np.testing.assert_array_equal(cp.unique(occ).get(), [0., 1.])
    gpu.mo_energy, gpu.mo_coeff, gpu.mo_occ = energy, coeff, occ
    # get_occ without arguments uses the stored orbital energies.
    np.testing.assert_array_equal(gpu.get_occ().get(), occ.get())
    # The occupied-orbital density satisfies D S D = D and Tr(D S) = N per cell.
    dm = gpu.make_rdm1()
    np.testing.assert_allclose((dm @ overlap @ dm).get(), dm.get(), atol=1e-10)
    electrons = cp.trace(dm @ overlap, axis1=-2, axis2=-1).real.mean()
    assert float(electrons) == pytest.approx(gpu.cell.nelectron, abs=1e-11)
    # The Fermi energy is the highest occupied eigenvalue.
    occupied_energy = energy[occ > 0]
    assert gpu.get_fermi() == pytest.approx(float(occupied_energy.max()), abs=1e-12)
    # Eigenvectors of the supplied Fock matrix have zero orbital gradient.
    np.testing.assert_allclose(gpu.get_grad(coeff, occ, field_h).get(), 0, atol=1e-10)
    # Level shifting raises only virtual-orbital energies by the requested amount.
    shifted = gpu.get_fock(h1e=field_h, s1e=overlap, vhf=cp.zeros_like(h),
                           dm=dm, cycle=1, diis=None, damp_factor=0.,
                           level_shift_factor=.2)
    delta_mo = coeff.swapaxes(-1, -2).conj() @ (shifted-field_h) @ coeff
    expected = cp.eye(2*nao)*(1-occ)[..., None, :]*.2
    np.testing.assert_allclose(delta_mo.get(), expected.get(), atol=1e-10)
    # With no supplied Fock matrix, get_grad builds it from the stored density.
    fock = h + gpu.get_veff()
    matrices = zip(coeff, occ, fock)
    expected_gradient = cp.hstack([(c[:, o == 0].conj().T @ f @ c[:, o > 0]).ravel()
                                  for c, o, f in matrices])
    np.testing.assert_allclose(gpu.get_grad(coeff, occ).get(), expected_gradient.get(), atol=1e-10)
    # Real-space charge density agrees with CPU integration of D_aa + D_bb.
    grids = gen_grid.UniformGrids(gpu.cell).build()
    rho = gpu.get_rho(dm=dm, grids=grids)
    scalar_dm = dm[..., :nao, :nao] + dm[..., nao:, nao:]
    ni = dft.numint.KNumInt()
    cpu_grids = dft.gen_grid.UniformGrids(gpu.cell).build()
    # GPU get_rho follows the spatially tiled block-loop order. Compare
    # densities at those same coordinates, not two different grid layouts.
    cpu_grids.coords = cp.asnumpy(grids.coords)[cp.asnumpy(grids.argsort())]
    points = gpu.kpts
    expected_rho = ni.get_rho(gpu.cell, scalar_dm.get(), cpu_grids, points)
    np.testing.assert_allclose(cp.asnumpy(rho), expected_rho, atol=1e-10, rtol=1e-10)


def test_gamma_hf_conversion(cell):
    gpu, cpu_hf, kpts = make_gamma_pair(cell, 'gdf')
    base, _, _ = occupied_spin_projectors(cell, kpts)
    dm = base[0]
    ks = gpu.to_ks('LDA,VWN')
    cpu = dft.GKS(cell, xc='LDA,VWN', exxdiv=None)
    cpu.with_df = cpu_hf.with_df
    ks.collinear = cpu.collinear = 'col'
    potential = ks.get_veff(dm=cp.asarray(dm))
    reference = cpu.get_veff(dm=dm)
    np.testing.assert_allclose(potential.get(), reference, atol=2e-8, rtol=1e-8)
    assert float(potential.exc) == pytest.approx(float(reference.exc), abs=2e-8)
    assert float(potential.ecoul) == pytest.approx(float(reference.ecoul), abs=2e-8)
    # Match the one-electron input: the low JK grid is not converged for
    # FFT pseudopotentials, while GPU hcore uses adaptive multigrid quadrature.
    hcore = ks.get_hcore()
    np.testing.assert_allclose(ks.energy_elec(cp.asarray(dm), hcore, potential),
                               cpu.energy_elec(dm, hcore.get(), reference), atol=2e-8)
    copied = ks.to_gks()
    assert copied is not ks and copied.xc == ks.xc
    changed = ks.to_gks('PBE')
    assert changed.xc == 'PBE' and ks.xc == 'LDA,VWN'
    assert not changed.converged
    back = ks.to_hf()
    assert type(back) is type(gpu)
    # The pure-DFT J-only cache must not become an incomplete HF exchange cache.
    actual_k = back.get_k(None, cp.asarray(dm), hermi=1)
    expected_k = cpu_hf.get_k(None, dm, hermi=1)
    np.testing.assert_allclose(actual_k.get(), expected_k, atol=2e-8, rtol=1e-8)


def test_kpoint_hf_conversion(cell):
    gpu, cpu_hf, kpts = make_kpoint_pair(cell, 'gdf')
    base, _, _ = occupied_spin_projectors(cell, kpts)
    dm = base
    ks = gpu.to_ks('LDA,VWN')
    cpu = dft.KGKS(cell, kpts, xc='LDA,VWN', exxdiv=None)
    cpu.with_df = cpu_hf.with_df
    ks.collinear = cpu.collinear = 'col'
    potential = ks.get_veff(dm=cp.asarray(dm))
    reference = cpu.get_veff(dm=dm)
    np.testing.assert_allclose(potential.get(), reference, atol=2e-8, rtol=1e-8)
    assert float(potential.exc) == pytest.approx(float(reference.exc), abs=2e-8)
    assert float(potential.ecoul) == pytest.approx(float(reference.ecoul), abs=2e-8)
    # Match the one-electron input: the low JK grid is not converged for
    # FFT pseudopotentials, while GPU hcore uses adaptive multigrid quadrature.
    hcore = ks.get_hcore()
    np.testing.assert_allclose(ks.energy_elec(cp.asarray(dm), hcore, potential),
                               cpu.energy_elec(dm, hcore.get(), reference), atol=2e-8)
    copied = ks.to_gks()
    assert copied is not ks and copied.xc == ks.xc
    changed = ks.to_gks('PBE')
    assert changed.xc == 'PBE' and ks.xc == 'LDA,VWN'
    assert not changed.converged
    back = ks.to_hf()
    assert type(back) is type(gpu)
    # The pure-DFT J-only cache must not become an incomplete HF exchange cache.
    actual_k = back.get_k(None, cp.asarray(dm), hermi=1)
    expected_k = cpu_hf.get_k(None, dm, hermi=1)
    np.testing.assert_allclose(actual_k.get(), expected_k, atol=2e-8, rtol=1e-8)
    roundtrip = ks.to_cpu()
    np.testing.assert_allclose(roundtrip.get_veff(dm=dm), reference, atol=2e-8, rtol=1e-8)


def test_checkpoint_basis_projection(cell, tmp_path):
    kpts = cell.make_kpts([2, 1, 1], wrap_around=True)
    base, _, rotation = occupied_spin_projectors(cell, kpts)
    dm = rotation @ base @ rotation.conj().T
    source = kghf.KGHF(cell, kpts, exxdiv=None)
    source.chkfile = str(tmp_path / 'k2.chk')
    source.conv_tol = 1e-9
    source.max_cycle = 50
    source.kernel(dm0=cp.asarray(dm))
    assert source.converged

    target = cell.copy()
    target.basis = 'gth-dzvp'
    target.build()
    reader = kghf.KGHF(target, source.kpts, exxdiv=None)
    dm = reader.init_guess_by_chkfile(source.chkfile, project=True)
    overlap = target.pbc_intor('int1e_ovlp', hermi=1, kpts=source.kpts)
    cross = gto.intor_cross('int1e_ovlp', target, cell, kpts=source.kpts)
    expected = []
    for s, cross_s, c, occ in zip(overlap, cross, source.mo_coeff.get(), source.mo_occ.get()):
        a = scipy.linalg.solve(s, cross_s @ c[:cell.nao], assume_a='pos')
        b = scipy.linalg.solve(s, cross_s @ c[cell.nao:], assume_a='pos')
        norm = np.einsum('pi,pi->i', a.conj(), s @ a).real
        norm += np.einsum('pi,pi->i', b.conj(), s @ b).real
        projected = np.vstack([a, b]) / np.sqrt(norm)
        expected.append((projected*occ) @ projected.conj().T)
    np.testing.assert_allclose(dm.get(), expected, atol=1e-9, rtol=1e-9)
    spin_overlap = np.stack([scipy.linalg.block_diag(s, s) for s in overlap])
    electrons = np.einsum('kij,kji->k', dm.get(), spin_overlap).real.mean()
    assert electrons == pytest.approx(cell.nelectron, abs=1e-10)


def test_kghf_reads_unrestricted_checkpoint(cell, tmp_path):
    kpts = cell.make_kpts([2, 1, 1])
    source = scf.KUHF(cell, kpts, exxdiv=None)
    # KUHF's occupation counts refer to the complete two-k-point system.
    source.nelec = (4, 2)
    source.chkfile = str(tmp_path / 'kuhf.chk')
    source.kernel()
    assert source.converged
    reader = kghf.KGHF(cell, kpts, exxdiv=None)
    dm = reader.init_guess_by_chkfile(source.chkfile, project=False)
    dm = source.make_rdm1()
    expected = np.stack([scipy.linalg.block_diag(a, b) for a, b in zip(dm[0], dm[1])])
    np.testing.assert_allclose(dm.get(), expected, atol=1e-10)


def test_gamma_minao_guesses_high_cost(cell):
    mf_minao = ghf.GHF(cell, exxdiv=None)
    dm = mf_minao.get_init_guess(key='minao')
    assert bool(cp.isfinite(dm).all())
    np.testing.assert_allclose(dm.get(), dm.get().swapaxes(-1, -2).conj(), atol=1e-11)
    mf_minao.conv_tol = 1e-9
    mf_minao.max_cycle = 50
    mf_minao.kernel(dm0=dm)
    assert mf_minao.converged
    overlap = mf_minao.get_ovlp()
    dm = mf_minao.make_rdm1()
    assert float(cp.trace(dm @ overlap, axis1=-2, axis2=-1).real.mean()) == pytest.approx(
        cell.nelectron, abs=1e-10)

    mf_atom = ghf.GHF(cell, exxdiv=None)
    dm = mf_atom.get_init_guess(key='atom')
    assert bool(cp.isfinite(dm).all())
    np.testing.assert_allclose(dm.get(), dm.get().swapaxes(-1, -2).conj(), atol=1e-11)
    mf_atom.conv_tol = 1e-9
    mf_atom.max_cycle = 50
    mf_atom.kernel(dm0=dm)
    assert mf_atom.converged
    overlap = mf_atom.get_ovlp()
    dm = mf_atom.make_rdm1()
    assert float(cp.trace(dm @ overlap, axis1=-2, axis2=-1).real.mean()) == pytest.approx(
        cell.nelectron, abs=1e-10)
    assert mf_minao.e_tot == pytest.approx(mf_atom.e_tot, abs=2e-8)


def test_kpoint_minao_guesses_high_cost(cell):
    kpts = cell.make_kpts([2, 1, 1], wrap_around=True)
    mf_minao = kghf.KGHF(cell, kpts, exxdiv=None)
    dm = mf_minao.get_init_guess(key='minao')
    assert bool(cp.isfinite(dm).all())
    np.testing.assert_allclose(dm.get(), dm.get().swapaxes(-1, -2).conj(), atol=1e-11)
    mf_minao.conv_tol = 1e-9
    mf_minao.max_cycle = 50
    mf_minao.kernel(dm0=dm)
    assert mf_minao.converged
    overlap = mf_minao.get_ovlp()
    dm = mf_minao.make_rdm1()
    assert float(cp.trace(dm @ overlap, axis1=-2, axis2=-1).real.mean()) == pytest.approx(
        cell.nelectron, abs=1e-10)

    mf_atom = kghf.KGHF(cell, kpts, exxdiv=None)
    dm = mf_atom.get_init_guess(key='atom')
    assert bool(cp.isfinite(dm).all())
    np.testing.assert_allclose(dm.get(), dm.get().swapaxes(-1, -2).conj(), atol=1e-11)
    mf_atom.conv_tol = 1e-9
    mf_atom.max_cycle = 50
    mf_atom.kernel(dm0=dm)
    assert mf_atom.converged
    overlap = mf_atom.get_ovlp()
    dm = mf_atom.make_rdm1()
    assert float(cp.trace(dm @ overlap, axis1=-2, axis2=-1).real.mean()) == pytest.approx(
        cell.nelectron, abs=1e-10)
    assert mf_minao.e_tot == pytest.approx(mf_atom.e_tot, abs=2e-8)
