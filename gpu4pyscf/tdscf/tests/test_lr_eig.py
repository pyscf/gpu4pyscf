# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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

import io
import re
import numpy as np
import cupy as cp
import pytest
import scipy.linalg
from pyscf.lib.linalg_helper import _eigs_cmplx2real
from gpu4pyscf.lib import logger
from gpu4pyscf.tdscf import _lr_eig as lr


def positive_roots(w, v, nroots, env):
    xp = cp.get_array_module(w)
    idx = xp.where((w.real > 1e-8) & (abs(w.imag) < 1e-8))[0]
    idx = idx[xp.argsort(w[idx].real)]
    return w[idx].real, v[:, idx], idx


def positive_response_roots(w, v, nroots, env):
    # A projected non-Hermitian problem may have complex Ritz values before
    # convergence, even though the stable full response has real eigenvalues.
    xp = cp.get_array_module(w)
    idx = xp.flatnonzero(w.real > 1e-8)
    if env['dtype'] == np.float64:
        idx = idx[abs(w[idx].imag) < 1e-8]
        return _eigs_cmplx2real(w, v, idx, real_eigenvectors=True)
    idx = idx[xp.argsort(w[idx].real)]
    return w[idx], v[:, idx], idx


def preconditioner(diagonal):
    def apply(residual, energy, *args):
        energy = cp.asarray(energy)
        if residual.ndim == 2:
            energy = energy.reshape(-1, 1)
        denominator = diagonal - energy
        denominator = cp.where(abs(denominator) < 1e-8, 1e-8, denominator)
        return residual / denominator
    return apply


def symmetric_matrix(n=12):
    matrix = np.diag(np.arange(2., n + 2))
    coupling = np.random.default_rng(81).normal(size=(n, n)) * .01
    return matrix + coupling + coupling.T


def response_matrix(n=6, complex_values=False):
    a = symmetric_matrix(n)
    b = np.diag(np.linspace(.05, .12, n))
    b += np.diag(np.full(n-1, .01), 1) + np.diag(np.full(n-1, .01), -1)
    if complex_values:
        a = a.astype(complex)
        a[0, 1] += .02j
        a[1, 0] -= .02j
        b = b.astype(complex)
        b[0, 2] += .01j
        b[2, 0] += .01j
    return a, b, np.block([[a, b], [-b.conj(), -a.conj()]])


def test_eigh_single_initial_guess():
    matrix = symmetric_matrix(6)
    a = cp.asarray(matrix)
    guess = cp.ones(6)
    converged, values, vectors = lr.eigh(lambda x: x @ a.T, guess,
                                        preconditioner(cp.diag(a)), nroots=1,
                                        pick=positive_roots, tol_residual=1e-10, lindep=1e-24,
                                        max_cycle=40, verbose=0)
    assert bool(cp.asarray(converged).all())
    np.testing.assert_allclose(cp.asnumpy(values), np.linalg.eigvalsh(matrix)[:1], atol=1e-10)
    np.testing.assert_allclose((vectors @ a.T - cp.asarray(values)[:, None] * vectors).get(), 0, atol=1e-9)


def test_eigh_with_symmetry():
    block = np.diag([2., 5., 8., 11.]) + .03 * (np.ones((4, 4)) - np.eye(4))
    matrix = scipy.linalg.block_diag(block, block + np.eye(4))
    a = cp.asarray(matrix)
    guess = cp.eye(8)[[0, 4]]
    converged, values, vectors = lr.eigh(lambda x: x @ a.T, guess,
                                        preconditioner(cp.diag(a)), nroots=2,
                                        x0sym=[0, 1], pick=positive_roots,
                                        tol_residual=1e-10, lindep=1e-24, max_cycle=30, verbose=0)
    assert bool(cp.asarray(converged).all())
    np.testing.assert_allclose(cp.asnumpy(values), np.linalg.eigvalsh(matrix)[:2], atol=1e-10)
    np.testing.assert_allclose((vectors @ a.T - cp.asarray(values)[:, None] * vectors).get(), 0, atol=1e-9)
    np.testing.assert_allclose(vectors[0, 4:].get(), 0, atol=1e-12)
    np.testing.assert_allclose(vectors[1, :4].get(), 0, atol=1e-12)


@pytest.mark.parametrize('complex_values', [False, True])
def test_eig(complex_values):
    a, b, matrix = response_matrix(complex_values=complex_values)
    operator = cp.asarray(matrix)
    n = a.shape[0]
    guess = cp.eye(2*n, dtype=operator.dtype)[:2]
    converged, values, vectors = lr.eig(lambda x: x @ operator.T, guess,
                                       preconditioner(cp.diag(operator)), nroots=2,
                                       pick=positive_response_roots,
                                       tol_residual=1e-10, lindep=1e-24,
                                       max_cycle=30, verbose=0)
    expected = scipy.linalg.eigvals(matrix)
    assert np.max(abs(expected.imag)) < 1e-12
    expected = np.sort(expected.real[expected.real > 0])[:2]
    assert bool(cp.asarray(converged).all())
    assert isinstance(vectors, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(values), expected, atol=1e-9)
    np.testing.assert_allclose((vectors @ operator.T - cp.asarray(values)[:, None] * vectors).get(),
                               0, atol=1e-9)


@pytest.mark.parametrize('complex_values', [False, True])
def test_symmetric_orth(complex_values):
    x = np.array([[1., .1, 0., .03, 0., .01], [.1, 1., .2, 0., .02, 0.]])
    if complex_values:
        x = x.astype(complex)
        x[0, 1] += .05j
        x[1, 5] += .03j
    vectors = lr._symmetric_orth(cp.asarray(x))
    assert isinstance(vectors, cp.ndarray)
    dual = cp.hstack([vectors[:, 3:], vectors[:, :3]]).conj()
    augmented = cp.vstack([vectors, dual])
    np.testing.assert_allclose((augmented.conj() @ augmented.T).get(), np.eye(4), atol=1e-11)


@pytest.mark.parametrize('gram_schmidt', [False, True])
def test_davidson(gram_schmidt):
    matrix = symmetric_matrix(6)
    a = cp.asarray(matrix, dtype=cp.float64)
    converged, values, vectors = lr.Davidson(lambda x: x @ a.T, cp.diag(a), N_states=2,
                                            conv_tol=1e-10, max_iter=25,
                                            gram_schmidt=gram_schmidt, single=False, verbose=0)
    assert bool(cp.asarray(converged).all())
    np.testing.assert_allclose(values.get(), np.linalg.eigvalsh(matrix)[:2], atol=1e-9, rtol=0)
    np.testing.assert_allclose((vectors @ a.T - cp.asarray(values)[:, None] * vectors).get(), 0, atol=1e-9)


@pytest.mark.parametrize('gram_schmidt', [False, True])
def test_davidson_casida(gram_schmidt):
    a, b, matrix = response_matrix(n=12)
    ag, bg = cp.asarray(a, dtype=cp.float64), cp.asarray(b, dtype=cp.float64)

    def product(X, Y):
        return X @ ag + Y @ bg, X @ bg + Y @ ag

    converged, values, x, y = lr.Davidson_Casida(product, cp.diag(ag), N_states=2,
                                               conv_tol=1e-10,
                                               max_iter=25, gram_schmidt=gram_schmidt,
                                               single=False, verbose=0)
    expected = scipy.linalg.eigvals(matrix).real
    expected = np.sort(expected[expected > 0])[:2]
    assert bool(cp.asarray(converged).all())
    np.testing.assert_allclose(values.get(), expected, atol=1e-9, rtol=0)
    vectors = cp.hstack([x, y])
    np.testing.assert_allclose((vectors @ cp.asarray(matrix).T - values[:, None] * vectors).get(),
                               0, atol=1e-9)
    np.testing.assert_allclose((x @ x.T - y @ y.T).get(), np.eye(2), atol=1e-9)


def test_davidson_casida_iteration_limit():
    a, b, _ = response_matrix()
    ag, bg = cp.asarray(a), cp.asarray(b)
    output = io.StringIO()
    log = logger.Logger(output, logger.WARN)
    converged, values, x, y = lr.Davidson_Casida(
        lambda X, Y: (X @ ag + Y @ bg, X @ bg + Y @ ag), cp.diag(ag),
        N_states=1, max_iter=1, conv_tol=1e-12, verbose=log)
    assert not bool(cp.asarray(converged).all())
    assert values.shape == (1,)
    assert bool(cp.isfinite(x).all() & cp.isfinite(y).all())
    assert 'not converged' in output.getvalue()


def test_real_eig_with_symmetry():
    a = scipy.linalg.block_diag(symmetric_matrix(3), symmetric_matrix(3) + .5*np.eye(3))
    b = np.eye(6) * .05
    matrix = cp.asarray(np.block([[a, b], [-b, -a]]))
    guess = cp.eye(12)[[0, 3]]
    converged, values, vectors = lr.real_eig(lambda v: v @ matrix.T, guess,
                                           preconditioner(cp.diag(matrix)), nroots=2,
                                           x0sym=[0, 1], tol_residual=1e-10, lindep=1e-24,
                                           max_cycle=30, verbose=0)
    assert bool(cp.asarray(converged).all())
    expected = np.sort(scipy.linalg.eigvals(matrix.get()).real)
    expected = expected[expected > 0][:2]
    np.testing.assert_allclose(cp.asnumpy(values), expected, atol=1e-9)
    np.testing.assert_allclose((vectors @ matrix.T - cp.asarray(values)[:, None] * vectors).get(),
                               0, atol=1e-9)


@pytest.mark.parametrize('complex_values', [False, True])
def test_eig_with_symmetry(complex_values):
    a0, b0, _ = response_matrix(n=3, complex_values=complex_values)
    a = scipy.linalg.block_diag(a0, a0 + .4*np.eye(3))
    b = scipy.linalg.block_diag(b0, b0*.8)
    matrix = np.block([[a, b], [-b.conj(), -a.conj()]])
    operator = cp.asarray(matrix)
    guesses = cp.eye(12, dtype=operator.dtype)[[0, 3]]
    output = io.StringIO()
    converged, values, vectors = lr.eig(
        lambda x: x @ operator.T, guesses, preconditioner(cp.diag(operator)),
        nroots=2, x0sym=np.array([0, 1]), pick=positive_response_roots,
        tol_residual=1e-10, lindep=1e-24, max_cycle=30,
        verbose=logger.Logger(output, logger.DEBUG))
    expected = scipy.linalg.eigvals(matrix)
    expected = np.sort(expected.real[expected.real > 0])[:2]
    assert bool(cp.asarray(converged).all())
    np.testing.assert_allclose(values, expected, atol=1e-9, rtol=0)
    np.testing.assert_allclose((vectors @ operator.T-cp.asarray(values)[:, None]*vectors).get(),
                               0, atol=1e-9)
    np.testing.assert_allclose(vectors[0, [3, 4, 5, 9, 10, 11]].get(), 0, atol=1e-12)
    np.testing.assert_allclose(vectors[1, [0, 1, 2, 6, 7, 8]].get(), 0, atol=1e-12)
    # Normalize each physical response root in the indefinite TD metric.
    metric_norm = cp.sum(abs(vectors[:, :6])**2-abs(vectors[:, 6:])**2, axis=1)
    normalized = vectors/cp.sqrt(metric_norm[:, None])
    np.testing.assert_allclose((normalized[:, :6].conj() @ normalized[:, :6].T-
                               normalized[:, 6:].conj() @ normalized[:, 6:].T).get(),
                               np.eye(2), atol=1e-10)


@pytest.mark.parametrize('complex_values', [False, True])
def test_davidson_nosym1_initial_guess_and_callback(complex_values):
    rng = np.random.default_rng(632)
    transform = np.eye(6)+.03*rng.normal(size=(6, 6))
    if complex_values:
        transform = transform+.02j*rng.normal(size=(6, 6))
    matrix = transform @ np.diag(np.arange(2., 8.)) @ np.linalg.inv(transform)
    operator = cp.asarray(matrix)
    guess = cp.ones(6, dtype=matrix.dtype)
    iterations = []
    converged, values, vectors = lr.davidson_nosym1(
        lambda x: x @ operator.T, guess, preconditioner(cp.diag(operator)),
        nroots=1, pick=positive_response_roots if complex_values else positive_roots,
        tol=1e-18, lindep=1e-24,
        max_cycle=30, callback=lambda env: iterations.append(env['icyc']), verbose=0)
    assert bool(cp.asarray(converged).all())
    assert iterations and len(iterations) < 30
    np.testing.assert_allclose(values, [2.], atol=1e-9, rtol=0)
    np.testing.assert_allclose((vectors @ operator.T-cp.asarray(values)[:, None]*vectors).get(),
                               0, atol=1e-9)
    reference = transform[:, 0]/np.linalg.norm(transform[:, 0])
    vector = vectors[0].get()/np.linalg.norm(vectors[0].get())
    np.testing.assert_allclose(np.outer(vector, vector.conj()),
                               np.outer(reference, reference.conj()), atol=1e-9)


def test_symmetric_orth_real():
    initial = cp.asarray([[1., .1, 0., .03, 0., .01], [.1, 1., .2, 0., .02, 0.]])
    orth = lr._symmetric_orth(initial)
    augmented = cp.vstack([orth, cp.hstack([orth[:, 3:], orth[:, :3]])])
    assert augmented.shape == (4, 6)
    np.testing.assert_allclose((augmented @ augmented.T).get(), np.eye(4), atol=1e-10)
    original = cp.vstack([initial, cp.hstack([initial[:, 3:], initial[:, :3]])]).get()
    q, _ = np.linalg.qr(original.T)
    np.testing.assert_allclose((augmented.T @ augmented).get(), q @ q.T, atol=1e-10)


def test_symmetric_orth_complex_linear_dependency():
    initial = cp.asarray([[1., .1j, 0., .03, 0., .01]])
    orth = lr._symmetric_orth(cp.vstack([initial, initial]))
    assert orth.shape == (1, 6)
    augmented = cp.vstack([orth, cp.hstack([orth[:, 3:], orth[:, :3]]).conj()])
    np.testing.assert_allclose((augmented.conj() @ augmented.T).get(), np.eye(2), atol=1e-11)


@pytest.mark.parametrize('solver', [lr.eigh, lr.eig, lr.real_eig, lr.davidson_nosym1])
def test_solver_restart_with_limited_memory(solver, monkeypatch):
    # MAX_SPACE_INC is a documented tuning option, not a numerical mock.
    monkeypatch.setattr(lr, 'MAX_SPACE_INC', 2)
    a, _, response = response_matrix(n=6)
    matrix = response if solver in (lr.eig, lr.real_eig) else a
    operator = cp.asarray(matrix)
    output = io.StringIO()
    options = dict(nroots=1, max_memory=0, max_cycle=50, lindep=1e-24,
                   verbose=logger.Logger(output, logger.DEBUG if solver is lr.eig else logger.WARN))
    if solver is lr.real_eig:
        options['tol_residual'] = 1e-10
    elif solver is lr.davidson_nosym1:
        options.update(pick=positive_roots, tol=1e-18)
    else:
        options.update(pick=positive_roots, tol_residual=1e-10)
    converged, values, vectors = solver(
        lambda x: x @ operator.T, cp.eye(len(matrix))[:1],
        preconditioner(cp.diag(operator)), **options)
    assert bool(cp.asarray(converged).all())
    expected = scipy.linalg.eigvals(matrix).real
    expected = np.sort(expected[expected > 0])[:1]
    np.testing.assert_allclose(cp.asnumpy(values), expected, atol=1e-9, rtol=0)
    np.testing.assert_allclose((vectors @ operator.T-cp.asarray(values)[:, None]*vectors).get(),
                               0, atol=1e-9)
    assert 'Not enough memory' in output.getvalue()
    if solver is lr.eig:
        sizes = [int(n) for n in re.findall(r'^lr_eig \d+ (\d+) ', output.getvalue(), re.MULTILINE)]
        assert any(new < old for old, new in zip(sizes, sizes[1:])), sizes


@pytest.mark.parametrize('solver', [lr.eigh, lr.eig, lr.real_eig])
def test_converged_initial_guess_with_insufficient_roots(solver):
    diagonal = cp.asarray([2., 3., 4.])
    if solver is not lr.eigh:
        diagonal = cp.hstack([diagonal, -diagonal])
    output = io.StringIO()
    kwargs = dict(nroots=2, verbose=logger.Logger(output, logger.WARN))
    if solver is not lr.real_eig:
        kwargs['pick'] = positive_roots
    converged, values, vectors = solver(
        lambda x: x*diagonal, cp.eye(len(diagonal))[:1],
        preconditioner(diagonal), **kwargs)
    assert len(values) == len(vectors) == 1
    assert bool(cp.asarray(converged).all())
    np.testing.assert_allclose(cp.asnumpy(values), [2.], atol=1e-12)
    np.testing.assert_allclose((vectors*diagonal-cp.asarray(values)[:, None]*vectors).get(), 0, atol=1e-12)
    assert 'Not enough eigenvectors' in output.getvalue()


def test_converged_vector_as_initial_guess():
    # Duplicate guesses exercise the QR rank filter (_lr_eig.py:1630-1634)
    # and the len(xt) != x0len warning branch (_lr_eig.py:1681-1687).
    diagonal = cp.asarray([2., 3., 4.])
    output = io.StringIO()
    converged, values, vectors = lr.davidson_nosym1(
        lambda x: x*diagonal, cp.asarray([[1., 0., 0.], [1., 0., 0.]]),
        preconditioner(diagonal), nroots=1, pick=positive_roots,
        verbose=logger.Logger(output, logger.WARN))
    assert bool(converged.all())
    np.testing.assert_allclose(values, [2.], atol=1e-12)
    np.testing.assert_allclose((vectors*diagonal-cp.asarray(values)[:, None]*vectors).get(), 0, atol=1e-12)
    assert 'QR decomposition removed 1 vectors' in output.getvalue()


def test_response_restart_linear_dependency(monkeypatch):
    # Retain an extra Ritz vector and its symmetry label (_lr_eig.py:463-465).
    monkeypatch.setattr(lr, 'MAX_SPACE_INC', 3)
    a0, b0, _ = response_matrix(n=6, complex_values=True)
    a = scipy.linalg.block_diag(a0, a0+.4*np.eye(6))
    b = scipy.linalg.block_diag(b0, b0*.8)
    matrix = cp.asarray(np.block([[a, b], [-b.conj(), -a.conj()]]))
    calls = []

    def product(x):
        calls.append(len(x))
        dual = cp.hstack([x[:, 12:], x[:, :12]]).conj()
        paired = cp.vstack([x, dual])
        # Check symmetry-sector reorthogonalization on restart (_lr_eig.py:373-384).
        np.testing.assert_allclose((paired.conj() @ paired.T).get(),
                                   np.eye(len(paired)), atol=1e-10)
        return x @ matrix.T

    diagonal_preconditioner = preconditioner(cp.diag(matrix))

    # max_memory=0 triggers the restart condition (_lr_eig.py:531).
    # Exercise the correction filter (_lr_eig.py:498-505): small independent
    # corrections must survive while the residual exceeds tol_residual.
    converged, values, vectors = lr.eig(
        product, cp.eye(24, dtype=complex)[[0, 6]], diagonal_preconditioner,
        x0sym=[0, 1], nroots=2, max_memory=0, max_cycle=50,
        tol_residual=1e-10, lindep=1e-24, pick=positive_response_roots, verbose=0)
    assert bool(cp.asarray(converged).all())
    assert len(calls) <= 50
    expected = scipy.linalg.eigvals(matrix.get())
    expected = np.sort(expected.real[expected.real > 0])[:2]
    np.testing.assert_allclose(values, expected, atol=1e-9, rtol=0)
    residual = cp.linalg.norm(vectors @ matrix.T-cp.asarray(values)[:, None]*vectors, axis=1)
    assert bool((residual <= 1e-10).all()), residual
    np.testing.assert_allclose(vectors[0, list(range(6, 12))+list(range(18, 24))].get(), 0, atol=1e-12)
    np.testing.assert_allclose(vectors[1, list(range(6))+list(range(12, 18))].get(), 0, atol=1e-12)
