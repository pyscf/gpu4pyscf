# Copyright 2025-2026 The PySCF Developers. All Rights Reserved.
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

import math
import numpy as np
import pytest
from gpu4pyscf.md.fssh_tully import FSSH_Tully


def diabatic_hamiltonian(model, x):
    # Independent 2x2 construction; do not use the analytic adiabatic transform.
    if model == 'sac':
        v11 = 0.01 * (1 - math.exp(-1.6*x)) if x >= 0 else -0.01 * (1 - math.exp(1.6*x))
        v22 = -v11
        v12 = 0.005 * math.exp(-x*x)
    elif model == 'dac':
        v11 = 0.
        v22 = 0.05 - 0.1 * math.exp(-0.28*x*x)
        v12 = 0.015 * math.exp(-0.06*x*x)
    elif model == 'ecr':
        v11, v22 = -0.0006, 0.0006
        v12 = 0.1 * math.exp(0.9*x) if x < 0 else 0.2 - 0.1 * math.exp(-0.9*x)
    else:
        raise AssertionError('Unknown reference model')
    return np.array([[v11, v12], [v12, v22]])


def aligned_eigenvectors(model, x, reference):
    _, vectors = np.linalg.eigh(diabatic_hamiltonian(model, x))
    overlap = np.sum(reference * vectors, axis=0)
    assert np.all(abs(overlap) > 0.9)
    return vectors * np.sign(overlap)


@pytest.mark.parametrize('model', ['sac', 'dac', 'ecr'])
@pytest.mark.parametrize('x', [-1.0, -0.25, 0.25, 1.0])
def test_energies_and_finite_difference_forces(model, x):
    dynamics = FSSH_Tully(model, mass=2000.)
    energies = np.linalg.eigvalsh(diabatic_hamiltonian(model, x))
    step = 1e-4
    plus = np.linalg.eigvalsh(diabatic_hamiltonian(model, x + step))
    minus = np.linalg.eigvalsh(diabatic_hamiltonian(model, x - step))
    forces = -(plus - minus) / (2 * step)
    for state in (0, 1):
        pes = dynamics.evaluate_pes([x], cur_state=state)
        # The production transform regularizes the squared gap by 1e-16.
        np.testing.assert_allclose(pes.energy, energies, atol=2e-13, rtol=0)
        np.testing.assert_allclose(pes.force, forces[state], atol=2e-9, rtol=0)
        assert np.isfinite(pes.force)


@pytest.mark.parametrize('model', ['sac', 'dac', 'ecr'])
@pytest.mark.parametrize('x', [-1.0, -0.25, 0.25, 1.0])
def test_derivative_coupling_from_eigenvector_differences(model, x):
    _, vectors = np.linalg.eigh(diabatic_hamiltonian(model, x))
    step = 1e-4
    plus = aligned_eigenvectors(model, x + step, vectors)
    minus = aligned_eigenvectors(model, x - step, vectors)
    coupling = vectors.T @ ((plus - minus) / (2 * step))

    dynamics = FSSH_Tully(model, mass=[2000.])
    for state in (0, 1):
        pes = dynamics.evaluate_pes(np.array([[x]]), cur_state=state)
        assert pes.nacv.shape == (2, 2, 1, 1)
        actual = pes.nacv[:, :, 0, 0]
        assert np.isfinite(actual).all()
        np.testing.assert_array_equal(actual, -actual.T)
        np.testing.assert_array_equal(actual.diagonal(), np.zeros(2))
        # The overall eigenvector phases are not returned by the PES API.
        np.testing.assert_allclose(abs(actual[0, 1]), abs(coupling[0, 1]),
                                   atol=2e-9, rtol=0)


@pytest.mark.parametrize('model', ['sac', 'dac', 'ecr'])
@pytest.mark.parametrize('state', [0, 1])
def test_crossing_origin(model, state):
    derivatives = {
        'sac': np.diag([0.016, -0.016]),
        'dac': np.zeros((2, 2)),
        'ecr': np.array([[0., 0.09], [0.09, 0.]]),
    }
    energies, vectors = np.linalg.eigh(diabatic_hamiltonian(model, 0.))
    # Hellmann-Feynman derivatives use the common one-sided limits at x=0.
    derivative = vectors.T @ derivatives[model] @ vectors
    force = -derivative[state, state]
    coupling = derivative[0, 1] / (energies[1] - energies[0])
    pes = FSSH_Tully(model, mass=2000.).evaluate_pes(0., cur_state=state)
    np.testing.assert_allclose(pes.energy, energies, atol=2e-13, rtol=0)
    np.testing.assert_allclose(pes.force, force, atol=2e-12, rtol=0)
    np.testing.assert_allclose(abs(pes.nacv[0, 1, 0, 0]), abs(coupling),
                               atol=2e-11, rtol=0)


def test_ecr_origin_is_continuous_from_both_sides():
    dynamics = FSSH_Tully('ecr', mass=2000.)
    for state in (0, 1):
        center = dynamics.evaluate_pes(0., cur_state=state)
        for x in (-1e-7, 1e-7):
            nearby = dynamics.evaluate_pes(x, cur_state=state)
            # The first derivatives are continuous; nearby changes are O(|x|).
            np.testing.assert_allclose(center.energy, nearby.energy, atol=2e-8, rtol=0)
            np.testing.assert_allclose(center.force, nearby.force, atol=2e-8, rtol=0)
            np.testing.assert_allclose(center.nacv, nearby.nacv, atol=2e-8, rtol=0)


@pytest.fixture
def trajectory_factory(tmp_path):
    random_state = np.random.get_state()

    def make(name, model='sac', coupling='nac', nsteps=12):
        dynamics = FSSH_Tully(model, mass=2000.)
        dynamics.verbose = 0
        dynamics.filename = str(tmp_path / name)
        dynamics.seed = 19
        dynamics.dt = .2
        dynamics.nsteps = nsteps
        dynamics.coupling_method = coupling
        dynamics.decoherence = 'none' if coupling == 'nac' else 'edc'
        dynamics.extra_dump = ('velocity', 'force')
        if coupling == 'nac':
            dynamics.extra_dump += ('nacv',)
        return dynamics

    yield make
    np.random.set_state(random_state)


@pytest.mark.parametrize('model', ['sac', 'dac', 'ecr'])
@pytest.mark.parametrize('coupling', ['nac', 'ktdc'])
def test_real_trajectory_restart_and_frame_observables(trajectory_factory, model, coupling):
    from gpu4pyscf.md.fssh import H5Trajectory
    position, velocity = np.array([[-.4]]), np.array([[.008]])
    coefficient = np.array([1.+0j, 0j])
    whole = trajectory_factory('whole.h5', model, coupling)
    callbacks = []
    whole.callback = lambda env: callbacks.append(
        (env['step'], env['self'].cur_step, env['position'].copy()))
    whole.kernel(position.copy(), velocity.copy(), coefficient.copy())
    assert [(step, current) for step, current, _ in callbacks] == [(i, i) for i in range(1, 13)]
    expected_rng = np.random.get_state()
    split = trajectory_factory('split.h5', model, coupling, nsteps=4)
    split.kernel(position.copy(), velocity.copy(), coefficient.copy())
    restored = trajectory_factory('unused.h5', model, coupling)
    assert restored.restore(split.filename) is restored
    assert restored.cur_step == 4
    restored.nsteps = 12
    restored.kernel()
    actual_rng = np.random.get_state()
    np.testing.assert_array_equal(actual_rng[1], expected_rng[1])
    assert actual_rng[2:] == expected_rng[2:]
    assert whole.cur_step == restored.cur_step == 12
    assert whole._trajectory is restored._trajectory is None
    for name in ('position', 'velocity', 'energy', 'coefficient'):
        np.testing.assert_allclose(getattr(restored, name), getattr(whole, name), atol=2e-14, rtol=0)
    with H5Trajectory(whole.filename, 'r') as full, H5Trajectory(restored.filename, 'r') as resumed:
        assert full.frame_steps() == resumed.frame_steps() == list(range(13))
        assert resumed.read_last_frame()[0] == 12  # Numeric, not lexical ordering.
        np.testing.assert_array_equal(resumed.read_checkpoint()['velocity'], restored.velocity)
        energies = []
        for step in full.frame_steps():
            a, b = full.read_frame(step), resumed.read_frame(step)
            assert a.keys() == b.keys()
            assert set(a) == {'position', 'energy', 'coeffs', 'cur_state', *whole.extra_dump}
            for key in a:
                np.testing.assert_allclose(b[key], a[key], atol=2e-14, rtol=0)
            x = np.asarray(a['position']).item()
            if step:
                np.testing.assert_array_equal(callbacks[step-1][2], a['position'])
            expected = np.linalg.eigvalsh(diabatic_hamiltonian(model, x))
            np.testing.assert_allclose(a['energy'], expected, atol=2e-13, rtol=0)
            np.testing.assert_allclose(np.vdot(a['coeffs'], a['coeffs']), 1, atol=2e-14, rtol=0)
            state = int(a['cur_state'])
            pes = whole.evaluate_pes(a['position'], state)
            np.testing.assert_allclose(a['force'], pes.force, atol=1e-14, rtol=0)
            if 'nacv' in a:
                np.testing.assert_allclose(a['nacv'], pes.nacv, atol=1e-14, rtol=0)
            energies.append(a['energy'][state] + .5*2000*np.sum(a['velocity']**2))
        # A short, small-step velocity-Verlet trajectory conserves total energy.
        assert np.max(abs(np.asarray(energies) - energies[0])) < 1e-8


@pytest.mark.parametrize('missing', ['velocity', 'history_energy'])
def test_restart_reports_missing_saved_data(trajectory_factory, missing):
    import h5py
    run = trajectory_factory('damaged.h5', coupling='ktdc', nsteps=4)
    run.kernel(np.array([[-.4]]), np.array([[.008]]), np.array([1.+0j, 0j]))
    with h5py.File(run.filename, 'a') as handle:
        del handle['velocity' if missing == 'velocity' else '3/energy']
    restored = trajectory_factory('unused.h5', coupling='ktdc')
    if missing == 'velocity':
        with pytest.raises(ValueError, match='checkpoint velocity is missing'):
            restored.restore(run.filename)
    else:
        restored.restore(run.filename)
        restored.nsteps = 5
        with pytest.raises(ValueError, match='history frame energy is missing'):
            restored.kernel()
    assert restored._trajectory is None


def test_default_velocity_explicit_initial_conditions(trajectory_factory):
    implicit = trajectory_factory('implicit.h5', nsteps=2)
    explicit = trajectory_factory('explicit.h5', nsteps=2)
    position = np.array([[-.4]])
    implicit.kernel(position.copy())
    explicit.kernel(position.copy(), np.zeros_like(position), np.array([1.+0j, 0j]))
    for name in ('position', 'velocity', 'coefficient', 'energy'):
        np.testing.assert_allclose(getattr(implicit, name), getattr(explicit, name), atol=1e-15, rtol=0)


def test_energy_and_gradient_curvature_couplings_converge_to_analytic_limit():
    from gpu4pyscf.md.fssh import ktdc_energy, ktdc_gradient
    model = FSSH_Tully('sac', mass=2000.)
    speed = .02
    velocity = np.array([[speed]])
    energy = model.evaluate_pes(0., 0).energy

    def gradient(x):
        return np.array([-model.evaluate_pes(x, state).force for state in (0, 1)]).reshape(2, 1, 1)

    # Analytic curvature of the SAC gap at x=0, including its stated regularizer.
    # Curvature-based TDC is an approximation, not the exact derivative coupling.
    expected = .5*speed*np.sqrt(((.01*1.6)**2 - 2*.005**2) / (.005**2 + 1e-16))
    errors = []
    for dt in (.01, .002):
        past = model.evaluate_pes(-speed*dt, 0).energy
        pastpast = model.evaluate_pes(-2*speed*dt, 0).energy
        from_energy = ktdc_energy(energy, past, pastpast, dt)
        from_gradient = ktdc_gradient(energy, gradient(0.), velocity,
                                      gradient(-speed*dt), velocity, dt)
        for matrix in (from_energy, from_gradient):
            np.testing.assert_array_equal(matrix, -matrix.T)
        errors.append([abs(from_energy[1, 0]-expected), abs(from_gradient[1, 0]-expected)])
    assert np.all(np.array(errors[1]) < .3*np.array(errors[0]))
    np.testing.assert_allclose([from_energy[1, 0], from_gradient[1, 0]], expected,
                               atol=1e-10, rtol=2e-4)
    with pytest.raises(ValueError, match='dt must be positive'):
        ktdc_energy(energy, energy, energy, 0)
    with pytest.raises(ValueError, match='dt must be positive'):
        ktdc_gradient(energy, gradient(0.), velocity, gradient(0.), velocity, 0)


@pytest.mark.parametrize('velocity', [-.006, .006, -.001, .001])
def test_real_surface_hop_rescaling_conserves_energy_or_reflects(velocity):
    from gpu4pyscf.md.fssh import rescale_velocity
    run = FSSH_Tully('sac', mass=2000.)
    pes = run.evaluate_pes(0., cur_state=0)
    before = np.array([[velocity]])
    allowed, after = rescale_velocity(before, [[2000.]], pes.energy, 0, 1, pes.nacv[0, 1])
    available = .5*2000*velocity**2
    gap = pes.energy[1] - pes.energy[0]
    assert bool(allowed) == (available >= gap)
    if allowed:
        expected = np.sign(velocity) * np.sqrt(velocity**2 - 2*gap/2000)
        np.testing.assert_allclose(after, expected, atol=1e-15, rtol=0)
        np.testing.assert_allclose(.5*2000*np.sum(after**2) + pes.energy[1],
                                   available + pes.energy[0], atol=1e-15, rtol=0)
    else:
        np.testing.assert_allclose(after, -before, atol=1e-15, rtol=0)
    np.testing.assert_array_equal(before, [[velocity]])
