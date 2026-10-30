# Copyright 2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Fourier and truncated Coulomb contracts for physical boundary conditions."""
import numpy as np
import pyscf
from packaging.version import Version
import cupy as cp
import pytest
from pyscf.pbc import gto
from pyscf.pbc.tools import pbc as cpu_tools
from gpu4pyscf.pbc.tools import pbc


@pytest.mark.parametrize('omega', [0., .8, -.8])
@pytest.mark.skipif(Version(pyscf.__version__) < Version('2.9'),
                     reason='PySCF 2.8: no spherical 0D Coulomb cutoff')
def test_isolated_cubic_coulomb_kernel(omega):
    cell = gto.M(atom='He 0 0 0', basis='sto-3g', a=np.eye(3)*8,
                 unit='Bohr', dimension=0, mesh=[7]*3, verbose=0)
    vectors = cell.get_Gv()
    actual = pbc.get_coulG(cell, omega=omega).get()
    reference = cpu_tools.get_coulG(cell, omega=omega)
    np.testing.assert_allclose(actual, reference, atol=1e-12, rtol=1e-13)
    radius = 4.
    if omega == 0:
        zero_mode = 2*np.pi*radius**2
    elif omega > 0:
        zero_mode = 2*np.pi*radius**2 - np.pi/omega**2
    else:
        zero_mode = np.pi/omega**2
    assert actual[0] == pytest.approx(zero_mode, abs=1e-12)
    assert np.isfinite(actual).all()
    with cell.with_range_coulomb(omega):
        np.testing.assert_allclose(pbc.get_coulG(cell, Gv=cp.asarray(vectors)).get(),
                                   actual, atol=1e-12, rtol=0)
    full = pbc.get_coulG(cell, omega=0)
    # Both attenuated operators satisfy |omega|*Rc>=2, as required by the
    # truncated isolated-cell approximation. No 2D attenuation is assumed.
    parts = pbc.get_coulG(cell, omega=.8) + pbc.get_coulG(cell, omega=-.8)
    np.testing.assert_allclose(parts.get(), full.get(), atol=2e-14, rtol=1e-13)


@pytest.mark.parametrize('shifted', [False, True])
def test_slab_coulomb_kernel(shifted):
    cell = gto.M(atom='He 0 0 0', basis='sto-3g',
                 a=np.diag([5., 6., 12.]), unit='Bohr', dimension=2,
                 low_dim_ft_type=None, mesh=[7, 9, 11], verbose=0)
    k = np.array([.17, -.11, 0.]) if shifted else np.zeros(3)
    actual = pbc.get_coulG(cell, k=k, wrap_around=False).get()
    reference = cpu_tools.get_coulG(cell, k=k, wrap_around=False)
    np.testing.assert_allclose(actual, reference, atol=1e-12, rtol=1e-13)
    if not shifted:
        assert actual[0] == pytest.approx(-2*np.pi*6**2, abs=1e-12)
    assert np.isfinite(actual).all()
