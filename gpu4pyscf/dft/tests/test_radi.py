# Copyright 2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Atomic-size corrections for Becke and Treutler grid partitions."""

import unittest
import cupy as cp
import numpy as np
from pyscf import gto
from pyscf.dft import radi as cpu_radi
from gpu4pyscf.dft import radi


def setUpModule():
    global mol
    mol = gto.M(atom='O 0 0 0; H .7 0 .6; H -.7 0 .6',
                basis='sto-3g', verbose=0)


def tearDownModule():
    global mol
    del mol


def pair_factors(adjust, natm):
    return np.array([[adjust(i, j, np.array(0.)) for j in range(natm)]
                     for i in range(natm)])


class KnownValues(unittest.TestCase):
    def test_becke_pair_factors(self):
        table = radi.BRAGG_RADII
        before = table.copy()
        actual = radi.get_becke_fac(mol, table).get()
        reference = cpu_radi.becke_atomic_radii_adjust(mol, table)
        expected = pair_factors(reference, mol.natm)
        np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-15)
        np.testing.assert_array_equal(actual, -actual.T)
        np.testing.assert_array_equal(actual.diagonal(), 0)
        np.testing.assert_array_equal(table, before)

    def test_treutler_pair_factors(self):
        table = radi.BRAGG_RADII
        before = table.copy()
        actual = radi.get_treutler_fac(mol, table).get()
        reference = cpu_radi.treutler_atomic_radii_adjust(mol, table)
        expected = pair_factors(reference, mol.natm)
        np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-15)
        np.testing.assert_array_equal(actual, -actual.T)
        np.testing.assert_array_equal(actual.diagonal(), 0)
        np.testing.assert_array_equal(table, before)

    def test_ghost_atomic_radii(self):
        ghosts = gto.M(atom=mol.atom + '; ghost-H 0 2 0; ghost-O 0 0 2',
                       basis='sto-3g', verbose=0)
        # Ghosts use their basis element's radius, not atomic number zero.
        for factory in (radi.get_becke_fac, radi.get_treutler_fac):
            with self.subTest(adjustment=factory.__name__):
                factors = factory(ghosts, radi.BRAGG_RADII).get()
                np.testing.assert_array_equal(factors[1], factors[3])
                np.testing.assert_array_equal(factors[0], factors[4])

    def test_treutler_strided_density(self):
        values = np.array([-1., -.75, -.2, 0., .3, .8, 1.])
        g = np.broadcast_to(values, (mol.natm, mol.natm, len(values)))
        storage = cp.full(g.shape[:-1] + (2*g.shape[-1],), 123.)
        storage[..., ::2] = cp.asarray(g)
        view = storage[..., ::2]
        adjust = radi.treutler_atomic_radii_adjust(mol, radi.BRAGG_RADII)
        actual = adjust(view)
        reference = cpu_radi.treutler_atomic_radii_adjust(mol, radi.BRAGG_RADII)
        expected = np.array([[reference(i, j, values) for j in range(mol.natm)]
                             for i in range(mol.natm)])
        np.testing.assert_allclose(actual.get(), expected, atol=2e-15, rtol=2e-15)
        np.testing.assert_array_equal(actual[..., 0].get(), -1)
        np.testing.assert_array_equal(actual[..., -1].get(), 1)
        reverse = adjust(-view.swapaxes(0, 1)).swapaxes(0, 1)
        np.testing.assert_allclose(actual.get(), -reverse.get(), atol=2e-15, rtol=0)
        np.testing.assert_array_equal(view.get(), g)
        np.testing.assert_array_equal(storage[..., 1::2].get(), 123.)

    def test_becke_radii_clipping(self):
        table = radi.BRAGG_RADII.copy()
        table[1], table[8] = 1e-4, 100.
        before = table.copy()
        factor = radi.get_becke_fac(mol, table)
        self.assertEqual(factor[1, 0], .5)
        self.assertEqual(factor[0, 1], -.5)
        values = cp.linspace(-1, 1, 17)
        adjusted = values + factor[:, :, None]*(1-values**2)
        np.testing.assert_array_equal(adjusted[..., 0].get(), -1)
        np.testing.assert_array_equal(adjusted[..., -1].get(), 1)
        self.assertTrue(bool(cp.all(cp.diff(adjusted, axis=-1) >= 0)))
        self.assertTrue(bool(cp.all(abs(adjusted) <= 1)))
        np.testing.assert_array_equal(table, before)


if __name__ == '__main__':
    unittest.main()
