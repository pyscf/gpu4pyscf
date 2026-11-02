import numpy as np

from pyscf import gto
from pyscf.df import addons

from gpu4pyscf.df.df import DF


def test_reset_preserves_auxmol_without_geometry_change():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    df = DF(mol, auxbasis='weigend')
    df.auxmol = addons.make_auxmol(mol, df.auxbasis)
    cached_auxmol = df.auxmol
    df.intopt = object()
    df.j_engine = object()

    df.reset()

    assert df.auxmol is cached_auxmol
    assert df.intopt is None
    assert df.j_engine is None


def test_reset_invalidates_auxmol_after_in_place_geometry_change():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    df = DF(mol, auxbasis='weigend')
    df.auxmol = addons.make_auxmol(mol, df.auxbasis)
    coords = mol.atom_coords()
    coords[1, 2] += 1e-3
    mol.set_geom_(coords, unit='Bohr', inplace=True)
    mol.build(dump_input=False)

    df.reset()

    assert df.auxmol is None


def test_reset_rebuilds_density_fitting_after_scanner_geometry_change():
    from gpu4pyscf.scf import hf

    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = hf.RHF(mol).density_fit(auxbasis='weigend')
    mf.kernel()
    df = mf.with_df
    cached_auxmol = df.auxmol
    cached_cderi = df._cderi

    coords = mol.atom_coords()
    coords[1, 2] += 1e-3
    mol.set_geom_(coords, unit='Bohr', inplace=True)
    mol.build(dump_input=False)
    mf.reset()

    assert df.auxmol is None
    assert df._cderi is None

    e_tot = mf.kernel()

    assert np.isfinite(e_tot)
    assert df.auxmol is not cached_auxmol
    assert df._cderi is not None and df._cderi is not cached_cderi
    assert np.allclose(df.auxmol.atom_coords(), mol.atom_coords())
