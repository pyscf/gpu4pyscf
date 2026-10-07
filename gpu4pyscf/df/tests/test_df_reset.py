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
