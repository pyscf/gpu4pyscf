import pytest
import numpy as np
import cupy as cp
import pyscf
from pyscf.df import incore
from gpu4pyscf.df import int3c2e_bdiv
from gpu4pyscf.lib.cupy_helper import contract

def test_int3c2e():
    mol = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': [[3, [1.5, 1.], [.9, 1.]],
                      [4, [2., 1.]]],
               'C2': 'ccpvdz'})
    auxmol = mol.copy()
    auxmol.basis = {
        'C1': '''
C    S
      2.9917624900           1.0000000000
C    P
     28.1325940100           1.0000000000
C    P
      9.8364318200           1.0000000000
C    P
      3.3490545000           1.0000000000
C    P
      1.4947618600           1.0000000000
C    P
      0.5769010900           1.0000000000
C    D
      0.1995412500           1.0000000000 ''',
        'C2': [[0, [.5, 1.]], [1, [.8, 1.]], [3, [.9, 1]]],
    }
    auxmol.build()
    dat = int3c2e_bdiv.aux_e2(mol, auxmol)
    ref = incore.aux_e2(mol, auxmol)
    assert abs(dat.get()-ref).max() < 1e-10

def test_int3c2e_bdiv():
    mol = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': [[3, [1.5, 1.], [.9, 1.]],
                      [4, [2., 1.]]],
               'C2': 'ccpvdz'})

    auxmol = mol.copy()
    auxmol.basis = {
        'C1':'''
C    S
      2.9917624900           1.0000000000
C    P
     28.1325940100           1.0000000000
C    P
      9.8364318200           1.0000000000
C    P
      3.3490545000           1.0000000000
C    P
      1.4947618600           1.0000000000
C    P
      0.5769010900           1.0000000000
C    D
      0.1995412500           1.0000000000 ''',
        'C2':[[0, [.5, 1.]], [1, [.8, 1.]], [3, [.9, 1]]],
    }
    auxmol.build()
    int3c2e_opt = int3c2e_bdiv.Int3c2eOpt(mol, auxmol).build()
    nao, nao_orig = int3c2e_opt.coeff.shape
    naux = int3c2e_opt.aux_coeff.shape[0]
    out = cp.zeros((nao*nao, naux))
    eri3c = next(int3c2e_opt.int3c2e_bdiv_generator())
    ao_pair_mapping = int3c2e_opt.create_ao_pair_mapping()
    out[ao_pair_mapping] = eri3c
    i, j = divmod(ao_pair_mapping, nao)
    out[j*nao+i] = eri3c
    out = out.reshape(nao, nao, naux)
    aux_coeff = cp.asarray(int3c2e_opt.aux_coeff)
    coeff = cp.asarray(int3c2e_opt.coeff)
    out = contract('pqr,rk->pqk', out, aux_coeff)
    out = contract('pqk,qj->pjk', out, coeff)
    out = contract('pjk,pi->ijk', out, coeff)
    ref = incore.aux_e2(mol, auxmol)
    assert abs(out.get()-ref).max() < 1e-10

    eri3c = int3c2e_opt.orbital_pair_cart2sph(eri3c)
    ao_pair_mapping = int3c2e_opt.create_ao_pair_mapping(cart=mol.cart)
    out = cp.zeros((nao_orig*nao_orig, naux))
    out[ao_pair_mapping] = eri3c
    i, j = divmod(ao_pair_mapping, nao_orig)
    out[j*nao_orig+i] = eri3c
    out = out.reshape(nao_orig, nao_orig, naux)
    out = contract('pqr,rk->pqk', out, aux_coeff)
    out = int3c2e_opt.unsort_orbitals(out, axis=(0,1))
    assert abs(out.get()-ref).max() < 1e-10

    eri3c, rows, cols = int3c2e_bdiv.compressed_aux_e2(mol, auxmol)
    out = cp.zeros((nao_orig, nao_orig, auxmol.nao))
    out[rows,cols] = eri3c
    out[cols,rows] = eri3c
    assert abs(out.get()-ref).max() < 1e-10

def test_int3c2e_sparse():
    mol = pyscf.M(
        atom='''
O       0.873    5.017    1.816
H       1.128    5.038    2.848
H       0.173    4.317    1.960
O       3.665    1.316    1.319
H       3.904    2.233    1.002
H       4.224    0.640    0.837
''',
        basis='def2-tzvp'
    )
    auxmol = mol.copy()
    auxmol.basis = 'ccpvdz-jkfit'
    auxmol.build()
    int3c2e_opt = int3c2e_bdiv.Int3c2eOpt(mol, auxmol).build()
    dat = int3c2e_bdiv.aux_e2(mol, auxmol)
    ref = incore.aux_e2(mol, auxmol)
    assert abs(dat.get()-ref).max() < 1e-10

    eri3c = next(int3c2e_opt.int3c2e_bdiv_generator())
    eri3c = int3c2e_opt.orbital_pair_cart2sph(eri3c)
    ao_pair_mapping = int3c2e_opt.create_ao_pair_mapping(cart=mol.cart)
    nao, nao_orig = int3c2e_opt.coeff.shape
    naux = int3c2e_opt.aux_coeff.shape[0]
    out = cp.zeros((nao_orig*nao_orig, naux))
    out[ao_pair_mapping] = eri3c
    i, j = divmod(ao_pair_mapping, nao_orig)
    out[j*nao_orig+i] = eri3c
    out = out.reshape(nao_orig, nao_orig, naux)
    aux_coeff = cp.asarray(int3c2e_opt.aux_coeff)
    out = contract('pqr,rk->pqk', out, aux_coeff)
    out = int3c2e_opt.unsort_orbitals(out, axis=(0,1))
    assert abs(out.get()-ref).max() < 1e-10

    eri3c, rows, cols = int3c2e_bdiv.compressed_aux_e2(mol, auxmol)
    out = cp.zeros((nao_orig, nao_orig, auxmol.nao))
    out[rows,cols] = eri3c
    out[cols,rows] = eri3c
    assert abs(out.get()-ref).max() < 1e-10

def test_group_blocks():
    assert int3c2e_bdiv.group_blocks([0, 1, 3, 6], 3) == [0, 2, 3]
    assert int3c2e_bdiv.group_blocks([0, 1, 3, 4], 3) == [0, 2, 3]
    with pytest.raises(RuntimeError):
        int3c2e_bdiv.group_blocks([0, 4, 9, 14], 3)

def test_contract_int3c2e():
    from gpu4pyscf.df.j_engine_3c2e import contract_int3c2e_dm
    from gpu4pyscf.df.int3c2e_bdiv import contract_int3c2e_auxvec
    mol = pyscf.M(
        atom = '''
        O   0.000   -0.    0.1174
        H  -0.757    8.   -0.4696
        H   0.757    4.   -0.4696
        C   1.      1.    0.
        ''',
        basis=('ccpvtz', [[1, [3.7, 1, .1]], [1, [2., .5, .3]], [1, [.8, .5, .8]]])
    )
    auxmol = mol.copy()
    auxmol.basis = ('weigend', [[3, [2, 1, .5], [1, .2, 1]]])
    auxmol.build(0, 0)
    np.random.seed(10)
    nao = mol.nao
    dm = np.random.rand(nao,nao)
    dm = dm.dot(dm.T)
    eri3c = incore.aux_e2(mol, auxmol)

    dat = contract_int3c2e_dm(mol, auxmol, dm)
    ref = np.einsum('ijP,ji->P', eri3c, dm)
    assert abs(dat.get() - ref).max() < 1e-9

    auxvec = np.random.rand(auxmol.nao)
    dat = contract_int3c2e_auxvec(mol, auxmol, auxvec)
    ref = np.einsum('ijP,P->ij', eri3c, auxvec)
    assert abs(dat.get() - ref).max() < 1e-9

# issue 540
def test_int3c2e_sparse1():
    mol = pyscf.M(
        atom='C 1. 1. 0.; O 8. 0. 0.',
        basis={
            'C': [[0, [1e4, -.2], [1e3, .8]],
                  [0, [10., 1]]],
            'O': [[0, [1e4, -.2], [3e3, .2], [1e3, .8]],
                  [0, [10., 1]]],},
    )
    dat = int3c2e_bdiv.aux_e2(mol, mol)
    ref = incore.aux_e2(mol, mol)
    assert abs(dat.get() - ref).max() < 1e-9

    int3c2e_opt = int3c2e_bdiv.Int3c2eOpt(mol, mol).build()
    ao_pair_mapping = int3c2e_opt.create_ao_pair_mapping()
    nao = mol.nao
    i, j = divmod(ao_pair_mapping, nao)
    coeff = cp.asarray(int3c2e_opt.coeff)
    aux_coeff = cp.asarray(int3c2e_opt.coeff)
    for eri3c_batch in int3c2e_opt.int3c2e_bdiv_generator():
        eri3c_batch = int3c2e_opt.orbital_pair_cart2sph(eri3c_batch, inplace=True)
        dat = cp.zeros((nao*nao, nao))
        dat[i*nao+j] = dat[j*nao+i] = eri3c_batch
        dat = dat.reshape(nao,nao,nao)
        dat = contract('pqr,rk->pqk', dat, aux_coeff)
        dat = contract('pqk,qj->pjk', dat, coeff)
        dat = contract('pjk,pi->ijk', dat, coeff)
        assert abs(dat.get() - ref).max() < 1e-9
