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
    eri3c = int3c2e_opt.int3c2e_bdiv_kernel()
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

    eri3c = int3c2e_opt.int3c2e_bdiv_kernel()
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
