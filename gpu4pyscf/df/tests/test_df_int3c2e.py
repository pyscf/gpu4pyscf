import pytest
import numpy as np
import cupy as cp
import pyscf
from pyscf import lib
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
        'C2': ('unc-weigend', [[0, [.5, 1.]], [1, [.8, 1.]], [3, [.9, 1]]]),
    }
    auxmol.build()
    dat = int3c2e_bdiv.aux_e2(mol, auxmol)
    ref = incore.aux_e2(mol, auxmol)
    assert abs(dat.get()-ref).max() < 1e-10

def test_int3c2e_1():
    mol = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
                C2   1.    .3      1.1
                C2   .1    1.1     -.1
                C2   .4    -.1     -.1
                C2   -.3    .2     -.7
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.5, 1.], [.9, 1.]],
                       [4, [2., 1.]]]),
               'C2': 'ccpvdz'},
    )
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
        'C2': ('unc-weigend', [[0, [.5, 1.]], [1, [.8, 1.]], [3, [.9, 1]]]),
    }
    auxmol.build()
    for cart in (True, False):
        mol.cart = cart
        auxmol.cart = cart
        nao = mol.nao
        naux = auxmol.nao
        int3c2e_opt = int3c2e_bdiv.Int3c2eOpt(mol, auxmol).build()
        results = []
        for reorder_aux in (True, False):
            eval_j3c, aux_sorting = int3c2e_opt.int3c2e_evaluator(
                reorder_aux=reorder_aux, cart=mol.cart)[:2]
            j3c = eval_j3c()
            aux_coef = int3c2e_opt.auxmol.ctr_coeff
            aux_coef, tmp = cp.empty_like(aux_coef), aux_coef
            aux_coef[aux_sorting] = tmp
            j3c = j3c.dot(aux_coef)
            pair_address = int3c2e_opt.pair_and_diag_indices()[0]
            rows, cols = divmod(pair_address, nao)
            dat = cp.zeros((nao, nao, naux))
            dat[cols,rows] = j3c
            dat[rows,cols] = j3c
            results.append(dat)
        #ref = incore.aux_e2(mol, auxmol)
        assert abs(results[0]-results[1]).max() < 1e-10
        if cart:
            assert abs(lib.fp(results[0].get()) - 1331.2232227224067) < 1e-9
        else:
            assert abs(lib.fp(results[0].get()) - 27.77438089588688) < 1e-10

def test_int3c2e_batch_evaluation():
    mol = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
                C2   1.    .3      1.1
                C2   .1    1.1     -.1
                C2   .4    -.1     -.1
                C2   -.3    .2     -.7
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.5, 1.], [.9, 1.]],
                       [4, [2., 1.]]]),
               'C2': 'ccpvdz'},
    )
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
        'C2': ('unc-weigend', [[0, [.5, 1.]], [1, [.8, 1.]], [3, [.9, 1]]]),
    }
    auxmol.build()
    for cart in (True, False):
        mol.cart = cart
        auxmol.cart = cart
        opt = int3c2e_bdiv.Int3c2eOpt(mol, auxmol).build()
        for reorder_aux in (True, False):
            eval_j3c, aux_sorting = opt.int3c2e_evaluator(
                reorder_aux=reorder_aux, cart=mol.cart)[:2]
            ref = eval_j3c()[:,aux_sorting]
            batch_size = int(ref.shape[0] *.23)

            eval_j3c, aux_sorting, ao_pair_offsets = opt.int3c2e_evaluator(
                ao_pair_batch_size=batch_size, reorder_aux=reorder_aux, cart=mol.cart)[:3]
            dat = cp.empty_like(ref)
            for i, (p0, p1) in enumerate(zip(ao_pair_offsets[:-1],
                                             ao_pair_offsets[1:])):
                dat[p0:p1] = eval_j3c(i)
            assert abs(dat[:,aux_sorting] - ref).max() < 1e-12

            batch_size = int(ref.shape[1] * 0.22)
            eval_j3c, aux_sorting, ao_pair_offsets, aux_offsets = opt.int3c2e_evaluator(
                aux_batch_size=batch_size, reorder_aux=reorder_aux, cart=mol.cart)[:4]
            dat = cp.empty_like(ref)
            for i, (p0, p1) in enumerate(zip(aux_offsets[:-1], aux_offsets[1:])):
                dat[:,p0:p1] = eval_j3c(aux_batch_id=i)
            assert abs(dat[:,aux_sorting] - ref).max() < 1e-12

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

    nao = mol.nao
    naux = auxmol.nao
    out = cp.zeros((nao, nao, naux))
    eri3c = next(int3c2e_opt.int3c2e_bdiv_generator())
    eri3c = eri3c.dot(int3c2e_opt.aux_coeff)
    ao_pair_mapping = int3c2e_opt.create_ao_pair_mapping()
    i, j = divmod(ao_pair_mapping, nao)
    out[j, i] = eri3c
    out[i, j] = eri3c
    ref = incore.aux_e2(mol, auxmol)
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

    nao = mol.nao
    naux = auxmol.nao
    eri3c = [x for x in int3c2e_opt.int3c2e_bdiv_generator(batch_size=8)]
    eri3c = cp.hstack(eri3c)
    eri3c = eri3c.dot(int3c2e_opt.aux_coeff)
    ao_pair_mapping = int3c2e_opt.create_ao_pair_mapping()
    i, j = divmod(ao_pair_mapping, nao)
    out = cp.zeros((nao, nao, naux))
    out[j, i] = eri3c
    out[i, j] = eri3c
    assert abs(out.get()-ref).max() < 1e-10

    eri3c, rows, cols = int3c2e_bdiv.compressed_aux_e2(mol, auxmol)
    out = cp.zeros((nao, nao, auxmol.nao))
    out[rows,cols] = eri3c
    out[cols,rows] = eri3c
    assert abs(out.get()-ref).max() < 1e-10

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

    nao = mol.nao
    int3c2e_opt = int3c2e_bdiv.Int3c2eOpt(mol, mol).build()
    ao_pair_mapping = int3c2e_opt.create_ao_pair_mapping()
    i, j = divmod(ao_pair_mapping, nao)
    eri3c = next(int3c2e_opt.int3c2e_bdiv_generator())
    eri3c = eri3c.dot(int3c2e_opt.aux_coeff)
    dat = cp.zeros((nao, nao, nao))
    dat[j, i] = eri3c
    dat[i, j] = eri3c
    dat = dat.reshape(nao,nao,nao)
    assert abs(dat.get() - ref).max() < 1e-9

def test_contract_int3c2e():
    from gpu4pyscf.df.int3c2e_bdiv import contract_int3c2e_auxvec, contract_int3c2e_dm
    from gpu4pyscf.df.j_engine_3c2e import contract_int3c2e_dm as j_engine
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
    dm = np.random.rand(nao,nao) - .5
    dm = dm.dot(dm.T)
    eri3c = incore.aux_e2(mol, auxmol)

    dat = j_engine(mol, auxmol, dm)
    ref = np.einsum('ijP,ji->P', eri3c, dm)
    assert abs(dat.get() - ref).max() < 1e-9

    dat = contract_int3c2e_dm(mol, auxmol, dm)
    assert abs(dat.get() - ref).max() < 1e-9

    auxvec = np.random.rand(auxmol.nao)
    dat = contract_int3c2e_auxvec(mol, auxmol, auxvec)
    ref = np.einsum('ijP,P->ij', eri3c, auxvec)
    assert abs(dat.get() - ref).max() < 1e-9

    dm = np.random.rand(6, nao, nao)
    dat = contract_int3c2e_dm(mol, auxmol, dm)
    ref = np.einsum('ijP,nji->nP', eri3c, dm)
    assert abs(dat.get() - ref).max() < 1e-9

def test_int2c2e():
    mol = pyscf.M(
        atom='''C1   1.3   .2       .3
                C2   .19   .1      1.1
                C1   .5   -.1      0.2
                C2   .04   .6       .5
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]
                     ),
               'C2': 'ccpvtz'}
    )
    j2c = int3c2e_bdiv.int2c2e(mol)
    ref = mol.intor('int2c2e')
    assert abs(j2c.get() - ref).max() < 3e-11

    j2c = int3c2e_bdiv.int2c2e_ip1(mol)
    ref = mol.intor('int2c2e_ip1')
    assert abs(j2c.get() - ref).max() < 1e-11

def test_int3c2e_rsh():
    mol = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis='ccpvdz'
    )
    auxmol = mol.copy()
    auxmol.basis = 'ccpvdz-jkfit'
    auxmol.build()
    int3c2e_opt = int3c2e_bdiv.Int3c2eOpt(mol, auxmol).build()

    nao = mol.nao
    naux = auxmol.nao
    out = cp.zeros((nao, nao, naux))
    omega = 0.33
    lr_factor = 0.65
    sr_factor = 0.19
    eval_j3c, aux_sorting = int3c2e_opt.int3c2e_evaluator(
        omega=omega, lr_factor=lr_factor, sr_factor=sr_factor)[:2]
    eri3c = eval_j3c()
    eri3c = eri3c[:,aux_sorting].dot(int3c2e_opt.aux_coeff)
    cp.cuda.get_current_stream().synchronize()
    pair_address = int3c2e_opt.pair_and_diag_indices()[0]
    i, j = divmod(pair_address, nao)
    out[j, i] = eri3c
    out[i, j] = eri3c
    with mol.with_range_coulomb(omega):
        ref = incore.aux_e2(mol, auxmol) * lr_factor
    with mol.with_range_coulomb(-omega):
        ref += incore.aux_e2(mol, auxmol) * sr_factor
    assert abs(out.get()-ref).max() < 1e-12
