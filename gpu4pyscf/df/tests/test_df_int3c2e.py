import pyscf
from pyscf.df import incore

def test_int3c2e():
    mol = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': [[3, [1.1, 1.]],
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
    dat = aux_e2(mol, auxmol)
    ref = incore.aux_e2(mol, auxmol)
    assert abs(dat.get()-ref).max() < 1e-10

def test_int3c2e_bdiv():
    mol = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': [[3, [1.1, 1.]],
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
    int3c2e_opt = Int3c2eOpt(mol, auxmol)
    nao, nao_orig = int3c2e_opt.coeff.shape
    naux = int3c2e_opt.aux_coeff.shape[0]
    out = cp.zeros((nao*nao, naux))
    eri3c, ao_pair_mapping, aux_mapping = int3c2e_opt.int3c2e_bdiv_kernel()
    out[ao_pair_mapping] = eri3c
    i, j = divmod(ao_pair_mapping, nao)
    out[j*nao+i] = eri3c
    out = out.reshape(nao, nao, naux)
    aux_coeff = cp.empty_like(int3c2e_opt.aux_coeff)
    aux_coeff[aux_mapping] = int3c2e_opt.aux_coeff
    out = contract('pqr,rk->pqk', out, aux_coeff)
    out = contract('pqk,qj->pjk', out, int3c2e_opt.coeff)
    out = contract('pjk,pi->ijk', out, int3c2e_opt.coeff)
    ref = incore.aux_e2(mol, auxmol)
    assert abs(out.get()-ref).max() < 1e-10
