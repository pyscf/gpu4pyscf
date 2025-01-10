import tempfile
import numpy as np
import pyscf
from pyscf.pbc.df.rsdf_builder import _RSGDFBuilder
from pyscf.pbc.df.df import _load3c
from gpu4pyscf.pbc.df.rsdf_builder import build_cderi

def test_gamma_point():
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': [[0, [1.1, 1.]],
                      [1, [2., 1.]]],
               'C2': 'ccpvdz'},
        a=np.diag([2.5, 1.9, 2.2])*3)

    auxcell = cell.copy()
    auxcell.basis = {
        'C1':'''
C    P
    102.9917624900           1.0000000000
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
        'C2':[[0, [.5, 1.]]],
    }
    auxcell.build()
    omega = 0.3
    kmesh = [1,1,1]
    gpu_dat, dat_neg = build_cderi(cell, auxcell, kmesh=kmesh, omega=omega)

    cell.precision = 1e-10
    auxcell.precision = 1e-10
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    dfbuilder = _RSGDFBuilder(cell, auxcell, kpts)
    dfbuilder.omega = omega
    dfbuilder.j2c_eig_always = True
    dfbuilder.fft_dd_block = True
    dfbuilder.exclude_d_aux = True
    naux = auxcell.nao
    nao = cell.nao
    with tempfile.NamedTemporaryFile() as tmpf:
        dfbuilder.make_j3c(tmpf.name, aosym='s1')
        for ki in range(nkpts):
            for kj in range(nkpts):
                with _load3c(tmpf.name, 'j3c', kpts[[ki,kj]]) as cderi:
                    ref = abs(cderi[:].reshape(naux,nao,nao))
                    dat = abs(gpu_dat[ki,kj].transpose(2,0,1).get())
                    print(ki,kj)
                    assert abs(dat - ref).max() < 3e-8

test_gamma_point()
