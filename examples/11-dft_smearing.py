#!/usr/bin/env python

'''Fermi-Dirac or Gaussian smearing for DFT calculation'''

import pyscf

mol = pyscf.M(
    atom='''
Fe 0 0 1
Fe 1 0 1
''',
    basis='ccpvdz',
    verbose=4,
)

# The .to_gpu() transfer must be executed before calling .smearing().
# Currently, to_gpu() does not support the transfer of the smearing setup.
mf = mol.RKS(xc='pbe').to_gpu().smearing(sigma=0.1).density_fit().run()
