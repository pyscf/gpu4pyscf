#!/usr/bin/env python

'''
Gamma point Hartree-Fock/DFT using density fitting approximation
'''

import numpy as np
import pyscf

cell = pyscf.M(
    a = np.eye(3)*3.5668,
    atom = '''C     0.      0.      0.    
              C     0.8917  0.8917  0.8917
              C     1.7834  1.7834  0.    
              C     2.6751  2.6751  0.8917
              C     1.7834  0.      1.7834
              C     2.6751  0.8917  2.6751
              C     0.      1.7834  1.7834
              C     0.8917  2.6751  2.6751''',
    basis = 'ccpvdz',
    verbose = 5,
)

mf = cell.RHF().to_gpu().density_fit().run()

mf = cell.RKS(xc='pbe0').to_gpu().density_fit().run()
