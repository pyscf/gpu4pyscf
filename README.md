GPU plugin for PySCF
====================

Examples
--------
```
import pyscf
# Load the gpu4pyscf to enable GPU mode
from gpu4pyscf import patch_pyscf

mol = pyscf.M(atom='''
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161''',
basis='ccpvdz',
verbose=5)
mf = mol.RHF().run()

# Run SCF with CPU
mf.device = 'cpu'
mf.run()
```
