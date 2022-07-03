'''
Aggresively patch all PySCF modules if applicable
This patch may break some pyscf modules.
'''

from gpu4pyscf.scf import patch_pyscf
from gpu4pyscf.dft import patch_pyscf
del patch_pyscf
