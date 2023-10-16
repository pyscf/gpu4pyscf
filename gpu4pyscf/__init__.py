from . import lib, grad, hessian, solvent, scf, dft
__version__ = '0.6.1'

import pyscf
from gpu4pyscf.lib.utils import patch_cpu_kernel

# patch tag_array for compatibility
pyscf.lib.tag_array = patch_cpu_kernel(pyscf.lib.tag_array)(lib.cupy_helper.tag_array)