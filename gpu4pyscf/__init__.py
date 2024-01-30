from . import lib, grad, hessian, solvent, scf, dft

__version__ = '0.7.0'

# monkey patch libxc reference due to a bug in nvcc
from pyscf.dft import libxc
libxc.__reference__ = 'unable to decode the reference due to https://github.com/NVIDIA/cuda-python/issues/29'

from gpu4pyscf.lib.utils import patch_cpu_kernel
from gpu4pyscf.lib.cupy_helper import tag_array
from pyscf import lib
lib.tag_array = tag_array
