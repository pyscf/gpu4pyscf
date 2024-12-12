__version__ = '1.2.0'

# monkey patch libxc reference due to a bug in nvcc
from pyscf.dft import libxc
libxc.__reference__ = 'unable to decode the reference due to https://github.com/NVIDIA/cuda-python/issues/29'

from . import lib, grad, hessian, solvent, scf, dft
