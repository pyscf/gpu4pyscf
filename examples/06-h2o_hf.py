from pyscf import gto, scf, grad
import numpy as np

mol = gto.M(atom=
'''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
''',
basis='cc-pvtz',
verbose=4)

mf = scf.hf.RHF(mol)
mf.direct_scf_tol = 1e-10
mf.kernel()

cpu_gradient = grad.rhf.Gradients(mf)
cpu_gradient.kernel()

import gpu4pyscf
gpu_gradient = gpu4pyscf.grad.rhf.Gradients(mf)
gpu_gradient.kernel()

assert(np.max(np.abs(cpu_gradient.de - gpu_gradient.de)) < 1e-10)
