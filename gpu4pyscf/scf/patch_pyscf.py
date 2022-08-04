'''
Patch pyscf SCF modules to make all subclass of SCF class support GPU mode.
'''

from gpu4pyscf.scf.hf import _get_jk, _eigh
from pyscf.scf.hf import SCF
from gpu4pyscf.lib.utils import patch_cpu_kernel

# The device attribute is patched to the pyscf base SCF module. It will be
# seen by all subclasses.
SCF.device = 'gpu'

print(f'{SCF} monkey-patched')
SCF.get_jk = patch_cpu_kernel(SCF.get_jk)(_get_jk)
SCF._eigh = patch_cpu_kernel(SCF._eigh)(_eigh)
