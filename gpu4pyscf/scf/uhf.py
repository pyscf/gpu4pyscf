from pyscf.scf import uhf
from gpu4pyscf.scf.hf import _get_jk
from gpu4pyscf.lib.utils import patch_cpu_kernel

class UHF(uhf.UHF):
    device = 'gpu'
    get_jk = patch_cpu_kernel(uhf.UHF.get_jk)(_get_jk)
