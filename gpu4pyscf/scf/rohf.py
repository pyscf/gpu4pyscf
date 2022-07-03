from pyscf.scf import rohf
from gpu4pyscf.scf.hf import _get_jk
from gpu4pyscf.lib.utils import patch_cpu_kernel


class ROHF(rohf.ROHF):
    device = 'gpu'
    get_jk = patch_cpu_kernel(rohf.ROHF.get_jk)(_get_jk)
