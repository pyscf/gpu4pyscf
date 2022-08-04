from pyscf.scf import rohf
from gpu4pyscf.scf.hf import _get_jk, _eigh
from gpu4pyscf.lib.utils import patch_cpu_kernel


class ROHF(rohf.ROHF):
    device = 'gpu'
    get_jk = patch_cpu_kernel(rohf.ROHF.get_jk)(_get_jk)
    _eigh = patch_cpu_kernel(rohf.ROHF._eigh)(_eigh)
