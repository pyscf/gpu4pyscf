from pyscf.dft import roks
from gpu4pyscf.dft import numint
from gpu4pyscf.scf.hf import _get_jk, _eigh
from gpu4pyscf.lib.utils import patch_cpu_kernel

class ROKS(roks.ROKS):
    def __init__(self, mol, xc='LDA,VWN'):
        super().__init__(mol, xc)
        self._numint = numint.NumInt()

    @property
    def device(self):
        return self._numint.device
    @device.setter
    def device(self, value):
        self._numint.device = value

    get_jk = patch_cpu_kernel(roks.ROKS.get_jk)(_get_jk)
    _eigh = patch_cpu_kernel(roks.ROKS._eigh)(_eigh)
