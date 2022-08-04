from pyscf.dft import gks
from gpu4pyscf.dft import numint
from gpu4pyscf.scf.ghf import get_jk, _eigh
from gpu4pyscf.lib.utils import patch_cpu_kernel

class GKS(gks.GKS):
    def __init__(self, mol, xc='LDA,VWN'):
        super().__init__(mol, xc)
        self._numint = numint.NumInt()

    @property
    def device(self):
        return self._numint.device
    @device.setter
    def device(self, value):
        self._numint.device = value

    @patch_cpu_kernel(gks.GKS.get_jk)
    def get_jk(self, mol=None, dm=None, hermi=0, with_j=True, with_k=True,
               omega=None):
        return get_jk(mol, dm, hermi, with_j, with_k, omega)

    _eigh = patch_cpu_kernel(gks.GKS._eigh)(_eigh)
