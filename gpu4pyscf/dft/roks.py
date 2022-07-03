from pyscf.dft import roks
from gpu4pyscf.dft import numint

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
