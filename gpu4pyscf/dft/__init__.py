from . import rks
from .rks import KohnShamDFT
from .uks import UKS
from .gks import GKS
from .roks import ROKS
from .gen_grid import Grids

def KS(mol, xc='LDA,VWN'):
    if mol.spin == 0:
        return RKS(mol, xc)
    else:
        return UKS(mol, xc)

def RKS(mol, xc='LDA,VWN'):
    from gpu4pyscf.lib.cupy_helper import get_avail_mem
    from . import rks_lowmem
    mem = get_avail_mem()
    nao = mol.nao
    if nao**2*40*8 > mem:
        return rks_lowmem.RKS(mol, xc)
    else:
        return rks.RKS(mol, xc)
