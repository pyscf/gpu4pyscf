from . import rks
from .rks import KohnShamDFT
from .rks_lowmem import RKS as LRKS
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
    if mol.spin == 0:
        return rks.RKS(mol, xc)
    else:
        return ROKS(mol, xc)
