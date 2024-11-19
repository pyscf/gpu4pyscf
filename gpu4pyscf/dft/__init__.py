from . import rks
from .rks import RKS, KohnShamDFT
from .uks import UKS
from .gks import GKS
from .roks import ROKS
from .gen_grid import Grids

def KS(mol, xc='LDA,VWN'):
    if mol.spin == 0:
        return RKS(mol, xc)
    else:
        return UKS(mol, xc)
