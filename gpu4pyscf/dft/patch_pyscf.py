'''
Patch pyscf DFT modules to make all subclass of DFT class support GPU mode.
'''

from pyscf.dft.rks import KohnShamDFT
from pyscf.dft.numint import NumInt
from gpu4pyscf.dft import numint as gpu_numint
from gpu4pyscf.lib.utils import patch_cpu_kernel

print(f'{NumInt} monkey-patched')
NumInt.device = 'gpu'
NumInt.get_rho = gpu_numint.get_rho
NumInt.nr_rks = gpu_numint.nr_rks
NumInt.nr_uks = gpu_numint.nr_uks
NumInt.nr_rks_fxc = gpu_numint.nr_rks_fxc
NumInt.nr_uks_fxc = gpu_numint.nr_uks_fxc
NumInt.nr_rks_fxc_st = gpu_numint.nr_rks_fxc_st
NumInt.cache_xc_kernel = gpu_numint.cache_xc_kernel

print(f'{KohnShamDFT} monkey-patched')
def _get_device(obj):
    return getattr(self._numint, 'device', 'cpu')
def _set_device(obj, value):
    self._numint.device = value
KohnShamDFT.device = property(_get_device, _set_device)
