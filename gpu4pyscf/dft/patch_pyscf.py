# gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
#
# Copyright (C) 2022 Qiming Sun
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
Patch pyscf DFT modules to make all subclass of DFT class support GPU mode.
'''

from pyscf.dft.rks import KohnShamDFT
from pyscf.dft.numint import NumInt
from gpu4pyscf.dft import numint as gpu_numint

print(f'{NumInt} monkey-patched')
NumInt.get_rho = gpu_numint.get_rho
NumInt.nr_rks = gpu_numint.nr_rks
NumInt.nr_uks = gpu_numint.nr_uks
NumInt.nr_rks_fxc = gpu_numint.nr_rks_fxc
NumInt.nr_uks_fxc = gpu_numint.nr_uks_fxc
NumInt.nr_rks_fxc_st = gpu_numint.nr_rks_fxc_st
NumInt.cache_xc_kernel = gpu_numint.cache_xc_kernel
