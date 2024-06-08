# Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
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
dispersion correction for HF and DFT
'''

from pyscf.scf import dispersion
from gpu4pyscf.scf import hf, uhf
from gpu4pyscf.dft import rks, uks

# Inject to SCF class
hf.SCF.do_disp = dispersion.check_disp

hf.RHF.get_dispersion = dispersion.get_dispersion
uhf.UHF.get_dispersion = dispersion.get_dispersion
rks.RKS.get_dispersion = dispersion.get_dispersion
uks.UKS.get_dispersion = dispersion.get_dispersion
