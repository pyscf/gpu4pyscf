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
gradient of dispersion correction for HF and DFT
'''

import numpy
from pyscf.grad import dispersion
from gpu4pyscf import dft

# Inject to Gradient
from gpu4pyscf.grad import rhf, uhf, rks, uks
rhf.Gradients.get_dispersion = dispersion.get_dispersion
uhf.Gradients.get_dispersion = dispersion.get_dispersion
rks.Gradients.get_dispersion = dispersion.get_dispersion
uks.Gradients.get_dispersion = dispersion.get_dispersion
