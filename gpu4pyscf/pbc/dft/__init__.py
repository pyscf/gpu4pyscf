#!/usr/bin/env python
#
# Copyright 2024 The GPU4PySCF Developers. All Rights Reserved.
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

'''Kohn-Sham DFT for periodic systems
'''

from .gen_grid import UniformGrids, BeckeGrids
from . import rks
#from . import uks
#from . import krks
#from . import kuks
from .rks import KohnShamDFT

RKS = rks.RKS
#UKS = uks.UKS
#KRKS = krks.KRKS
#KUKS = kuks.KUKS
