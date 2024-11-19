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

'''Hartree-Fock for periodic systems
'''

from .import hf
#from . import uhf
#from . import khf
#from . import kuhf

rhf = hf
#krhf = khf

#UHF = uhf.UHF
RHF = rhf.RHF
#KRHF = krhf.KRHF
#KUHF = kuhf.KRHF
