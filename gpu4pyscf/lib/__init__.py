# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
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

from importlib.util import find_spec

import os
import numpy
from gpu4pyscf.lib import diis


has_dpctl = find_spec("dpctl")

if not has_dpctl:
    from gpu4pyscf.lib import cupy_helper
    from gpu4pyscf.lib import cutensor
else:
    from gpu4pyscf.lib import dpnp_helper

try:
    from gpu4pyscf.lib import dftd3
except Exception:
    print('failed to load DFTD3')

try:
    from gpu4pyscf.lib import dftd4
except Exception:
    print('failed to load DFTD4')
