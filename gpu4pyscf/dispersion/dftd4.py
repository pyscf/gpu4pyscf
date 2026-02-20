# Copyright 2024 The PySCF Developers. All Rights Reserved.
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

from pyscf import dispersion
from pyscf.dispersion.dftd4 import *

_version_major, _version_minor = dispersion.__version__.split('.')[:2]
if int(_version_major) == 1 and int(_version_minor) < 4:
    # Override DFTD4Dispersion class for compatibility with older versions
    class DFTD4Dispersion(DFTD4Dispersion):
        def __init__(self, mol, xc, version='d4', ga=None, gc=None, wf=None, atm=False):
            if version != 'd4':
                raise RuntimeError(
                    f'pyscf-dispersion {dispersion.__version__} does not support '
                    f'dftd4 code {version}. It is available in pyscf-dispersion>=1.5')
            super().__init__(mol, xc, ga, gc, wf, atm)
