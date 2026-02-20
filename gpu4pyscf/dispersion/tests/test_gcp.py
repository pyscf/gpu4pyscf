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

import numpy as np
import pytest
import pyscf
from gpu4pyscf.dispersion.gcp import GCP

def test_gcp_unknown_xc():
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1')
    with pytest.raises(RuntimeError):
        model = GCP(mol, method='xx')
        model.get_counterpoise()

def test_b973c_energy():
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1')
    model = GCP(mol, method='b973c')
    out = model.get_counterpoise(grad=True)
    assert abs(np.linalg.norm(out['energy']) - 0.000818489452022319) < 1e-8
    assert abs(np.linalg.norm(out['gradient']) - 0.0028068213098591467) < 1e-8
    assert abs(np.linalg.norm(out['virial']) - 0.0037505817348592453) < 1e-8

def test_r2scan3c_energy():
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1')
    model = GCP(mol, method='r2scan3c')
    out = model.get_counterpoise(grad=True)
    assert abs(np.linalg.norm(out['energy']) - 0.0002982079859909563) < 1e-8
    assert abs(np.linalg.norm(out['gradient']) - 0.0007006492552500862) < 1e-8
    assert abs(np.linalg.norm(out['virial']) - 0.0009362342697247007) < 1e-8 

# wb97x-3c does not require GCP
def test_wb97x3c_energy():
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1')
    model = GCP(mol, method='wb97x3c')
    out = model.get_counterpoise(grad=True)
    assert abs(np.linalg.norm(out['energy'])) < 1e-8
    assert abs(np.linalg.norm(out['gradient'])) < 1e-8
    assert abs(np.linalg.norm(out['virial'])) < 1e-8
