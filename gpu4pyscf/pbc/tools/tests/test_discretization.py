# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import numpy as np
from gpu4pyscf.pbc.tools.discretization import freeze_mesh


class _Cell:
    def __init__(self, mesh, mesh_from_build=True):
        self._mesh = np.asarray(mesh)
        self._mesh_from_build = mesh_from_build

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        self._mesh = value
        self._mesh_from_build = False


def _method(cell_mesh=(20, 22, 24), object_mesh=(10, 10, 10)):
    cell = _Cell(cell_mesh)
    return SimpleNamespace(
        cell=cell,
        _numint=SimpleNamespace(mesh=object_mesh),
        with_df=SimpleNamespace(mesh=object_mesh),
        grids=SimpleNamespace(mesh=object_mesh),
    )


def test_freezes_automatic_cell_mesh_and_synchronizes_objects():
    method = _method()

    mesh = freeze_mesh(method)

    assert mesh == (20, 22, 24)
    assert method.cell._mesh_from_build is False
    for obj in (method.cell, method._numint, method.with_df, method.grids):
        np.testing.assert_array_equal(obj.mesh, mesh)


def test_reuses_reference_mesh():
    method = _method(cell_mesh=(22, 24, 26))

    mesh = freeze_mesh(method, mesh=(20, 22, 24))

    assert mesh == (20, 22, 24)
    for obj in (method.cell, method._numint, method.with_df, method.grids):
        np.testing.assert_array_equal(obj.mesh, mesh)
