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

import numpy as np


def freeze_mesh(method, cell=None, mesh=None):
    """Make the current PBC mesh explicit and share it with method objects."""
    if cell is None:
        cell = method.cell
    if mesh is None:
        mesh = cell.mesh
    if mesh is None:
        raise RuntimeError("PBC mesh is not initialized")

    mesh = np.asarray(mesh, dtype=np.int32).copy()
    cell.mesh = mesh.copy()
    objects = (
        getattr(method, "_numint", None),
        getattr(method, "with_df", None),
        getattr(method, "grids", None),
    )
    for obj in objects:
        if obj is not None and hasattr(obj, "mesh"):
            obj.mesh = mesh.copy()
    return tuple(int(value) for value in mesh)
