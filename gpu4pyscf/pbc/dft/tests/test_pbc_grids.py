# Copyright 2025 The PySCF Developers. All Rights Reserved.
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

import unittest
import numpy as np
import pyscf
from gpu4pyscf.pbc.dft import gen_grid

class KnownValues(unittest.TestCase):
    def test_argsort(self):
        cell = pyscf.M(atom='He 0 0 0', a=np.eye(3)*3)
        grids = gen_grid.UniformGrids(cell)
        grids.mesh = [19] * 3
        for tile in [3, 4, 6, 8]:
            idx = grids.argsort(tile=tile)
            self.assertEqual(len(np.unique(idx)), 19**3)

if __name__ == '__main__':
    print("Full Tests for pbc.dft.numint")
    unittest.main()
