# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
import cupy as cp
from gpu4pyscf.lib.multi_gpu import lru_cache
from gpu4pyscf.__config__ import num_devices

@unittest.skipIf(num_devices == 1, 'Single GPU')
class KnownValues(unittest.TestCase):
    def test_lru_cache(self):
        counts = 0

        @lru_cache(10)
        def fn():
            nonlocal counts
            counts += 1

        with cp.cuda.Device(0):
            fn()
        with cp.cuda.Device(1):
            fn()
        assert counts == 2

if __name__ == "__main__":
    print("Full tests for multi_gpu helper functions")
    unittest.main()
