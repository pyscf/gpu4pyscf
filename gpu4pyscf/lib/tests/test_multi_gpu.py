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
import numpy as np
import cupy as cp
from gpu4pyscf.lib import multi_gpu
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

    def test_reduce_promotes_dtype_and_forwards_arguments(self):
        def term(value, offset, *, scale):
            dtype = cp.int32 if value == 1 else cp.float64
            return cp.asarray([value, value + offset], dtype=dtype) * scale

        result = multi_gpu.reduce(term, [1, 2], args=(3,), kwargs={'scale': 2})
        self.assertEqual(result.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(result.get(), [6, 18])

    def test_array_broadcast(self):
        expected = np.arange(6, dtype=np.float64).reshape(2, 3)
        with cp.cuda.Device(0):
            result = multi_gpu.array_broadcast(cp.asarray(expected))
        self.assertEqual(len(result), num_devices)
        for device_id, array in enumerate(result):
            with self.subTest(device=device_id):
                self.assertEqual(array.device.id, device_id)
                np.testing.assert_array_equal(array.get(), expected)

    def test_array_reduce(self):
        base = np.arange(6, dtype=np.float64).reshape(2, 3)
        expected = base * sum(range(1, num_devices + 1))
        for inplace in (False, True):
            with self.subTest(inplace=inplace):
                arrays = []
                for device_id in range(num_devices):
                    with cp.cuda.Device(device_id):
                        arrays.append(cp.asarray(base * (device_id + 1)))
                with cp.cuda.Device(0):
                    result = multi_gpu.array_reduce(arrays, inplace=inplace)
                self.assertEqual(result.device.id, 0)
                np.testing.assert_array_equal(result.get(), expected)

    def test_synchronize_waits_for_nondefault_stream(self):
        array = cp.empty(4)
        stream = cp.cuda.Stream(non_blocking=True)
        event = cp.cuda.Event()
        with stream:
            array.fill(7.)
            event.record(stream)
        multi_gpu.synchronize()
        self.assertTrue(event.done)
        np.testing.assert_array_equal(array.get(), np.full(4, 7.))

if __name__ == "__main__":
    print("Full tests for multi_gpu helper functions")
    unittest.main()
