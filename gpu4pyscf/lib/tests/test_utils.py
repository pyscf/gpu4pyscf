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
import numpy
from gpu4pyscf.lib import utils


class KnownValues(unittest.TestCase):
    def test_splits_by_blocksize_group_boundaries(self):
        offsets = numpy.array([0, 2, 5, 9], dtype=numpy.int32)
        self.assertEqual(utils.splits_by_blocksize(offsets, 5), [0, 2, 3])
        numpy.testing.assert_array_equal(offsets, [0, 2, 5, 9])

    def test_splits_by_blocksize_oversized_group(self):
        # A single group cannot be divided, even when larger than the target.
        offsets = numpy.array([0, 8, 10], dtype=numpy.int32)
        self.assertEqual(utils.splits_by_blocksize(offsets, 3), [0, 1, 2])

    def test_splits_by_blocksize_empty_groups(self):
        offsets = numpy.array([0, 0, 3, 3, 7], dtype=numpy.int32)
        self.assertEqual(utils.splits_by_blocksize(offsets, 3), [0, 3, 4])

    def test_splits_by_blocksize_single_batch(self):
        for offsets, block_size, expected in (
                ([0], 3, [0]),
                ([0, 2, 5], 5, [0, 2]),
                ([0, 2, 5], 8, [0, 2])):
            with self.subTest(offsets=offsets, block_size=block_size):
                self.assertEqual(utils.splits_by_blocksize(
                    numpy.asarray(offsets), block_size), expected)

    def test_nearest_power2_scalar_rounding(self):
        for n, lower, upper in ((1, 1, 1), (2, 2, 2), (3, 2, 4),
                                 (7, 4, 8), (8, 8, 8), (9, 8, 16), (16, 16, 16)):
            with self.subTest(n=n):
                self.assertEqual(utils.nearest_power2(n), lower)
                self.assertEqual(utils.nearest_power2(n, return_leq=False), upper)

    def test_nearest_power2_array_rounding(self):
        counts = numpy.array([[1, 2, 3, 4], [7, 8, 9, 16]], dtype=numpy.int32)
        lower = utils.nearest_power2(counts)
        upper = utils.nearest_power2(counts, return_leq=False)
        numpy.testing.assert_array_equal(lower, [[1, 2, 2, 4], [4, 8, 8, 16]])
        numpy.testing.assert_array_equal(upper, [[1, 2, 4, 4], [8, 8, 16, 16]])
        numpy.testing.assert_array_equal(counts, [[1, 2, 3, 4], [7, 8, 9, 16]])

    def test_indices_within_groups_empty_group(self):
        counts = [3, 0, 2, 1]
        indices = utils.indices_within_groups(counts)
        numpy.testing.assert_array_equal(indices, [0, 1, 2, 0, 1, 0])
        self.assertTrue(numpy.issubdtype(indices.dtype, numpy.integer))
        self.assertEqual(counts, [3, 0, 2, 1])

    def test_indices_within_groups_empty_output(self):
        for counts in (numpy.empty(0, dtype=numpy.int32),
                       numpy.zeros(3, dtype=numpy.int32)):
            with self.subTest(counts=counts):
                indices = utils.indices_within_groups(counts)
                numpy.testing.assert_array_equal(indices, numpy.empty(0, dtype=int))
                self.assertTrue(numpy.issubdtype(indices.dtype, numpy.integer))


if __name__ == '__main__':
    unittest.main()
