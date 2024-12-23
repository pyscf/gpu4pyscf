# Copyright 2024 The PySCF Developers. All Rights Reserved.
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

from functools import reduce
from fractions import Fraction
import itertools
import numpy as np
from pyscf.lib import logger

# This version of kpts_to_kmesh will be available in PySCF-2.8
def kpts_to_kmesh(cell, kpts, precision=None, max_images=10000):
    '''Find the minimal k-points mesh to include all input kpts'''
    scaled_kpts = cell.get_scaled_kpts(kpts)
    logger.debug3(cell, '    scaled_kpts kpts %s', scaled_kpts)
    # cell.nimgs are the upper limits for kmesh
    kmesh = np.asarray(cell.nimgs) * 2 + 1
    if precision is None:
        precision = cell.precision * 1e2
    for i in range(3):
        floats = scaled_kpts[:,i]
        uniq_floats_idx = np.unique(floats.round(6), return_index=True)[1]
        uniq_floats = floats[uniq_floats_idx]
        # Limit the number of images to 30 in each direction
        fracs = [Fraction(x).limit_denominator(int(kmesh[i])) for x in uniq_floats]
        denominators = np.unique([x.denominator for x in fracs])
        common_denominator = reduce(np.lcm, denominators)
        fs = common_denominator * uniq_floats
        if abs(uniq_floats - np.rint(fs)/common_denominator).max() < precision:
            kmesh[i] = min(kmesh[i], common_denominator)
        if cell.verbose >= logger.DEBUG3:
            logger.debug3(cell, 'dim=%d common_denominator %d  error %g',
                          i, common_denominator, abs(fs - np.rint(fs)).max())
            logger.debug3(cell, '    unique kpts %s', uniq_floats)
            logger.debug3(cell, '    frac kpts %s', fracs)

    assert max_images > 0
    if np.prod(kmesh) > max_images:
        kmesh_raw = kmesh.copy()
        for i in itertools.cycle(np.argsort(kmesh)[::-1]):
            kmesh[i] = int(kmesh[i] * .8)
            if np.prod(kmesh) < max_images:
                break
        logger.warn(cell, 'kmesh (%s) exceeds max_images (%d); reduced to %s',
                    kmesh_raw, max_images, kmesh)
    return kmesh
