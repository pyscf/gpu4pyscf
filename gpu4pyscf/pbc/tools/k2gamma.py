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

def kpts_to_kmesh(cell, kpts, precision=None, rcut=None, bound_by_supmol=True):
    '''Search the minimal BvK mesh or Monkhorst-Pack k-point mesh

    bound_by_supmol:
        If True, the largest k-mesh is constrained within the supmol.
        If False, the k-mesh must exactly reproduce the provided k-points.
    '''
    if kpts is None:
        return np.ones(3, dtype=int)

    assert kpts.ndim == 2
    scaled_kpts = cell.get_scaled_kpts(kpts)
    logger.debug3(cell, '    scaled_kpts kpts %s', scaled_kpts)
    if rcut is None:
        kmesh = np.asarray(cell.nimgs) * 2 + 1
    else:
        nimgs = cell.get_bounding_sphere(rcut)
        kmesh = nimgs * 2 + 1

    if precision is None:
        precision = max(1e-6, cell.precision * 1e2)
    for i in range(3):
        floats = scaled_kpts[:,i]
        uniq_floats_idx = np.unique((floats/precision+.5).astype(int), return_index=True)[1]
        uniq_floats = floats[uniq_floats_idx]
        fracs = [Fraction(x).limit_denominator(int(kmesh[i])) for x in uniq_floats]
        denominators = np.unique([x.denominator for x in fracs])
        common_denominator = reduce(np.lcm, denominators)
        fs = [(x * common_denominator).numerator for x in fracs]
        if cell.verbose >= logger.DEBUG3:
            logger.debug3(cell, 'dim=%d common_denominator %d  error %g',
                          i, common_denominator, abs(fs - np.rint(fs)).max())
            logger.debug3(cell, '    unique kpts %s', uniq_floats)
            logger.debug3(cell, '    frac kpts %s', fracs)
        if abs(uniq_floats - np.rint(fs)/common_denominator).max() < precision:
            kmesh[i] = common_denominator
        elif not bound_by_supmol:
            raise RuntimeError(f'Unable to find Monkhorst-Pack k-point mesh for {kpts}')
    return kmesh
