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

import json
import numpy as np
from pyscf.pbc.gto.cell import make_kpts
from pyscf.pbc.lib.kpts_helper import is_zero
from gpu4pyscf import __config__

WRAP_AROUND = getattr(__config__, 'pbc_gto_cell_make_kpts_wrap_around', False)
WITH_GAMMA = getattr(__config__, 'pbc_gto_cell_make_kpts_with_gamma', True)


class KPointSampling:
    '''K-point coordinates with optional Monkhorst-Pack mesh metadata.

    Unlike pyscf.pbc.lib.kpts.KPoints, this class does not perform symmetry
    reduction. Coordinates are NumPy arrays; use kpts or to_cpu() when an
    array is required.

    Args:
        lattice_vectors : array_like, shape (3, 3)
            Lattice vectors as rows, in Bohr.

    Attributes:
        lattice_vectors : numpy.ndarray, shape (3, 3)
            Lattice vectors as rows, in Bohr. A snapshot of the input cell's
            lattice, updated explicitly by reset().
        kpts : numpy.ndarray, shape (nkpts, 3)
            Cartesian k-points in Bohr^-1, in input order.
        kpts_scaled : numpy.ndarray, shape (nkpts, 3)
            Fractional reciprocal coordinates.
        kmesh : numpy.ndarray, shape (3,), or None
            Sampling dimensions passed to cell.make_kpts, not BvK dimensions.
            None for explicit coordinates.
        wrap_around, with_gamma_point : bool or None
            Options passed to the CPU point generator. Wrapping precedes
            the addition of scaled_center. None for explicit coordinates.
        scaled_center : numpy.ndarray, shape (3,), or None
            Requested shift. None is preserved because an explicit center
            changes grid generation when with_gamma_point is False.
            Also None when the generation parameters are unknown.
    '''

    def __init__(self, lattice_vectors):
        self.lattice_vectors = np.array(lattice_vectors, dtype=np.float64, copy=True)
        self.kpts_scaled = None
        self.kmesh = None
        self.wrap_around = None
        self.with_gamma_point = None
        self.scaled_center = None

    @classmethod
    def from_mesh(cls, cell, kmesh, wrap_around=WRAP_AROUND,
                  with_gamma_point=WITH_GAMMA, scaled_center=None,
                  space_group_symmetry=False, time_reversal_symmetry=False):
        '''Generate a full MP mesh and retain its generation parameters.

        Options follow pyscf.pbc.gto.cell.make_kpts. The returned object's
        kmesh stores the sampling dimensions, including for shifted meshes.
        Symmetry reduction is unsupported.
        '''
        if space_group_symmetry or time_reversal_symmetry:
            raise NotImplementedError('K-point symmetry reduction is not supported')
        # cell may be an instance of GPU Cell, which overrides make_kpts method.
        kpts = make_kpts(cell, kmesh, wrap_around, with_gamma_point, scaled_center)
        self = cls.from_kpts(cell, kpts)
        self.kmesh = np.asarray(kmesh)
        self.with_gamma_point = with_gamma_point
        self.wrap_around = wrap_around
        if scaled_center is not None:
            self.scaled_center = np.asarray(scaled_center)
        return self

    @classmethod
    def gamma_point(cls, cell):
        '''Create a single Gamma-point sampling with kmesh (1, 1, 1).'''
        kpts = cls(cell.lattice_vectors())
        kpts.kpts_scaled = np.zeros((1, 3))
        kpts.kmesh = np.asarray([1, 1, 1])
        kpts.with_gamma_point = True
        kpts.wrap_around = True
        return kpts

    @classmethod
    def from_kpts(cls, cell, kpts):
        if isinstance(kpts, KPointSampling):
            return kpts
        self = cls(cell.lattice_vectors())
        self.kpts_scaled = cell.get_scaled_kpts(np.asarray(kpts))
        return self

    @property
    def kpts(self):
        return self.kpts_scaled @ self.reciprocal_vectors()

    def reciprocal_vectors(self):
        '''Reciprocal lattice vectors as rows, in Bohr^-1.'''
        return 2 * np.pi * np.linalg.inv(self.lattice_vectors.T)

    @property
    def nkpts(self):
        return len(self.kpts_scaled)

    def __len__(self):
        return self.nkpts

    def is_gamma_only(self):
        '''Whether all Cartesian k-points are zero within PySCF's tolerance.'''
        return is_zero(self.kpts_scaled)

    def __array__(self, dtype=None, copy=None):
        '''Return Cartesian coordinates as a NumPy array.'''
        return np.asarray(self.kpts, dtype=dtype, copy=copy)

    def __str__(self):
        s = ['\nScaled k-points']
        s.extend(['%d:  %11.8f, %11.8f, %11.8f' % (k, *ks)
                  for k, ks in enumerate(self.kpts_scaled)])
        return '\n'.join(s)

    def reset(self, cell):
        '''Update the lattice while preserving fractional k-points.'''
        self.lattice_vectors = cell.lattice_vectors()
        return self

    def dumps(self):
        state = {key: getattr(self, key) for key in (
            'lattice_vectors', 'kpts_scaled', 'kmesh', 'wrap_around',
            'with_gamma_point', 'scaled_center')}
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                state[key] = value.tolist()
        return json.dumps(state)

    @classmethod
    def loads(cls, text):
        state = json.loads(text)
        self = cls(state['lattice_vectors'])
        for key, dtype in (('kpts_scaled', float), ('kmesh', int),
                           ('scaled_center', float)):
            value = state[key]
            if value is not None:
                value = np.asarray(value, dtype=dtype)
            setattr(self, key, value)
        self.wrap_around = state['wrap_around']
        self.with_gamma_point = state['with_gamma_point']
        return self

    def to_cpu(self):
        '''Return the full Cartesian k-point array for CPU calculations.'''
        return self.kpts
