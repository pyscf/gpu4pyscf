#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
# Modified by Xiaojie Wu <wxj6000@gmail.com>

'''radii grids'''

import numpy
import cupy
import pyscf
from pyscf.data import radii
from pyscf.data.elements import charge as elements_proton

BRAGG_RADII = radii.BRAGG
COVALENT_RADII = radii.COVALENT
SG1RADII = pyscf.dft.radi.SG1RADII

gauss_chebyshev = pyscf.dft.radi.gauss_chebyshev
treutler = pyscf.dft.radi.treutler

def treutler_atomic_radii_adjust(mol, atomic_radii):
    '''Treutler atomic radii adjust function: [JCP 102, 346 (1995); DOI:10.1063/1.469408]'''
# JCP 102, 346 (1995)
# i > j
# fac(i,j) = \frac{1}{4} ( \frac{ra(j)}{ra(i)} - \frac{ra(i)}{ra(j)}
# fac(j,i) = -fac(i,j)
    charges = [elements_proton(x) for x in mol.elements]
    rad = cupy.sqrt(atomic_radii[charges]) + 1e-200
    rr = rad.reshape(-1,1) * (1./rad)
    a = .25 * (rr.T - rr)
    a[a<-.5] = -.5
    a[a>0.5] = 0.5
    #:return lambda i,j,g: g + a[i,j]*(1-g**2)
    def fadjust(g):
        g1 = g**2
        g1 -= 1.
        g1 *= -a[:,:,None]
        g1 += g
        return g1
    return fadjust

def get_treutler_fac(mol, atomic_radii):
    '''
    # fac(i,j) = \frac{1}{4} ( \frac{ra(j)}{ra(i)} - \frac{ra(i)}{ra(j)}
    # fac(j,i) = -fac(i,j)
    '''
    charges = [elements_proton(x) for x in mol.elements]
    #atomic_radii = cupy.asarray(atomic_radii[charges])
    rad = numpy.sqrt(atomic_radii[charges]) + 1e-200
    rr = rad.reshape(-1,1) * (1./rad)
    a = .25 * (rr.T - rr)
    a[a<-.5] = -.5
    a[a>0.5] = 0.5
    return cupy.asarray(a)

def get_becke_fac(mol, atomic_radii):
    charges = [elements_proton(x) for x in mol.elements]
    atomic_radii = numpy.asarray(atomic_radii[charges])
    rad = atomic_radii[charges] + 1e-200
    rr = rad.reshape(-1,1) * (1./rad)
    a = .25 * (rr.T - rr)
    a[a<-.5] = -.5
    a[a>0.5] = 0.5
    return cupy.asarray(a)
