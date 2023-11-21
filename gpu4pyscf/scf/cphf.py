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
# Modified by: Xiaojie Wu <wxj6000@gmail.com>

'''
Restricted coupled pertubed Hartree-Fock solver
'''


import numpy
import cupy
from pyscf import lib
from gpu4pyscf.lib.cupy_helper import krylov
from gpu4pyscf.lib import logger

def solve(fvind, mo_energy, mo_occ, h1, s1=None,
          max_cycle=50, tol=1e-7, hermi=False, verbose=logger.WARN):
    '''
    Args:
        fvind : function
            Given density matrix, compute (ij|kl)D_{lk}*2 - (ij|kl)D_{jk}
    Kwargs:
        hermi : boolean
            Whether the matrix defined by fvind is Hermitian or not.
    '''

    if s1 is None:
        return solve_nos1(fvind, mo_energy, mo_occ, h1,
                          max_cycle, tol, hermi, verbose)
    else:
        return solve_withs1(fvind, mo_energy, mo_occ, h1, s1,
                            max_cycle, tol, hermi, verbose)
kernel = solve

# h1 shape is (:,nvir,nocc)
def solve_nos1(fvind, mo_energy, mo_occ, h1,
               max_cycle=20, tol=1e-9, hermi=False, verbose=logger.WARN):
    '''For field independent basis. First order overlap matrix is zero'''
    log = logger.new_logger(verbose=verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    e_a = mo_energy[mo_occ==0]
    e_i = mo_energy[mo_occ>0]
    I, A = cupy.meshgrid(e_i, e_a)
    e_ai = 1 / (A - I)#cupy.einsum('a,i->ai', e_a, e_i)
    mo1base = h1 * -e_ai

    def vind_vo(mo1):
        v = fvind(mo1.reshape(h1.shape)).reshape(h1.shape)
        v *= e_ai
        return v.ravel()
    mo1 = krylov(vind_vo, mo1base.ravel(),
                     tol=tol, max_cycle=max_cycle, hermi=hermi, verbose=log)
    log.timer('krylov solver in CPHF', *t0)
    return mo1.reshape(h1.shape), None

# h1 shape is (:,nocc+nvir,nocc)
def solve_withs1(fvind, mo_energy, mo_occ, h1, s1,
                 max_cycle=50, tol=1e-9, hermi=False, verbose=logger.WARN):
    '''For field dependent basis. First order overlap matrix is non-zero.
    The first order orbitals are set to
    C^1_{ij} = -1/2 S1
    e1 = h1 - s1*e0 + (e0_j-e0_i)*c1 + vhf[c1]
    Kwargs:
        hermi : boolean
            Whether the matrix defined by fvind is Hermitian or not.
    Returns:
        First order orbital coefficients (in MO basis) and first order orbital
        energy matrix
    '''
    log = logger.new_logger(verbose=verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    occidx = mo_occ > 0
    viridx = mo_occ == 0
    e_a = mo_energy[viridx]
    e_i = mo_energy[occidx]
    I, A = cupy.meshgrid(e_i, e_a)
    e_ai = 1 / (A - I)
    nvir, nocc = e_ai.shape
    nmo = nocc + nvir

    s1 = s1.reshape(-1,nmo,nocc)
    hs = mo1base = h1.reshape(-1,nmo,nocc) - s1*e_i
    mo_e1 = hs[:,occidx,:].copy()

    mo1base[:,viridx] *= -e_ai
    mo1base[:,occidx] = -s1[:,occidx] * .5

    def vind_vo(mo1):
        v = fvind(mo1.reshape(h1.shape)).reshape(-1,nmo,nocc)
        v[:,viridx,:] *= e_ai
        v[:,occidx,:] = 0
        return v.ravel()
    mo1 = krylov(vind_vo, mo1base.ravel(),
                     tol=tol, max_cycle=max_cycle, hermi=hermi, verbose=log)
    mo1 = mo1.reshape(mo1base.shape)
    log.timer('krylov solver in CPHF', *t0)

    v1mo = fvind(mo1.reshape(h1.shape)).reshape(-1,nmo,nocc)
    mo1[:,viridx] = mo1base[:,viridx] - v1mo[:,viridx]*e_ai

    # mo_e1 has the same symmetry as the first order Fock matrix (hermitian or
    # anti-hermitian). mo_e1 = v1mo - s1*lib.direct_sum('i+j->ij',e_i,e_i)
    It, I = cupy.meshgrid(e_i, e_i)
    mo_e1 += mo1[:,occidx] * (I - It)#(cupy.einsum('i,j->ij', e_i, e_i)
    mo_e1 += v1mo[:,occidx,:]

    if h1.ndim == 3:
        return mo1, mo_e1
    else:
        return mo1.reshape(h1.shape), mo_e1.reshape(nocc,nocc)
