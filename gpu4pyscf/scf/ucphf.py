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


'''
Unrestricted coupled pertubed Hartree-Fock solver
'''

import numpy
import cupy
from pyscf import lib
from gpu4pyscf.lib.cupy_helper import krylov
from gpu4pyscf.lib import logger

def solve(fvind, mo_energy, mo_occ, h1, s1=None,
          max_cycle=50, tol=1e-7, hermi=False, verbose=logger.WARN,
          level_shift=0):

    if s1 is None:
        return solve_nos1(fvind, mo_energy, mo_occ, h1,
                          max_cycle, tol, hermi, verbose, level_shift)
    else:
        return solve_withs1(fvind, mo_energy, mo_occ, h1, s1,
                            max_cycle, tol, hermi, verbose, level_shift)
kernel = solve

# h1 shape is (:,nvir,nocc)
def solve_nos1(fvind, mo_energy, mo_occ, h1,
               max_cycle=20, tol=1e-9, hermi=False, verbose=logger.WARN,
               level_shift=0):
    '''For field independent basis. First order overlap matrix is zero'''
    log = logger.new_logger(verbose=verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    viridxa = ~occidxa
    viridxb = ~occidxb
    nocca = int(cupy.count_nonzero(occidxa))
    noccb = int(cupy.count_nonzero(occidxb))
    nvira = int(mo_occ[0].size - nocca)
    nvirb = int(mo_occ[1].size - noccb)
    mo_ea, mo_eb = mo_energy
    e_ai = cupy.hstack(
        ((mo_ea[viridxa,None]+level_shift - mo_ea[occidxa]).ravel(),
         (mo_eb[viridxb,None]+level_shift - mo_eb[occidxb]).ravel()))
    e_ai = 1 / e_ai
    mo1base = cupy.hstack((h1[0].reshape(-1,nvira*nocca),
                            h1[1].reshape(-1,nvirb*noccb)))
    mo1base *= -e_ai
    nov = e_ai.size

    def vind_vo(mo1):
        nd = mo1.shape[0]
        v = fvind(mo1).reshape(nd, nov)
        if level_shift != 0:
            v -= mo1 * level_shift
        v *= e_ai
        return v.reshape(-1, nov)
    mo1 = krylov(vind_vo, mo1base.reshape(-1, nov),
                     tol=tol, max_cycle=max_cycle, hermi=hermi, verbose=log)
    log.timer('krylov solver in CPHF', *t0)

    mo1 = mo1.reshape(mo1base.shape)
    mo1_a = mo1[:,:nvira*nocca].reshape(-1,nvira,nocca)
    mo1_b = mo1[:,nvira*nocca:].reshape(-1,nvirb,noccb)
    if isinstance(h1[0], cupy.ndarray) and h1[0].ndim == 2:
        mo1 = (mo1_a[0], mo1_b[0])
    else:
        assert h1[0].ndim == 3
        mo1 = (mo1_a, mo1_b)
    return mo1, None

# h1 shape is (:,nocc+nvir,nocc)
def solve_withs1(fvind, mo_energy, mo_occ, h1, s1,
                 max_cycle=50, tol=1e-9, hermi=False, verbose=logger.WARN,
                 level_shift=0):
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

    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0

    viridxa = mo_occ[0] == 0
    viridxb = mo_occ[1] == 0

    nocca = cupy.sum(mo_occ[0] > 0).get()
    noccb = cupy.sum(mo_occ[1] > 0).get()
    nmoa, nmob = mo_occ[0].size, mo_occ[1].size

    mo_ea, mo_eb = mo_energy
    ea_a = mo_ea[mo_occ[0]==0]
    ei_a = mo_ea[mo_occ[0]>0]
    eai_a = 1 / (ea_a[:,None] + level_shift - ei_a)

    ea_b = mo_eb[mo_occ[1]==0]
    ei_b = mo_eb[mo_occ[1]>0]
    eai_b = 1 / (ea_b[:,None] + level_shift - ei_b)

    s1_a = s1[0].reshape(-1,nmoa,nocca)
    nset = s1_a.shape[0]
    s1_b = s1[1].reshape(nset,nmob,noccb)
    hs_a = h1[0].reshape(nset,nmoa,nocca) - s1_a * ei_a
    hs_b = h1[1].reshape(nset,nmob,noccb) - s1_b * ei_b

    mo1base_a = hs_a.copy()
    mo1base_b = hs_b.copy()
    mo1base_a[:,viridxa] *= -eai_a
    mo1base_b[:,viridxb] *= -eai_b
    mo1base_a[:,occidxa] = -s1_a[:,occidxa] * .5
    mo1base_b[:,occidxb] = -s1_b[:,occidxb] * .5
    mo1base = cupy.hstack((mo1base_a.reshape(nset,-1), mo1base_b.reshape(nset,-1)))
    
    def vind_vo(mo1):
        mo1 = mo1.reshape(-1,nmoa*nocca+nmob*noccb)
        v = fvind(mo1).reshape(-1,nmoa*nocca+nmob*noccb)
        if level_shift != 0:
            v -= mo1 * level_shift
        v1a = v[:,:nmoa*nocca].reshape(-1,nmoa,nocca)
        v1b = v[:,nmoa*nocca:].reshape(-1,nmob,noccb)
        v1a[:,viridxa] *= eai_a
        v1b[:,viridxb] *= eai_b
        v1a[:,occidxa] = 0
        v1b[:,occidxb] = 0
        return v.reshape(-1,nmoa*nocca+nmob*noccb)
    mo1 = krylov(vind_vo, mo1base.reshape(-1,nmoa*nocca+nmob*noccb),
                     tol=tol, max_cycle=max_cycle, hermi=hermi, verbose=log)

    mo1 = mo1.reshape(mo1base.shape)
    mo1_a = mo1[:,:nmoa*nocca].reshape(nset,nmoa,nocca)
    mo1_b = mo1[:,nmoa*nocca:].reshape(nset,nmob,noccb)
    mo1_a[:,occidxa] = mo1base_a[:,occidxa]
    mo1_b[:,occidxb] = mo1base_b[:,occidxb]
    log.timer('krylov solver in CPHF', *t0)

    v1mo = fvind(mo1).reshape(mo1base.shape)
    hs_a += v1mo[:,:nmoa*nocca].reshape(nset,nmoa,nocca)
    hs_b += v1mo[:,nmoa*nocca:].reshape(nset,nmob,noccb)
    mo1_a[:,viridxa] = hs_a[:,viridxa] / (ei_a - ea_a[:,None])
    mo1_b[:,viridxb] = hs_b[:,viridxb] / (ei_b - ea_b[:,None])

    mo_e1_a = hs_a[:,occidxa]
    mo_e1_b = hs_b[:,occidxb]
    mo_e1_a += mo1_a[:,occidxa] * (ei_a[:,None] - ei_a)
    mo_e1_b += mo1_b[:,occidxb] * (ei_b[:,None] - ei_b)

    if isinstance(h1[0], numpy.ndarray) and h1[0].ndim == 2:
        mo1_a, mo1_b = mo1_a[0], mo1_b[0]
        mo_e1_a, mo_e1_b = mo_e1_a[0], mo_e1_b[0]
    return (mo1_a, mo1_b), (mo_e1_a, mo_e1_b)
