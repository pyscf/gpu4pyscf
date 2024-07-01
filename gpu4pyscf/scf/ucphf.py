# Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


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
    nocca = cupy.sum(mo_occ[0] > 0)
    noccb = cupy.sum(mo_occ[1] > 0)

    nvira = mo_occ[0].size - nocca
    nvirb = mo_occ[1].size - noccb

    mo_ea, mo_eb = mo_energy
    e_a = mo_ea[mo_occ[0]==0]
    e_i = mo_ea[mo_occ[0]>0]
    ea_ai = 1 / (e_a[:,None] + level_shift - e_i)

    e_a = mo_eb[mo_occ[1]==0]
    e_i = mo_eb[mo_occ[1]>0]
    eb_ai = 1 / (e_a[:,None] + level_shift - e_i)

    e_ai = cupy.hstack([ea_ai.ravel(),eb_ai.ravel()])

    mo1base = cupy.hstack((h1[0].reshape(-1, nvira*nocca),
                            h1[1].reshape(-1,nvirb*noccb)))
    mo1base *= -e_ai

    def vind_vo(mo1):
        v = fvind(mo1.reshape(h1.shape)).reshape(h1.shape)
        if level_shift != 0:
            v -= mo1 * level_shift
        v *= e_ai
        return v.ravel()
    mo1 = krylov(vind_vo, mo1base.ravel(),
                     tol=tol, max_cycle=max_cycle, hermi=hermi, verbose=log)
    log.timer('krylov solver in UCPHF', *t0)
    mo1 = mo1.reshape(mo1base.shape)
    mo1_a = mo1[:,:nvira*nocca].reshape(-1,nvira,nocca)
    mo1_b = mo1[:,nvira*nocca:].reshape(-1,nvirb,noccb)
    mo1 = (mo1_a, mo1_b)
    return mo1.reshape(h1.shape), None

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
