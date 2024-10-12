#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
#  modified by: Xiaojie Wu <wxj6000@gmail.com>
#


'''
Non-relativistic UHF analytical Hessian
'''

from functools import reduce
import ctypes
import numpy
import numpy as np
import cupy
from pyscf.hessian import rhf as rhf_hess_cpu
from pyscf.hessian import uhf as uhf_hess_cpu
from pyscf import lib
from pyscf import gto
from pyscf.scf import _vhf

# import _response_functions to load gen_response methods in SCF class
from gpu4pyscf.scf import _response_functions  # noqa
# import pyscf.grad.rhf to activate nuc_grad_method method
from gpu4pyscf.scf import ucphf
from gpu4pyscf.gto.mole import sort_atoms
from gpu4pyscf.lib.cupy_helper import contract, tag_array, print_mem_info, get_avail_mem
from gpu4pyscf.lib import logger
from gpu4pyscf.df import int3c2e
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.hessian.rhf import HessianBase, _partial_hess_ejk

GB = 1024*1024*1024
ALIGNED = 4

def hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
              mo1=None, mo_e1=None, h1mo=None,
              atmlst=None, max_memory=4000, verbose=None):
    ''' Different from PySF, using h1mo instead of h1ao for saving memory
    '''
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (logger.process_clock(), logger.perf_counter())

    mol = hessobj.mol
    mf = hessobj.base
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)

    mo_energy = cupy.asarray(mo_energy)
    mo_occ = cupy.asarray(mo_occ)
    mo_coeff = cupy.asarray(mo_coeff)
    de2 = hessobj.partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst,
                                    max_memory, log)
    t1 = log.timer_debug1('hess elec', *t1)
    if h1mo is None:
        h1mo = hessobj.make_h1(mo_coeff, mo_occ, None, atmlst, log)
        t1 = log.timer_debug1('making H1', *t1)
    if mo1 is None or mo_e1 is None:
        mo1, mo_e1 = hessobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1mo,
                                       None, atmlst, max_memory, log)
        t1 = log.timer_debug1('solving MO1', *t1)
    mo1a, mo1b = mo1
    mo_e1a, mo_e1b = mo_e1
    h1aoa, h1aob = h1mo

    nao, _ = mo_coeff[0].shape
    mocca = cupy.array(mo_coeff[0][:,mo_occ[0]>0])
    moccb = cupy.array(mo_coeff[1][:,mo_occ[1]>0])
    mo_energy = cupy.array(mo_energy)
    mo_ea = mo_energy[0][mo_occ[0]>0]
    mo_eb = mo_energy[1][mo_occ[1]>0]

    s1a = -mol.intor('int1e_ipovlp', comp=3)
    s1a = cupy.asarray(s1a)
    aoslices = mol.aoslice_by_atom()
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        s1ao = cupy.zeros((3,nao,nao))
        s1ao[:,p0:p1] += s1a[:,p0:p1]
        s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)

        tmp = contract('xpq,pi->xiq', s1ao, mocca)
        s1ooa = contract('xiq,qj->xij', tmp, mocca)

        tmp = contract('xpq,pi->xiq', s1ao, moccb)
        s1oob = contract('xiq,qj->xij', tmp, moccb)

        #s1oo = cupy.einsum('xpq,pi,qj->xij', s1ao, mocc, mocc)
        s1moa = contract('xij,ip->xpj', s1ao, mo_coeff[0])
        s1mob = contract('xij,ip->xpj', s1ao, mo_coeff[1])
        for j0 in range(i0+1):
            ja = atmlst[j0]
            q0, q1 = aoslices[ja][2:]
# *2 for double occupancy, *2 for +c.c.
            #dm1 = cupy.einsum('ypi,qi->ypq', mo1[ja], mocc)
            #de2_gpu[i0,j0] += cupy.einsum('xpq,ypq->xy', h1ao[ia], dm1) * 4
            de2[i0,j0] += contract('xpi,ypi->xy', h1aoa[ia], mo1a[ja]) * 2
            de2[i0,j0] += contract('xpi,ypi->xy', h1aob[ia], mo1b[ja]) * 2
            dm1a = contract('ypi,qi->ypq', mo1a[ja], mocca*mo_ea)
            dm1b = contract('ypi,qi->ypq', mo1b[ja], moccb*mo_eb)
            de2[i0,j0] -= contract('xpq,ypq->xy', s1moa, dm1a) * 2
            de2[i0,j0] -= contract('xpq,ypq->xy', s1mob, dm1b) * 2
            de2[i0,j0] -= contract('xpq,ypq->xy', s1ooa, mo_e1a[ja])
            de2[i0,j0] -= contract('xpq,ypq->xy', s1oob, mo_e1b[ja])
        for j0 in range(i0):
            de2[j0,i0] = de2[i0,j0].T

    log.timer('UHF hessian', *time0)

    return de2

def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    '''Partial derivative
    '''
    e1, ej, ek = _partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ,
                                   atmlst, max_memory, verbose, True)
    return e1 + ej - ek  # (A,B,dR_A,dR_B)

def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    mol = hessobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    mo_a, mo_b = mo_coeff
    mocca = mo_a[:,mo_occ[0]>0]
    moccb = mo_b[:,mo_occ[1]>0]
    nmoa = mo_a.shape[1]
    nmob = mo_b.shape[1]
    nocca = mocca.shape[1]
    noccb = moccb.shape[1]
    dm0a = mocca.dot(mocca.T)
    dm0b = moccb.dot(moccb.T)
    vja, vka = rhf_grad.get_jk(mol, dm0a)
    vjb, vkb = rhf_grad.get_jk(mol, dm0b)
    vj = vja + vjb
    vhfa = vj - vka
    vhfb = vj - vkb

    hcore_deriv = hessobj.base.nuc_grad_method().hcore_generator(mol)
    natm = mol.natm
    for ia in range(natm):
        hcore1 = hcore_deriv(ia)
        vhfa[ia*3:ia*3+3] += hcore1
        vhfb[ia*3:ia*3+3] += hcore1

    h1aoa = np.empty((natm*3,nmoa,nocca))
    h1aob = np.empty((natm*3,nmob,noccb))
    for i, v in enumerate(vhfa):
        h1aoa[i] = mo_a.T.dot(v.dot(mocca)).get()
    for i, v in enumerate(vhfb):
        h1aob[i] = mo_b.T.dot(v.dot(moccb)).get()
    return h1aoa.reshape(natm,3,nmoa,nocca), h1aob.reshape(natm,3,nmob,noccb)

def get_hcore(mol):
    '''Part of the second derivatives of core Hamiltonian'''
    h1aa = mol.intor('int1e_ipipkin', comp=9)
    h1ab = mol.intor('int1e_ipkinip', comp=9)
    if mol._pseudo:
        NotImplementedError('Nuclear hessian for GTH PP')
    else:
        h1aa+= mol.intor('int1e_ipipnuc', comp=9)
        h1ab+= mol.intor('int1e_ipnucip', comp=9)
    if mol.has_ecp():
        h1aa += mol.intor('ECPscalar_ipipnuc', comp=9)
        h1ab += mol.intor('ECPscalar_ipnucip', comp=9)
    nao = h1aa.shape[-1]
    return h1aa.reshape(3,3,nao,nao), h1ab.reshape(3,3,nao,nao)

def get_ovlp(mol):
    s1a =-mol.intor('int1e_ipovlp', comp=3)
    nao = s1a.shape[-1]
    s1aa = mol.intor('int1e_ipipovlp', comp=9).reshape(3,3,nao,nao)
    s1ab = mol.intor('int1e_ipovlpip', comp=9).reshape(3,3,nao,nao)
    return s1aa, s1ab, s1a

def solve_mo1(mf, mo_energy, mo_coeff, mo_occ, h1mo,
              fx=None, atmlst=None, max_memory=4000, verbose=None,
              max_cycle=50, level_shift=0):
    '''Solve the first order equation
    Kwargs:
        fx : function(dm_mo) => v1_mo
            A function to generate the induced potential.
            See also the function gen_vind.
    '''
    mol = mf.mol
    log = logger.new_logger(mf, verbose)
    if atmlst is None: atmlst = range(mol.natm)

    nao, nmo = mo_coeff[0].shape
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    nocca = mocca.shape[1]
    noccb = moccb.shape[1]
    if fx is None:
        fx = gen_vind(mf, mo_coeff, mo_occ)
    s1a = -mol.intor('int1e_ipovlp', comp=3)
    s1a = cupy.asarray(s1a)

    def _ao2mo(mat, mo, mocc):
        tmp = contract('xij,jo->xio', mat, mocc)
        return contract('xik,ip->xpk', tmp, mo)
    cupy.get_default_memory_pool().free_all_blocks()

    avail_mem = get_avail_mem()
    blksize = int(avail_mem*0.4) // (8*3*nao*nao*4) // ALIGNED * ALIGNED
    blksize = min(8, blksize)
    log.debug(f'GPU memory {avail_mem/GB:.1f} GB available')
    log.debug(f'{blksize} atoms in each block CPHF equation')

    # sort atoms to improve the convergence
    sorted_idx = sort_atoms(mol)
    atom_groups = []
    for p0,p1 in lib.prange(0,mol.natm,blksize):
        blk = sorted_idx[p0:p1]
        atom_groups.append(blk)

    mo1sa = [None] * mol.natm
    mo1sb = [None] * mol.natm
    e1sa = [None] * mol.natm
    e1sb = [None] * mol.natm
    aoslices = mol.aoslice_by_atom()
    for group in atom_groups:
        s1voa = []
        s1vob = []
        h1voa = []
        h1vob = []
        for ia in group:
            shl0, shl1, p0, p1 = aoslices[ia]
            s1ao = cupy.zeros((3,nao,nao))
            s1ao[:,p0:p1] += s1a[:,p0:p1]
            s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
            s1voa.append(_ao2mo(s1ao, mo_coeff[0], mocca))
            s1vob.append(_ao2mo(s1ao, mo_coeff[1], moccb))
            h1voa.append(h1mo[0][ia])
            h1vob.append(h1mo[1][ia])

        log.info(f'Solving CPHF equation for atoms {len(group)}/{mol.natm}')
        h1vo = (cupy.vstack(h1voa), cupy.vstack(h1vob))
        s1vo = (cupy.vstack(s1voa), cupy.vstack(s1vob))
        tol = mf.conv_tol_cpscf
        mo1, e1 = ucphf.solve(fx, mo_energy, mo_occ, h1vo, s1vo,
                              max_cycle=max_cycle, level_shift=level_shift, tol=tol, verbose=verbose)

        mo1a = mo1[0].reshape(-1,3,nao,nocca)
        mo1b = mo1[1].reshape(-1,3,nao,noccb)
        e1a = e1[0].reshape(-1,3,nocca,nocca)
        e1b = e1[1].reshape(-1,3,noccb,noccb)
        for k, ia in enumerate(group):
            mo1sa[ia] = mo1a[k]
            mo1sb[ia] = mo1b[k]
            e1sa[ia] = e1a[k].reshape(3,nocca,nocca)
            e1sb[ia] = e1b[k].reshape(3,noccb,noccb)
        mo1 = e1 = None
    return (mo1sa, mo1sb), (e1sa, e1sb)

def gen_vind(mf, mo_coeff, mo_occ):
    # Move data to GPU
    mo_coeff = cupy.asarray(mo_coeff)
    mo_occ = cupy.asarray(mo_occ)
    nao, nmoa = mo_coeff[0].shape
    nao, nmob = mo_coeff[1].shape
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    nocca = mocca.shape[1]
    noccb = moccb.shape[1]
    vresp = mf.gen_response(mo_coeff, mo_occ, hermi=1)

    def fx(mo1):
        mo1 = cupy.asarray(mo1)
        mo1 = mo1.reshape(-1,nmoa*nocca+nmob*noccb)
        nset = len(mo1)

        x = mo1[:,:nmoa*nocca].reshape(nset,nmoa,nocca)
        mo1_moa = contract('npo,ip->nio', x, mo_coeff[0])
        dma = contract('nio,jo->nij', mo1_moa, mocca)

        x = mo1[:,nmoa*nocca:].reshape(nset,nmob,noccb)
        mo1_mob = contract('npo,ip->nio', x, mo_coeff[1])
        dmb = contract('nio,jo->nij', mo1_mob, moccb)

        dm1 = cupy.empty([2,nset,nao,nao])
        dm1[0] = dma + dma.transpose(0,2,1)
        dm1[1] = dmb + dmb.transpose(0,2,1)

        dm1 = tag_array(dm1, mo1=[mo1_moa,mo1_mob], occ_coeff=[mocca,moccb], mo_occ=mo_occ)
        v1 = vresp(dm1)
        v1vo = cupy.empty_like(mo1)
        tmp = contract('nij,jo->nio', v1[0], mocca)
        v1vo[:,:nmoa*nocca] = contract('nio,ip->npo', tmp, mo_coeff[0]).reshape(nset,-1)

        tmp = contract('nij,jo->nio', v1[1], moccb)
        v1vo[:,nmoa*nocca:] = contract('nio,ip->npo', tmp, mo_coeff[1]).reshape(nset,-1)
        return v1vo
    return fx


def gen_hop(hobj, mo_energy=None, mo_coeff=None, mo_occ=None, verbose=None):
    log = logger.new_logger(hobj, verbose)
    mol = hobj.mol
    mf = hobj.base

    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff

    natm = mol.natm
    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]

    atmlst = range(natm)
    max_memory = max(2000, hobj.max_memory - lib.current_memory()[0])
    de2 = hobj.partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst,
                                 max_memory, log)
    de2 += hobj.hess_nuc()

    h1ao_cache = hobj.make_h1(mo_coeff, mo_occ, None, atmlst, log)

    aoslices = mol.aoslice_by_atom()
    s1a = -mol.intor('int1e_ipovlp', comp=3)

    fvind = gen_vind(mf, mo_coeff, mo_occ)
    def h_op(x):
        x = x.reshape(natm,3)
        hx = numpy.einsum('abxy,ax->by', de2, x)
        h1ao = 0
        s1ao = 0
        for ia in range(natm):
            shl0, shl1, p0, p1 = aoslices[ia]
            h1ao += numpy.einsum('x,xij->ij', x[ia], h1ao_cache[ia])
            s1ao_i = numpy.zeros((3,nao,nao))
            s1ao_i[:,p0:p1] += s1a[:,p0:p1]
            s1ao_i[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
            s1ao += numpy.einsum('x,xij->ij', x[ia], s1ao_i)

        s1vo = reduce(numpy.dot, (mo_coeff.T, s1ao, mocc))
        h1vo = reduce(numpy.dot, (mo_coeff.T, h1ao, mocc))
        mo1, mo_e1 = ucphf.solve(fvind, mo_energy, mo_occ, h1vo, s1vo)
        mo1 = numpy.dot(mo_coeff, mo1)
        mo_e1 = mo_e1.reshape(nocc,nocc)
        dm1 = numpy.einsum('pi,qi->pq', mo1, mocc)
        dme1 = numpy.einsum('pi,qi,i->pq', mo1, mocc, mo_energy[mo_occ>0])
        dme1 = dme1 + dme1.T + reduce(numpy.dot, (mocc, mo_e1.T, mocc.T))

        for ja in range(natm):
            q0, q1 = aoslices[ja][2:]
            hx[ja] += numpy.einsum('xpq,pq->x', h1ao_cache[ja], dm1) * 4
            hx[ja] -= numpy.einsum('xpq,pq->x', s1a[:,q0:q1], dme1[q0:q1]) * 2
            hx[ja] -= numpy.einsum('xpq,qp->x', s1a[:,q0:q1], dme1[:,q0:q1]) * 2
        return hx.ravel()

    hdiag = numpy.einsum('aaxx->ax', de2).ravel()
    return h_op, hdiag


class Hessian(HessianBase):
    '''Non-relativistic unrestricted Hartree-Fock hessian'''

    partial_hess_elec = partial_hess_elec
    hess_elec = hess_elec
    make_h1 = make_h1
    kernel = NotImplemented
    hess = NotImplemented

    def solve_mo1(self, mo_energy, mo_coeff, mo_occ, h1mo,
                  fx=None, atmlst=None, max_memory=4000, verbose=None):
        return solve_mo1(self.base, mo_energy, mo_coeff, mo_occ, h1mo,
                         fx, atmlst, max_memory, verbose,
                         max_cycle=self.max_cycle, level_shift=self.level_shift)

    gen_hop = gen_hop

# Inject to UHF class
from gpu4pyscf import scf
scf.uhf.UHF.Hessian = lib.class_as_method(Hessian)
