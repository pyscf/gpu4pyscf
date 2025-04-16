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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#  modified by: Xiaojie Wu <wxj6000@gmail.com>
#


'''
Non-relativistic UHF analytical Hessian
'''

import numpy as np
import cupy
import cupy as cp
from pyscf import lib
from pyscf.scf import ucphf
from gpu4pyscf.gto.ecp import get_ecp_ip
from gpu4pyscf.lib.cupy_helper import (contract, transpose_sum, get_avail_mem,
                                       krylov, tag_array)
from gpu4pyscf.lib import logger
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.hessian import rhf as rhf_hess_gpu
from gpu4pyscf.hessian import jk

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
    if atmlst is not None:
        assert len(atmlst) == mol.natm

    mo_energy = cupy.asarray(mo_energy)
    mo_occ = cupy.asarray(mo_occ)
    mo_coeff = cupy.asarray(mo_coeff)
    de2 = hessobj.partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst,
                                    max_memory, log)
    t1 = log.timer_debug1('hess elec', *t1)
    if h1mo is None:
        h1mo = hessobj.make_h1(mo_coeff, mo_occ, None, atmlst, log)
        if h1mo[0].size * 8 * 10 > get_avail_mem():
            # Reduce GPU memory footprint
            h1mo = (h1mo[0].get(), h1mo[1].get())
        t1 = log.timer_debug1('making H1', *t1)
    if mo1 is None or mo_e1 is None:
        fx = hessobj.gen_vind(mo_coeff, mo_occ)
        mo1, mo_e1 = hessobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1mo,
                                       fx, atmlst, max_memory, log)
        t1 = log.timer_debug1('solving MO1', *t1)

    mo1a = cupy.asarray(mo1[0])
    mo1b = cupy.asarray(mo1[1])
    de2 += contract('kxpi,lypi->klxy', cupy.asarray(h1mo[0]), mo1a) * 2
    de2 += contract('kxpi,lypi->klxy', cupy.asarray(h1mo[1]), mo1b) * 2
    mo1a = contract('kxai,pa->kxpi', mo1a, mo_coeff[0])
    mo1b = contract('kxai,pa->kxpi', mo1b, mo_coeff[1])

    mo_e1a = cupy.asarray(mo_e1[0])
    mo_e1b = cupy.asarray(mo_e1[1])

    nao, _ = mo_coeff[0].shape
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    mo_ea = mo_energy[0][mo_occ[0]>0]
    mo_eb = mo_energy[1][mo_occ[1]>0]
    mocca_e = mocca * mo_ea
    moccb_e = moccb * mo_eb
    s1a = -mol.intor('int1e_ipovlp', comp=3)
    s1a = cupy.asarray(s1a)

    aoslices = mol.aoslice_by_atom()
    for i0, (p0, p1) in enumerate(aoslices[:,2:]):
        s1ao = cupy.zeros((3,nao,nao))
        s1ao[:,p0:p1] += s1a[:,p0:p1]
        s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)

        tmp = contract('xpq,pi->xiq', s1ao, mocca)
        s1oo = contract('xiq,qj->xij', tmp, mocca)
        de2[i0] -= contract('xij,kyij->kxy', s1oo, mo_e1a)

        tmp = contract('xpq,pi->xiq', s1ao, moccb)
        s1oo = contract('xiq,qj->xij', tmp, moccb)
        de2[i0] -= contract('xij,kyij->kxy', s1oo, mo_e1b)

        s1mo = contract('xpq,qi->xpi', s1ao, mocca_e)
        de2[i0] -= contract('xpi,kypi->kxy', s1mo, mo1a) * 2

        s1mo = contract('xpq,qi->xpi', s1ao, moccb_e)
        de2[i0] -= contract('xpi,kypi->kxy', s1mo, mo1b) * 2

    de2 = de2 + de2.transpose(1,0,3,2)
    de2 *= .5
    log.timer('UHF hessian', *time0)
    return de2

def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    '''Partial derivative
    '''
    e1, ejk = _partial_hess_ejk(
        hessobj, mo_energy, mo_coeff, mo_occ, atmlst, max_memory, verbose, True)
    return e1 + ejk  # (A,B,dR_A,dR_B)

def _partial_hess_ejk(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None,
                      j_factor=1., k_factor=1.):
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (logger.process_clock(), logger.perf_counter())

    mol = hessobj.mol
    mf = hessobj.base
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    assert atmlst is None
    atmlst = range(mol.natm)

    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    dm0a = mocca.dot(mocca.T)
    dm0b = moccb.dot(moccb.T)
    dm0 = cp.asarray((dm0a, dm0b))
    vhfopt = mf._opt_gpu.get(None, None)
    ejk = rhf_hess_gpu._partial_ejk_ip2(mol, dm0, vhfopt, j_factor, k_factor,
                                        verbose=log)
    t1 = log.timer_debug1('hessian of 2e part', *t1)

    # Energy weighted density matrix
    mo_ea = mo_energy[0][mo_occ[0]>0]
    mo_eb = mo_energy[1][mo_occ[1]>0]
    dme0 = (mocca*mo_ea).dot(mocca.T)
    dme0+= (moccb*mo_eb).dot(moccb.T)
    de_hcore = rhf_hess_gpu._e_hcore_generator(hessobj, dm0a+dm0b)
    s1aa, s1ab, s1a = rhf_hess_gpu.get_ovlp(mol)

    aoslices = mol.aoslice_by_atom()
    e1 = cupy.zeros((mol.natm,mol.natm,3,3))
    for i0, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia][2:]
        e1[i0,i0] -= contract('xypq,pq->xy', s1aa[:,:,p0:p1], dme0[p0:p1])*2

        for j0, ja in enumerate(atmlst[:i0+1]):
            q0, q1 = aoslices[ja][2:]
            e1[i0,j0] -= contract('xypq,pq->xy', s1ab[:,:,p0:p1,q0:q1], dme0[p0:p1,q0:q1])*2
            e1[i0,j0] += de_hcore(ia, ja)

        for j0 in range(i0):
            e1[j0,i0] = e1[i0,j0].T

    log.timer('UHF partial hessian', *time0)
    return e1, ejk

def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    assert atmlst is None
    mol = hessobj.mol
    natm = mol.natm

    mo_a, mo_b = mo_coeff
    mocca = mo_a[:,mo_occ[0]>0]
    moccb = mo_b[:,mo_occ[1]>0]
    dm0a = mocca.dot(mocca.T)
    dm0b = moccb.dot(moccb.T)
    grad_obj = hessobj.base.Gradients()
    h1moa = rhf_grad.get_grad_hcore(grad_obj, mo_a, mo_occ[0])
    h1mob = rhf_grad.get_grad_hcore(grad_obj, mo_b, mo_occ[1])

    # Estimate the size of intermediate variables
    # dm, vj, and vk in [natm,3,nao_cart,nao_cart]
    nao_cart = mol.nao_cart()
    avail_mem = get_avail_mem()
    slice_size = int(avail_mem*0.5) // (8*3*nao_cart*nao_cart*6)
    for atoms_slice in lib.prange(0, natm, slice_size):
        vja, vka = rhf_hess_gpu._get_jk_ip1(mol, dm0a, atoms_slice=atoms_slice, verbose=verbose)
        vjb, vkb = rhf_hess_gpu._get_jk_ip1(mol, dm0b, atoms_slice=atoms_slice, verbose=verbose)
        #:vhfa = vja+vjb - vka
        #:vhfb = vja+vjb - vkb
        vhfa = vka
        vhfb = vkb
        vhfa *= -1
        vhfb *= -1
        vj = vja + vjb
        vhfa += vj
        vhfb += vj
        atom0, atom1 = atoms_slice
        for i, ia in enumerate(range(atom0, atom1)):
            for ix in range(3):
                h1moa[ia,ix] += mo_a.T.dot(vhfa[i,ix].dot(mocca))
                h1mob[ia,ix] += mo_b.T.dot(vhfb[i,ix].dot(moccb))
        vj = vja = vjb = vka = vkb = vhfa = vhfb = None
    return h1moa, h1mob

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
        h1aa += get_ecp_ip(mol, 'ipipv')
        h1ab += get_ecp_ip(mol, 'ipvip')
        #h1aa += mol.intor('ECPscalar_ipipnuc', comp=9)
        #h1ab += mol.intor('ECPscalar_ipnucip', comp=9)
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
    '''Solve the CPHF equation for the first orbitals.
    Note: These orbitals are represented in MO basis. This is different to the
    solve_mo1 function in the PySCF CPU version, which transforms the mo1 to AO
    basis.

    Kwargs:
        fx : function(dm_mo) => v1_mo
            A function to generate the induced potential.
            See also the function gen_vind.
    '''
    mol = mf.mol
    log = logger.new_logger(mf, verbose)
    t0 = log.init_timer()

    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    viridxa = mo_occ[0] == 0
    viridxb = mo_occ[1] == 0
    mo_ea, mo_eb = mo_energy
    ei_a = mo_ea[occidxa]
    ei_b = mo_eb[occidxb]
    ea_a = mo_ea[viridxa]
    ea_b = mo_eb[viridxb]
    eai_a = 1 / (ea_a[:,None] + level_shift - ei_a)
    eai_b = 1 / (ea_b[:,None] + level_shift - ei_b)
    nvira, nocca = eai_a.shape
    nvirb, noccb = eai_b.shape
    nocc = nocca + noccb

    nao, nmo = mo_coeff[0].shape
    mocca = mo_coeff[0][:,occidxa]
    moccb = mo_coeff[1][:,occidxb]
    natm = mol.natm

    if fx is None:
        fx = gen_vind(mf, mo_coeff, mo_occ)

    def fvind_vo(mo1):
        mo1 = mo1.reshape(-1,nmo*nocc)
        v = fx(mo1).reshape(-1,nmo*nocc)
        if level_shift != 0:
            v -= mo1 * level_shift
        v1a = v[:,:nmo*nocca].reshape(-1,nmo,nocca)
        v1b = v[:,nmo*nocca:].reshape(-1,nmo,noccb)
        v1a[:,viridxa] *= eai_a
        v1b[:,viridxb] *= eai_b
        v1a[:,occidxa] = 0
        v1b[:,occidxb] = 0
        return v.reshape(-1,nmo*nocc)

    ipovlp = -mol.intor('int1e_ipovlp', comp=3)
    ipovlp = cp.asarray(ipovlp)
    cp.get_default_memory_pool().free_all_blocks()

    avail_mem = get_avail_mem()
    # *8 for spin-up/down input dm, vj, vk, and vxc
    blksize = int(min(avail_mem*.3 / (8*3*nao*nocc*8),
                      avail_mem*.3 / (8*nao*nao*3*6)))  # in vj, vk, dm in AO
    if blksize < ALIGNED**2:
        raise RuntimeError('GPU memory insufficient')

    blksize = (blksize // ALIGNED**2) * ALIGNED**2
    log.debug(f'GPU memory {avail_mem/GB:.1f} GB available')
    log.debug(f'{blksize} atoms in each block CPHF equation')

    natm = mol.natm
    h1moa, h1mob = h1mo
    mo1sa = np.zeros(h1moa.shape)
    mo1sb = np.zeros(h1mob.shape)
    e1sa = np.zeros((natm, 3, nocca, nocca))
    e1sb = np.zeros((natm, 3, noccb, noccb))
    aoslices = mol.aoslice_by_atom()
    for i0, i1 in lib.prange(0, natm, blksize):
        log.info('Solving CPHF equation for atoms [%d:%d]', i0, i1)

        h1a_blk = h1moa[i0:i1]
        h1b_blk = h1mob[i0:i1]
        if not isinstance(h1moa, cp.ndarray):
            h1a_blk = cp.asarray(h1a_blk)
            h1b_blk = cp.asarray(h1b_blk)
        s1a_blk = cp.empty_like(h1a_blk)
        s1b_blk = cp.empty_like(h1b_blk)
        for k, (p0, p1) in enumerate(aoslices[i0:i1,2:]):
            s1ao = cp.zeros((3,nao,nao))
            s1ao[:,p0:p1] += ipovlp[:,p0:p1]
            s1ao[:,:,p0:p1] += ipovlp[:,p0:p1].transpose(0,2,1)
            tmp = contract('xij,jo->xio', s1ao, mocca)
            s1a_blk[k] = contract('xio,ip->xpo', tmp, mo_coeff[0])
            tmp = contract('xij,jo->xio', s1ao, moccb)
            s1b_blk[k] = contract('xio,ip->xpo', tmp, mo_coeff[1])

        mo1a = hs_a = h1a_blk - s1a_blk * ei_a
        mo1b = hs_b = h1b_blk - s1b_blk * ei_b
        mo_e1a = hs_a[:,:,occidxa]
        mo_e1b = hs_b[:,:,occidxb]
        mo1a[:,:,viridxa] *= -eai_a
        mo1b[:,:,viridxb] *= -eai_b
        mo1a[:,:,occidxa] = -s1a_blk[:,:,occidxa] * .5
        mo1b[:,:,occidxb] = -s1b_blk[:,:,occidxb] * .5
        nset = (i1 - i0) * 3
        mo1 = cp.hstack((mo1a.reshape(nset,-1), mo1b.reshape(nset,-1)))
        hs_a = hs_b = h1a_blk = h1b_blk = s1a_blk = s1b_blk = None

        tol = mf.conv_tol_cpscf * (i1 - i0)
        raw_mo1 = krylov(fvind_vo, mo1.reshape(-1,nmo*nocc),
                         tol=tol, max_cycle=max_cycle, verbose=log)
        raw_mo1a = mo1[:,:nmo*nocca].reshape(i1-i0,3,nmo,nocca)
        raw_mo1b = mo1[:,nmo*nocca:].reshape(i1-i0,3,nmo,noccb)

        # The occ-occ block of mo1 is non-canonical
        raw_mo1a[:,:,occidxa] = mo1a[:,:,occidxa]
        raw_mo1b[:,:,occidxb] = mo1b[:,:,occidxb]

        v1 = fx(raw_mo1)
        v1a = v1[:,:nmo*nocca].reshape(i1-i0,3,nmo,nocca)
        v1b = v1[:,nmo*nocca:].reshape(i1-i0,3,nmo,noccb)
        mo1a[:,:,viridxa] -= v1a[:,:,viridxa] * eai_a
        mo1b[:,:,viridxb] -= v1b[:,:,viridxb] * eai_b
        mo_e1a += v1a[:,:,occidxa]
        mo_e1b += v1b[:,:,occidxb]
        mo_e1a += mo1a[:,:,occidxa] * (ei_a[:,None] - ei_a)
        mo_e1b += mo1b[:,:,occidxb] * (ei_b[:,None] - ei_b)

        mo1sa[i0:i1] = mo1a.get()
        mo1sb[i0:i1] = mo1b.get()
        e1sa[i0:i1] = mo_e1a.get()
        e1sb[i0:i1] = mo_e1b.get()
        mo1a = mo1b = mo1 = mo_e1a = mo_e1b = None
        raw_mo1a = raw_mo1b = raw_mo1 = None
        v1a = v1b = v1 = None
    log.timer('CPHF solver', *t0)
    return (mo1sa, mo1sb), (e1sa, e1sb)

def gen_vind(hessobj, mo_coeff, mo_occ):
    # Move data to GPU
    mol = hessobj.mol
    mo_coeff = cupy.asarray(mo_coeff)
    mo_occ = cupy.asarray(mo_occ)
    nao, nmoa = mo_coeff[0].shape
    nao, nmob = mo_coeff[1].shape
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    nocca = mocca.shape[1]
    noccb = moccb.shape[1]

    def fx(mo1):
        mo1 = cupy.asarray(mo1)
        mo1 = mo1.reshape(-1,nmoa*nocca+nmob*noccb)
        nset = len(mo1)

        dm1 = cupy.empty([2,nset,nao,nao])

        x = mo1[:,:nmoa*nocca].reshape(nset,nmoa,nocca)
        mo1_moa = contract('npo,ip->nio', x, mo_coeff[0])
        dma = contract('nio,jo->nij', mo1_moa, mocca)
        dm1[0] = transpose_sum(dma)

        x = mo1[:,nmoa*nocca:].reshape(nset,nmob,noccb)
        mo1_mob = contract('npo,ip->nio', x, mo_coeff[1])
        dmb = contract('nio,jo->nij', mo1_mob, moccb)
        dm1[1] = transpose_sum(dmb)

        dm1 = tag_array(dm1, mo1=[mo1_moa,mo1_mob], occ_coeff=[mocca,moccb], mo_occ=mo_occ)
        return hessobj.get_veff_resp_mo(mol, dm1, mo_coeff, mo_occ, hermi=1)
    return fx

def _get_veff_resp_mo(hessobj, mol, dms, mo_coeff, mo_occ, hermi=1):
    vj, vk = hessobj.get_jk_mo(mol, dms, mo_coeff, mo_occ,
                               hermi=hermi, with_j=True, with_k=True)
    return vj - vk

class Hessian(rhf_hess_gpu.HessianBase):
    '''Non-relativistic unrestricted Hartree-Fock hessian'''

    from gpu4pyscf.lib.utils import to_gpu, device

    __init__ = rhf_hess_gpu.Hessian.__init__
    partial_hess_elec = partial_hess_elec
    hess_elec = hess_elec
    make_h1 = make_h1
    gen_vind = gen_vind
    get_jk_mo = rhf_hess_gpu._get_jk_mo
    get_veff_resp_mo = _get_veff_resp_mo

    def solve_mo1(self, mo_energy, mo_coeff, mo_occ, h1mo,
                  fx=None, atmlst=None, max_memory=4000, verbose=None):
        return solve_mo1(self.base, mo_energy, mo_coeff, mo_occ, h1mo,
                         fx, atmlst, max_memory, verbose,
                         max_cycle=self.max_cycle, level_shift=self.level_shift)

    gen_hop = NotImplemented

# Inject to UHF class
from gpu4pyscf import scf
scf.uhf.UHF.Hessian = lib.class_as_method(Hessian)
