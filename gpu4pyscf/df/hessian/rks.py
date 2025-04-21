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
#   Modified by Xiaojie Wu <wxj6000@gmail.com>

'''
Non-relativistic RKS analytical Hessian
'''


import numpy
import cupy
from pyscf import lib
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.hessian import rhf as rhf_hess
from gpu4pyscf.hessian import rks as rks_hess
from gpu4pyscf.hessian.rks import _get_enlc_deriv2, _get_vnlc_deriv1
from gpu4pyscf.df.hessian import rhf as df_rhf_hess
from gpu4pyscf.df.hessian.rhf import _get_jk_ip, _partial_hess_ejk
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract

def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = log.init_timer()

    mol = hessobj.mol
    mf = hessobj.base
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)

    mocc = mo_coeff[:,mo_occ>0]
    dm0 = numpy.dot(mocc, mocc.T) * 2

    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    with_k = mf._numint.libxc.is_hybrid_xc(mf.xc)
    de2, ej, ek = _partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ,
                                    atmlst, max_memory, verbose,
                                    with_j=True, with_k=with_k)
    de2 += ej  # (A,B,dR_A,dR_B)
    if with_k:
        de2 -= hyb * ek

    if abs(omega) > 1e-10 and abs(alpha-hyb) > 1e-10:
        ek_lr = _partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ,
                                  atmlst, max_memory, verbose,
                                  with_j=False, with_k=True, omega=omega)[2]
        de2 -= (alpha - hyb) * ek_lr

    max_memory = None
    t1 = log.timer_debug1('computing ej, ek', *t1)
    veff_diag = rks_hess._get_vxc_diag(hessobj, mo_coeff, mo_occ, max_memory)
    t1 = log.timer_debug1('computing veff_diag', *t1)
    aoslices = mol.aoslice_by_atom()
    vxc_dm = rks_hess._get_vxc_deriv2(hessobj, mo_coeff, mo_occ, max_memory)
    t1 = log.timer_debug1('computing veff_deriv2', *t1)
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        veff_dm = vxc_dm[ia]
        de2[i0,i0] += contract('xypq,pq->xy', veff_diag[:,:,p0:p1], dm0[p0:p1])*2
        for j0, ja in enumerate(atmlst[:i0+1]):
            q0, q1 = aoslices[ja][2:]
            #:contract('xypq,pq->xy', veff[:,:,q0:q1], dm0[q0:q1])*2
            de2[i0,j0] += 2.0*cupy.sum(veff_dm[:,:,q0:q1], axis=2)
        for j0 in range(i0):
            de2[j0,i0] = de2[i0,j0].T

    if mf.do_nlc():
        de2 += _get_enlc_deriv2(hessobj, mo_coeff, mo_occ, max_memory)

    log.timer('RKS partial hessian', *time0)
    return de2

def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    mol = hessobj.mol
    natm = mol.natm
    assert atmlst is None or atmlst ==range(natm)
    mf = hessobj.base
    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)

    with_k = ni.libxc.is_hybrid_xc(mf.xc)
    vj1, vk1 = _get_jk_ip(hessobj, mo_coeff, mo_occ, chkfile,
                          atmlst, verbose, with_j=True, with_k=with_k)
    h1mo = vj1
    if with_k:
        h1mo -= .5 * hyb * vk1
    vj1 = vk1 = None

    if abs(omega) > 1e-10 and abs(alpha-hyb) > 1e-10:
        _, vk1_lr = _get_jk_ip(hessobj, mo_coeff, mo_occ, chkfile, atmlst,
                               verbose, with_j=False, with_k=True, omega=omega)
        h1mo -= .5 * (alpha - hyb) * vk1_lr
        vk1_lr = None

    h1mo += rhf_grad.get_grad_hcore(hessobj.base.nuc_grad_method())
    h1mo += rks_hess._get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory)
    if mf.do_nlc():
        h1mo += _get_vnlc_deriv1(hessobj, mo_coeff, mo_occ, max_memory)
    return h1mo

class Hessian(rks_hess.Hessian):
    '''Non-relativistic RKS hessian'''
    from gpu4pyscf.lib.utils import to_gpu, device

    auxbasis_response = 1
    partial_hess_elec = partial_hess_elec
    make_h1 = make_h1
    get_jk_mo = df_rhf_hess._get_jk_mo
