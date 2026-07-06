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


import cupy
from pyscf import lib
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.hessian import rhf as rhf_hess
from gpu4pyscf.hessian import uhf as uhf_hess
from gpu4pyscf.hessian import uks as uks_hess
from gpu4pyscf.df.hessian import rhf as df_rhf_hess
from gpu4pyscf.df.hessian import uhf as df_uhf_hess
from gpu4pyscf.hessian.rks import _get_enlc_deriv2, _get_vnlc_deriv1
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

    if mf.with_df.intopt is None:
        mf.with_df.build(build_cderi=False)
    intopt = mf.with_df.intopt
    mf.with_df.reset() # Release GPU memory

    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    de2 = df_uhf_hess._jk_energy_per_atom(intopt, dm0, 1., hyb, verbose=log)

    if abs(omega) > 1e-10 and abs(alpha-hyb) > 1e-10:
        de2 += df_uhf_hess._jk_energy_per_atom(intopt, dm0, 0., alpha-hyb,
                                               omega, verbose=log)
    t1 = log.timer_debug1('computing ej, ek', *t1)

    # Energy weighted density matrix
    dme0_sf  = (mo_coeff[0] * (mo_occ[0]*mo_energy[0])).dot(mo_coeff[0].T)
    dme0_sf += (mo_coeff[1] * (mo_occ[1]*mo_energy[1])).dot(mo_coeff[1].T)
    dm0_sf = dm0[0] + dm0[1]
    de2 += df_uhf_hess._hcore_energy(hessobj, dm0_sf, dme0_sf)
    log.timer_debug1('hcore contribution', *t1)

    max_memory = None
    de2 += uks_hess._get_exc_deriv2(hessobj, mo_coeff, mo_occ, (dm0a, dm0b), max_memory)
    if mf.do_nlc():
        de2 += _get_enlc_deriv2(hessobj, mo_coeff, mo_occ, max_memory)

    log.timer('RKS partial hessian', *time0)
    return de2

def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    mol = hessobj.mol
    natm = mol.natm
    assert atmlst is None or atmlst ==range(natm)
    mf = hessobj.base
    if mf.with_df.intopt is None:
        mf.with_df.build(build_cderi=False)
    intopt = mf.with_df.intopt

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)

    h1mo = df_uhf_hess._get_veff(intopt, mo_coeff, mo_occ, 1., hyb)

    if abs(omega) > 1e-10 and abs(alpha-hyb) > 1e-10:
        veff_lr = df_uhf_hess._get_veff(intopt, mo_coeff, mo_occ, 0., alpha-hyb, omega)
        h1mo[0] += veff_lr[0]
        h1mo[1] += veff_lr[1]

    gobj = hessobj.base.nuc_grad_method()
    h1moa += rhf_grad.get_grad_hcore(gobj, mo_coeff[0], mo_occ[0])
    h1mob += rhf_grad.get_grad_hcore(gobj, mo_coeff[1], mo_occ[1])

    v1moa, v1mob = uks_hess._get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory)
    h1moa += v1moa
    h1mob += v1mob

    if mf.do_nlc():
        h1moa_nlc, h1mob_nlc = _get_vnlc_deriv1(hessobj, mo_coeff, mo_occ, max_memory)
        h1moa += h1moa_nlc
        h1mob += h1mob_nlc

    return h1moa, h1mob

class Hessian(uks_hess.Hessian):
    '''Non-relativistic RKS hessian'''

    _keys = {'auxbasis_response',}

    auxbasis_response = 2
    partial_hess_elec = partial_hess_elec
    make_h1 = make_h1
    get_jk_mo = df_rhf_hess._get_jk_mo
    to_cpu = df_uhf_hess.Hessian.to_cpu
