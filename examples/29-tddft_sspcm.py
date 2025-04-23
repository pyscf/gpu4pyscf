#!/usr/bin/env python
# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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

###################################
#  Example of TDDFT with State-Specific Solvent
###################################

"""
[Experimental Feature Notice]
-----------------------------------------------------------------------
This implementation is EXPERIMENTAL and subject to change without warning.
Users must fully understand the theoretical assumptions and limitations 
before application in production systems. Misuse may lead to incorrect results.
"""
import numpy as np
import pyscf
import cupy as cp
import gpu4pyscf
from pyscf import gto, scf, dft, tddft
from gpu4pyscf.solvent._attach_solvent import SCFWithSolvent
from gpu4pyscf.lib.cupy_helper import tag_array, pack_tril
from gpu4pyscf.scf.hf_lowmem import WaveFunction
from gpu4pyscf.lib.cupy_helper import contract
from pyscf import lib
import copy
from functools import reduce
from gpu4pyscf.scf import cphf

def get_total_density(td_grad, mf, x_y, singlet=True, relaxed=True):
    mol = mf.mol
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    nmo = mo_coeff.shape[1]
    nocc = int((mo_occ > 0).sum())
    nvir = nmo - nocc
    orbv = mo_coeff[:, nocc:]
    orbo = mo_coeff[:, :nocc]
    x, y = x_y
    x = cp.asarray(x)
    y = cp.asarray(y)
    xpy = (x + y).reshape(nocc, nvir).T
    xmy = (x - y).reshape(nocc, nvir).T
    dvv = contract("ai,bi->ab", xpy, xpy) + contract("ai,bi->ab", xmy, xmy)  # 2 T_{ab}
    doo = -contract("ai,aj->ij", xpy, xpy) - contract("ai,aj->ij", xmy, xmy)  # 2 T_{ij}
    dmxpy = reduce(cp.dot, (orbv, xpy, orbo.T))  # (X+Y) in ao basis
    dmxmy = reduce(cp.dot, (orbv, xmy, orbo.T))  # (X-Y) in ao basis
    dmzoo = reduce(cp.dot, (orbo, doo, orbo.T))  # T_{ij}*2 in ao basis
    dmzoo += reduce(cp.dot, (orbv, dvv, orbv.T))  # T_{ij}*2 + T_{ab}*2 in ao basis
    vj0, vk0 = mf.get_jk(mol, dmzoo, hermi=0)
    vj1, vk1 = mf.get_jk(mol, dmxpy + dmxpy.T, hermi=0)
    vj2, vk2 = mf.get_jk(mol, dmxmy - dmxmy.T, hermi=0)
    if not isinstance(vj0, cp.ndarray):
        vj0 = cp.asarray(vj0)
    if not isinstance(vk0, cp.ndarray):
        vk0 = cp.asarray(vk0)
    if not isinstance(vj1, cp.ndarray):
        vj1 = cp.asarray(vj1)
    if not isinstance(vk1, cp.ndarray):
        vk1 = cp.asarray(vk1)
    if not isinstance(vj2, cp.ndarray):
        vj2 = cp.asarray(vj2)
    if not isinstance(vk2, cp.ndarray):
        vk2 = cp.asarray(vk2)
    vj = cp.stack((vj0, vj1, vj2))
    vk = cp.stack((vk0, vk1, vk2))
    veff0doo = vj[0] * 2 - vk[0]  # 2 for alpha and beta
    veff0doo += td_grad.solvent_response(dmzoo)
    wvo = reduce(cp.dot, (orbv.T, veff0doo, orbo)) * 2
    if singlet:
        veff = vj[1] * 2 - vk[1]
    else:
        veff = -vk[1]
    veff += td_grad.solvent_response(dmxpy + dmxpy.T)
    veff0mop = reduce(cp.dot, (mo_coeff.T, veff, mo_coeff))
    wvo -= contract("ki,ai->ak", veff0mop[:nocc, :nocc], xpy) * 2  # 2 for dm + dm.T
    wvo += contract("ac,ai->ci", veff0mop[nocc:, nocc:], xpy) * 2
    veff = -vk[2]
    veff0mom = reduce(cp.dot, (mo_coeff.T, veff, mo_coeff))
    wvo -= contract("ki,ai->ak", veff0mom[:nocc, :nocc], xmy) * 2
    wvo += contract("ac,ai->ci", veff0mom[nocc:, nocc:], xmy) * 2

    # set singlet=None, generate function for CPHF type response kernel
    vresp = td_grad.base.gen_response(singlet=None, hermi=1)

    def fvind(x):  # For singlet, closed shell ground state
        dm = reduce(cp.dot, (orbv, x.reshape(nvir, nocc) * 2, orbo.T))  # 2 for double occupancy
        v1ao = vresp(dm + dm.T)  # for the upused 2
        return reduce(cp.dot, (orbv.T, v1ao, orbo)).ravel()

    z1 = cphf.solve(
        fvind,
        mo_energy,
        mo_occ,
        wvo,
        max_cycle=td_grad.cphf_max_cycle,
        tol=td_grad.cphf_conv_tol)[0]
    z1 = z1.reshape(nvir, nocc)

    z1ao = reduce(cp.dot, (orbv, z1, orbo.T))

    if relaxed:
        dmz1doo = z1ao + dmzoo
    else:
        dmz1doo = dmzoo
    return (dmz1doo + dmz1doo.T) * 0.5 + mf.make_rdm1()


def get_phi(pcmobj, sigma):
    mol = pcmobj.mol
    int2c2e = mol._add_suffix('int2c2e')
    grid_coords = pcmobj.surface['grid_coords']
    charge_exp  = pcmobj.surface['charge_exp']
    fakemol_charge = gto.fakemol_for_charges(grid_coords.get(), expnt=charge_exp.get()**2)
    v_ng = gto.mole.intor_cross(int2c2e, fakemol_charge, fakemol_charge)
    v_ng = cp.asarray(v_ng)
    phi_s = sigma@v_ng
    return phi_s


def get_deltaG(mfeq, mfneq, tdneq, tdgneq, eps_optical=1.78):
    pcmobj = mfeq.with_solvent
    eps = pcmobj.eps

    pcmobj_optical = copy.copy(pcmobj)
    pcmobj_optical.dmprev = None
    pcmobj_optical.eps = eps_optical
    pcmobj_optical.build()
    dm1_eq = mfeq.make_rdm1()

    dm2 = get_total_density(tdgneq, mfneq, tdneq.xy[0])
    chi_e = (eps - 1)/(4 * np.pi)
    chi_fast = (eps_optical - 1)/(4 * np.pi)
    chi_slow = (eps - eps_optical)/(4 * np.pi)
    q_1 = pcmobj._get_qsym(dm1_eq, with_nuc=True)[0]
    q_1_s = chi_slow/chi_e * q_1
    q_2 = pcmobj._get_qsym(dm2, with_nuc=True)[0]
    q_2_s = chi_slow/chi_e * q_2
    v1 = pcmobj._get_vgrids(dm1_eq, with_nuc=True)
    v2 = pcmobj._get_vgrids(dm2, with_nuc=True)
    phi_1_s = get_phi(pcmobj, q_1_s)
    vgrids_1 = v1 + phi_1_s
    vgrids_2 = v2 + phi_1_s
    b = pcmobj_optical.left_multiply_R(vgrids_1.T)
    q = pcmobj_optical.left_solve_K(b).T
    vK_1 = pcmobj_optical.left_solve_K(vgrids_1.T, K_transpose = True)
    qt = pcmobj_optical.left_multiply_R(vK_1, R_transpose = True).T
    q_1_f = (q + qt)/2.0
    b = pcmobj_optical.left_multiply_R(vgrids_2.T)
    q = pcmobj_optical.left_solve_K(b).T
    vK_1 = pcmobj_optical.left_solve_K(vgrids_2.T, K_transpose = True)
    qt = pcmobj_optical.left_multiply_R(vK_1, R_transpose = True).T
    q_2_f = (q + qt)/2.0
    v2_rho = pcmobj._get_vgrids(dm2)
    v1_rho = pcmobj._get_vgrids(dm1_eq)
    phi2_f = get_phi(pcmobj, q_2_f)
    phi1_f = get_phi(pcmobj, q_1_f)

    delta_G = 0.5*q_2_f.T@v2_rho + q_1_s.T@v2_rho - 0.5*q_1_s.T@v1_rho + 0.5*q_1_s@phi2_f - 0.5*q_1_s@phi1_f
    nuc_cor = 0.5*q_1_s.T@pcmobj.v_grids_n + 0.5*q_2_f.T@pcmobj.v_grids_n
    e_ss = tdneq.e[0] + mfneq.e_tot + delta_G + mfeq.with_solvent.e + nuc_cor
    e_ss, delta_G + mfeq.with_solvent.e + nuc_cor


def external_iteration():
    pass