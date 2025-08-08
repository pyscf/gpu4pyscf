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

import time
import ctypes
import numpy as np
import cupy
import numpy
from pyscf import lib, gto
from pyscf.grad import uhf
from pyscf.grad import rhf as rhf_grad_cpu
from gpu4pyscf.gto.ecp import get_ecp_ip
from gpu4pyscf.lib import utils
from gpu4pyscf.lib.cupy_helper import tag_array, contract, ensure_numpy
from gpu4pyscf.df import int3c2e      #TODO: move int3c2e to out of df
from gpu4pyscf.lib import logger
from gpu4pyscf.grad import rhf as rhf_grad

def make_rdm1e(mo_energy, mo_coeff, mo_occ):
    '''Energy weighted density matrix'''
    return cupy.asarray((rhf_grad_cpu.make_rdm1e(mo_energy[0], mo_coeff[0], mo_occ[0]),
                         rhf_grad_cpu.make_rdm1e(mo_energy[1], mo_coeff[1], mo_occ[1])))

def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    '''
    Electronic part of UHF/UKS gradients

    Args:
        mf_grad : grad.uhf.Gradients or grad.uks.Gradients object
    '''
    mf = mf_grad.base
    mol = mf_grad.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()

    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)
    t0 = t1 = log.init_timer()

    mo_energy = cupy.asarray(mo_energy)
    mo_occ = cupy.asarray(mo_occ)
    mo_coeff = cupy.asarray(mo_coeff)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)
    dm0 = tag_array(dm0, mo_coeff=mo_coeff, mo_occ=mo_occ)
    dm0_sf = dm0[0] + dm0[1]
    dme0_sf = dme0[0] + dme0[1]

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = cupy.zeros((len(atmlst),3))
    
    # (\nabla i | hcore | j) - (\nabla i | j)
    h1 = cupy.asarray(mf_grad.get_hcore(mol, exclude_ecp=True))
    s1 = cupy.asarray(mf_grad.get_ovlp(mol))

    # (i | \nabla hcore | j)
    dh1e = int3c2e.get_dh1e(mol, dm0_sf)

    # Calculate ECP contributions in (i | \nabla hcore | j) and 
    # (\nabla i | hcore | j) simultaneously
    if mol.has_ecp():
        ecp_atoms = sorted(set(mol._ecpbas[:,gto.ATOM_OF]))
        h1_ecp = get_ecp_ip(mol, ecp_atoms=ecp_atoms)
        h1 -= h1_ecp.sum(axis=0)

        dh1e[ecp_atoms] += 2.0 * contract('nxij,ij->nx', h1_ecp, dm0_sf)
    t1 = log.timer_debug1('gradients of h1e', *t1)
    log.debug('Computing Gradients of NR-HF Coulomb repulsion')
    dvhf = mf_grad.get_veff(mol, dm0)

    extra_force = np.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        extra_force[k] += ensure_numpy(mf_grad.extra_force(ia, locals()))
    log.timer_debug1('gradients of 2e part', *t1)

    dh = contract('xij,ij->xi', h1, dm0_sf)
    ds = contract('xij,ij->xi', s1, dme0_sf)
    delec = 2.0*(dh - ds)
    delec = cupy.asarray([cupy.sum(delec[:, p0:p1], axis=1) for p0, p1 in aoslices[:,2:]])
    de = ensure_numpy(2.0 * dvhf + dh1e + delec)
    de += extra_force
    log.timer_debug1('gradients of electronic part', *t0)
    return de


class Gradients(rhf_grad.GradientsBase):

    to_cpu = utils.to_cpu
    to_gpu = utils.to_gpu
    device = utils.device

    grad_elec = grad_elec

    def get_veff(self, mol, dm, verbose=None):
        '''
        Computes the first-order derivatives of the energy contributions from
        Veff per atom.

        NOTE: This function is incompatible to the one implemented in PySCF CPU version.
        In the CPU version, get_veff returns the first order derivatives of Veff matrix.
        '''
        vhfopt = self.base._opt_gpu.get(None, None)
        ejk = rhf_grad._jk_energy_per_atom(mol, dm, vhfopt, verbose=verbose)
        return ejk

    def make_rdm1e(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        return make_rdm1e(mo_energy, mo_coeff, mo_occ)

Grad = Gradients

from gpu4pyscf import scf
scf.uhf.UHF.Gradients = lib.class_as_method(Gradients)
