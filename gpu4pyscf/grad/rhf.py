# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
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

import cupy
import numpy
from pyscf import lib, gto
from pyscf.lib import logger
from pyscf.grad import rhf
from gpu4pyscf.lib.utils import patch_cpu_kernel
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.df import int3c2e      #TODO: move int3c2e to out of df

def get_dh1e_ecp(mol, dm):
    natom = mol.natm
    dh1e_ecp = cupy.zeros([natom,3])
    with_ecp = mol.has_ecp()
    if not with_ecp:
        return dh1e_ecp
    ecp_atoms = set(mol._ecpbas[:,gto.ATOM_OF])
    for ia in ecp_atoms:
        with mol.with_rinv_at_nucleus(ia):
            ecp = mol.intor('ECPscalar_iprinv', comp=3)
            dh1e_ecp[ia] = cupy.einsum('xij,ij->x', ecp, dm)
    return 2.0 * dh1e_ecp

def _grad_nuc(mol, atmlst=None):
    '''
    Derivatives of nuclear repulsion energy wrt nuclear coordinates
    '''
    z = mol.atom_charges()
    r = mol.atom_coords()
    dr = r[:,None,:] - r
    dist = numpy.linalg.norm(dr, axis=2)
    diag_idx = numpy.diag_indices(z.size)
    dist[diag_idx] = 1e100
    rinv = 1./dist
    rinv[diag_idx] = 0.
    gs = numpy.einsum('i,j,ijx,ij->ix', -z, z, dr, rinv**3)
    if atmlst is not None:
        gs = gs[atmlst]
    return gs

def _grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    '''
    Electronic part of RHF/RKS gradients
    Args:
        mf_grad : grad.rhf.Gradients or grad.rks.Gradients object
    '''
    mf = mf_grad.base
    mol = mf_grad.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    t0 = (logger.process_clock(), logger.perf_counter())
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)
    
    mo_energy = cupy.asarray(mo_energy)
    mo_occ = cupy.asarray(mo_occ)
    mo_coeff = cupy.asarray(mo_coeff)

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)
    
    # CPU tasks are executed on background
    def calculate_h1e(h1_gpu, s1_gpu):
        # (\nabla i | hcore | j) - (\nabla i | j)
        h1_cpu = mf_grad.get_hcore(mol)
        s1_cpu = mf_grad.get_ovlp(mol)
        h1_gpu[:] = cupy.asarray(h1_cpu)
        s1_gpu[:] = cupy.asarray(s1_cpu)
        return 

    h1 = cupy.empty([3, dm0.shape[0], dm0.shape[1]])
    s1 = cupy.empty([3, dm0.shape[0], dm0.shape[1]])
    with lib.call_in_background(calculate_h1e) as calculate_hs:
        calculate_hs(h1, s1)
        # (i | \nabla hcore | j)
        dh1e = int3c2e.get_dh1e(mol, dm0)
        if mol.has_ecp():
            dh1e += get_dh1e_ecp(mol, dm0)
    
        t1 = log.timer_debug1('gradients of h1e', *t0)
        log.debug('Computing Gradients of NR-HF Coulomb repulsion')

        dm0 = tag_array(dm0, mo_coeff=mo_coeff, mo_occ=mo_occ)
        dvhf, extra_force = mf_grad.get_veff(mol, dm0)
        t2 = log.timer_debug1('gradients of 2e part', *t1)

    dh = cupy.einsum('xij,ij->xi', h1, dm0)
    ds = cupy.einsum('xij,ij->xi', s1, dme0)
    delec = 2.0*(dh - ds)

    aoslices = mol.aoslice_by_atom()
    delec = cupy.asarray([cupy.sum(delec[:, p0:p1], axis=1) for p0, p1 in aoslices[:,2:]])
    de = 2.0 * dvhf + dh1e + delec + extra_force

    if(hasattr(mf, 'disp') and mf.disp is not None):
        g_disp = mf_grad.get_dispersion()
        mf_grad.grad_disp = g_disp
        mf_grad.grad_mf = de
        de += cupy.asarray(g_disp)

    if log.verbose >= logger.DEBUG:
        log.timer_debug1('gradients of electronic part', *t0)
    
    # net force should be zero
    de -= cupy.sum(de, axis=0)/len(atmlst)
    return de.get()

class Gradients(rhf.Gradients):
    device = 'gpu'
    grad_elec = patch_cpu_kernel(rhf.Gradients.grad_elec)(_grad_elec)
    grad_elec = patch_cpu_kernel(rhf.Gradients.grad_nuc)(_grad_nuc)
    #TODO: get_jk
