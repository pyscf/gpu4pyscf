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
from pyscf import lib, gto
from pyscf.lib import logger
from pyscf.grad import rhf
from gpu4pyscf.lib.utils import patch_cpu_kernel
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.df import int3c2e      #TODO: move int3c2e to out of df


LMAX_ON_GPU = 3
FREE_CUPY_CACHE = True
BINSIZE = 128
libgvhf = load_library('libgvhf')

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

def make_rdm1e(mo_energy, mo_coeff, mo_occ):
    '''Energy weighted density matrix'''
    mo0 = mo_coeff[:,mo_occ>0]
    mo0e = mo0 * (mo_energy[mo_occ>0] * mo_occ[mo_occ>0])
    return cupy.dot(mo0e, mo0.T.conj())


def _make_rdm1e(mf_grad, mo_energy, mo_coeff, mo_occ):
    if mo_energy is None: mo_energy = mf_grad.base.mo_energy
    if mo_coeff is None: mo_coeff = mf_grad.base.mo_coeff
    if mo_occ is None: mo_occ = mf_grad.base.mo_occ
    return make_rdm1e(mo_energy, mo_coeff, mo_occ)

=======
import cupy
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
>>>>>>> master

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
        de += cupy.asarray(g_disp)

    if log.verbose >= logger.DEBUG:
        log.timer_debug1('gradients of electronic part', *t0)

    # net force should be zero
    de -= cupy.sum(de, axis=0)/len(atmlst)
    return de.get()

class Gradients(rhf.Gradients):
    device = 'gpu'
    grad_elec = patch_cpu_kernel(rhf.Gradients.grad_elec)(_grad_elec)
    # get_jk = patch_cpu_kernel(grad.rhf.Gradients.get_jk)(_get_jk)
    # make_rdm1e = patch_cpu_kernel(grad.rhf.Gradients.make_rdm1e)(_make_rdm1e)
    # kernel = patch_cpu_kernel(grad.rhf.Gradients.kernel)(_kernel)

Grad = Gradients
=======
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
        de += cupy.asarray(g_disp)

    if log.verbose >= logger.DEBUG:
        log.timer_debug1('gradients of electronic part', *t0)
    
    # net force should be zero
    de -= cupy.sum(de, axis=0)/len(atmlst)
    return de.get()

class Gradients(rhf.Gradients):
    device = 'gpu'
    grad_elec = patch_cpu_kernel(rhf.Gradients.grad_elec)(_grad_elec)

    #TODO: get_jk
