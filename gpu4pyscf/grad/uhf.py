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

import time
import ctypes
import numpy as np
import cupy
import numpy
from pyscf import lib, gto
from pyscf.grad import uhf
from pyscf.grad import rhf as rhf_grad_cpu
from gpu4pyscf.lib.cupy_helper import load_library
from gpu4pyscf.lib.cupy_helper import tag_array, contract
from gpu4pyscf.df import int3c2e      #TODO: move int3c2e to out of df
from gpu4pyscf.lib import logger
from gpu4pyscf.scf.hf import _VHFOpt
from gpu4pyscf.grad import rhf as rhf_grad

LMAX_ON_GPU = 3
FREE_CUPY_CACHE = True
BINSIZE = 128
libgvhf = load_library('libgvhf')

def get_jk(mol, dm, hermi=1, vhfopt=None, with_j=True, with_k=True, omega=None,
           verbose=None, atmlst=None):
    if atmlst is None:
        atmlst = range(mol.natm)

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mol, verbose)
    if hermi != 1:
        raise NotImplementedError('JK-builder only supports hermitian density matrix')
    if omega is None:
        omega = 0.0
    if vhfopt is None:
        vhfopt = _VHFOpt(mol, 'int2e').build(diag_block_with_triu=False)
    out_cupy = isinstance(dm, cupy.ndarray)
    if not isinstance(dm, cupy.ndarray):
        dm = cupy.asarray(dm)
    coeff = cupy.asarray(vhfopt.coeff)
    nao, nao0 = coeff.shape
    dm0 = dm
    dms = cupy.asarray(dm0.reshape(-1,nao0,nao0))
    dms = [cupy.einsum('pi,ij,qj->pq', coeff, x, coeff) for x in dms]
    if dm0.ndim == 2:
        dms = cupy.asarray(dms[0], order='C').reshape(1,nao,nao)
    else:
        dms = cupy.asarray(dms, order='C')
    n_dm = dms.shape[0]
    scripts = []
    vj = vk = None
    vj_ptr = vk_ptr = lib.c_null_ptr()
    gradient_shape = list(dms.shape)
    gradient_shape[0] *= 3
    if with_j:
        vj = cupy.zeros(gradient_shape).transpose(0, 2, 1)
        vj_ptr = ctypes.cast(vj.data.ptr, ctypes.c_void_p)
        scripts.append('ji->s2kl')
    if with_k:
        vk = cupy.zeros(gradient_shape).transpose(0, 2, 1)
        vk_ptr = ctypes.cast(vk.data.ptr, ctypes.c_void_p)
        if hermi == 1:
            scripts.append('jk->s2il')
        else:
            scripts.append('jk->s1il')

    l_symb = lib.param.ANGULAR
    log_qs = vhfopt.log_qs
    direct_scf_tol = vhfopt.direct_scf_tol
    ncptype = len(log_qs)
    cp_idx, cp_jdx = np.tril_indices(ncptype)
    l_ctr_shell_locs = vhfopt.l_ctr_offsets
    l_ctr_ao_locs = vhfopt.mol.ao_loc[l_ctr_shell_locs]
    dm_ctr_cond = np.max(
        [lib.condense('absmax', x, l_ctr_ao_locs) for x in dms.get()], axis=0)

    dm_shl = cupy.zeros([l_ctr_shell_locs[-1], l_ctr_shell_locs[-1]])
    assert dms.flags.c_contiguous
    size_l = np.array([1,3,6,10,15,21,28])
    l_ctr = vhfopt.uniq_l_ctr[:,0]
    r = 0

    for i, li in enumerate(l_ctr):
        i0 = l_ctr_ao_locs[i]
        i1 = l_ctr_ao_locs[i+1]
        ni_shls = (i1-i0)//size_l[li]
        c = 0
        for j, lj in enumerate(l_ctr):
            j0 = l_ctr_ao_locs[j]
            j1 = l_ctr_ao_locs[j+1]
            nj_shls = (j1-j0)//size_l[lj]
            sub_dm = dms[0][i0:i1,j0:j1].reshape([ni_shls, size_l[li], nj_shls, size_l[lj]])
            dm_shl[r:r+ni_shls, c:c+nj_shls] = cupy.max(sub_dm, axis=[1,3])
            c += nj_shls
        r += ni_shls

    dm_shl = cupy.asarray(np.log(dm_shl))
    nshls = dm_shl.shape[0]
    t0 = time.perf_counter()

    if hermi != 1:
        dm_ctr_cond = (dm_ctr_cond + dm_ctr_cond.T) * .5
    fn = libgvhf.GINTbuild_ip1_jk
    for cp_ij_id, log_q_ij in enumerate(log_qs):
        cpi = cp_idx[cp_ij_id]
        cpj = cp_jdx[cp_ij_id]
        li = vhfopt.uniq_l_ctr[cpi,0]
        lj = vhfopt.uniq_l_ctr[cpj,0]
        if li > LMAX_ON_GPU or lj > LMAX_ON_GPU or log_q_ij.size == 0:
            continue

        for cp_kl_id, log_q_kl in enumerate(log_qs):
            cpk = cp_idx[cp_kl_id]
            cpl = cp_jdx[cp_kl_id]
            lk = vhfopt.uniq_l_ctr[cpk,0]
            ll = vhfopt.uniq_l_ctr[cpl,0]
            if lk > LMAX_ON_GPU or ll > LMAX_ON_GPU or log_q_kl.size == 0:
                continue

            # TODO: determine cutoff based on the relevant maximum value of dm blocks?
            sub_dm_cond = max(dm_ctr_cond[cpi,cpj], dm_ctr_cond[cpk,cpl],
                              dm_ctr_cond[cpi,cpk], dm_ctr_cond[cpj,cpk],
                              dm_ctr_cond[cpi,cpl], dm_ctr_cond[cpj,cpl])

            if sub_dm_cond < direct_scf_tol * 1e3:
                continue

            log_cutoff = np.log(direct_scf_tol)
            sub_dm_cond = np.log(sub_dm_cond)

            bins_locs_ij = vhfopt.bins[cp_ij_id]
            bins_locs_kl = vhfopt.bins[cp_kl_id]

            log_q_ij = cupy.asarray(log_q_ij, dtype=np.float64)
            log_q_kl = cupy.asarray(log_q_kl, dtype=np.float64)

            bins_floor_ij = vhfopt.bins_floor[cp_ij_id]
            bins_floor_kl = vhfopt.bins_floor[cp_kl_id]

            nbins_ij = len(bins_locs_ij) - 1
            nbins_kl = len(bins_locs_kl) - 1

            err = fn(vhfopt.bpcache, vj_ptr, vk_ptr,
                     ctypes.cast(dms.data.ptr, ctypes.c_void_p),
                     ctypes.c_int(nao), ctypes.c_int(n_dm),
                     bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                     bins_locs_kl.ctypes.data_as(ctypes.c_void_p),
                     bins_floor_ij.ctypes.data_as(ctypes.c_void_p),
                     bins_floor_kl.ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_int(nbins_ij),
                     ctypes.c_int(nbins_kl),
                     ctypes.c_int(cp_ij_id),
                     ctypes.c_int(cp_kl_id),
                     ctypes.c_double(omega),
                     ctypes.c_double(log_cutoff),
                     ctypes.c_double(sub_dm_cond),
                     ctypes.cast(dm_shl.data.ptr, ctypes.c_void_p),
                     ctypes.c_int(nshls),
                     ctypes.cast(log_q_ij.data.ptr, ctypes.c_void_p),
                     ctypes.cast(log_q_kl.data.ptr, ctypes.c_void_p))
            if err != 0:
                detail = f'CUDA Error for ({l_symb[li]}{l_symb[lj]}|{l_symb[lk]}{l_symb[ll]})'
                raise RuntimeError(detail)
            log.debug1('(%s%s|%s%s) on GPU %.3fs',
                       l_symb[li], l_symb[lj], l_symb[lk], l_symb[ll],
                       time.perf_counter() - t0)
    if with_j:
        vj = cupy.asarray([coeff.T @ vj_slice @ coeff * 2 for vj_slice in vj])
        # *2 because only the lower triangle part of dm was used in J contraction
    if with_k:
        vk = cupy.asarray([coeff.T @ vk_slice @ coeff for vk_slice in vk])

    cput0 = log.timer_debug1('get_jk pass 1 on gpu', *cput0)

    #TODO: h_shls untested
    h_shls = vhfopt.h_shls
    if h_shls:
        log.debug3('Integrals for %s functions on CPU', l_symb[LMAX_ON_GPU+1])
        pmol = vhfopt.mol
        shls_excludes = [0, h_shls[0]] * 4
        vs_h = vhfopt.direct_mapdm('int2e_cart', 's8', scripts,
                                 dms.get(), 1, pmol._atm, pmol._bas, pmol._env,
                                 vhfopt=vhfopt, shls_excludes=shls_excludes)
        coeff = vhfopt.coeff
        pnao = coeff.shape[0]
        idx, idy = np.tril_indices(pnao, -1)
        if with_j and with_k:
            vj1 = vs_h[0]
            vk1 = vs_h[1]
        elif with_j:
            vj1 = vs_h[0]
        else:
            vk1 = vs_h[0]

        if with_j:
            vj1[:,idy,idx] = vj1[:,idx,idy]
            for i, v in enumerate(vj1):
                vj[i] += coeff.T.dot(v).dot(coeff)
        if with_k:
            if hermi:
                vk1[:,idy,idx] = vk1[:,idx,idy]
            for i, v in enumerate(vk1):
                vk[i] += coeff.T.dot(v).dot(coeff)
        cput0 = log.timer_debug1('get_jk pass 2 for l>4 basis on cpu', *cput0)

    if FREE_CUPY_CACHE:
        coeff = dms = None
        cupy.get_default_memory_pool().free_all_blocks()

    if dm0.ndim != 2:
        if with_j:
            vj = vj.reshape(2,3,nao0,nao0)
        if with_k:
            vk = vk.reshape(2,3,nao0,nao0)

    if out_cupy:
        return vj, vk
    else:
        return vj.get() if vj is not None else None, \
            vk.get() if vk is not None else None

def get_veff(mf_grad, mol, dm):
    vk0 = mf_grad.get_jk(mol, dm[0])[1]
    vk1 = mf_grad.get_jk(mol, dm[1])[1]
    vj0 = mf_grad.get_jk(mol, dm[0]+dm[1])[0]
    return vj0 - vk0 - vk1


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
    t0 = log.init_timer()

    mo_energy = cupy.asarray(mo_energy)
    mo_occ = cupy.asarray(mo_occ)
    mo_coeff = cupy.asarray(mo_coeff)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)
    dm0 = tag_array(dm0, mo_coeff=mo_coeff, mo_occ=mo_occ)
    dm0_sf = dm0[0] + dm0[1]
    dme0_sf = dme0[0] + dme0[1]

    s1 = mf_grad.get_ovlp(mol)

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = cupy.zeros((len(atmlst),3))

    def calculate_h1e(h1_gpu, s1_gpu):
        # (\nabla i | hcore | j) - (\nabla i | j)
        h1_cpu = mf_grad.get_hcore(mol)
        s1_cpu = mf_grad.get_ovlp(mol)
        h1_gpu[:] = cupy.asarray(h1_cpu)
        s1_gpu[:] = cupy.asarray(s1_cpu)
        return

    h1 = cupy.empty([3, dm0.shape[1], dm0.shape[2]])
    s1 = cupy.empty([3, dm0.shape[1], dm0.shape[2]])
    with lib.call_in_background(calculate_h1e) as calculate_hs:
        calculate_hs(h1, s1)
        # (i | \nabla hcore | j)
        t3 = log.init_timer()
        dh1e = int3c2e.get_dh1e(mol, dm0_sf)

        log.timer_debug1("get_dh1e", *t3)
        if mol.has_ecp():
            dh1e += rhf_grad.get_dh1e_ecp(mol, dm0_sf)
        t1 = log.timer_debug1('gradients of h1e', *t0)
        log.debug('Computing Gradients of NR-HF Coulomb repulsion')
        dvhf = mf_grad.get_veff(mol, dm0)
        
        extra_force = cupy.zeros((len(atmlst),3))
        for k, ia in enumerate(atmlst):
            extra_force[k] += mf_grad.extra_force(ia, locals())
        log.timer_debug1('gradients of 2e part', *t1)

    dh = contract('xij,ij->xi', h1, dm0_sf)
    ds = contract('xij,ij->xi', s1, dme0_sf)
    delec = 2.0*(dh - ds)
    delec = cupy.asarray([cupy.sum(delec[:, p0:p1], axis=1) for p0, p1 in aoslices[:,2:]])

    de = 2.0 * dvhf + dh1e + delec + extra_force

    # for backward compatiability
    if(hasattr(mf, 'disp') and mf.disp is not None):
        g_disp = mf_grad.get_dispersion()
        mf_grad.grad_disp = g_disp
        mf_grad.grad_mf = de

    de -= cupy.sum(de, axis=0)/len(atmlst)
    return de.get()


class Gradients(rhf_grad.GradientsBase):
    from gpu4pyscf.lib.utils import to_gpu, device

    grad_elec = grad_elec
    grad_nuc = rhf_grad.grad_nuc
    get_veff =  get_veff
    get_jk = rhf_grad._get_jk

    def get_j(self, mol=None, dm=None, hermi=0, omega=None):
        vj, _ = self.get_jk(mol, dm, with_k=False, omega=omega)
        return vj

    def get_k(self, mol=None, dm=None, hermi=0, omega=None):
        _, vk = self.get_jk(mol, dm, with_j=False, omega=omega)
        return vk

    def make_rdm1e(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        return make_rdm1e(mo_energy, mo_coeff, mo_occ)

    def extra_force(self, atom_id, envs):
        '''
        grid response is implemented get_veff
        '''
        return 0
