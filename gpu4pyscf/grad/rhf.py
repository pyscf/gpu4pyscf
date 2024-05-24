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

import time
import ctypes
import numpy as np
import cupy
import numpy
from pyscf import lib, gto
from pyscf.grad import rhf
from gpu4pyscf.lib.cupy_helper import load_library
from gpu4pyscf.scf.hf import _VHFOpt, KohnShamDFT
from gpu4pyscf.lib.cupy_helper import tag_array, contract, take_last2d
from gpu4pyscf.df import int3c2e      #TODO: move int3c2e to out of df
from gpu4pyscf.lib import logger

LMAX_ON_GPU = 3
FREE_CUPY_CACHE = True
BINSIZE = 128
libgvhf = load_library('libgvhf')

'''
def get_jk(mol, dm, hermi=1, vhfopt=None, with_j=True, with_k=True, omega=None,
           verbose=None):

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
            vj = vj.reshape((3,) + dm0.shape)
        if with_k:
            vk = vk.reshape((3,) + dm0.shape)

    if out_cupy:
        return vj, vk
    else:
        return vj.get() if vj is not None else None, \
            vk.get() if vk is not None else None

def _get_jk(gradient_object, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
            omega=None):
    if omega is not None:
        raise NotImplementedError('Range separated Coulomb integrals')
    mf = gradient_object.base
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(gradient_object)
    log.debug3('apply get_grad_jk on gpu')
    if hasattr(mf, '_opt_gpu'):
        vhfopt = mf._opt_gpu
    else:
        vhfopt = _VHFOpt(mol, getattr(mf.opt, '_intor', 'int2e'),
                         getattr(mf.opt, 'prescreen', 'CVHFnrs8_prescreen'),
                         getattr(mf.opt, '_qcondname', 'CVHFsetnr_direct_scf'),
                         getattr(mf.opt, '_dmcondname', 'CVHFsetnr_direct_scf_dm'))
        vhfopt.build(mf.direct_scf_tol)
        mf._opt_gpu = vhfopt
    vj, vk = get_jk(mol, dm, hermi, vhfopt, with_j, with_k, omega, verbose=log)
    log.timer('vj and vk gradient on gpu', *cput0)
    return vj, vk
'''

def get_jk(mol, dm, hermi=1, vhfopt=None, with_j=True, with_k=True, omega=None,
             verbose=None, atmlst=None):
    if atmlst is None:
        atmlst = range(mol.natm)

    log = logger.new_logger(mol, verbose)
    cput0 = log.init_timer()
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

    vj_per_atom = vk_per_atom = None
    if with_j:
        vj = cupy.zeros([vhfopt.mol.nbas, 3])
        vj_per_atom = cupy.zeros([len(atmlst), 3])
        vj_ptr = ctypes.cast(vj.data.ptr, ctypes.c_void_p)
        scripts.append('ji->s2kl')
    if with_k:
        vk = cupy.zeros([vhfopt.mol.nbas, 3])
        vk_per_atom = cupy.zeros([len(atmlst), 3])
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
    fn = libgvhf.GINTget_veff_ip1
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
                     ctypes.cast(log_q_kl.data.ptr, ctypes.c_void_p)
                     )
            if err != 0:
                detail = f'CUDA Error for ({l_symb[li]}{l_symb[lj]}|{l_symb[lk]}{l_symb[ll]})'
                raise RuntimeError(detail)
            log.debug1('(%s%s|%s%s) on GPU %.3fs',
                       l_symb[li], l_symb[lj], l_symb[lk], l_symb[ll],
                       time.perf_counter() - t0)

    if with_j:
        for atom in atmlst:
            shell_ids = vhfopt.mol.atom_shell_ids(atom)
            vj_per_atom[atom] += cupy.sum(vj[shell_ids], axis=0)

        vj_per_atom *= 2

    if with_k:
        for atom in atmlst:
            shell_ids = vhfopt.mol.atom_shell_ids(atom)
            vk_per_atom[atom] += cupy.sum(vk[shell_ids], axis=0)


    cput0 = log.timer_debug1('get_jk pass 1 on gpu', *cput0)

    if FREE_CUPY_CACHE:
        coeff = dms = None
        cupy.get_default_memory_pool().free_all_blocks()

    if out_cupy:
        return vj_per_atom, vk_per_atom
    else:
        return vj_per_atom.get() if vj is not None else None, \
            vk_per_atom.get() if vk is not None else None

def _get_jk(gradient_object, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
            omega=None):
    mf = gradient_object.base
    log = logger.new_logger(gradient_object)
    cput0 = log.init_timer()
    log.debug3('apply get_grad_jk on gpu')
    if hasattr(mf, '_opt_gpu'):
        vhfopt = mf._opt_gpu
    else:
        vhfopt = _VHFOpt(mol, getattr(mf.opt, '_intor', 'int2e'),
                         getattr(mf.opt, 'prescreen', 'CVHFnrs8_prescreen'),
                         getattr(mf.opt, '_qcondname', 'CVHFsetnr_direct_scf'),
                         getattr(mf.opt, '_dmcondname', 'CVHFsetnr_direct_scf_dm'))
        vhfopt.build(mf.direct_scf_tol)
        mf._opt_gpu = vhfopt
    vj, vk = get_jk(mol, dm, hermi, vhfopt, with_j, with_k, omega, verbose=log)
    log.timer('vj and vk gradient on gpu', *cput0)
    return vj, vk

def get_veff(mf_grad, mol, dm):
    vj, vk = mf_grad.get_jk(mol, dm)
    return vj - vk * .5

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
            dh1e_ecp[ia] = contract('xij,ij->x', cupy.asarray(ecp), dm)
    return 2.0 * dh1e_ecp

def grad_nuc(mf_grad, atmlst=None):
    '''
    Derivatives of nuclear repulsion energy wrt nuclear coordinates
    '''
    z = mf_grad.mol.atom_charges()
    r = mf_grad.mol.atom_coords()
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

def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    '''
    Electronic part of RHF/RKS gradients
    Args:
        mf_grad : grad.rhf.Gradients or grad.rks.Gradients object
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
        t3 = log.init_timer()
        dh1e = int3c2e.get_dh1e(mol, dm0)

        if mol.has_ecp():
            dh1e += get_dh1e_ecp(mol, dm0)
        t3 = log.timer_debug1('gradients of h1e', *t3)

        dvhf = mf_grad.get_veff(mol, dm0)
        log.timer_debug1('gradients of veff', *t3)
        log.debug('Computing Gradients of NR-HF Coulomb repulsion')

        dm0 = tag_array(dm0, mo_coeff=mo_coeff, mo_occ=mo_occ)
        extra_force = cupy.zeros((len(atmlst),3))
        for k, ia in enumerate(atmlst):
            extra_force[k] += mf_grad.extra_force(ia, locals())

        log.timer_debug1('gradients of 2e part', *t3)

    dh = contract('xij,ij->xi', h1, dm0)
    ds = contract('xij,ij->xi', s1, dme0)
    delec = 2.0*(dh - ds)

    delec = cupy.asarray([cupy.sum(delec[:, p0:p1], axis=1) for p0, p1 in aoslices[:,2:]])
    de = 2.0 * dvhf + dh1e + delec + extra_force

    # for backforward compatiability
    if(hasattr(mf, 'disp') and mf.disp is not None):
        g_disp = mf_grad.get_dispersion()
        mf_grad.grad_disp = g_disp
        mf_grad.grad_mf = de

    if log.verbose >= logger.DEBUG:
        log.timer_debug1('gradients of electronic part', *t0)

    ## net force should be zero
    #de -= cupy.sum(de, axis=0)/len(atmlst)
    return de.get()

def get_grad_hcore(mf_grad, mo_coeff=None, mo_occ=None):
    '''
    derivative of hcore in MO
    '''
    mf = mf_grad.base
    mol = mf.mol
    natm = mol.natm
    nao = mol.nao
    if mo_coeff is None: mo_coeff = cupy.asarray(mf.mo_coeff)
    if mo_occ is None: mo_occ = mf.mo_occ

    orbo = mo_coeff[:,mo_occ>0]
    nocc = orbo.shape[1]

    # derivative w.r.t nuclie position
    dh1e = cupy.zeros([3,natm,nao,nocc])
    coords = mol.atom_coords()
    charges = cupy.asarray(mol.atom_charges(), dtype=np.float64)
    fakemol = gto.fakemol_for_charges(coords)
    intopt = int3c2e.VHFOpt(mol, fakemol, 'int2e')
    intopt.build(1e-14, diag_block_with_triu=True, aosym=False, group_size=int3c2e.BLKSIZE, group_size_aux=int3c2e.BLKSIZE)
    orbo_sorted = orbo[intopt.ao_idx]
    mo_coeff_sorted = mo_coeff[intopt.ao_idx]
    for i0,i1,j0,j1,k0,k1,int3c_blk in int3c2e.loop_int3c2e_general(intopt, ip_type='ip1'):
        dh1e[:,k0:k1,j0:j1,:] += contract('xkji,io->xkjo', int3c_blk, orbo_sorted[i0:i1])
        dh1e[:,k0:k1,i0:i1,:] += contract('xkji,jo->xkio', int3c_blk, orbo_sorted[j0:j1])
    dh1e = contract('xkjo,k->xkjo', dh1e, -charges)
    dh1e = contract('xkjo,jp->xkpo', dh1e, mo_coeff_sorted)

    # derivative w.r.t. atomic orbitals
    h1 = mf_grad.get_hcore(mol)
    aoslices = mol.aoslice_by_atom()
    with_ecp = mol.has_ecp()
    if with_ecp:
        ecp_atoms = set(mol._ecpbas[:,gto.ATOM_OF])
    else:
        ecp_atoms = ()
    for atm_id in range(natm):
        shl0, shl1, p0, p1 = aoslices[atm_id]
        h1ao = numpy.zeros([3,nao,nao])
        with mol.with_rinv_at_nucleus(atm_id):
            if with_ecp and atm_id in ecp_atoms:
                h1ao += mol.intor('ECPscalar_iprinv', comp=3)
        h1ao[:,p0:p1] += h1[:,p0:p1]
        h1ao += h1ao.transpose([0,2,1])
        h1ao = cupy.asarray(h1ao)
        h1mo = contract('xij,jo->xio', h1ao, orbo)
        dh1e[:,atm_id] += contract('xio,ip->xpo', h1mo, mo_coeff)
    return dh1e

def as_scanner(mf_grad):
    if isinstance(mf_grad, lib.GradScanner):
        return mf_grad
    logger.info(mf_grad, 'Create scanner for %s', mf_grad.__class__)
    name = mf_grad.__class__.__name__ + SCF_GradScanner.__name_mixin__
    return lib.set_class(SCF_GradScanner(mf_grad),
                         (SCF_GradScanner, mf_grad.__class__), name)

class SCF_GradScanner(lib.GradScanner):
    def __init__(self, g):
        lib.GradScanner.__init__(self, g)

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            assert mol_or_geom.__class__ == gto.Mole
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        self.reset(mol)
        mf_scanner = self.base
        e_tot = mf_scanner(mol)

        if isinstance(mf_scanner, KohnShamDFT):
            if getattr(self, 'grids', None):
                self.grids.reset(mol)
            if getattr(self, 'nlcgrids', None):
                self.nlcgrids.reset(mol)

        de = self.kernel(**kwargs)
        return e_tot, de

class GradientsBase(lib.StreamObject):
    '''
    Basic nuclear gradient functions for non-relativistic methods
    '''

    _keys = {'mol', 'base', 'unit', 'atmlst', 'de'}
    __init__ = rhf.GradientsBase.__init__

    def dump_flags(self, verbose=None):
        return

    reset       = rhf.GradientsBase.reset
    get_hcore   = rhf.GradientsBase.get_hcore
    get_ovlp    = rhf.GradientsBase.get_ovlp
    get_jk      = rhf.GradientsBase.get_jk
    get_j       = rhf.GradientsBase.get_j
    get_k       = rhf.GradientsBase.get_k
    get_veff    = NotImplemented
    make_rdm1e  = rhf.GradientsBase.make_rdm1e
    grad_nuc    = rhf.GradientsBase.grad_nuc
    optimizer   = rhf.GradientsBase.optimizer
    extra_force = rhf.GradientsBase.extra_force
    kernel      = rhf.GradientsBase.kernel
    grad        = rhf.GradientsBase.grad
    _finalize   = rhf.GradientsBase._finalize
    _write      = rhf.GradientsBase._write
    as_scanner  = as_scanner
    _tag_rdm1   = rhf.GradientsBase._tag_rdm1

    # to_cpu can be reused only when __init__ still takes mf
    def to_cpu(self):
        mf = self.base.to_cpu()
        from importlib import import_module
        mod = import_module(self.__module__.replace('gpu4pyscf', 'pyscf'))
        cls = getattr(mod, self.__class__.__name__)
        obj = cls(mf)
        return obj

class Gradients(GradientsBase):
    from gpu4pyscf.lib.utils import to_gpu, device

    make_rdm1e = rhf.Gradients.make_rdm1e
    grad_elec = grad_elec
    grad_nuc = grad_nuc
    get_veff = get_veff
    get_jk = _get_jk

    _keys = {'auxbasis_response', 'grad_disp', 'grad_mf'}

    def get_j(self, mol=None, dm=None, hermi=0, omega=None):
        vj, _ = self.get_jk(mol, dm, with_k=False, omega=omega)
        return vj

    def get_k(self, mol=None, dm=None, hermi=0, omega=None):
        _, vk = self.get_jk(mol, dm, with_j=False, omega=omega)
        return vk

    def extra_force(self, atom_id, envs):
        '''
        grid response is implemented get_veff
        '''
        return 0

Grad = Gradients