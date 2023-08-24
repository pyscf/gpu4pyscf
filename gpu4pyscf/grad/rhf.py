# gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
#
# Copyright (C) 2022 Qiming Sun
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
from pyscf import lib, gto, grad
from pyscf.lib import logger
from pyscf.scf import hf, jk, _vhf
from gpu4pyscf.lib.cupy_helper import load_library
from gpu4pyscf.scf.hf import _VHFOpt
from gpu4pyscf.lib.utils import patch_cpu_kernel

LMAX_ON_GPU = 4
FREE_CUPY_CACHE = True
BINSIZE = 128
libgvhf = load_library('libgvhf')


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
    fn = libgvhf.GINTbuild_jk_nabla1i
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
        vj = cupy.einsum('pi,apq,qj->aij', coeff, vj, coeff) * 2
        # *2 because only the lower triangle part of dm was used in J contraction
    if with_k:
        vk = cupy.einsum('pi,apq,qj->aij', coeff, vk, coeff)

    cput0 = log.timer_debug1('get_jk pass 1 on gpu', *cput0)

    #TODO: h_shls untested
    h_shls = vhfopt.h_shls
    if h_shls:
        log.debug3('Integrals for %s functions on CPU', l_symb[LMAX_ON_GPU+1])
        pmol = vhfopt.mol
        shls_excludes = [0, h_shls[0]] * 4
        vs_h = _vhf.direct_mapdm('int2e_cart', 's8', scripts,
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


def _grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    '''
    Electronic part of RHF/RKS gradients

    Args:
        mf_grad : grad.rhf.Gradients or grad.rks.Gradients object
    '''
    mf = mf_grad.base
    mol = mf_grad.mol

    if mo_energy is None: mo_energy = cupy.asarray(mf.mo_energy)
    if mo_occ is None:    mo_occ = cupy.asarray(mf.mo_occ)
    if mo_coeff is None:  mo_coeff = cupy.asarray(mf.mo_coeff)
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = cupy.asarray(mf_grad.get_ovlp(mol))
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Computing Gradients of NR-HF Coulomb repulsion')
    vhf = mf_grad.get_veff(mol, dm0)
    log.timer('gradients of 2e part', *t0)

    dme0 = make_rdm1e(mo_energy, mo_coeff, mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = cupy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices [ia,2:]
        h1ao = cupy.asarray(hcore_deriv(ia))
        de[k] += cupy.einsum('xij,ij->x', h1ao, dm0)
        # nabla was applied on bra in vhf, *2 for the contributions of nabla|ket>
        de[k] += cupy.einsum('xij,ij->x', vhf[:,p0:p1], dm0[p0:p1]) * 2
        de[k] -= cupy.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1]) * 2

        de[k] += mf_grad.extra_force(ia, locals())

    if log.verbose >= logger.DEBUG:
        log.debug('gradients of electronic part')
        grad.rhf._write(log, mol, de, atmlst)
    return de


def _kernel(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    cput0 = (logger.process_clock(), logger.perf_counter())

    if mo_energy is None: mo_energy = cupy.asarray(mf_grad.base.mo_energy)
    if mo_coeff is None: mo_coeff = cupy.asarray(mf_grad.base.mo_coeff)
    if mo_occ is None: mo_occ = cupy.asarray(mf_grad.base.mo_occ)
    if atmlst is None:
        atmlst = mf_grad.atmlst
    else:
        mf_grad.atmlst = atmlst

    if mf_grad.verbose >= logger.WARN:
        mf_grad.check_sanity()
    if mf_grad.verbose >= logger.INFO:
        mf_grad.dump_flags()

    de = mf_grad.grad_elec(mo_energy, mo_coeff, mo_occ, atmlst)
    mf_grad.de = de + cupy.asarray(mf_grad.grad_nuc(atmlst=atmlst))
    if mf_grad.mol.symmetry:
        mf_grad.de = mf_grad.symmetrize(mf_grad.de, atmlst)
    logger.timer(mf_grad, 'SCF gradients', *cput0)
    mf_grad._finalize()
    return mf_grad.de


class Gradients(grad.rhf.Gradients):
    screen_tol = 1e-14
    device = 'gpu'

    get_jk = patch_cpu_kernel(grad.rhf.Gradients.get_jk)(_get_jk)
    make_rdm1e = patch_cpu_kernel(grad.rhf.Gradients.make_rdm1e)(_make_rdm1e)
    grad_elec = patch_cpu_kernel(grad.rhf.Gradients.grad_elec)(_grad_elec)
    kernel = patch_cpu_kernel(grad.rhf.Gradients.grad_elec)(_kernel)


Grad = Gradients
