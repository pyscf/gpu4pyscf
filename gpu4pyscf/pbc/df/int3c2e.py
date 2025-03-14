# Copyright 2024-2025 The PySCF Developers. All Rights Reserved.
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

'''
Perodic 3-center 2-electron short-range Coulomb integral helper functions
'''

import ctypes
import math
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.lib.parameters import ANGULAR
from pyscf.gto.mole import ANG_OF, ATOM_OF, PTR_COORD, PTR_EXP, conc_env
from pyscf.pbc import tools as pbctools
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.lib.kpts_helper import is_zero
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.gto.mole import group_basis, PTR_BAS_COORD
from gpu4pyscf.scf.jk import _nearest_power2, _scale_sp_ctr_coeff, SHM_SIZE
from gpu4pyscf.gto.mole import extract_pgto_params
from gpu4pyscf.pbc.df.ft_ao import libpbc, init_constant

__all__ = [
    'sr_aux_e2',
]

libpbc.fill_int3c2e.restype = ctypes.c_int

LMAX = 4
L_AUX_MAX = 6
GOUT_WIDTH = 45
THREADS = 256
BVK_CELL_SHELLS = 2400

def sr_aux_e2(cell, auxcell, omega, kpts=None, bvk_kmesh=None, j_only=False):
    r'''
    Short-range 3-center integrals (ij|k). The auxiliary basis functions are
    placed at the second electron.
    '''
    if bvk_kmesh is None and kpts is not None:
        if j_only:
            # Coulomb integrals requires smaller kmesh to converge finite-size effects
            bvk_kmesh = kpts_to_kmesh(cell, bvk_kmesh)
        else:
            # The remote images may contribute to certain k-point mesh,
            # contributing to the finite-size effects in exchange matrix.
            rcut = estimate_rcut(cell, auxcell, omega).max()
            bvk_kmesh = kpts_to_kmesh(cell, kpts, rcut=rcut)
    bvk_kmesh, bvk_kmesh_inp = guess_bvk_kmesh(cell, bvk_kmesh), bvk_kmesh
    logger.debug(cell, 'BvK input %s, set to %s for sr_aux_e2', bvk_kmesh_inp, bvk_kmesh)
    int3c2e_opt = SRInt3c2eOpt(cell, auxcell, omega, bvk_kmesh)
    nao, nao_orig = int3c2e_opt.coeff.shape
    naux = int3c2e_opt.aux_coeff.shape[0]

    gamma_point = kpts is None or (kpts.ndim == 1 and is_zero(kpts))
    if gamma_point:
        out = cp.zeros((nao, nao, naux))
    else:
        kpts = np.asarray(kpts).reshape(-1, 3)
        expLk = cp.exp(1j*cp.asarray(int3c2e_opt.bvkmesh_Ls.dot(kpts.T)))
        nL, nkpts = expLk.shape
        if j_only:
            expLLk = contract('Lk,Mk->LMk', expLk.conj(), expLk)
            expLLk = expLLk.view(np.float64).reshape(nL,nL,nkpts,2)
            out = cp.zeros((nkpts, nao, nao, naux), dtype=np.complex128)
        else:
            out = cp.zeros((nkpts, nkpts, nao, nao, naux), dtype=np.complex128)

    ao_loc = int3c2e_opt.sorted_cell.ao_loc
    aux_loc = int3c2e_opt.sorted_auxcell.ao_loc

    for shls_slice, eri3c in int3c2e_opt.int3c2e_kernel():
        i0, i1, j0, j1 = ao_loc[list(shls_slice[:4])]
        k0, k1 = aux_loc[list(shls_slice[4:])]
        if gamma_point:
            out[i0:i1,j0:j1,k0:k1] = tmp = eri3c.sum(axis=(0,2))
            if i0 != j0:
                out[j0:j1,i0:i1,k0:k1] = tmp.transpose(1,0,2)
        elif j_only:
            tmp = contract('LMkz,LpMqr->kpqrz', expLLk, eri3c)
            tmp = tmp.view(np.complex128)[...,0]
            out[:,i0:i1,j0:j1,k0:k1] = tmp
            if i0 != j0:
                out[:,j0:j1,i0:i1,k0:k1] = tmp.transpose(0,2,1,3).conj()
        else:
            expLkz = expLk.view(np.float64).reshape(nL,nkpts,2)
            tmp = contract('Lkz,MpLqr->Mkpqrz', expLkz, eri3c)
            tmp = tmp.view(np.complex128)[...,0]
            tmp = contract('Mk,Mlpqr->klpqr', expLk.conj(), tmp)
            out[:,:,i0:i1,j0:j1,k0:k1] = tmp
            if i0 != j0:
                out[:,:,j0:j1,i0:i1,k0:k1] = tmp.transpose(1,0,3,2,4).conj()
        tmp = None

    if kpts is None:
        out = contract('pqr,rk->pqk', out, int3c2e_opt.aux_coeff)
        out = contract('pqk,qj->pjk', out, int3c2e_opt.coeff)
        out = contract('pjk,pi->ijk', out, int3c2e_opt.coeff)
    elif j_only:
        #:out = einsum('MpNqr,pi,qj,rk->MiNjk', out, coeff, coeff, auxcoeff)
        out = contract('Npqr,rk->Npqk', out, int3c2e_opt.aux_coeff)
        out = contract('Npqk,qj->Npjk', out, int3c2e_opt.coeff)
        out = contract('Npjk,pi->Nijk', out, int3c2e_opt.coeff)
    else:
        #:out = einsum('MpNqr,pi,qj,rk->MiNjk', out, coeff, coeff, auxcoeff)
        out = contract('MNpqr,rk->MNpqk', out, int3c2e_opt.aux_coeff)
        out = contract('MNpqk,qj->MNpjk', out, int3c2e_opt.coeff)
        out = contract('MNpjk,pi->MNijk', out, int3c2e_opt.coeff)
    return out

def create_img_idx(cell, bvkcell, auxcell, Ls, int3c2e_envs):
    '''integral screening'''
    # consider only the most diffused component of a basis
    exps, cs = extract_pgto_params(cell, 'diffused')
    ls = cell._bas[:,ANG_OF]
    exps = cp.asarray(exps, dtype=np.float32)
    log_cs = np.log(np.abs(cs * ((2*ls+1)/(4*np.pi))**.5))
    log_cs = cp.asarray(log_cs, np.float32)
    nbas = cell.nbas
    nk = bvkcell.nbas // nbas

    # Search the most diffused functions on each atom
    aux_exps, aux_cs = extract_pgto_params(auxcell, 'diffused')
    aux_ls = auxcell._bas[:,ANG_OF]
    r2_aux = np.log(aux_cs**2 / cell.precision * 10**aux_ls) / aux_exps
    atom_aux_exps = []
    atoms = auxcell._bas[:,ATOM_OF]
    atom_aux_exps = cp.full(cell.natm, 1e8, dtype=np.float32)
    for ia in range(cell.natm):
        bas_mask = atoms == ia
        es = aux_exps[bas_mask]
        if len(es) > 0:
            atom_aux_exps[ia] = es[r2_aux[bas_mask].argmax()]

    def gen_img_idx(ish0, ish1, jsh0, jsh1):
        nish = ish1 - ish0
        njsh = jsh1 - jsh0
        #TODO: only tril part when i == j
        ij_pairs = nk * nish * nk * njsh
        img_counts = cp.zeros(ij_pairs, dtype=np.int32)
        err = libpbc.int3c2e_img_counts(
            ctypes.cast(img_counts.data.ptr, ctypes.c_void_p),
            ctypes.byref(int3c2e_envs),
            (ctypes.c_int*4)(ish0, ish1, jsh0, jsh1),
            ctypes.cast(exps.data.ptr, ctypes.c_void_p),
            ctypes.cast(log_cs.data.ptr, ctypes.c_void_p),
            ctypes.cast(atom_aux_exps.data.ptr, ctypes.c_void_p),
            ctypes.c_int(nk), ctypes.c_int(cell.natm))
        if err != 0:
            raise RuntimeError('int3c2e_img_counts failed')

        remaining_idx = np.nonzero(img_counts > 0)[0]
        remaining_idx = remaining_idx[img_counts[remaining_idx].argsort()[::-1]]
        remaining_idx = cp.asarray(remaining_idx, dtype=np.int32, order='C')
        ij_pairs = remaining_idx.size
        img_offsets = cp.empty(ij_pairs+1, dtype=np.int32)
        cp.cumsum(img_counts[remaining_idx], out=img_offsets[1:])
        img_offsets[0] = 0

        img_idx = cp.empty(int(img_offsets[-1]), dtype=np.int32)
        err = libpbc.int3c2e_img_idx(
            ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(img_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(remaining_idx.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ij_pairs),
            ctypes.byref(int3c2e_envs),
            (ctypes.c_int*4)(ish0, ish1, jsh0, jsh1),
            ctypes.cast(exps.data.ptr, ctypes.c_void_p),
            ctypes.cast(log_cs.data.ptr, ctypes.c_void_p),
            ctypes.cast(atom_aux_exps.data.ptr, ctypes.c_void_p),
            ctypes.c_int(nk), ctypes.c_int(cell.natm))
        if err != 0:
            raise RuntimeError('int3c2e_img_idx failed')

        Ki, i, Kj, j = cp.unravel_index(remaining_idx, (nk, nish, nk, njsh))
        i += ish0
        j += jsh0
        # one-dimensional indices corresponding to [Ki,i,Kj,j]
        bas_ij = cp.ravel_multi_index((Ki, i, Kj, j), (nk, nbas, nk, nbas))
        bas_ij = cp.asarray(bas_ij, dtype=np.int32)
        return img_idx, img_offsets, bas_ij
    return gen_img_idx

class SRInt3c2eOpt:
    def __init__(self, cell, auxcell, omega, bvk_kmesh=None):
        assert omega < 0
        self.omega = omega

        self.cell = cell
        cell, coeff, uniq_l_ctr, l_ctr_counts = group_basis(cell, tile=1)
        self.sorted_cell = cell
        self.uniq_l_ctr = uniq_l_ctr
        self.l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
        self.coeff = cp.asarray(coeff)
        self.sorted_cell.omega = omega

        self.auxcell = auxcell
        auxcell, coeff, uniq_l_ctr, l_ctr_counts = group_basis(auxcell, tile=1)
        self.sorted_auxcell = auxcell
        self.uniq_l_ctr_aux = uniq_l_ctr
        self.l_ctr_aux_offsets = np.append(0, np.cumsum(l_ctr_counts))
        self.aux_coeff = cp.asarray(coeff)
        self.sorted_auxcell.omega = omega

        if bvk_kmesh is None:
            bvk_kmesh = np.ones(3, dtype=int)
        self.bvk_kmesh = bvk_kmesh
        self.bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(cell, bvk_kmesh, True)

        if np.prod(bvk_kmesh) == 1:
            bvkcell = cell
        else:
            bvkcell = pbctools.super_cell(cell, bvk_kmesh, wrap_around=True)
            # PTR_BAS_COORD was not initialized in pbctools.supe_rcell
            bvkcell._bas[:,PTR_BAS_COORD] = bvkcell._atm[bvkcell._bas[:,ATOM_OF],PTR_COORD]
        self.bvkcell = bvkcell

    def int3c2e_kernel(self, cutoff=None, verbose=None):
        cell = self.sorted_cell
        auxcell = self.sorted_auxcell
        uniq_l_ctr = self.uniq_l_ctr
        l_ctr_offsets = self.l_ctr_offsets
        l_ctr_aux_offsets = self.l_ctr_aux_offsets
        bvkcell = self.bvkcell

        log = logger.new_logger(cell, verbose)
        cput0 = log.init_timer()
        rcut = estimate_rcut(cell, auxcell, self.omega).max()
        Ls = cp.asarray(bvkcell.get_lattice_Ls(rcut=rcut))
        Ls = Ls[cp.linalg.norm(Ls-.5, axis=1).argsort()]
        nimgs = len(Ls)
        log.debug('int3c2e_kernel rcut = %g, nimgs = %d', rcut, nimgs)

        if cutoff is None:
            omega = cell.omega
            aux_exp, _, aux_l = most_diffused_pgto(auxcell)
            cell_exp, _, cell_l = most_diffused_pgto(cell)
            if omega == 0:
                theta = 1./(1./cell_exp + 1./aux_exp)
            else:
                theta = 1./(1./cell_exp + 1./aux_exp + omega**-2)
            lsum = cell_l * 2 + aux_l + 1
            rad = cell.vol**(-1./3) * rcut + 1
            surface = 4*np.pi * rad**2
            lattice_sum_factor = 2*np.pi*rcut*lsum/(cell.vol*theta) + surface
            cutoff = cell.precision / lattice_sum_factor
            log.debug1('int3c_kernel integral omega=%g theta=%g cutoff=%g',
                       omega, theta, cutoff)

        _atm_cpu, _bas_cpu, _env_cpu = conc_env(
            bvkcell._atm, bvkcell._bas, _scale_sp_ctr_coeff(bvkcell),
            auxcell._atm, auxcell._bas, _scale_sp_ctr_coeff(auxcell))
        #NOTE: PTR_BAS_COORD is not updated in conc_env()
        off = _bas_cpu[bvkcell.nbas,PTR_EXP] - auxcell._bas[0,PTR_EXP]
        _bas_cpu[bvkcell.nbas:,PTR_BAS_COORD] += off

        bvk_ao_loc = bvkcell.ao_loc
        aux_loc = auxcell.ao_loc

        _atm = cp.array(_atm_cpu, dtype=np.int32)
        _bas = cp.array(_bas_cpu, dtype=np.int32)
        _env = cp.array(_env_cpu, dtype=np.float64)
        ao_loc = _conc_locs(bvk_ao_loc, aux_loc)
        bvk_ncells = bvkcell.nbas // cell.nbas
        int3c2e_envs = Int3c2eEnvVars(
            cell.natm, cell.nbas, bvk_ncells, nimgs,
            _atm.data.ptr, _bas.data.ptr, _env.data.ptr, ao_loc.data.ptr,
            Ls.data.ptr, math.log(cutoff),
        )
        # Keep a reference to these arrays, prevent releasing them upon returning the closure
        int3c2e_envs._env_ref_holder = (_atm, _bas, _env, ao_loc, Ls)

        gen_img_idx = create_img_idx(cell, bvkcell, auxcell, Ls, int3c2e_envs)

        uniq_l = uniq_l_ctr[:,0]
        assert uniq_l.max() <= LMAX
        n_groups = len(uniq_l)
        init_constant(cell)
        kern = libpbc.fill_int3c2e
        cp.cuda.Stream.null.synchronize()
        t1 = log.timer_debug1('initialize int3c2e_kernel', *cput0)
        timing_collection = {}
        kern_counts = 0

        cell_ao_loc = cell.ao_loc
        di = (cell_ao_loc[l_ctr_offsets[1:]] - cell_ao_loc[l_ctr_offsets[:-1]]).max()
        dk = (aux_loc[l_ctr_aux_offsets[1:]] - aux_loc[l_ctr_aux_offsets[:-1]]).max()
        buf = cp.empty((bvk_ncells,di, bvk_ncells,di, dk))

        ij_tasks = ((i, j) for i in range(n_groups) for j in range(i+1))
        for i, j in ij_tasks:
            li = uniq_l[i]
            lj = uniq_l[j]
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
            nrow = bvk_ao_loc[ish1] - bvk_ao_loc[ish0]
            ncol = bvk_ao_loc[jsh1] - bvk_ao_loc[jsh0]
            img_idx, img_offsets, bas_ij_idx = gen_img_idx(ish0, ish1, jsh0, jsh1)

            for k, lk in enumerate(self.uniq_l_ctr_aux[:,0]):
                ksh0, ksh1 = l_ctr_aux_offsets[k:k+2]
                naux = aux_loc[ksh1] - aux_loc[ksh0]
                shls_slice = ish0, ish1, jsh0, jsh1, ksh0, ksh1
                eri3c = cp.ndarray((bvk_ncells, nrow, bvk_ncells, ncol, naux),
                                   dtype=np.float64, memptr=buf.data)
                eri3c.fill(0.)
                lll = f'({ANGULAR[li]}{ANGULAR[lj]}|{ANGULAR[lk]})'
                scheme = int3c2e_scheme(li, lj, lk)
                log.debug2('int3c2e_scheme for %s: %s', lll, scheme)
                err = kern(
                    ctypes.cast(eri3c.data.ptr, ctypes.c_void_p),
                    ctypes.byref(int3c2e_envs), (ctypes.c_int*3)(*scheme),
                    (ctypes.c_int*6)(*shls_slice),
                    ctypes.c_int(bvk_ncells), ctypes.c_int(nrow),
                    ctypes.c_int(ncol), ctypes.c_int(naux),
                    ctypes.c_int(bas_ij_idx.size),
                    ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(img_offsets.data.ptr, ctypes.c_void_p),
                    _atm_cpu.ctypes, ctypes.c_int(bvkcell.natm),
                    _bas_cpu.ctypes, ctypes.c_int(bvkcell.nbas), _env_cpu.ctypes)
                if err != 0:
                    raise RuntimeError(f'fill_int3c2e kernel for {lll} failed')
                if log.verbose >= logger.DEBUG1:
                    t1, t1p = log.timer_debug1(f'processing {lll}', *t1), t1
                    if lll not in timing_collection:
                        timing_collection[lll] = 0
                    timing_collection[lll] += t1[1] - t1p[1]
                    kern_counts += 1
                yield shls_slice, eri3c

        if log.verbose >= logger.DEBUG1:
            log.timer('int3c2e', *cput0)
            log.debug1('kernel launches %d', kern_counts)
            for lll, t in timing_collection.items():
                log.debug1('%s wall time %.2f', lll, t)

class Int3c2eEnvVars(ctypes.Structure):
    _fields_ = [
        ('cell0_natm', ctypes.c_uint16),
        ('cell0_nbas', ctypes.c_uint16),
        ('bvk_ncells', ctypes.c_uint16),
        ('nimgs', ctypes.c_uint16),
        ('atm', ctypes.c_void_p),
        ('bas', ctypes.c_void_p),
        ('env', ctypes.c_void_p),
        ('ao_loc', ctypes.c_void_p),
        ('img_coords', ctypes.c_void_p),
        ('log_cutoff', ctypes.c_float),
    ]

def _conc_locs(ao_loc1, ao_loc2):
    comp_loc = np.append(ao_loc1[:-1], ao_loc1[-1] + ao_loc2)
    return cp.array(comp_loc, dtype=np.int32)

def int3c2e_scheme(li, lj, lk, shm_size=SHM_SIZE):
    order = li + lj + lk
    nroots = (order//2 + 1) * 2

    g_size = (li+1)*(lj+1)*(lk+1)
    unit = g_size*3 + nroots*2 + 6
    nksp_max = shm_size//(unit*8)
    nksp_max = _nearest_power2(nksp_max)

    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    nfk = (lk + 1) * (lk + 2) // 2
    gout_size = nfi * nfj * nfk
    gout_stride = (gout_size + GOUT_WIDTH-1) // GOUT_WIDTH
    # Round up to the next 2^n
    gout_stride = _nearest_power2(gout_stride, return_leq=False)

    # Align nksh*gout_stride to warp size
    if gout_stride < 32:
        nksh_per_block = 32 // gout_stride
        nsp_per_block = min(THREADS // 32, nksp_max // nksh_per_block)
    else:
        nksh_per_block = THREADS // gout_stride
        nsp_per_block = 1
    if nksp_max < nksh_per_block:
        raise RuntimeError('GOUT_WIDTH too small or not enough shared memory')

    gout_stride = THREADS // (nksh_per_block*nsp_per_block)
    return nksh_per_block, gout_stride, nsp_per_block

def most_diffused_pgto(cell):
    exps, cs = extract_pgto_params(cell, 'diffused')
    ls = cell._bas[:,ANG_OF]
    r2 = np.log(cs**2 / cell.precision * 10**ls) / exps
    idx = r2.argmax()
    return exps[idx], cs[idx], ls[idx]

# This modified rcut estimation function will be available in pyscf-2.8 or newer
def estimate_rcut(cell, auxcell, omega):
    '''Estimate rcut for 3c2e SR-integrals'''
    if cell.nbas == 0 or auxcell.nbas == 0:
        return np.zeros(1)

    if omega == 0:
        # No SR integrals in int3c2e if omega=0
        assert cell.dimension == 0
        return np.zeros(1)

    precision = cell.precision
    ak, ck, lk = most_diffused_pgto(auxcell)

    # the most diffused orbital basis
    cell_exps, cs = extract_pgto_params(cell, 'diffused')
    ls = cell._bas[:,ANG_OF]
    r2_cell = np.log(cs**2 / precision * 10**ls) / cell_exps
    ai_idx = r2_cell.argmax()
    ai = cell_exps[ai_idx]
    aj = cell_exps
    li = ls[ai_idx]
    lj = ls
    ci = cs[ai_idx]
    cj = cs

    aij = ai + aj
    lij = li + lj
    l3 = lij + lk
    theta = 1./(omega**-2 + 1./aij + 1./ak)
    norm_ang = ((2*li+1)*(2*lj+1))**.5/(4*np.pi)
    c1 = ci * cj * ck * norm_ang
    sfac = aij*aj/(aij*aj + ai*theta)
    fl = 2
    fac = 2**li*np.pi**2.5*c1 * theta**(l3-.5)
    rad = cell.vol**(-1./3) * cell.rcut + 1
    surface = 4*np.pi * rad**2
    lattice_sum_factor = 2*np.pi*cell.rcut/(cell.vol*theta) + surface
    fac *= lattice_sum_factor
    fac /= aij**(li+1.5) * ak**(lk+1.5) * aj**lj
    fac *= fl / precision

    r0 = cell.rcut  # initial guess
    r0 = (np.log(fac * (sfac*r0)**(l3-1) + 1.) / (sfac*theta))**.5
    r0 = (np.log(fac * (sfac*r0)**(l3-1) + 1.) / (sfac*theta))**.5
    rcut = r0
    return rcut

def guess_bvk_kmesh(cell, bvk_kmesh, target_size=BVK_CELL_SHELLS):
    '''Generate a sufficient large bvk cell for fill_int3c2e kernel to achieve
    better load balance'''
    if bvk_kmesh is None:
        bvk_kmesh = np.ones(3, dtype=int)
    else:
        bvk_kmesh = bvk_kmesh.copy()
    bvk_ncells = np.prod(bvk_kmesh)

    # produce a cell with ~2000 shells
    replica = target_size / (bvk_ncells * cell.nbas)
    if replica < 1:
        return bvk_kmesh

    mesh_max = cell.nimgs * 2 + 1
    bvk_multiplier = mesh_max / bvk_kmesh
    if cell.dimension == 2:
        fac = (replica / np.prod(bvk_multiplier[:2]))**.5
        fac = min(fac, 1)
        bvk_kmesh[:2] *= (fac * bvk_multiplier[:2]).astype(int)
    else:
        # The replica on each axis should be proportional to the required nimg
        # along each direction.
        fac = (replica / np.prod(bvk_multiplier))**(1./3)
        # The replica is not necessary to be more than the required nimg.
        fac = min(fac, 1)
        bvk_kmesh *= (fac * bvk_multiplier).astype(int)

    return bvk_kmesh
