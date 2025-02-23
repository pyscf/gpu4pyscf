#!/usr/bin/env python
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

import itertools
import ctypes
from dataclasses import dataclass
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.gto import ATOM_OF, ANG_OF, NPRIM_OF, NCTR_OF, PTR_EXP, PTR_COEFF, PTR_COORD
from pyscf.pbc.gto import pseudo
from pyscf.pbc.df.df_jk import _format_kpts_band
from pyscf.pbc.lib.kpts_helper import is_zero
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import utils
from gpu4pyscf.lib.cupy_helper import (
    load_library, tag_array, contract, sandwich_dot, block_diag)
from gpu4pyscf.gto.mole import cart2sph_by_l
from gpu4pyscf.dft import numint
from gpu4pyscf.pbc import tools
from gpu4pyscf.pbc.df.fft import get_SI, _check_kpts
from gpu4pyscf.pbc.df.fft_jk import _format_dms, _format_jks
from gpu4pyscf.pbc.df.ft_ao import ft_ao
from gpu4pyscf.__config__ import shm_size
from gpu4pyscf.__config__ import props as gpu_specs

__all__ = ['MultiGridNumInt']

libmgrid = load_library('libmgrid')
libmgrid.MG_eval_rho_orth.restype = ctypes.c_int
libmgrid.MG_eval_mat_lda_orth.restype = ctypes.c_int
libmgrid.MG_init_constant.restype = ctypes.c_int

PRIMBAS_ANG = 0
PRIMBAS_EXP = 1
PRIMBAS_COEFF = 2
PRIMBAS_COORD = 3
LMAX = 4
SHARED_RHO_SIZE = 2097152 # 128^3
SHM_SIZE = shm_size - 1024
del shm_size
WARP_SIZE = 32

def get_j_kpts(ni, dm_kpts, hermi=1, kpts=None, kpts_band=None):
    '''Get the Coulomb (J) AO matrix at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.  If a list of k-point DMs, eg,
            UHF alpha and beta DM, the alpha and beta DMs are contracted
            separately.
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpts_band : ``(3,)`` ndarray or ``(*,3)`` ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        or list of vj if the input dm_kpts is a list of DMs
    '''
    assert kpts is None
    kpts = np.zeros((1, 3))

    cell = ni.cell
    dm_kpts = cp.asarray(dm_kpts)
    rhoG = _eval_rhoG(ni, dm_kpts, hermi, kpts, deriv=0)
    coulG = tools.get_coulG(cell, mesh=cell.mesh)
    #:vG = np.einsum('ng,g->ng', rhoG[:,0], coulG)
    vG = rhoG[:,0]
    vG *= coulG

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    vj_kpts = _get_j_pass2(ni, vG, hermi, kpts_band)
    return _format_jks(vj_kpts, dm_kpts, input_band, kpts)

def _eval_rhoG(ni, dm_kpts, hermi=1, kpts=None, deriv=0):
    cell = ni.cell
    log = logger.new_logger(cell)
    t0 = log.init_timer()

    dm_kpts = cp.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts = dms.shape[:2]
    assert nkpts == 1

    dms = ni.sort_orbitals(dms)
    lmax = cell._bas[:,ANG_OF].max()
    nao = dms.shape[-1]

    assert (deriv < 2)
    #hermi = hermi and abs(dms - dms.transpose(0,1,3,2).conj()).max() < 1e-9
    gga_high_order = False
    if deriv == 0:
        #xctype = 'LDA'
        rhodim = 1

    elif deriv == 1:
        gga_high_order = True
        #xctype = 'LDA'
        rhodim = 1
        deriv = 0

    elif deriv == 2:  # meta-GGA
        raise NotImplementedError

    ignore_imag = (hermi == 1)
    assert ignore_imag

    a = cell.lattice_vectors()
    assert abs(a - np.diag(a.diagonal())).max() < 1e-5, 'Must be orthogonal lattice'
    lattice_params = cp.asarray(a.diagonal(), order='C')
    supmol_bas = cp.asarray(ni.supmol_bas, dtype=np.int32)
    supmol_env = cp.asarray(ni.supmol_env)
    ao_loc_in_cell0 = cp.asarray(ni.ao_loc_in_cell0, dtype=np.int32)
    mg_envs = MGridEnvVars(
        ni.primitive_nbas, len(supmol_bas), nao, supmol_bas.data.ptr,
        supmol_env.data.ptr, ao_loc_in_cell0.data.ptr, lattice_params.data.ptr)
    mg_envs._env_ref_holder = (supmol_bas, supmol_env, ao_loc_in_cell0, lattice_params)
    tasks = ni.tasks
    workers = gpu_specs['multiProcessorCount']
    nf3 = (lmax*2+1)*(lmax*2+2)*(lmax*2+3)//6
    ngrid_span = max(task.n_radius*2 for task in itertools.chain(*tasks))
    cache_size = ((lmax*2+1)*ngrid_span*3 + nf3 + 3) * WARP_SIZE
    pool = cp.empty((workers, cache_size))

    init_constant(cell)
    kern = libmgrid.MG_eval_rho_orth
    assert rhodim == 1
    rhoG = None

    for i, sub_tasks in enumerate(tasks):
        if not sub_tasks: continue
        task = sub_tasks[0]
        mesh = task.mesh
        ngrids = np.prod(mesh)
        rhoR = cp.empty((nset, *mesh))
        for iset in range(nset):
            if ngrids < SHARED_RHO_SIZE:
                rho_local = cp.zeros((workers*8, *mesh))
            else:
                rho_local = cp.zeros(mesh)
            for task in sub_tasks:
                kern(ctypes.cast(rho_local.data.ptr, ctypes.c_void_p),
                     ctypes.cast(dms[iset].data.ptr, ctypes.c_void_p),
                     mg_envs, ctypes.c_int(task.l), ctypes.c_int(task.n_radius),
                     (ctypes.c_int*3)(*task.mesh),
                     ctypes.c_uint32(len(task.shl_pair_idx)),
                     ctypes.cast(task.shl_pair_idx.data.ptr, ctypes.c_void_p),
                     ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                     ctypes.c_int(workers))
            if ngrids < SHARED_RHO_SIZE:
                rho_local = rho_local.sum(axis=0)
            rhoR[iset] = rho_local

        weight = 1./nkpts * cell.vol/ngrids
        rho_freq = tools.fft(rhoR.reshape(nset*rhodim, *mesh), mesh)
        rho_freq *= weight
        if rhoG is None:
            rhoG = rho_freq.reshape(-1, *mesh)
        else:
            _takebak_4d(rhoG, rho_freq.reshape(-1, *mesh), mesh)
    # TODO: for diffused basis functions lower than minimal Ecut, compute the
    # rhoR using normal FFTDF code

    rhoG = rhoG.reshape(nset,rhodim,-1)

    if gga_high_order:
        Gv = cp.asarray(cell.get_Gv(ni.mesh))
        rhoG1 = cp.einsum('np,px->nxp', 1j*rhoG[:,0], Gv)
        rhoG = cp.concatenate([rhoG, rhoG1], axis=1)
    log.timer_debug1('eval_rhoG', *t0)
    return rhoG

def _get_j_pass2(ni, vG, hermi=1, kpts=None, verbose=None):
    cell = ni.cell
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()
    nkpts = len(kpts)
    assert nkpts == 1, 'gamma point only'
    nao = cell.nao_nr(cart=True)
    nset = vG.shape[0]
    lmax = cell._bas[:,ANG_OF].max()

    a = cell.lattice_vectors()
    assert abs(a - np.diag(a.diagonal())).max() < 1e-5, 'Must be orthogonal lattice'
    lattice_params = cp.asarray(a.diagonal(), order='C')
    supmol_bas = cp.asarray(ni.supmol_bas, dtype=np.int32)
    supmol_env = cp.asarray(ni.supmol_env)
    ao_loc_in_cell0 = cp.asarray(ni.ao_loc_in_cell0, dtype=np.int32)
    mg_envs = MGridEnvVars(
        ni.primitive_nbas, len(supmol_bas), nao, supmol_bas.data.ptr,
        supmol_env.data.ptr, ao_loc_in_cell0.data.ptr, lattice_params.data.ptr)
    mg_envs._env_ref_holder = (supmol_bas, supmol_env, ao_loc_in_cell0, lattice_params)
    workers = gpu_specs['multiProcessorCount']
    tasks = ni.tasks
    nf2 = (lmax*2+1)*(lmax*2+2)//2
    ngrid_span = max(task.n_radius*2 for task in itertools.chain(*tasks))
    cache_size = ((lmax*2+1)*ngrid_span*3 + nf2*ngrid_span + 3) * WARP_SIZE
    pool = cp.empty((workers, cache_size))

    mesh_largest = tasks[0][0].mesh
    vG = vG.reshape(-1, *mesh_largest)

    init_constant(cell)
    kern = libmgrid.MG_eval_mat_lda_orth
    # TODO: might be complex array when tddft amplitudes are complex
    vj = cp.zeros((nset,nao,nao))

    for i, sub_tasks in enumerate(tasks):
        if not sub_tasks: continue
        task = sub_tasks[0]
        mesh = task.mesh
        ngrids = np.prod(mesh)
        sub_vG = _take_4d(vG, mesh).reshape(nset,ngrids)
        v_rs = tools.ifft(sub_vG, mesh).reshape(nset,ngrids)
        imag_max = abs(v_rs.imag).max()
        if imag_max > 1e-5:
            msg = f'Imaginary values {imag_max} in potential. mesh {mesh} might be insufficient'
            #raise RuntimeError(msg)
            logger.warn(cell, msg)

        vR = cp.asarray(v_rs.real, order='C')
        for iset in range(nset):
            for task in sub_tasks:
                kern(ctypes.cast(vj[iset].data.ptr, ctypes.c_void_p),
                     ctypes.cast(vR[iset].data.ptr, ctypes.c_void_p),
                     mg_envs, ctypes.c_int(task.l), ctypes.c_int(task.n_radius),
                     (ctypes.c_int*3)(*task.mesh),
                     ctypes.c_uint32(len(task.shl_pair_idx)),
                     ctypes.cast(task.shl_pair_idx.data.ptr, ctypes.c_void_p),
                     ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                     ctypes.c_int(workers))

    # TODO: for diffused basis functions lower than minimal Ecut, compute the
    # vj using normal FFTDF code
    vj = ni.unsort_orbitals(vj)
    nao = vj.shape[-1]
    vj = vj.reshape(nset,nkpts,nao,nao)
    log.timer_debug1('get_j pass2', *t0)
    return vj

def _get_gga_pass2(ni, vG, hermi=1, kpts=np.zeros((1,3)), verbose=None):
    raise NotImplementedError

def nr_rks(ni, cell, grids, xc_code, dm_kpts, relativity=0, hermi=1,
           kpts=None, kpts_band=None, with_j=False, verbose=None):
    '''Compute the XC energy and RKS XC matrix at sampled k-points.
    multigrid version of function pbc.dft.numint.nr_rks.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpts_band : ``(3,)`` ndarray or ``(*,3)`` ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.
        with_j : bool
            Whether to add the Coulomb matrix into the XC matrix.

    Returns:
        exc : XC energy
        nelec : number of electrons obtained from the numerical integration
        veff : (nkpts, nao, nao) ndarray
            or list of veff if the input dm_kpts is a list of DMs
    '''
    assert kpts is None or all(kpts == 0)
    kpts = np.zeros((1, 3))

    log = logger.new_logger(cell, verbose)
    cell = ni.cell
    dm_kpts = cp.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band

    xctype = ni._xc_type(xc_code)
    if xctype == 'LDA':
        deriv = 0
    elif xctype == 'GGA':
        deriv = 1
        raise NotImplementedError
    elif xctype == 'MGGA':
        deriv = 1
        raise NotImplementedError
    rhoG = _eval_rhoG(ni, dms, hermi, kpts, deriv)

    mesh = ni.mesh
    ngrids = np.prod(mesh)
    coulG = tools.get_coulG(cell, mesh=mesh)
    vG = rhoG[0,0] * coulG
    ecoul = .5 * float(rhoG[0,0].conj().dot(vG).real)
    ecoul /= cell.vol
    log.debug('Multigrid Coulomb energy %s', ecoul)

    weight = cell.vol / ngrids
    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    rhoR = tools.ifft(rhoG, mesh).real * (1./weight)
    rhoR = cp.asarray(rhoR.reshape(nset,-1,ngrids), order='C')
    nelec = float(rhoR[0,0].sum()) * weight

    wv_freq = []
    excsum = 0
    for i in range(nset):
        if xctype == 'LDA':
            exc, vxc = ni.eval_xc_eff(xc_code, rhoR[i,0], deriv=1, xctype=xctype)[:2]
        else:
            exc, vxc = ni.eval_xc_eff(xc_code, rhoR[i], deriv=1, xctype=xctype)[:2]
        if i == 0:
            excsum += float(rhoR[0,0].dot(exc[:,0])) * weight
        wv = weight * vxc
        wv_freq.append(tools.fft(wv, mesh))
    wv_freq = cp.asarray(wv_freq).reshape(nset,-1,*mesh)
    rhoR = rhoG = None
    log.debug('Multigrid exc %s  nelec %s', excsum, nelec)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    if xctype == 'LDA':
        if with_j:
            wv_freq[:,0] += vG.reshape(nset,*mesh)
        veff = _get_j_pass2(ni, wv_freq, hermi, kpts_band, verbose=log)
    elif xctype == 'GGA':
        if with_j:
            wv_freq[:,0] += vG.reshape(nset,*mesh)
        # *.5 because v+v.T is always called in _get_gga_pass2
        wv_freq[:,0] *= .5
        veff = _get_gga_pass2(ni, wv_freq, hermi, kpts_band, verbose=log)
    veff = _format_jks(veff, dm_kpts, input_band, kpts)

    shape = list(dm_kpts.shape)
    if len(shape) == 3 and shape[0] != kpts_band.shape[0]:
        shape[0] = kpts_band.shape[0]
    veff = veff.reshape(shape)
    veff = tag_array(veff, ecoul=ecoul, exc=excsum, vj=None, vk=None)
    return nelec, excsum, veff

# Note nr_uks handles only one set of KUKS density matrices (alpha, beta) in
# each call (nr_rks supports multiple sets of KRKS density matrices)
def nr_uks(ni, cell, grids, xc_code, dm_kpts, relativity=0, hermi=1,
           kpts=None, kpts_band=None, with_j=False, verbose=None):
    raise NotImplementedError

def get_rho(ni, dm, kpts=None):
    '''Density in real space
    '''
    assert kpts is None
    kpts = np.zeros((1, 3))

    cell = ni.cell
    hermi = 1
    rhoG = _eval_rhoG(ni, cp.asarray(dm), hermi, kpts, deriv=0)

    mesh = ni.mesh
    ngrids = np.prod(mesh)
    weight = cell.vol / ngrids
    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    rhoR = tools.ifft(rhoG.reshape(ngrids), mesh).real * (1./weight)
    return rhoR

def get_nuc(ni, kpts=None):
    assert kpts is None or all(kpts == 0)
    if kpts is None or kpts.ndim == 1:
        is_single_kpt = True
    kpts = np.zeros((1, 3))

    cell = ni.cell
    mesh = ni.mesh
    charge = cp.asarray(-cell.atom_charges())
    Gv = cell.get_Gv(mesh)
    # TODO: SI size = prod(mesh)*natm, may exceed the GPU memory size
    SI = get_SI(cell, mesh=mesh)
    rhoG = charge.dot(SI)

    coulG = tools.get_coulG(cell, mesh=mesh, Gv=Gv)
    vneG = rhoG
    vneG *= coulG
    hermi = 1
    vne = _get_j_pass2(ni, vneG[None,:], hermi, kpts)[0]
    if is_single_kpt:
        vne = vne[0]
    return vne

def get_pp(ni, kpts=None):
    '''Get the periodic pseudopotential nuc-el AO matrix, with G=0 removed.
    '''
    from pyscf import gto
    from pyscf.pbc.gto.pseudo import pp_int
    assert kpts is None or all(kpts == 0)
    if kpts is None or kpts.ndim == 1:
        is_single_kpt = True
    kpts = np.zeros((1, 3))

    cell = ni.cell
    log = logger.new_logger(cell)
    t0 = log.init_timer()
    mesh = ni.mesh
    Gv = cell.get_Gv(mesh)
    # TODO: SI size = prod(mesh)*natm, may exceed the GPU memory size
    SI = get_SI(cell, mesh=mesh)
    vpplocG = cp.asarray(pseudo.get_vlocG(cell, Gv))
    vpplocG = -contract('ij,ij->j', SI, vpplocG)
    vpp = _get_j_pass2(ni, vpplocG[None,:], kpts=kpts)[0]

    # vppnonloc evaluated in reciprocal space
    fakemol = gto.Mole()
    fakemol._atm = np.zeros((1,gto.ATM_SLOTS), dtype=np.int32)
    fakemol._bas = np.zeros((1,gto.BAS_SLOTS), dtype=np.int32)
    ptr = gto.PTR_ENV_START
    fakemol._env = np.zeros(ptr+10)
    fakemol._bas[0,gto.NPRIM_OF ] = 1
    fakemol._bas[0,gto.NCTR_OF  ] = 1
    fakemol._bas[0,gto.PTR_EXP  ] = ptr+3
    fakemol._bas[0,gto.PTR_COEFF] = ptr+4

    # buf for SPG_lmi upto l=0..3 and nl=3
    ngrids = np.prod(mesh)
    buf = np.empty((48,ngrids), dtype=np.complex128)
    def vppnl_by_k(kpt):
        Gk = Gv + kpt
        G_rad = lib.norm(Gk, axis=1)
        aokG = ft_ao(cell, Gv, kpt=kpt) * (1/cell.vol)**.5
        vppnl = 0
        for ia in range(cell.natm):
            symb = cell.atom_symbol(ia)
            if symb not in cell._pseudo:
                continue
            pp = cell._pseudo[symb]
            p1 = 0
            for l, proj in enumerate(pp[5:]):
                rl, nl, hl = proj
                if nl > 0:
                    fakemol._bas[0,gto.ANG_OF] = l
                    fakemol._env[ptr+3] = .5*rl**2
                    fakemol._env[ptr+4] = rl**(l+1.5)*np.pi**1.25
                    pYlm_part = fakemol.eval_gto('GTOval', Gk)

                    p0, p1 = p1, p1+nl*(l*2+1)
                    # pYlm is real, SI[ia] is complex
                    pYlm = np.ndarray((nl,l*2+1,ngrids), dtype=np.complex128, buffer=buf[p0:p1])
                    for k in range(nl):
                        qkl = pseudo.pp._qli(G_rad*rl, l, k)
                        pYlm[k] = pYlm_part.T * qkl
                    #:SPG_lmi = np.einsum('g,nmg->nmg', SI[ia].conj(), pYlm)
                    #:SPG_lm_aoG = np.einsum('nmg,gp->nmp', SPG_lmi, aokG)
                    #:tmp = np.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                    #:vppnl += np.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
            if p1 > 0:
                SPG_lmi = cp.asarray(buf[:p1])
                SPG_lmi *= SI[ia].conj()
                SPG_lm_aoGs = SPG_lmi.dot(aokG)
                p1 = 0
                for l, proj in enumerate(pp[5:]):
                    rl, nl, hl = proj
                    if nl > 0:
                        p0, p1 = p1, p1+nl*(l*2+1)
                        hl = cp.asarray(hl)
                        SPG_lm_aoG = SPG_lm_aoGs[p0:p1].reshape(nl,l*2+1,-1)
                        tmp = contract('ij,jmp->imp', hl, SPG_lm_aoG)
                        vppnl += contract('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
        return vppnl * (1./cell.vol)

    for k, kpt in enumerate(kpts):
        vppnl = vppnl_by_k(kpt)
        if is_zero(kpt):
            vpp[k] += cp.asarray(vppnl.real)
        else:
            vpp[k] += cp.asarray(vppnl)

    if is_single_kpt:
        vpp = vpp[0]
    log.timer('get_pp', *t0)
    return vpp

def to_primitive_bas(cell):
    '''Decontract the cell basis sets into primitive bases in super-mole'''
    bas_templates = {}
    prim_bas = []
    prim_env = cell._env.copy()
    # In kernel, ao_loc are access in Cartesian bases
    ao_loc = cell.ao_loc_nr(cart=True)
    ao_loc_in_cell0 = []
    aoslices = cell.aoslice_by_atom()
    for ia, (ib0, ib1) in enumerate(aoslices[:,:2]):
        ptr_coord = cell._atm[ia,PTR_COORD]
        key = tuple(cell._bas[ib0:ib1,PTR_COEFF])
        if key in bas_templates:
            bas_of_ia, local_ao_mapping = bas_templates[key]
            bas_of_ia = bas_of_ia.copy()
            bas_of_ia[:,PRIMBAS_COORD] = ptr_coord
        else:
            # Generate the template for decontracted basis
            local_ao_mapping = []
            off = 0
            bas_of_ia = []
            for shell in cell._bas[ib0:ib1]:
                l = shell[ANG_OF]
                nf = (l + 1) * (l + 2) // 2
                nctr = shell[NCTR_OF]
                nprim = shell[NPRIM_OF]
                pexp = shell[PTR_EXP]
                pcoeff = shell[PTR_COEFF]
                bs = np.empty((nprim*nctr, 4), dtype=np.int32)
                bs[:,PRIMBAS_ANG] = l
                bs[:,PRIMBAS_EXP] = _repeat(np.arange(pexp, pexp+nprim), nctr)
                bs[:,PRIMBAS_COEFF] = np.arange(pcoeff, pcoeff+nprim*nctr)
                bs[:,PRIMBAS_COORD] = ptr_coord
                bas_of_ia.append(bs)
                if l <= 1:
                    # sort_orbitals and unsort_orbitals do not transform the
                    # s and p orbitals. The special normalization coefficients
                    # should be applied into the contraction coefficients.
                    prim_env[pcoeff:pcoeff+nprim*nctr] *= ((2*l+1)/(4*np.pi))**.5
                # idx = [ao_loc[ib], ao_loc[ib], ... nprim terms ...,
                #        ao_loc[ib]+nf, ao_loc[ib]+nv, ... nprim terms ...]
                idx = np.repeat(np.arange(off, off+nf*nctr, nf), nprim)
                local_ao_mapping.append(idx)
                off += nf * nctr

            bas_of_ia = np.vstack(bas_of_ia)
            local_ao_mapping = np.hstack(local_ao_mapping)
            bas_templates[key] = (bas_of_ia, local_ao_mapping)

        prim_bas.append(bas_of_ia)
        ao_loc_in_cell0.append(ao_loc[ib0] + local_ao_mapping)

    prim_bas = np.asarray(np.vstack(prim_bas), dtype=np.int32)
    ao_loc_in_cell0 = np.asarray(np.hstack(ao_loc_in_cell0), dtype=np.int32)

    # Transform to super-mole
    Ls = cell.get_lattice_Ls()
    Ls = Ls[np.linalg.norm(Ls+0.5, axis=1).argsort()]
    nimgs = len(Ls)
    ptr_coords = prim_bas[:,PRIMBAS_COORD]
    bas_coords = cell._env[ptr_coords[:,None] + np.arange(3)]
    # Keep the unit cell at the beginning
    basLr = (bas_coords + Ls[1:,None]).reshape(-1, 3)
    dr = np.linalg.norm(bas_coords[:,None,:] - basLr, axis=2)
    # TODO: optimize the rcut for each basis
    idx = np.where(dr.min(axis=0) < cell.rcut)[0]
    _env = np.hstack([prim_env, basLr[idx].ravel()])
    extended_bas = _repeat(prim_bas, nimgs-1)[idx]
    extended_bas[:,PRIMBAS_COORD] = len(prim_env) + np.arange(idx.size) * 3
    supmol_bas = np.vstack([prim_bas, extended_bas])

    ao_loc_in_cell0 = np.append(
        ao_loc_in_cell0, np.hstack([ao_loc_in_cell0]*(nimgs-1))[idx])
    ao_loc_in_cell0 = np.asarray(ao_loc_in_cell0, dtype=np.int32)
    return supmol_bas, _env, ao_loc_in_cell0

def _repeat(a, repeats):
    '''repeats vertically'''
    ap = np.repeat(a[np.newaxis], repeats, axis=0)
    return ap.reshape(-1, *a.shape[1:])

@dataclass
class Task:
    mesh: tuple
    n_radius: int
    l: int
    shl_pair_idx: np.ndarray

def create_tasks(cell, prim_bas, supmol_bas, supmol_env):
    a = cell.lattice_vectors()
    assert abs(a - np.diag(a.diagonal())).max() < 1e-5, 'Must be orthogonal lattice'

    vol = cell.vol
    weight_penalty = vol
    precision = cell.precision / max(weight_penalty, 1)

    cell0_nprims = len(prim_bas)
    ls = cp.asarray(supmol_bas[:,PRIMBAS_ANG])
    es = cp.asarray(supmol_env[supmol_bas[:,PRIMBAS_EXP]])
    cs = cp.asarray(abs(supmol_env[supmol_bas[:,PRIMBAS_COEFF]]))
    norm = cs * ((2*ls+1)/(4*np.pi))**.5
    ptr_coords = supmol_bas[:,PRIMBAS_COORD]
    bas_coords = cp.asarray(supmol_env[ptr_coords[:,None] + np.arange(3)])

    # Estimate <cell0|supmol> overlap
    li = ls[:cell0_nprims,None]
    lj = ls[None,:]
    lij = li + lj
    aij = es[:cell0_nprims,None] + es
    fi = es[:cell0_nprims,None] / aij
    fj = es[None,:] / aij
    theta = es[:cell0_nprims,None] * fj
    rirj = bas_coords[:cell0_nprims,None,:] - bas_coords
    dr = cp.linalg.norm(rirj, axis=2)
    dri = fj * dr
    drj = fi * dr
    fac_dri = (li * .5/aij + dri**2) ** (li*.5)
    fac_drj = (lj * .5/aij + drj**2) ** (lj*.5)
    rad = cell.vol**(-1./3) * dr + 1
    surface = 4*np.pi * rad**2
    fl = cp.where(surface > 1, surface, 1)
    fac_norm = norm[:cell0_nprims,None]*norm * (np.pi/aij)**1.5
    ovlp = fac_norm * cp.exp(-theta*dr**2) * fac_dri * fac_drj * fl
    ovlp[ovlp > 1.] = 1.

    # Ecut estimation based on pyscf.pbc.gto.cell.estimate_ke_cutoff
    # Factors for Ecut estimation should be
    #     fac = norm[:,None]*norm * cp.exp(-theta*dr**2) * fac_dri * fac_drj * fl
    # where
    #     fac_dri = (li * .5/aij + dri**2 + Ecut/2/aij**2)**(li*.5)
    #             ~= (li * .5/aij + dri**2 + log(1./precision)/aij)**(li*.5)
    #     fac_drj = (lj * .5/aij + drj**2 + Ecut/2/aij**2)**(lj*.5)
    #             ~= (lj * .5/aij + drj**2 + log(1./precision)/aij)**(lj*.5)
    # Here, this fac is approximately derived from the overlap integral
    fac = ovlp / precision
    Ecut = cp.log(fac + 1.) * 2*aij

    # Estimate radius:
    # rho[r-Rp] = fl*norm[:cell0_nprims,None]*norm * exp(-theta*dr**2)
    #             * r**lij * exp(-aij*r**2)
    radius = 2.
    radius = (cp.log(ovlp/precision * radius**lij + 1.) / aij)**.5
    radius = (cp.log(ovlp/precision * radius**lij + 1.) / aij)**.5

    lmax = cell._bas[:,ANG_OF].max()
    assert lmax <= LMAX
    cell_len = a.diagonal()

    def sub_tasks_for_l(mesh, n_radius, mask):
        sub_tasks = []
        dh = float((cell_len / mesh).min())
        rcut_threshold = n_radius * dh
        for l in range(lmax*2+1):
            pair_idx = cp.where((mask & (l == lij)).ravel())[0]
            n_pairs = len(pair_idx)
            if n_pairs > 0:
                task = Task(mesh=mesh, n_radius=n_radius, l=l,
                            shl_pair_idx=cp.asarray(pair_idx, dtype=np.int32))
                sub_tasks.append(task)
                logger.debug1(cell, 'Add task: mesh=%s, n_radius=%d, rcut=%.5g, l=%d, n_pairs=%d',
                              mesh, n_radius, rcut_threshold, l, n_pairs)
        return sub_tasks

    remaining_mask = ovlp >= precision
    Gbase = 2*np.pi / cell_len
    ngrid_min = 512
    mesh = np.asarray(cell.mesh)
    assert all(mesh < 1000)
    tasks = []
    if 1:
        # partition tasks based on Ecut.
        # TODO: benchmark against the radius-based task Partitioning
        Ecut_threshold = ((mesh+1)//2 * Gbase).max()**2 / 4
        # Ecut/2 ~ mesh*.7 ~ radius*1.5
        for _ in range(20):
            if np.prod(mesh) <= ngrid_min:
                break
            dh = float((cell_len / mesh).min())
            mask = remaining_mask & (Ecut > Ecut_threshold)
            r_active = radius[mask]
            if r_active.size == 0:
                continue

            n_radius = int(np.ceil(r_active.max() / dh))
            n_radius = max(n_radius, 4)
            sub_tasks = sub_tasks_for_l(mesh, n_radius, mask)
            tasks.append(sub_tasks)

            remaining_mask &= Ecut <= Ecut_threshold
            # Determine mesh for the next task
            Gmax = (2*Ecut_threshold)**.5
            mesh = np.floor(Gmax/Gbase).astype(int) * 2 + 1

            Ecut_threshold /= 2

        if cp.any(remaining_mask):
            # TODO: Using a regular FFTDF task than the MG algorithm?
            dh = (cell_len / mesh).min()
            n_radius = int(np.ceil(radius[remaining_mask].max() / dh))
            n_radius = max(n_radius, 4)
            sub_tasks = sub_tasks_for_l(mesh, n_radius, remaining_mask)
            tasks.append(sub_tasks)
        return tasks

    # The derivation of n_radius
    # Ecut -> Gmax -> mesh -> resolution -> n_radius = radius/resolution
    # can be simplified to
    # n_radius ~= sqrt(2)/pi * Ecut**.5*radius
    # This n_radius for compact GTOs is larger than that for diffused GTOs.
    n_radius = int(2**.5/np.pi * (Ecut**.5*radius).max() * .8)
    # 8 aligned for ngrid_span in CUDA kernel
    n_radius = (n_radius + 3) // 4 * 4
    logger.debug1(cell, 'n_radius = %d', n_radius)

    for _ in range(20):
        if np.prod(mesh) <= ngrid_min:
            break
        dh = float((cell_len / mesh).min())
        # n_radius=16 can produce best performance. However, when cell.precision is
        # very tight, it might be insufficient to converge the value of GTO-pair and
        # the integration of GTO pair simultaneously.
        # Take several trials to adjust n_radius
        for inc in [0, 4, 8]:
            # The derivation of n_radius
            # Ecut -> Gmax -> mesh -> resolution -> n_radius = radius/resolution
            # can be simplified to
            # n_radius ~= sqrt(2)/pi * Ecut**.5*radius
            # This n_radius for compact GTOs is larger than that for diffused GTOs.
            # The mesh generation (from Ecut_max) may not changed during
            # iterations, leading to endless loop and empty tasks.
            # This issue can happen for tight cell.precision.
            # Increase the n_radius a little bit to handle this issue.
            rcut_threshold = dh * (n_radius + inc)
            mask = remaining_mask & (radius < rcut_threshold)
            sub_tasks = sub_tasks_for_l(mesh, n_radius+inc, mask)
            if sub_tasks:
                break
        else:
            raise RuntimeError(f'Failed to create multigrid task for cell.precision={cell.precision}')

        tasks.append(sub_tasks)
        remaining_mask &= radius >= rcut_threshold
        Ecut_remaining = Ecut[remaining_mask]
        if Ecut_remaining.size > 0:
            Ecut_max = float(Ecut_remaining.max())
            Gmax = (2*Ecut_max)**.5
            mesh = np.floor(Gmax/Gbase).astype(int) * 2 + 1
        else:
            break

    if cp.any(remaining_mask):
        # TODO: Using a regular FFTDF task than the MG algorithm?
        dh = (cell_len / mesh).min()
        n_radius = int(np.ceil(radius[remaining_mask].max() / dh))
        n_radius = max(n_radius, 4)
        sub_tasks = sub_tasks_for_l(mesh, n_radius, remaining_mask)
        tasks.append(sub_tasks)
    return tasks

def init_constant(cell):
    err = libmgrid.MG_init_constant(ctypes.c_int(SHM_SIZE))
    if err != 0:
        raise RuntimeError('CUDA kernel initialization')

def _take_4d(a, mesh):
    gx = np.fft.fftfreq(mesh[0], 1./mesh[0]).astype(np.int32)
    gy = np.fft.fftfreq(mesh[1], 1./mesh[1]).astype(np.int32)
    gz = np.fft.fftfreq(mesh[2], 1./mesh[2]).astype(np.int32)
    return a[:,gx[:,None,None],gy[:,None],gz]

def _takebak_4d(out, a, mesh):
    gx = np.fft.fftfreq(mesh[0], 1./mesh[0]).astype(np.int32)
    gy = np.fft.fftfreq(mesh[1], 1./mesh[1]).astype(np.int32)
    gz = np.fft.fftfreq(mesh[2], 1./mesh[2]).astype(np.int32)
    out[:,gx[:,None,None],gy[:,None],gz] += a.reshape(-1, *mesh)
    return out

class MGridEnvVars(ctypes.Structure):
    _fields_ = [
        ('nbas_i', ctypes.c_int),
        ('nbas_j', ctypes.c_int),
        ('nao', ctypes.c_int),
        ('bas', ctypes.c_void_p),
        ('env', ctypes.c_void_p),
        ('ao_loc', ctypes.c_void_p),
        ('lattice_params', ctypes.c_void_p),
    ]


class MultiGridNumInt(lib.StreamObject, numint.LibXCMixin):
    def __init__(self, cell):
        self.cell = cell
        self.mesh = cell.mesh
        # ao_loc_in_cell0 is the address of Cartesian AO in cell-0 for each
        # primitive GTOs in the super-mole.
        supmol_bas, supmol_env, ao_loc_in_cell0 = to_primitive_bas(cell)
        self.supmol_bas = supmol_bas
        self.supmol_env = supmol_env
        self.ao_loc_in_cell0 = ao_loc_in_cell0
        # Number of primitive shells
        self.primitive_nbas = cell._bas[:,NPRIM_OF].dot(cell._bas[:,NCTR_OF])
        # A list of integral meshgrids for each task
        #self.tasks = multigrid_tasks(cell)
        prim_bas = supmol_bas[:self.primitive_nbas]
        self.tasks = create_tasks(cell, prim_bas, supmol_bas, supmol_env)
        logger.debug(cell, 'Multigrid ntasks %s', len(self.tasks))

    def reset(self, cell=None):
        if cell is not None:
            self.cell = cell
        self.tasks = None
        return self

    def sort_orbitals(self, mat):
        ''' Transform bases of a matrix into Cartesian bases
        '''
        cell = self.cell
        if cell.cart:
            return mat
        c2s = block_diag([cart2sph_by_l(l)
                          for l, nc in zip(cell._bas[:,ANG_OF], cell._bas[:,NCTR_OF])
                          for _ in range(nc)])
        return sandwich_dot(mat, c2s.T)

    def unsort_orbitals(self, mat):
        ''' Transform bases of a matrix into original AOs
        '''
        cell = self.cell
        if cell.cart:
            return mat
        c2s = block_diag([cart2sph_by_l(l)
                          for l, nc in zip(cell._bas[:,ANG_OF], cell._bas[:,NCTR_OF])
                          for _ in range(nc)])
        return sandwich_dot(mat, c2s)

    def get_j(self, dm, hermi=1, kpts=None, kpts_band=None,
              omega=None, exxdiv='ewald'):
        if kpts is not None:
            raise NotImplementedError
        vj = get_j_kpts(self, dm, hermi, kpts, kpts_band)
        return vj

    get_nuc = get_nuc
    get_pp = get_pp

    get_rho = get_rho
    nr_rks = nr_rks
    nr_uks = nr_uks
    get_vxc = nr_vxc = NotImplemented #numint_cpu.KNumInt.nr_vxc

    eval_xc_eff = numint.eval_xc_eff
    _init_xcfuns = numint.NumInt._init_xcfuns

    nr_rks_fxc = NotImplemented
    nr_uks_fxc = NotImplemented
    nr_rks_fxc_st = NotImplemented
    cache_xc_kernel  = NotImplemented
    cache_xc_kernel1 = NotImplemented

    to_gpu = utils.to_gpu
    device = utils.device

    def to_cpu(self):
        raise RuntimeError('Not available')
