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

import ctypes
import numpy as np
import cupy as cp
import scipy.linalg
from pyscf.gto import ATOM_OF, ANG_OF, NPRIM_OF, NCTR_OF, PTR_EXP, PTR_COEFF, PTR_COORD
from pyscf.pbc.tools.pbc import mesh_to_cutoff, cutoff_to_mesh
from pyscf.pbc.df.df_jk import _format_kpts_band
from gpu4pyscf.pbc import tools
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import utils
from gpu4pyscf.lib.cupy_helper import load_library, tag_array, sandwich_dot, block_diag
from gpu4pyscf.gto.mole import cart2sph_by_l, extract_pgto_params
from gpu4pyscf.pbc.df.fft_jk import _format_dms, _format_jks
from gpu4pyscf.__config__ import shm_size
from gpu4pyscf.__config__ import props as gpu_specs

__all__ = ['MultiGridFFTDF']

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

def eval_mat(cell, weights, shls_slice=None, comp=1, hermi=0,
             xctype='LDA', kpts=None, mesh=None, offset=None, submesh=None):
    raise NotImplementedError

def get_j_kpts(mydf, dm_kpts, hermi=1, kpts=None, kpts_band=None):
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

    cell = mydf.cell
    dm_kpts = cp.asarray(dm_kpts)
    rhoG = _eval_rhoG(mydf, dm_kpts, hermi, kpts, deriv=0)
    coulG = tools.get_coulG(cell, mesh=cell.mesh)
    #:vG = np.einsum('ng,g->ng', rhoG[:,0], coulG)
    vG = rhoG[:,0]
    vG *= coulG

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    vj_kpts = _get_j_pass2(mydf, vG, hermi, kpts_band)
    return _format_jks(vj_kpts, dm_kpts, input_band, kpts)


def _eval_rhoG(mydf, dm_kpts, hermi=1, kpts=None, deriv=0):
    cell = mydf.cell
    log = logger.new_logger(cell)
    t0 = log.init_timer()

    dm_kpts = cp.asarray(dm_kpts, order='C')
    dm_kpts = mydf.sort_orbitals(dm_kpts)
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    lmax = cell._bas[:,ANG_OF].max()
    assert lmax <= LMAX

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
    supmol_bas = cp.asarray(mydf.supmol_bas, dtype=np.int32)
    supmol_env = cp.asarray(mydf.supmol_env)
    tasks = mydf.tasks
    ao_loc_in_cell0 = cp.asarray(mydf.ao_loc_in_cell0, dtype=np.int32)
    mg_envs = MGridEnvVars(
        mydf.primitive_nbas, len(supmol_bas), nao, supmol_bas.data.ptr,
        supmol_env.data.ptr, ao_loc_in_cell0.data.ptr, lattice_params.data.ptr)
    mg_envs._env_ref_holder = (supmol_bas, supmol_env, ao_loc_in_cell0, lattice_params)
    sp_groups = group_shl_pairs(cell, supmol_bas[:mydf.primitive_nbas],
                                mydf.supmol_bas, mydf.supmol_env, tasks)
    workers = gpu_specs['multiProcessorCount']
    nf3 = (lmax*2+1)*(lmax*2+2)*(lmax*2+3)//6
    ngrid_span = max(task.n_radius*2 for task in tasks)
    cache_size = ((lmax*2+1)*ngrid_span*3 + nf3 + 6) * 32
    pool = cp.empty((workers, cache_size))

    init_constant(cell)
    kern = libmgrid.MG_eval_rho_orth
    assert rhodim == 1
    rhoG = None

    for i, task in enumerate(tasks):
        mesh = task.mesh
        n_radius = task.n_radius
        ngrids = np.prod(mesh)
        rhoR = cp.empty((nset, *mesh))
        for iset in range(nset):
            if ngrids < SHARED_RHO_SIZE:
                rho_local = cp.zeros((workers*8, *mesh))
            else:
                rho_local = cp.zeros(mesh)
            for l in range(lmax*2+1):
                if (i, l) not in sp_groups: continue
                pair_idx = sp_groups[i,l]
                kern(ctypes.cast(rho_local.data.ptr, ctypes.c_void_p),
                     ctypes.cast(dms[iset].data.ptr, ctypes.c_void_p),
                     mg_envs, ctypes.c_int(l), ctypes.c_int(n_radius),
                     (ctypes.c_int*3)(*mesh),
                     ctypes.c_uint32(len(pair_idx)),
                     ctypes.cast(pair_idx.data.ptr, ctypes.c_void_p),
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
        Gv = cp.asarray(cell.get_Gv(mydf.mesh))
        rhoG1 = cp.einsum('np,px->nxp', 1j*rhoG[:,0], Gv)
        rhoG = cp.concatenate([rhoG, rhoG1], axis=1)
    log.timer_debug1('eval_rhoG', *t0)
    return rhoG

def _get_j_pass2(mydf, vG, hermi=1, kpts=None, verbose=None):
    cell = mydf.cell
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()
    nkpts = len(kpts)
    assert nkpts == 1, 'gamma point only'
    nao = cell.nao_nr(cart=True)
    nset = vG.shape[0]
    lmax = cell._bas[:,ANG_OF].max()
    assert lmax <= LMAX

    tasks = mydf.tasks
    mesh_largest = tasks[0].mesh
    vG = vG.reshape(-1, *mesh_largest)

    a = cell.lattice_vectors()
    assert abs(a - np.diag(a.diagonal())).max() < 1e-5, 'Must be orthogonal lattice'
    lattice_params = cp.asarray(a.diagonal(), order='C')
    supmol_bas = cp.asarray(mydf.supmol_bas, dtype=np.int32)
    supmol_env = cp.asarray(mydf.supmol_env)
    ao_loc_in_cell0 = cp.asarray(mydf.ao_loc_in_cell0, dtype=np.int32)
    mg_envs = MGridEnvVars(
        mydf.primitive_nbas, len(supmol_bas), nao, supmol_bas.data.ptr,
        supmol_env.data.ptr, ao_loc_in_cell0.data.ptr, lattice_params.data.ptr)
    mg_envs._env_ref_holder = (supmol_bas, supmol_env, ao_loc_in_cell0, lattice_params)
    sp_groups = group_shl_pairs(cell, supmol_bas[:mydf.primitive_nbas],
                                mydf.supmol_bas, mydf.supmol_env, tasks)

    workers = gpu_specs['multiProcessorCount']
    nf2 = (lmax*2+1)*(lmax*2+2)//2
    ngrid_span = max(task.n_radius*2 for task in tasks)
    cache_size = ((lmax*2+1)*ngrid_span*3 + nf2*ngrid_span + 6) * 32
    pool = cp.empty((workers, cache_size))

    init_constant(cell)
    kern = libmgrid.MG_eval_mat_lda_orth
    # TODO: might be complex array when tddft amplitudes are complex
    vj = cp.zeros((nset,nao,nao))

    for i, task in enumerate(tasks):
        mesh = task.mesh
        n_radius = task.n_radius
        ngrids = np.prod(mesh)
        sub_vG = _take_4d(vG, mesh).reshape(nset,ngrids)
        v_rs = tools.ifft(sub_vG, mesh).reshape(nset,ngrids)
        if abs(v_rs.imag).max() > 1e-8:
            raise RuntimeError('Complex values in potential. mesh might be insufficient')

        vR = cp.asarray(v_rs.real, order='C')
        for iset in range(nset):
            for l in range(lmax*2+1):
                pair_idx = sp_groups[i,l]
                kern(ctypes.cast(vj[iset].data.ptr, ctypes.c_void_p),
                     ctypes.cast(vR[iset].data.ptr, ctypes.c_void_p),
                     mg_envs, ctypes.c_int(l), ctypes.c_int(n_radius),
                     (ctypes.c_int*3)(*mesh),
                     ctypes.c_uint32(len(pair_idx)),
                     ctypes.cast(pair_idx.data.ptr, ctypes.c_void_p),
                     ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                     ctypes.c_int(workers))
    # TODO: for diffused basis functions lower than minimal Ecut, compute the
    # vj using normal FFTDF code
    vj = mydf.unsort_orbitals(vj)
    vj = vj.reshape(nset,nkpts,nao,nao)
    log.timer_debug1('get_j pass2', *t0)
    return vj

def _get_gga_pass2(mydf, vG, hermi=1, kpts=np.zeros((1,3)), verbose=None):
    raise NotImplementedError

def nr_rks(mydf, xc_code, dm_kpts, hermi=1, kpts=None,
           kpts_band=None, with_j=False, return_j=False, verbose=None):
    '''Compute the XC energy and RKS XC matrix at sampled k-points.
    multigrid version of function pbc.dft.numint.nr_rks.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpts_band : ``(3,)`` ndarray or ``(*,3)`` ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        exc : XC energy
        nelec : number of electrons obtained from the numerical integration
        veff : (nkpts, nao, nao) ndarray
            or list of veff if the input dm_kpts is a list of DMs
        vj : (nkpts, nao, nao) ndarray
            or list of vj if the input dm_kpts is a list of DMs
    '''
    assert kpts is None
    kpts = np.zeros((1, 3))

    log = logger.new_logger(mydf, verbose)
    cell = mydf.cell
    dm_kpts = cp.asarray(dm_kpts, order='C')
    dm_kpts = mydf.sort_orbitals(dm_kpts)
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band

    ni = mydf._numint
    xctype = ni._xc_type(xc_code)

    if xctype == 'LDA':
        deriv = 0
    elif xctype == 'GGA':
        deriv = 1
    elif xctype == 'MGGA':
        deriv = 1
        raise NotImplementedError
    rhoG = _eval_rhoG(mydf, dm_kpts, hermi, kpts, deriv)

    mesh = mydf.mesh
    ngrids = np.prod(mesh)
    coulG = tools.get_coulG(cell, mesh=mesh)
    vG = rhoG[:,0] * coulG
    ecoul = .5 * cp.einsum('ng,ng->n', rhoG[:,0].real, vG.real)
    ecoul+= .5 * cp.einsum('ng,ng->n', rhoG[:,0].imag, vG.imag)
    ecoul /= cell.vol
    log.debug('Multigrid Coulomb energy %s', ecoul)

    weight = cell.vol / ngrids
    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    rhoR = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (1./weight)
    rhoR = cp.asarray(rhoR.reshape(nset,-1,ngrids), order='C')
    nelec = rhoR[:,0].sum(axis=1) * weight

    wv_freq = []
    excsum = np.zeros(nset)
    for i in range(nset):
        if xctype == 'LDA':
            exc, vxc = ni.eval_xc_eff(xc_code, rhoR[i,0], deriv=1, xctype=xctype)[:2]
        else:
            exc, vxc = ni.eval_xc_eff(xc_code, rhoR[i], deriv=1, xctype=xctype)[:2]
        excsum[i] += float((rhoR[i,0]*exc).sum()) * weight
        wv = weight * vxc
        wv_freq.append(tools.fft(wv, mesh))
    wv_freq = cp.asarray(wv_freq).reshape(nset,-1,*mesh)
    rhoR = rhoG = None

    if nset == 1:
        ecoul = ecoul[0]
        nelec = nelec[0]
        excsum = excsum[0]
    log.debug('Multigrid exc %s  nelec %s', excsum, nelec)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    if xctype == 'LDA':
        if with_j:
            wv_freq[:,0] += vG.reshape(nset,*mesh)
        veff = _get_j_pass2(mydf, wv_freq, hermi, kpts_band, verbose=log)
    elif xctype == 'GGA':
        if with_j:
            wv_freq[:,0] += vG.reshape(nset,*mesh)
        # *.5 because v+v.T is always called in _get_gga_pass2
        wv_freq[:,0] *= .5
        veff = _get_gga_pass2(mydf, wv_freq, hermi, kpts_band, verbose=log)
    veff = _format_jks(veff, dm_kpts, input_band, kpts)

    if return_j:
        vj = _get_j_pass2(mydf, vG, hermi, kpts_band, verbose=log)
        vj = _format_jks(veff, dm_kpts, input_band, kpts)
    else:
        vj = None

    shape = list(dm_kpts.shape)
    if len(shape) == 3 and shape[0] != kpts_band.shape[0]:
        shape[0] = kpts_band.shape[0]
    veff = veff.reshape(shape)
    veff = tag_array(veff, ecoul=ecoul, exc=excsum, vj=vj, vk=None)
    return nelec, excsum, veff

# Note nr_uks handles only one set of KUKS density matrices (alpha, beta) in
# each call (nr_rks supports multiple sets of KRKS density matrices)
def nr_uks(mydf, xc_code, dm_kpts, hermi=1, kpts=None,
           kpts_band=None, with_j=False, return_j=False, verbose=None):
    raise NotImplementedError

def get_rho(mydf, dm, kpts=None):
    '''Density in real space
    '''
    assert kpts is None
    kpts = np.zeros((1, 3))

    cell = mydf.cell
    hermi = 1
    rhoG = _eval_rhoG(mydf, cp.asarray(dm), hermi, kpts, deriv=0)

    mesh = mydf.mesh
    ngrids = np.prod(mesh)
    weight = cell.vol / ngrids
    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    rhoR = tools.ifft(rhoG.reshape(ngrids), mesh).real * (1./weight)
    return rhoR

def _to_primitive_bas(cell):
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

def group_shl_pairs(cell, prim_bas, supmol_bas, supmol_env, tasks):
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

    # TODO: rewritten in C
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

    #FIXME: Ecut estimation based on pyscf.pbc.gto.cell.estimate_ke_cutoff
    #scale = norm[:,None]*norm * cp.exp(-theta*dr**2) * fl
    #scale * 32*np.pi**2*(2*np.pi)**1.5 * norm[:,None]*norm / aij**(lij+.5) / precision
    fac = ovlp / precision
    Ecut = 20.
    Ecut = cp.log(fac * (Ecut*2)**(lij-.5) + 1.) * 2*aij
    Ecut = cp.log(fac * (Ecut*2)**(lij-.5) + 1.) * 2*aij
    Ecut[ovlp < cell.precision] = -1

    task_Ecuts = [task.Ecut for task in tasks]
    task_Ecuts[0] = Ecut.max()
    task_Ecuts.append(0.)

    ntasks = len(tasks)
    lmax = cell._bas[:,ANG_OF].max()
    shl_pairs_groups = {}
    for i in range(ntasks): # Exclude the last
        # task_Ecuts[i:i+2] defines the window. Shell pairs within this window
        # uses the same task.n_radius and mesh in kernel.
        # TODO: adjust n_radius and mesh for high angular l.
        E_mask = Ecut <= task_Ecuts[i]
        E_mask &= Ecut > task_Ecuts[i+1]
        for l in range(lmax*2+1):
            mask = l == lij
            mask &= E_mask
            pair_idx = cp.asarray(cp.where(mask.ravel())[0], dtype=np.int32)
            shl_pairs_groups[i,l] = pair_idx
    return shl_pairs_groups

def multigrid_tasks(cell):
    a = cell.lattice_vectors()
    assert abs(a - np.diag(a.diagonal())).max() < 1e-5, 'Must be orthogonal lattice'
    cell_len = a.diagonal()
    Gbase = 2*np.pi / cell_len
    penalty = 1
    precision = cell.precision * penalty

    # * Estimate the proper radius and Ecut boundaries in each task. Given mesh
    # in each task, a resolution can be obtained to determine the radius and
    # Ecut boundaries. The largest radius in each task ~= n_radius * resolution.
    # This means that the numerical integration with n_radius*2 grids for the most
    # diffused orbital-product is sufficient to produce the required precision.
    # The largest Ecut in each task ~= cutoff_to_mesh(mesh), which is sufficient
    # to produce the required precision for the most compact orbital-product.
    # * Approximately, n_radius = np.log(1/precision) * 2/np.pi would produce a
    # balanced n_radius. The "balance" means that given the pair-GTO exponent, a
    # Ecut can be determined, which corresponds to a mesh grid in real space.
    # The resolution for this mesh and n_radius can determine a radius, at the
    # boundary of which, the pair-GTO can converge to the required precision.
    # * We choose a n_radius ~50% larger than the balanced one. By doing so, a
    # window of Ecut can be obtained. In this Ecut window, the mesh is
    # sufficient to converge all GTO pairs, as the mesh is determined from the
    # upper bound of the Ecut window. The n_radius is sufficient to converge the
    # most diffused GTOs, as the lower bound of the Ecut window is estimated
    # based those diffused GTO pairs.
    n_radius = int(np.log(1./precision))

    # for diffused basis functions lower than minimal Ecut, skip the multi-grid
    # computation. The regular FFTDF algorithm in this case may be more accurate
    # and efficient.
    Ecut_min = ((Gbase * 4)**2 / 2).min()
    mesh = cell.mesh
    Ecut = mesh_to_cutoff(a, mesh).max()
    tasks = []
    for _ in range(20):
        if Ecut < Ecut_min:
            break
        tasks.append(_Task(Ecut=Ecut, mesh=mesh, n_radius=n_radius, algorithm='MG'))
        dh = (cell_len / mesh).min()
        radius = dh * n_radius
        aij = np.log(1./precision) / radius**2
        Ecut = np.log(1./precision) * (2*aij)
        Gmax = (2*Ecut)**.5
        mesh = np.floor(Gmax / Gbase).astype(int) * 2 + 1
    else:
        raise RuntimeError(f'failed to generate mesh tasks for precision {cell.precision}.')
    # TODO: Attach a regular FFTDF task?
    #tasks.append(_Task(Ecut=Ecut, mesh=mesh, n_radius=None, algorithm='FFTDF'))

    # estimate n_radius for the most diffused GTO pairs
    es, _ = extract_pgto_params(cell)
    aij = es.min() * 2
    radius = (np.log(1./precision) / aij)**.5
    dh = (cell_len / mesh).min()
    tasks.append(_Task(Ecut=Ecut, mesh=mesh, n_radius=n_radius, algorithm='MG'))
    return tasks

from dataclasses import dataclass
@dataclass
class _Task:
    Ecut: float
    mesh: tuple
    n_radius: int
    algorithm: str
    shl_pair_idx: np.ndarray = None

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
        ('nbas_i', ctypes.c_uint16),
        ('nbas_j', ctypes.c_uint16),
        ('nao', ctypes.c_int),
        ('bas', ctypes.c_void_p),
        ('env', ctypes.c_void_p),
        ('ao_loc', ctypes.c_void_p),
        ('lattice_params', ctypes.c_void_p),
    ]


class MultiGridFFTDF:
    def __init__(self, cell):
        self.cell = cell
        self.mesh = cell.mesh
        # ao_loc_in_cell0 is the address of Cartesian AO in cell-0 for each
        # primitive GTOs in the super-mole.
        self.supmol_bas, self.supmol_env, self.ao_loc_in_cell0 = \
                _to_primitive_bas(cell)
        # Number of primitive shells
        self.primitive_nbas = cell._bas[:,NPRIM_OF].dot(cell._bas[:,NCTR_OF])
        # A list of integral meshgrids for each task
        self.tasks = multigrid_tasks(cell)
        logger.debug(cell, 'Multigrid ntasks %s', len(self.tasks))

    def reset(self, cell=None):
        if cell is not None:
            self.cell = cell
        self.tasks = None
        return self

    def get_j(self, dm, hermi=1, kpts=None, kpts_band=None,
              omega=None, exxdiv='ewald'):
        if kpts is not None:
            raise NotImplementedError

        vj = get_j_kpts(self, dm, hermi, kpts, kpts_band)
        return vj

    def get_veff(self, dm, xc):
        '''Computing XC matrix and Coulomb interactions'''
        raise NotImplementedError

    get_rho = get_rho

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
