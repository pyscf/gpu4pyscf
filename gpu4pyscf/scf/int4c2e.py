# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
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
Compute four-center ERIs
'''

import ctypes
import numpy as np
import cupy as cp
import math
from pyscf import gto, lib
from gpu4pyscf.__config__ import shm_size
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import fill_symmetric, contract
from gpu4pyscf.scf.jk import libvhf_rys, _VHFOpt, QUEUE_DEPTH, LMAX
from gpu4pyscf.gto.mole import SortedMole

def get_int4c2e(mol, vhfopt=None, direct_scf_tol=1e-13, aosym=True, omega=None):
    if vhfopt is None:
        mol = SortedMole.from_mol(mol, decontract=False)
        vhfopt = _VHFOpt(mol).build()

    mo = vhfopt.sorted_mol.ctr_coeff
    return _ao2mo_general(vhfopt, [mo]*4, omega)

def get_int4c2e_ovov(mol, orbo, orbv, vhfopt=None, direct_scf_tol=1e-13, stream=None, omega=None):
    '''
    Generate 2-electron integrals (ov|ov) on GPU
    '''
    if vhfopt is None:
        mol = SortedMole.from_mol(mol, decontract=False)
        vhfopt = _VHFOpt(mol).build()

    orbo = vhfopt.sorted_mol.apply_C_dot(orbo)
    orbv = vhfopt.sorted_mol.apply_C_dot(orbv)
    return _ao2mo_general(vhfopt, [orbo,orbv,orbo,orbv], omega)

def _ao2mo_general(vhfopt, mo_coeffs, omega=None):
    assert isinstance(vhfopt, _VHFOpt)
    mol = vhfopt.sorted_mol
    log = logger.new_logger(mol)
    t0 = log.init_timer()

    if omega is None:
        omega = mol.omega

    ao_loc = mol.ao_loc_nr()
    nao = int(ao_loc[-1])
    ao_loc = cp.asarray(ao_loc)

    assert mo_coeffs[0].shape[0] == nao
    nmo0, nmo1, nmo2, nmo3 = [x.shape[1] for x in mo_coeffs]
    swap_2e = min(nmo0, nmo1) < min(nmo2, nmo3)
    idx = [0, 1, 2, 3]
    if swap_2e:
        idx = [2, 3, 0, 1]
        nmo0, nmo1, nmo2, nmo3 = nmo2, nmo3, nmo0, nmo1
    swap_ij = nmo0 > nmo1
    swap_kl = nmo2 > nmo3
    if swap_ij:
        idx = [idx[1], idx[0], idx[2], idx[3]]
    if swap_kl:
        idx = [idx[0], idx[1], idx[3], idx[2]]
    mo_coeffs = [mo_coeffs[i] for i in idx]

    uniq_l = mol.uniq_l_ctr[:,0]
    nf = (uniq_l + 1) * (uniq_l + 2) // 2
    carts = [cp.arange(n) for n in nf]

    bas_pair_cache = {k: [cp.asarray(x) for x in v]
                      for k, v in vhfopt.bas_pair_cache.items()}
    bas_ij_idx = []
    pair_loc = []
    ao_pair_addresses = []
    for i, j in bas_pair_cache:
        bas_ij = bas_pair_cache[i, j][0]
        bas_ij_idx.append(bas_ij)
        pair_loc.append(cp.arange(len(bas_ij)+0, dtype=np.int32) * (nf[i] * nf[j]))
        ish, jsh = divmod(bas_ij, mol.nbas)
        iaddr = ao_loc[ish,None] + carts[i]
        jaddr = ao_loc[jsh,None] + carts[j]
        ao_pair_addresses.append((iaddr[:,None,:] * nao + jaddr[:,:,None]).ravel())
    bas_ij_idx = cp.hstack(bas_ij_idx, dtype=np.int32)
    ao_pair_addresses = cp.hstack(ao_pair_addresses, dtype=np.int32)
    nao_pairs = len(ao_pair_addresses)

    log_cutoff = math.log(vhfopt.direct_scf_tol)

    workers = gpu_specs['multiProcessorCount']
    # An additional integer to count for the proccessed pair_ijs
    pool = cp.empty(workers*QUEUE_DEPTH+1, dtype=np.int32)

    l_ctr_bas_loc = np.append(0, np.cumsum(mol.l_ctr_counts))
    l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
    assert all(uniq_l <= LMAX)

    rys_envs = vhfopt.rys_envs
    kern = libvhf_rys.RYS_fill_int4c2e

    kl0 = kl1 = 0
    eri_mo = cp.empty((nao_pairs, nmo2, nmo3))
    for kl_id, (k, l) in enumerate(bas_pair_cache):
        pair_kl_mapping, q_cond_kl, s_cond_kl = bas_pair_cache[k,l]
        npairs_kl = pair_kl_mapping.size
        if npairs_kl == 0:
            continue
        kl0, kl1 = kl1, kl1 + npairs_kl * nf[k] * nf[l]
        eri = cp.empty((nao_pairs, kl1-kl0))
        ij0 = ij1 = 0
        for ij_id, (i, j) in enumerate(bas_pair_cache):
            shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
            pair_ij_mapping, q_cond_ij, s_cond_ij = bas_pair_cache[i,j]
            npairs_ij = pair_ij_mapping.size
            if npairs_ij == 0:
                continue
            ij0, ij1 = ij1, ij1 + npairs_ij * nf[i] * nf[j]
            err = kern(
                ctypes.cast(eri[ij0:ij1].data.ptr, ctypes.c_void_p),
                ctypes.c_int(kl1 - kl0),
                ctypes.c_double(omega),
                ctypes.byref(rys_envs), (ctypes.c_int*8)(*shls_slice),
                ctypes.c_int(shm_size),
                ctypes.c_int(npairs_ij), ctypes.c_int(npairs_kl),
                ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                ctypes.cast(pair_kl_mapping.data.ptr, ctypes.c_void_p),
                ctypes.cast(q_cond_ij.data.ptr, ctypes.c_void_p),
                ctypes.cast(q_cond_kl.data.ptr, ctypes.c_void_p),
                ctypes.cast(pair_loc[ij_id].data.ptr, ctypes.c_void_p),
                ctypes.cast(pair_loc[kl_id].data.ptr, ctypes.c_void_p),
                ctypes.c_float(log_cutoff),
                ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                mol._bas.ctypes)
            if err != 0:
                llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                raise RuntimeError(f'fill_int4c2e kernel for {llll} failed')
        eri = fill_symmetric(eri, ao_pair_addresses, nao)
        eri = contract('pqr,pi->iqr', eri, mo_coeffs[2])
        contract('iqr,qj->rij', eri, mo_coeffs[3], out=eri_mo[kl0:kl1])
    eri_mo = fill_symmetric(eri_mo.reshape(nao_pairs,-1), ao_pair_addresses, nao)
    eri_mo = eri_mo.reshape(nao,nao,nmo2,nmo3)
    eri_mo = contract('pqkl,pi->iqkl', eri_mo, mo_coeffs[0])
    eri_mo = contract('iqkl,qj->ijkl', eri_mo, mo_coeffs[1])
    if swap_ij:
        eri_mo = eri_mo.transpose(1,0,2,3)
    if swap_kl:
        eri_mo = eri_mo.transpose(0,1,3,2)
    if swap_2e:
        eri_mo = eri_mo.transpose(2,3,0,1)
    log.timer_debug1('AO2MO transform 4c2e', *t0)
    return eri_mo

def get_int4c2e_jk(mol, dm, vhfopt=None, direct_scf_tol=1e-13, with_k=True, omega=None, stream=None):
    raise DeprecationWarning

def loop_int4c2e_general(intopt, ip_type='', direct_scf_tol=1e-13, omega=None, stream=None):
    raise DeprecationWarning

class BasisProdCache(ctypes.Structure):
    pass
