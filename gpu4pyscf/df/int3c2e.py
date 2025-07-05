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

import itertools
from concurrent.futures import ThreadPoolExecutor
import ctypes
import numpy as np
import cupy
from pyscf import gto, df, lib
from pyscf.scf import _vhf
from gpu4pyscf.scf.int4c2e import BasisProdCache, libgvhf, libgint
from gpu4pyscf.lib.cupy_helper import (block_c2s_diag, cart2sph, contract, get_avail_mem,
                                       reduce_to_device, copy_array, transpose_sum)
from gpu4pyscf.lib import logger
from gpu4pyscf.gto.mole import basis_seg_contraction
from gpu4pyscf.__config__ import num_devices, _streams

LMAX_ON_GPU = 8
FREE_CUPY_CACHE = True
STACK_SIZE_PER_THREAD = 8192 * 4
BLKSIZE = 256
NROOT_ON_GPU = 7

def make_fake_mol():
    '''
    fake mol for pairing with auxiliary basis
    '''
    fakemol = gto.mole.Mole()
    fakemol._atm = np.zeros((1,gto.ATM_SLOTS), dtype=np.int32)
    fakemol._atm[0][[0,1,2,3]] = np.array([2,20,1,23])

    ptr = gto.mole.PTR_ENV_START
    fakemol._bas = np.zeros((1,gto.BAS_SLOTS), dtype=np.int32)
    fakemol._bas[0,gto.NPRIM_OF ] = 1
    fakemol._bas[0,gto.NCTR_OF  ] = 1
    fakemol._bas[0,gto.PTR_EXP  ] = ptr+4
    fakemol._bas[0,gto.PTR_COEFF] = ptr+5

    fakemol._env = np.zeros(ptr+6)
    ptr_coeff = fakemol._bas[0,gto.PTR_COEFF]
    ptr_exp = fakemol._bas[0,gto.PTR_EXP]
    '''
    due to the common factor of normalization
    https://github.com/sunqm/libcint/blob/be610546b935049d0cf65c1099244d45b2ff4e5e/src/g1e.c
    '''
    fakemol._env[ptr_coeff] = 1.0/0.282094791773878143
    fakemol._env[ptr_exp] = 0.0
    fakemol._built = True
    return fakemol

class VHFOpt(_vhf.VHFOpt):
    def __init__(self, mol, auxmol, intor, prescreen='CVHFnoscreen',
                 qcondname='CVHFsetnr_direct_scf', dmcondname=None):
        self.mol = mol              # original mol
        self.auxmol = auxmol        # original auxiliary mol
        self._sorted_mol = None     # sorted mol
        self._sorted_auxmol = None  # sorted auxilary mol

        self._ao_idx = None
        self._aux_ao_idx = None

        self._intor = intor
        self._prescreen = prescreen
        self._qcondname = qcondname
        self._dmcondname = dmcondname

        self.cart_ao_loc = []
        self.cart_aux_loc = []
        self.sph_ao_loc = []
        self.sph_aux_loc = []

        self.angular = None
        self.aux_angular = None

        self.cp_idx = None
        self.cp_jdx = None

        self.log_qs = None
        self.aux_log_qs = None

    init_cvhf_direct = _vhf.VHFOpt.init_cvhf_direct

    def clear(self):
        _vhf.VHFOpt.__del__(self)
        for n, bpcache in self._bpcache.items():
            libgvhf.GINTdel_basis_prod(ctypes.byref(bpcache))
        return self

    def __del__(self):
        try:
            self.clear()
        except AttributeError:
            pass

    def build(self, cutoff=1e-14, group_size=None, group_size_aux=None, 
              diag_block_with_triu=False, aosym=False, verbose=None):
        '''
        int3c2e is based on int2e with (ao,ao|aux,1)
        a tot_mol is created with concatenating [mol, fake_mol, aux_mol]
        we will pair (ao,ao) and (aux,1) separately.
        '''
        _mol = self.mol
        _auxmol = self.auxmol

        mol = basis_seg_contraction(_mol, allow_replica=True)[0]
        auxmol = basis_seg_contraction(_auxmol, allow_replica=True)[0]

        if verbose is None:
            verbose = _mol.verbose
        log = logger.new_logger(_mol, verbose)
        cput0 = log.init_timer()
        _sorted_mol, sorted_idx, uniq_l_ctr, l_ctr_counts = sort_mol(mol, log=log)

        if group_size is not None :
            uniq_l_ctr, l_ctr_counts = _split_l_ctr_groups(uniq_l_ctr, l_ctr_counts, group_size)
        self.nctr = len(uniq_l_ctr)
        self.l_ctr_counts = l_ctr_counts

        # sort fake mol
        fake_mol = make_fake_mol()
        _, _, fake_uniq_l_ctr, fake_l_ctr_counts = sort_mol(fake_mol, log=log)

        # sort auxiliary mol
        _sorted_auxmol, sorted_aux_idx, aux_uniq_l_ctr, aux_l_ctr_counts = sort_mol(auxmol, log=log)
        if group_size_aux is not None:
            aux_uniq_l_ctr, aux_l_ctr_counts = _split_l_ctr_groups(aux_uniq_l_ctr, aux_l_ctr_counts, group_size_aux)
        self.aux_l_ctr_counts = aux_l_ctr_counts

        _tot_mol = _sorted_mol + fake_mol + _sorted_auxmol
        _tot_mol.cart = True

        # shift atom indices back to actual atom indices
        nbas = _sorted_mol.nbas + 1
        _tot_mol._bas[nbas:, gto.ATOM_OF] -= (mol.natm+1) 
        self._tot_mol = _tot_mol

        # Initialize vhfopt after reordering mol._bas
        _vhf.VHFOpt.__init__(self, _sorted_mol, self._intor, self._prescreen,
                             self._qcondname, self._dmcondname)
        self.direct_scf_tol = cutoff

        # TODO: is it more accurate to filter with overlap_cond (or exp_cond)?
        q_cond = self.get_q_cond()
        cput1 = log.timer_debug1('Initialize q_cond', *cput0)
        l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
        log_qs, pair2bra, pair2ket = get_pairing(
            l_ctr_offsets, l_ctr_offsets, q_cond,
            diag_block_with_triu=diag_block_with_triu, aosym=aosym)
        self.log_qs = log_qs.copy()
        cput1 = log.timer_debug1('Get AO pairing', *cput1)

        # contraction coefficient for ao basis
        cart_ao_loc = _sorted_mol.ao_loc_nr(cart=True)
        sph_ao_loc = _sorted_mol.ao_loc_nr(cart=False)
        self.cart_ao_loc = [cart_ao_loc[cp] for cp in l_ctr_offsets]
        self.sph_ao_loc = [sph_ao_loc[cp] for cp in l_ctr_offsets]
        self.angular = [l[0] for l in uniq_l_ctr]

        # Sorted AO indices
        ao_loc = mol.ao_loc_nr(cart=_mol.cart)
        ao_idx = np.array_split(np.arange(_mol.nao), ao_loc[1:-1])
        self._ao_idx = np.hstack([ao_idx[i] for i in sorted_idx])
        cput1 = log.timer_debug1('AO indices', *cput1)

        # pairing auxiliary basis with fake basis set
        fake_l_ctr_offsets = np.append(0, np.cumsum(fake_l_ctr_counts))
        fake_l_ctr_offsets += l_ctr_offsets[-1]
        aux_l_ctr_offsets = np.append(0, np.cumsum(aux_l_ctr_counts))

        # contraction coefficient for auxiliary basis
        cart_aux_loc = _sorted_auxmol.ao_loc_nr(cart=True)
        sph_aux_loc = _sorted_auxmol.ao_loc_nr(cart=False)
        self.cart_aux_loc = [cart_aux_loc[cp] for cp in aux_l_ctr_offsets]
        self.sph_aux_loc = [sph_aux_loc[cp] for cp in aux_l_ctr_offsets]
        self.aux_angular = [l[0] for l in aux_uniq_l_ctr]

        aux_loc = _auxmol.ao_loc_nr(cart=_auxmol.cart)
        ao_idx = np.array_split(np.arange(_auxmol.nao), aux_loc[1:-1])
        self._aux_ao_idx = np.hstack([ao_idx[i] for i in sorted_aux_idx])
        cput1 = log.timer_debug1('Aux AO indices', *cput1)

        ao_loc = _sorted_mol.ao_loc_nr(cart=_mol.cart)
        self.ao_pairs_row, self.ao_pairs_col = get_ao_pairs(pair2bra, pair2ket, ao_loc)
        cderi_row = np.hstack(self.ao_pairs_row)
        cderi_col = np.hstack(self.ao_pairs_col)
        self.cderi_row = cderi_row
        self.cderi_col = cderi_col
        self.cderi_diag = np.argwhere(cderi_row == cderi_col)[:,0]
        cput1 = log.timer_debug1('Get AO pairs', *cput1)

        aux_pair2bra = []
        aux_pair2ket = []
        aux_log_qs = []
        aux_l_ctr_offsets += fake_l_ctr_offsets[-1]
        for p0, p1 in zip(aux_l_ctr_offsets[:-1], aux_l_ctr_offsets[1:]):
            aux_pair2bra.append(np.arange(p0,p1,dtype=np.int32))
            aux_pair2ket.append(fake_l_ctr_offsets[0] * np.ones(p1-p0, dtype=np.int32))
            aux_log_qs.append(np.ones(p1-p0, dtype=np.int32))

        self.aux_log_qs = aux_log_qs.copy()
        pair2bra += aux_pair2bra
        pair2ket += aux_pair2ket

        self.aux_pair2bra = aux_pair2bra
        self.aux_pair2ket = aux_pair2ket

        uniq_l_ctr = np.concatenate([uniq_l_ctr, fake_uniq_l_ctr, aux_uniq_l_ctr])
        l_ctr_offsets = np.concatenate([
            l_ctr_offsets,
            fake_l_ctr_offsets[1:],
            aux_l_ctr_offsets[1:]])

        self.pair2bra = pair2bra
        self.pair2ket = pair2ket
        self.l_ctr_offsets = l_ctr_offsets

        self._bpcache = {}

        bas_pairs_locs = np.append(0, np.cumsum([x.size for x in pair2bra])).astype(np.int32)
        self.bas_pairs_locs = bas_pairs_locs
        ncptype = len(self.log_qs)
        self.aosym = aosym
        if aosym:
            self.cp_idx, self.cp_jdx = np.tril_indices(ncptype)
        else:
            nl = int(round(np.sqrt(ncptype)))
            self.cp_idx, self.cp_jdx = np.unravel_index(np.arange(ncptype), (nl, nl))

        if _mol.cart:
            self.ao_loc = self.cart_ao_loc
        else:
            self.ao_loc = self.sph_ao_loc
        if _auxmol.cart:
            self.aux_ao_loc = self.cart_aux_loc
        else:
            self.aux_ao_loc = self.sph_aux_loc

        self._sorted_mol = _sorted_mol
        self._sorted_auxmol = _sorted_auxmol

    @property
    def bpcache(self):
        device_id = cupy.cuda.Device().id
        if device_id not in self._bpcache:
            with cupy.cuda.Device(device_id), _streams[device_id]:
                log = logger.new_logger(self.mol, self.mol.verbose)
                cput0 = log.init_timer()
                bpcache = ctypes.POINTER(BasisProdCache)()
                scale_shellpair_diag = 1.
                _tot_mol = self._tot_mol
                log_qs = self.log_qs + self.aux_log_qs
                ao_loc = _tot_mol.ao_loc_nr(cart=True)
                bas_pair2shls = np.hstack(self.pair2bra + self.pair2ket).astype(np.int32).reshape(2,-1)
                ncptype = len(log_qs)
                libgint.GINTinit_basis_prod(
                    ctypes.byref(bpcache), ctypes.c_double(scale_shellpair_diag),
                    ao_loc.ctypes.data_as(ctypes.c_void_p),
                    bas_pair2shls.ctypes.data_as(ctypes.c_void_p),
                    self.bas_pairs_locs.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ncptype),
                    _tot_mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(_tot_mol.natm),
                    _tot_mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(_tot_mol.nbas),
                    _tot_mol._env.ctypes.data_as(ctypes.c_void_p))
                self._bpcache[device_id] = bpcache
                cput0 = log.timer_debug1(f'Initialize GPU cache on Device {device_id}', *cput0)
        bpcache = self._bpcache[device_id]
        return bpcache

    def sort_orbitals(self, mat, axis=[], aux_axis=[]):
        ''' Transform given axis of a matrix into sorted AO,
        and transform given auxiliary axis of a matrix into sorted auxiliary AO
        '''
        idx = self._ao_idx
        aux_idx = self._aux_ao_idx
        shape_ones = (1,) * mat.ndim
        fancy_index = []
        for dim, n in enumerate(mat.shape):
            if dim in axis:
                assert n == len(idx)
                indices = idx
            elif dim in aux_axis:
                assert n == len(aux_idx)
                indices = aux_idx
            else:
                indices = np.arange(n)
            idx_shape = shape_ones[:dim] + (-1,) + shape_ones[dim+1:]
            fancy_index.append(indices.reshape(idx_shape))
        return mat[tuple(fancy_index)]

    def unsort_orbitals(self, sorted_mat, axis=[], aux_axis=[]):
        ''' Transform given axis of a matrix into sorted AO,
        and transform given auxiliary axis of a matrix into original auxiliary AO
        '''
        idx = self._ao_idx
        aux_idx = self._aux_ao_idx
        shape_ones = (1,) * sorted_mat.ndim
        fancy_index = []
        for dim, n in enumerate(sorted_mat.shape):
            if dim in axis:
                assert n == len(idx)
                indices = idx
            elif dim in aux_axis:
                assert n == len(aux_idx)
                indices = aux_idx
            else:
                indices = np.arange(n)
            idx_shape = shape_ones[:dim] + (n,) + shape_ones[dim+1:]
            fancy_index.append(indices.reshape(idx_shape))
        mat = cupy.empty_like(sorted_mat)
        mat[tuple(fancy_index)] = sorted_mat
        return mat

    @property
    def cart2sph(self):
        return block_c2s_diag(self.angular, self.l_ctr_counts)

    @property
    def aux_cart2sph(self):
        return block_c2s_diag(self.aux_angular, self.aux_l_ctr_counts)

    @property
    def coeff(self):
        nao = self.mol.nao
        if self.mol.cart:
            coeff = cupy.eye(nao)
            self._coeff = self.unsort_orbitals(coeff, axis=[1])
        else:
            self._coeff = self.unsort_orbitals(self.cart2sph, axis=[1])
        return self._coeff

    @property
    def aux_coeff(self):
        naux = self.auxmol.nao
        if self.auxmol.cart:
            coeff = cupy.eye(naux)
            self._aux_coeff = self.unsort_orbitals(coeff, aux_axis=[1])
        else:
            self._aux_coeff = self.unsort_orbitals(self.aux_cart2sph, aux_axis=[1])
        return self._aux_coeff

def get_int3c2e_wjk(mol, auxmol, dm0_tag, thred=1e-12, omega=None, with_j=True, with_k=True):
    log = logger.new_logger(mol, mol.verbose)
    intopt = VHFOpt(mol, auxmol, 'int2e')
    intopt.build(thred, diag_block_with_triu=True, aosym=True,
                 group_size=BLKSIZE, group_size_aux=BLKSIZE)
    orbo = dm0_tag.occ_coeff
    nao = mol.nao
    naux = auxmol.nao
    nocc = orbo.shape[1]

    wj = None
    if with_j:
        wj = cupy.empty([naux])

    wk = None
    if with_k:
        avail_mem = get_avail_mem()
        use_gpu_memory = True
        if naux*nao*nocc*8 < 0.4*avail_mem:
            try:
                wk = cupy.empty([naux,nao,nocc])
            except Exception:
                use_gpu_memory = False
        else:
            use_gpu_memory = False

        if not use_gpu_memory:
            log.debug('Saving int3c2e_wjk on CPU memory')
            mem = cupy.cuda.alloc_pinned_memory(naux*nao*nocc*8)
            wk = np.ndarray([naux,nao,nocc], dtype=np.float64, order='C', buffer=mem)

    # TODO: async data transfer
    for cp_kl_id, _ in enumerate(intopt.aux_log_qs):
        k0 = intopt.aux_ao_loc[cp_kl_id]
        k1 = intopt.aux_ao_loc[cp_kl_id+1]
        if with_j:
            rhoj_tmp = cupy.zeros([k1-k0], order='C')
        if with_k:
            rhok_tmp = cupy.zeros([k1-k0, nao, nocc], order='C')

        for cp_ij_id, _ in enumerate(intopt.log_qs):
            cpi = intopt.cp_idx[cp_ij_id]
            cpj = intopt.cp_jdx[cp_ij_id]
            li = intopt.angular[cpi]
            lj = intopt.angular[cpj]
            int3c_blk = get_int3c2e_slice(intopt, cp_ij_id, cp_kl_id, omega=omega)
            if not intopt.mol.cart:
                int3c_blk = cart2sph(int3c_blk, axis=1, ang=lj)
                int3c_blk = cart2sph(int3c_blk, axis=2, ang=li)
            i0, i1 = intopt.ao_loc[cpi], intopt.ao_loc[cpi+1]
            j0, j1 = intopt.ao_loc[cpj], intopt.ao_loc[cpj+1]
            if with_j:
                tmp = contract('Lji,ij->L', int3c_blk, dm0_tag[i0:i1,j0:j1])
                rhoj_tmp += tmp
                if cpi != cpj:
                    rhoj_tmp += tmp
            if with_k:
                rhok_tmp[:,j0:j1] += contract('Lji,io->Ljo', int3c_blk, orbo[i0:i1])
                if cpi != cpj:
                    rhok_tmp[:,i0:i1] += contract('Lji,jo->Lio', int3c_blk, orbo[j0:j1])
        if with_j:
            wj[k0:k1] = rhoj_tmp
        if with_k:
            if isinstance(wk, cupy.ndarray):
                wk[k0:k1] = rhok_tmp
            else:
                #rhok_tmp.get(out=wk[k0:k1])
                copy_array(rhok_tmp, wk[k0:k1])
    return wj, wk

def get_int3c2e_ip_jk(intopt, cp_aux_id, ip_type, rhoj, rhok, dm, omega=None, stream=None):
    '''
    build jk with int3c2e slice (sliced in k dimension)
    '''
    if omega is None: omega = 0.0
    if stream is None: stream = cupy.cuda.get_current_stream()

    fn = getattr(libgvhf, 'GINTbuild_int3c2e_' + ip_type + '_jk')
    nao = intopt._sorted_mol.nao
    natm = intopt._sorted_mol.natm
    n_dm = 1
    cp_kl_id = cp_aux_id + len(intopt.log_qs)
    log_q_kl = intopt.aux_log_qs[cp_aux_id]

    k0, k1 = intopt.cart_aux_loc[cp_aux_id], intopt.cart_aux_loc[cp_aux_id+1]
    ao_offsets = np.array([0,0,nao+1+k0,nao], dtype=np.int32)
    nk = k1 - k0

    ej_ptr = ek_ptr = lib.c_null_ptr()
    rhoj_ptr = rhok_ptr = lib.c_null_ptr()
    ej = ek = None
    if rhoj is not None:
        assert(rhoj.flags['C_CONTIGUOUS'])
        rhoj_ptr = ctypes.cast(rhoj.data.ptr, ctypes.c_void_p)
        if ip_type == 'ip1':
            ej = cupy.zeros([natm,3], order='C')
        elif ip_type == 'ip2':
            ej = cupy.zeros([natm,3], order='C')
        ej_ptr = ctypes.cast(ej.data.ptr, ctypes.c_void_p)
    if rhok is not None:
        assert(rhok.flags['C_CONTIGUOUS'])
        rhok_ptr = ctypes.cast(rhok.data.ptr, ctypes.c_void_p)
        if ip_type == 'ip1':
            ek = cupy.zeros([natm,3], order='C')
        elif ip_type == 'ip2':
            ek = cupy.zeros([natm,3], order='C')
        ek_ptr = ctypes.cast(ek.data.ptr, ctypes.c_void_p)
    num_cp_ij = [len(log_qs) for log_qs in intopt.log_qs]
    bins_locs_ij = np.append(0, np.cumsum(num_cp_ij)).astype(np.int32)
    ntasks_kl = len(log_q_kl)
    ncp_ij = len(intopt.log_qs)
    err = fn(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        intopt.bpcache,
        ej_ptr,
        ek_ptr,
        ctypes.cast(dm.data.ptr, ctypes.c_void_p),
        rhoj_ptr,
        rhok_ptr,
        ao_offsets.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao),
        ctypes.c_int(nk),
        ctypes.c_int(n_dm),
        bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(ntasks_kl),
        ctypes.c_int(ncp_ij),
        ctypes.c_int(cp_kl_id),
        ctypes.c_double(omega))
    if err != 0:
        raise RuntimeError(f'GINT_getjk_int2e_ip failed, err={err}')
    return ej, ek

def loop_int3c2e_general(intopt, task_list=None, ip_type='', omega=None, stream=None):
    '''
    loop over all int3c2e blocks
    - outer loop for k
    - inner loop for ij pair
    '''
    fn = getattr(libgint, 'GINTfill_int3c2e_' + ip_type)
    if ip_type == '':       order = 0
    if ip_type == 'ip1':    order = 1
    if ip_type == 'ip2':    order = 1
    if ip_type == 'ipip1':  order = 2
    if ip_type == 'ip1ip2': order = 2
    if ip_type == 'ipvip1': order = 2
    if ip_type == 'ipip2':  order = 2

    if omega is None: omega = 0.0
    if stream is None: stream = cupy.cuda.get_current_stream()
    assert omega >= 0

    nao = intopt._sorted_mol.nao
    naux = intopt._sorted_auxmol.nao
    norb = nao + naux + 1
    ao_loc = intopt.ao_loc
    aux_ao_loc = intopt.aux_ao_loc
    comp = 3**order
    nbins = 1

    # If task_list is not given, generate all the tasks
    if task_list is None:
        ncp_k = len(intopt.aux_log_qs)
        ncp_ij = len(intopt.log_qs)
        task_list = itertools.product(range(ncp_k), range(ncp_ij))

    for aux_id, cp_ij_id in task_list:
        cp_kl_id = aux_id + len(intopt.log_qs)
        lk = intopt.aux_angular[aux_id]

        cpi = intopt.cp_idx[cp_ij_id]
        cpj = intopt.cp_jdx[cp_ij_id]
        li = intopt.angular[cpi]
        lj = intopt.angular[cpj]

        i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi+1]
        j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj+1]
        k0, k1 = intopt.cart_aux_loc[aux_id], intopt.cart_aux_loc[aux_id+1]
        ni = i1 - i0
        nj = j1 - j0
        nk = k1 - k0

        log_q_ij = intopt.log_qs[cp_ij_id]
        log_q_kl = intopt.aux_log_qs[aux_id]

        bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)
        bins_locs_kl = np.array([0, len(log_q_kl)], dtype=np.int32)

        ao_offsets = np.array([i0,j0,nao+1+k0,nao], dtype=np.int32)
        strides = np.array([1, ni, ni*nj, ni*nj*nk], dtype=np.int32)

        # Use GPU kernels for low-angular momentum
        if (li + lj + lk + order)//2 + 1 < NROOT_ON_GPU:
            int3c_blk = cupy.zeros([comp, nk, nj, ni], order='C', dtype=np.float64)
            err = fn(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                intopt.bpcache,
                ctypes.cast(int3c_blk.data.ptr, ctypes.c_void_p),
                ctypes.c_int(norb),
                strides.ctypes.data_as(ctypes.c_void_p),
                ao_offsets.ctypes.data_as(ctypes.c_void_p),
                bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                bins_locs_kl.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbins),
                ctypes.c_int(cp_ij_id),
                ctypes.c_int(cp_kl_id),
                ctypes.c_double(omega))
            if err != 0:
                raise RuntimeError(f'GINT_fill_int3c2e general failed, err={err}')
        else:
            from pyscf.gto.moleintor import getints, make_cintopt
            pmol = intopt._tot_mol
            intor = pmol._add_suffix('int3c2e_' + ip_type)
            opt = make_cintopt(pmol._atm, pmol._bas, pmol._env, intor)

            # TODO: sph2cart in CPU?
            ishl0, ishl1 = intopt.l_ctr_offsets[cpi], intopt.l_ctr_offsets[cpi+1]
            jshl0, jshl1 = intopt.l_ctr_offsets[cpj], intopt.l_ctr_offsets[cpj+1]
            kshl0, kshl1 = intopt.l_ctr_offsets[aux_id+1+intopt.nctr], intopt.l_ctr_offsets[aux_id+1+intopt.nctr+1]
            shls_slice = np.array([ishl0, ishl1, jshl0, jshl1, kshl0, kshl1], dtype=np.int64)
            int3c_cpu = getints(intor, pmol._atm, pmol._bas, pmol._env, shls_slice, cintopt=opt).transpose([0,3,2,1])
            int3c_blk = cupy.asarray(int3c_cpu)

        if not intopt.auxmol.cart:
            int3c_blk = cart2sph(int3c_blk, axis=1, ang=lk)
        if not intopt.mol.cart:
            int3c_blk = cart2sph(int3c_blk, axis=2, ang=lj)
            int3c_blk = cart2sph(int3c_blk, axis=3, ang=li)

        i0, i1 = ao_loc[cpi], ao_loc[cpi+1]
        j0, j1 = ao_loc[cpj], ao_loc[cpj+1]
        k0, k1 = aux_ao_loc[aux_id], aux_ao_loc[aux_id+1]

        yield i0,i1,j0,j1,k0,k1,int3c_blk

def loop_aux_jk(intopt, ip_type='', omega=None, stream=None):
    '''
    **** deprecated **********
    loop over all int3c2e blocks
    - outer loop for k
    - inner loop for ij pair
    '''
    fn = getattr(libgint, 'GINTfill_int3c2e_' + ip_type)
    if ip_type == '':       order = 0
    if ip_type == 'ip1':    order = 1
    if ip_type == 'ip2':    order = 1
    if ip_type == 'ipip1':  order = 2
    if ip_type == 'ip1ip2': order = 2
    if ip_type == 'ipvip1': order = 2
    if ip_type == 'ipip2':  order = 2

    if omega is None: omega = 0.0
    if stream is None: stream = cupy.cuda.get_current_stream()
    assert omega >= 0

    nao = intopt.mol.nao
    nao_cart = intopt._sorted_mol.nao
    naux_cart = intopt._sorted_auxmol.nao
    norb_cart = nao_cart + naux_cart + 1
    ao_loc = intopt.ao_loc
    aux_ao_loc = intopt.aux_ao_loc
    comp = 3**order

    nbins = 1
    for aux_id, log_q_kl in enumerate(intopt.aux_log_qs):
        cp_kl_id = aux_id + len(intopt.log_qs)
        k0, k1 = intopt.aux_ao_loc[aux_id], intopt.aux_ao_loc[aux_id+1]
        lk = intopt.aux_angular[aux_id]

        ints_slices = cupy.zeros([comp, k1-k0, nao, nao])
        for cp_ij_id, log_q_ij in enumerate(intopt.log_qs):
            cpi = intopt.cp_idx[cp_ij_id]
            cpj = intopt.cp_jdx[cp_ij_id]
            li = intopt.angular[cpi]
            lj = intopt.angular[cpj]

            i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi+1]
            j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj+1]
            k0, k1 = intopt.cart_aux_loc[aux_id], intopt.cart_aux_loc[aux_id+1]
            ni = i1 - i0
            nj = j1 - j0
            nk = k1 - k0

            bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)
            bins_locs_kl = np.array([0, len(log_q_kl)], dtype=np.int32)

            ao_offsets = np.array([i0,j0,nao_cart+1+k0,nao_cart], dtype=np.int32)
            strides = np.array([1, ni, ni*nj, ni*nj*nk], dtype=np.int32)

            int3c_blk = cupy.zeros([comp, nk, nj, ni], order='C', dtype=np.float64)
            err = fn(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                intopt.bpcache,
                ctypes.cast(int3c_blk.data.ptr, ctypes.c_void_p),
                ctypes.c_int(norb_cart),
                strides.ctypes.data_as(ctypes.c_void_p),
                ao_offsets.ctypes.data_as(ctypes.c_void_p),
                bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                bins_locs_kl.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbins),
                ctypes.c_int(cp_ij_id),
                ctypes.c_int(cp_kl_id),
                ctypes.c_double(omega))
            if err != 0:
                raise RuntimeError(f'GINT_fill_int3c2e general failed, err={err}')

            if not intopt._auxmol.cart:
                int3c_blk = cart2sph(int3c_blk, axis=1, ang=lk)
            if not intopt._mol.cart:
                int3c_blk = cart2sph(int3c_blk, axis=2, ang=lj)
                int3c_blk = cart2sph(int3c_blk, axis=3, ang=li)

            i0, i1 = ao_loc[cpi], ao_loc[cpi+1]
            j0, j1 = ao_loc[cpj], ao_loc[cpj+1]
            k0, k1 = aux_ao_loc[aux_id], aux_ao_loc[aux_id+1]
            ints_slices[:, :, j0:j1, i0:i1] = int3c_blk
        int3c_blk = None
        yield aux_id, ints_slices

def get_ao2atom(intopt, aoslices):
    nao = intopt.mol.nao
    ao2atom = cupy.zeros([nao, len(aoslices)])
    for ia, aoslice in enumerate(aoslices):
        _, _, p0, p1 = aoslice
        ao2atom[p0:p1,ia] = 1.0
    return intopt.sort_orbitals(ao2atom, axis=[0])

def get_aux2atom(intopt, auxslices):
    naux = intopt.auxmol.nao
    aux2atom = cupy.zeros([naux, len(auxslices)])
    for ia, auxslice in enumerate(auxslices):
        _, _, p0, p1 = auxslice
        aux2atom[p0:p1,ia] = 1.0
    return intopt.sort_orbitals(aux2atom, aux_axis=[0])

def get_j_int3c2e_pass1(intopt, dm0, sort_j=True, stream=None):
    '''
    get rhoj pass1 for int3c2e
    '''
    if stream is None: stream = cupy.cuda.get_current_stream()

    n_dm = 1

    naux = intopt._sorted_auxmol.nao

    coeff = intopt.coeff
    if dm0.ndim == 3:
        dm0 = dm0[0] + dm0[1]
    dm_cart = coeff @ dm0 @ coeff.T

    num_cp_ij = [len(log_qs) for log_qs in intopt.log_qs]
    num_cp_kl = [len(log_qs) for log_qs in intopt.aux_log_qs]

    bins_locs_ij = np.append(0, np.cumsum(num_cp_ij)).astype(np.int32)
    bins_locs_kl = np.append(0, np.cumsum(num_cp_kl)).astype(np.int32)

    ncp_ij = len(intopt.log_qs)
    ncp_kl = len(intopt.aux_log_qs)
    norb = dm_cart.shape[0]

    rhoj = cupy.zeros([naux])

    err = libgvhf.GINTbuild_j_int3c2e_pass1(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        intopt.bpcache,
        ctypes.cast(dm_cart.data.ptr, ctypes.c_void_p),
        ctypes.cast(rhoj.data.ptr, ctypes.c_void_p),
        ctypes.c_int(norb),
        ctypes.c_int(naux),
        ctypes.c_int(n_dm),
        bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
        bins_locs_kl.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(ncp_ij),
        ctypes.c_int(ncp_kl))
    if err != 0:
        raise RuntimeError('CUDA error in get_j_pass1')

    if sort_j:
        aux_coeff = intopt.aux_coeff
        rhoj = cupy.dot(rhoj, aux_coeff)
    return rhoj

def get_j_int3c2e_pass2(intopt, rhoj, stream=None):
    '''
    get vj pass2 for int3c2e
    '''
    if stream is None: stream = cupy.cuda.get_current_stream()

    n_dm = 1
    norb = intopt._sorted_mol.nao
    naux = intopt._sorted_auxmol.nao
    vj = cupy.zeros([norb, norb])

    num_cp_ij = [len(log_qs) for log_qs in intopt.log_qs]
    num_cp_kl = [len(log_qs) for log_qs in intopt.aux_log_qs]

    bins_locs_ij = np.append(0, np.cumsum(num_cp_ij)).astype(np.int32)
    bins_locs_kl = np.append(0, np.cumsum(num_cp_kl)).astype(np.int32)

    ncp_ij = len(intopt.log_qs)
    ncp_kl = len(intopt.aux_log_qs)

    rhoj = intopt.sort_orbitals(rhoj, aux_axis=[0])
    if not intopt.auxmol.cart:
        rhoj = intopt.aux_cart2sph @ rhoj

    err = libgvhf.GINTbuild_j_int3c2e_pass2(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        intopt.bpcache,
        ctypes.cast(vj.data.ptr, ctypes.c_void_p),
        ctypes.cast(rhoj.data.ptr, ctypes.c_void_p),
        ctypes.c_int(norb),
        ctypes.c_int(naux),
        ctypes.c_int(n_dm),
        bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
        bins_locs_kl.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(ncp_ij),
        ctypes.c_int(ncp_kl))

    if err != 0:
        raise RuntimeError('CUDA error in get_j_pass2')

    if not intopt.mol.cart:
        cart2sph = intopt.cart2sph
        vj = cart2sph.T @ vj @ cart2sph
    vj = intopt.unsort_orbitals(vj, axis=[0,1])
    vj = vj + vj.T
    return vj

def _int3c2e_jk_task(intopt, task_k_list, dm0, mocc, device_id=0, omega=None):
    with cupy.cuda.Device(device_id), _streams[device_id]:
        log = logger.new_logger(intopt.mol, intopt.mol.verbose)
        t0 = log.init_timer()
        mocc = cupy.asarray(mocc)
        dm0 = cupy.asarray(dm0)
        naux = intopt.auxmol.nao
        nocc = mocc.shape[1]
        rhoj = cupy.zeros([naux])
        rhok = cupy.zeros([naux,nocc,nocc])
        for cp_kl_id in task_k_list:
            k0 = intopt.aux_ao_loc[cp_kl_id]
            k1 = intopt.aux_ao_loc[cp_kl_id+1]
            rhoj_tmp = cupy.zeros([k1-k0], order='C')
            rhok_tmp = cupy.zeros([k1-k0, nocc, nocc], order='C')
            for cp_ij_id, _ in enumerate(intopt.log_qs):
                cpi = intopt.cp_idx[cp_ij_id]
                cpj = intopt.cp_jdx[cp_ij_id]
                li = intopt.angular[cpi]
                lj = intopt.angular[cpj]
                int3c_blk = get_int3c2e_slice(intopt, cp_ij_id, cp_kl_id, omega=omega)
                if not intopt.mol.cart:
                    int3c_blk = cart2sph(int3c_blk, axis=1, ang=lj)
                    int3c_blk = cart2sph(int3c_blk, axis=2, ang=li)
                i0, i1 = intopt.ao_loc[cpi], intopt.ao_loc[cpi+1]
                j0, j1 = intopt.ao_loc[cpj], intopt.ao_loc[cpj+1]
                if cpi == cpj and intopt.aosym:
                    int3c_blk *= 0.5

                rhoj_tmp += contract('pji,ij->p', int3c_blk, dm0[i0:i1,j0:j1])
                ints_o = contract('pji,jo->poi', int3c_blk, mocc[j0:j1])
                rhok_tmp += contract('poi,ir->por', ints_o, mocc[i0:i1])
                int3c_blk = ints_o = None
            if intopt.aosym:
                rhoj[k0:k1] = 2.0 * rhoj_tmp
                rhok[k0:k1] = transpose_sum(rhok_tmp)
            else:
                rhoj[k0:k1] = rhoj_tmp
                rhok[k0:k1] = rhok_tmp
        t0 = log.timer_debug1(f'int3c2e_vjk on Device {device_id}', *t0)
    return rhoj, rhok

def get_int3c2e_jk(mol, auxmol, dm0_tag, with_k=True, omega=None):
    '''
    get rhoj and rhok for int3c2e
    '''
    intopt = VHFOpt(mol, auxmol, 'int2e')
    intopt.build(1e-14, diag_block_with_triu=True, aosym=True, group_size=BLKSIZE, group_size_aux=BLKSIZE)

    orbo = cupy.asarray(dm0_tag.occ_coeff, order='C')
    futures = []
    aux_ao_loc = np.array(intopt.aux_ao_loc)
    loads = aux_ao_loc[1:] - aux_ao_loc[:-1]
    task_list = _split_tasks(loads, num_devices)

    cupy.cuda.get_current_stream().synchronize()
    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        for device_id in range(num_devices):
            future = executor.submit(
                _int3c2e_jk_task, intopt, task_list[device_id],
                dm0_tag, orbo, device_id=device_id, omega=omega)
            futures.append(future)

    rhoj_total = []
    rhok_total = []
    for future in futures:
        rhoj, rhok = future.result()
        rhoj_total.append(rhoj)
        rhok_total.append(rhok)

    rhoj = rhok = None
    rhoj = reduce_to_device(rhoj_total, inplace=True)
    if with_k:
        rhok = reduce_to_device(rhok_total, inplace=True)
    return rhoj, rhok

def _split_tasks(loads, ngroups):
    ''' Split a list of numbers into sublists with sums as close as possible
    '''
    if ngroups == 1:
        return [range(len(loads))]
    groups = [[] for _ in range(ngroups)]
    sums = [0] * ngroups

    sorted_indices = np.argsort(loads)[::-1]
    for idx in sorted_indices:
        min_index = sums.index(min(sums))
        groups[min_index].append(idx)
        sums[min_index] += loads[idx]
    return groups

def _int3c2e_ip1_vjk_task(intopt, task_k_list, rhoj, rhok, dm0, orbo, device_id=0,
                          with_j=True, with_k=True, omega=None):
    natom = intopt.mol.natm
    nao = intopt.mol.nao
    aoslices = intopt.mol.aoslice_by_atom()
    vj1_buf = vk1_buf = vj1 = vk1 = None

    with cupy.cuda.Device(device_id), _streams[device_id]:
        log = logger.new_logger(intopt.mol, intopt.mol.verbose)
        t0 = log.init_timer()
        ao2atom = get_ao2atom(intopt, aoslices)
        dm0 = cupy.asarray(dm0)
        orbo = cupy.asarray(orbo)
        nocc = orbo.shape[1]
        if with_j:
            rhoj = cupy.asarray(rhoj)
            vj1_buf = cupy.zeros([3,nao,nao])
            vj1 = cupy.zeros([natom,3,nao,nocc])
        if with_k:
            vk1_buf = cupy.zeros([3,nao,nao])
            vk1 = cupy.zeros([natom,3,nao,nocc])
        aux_ao_loc = intopt.aux_ao_loc
        ncp_ij = len(intopt.log_qs)
        for cp_k in task_k_list:
            task_list = [(cp_k, cp_ij) for cp_ij in range(ncp_ij)]
            k0, k1 = aux_ao_loc[cp_k], aux_ao_loc[cp_k+1]
            #rhok_tmp = cupy.asarray(rhok[k0:k1])
            rhok_tmp = copy_array(rhok[k0:k1])
            if with_k:
                rhok0 = contract('pio,ir->pro', rhok_tmp, orbo)
                rhok0 = contract('pro,Jo->prJ', rhok0, orbo)
                int3c_ip1_occ = cupy.zeros([3,k1-k0,nao,nocc])
            if with_j:
                rhoj0 = cupy.zeros([3,k1-k0,nao])

            for i0,i1,j0,j1,k0,k1,int3c_blk in loop_int3c2e_general(intopt, task_list=task_list,
                                                                     ip_type='ip1', omega=omega):
                if with_j:
                    vj1_buf[:,i0:i1,j0:j1] += contract('xpji,p->xij', int3c_blk, rhoj[k0:k1])
                    rhoj0[:,:,i0:i1] += contract('xpji,ij->xpi', int3c_blk, dm0[i0:i1,j0:j1])
                if with_k:
                    int3c_ip1_occ[:,:,i0:i1] += contract('xpji,jo->xpio', int3c_blk, orbo[j0:j1])

                    vk1_ao = contract('xpji,poi->xijo', int3c_blk, rhok0[:,:,i0:i1])
                    vk1[:,:,j0:j1] += contract('xijo,ia->axjo', vk1_ao, ao2atom[i0:i1])
                    vk1_ao = int3c_blk = None
            if with_j:
                rhoj0_atom = contract('xpi,ia->xpa', rhoj0, 2.0*ao2atom)
                vj1 += contract('pJo,xpa->axJo', rhok_tmp, rhoj0_atom)
                rhoj0_atom = rhoj0 = None
            if with_k:
                rhok0 = None
                vk1_buf += contract('xpio,plo->xil', int3c_ip1_occ, rhok_tmp)
                mem_avail = get_avail_mem()
                blksize = min(int(mem_avail * 0.2 / ((k1-k0) * nao) * 8),
                              int(mem_avail * 0.2 / (nocc * nao * 3 * 8)))
                for p0, p1, in lib.prange(0, nao, blksize):
                    rhok0_slice = contract('pJr,ir->pJi', rhok_tmp[:,p0:p1], orbo)
                    vk1_ao = contract('xpio,pJi->xiJo', int3c_ip1_occ, rhok0_slice)
                    vk1[:,:,p0:p1] += contract('xiJo,ia->axJo', vk1_ao, ao2atom)
                    rhok0_slice = vk1_ao = None
            rhok_tmp = int3c_ip1_occ = None
        t0 = log.timer_debug1(f'int3c2e_ip1_vjk on Device {device_id}', *t0)
    # TODO: absorbe vj1_buf and vk1_buf into vj1 and vk1
    return vj1_buf, vk1_buf, vj1, vk1

def get_int3c2e_ip1_vjk(intopt, rhoj, rhok, dm0_tag, aoslices, with_j=True,
                        with_k=True, omega=None):
    orbo = cupy.asarray(dm0_tag.occ_coeff, order='C')
    futures = []

    aux_ao_loc = np.array(intopt.aux_ao_loc)
    loads = aux_ao_loc[1:] - aux_ao_loc[:-1]
    task_list = _split_tasks(loads, num_devices)

    cupy.cuda.get_current_stream().synchronize()
    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        for device_id in range(num_devices):
            future = executor.submit(
                _int3c2e_ip1_vjk_task, intopt, task_list[device_id],
                rhoj, rhok, dm0_tag, orbo, with_j=with_j, with_k=with_k,
                device_id=device_id, omega=omega)
            futures.append(future)

    vj1_buf_total = []
    vk1_buf_total = []
    vj1_total = []
    vk1_total = []
    for future in futures:
        vj1_buf, vk1_buf, vj1, vk1 = future.result()
        vj1_buf_total.append(vj1_buf)
        vk1_buf_total.append(vk1_buf)
        vj1_total.append(vj1)
        vk1_total.append(vk1)

    vj1 = vk1 = vj1_buf = vk1_buf = None
    if with_j:
        vj1 = reduce_to_device(vj1_total, inplace=True)
        vj1_buf = reduce_to_device(vj1_buf_total, inplace=True)
    if with_k:
        vk1 = reduce_to_device(vk1_total, inplace=True)
        vk1_buf = reduce_to_device(vk1_buf_total, inplace=True)
    return vj1_buf, vk1_buf, vj1, vk1


def _int3c2e_ip2_vjk_task(intopt, task_k_list, rhoj, rhok, dm0, orbo,
                          device_id=0, with_j=True, with_k=True, omega=None):
    natom = intopt.mol.natm
    nao = intopt.mol.nao
    auxslices = intopt.auxmol.aoslice_by_atom()
    vj1 = vk1 = None
    with cupy.cuda.Device(device_id), _streams[device_id]:
        log = logger.new_logger(intopt.mol, intopt.mol.verbose)
        t0 = log.init_timer()
        aux2atom = get_aux2atom(intopt, auxslices)
        dm0 = cupy.asarray(dm0)
        orbo = cupy.asarray(orbo)
        nocc = orbo.shape[1]
        if with_j:
            rhoj = cupy.asarray(rhoj)
            vj1 = cupy.zeros([natom,3,nao,nocc])
        if with_k:
            vk1 = cupy.zeros([natom,3,nao,nocc])
        aux_ao_loc = intopt.aux_ao_loc
        ncp_ij = len(intopt.log_qs)
        for cp_k in task_k_list:
            task_list = [(cp_k, cp_ij) for cp_ij in range(ncp_ij)]
            k0, k1 = aux_ao_loc[cp_k], aux_ao_loc[cp_k+1]
            if with_j:
                wj2 = cupy.zeros([3,k1-k0])

            wk2_P__ = cupy.zeros([3,k1-k0,nao,nocc])
            for i0,i1,j0,j1,k0,k1,int3c_blk in loop_int3c2e_general(intopt, task_list=task_list,
                                                                     ip_type='ip2', omega=omega):
                # contraction
                if with_j:
                    wj2 += contract('xpji,ji->xp', int3c_blk, dm0[j0:j1,i0:i1])

                wk2_P__[:,:,i0:i1] += contract('xpji,jo->xpio', int3c_blk, orbo[j0:j1])
                int3c_blk = None
            #rhok_tmp = cupy.asarray(rhok[k0:k1])
            rhok_tmp = copy_array(rhok[k0:k1])
            if with_j:
                vj1_tmp = -contract('pio,xp->xpio', rhok_tmp, wj2)
                vj1_tmp -= contract('xpio,p->xpio', wk2_P__, rhoj[k0:k1])

                vj1 += contract('xpio,pa->axio', vj1_tmp, aux2atom[k0:k1])
                vj1_tmp = wj2 = None
            if with_k:
                rhok0_slice = contract('xpjo,jr->xpro', wk2_P__, orbo)
                vk1_tmp = -contract('xpro,pir->xpio', rhok0_slice, rhok_tmp)

                rhok0_oo = contract('pio,ir->pro', rhok_tmp, orbo)
                vk1_tmp -= contract('xpio,pro->xpir', wk2_P__, rhok0_oo)

                vk1 += contract('xpir,pa->axir', vk1_tmp, aux2atom[k0:k1])
                vk1_tmp = rhok0_oo = rhok0_slice = None
            rhok_tmp = wk2_P__ = None
        t0 = log.timer_debug1(f'int3c2e_ip2_vjk on Device {device_id}', *t0)
    return vj1, vk1

def get_int3c2e_ip2_vjk(intopt, rhoj, rhok, dm0_tag, auxslices,
                        with_j=True, with_k=True, omega=None):
    '''
    vj and vk responses (due to int3c2e_ip2) to changes in atomic positions
    '''
    orbo = cupy.asarray(dm0_tag.occ_coeff, order='C')
    futures = []

    aux_ao_loc = np.array(intopt.aux_ao_loc)
    loads = aux_ao_loc[1:] - aux_ao_loc[:-1]
    task_list = _split_tasks(loads, num_devices)

    cupy.cuda.get_current_stream().synchronize()
    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        for device_id in range(num_devices):
            future = executor.submit(
                _int3c2e_ip2_vjk_task, intopt, task_list[device_id],
                rhoj, rhok, dm0_tag, orbo, with_j=with_j,
                with_k=with_k, device_id=device_id, omega=omega)
            futures.append(future)

    vj_total = []
    vk_total = []
    for future in futures:
        vj, vk = future.result()
        vj_total.append(vj)
        vk_total.append(vk)

    vj = vk = None
    if with_j:
        vj = reduce_to_device(vj_total, inplace=True)
    if with_k:
        vk = reduce_to_device(vk_total, inplace=True)
    return vj, vk

def _int3c2e_ip1_wjk_task(intopt, task_k_list, dm0, orbo, wk, device_id=0, with_k=True, omega=None):
    nao = intopt.mol.nao
    naux = intopt.auxmol.nao
    aux_ao_loc = intopt.aux_ao_loc
    with cupy.cuda.Device(device_id), _streams[device_id]:
        log = logger.new_logger(intopt.mol, intopt.mol.verbose)
        t0 = log.init_timer()
        ncp_ij = len(intopt.log_qs)
        nocc = orbo.shape[1]
        wj = cupy.zeros([naux,nao,3])
        dm0 = cupy.asarray(dm0)
        orbo = cupy.asarray(orbo)
        for cp_k in task_k_list:
            k0, k1 = aux_ao_loc[cp_k], aux_ao_loc[cp_k+1]
            if with_k:
                wk_tmp = cupy.zeros([k1-k0,nao,nocc,3])
            task_list = [(cp_k, cp_ij) for cp_ij in range(ncp_ij)]
            for i0,i1,j0,j1,k0,k1,int3c_blk in loop_int3c2e_general(intopt, task_list=task_list,
                                                                    ip_type='ip1', omega=omega):
                wj[k0:k1,i0:i1] += contract('xpji,ij->pix', int3c_blk, dm0[i0:i1,j0:j1])
                if with_k:
                    wk_tmp[:,i0:i1] += contract('xpji,jo->piox', int3c_blk, orbo[j0:j1])
                int3c_blk = None
            if with_k:
                #wk_tmp.get(out=wk[k0:k1])
                copy_array(wk_tmp, wk[k0:k1])
            wk_tmp = None
        t0 = log.timer_debug1(f'int3c2e_ip1_wjk on Device {device_id}', *t0)
    return wj

def get_int3c2e_ip1_wjk(intopt, dm0_tag, with_k=True, omega=None):
    ''' wj in GPU, wk in CPU
    '''
    orbo = cupy.asarray(dm0_tag.occ_coeff, order='C')
    futures = []

    aux_ao_loc = np.array(intopt.aux_ao_loc)
    loads = aux_ao_loc[1:] - aux_ao_loc[:-1]
    task_list = _split_tasks(loads, num_devices)

    nao = intopt.mol.nao
    naux = intopt.auxmol.nao
    nocc = orbo.shape[1]
    wk = None
    if with_k:
        mem = cupy.cuda.alloc_pinned_memory(nao*naux*nocc*3*8)
        wk = np.ndarray([naux,nao,nocc,3], dtype=np.float64, order='C', buffer=mem)

    cupy.cuda.get_current_stream().synchronize()
    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        for device_id in range(num_devices):
            future = executor.submit(
                _int3c2e_ip1_wjk_task, intopt, task_list[device_id],
                dm0_tag, orbo, wk, with_k=with_k, device_id=device_id, omega=omega)
            futures.append(future)
    wj_total = []
    for future in futures:
        wj = future.result()
        wj_total.append(wj)
    wj = reduce_to_device(wj_total, inplace=True)
    return wj, wk

def _int3c2e_ip2_wjk(intopt, task_list, dm0, orbo, with_k=True, omega=None, device_id=0):
    aux_ao_loc = intopt.aux_ao_loc
    with cupy.cuda.Device(device_id), _streams[device_id]:
        cupy.get_default_memory_pool().free_all_blocks()
        log = logger.new_logger(intopt.mol, intopt.mol.verbose)
        t0 = log.init_timer()
        ncp_ij = len(intopt.log_qs)
        dm0 = cupy.asarray(dm0)
        orbo = cupy.asarray(orbo)
        naux = intopt.auxmol.nao
        nocc = orbo.shape[1]
        wj = cupy.zeros([naux,3])
        wk = None
        if with_k:
            wk = cupy.zeros([naux,nocc,nocc,3])
        for cp_k in task_list:
            k0, k1 = aux_ao_loc[cp_k], aux_ao_loc[cp_k+1]
            task_list = [(cp_k, cp_ij) for cp_ij in range(ncp_ij)]

            for i0,i1,j0,j1,k0,k1,int3c_blk in loop_int3c2e_general(intopt, task_list=task_list,
                                                                    ip_type='ip2', omega=omega):
                wj[k0:k1] += contract('xpji,ji->px', int3c_blk, dm0[j0:j1,i0:i1])
                if with_k:
                    tmp = contract('xpji,jo->piox', int3c_blk, orbo[j0:j1])
                    wk[k0:k1] += contract('piox,ir->prox', tmp, orbo[i0:i1])
                    tmp = None
                int3c_blk = None
        t0 = log.timer_debug1(f'int3c2e_ip2_wjk on Device {device_id}', *t0)
    return wj, wk

def get_int3c2e_ip2_wjk(intopt, dm0_tag, with_k=True, omega=None):
    orbo = cupy.asarray(dm0_tag.occ_coeff, order='C')
    futures = []

    aux_ao_loc = np.array(intopt.aux_ao_loc)
    loads = aux_ao_loc[1:] - aux_ao_loc[:-1]
    task_list = _split_tasks(loads, num_devices)

    cupy.cuda.get_current_stream().synchronize()
    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        for device_id in range(num_devices):
            future = executor.submit(
                _int3c2e_ip2_wjk, intopt, task_list[device_id],
                dm0_tag, orbo, with_k=with_k, device_id=device_id, omega=omega)
            futures.append(future)

    wj_total = []
    wk_total = []
    for future in futures:
        wj, wk = future.result()
        wj_total.append(wj)
        wk_total.append(wk)

    wj = wk = None
    wj = reduce_to_device(wj_total, inplace=True)
    if with_k:
        wk = reduce_to_device(wk_total, inplace=True)
    return wj, wk

def get_int3c2e_ip_slice(intopt, cp_aux_id, ip_type, out=None, omega=None, stream=None):
    '''
    Generate int3c2e_ip slice along k, full dimension in ij
    '''
    if omega is None: omega = 0.0
    if stream is None: stream = cupy.cuda.get_current_stream()
    nao = intopt.mol.nao
    naux = intopt.auxmol.nao

    norb = nao + naux + 1
    nbins = 1

    cp_kl_id = cp_aux_id + len(intopt.log_qs)
    log_q_kl = intopt.aux_log_qs[cp_aux_id]

    bins_locs_kl = np.array([0, len(log_q_kl)], dtype=np.int32)
    k0, k1 = intopt.cart_aux_loc[cp_aux_id], intopt.cart_aux_loc[cp_aux_id+1]

    nk = k1 - k0

    ao_offsets = np.array([0,0,nao+1+k0,nao], dtype=np.int32)
    if out is None:
        int3c_blk = cupy.zeros([3, nk, nao, nao], order='C', dtype=np.float64)
        strides = np.array([1, nao, nao*nao, nao*nao*nk], dtype=np.int32)
    else:
        int3c_blk = out
        # will be filled in f-contiguous
        strides = np.array([1, nao, nao*nao, nao*nao*nk], dtype=np.int32)
    if ip_type == 1:
        fn = libgint.GINTfill_int3c2e_ip1
    elif ip_type == 2:
        fn = libgint.GINTfill_int3c2e_ip2
    else:
        raise
    for cp_ij_id, log_q_ij in enumerate(intopt.log_qs):
        bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)
        err = fn(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            intopt.bpcache,
            ctypes.cast(int3c_blk.data.ptr, ctypes.c_void_p),
            ctypes.c_int(norb),
            strides.ctypes.data_as(ctypes.c_void_p),
            ao_offsets.ctypes.data_as(ctypes.c_void_p),
            bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
            bins_locs_kl.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nbins),
            ctypes.c_int(cp_ij_id),
            ctypes.c_int(cp_kl_id),
            ctypes.c_double(omega))

        if err != 0:
            raise RuntimeError(f'GINT_fill_int2e_ip failed, err={err}')

    return int3c_blk

def get_int3c2e_ip(mol, auxmol=None, ip_type=1, auxbasis='weigend+etb', direct_scf_tol=1e-13, omega=None, stream=None):
    '''
    Generate full int3c2e_ip tensor on GPU
    ip_type == 1: int3c2e_ip1
    ip_type == 2: int3c2e_ip2
    '''
    fn = getattr(libgint, 'GINTfill_int3c2e_' + ip_type)
    if omega is None: omega = 0.0
    if stream is None: stream = cupy.cuda.get_current_stream()
    if auxmol is None:
        auxmol = df.addons.make_auxmol(mol, auxbasis)

    nao = mol.nao
    naux = auxmol.nao

    intopt = VHFOpt(mol, auxmol, 'int2e')
    intopt.build(direct_scf_tol, diag_block_with_triu=True, aosym=False, group_size=BLKSIZE, group_size_aux=BLKSIZE)

    nao_cart = intopt.mol.nao
    naux_cart = intopt.auxmol.nao
    norb_cart = nao_cart + naux_cart + 1
    ao_loc = intopt.ao_loc
    aux_ao_loc = intopt.aux_ao_loc
    int3c = cupy.zeros([3, naux, nao, nao], order='C')
    nbins = 1
    for cp_ij_id, log_q_ij in enumerate(intopt.log_qs):
        cpi = intopt.cp_idx[cp_ij_id]
        cpj = intopt.cp_jdx[cp_ij_id]
        li = intopt.angular[cpi]
        lj = intopt.angular[cpj]

        for aux_id, log_q_kl in enumerate(intopt.aux_log_qs):
            cp_kl_id = aux_id + len(intopt.log_qs)
            i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi+1]
            j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj+1]
            k0, k1 = intopt.cart_aux_loc[aux_id], intopt.cart_aux_loc[aux_id+1]
            ni = i1 - i0
            nj = j1 - j0
            nk = k1 - k0
            lk = intopt.aux_angular[aux_id]

            bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)
            bins_locs_kl = np.array([0, len(log_q_kl)], dtype=np.int32)

            ao_offsets = np.array([i0,j0,nao+1+k0,nao], dtype=np.int32)
            strides = np.array([1, ni, ni*nj, ni*nj*nk], dtype=np.int32)

            int3c_blk = cupy.zeros([3, nk, nj, ni], order='C', dtype=np.float64)
            err = fn(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                intopt.bpcache,
                ctypes.cast(int3c_blk.data.ptr, ctypes.c_void_p),
                ctypes.c_int(norb_cart),
                strides.ctypes.data_as(ctypes.c_void_p),
                ao_offsets.ctypes.data_as(ctypes.c_void_p),
                bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                bins_locs_kl.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbins),
                ctypes.c_int(cp_ij_id),
                ctypes.c_int(cp_kl_id),
                ctypes.c_double(omega))

            if err != 0:
                raise RuntimeError("int3c2e_ip failed\n")

            if not intopt.auxmol.cart:
                int3c_blk = cart2sph(int3c_blk, axis=1, ang=lk)
            if not intopt.mol.cart:
                int3c_blk = cart2sph(int3c_blk, axis=2, ang=lj)
                int3c_blk = cart2sph(int3c_blk, axis=3, ang=li)

            i0, i1 = ao_loc[cpi], ao_loc[cpi+1]
            j0, j1 = ao_loc[cpj], ao_loc[cpj+1]
            k0, k1 = aux_ao_loc[aux_id], aux_ao_loc[aux_id+1]

            int3c[:, k0:k1, j0:j1, i0:i1] = int3c_blk
    int3c = intopt.unsort_orbitals(int3c, aux_axis=[1], axis=[2,3])
    return int3c.transpose([0,3,2,1])

def get_int3c2e_general(mol, auxmol=None, ip_type='', auxbasis='weigend+etb', direct_scf_tol=1e-13, omega=None, stream=None):
    '''
    Generate full int3c2e type tensor on GPU
    '''
    fn = getattr(libgint, 'GINTfill_int3c2e_' + ip_type)
    if ip_type == '':       order = 0
    if ip_type == 'ip1':    order = 1
    if ip_type == 'ip2':    order = 1
    if ip_type == 'ipip1':  order = 2
    if ip_type == 'ip1ip2': order = 2
    if ip_type == 'ipvip1': order = 2
    if ip_type == 'ipip2':  order = 2

    if omega is None: omega = 0.0
    if stream is None: stream = cupy.cuda.get_current_stream()
    if auxmol is None:
        auxmol = df.addons.make_auxmol(mol, auxbasis)
    assert omega >= 0

    nao = mol.nao
    naux = auxmol.nao

    intopt = VHFOpt(mol, auxmol, 'int2e')
    intopt.build(direct_scf_tol, diag_block_with_triu=True, aosym=False, group_size=BLKSIZE, group_size_aux=BLKSIZE)

    nao_cart = intopt._sorted_mol.nao
    naux_cart = intopt._sorted_auxmol.nao
    norb_cart = nao_cart + naux_cart + 1
    ao_loc = intopt.ao_loc
    aux_ao_loc = intopt.aux_ao_loc
    comp = 3**order
    int3c = cupy.zeros([comp, naux, nao, nao], order='C')
    nbins = 1
    for cp_ij_id, log_q_ij in enumerate(intopt.log_qs):
        cpi = intopt.cp_idx[cp_ij_id]
        cpj = intopt.cp_jdx[cp_ij_id]
        li = intopt.angular[cpi]
        lj = intopt.angular[cpj]

        for aux_id, log_q_kl in enumerate(intopt.aux_log_qs):
            cp_kl_id = aux_id + len(intopt.log_qs)
            i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi+1]
            j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj+1]
            k0, k1 = intopt.cart_aux_loc[aux_id], intopt.cart_aux_loc[aux_id+1]
            ni = i1 - i0
            nj = j1 - j0
            nk = k1 - k0
            lk = intopt.aux_angular[aux_id]

            bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)
            bins_locs_kl = np.array([0, len(log_q_kl)], dtype=np.int32)

            ao_offsets = np.array([i0,j0,nao_cart+1+k0,nao_cart], dtype=np.int32)
            strides = np.array([1, ni, ni*nj, ni*nj*nk], dtype=np.int32)

            # Use GPU kernels for low-angular momentum
            if (li + lj + lk + order)//2 + 1 < NROOT_ON_GPU:
                int3c_blk = cupy.zeros([comp, nk, nj, ni], order='C', dtype=np.float64)
                err = fn(
                    ctypes.cast(stream.ptr, ctypes.c_void_p),
                    intopt.bpcache,
                    ctypes.cast(int3c_blk.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(norb_cart),
                    strides.ctypes.data_as(ctypes.c_void_p),
                    ao_offsets.ctypes.data_as(ctypes.c_void_p),
                    bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                    bins_locs_kl.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nbins),
                    ctypes.c_int(cp_ij_id),
                    ctypes.c_int(cp_kl_id),
                    ctypes.c_double(omega))
                if err != 0:
                    raise RuntimeError("int3c2e failed\n")
            else:
                from pyscf.gto.moleintor import getints, make_cintopt
                pmol = intopt._tot_mol
                intor = pmol._add_suffix('int3c2e_' + ip_type)
                opt = make_cintopt(pmol._atm, pmol._bas, pmol._env, intor)

                # TODO: sph2cart in CPU?
                ishl0, ishl1 = intopt.l_ctr_offsets[cpi], intopt.l_ctr_offsets[cpi+1]
                jshl0, jshl1 = intopt.l_ctr_offsets[cpj], intopt.l_ctr_offsets[cpj+1]
                kshl0, kshl1 = intopt.l_ctr_offsets[aux_id+1+intopt.nctr], intopt.l_ctr_offsets[aux_id+1+intopt.nctr+1]
                shls_slice = np.array([ishl0, ishl1, jshl0, jshl1, kshl0, kshl1], dtype=np.int64)
                int3c_cpu = getints(intor, pmol._atm, pmol._bas, pmol._env, shls_slice, cintopt=opt).transpose([0,3,2,1])
                int3c_blk = cupy.asarray(int3c_cpu)

            if not intopt.auxmol.cart:
                int3c_blk = cart2sph(int3c_blk, axis=1, ang=lk)
            if not intopt.mol.cart:
                int3c_blk = cart2sph(int3c_blk, axis=2, ang=lj)
                int3c_blk = cart2sph(int3c_blk, axis=3, ang=li)

            i0, i1 = ao_loc[cpi], ao_loc[cpi+1]
            j0, j1 = ao_loc[cpj], ao_loc[cpj+1]
            k0, k1 = aux_ao_loc[aux_id], aux_ao_loc[aux_id+1]

            int3c[:, k0:k1, j0:j1, i0:i1] = int3c_blk

    int3c = intopt.unsort_orbitals(int3c, aux_axis=[1], axis=[2,3])
    return int3c.transpose([0,3,2,1])

def get_dh1e(mol, dm0):
    '''
    Contract 'int1e_iprinv', with density matrix
    xijk,ij->kx
    '''
    coords = mol.atom_coords()
    charges = cupy.asarray(mol.atom_charges(), dtype=np.float64)
    from gpu4pyscf.gto.int3c1e_ip import int1e_grids_ip2
    de0 = int1e_grids_ip2(mol, coords, charges=charges, dm=dm0)
    return de0.T

def get_d2h1e(mol, dm0):
    natm = mol.natm
    coords = mol.atom_coords()
    charges = cupy.asarray(mol.atom_charges(), dtype=np.float64)
    fakemol = gto.fakemol_for_charges(coords)
    fakemol.output = mol.output
    fakemol.stdout = mol.stdout
    fakemol.verbose = mol.verbose
    nao = mol.nao
    d2h1e_diag = cupy.zeros([natm,9])
    d2h1e_offdiag = cupy.zeros([natm, nao, 9])
    intopt = VHFOpt(mol, fakemol, 'int2e')
    intopt.build(1e-14, diag_block_with_triu=True, aosym=False, group_size=BLKSIZE, group_size_aux=BLKSIZE)
    dm0_sorted = intopt.sort_orbitals(dm0, axis=[0,1])
    for i0,i1,j0,j1,k0,k1,int3c_blk in loop_int3c2e_general(intopt, ip_type='ipip1'):
        d2h1e_diag[k0:k1,:9] -= contract('xaji,ij->ax', int3c_blk, dm0_sorted[i0:i1,j0:j1])
        d2h1e_offdiag[k0:k1,i0:i1,:9] += contract('xaji,ij->aix', int3c_blk, dm0_sorted[i0:i1,j0:j1])

    for i0,i1,j0,j1,k0,k1,int3c_blk in loop_int3c2e_general(intopt, ip_type='ipvip1'):
        d2h1e_diag[k0:k1,:9] -= contract('xaji,ij->ax', int3c_blk, dm0_sorted[i0:i1,j0:j1])
        d2h1e_offdiag[k0:k1,i0:i1,:9] += contract('xaji,ij->aix', int3c_blk, dm0_sorted[i0:i1,j0:j1])
    aoslices = mol.aoslice_by_atom()
    ao2atom = get_ao2atom(intopt, aoslices)
    d2h1e = contract('aix,ib->abx', d2h1e_offdiag, ao2atom)
    d2h1e[np.diag_indices(natm), :] += d2h1e_diag
    return 2.0 * contract('abx,a->xab', d2h1e, charges)

def get_int3c2e_slice(intopt, cp_ij_id, cp_aux_id, cart=False, aosym=None, out=None, omega=None, stream=None):
    '''
    Generate one int3c2e block for given ij, k
    '''
    if stream is None: stream = cupy.cuda.get_current_stream()
    if omega is None: omega = 0.0
    assert omega >= 0
    nao_cart = intopt._sorted_mol.nao
    naux_cart = intopt._sorted_auxmol.nao
    norb_cart = nao_cart + naux_cart + 1

    cpi = intopt.cp_idx[cp_ij_id]
    cpj = intopt.cp_jdx[cp_ij_id]
    cp_kl_id = cp_aux_id + len(intopt.log_qs)

    log_q_ij = intopt.log_qs[cp_ij_id]
    log_q_kl = intopt.aux_log_qs[cp_aux_id]

    nbins = 1
    bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)
    bins_locs_kl = np.array([0, len(log_q_kl)], dtype=np.int32)

    cart_ao_loc = intopt.cart_ao_loc
    cart_aux_loc = intopt.cart_aux_loc
    i0, i1 = cart_ao_loc[cpi], cart_ao_loc[cpi+1]
    j0, j1 = cart_ao_loc[cpj], cart_ao_loc[cpj+1]
    k0, k1 = cart_aux_loc[cp_aux_id], cart_aux_loc[cp_aux_id+1]

    ni = i1 - i0
    nj = j1 - j0
    nk = k1 - k0
    lk = intopt.aux_angular[cp_aux_id]

    ao_offsets = np.array([i0,j0,nao_cart+1+k0,nao_cart], dtype=np.int32)
    '''
    # if possible, write the data into the given allocated space
    # otherwise, need a temporary space for cart2sph
    '''
    if out is None or (lk > 1 and not intopt.auxmol.cart):
        int3c_blk = cupy.zeros([nk,nj,ni], order='C')
        strides = np.array([1, ni, ni*nj, 1], dtype=np.int32)
    else:
        int3c_blk = out
        s = int3c_blk.strides
        # will be filled in F order
        strides = np.array([s[2]//8 ,s[1]//8, s[0]//8, 1], dtype=np.int32)

    err = libgint.GINTfill_int3c2e(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        intopt.bpcache,
        ctypes.cast(int3c_blk.data.ptr, ctypes.c_void_p),
        ctypes.c_int(norb_cart),
        strides.ctypes.data_as(ctypes.c_void_p),
        ao_offsets.ctypes.data_as(ctypes.c_void_p),
        bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
        bins_locs_kl.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nbins),
        ctypes.c_int(cp_ij_id),
        ctypes.c_int(cp_kl_id),
        ctypes.c_double(omega))

    if err != 0:
        raise RuntimeError('GINT_fill_int2e failed')

    # move this operation to j2c?
    if lk > 1 and intopt.auxmol.cart == 0:
        int3c_blk = cart2sph(int3c_blk, axis=0, ang=lk, out=out)

    stream.synchronize()

    return int3c_blk

def get_int3c2e(mol, auxmol=None, auxbasis='weigend+etb', direct_scf_tol=1e-13, aosym=True, omega=None):
    '''
    Generate full int3c2e tensor on GPU
    '''
    if auxmol is None:
        auxmol = df.addons.make_auxmol(mol, auxbasis)

    nao = mol.nao
    naux = auxmol.nao
    intopt = VHFOpt(mol, auxmol, 'int2e')
    intopt.build(direct_scf_tol, diag_block_with_triu=True, aosym=aosym, group_size=BLKSIZE, group_size_aux=BLKSIZE)
    int3c = cupy.zeros([naux, nao, nao], order='C')
    for cp_ij_id, _ in enumerate(intopt.log_qs):
        cpi = intopt.cp_idx[cp_ij_id]
        cpj = intopt.cp_jdx[cp_ij_id]
        li = intopt.angular[cpi]
        lj = intopt.angular[cpj]
        i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi+1]
        j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj+1]

        int3c_slice = cupy.zeros([naux, j1-j0, i1-i0], order='C')
        for cp_kl_id, _ in enumerate(intopt.aux_log_qs):
            k0, k1 = intopt.aux_ao_loc[cp_kl_id], intopt.aux_ao_loc[cp_kl_id+1]
            get_int3c2e_slice(intopt, cp_ij_id, cp_kl_id, out=int3c_slice[k0:k1], omega=omega)
        i0, i1 = intopt.ao_loc[cpi], intopt.ao_loc[cpi+1]
        j0, j1 = intopt.ao_loc[cpj], intopt.ao_loc[cpj+1]
        if not mol.cart:
            int3c_slice = cart2sph(int3c_slice, axis=1, ang=lj)
            int3c_slice = cart2sph(int3c_slice, axis=2, ang=li)
        int3c[:, j0:j1, i0:i1] = int3c_slice
    if aosym:
        row, col = np.tril_indices(nao)
        int3c[:, row, col] = int3c[:, col, row]
    int3c = intopt.unsort_orbitals(int3c, aux_axis=[0], axis=[1,2])
    return int3c.transpose([2,1,0])

def sort_mol(mol0, cart=True, log=None):
    '''
    # Sort basis according to angular momentum and contraction patterns so
    # as to group the basis functions to blocks in GPU kernel.
    '''
    if log is None:
        log = logger.new_logger(mol0, mol0.verbose)
    mol = mol0.copy(deep=True)
    l_ctrs = mol._bas[:,[gto.ANG_OF, gto.NPRIM_OF]]
    uniq_l_ctr, _, inv_idx, l_ctr_counts = np.unique(
        l_ctrs, return_index=True, return_inverse=True, return_counts=True, axis=0)

    if mol.verbose >= logger.DEBUG:
        log.debug1('Number of shells for each [l, nctr] group')
        for l_ctr, n in zip(uniq_l_ctr, l_ctr_counts):
            log.debug('    %s : %s', l_ctr, n)

    sorted_idx = np.argsort(inv_idx.ravel(), kind='stable').astype(np.int32)

    # Sort basis inplace
    mol._bas = mol._bas[sorted_idx]
    return mol, sorted_idx, uniq_l_ctr, l_ctr_counts

def get_pairing(p_offsets, q_offsets, q_cond,
                cutoff=1e-14, diag_block_with_triu=True, aosym=True):
    '''
    pair shells and return pairing indices
    '''
    log_qs = []
    pair2bra = []
    pair2ket = []

    for p0, p1 in zip(p_offsets[:-1], p_offsets[1:]):
        for q0, q1 in zip(q_offsets[:-1], q_offsets[1:]):
            if aosym and q0 < p0 or not aosym:
                q_sub = q_cond[p0:p1,q0:q1].ravel()
                mask = q_sub > cutoff
                ishs, jshs = np.indices((p1-p0,q1-q0))
                ishs = ishs.ravel()[mask]
                jshs = jshs.ravel()[mask]
                ishs += p0
                jshs += q0
                pair2bra.append(ishs)
                pair2ket.append(jshs)
                log_q = np.log(q_sub[mask])
                log_q[log_q > 0] = 0
                log_qs.append(log_q)
            elif aosym and p0 == q0 and p1 == q1:
                q_sub = q_cond[p0:p1,p0:p1].ravel()
                ishs, jshs = np.indices((p1-p0, p1-p0))
                ishs = ishs.ravel()
                jshs = jshs.ravel()
                mask = q_sub > cutoff
                if not diag_block_with_triu:
                    # Drop the shell pairs in the upper triangle for diagonal blocks
                    mask &= ishs >= jshs

                ishs = ishs[mask]
                jshs = jshs[mask]
                ishs += p0
                jshs += p0
                if len(ishs) == 0 and len(jshs) == 0: continue

                pair2bra.append(ishs)
                pair2ket.append(jshs)

                log_q = np.log(q_sub[mask])
                log_q[log_q > 0] = 0
                log_qs.append(log_q)
    return log_qs, pair2bra, pair2ket

def _split_l_ctr_groups(uniq_l_ctr, l_ctr_counts, group_size):
    '''
    Splits l_ctr patterns into small groups with group_size the maximum
    number of AOs in each group
    '''
    l = uniq_l_ctr[:,0]
    nf = l * (l + 1) // 2
    _l_ctrs = []
    _l_ctr_counts = []
    for l_ctr, counts in zip(uniq_l_ctr, l_ctr_counts):
        l = l_ctr[0]
        nf = (l + 1) * (l + 2) // 2
        aligned_size = (group_size // nf // 1) * 1
        max_shells = max(aligned_size, 2)
        assert max_shells * nf <= group_size
        if l > LMAX_ON_GPU or counts <= max_shells:
            _l_ctrs.append(l_ctr)
            _l_ctr_counts.append(counts)
            continue

        nsubs, rests = counts.__divmod__(max_shells)
        _l_ctrs.extend([l_ctr] * nsubs)
        _l_ctr_counts.extend([max_shells] * nsubs)
        if rests > 0:
            _l_ctrs.append(l_ctr)
            _l_ctr_counts.append(rests)
    uniq_l_ctr = np.vstack(_l_ctrs)
    l_ctr_counts = np.hstack(_l_ctr_counts)
    return uniq_l_ctr, l_ctr_counts

def get_ao_pairs(pair2bra, pair2ket, ao_loc):
    """
    Compute the AO-pairs for the given pair2bra and pair2ket
    """
    bra_ctr = []
    ket_ctr = []
    for bra_shl, ket_shl in zip(pair2bra, pair2ket):
        if len(bra_shl) == 0 or len(ket_shl) == 0:
            bra_ctr.append(np.array([], dtype=np.int64))
            ket_ctr.append(np.array([], dtype=np.int64))
            continue

        i = bra_shl[0]
        j = ket_shl[0]
        indices = np.mgrid[:ao_loc[i+1]-ao_loc[i], :ao_loc[j+1]-ao_loc[j]]
        ao_bra = indices[0].reshape(-1,1) + ao_loc[bra_shl]
        ao_ket = indices[1].reshape(-1,1) + ao_loc[ket_shl]
        mask = ao_bra >= ao_ket
        bra_ctr.append(ao_bra[mask])
        ket_ctr.append(ao_ket[mask])
    return bra_ctr, ket_ctr
