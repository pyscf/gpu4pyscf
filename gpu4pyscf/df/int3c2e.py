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


import ctypes
import copy
import numpy as np
import cupy
from pyscf import gto, df, lib
from pyscf.scf import _vhf
from gpu4pyscf.scf.hf import BasisProdCache, _make_s_index_offsets
from gpu4pyscf.lib.cupy_helper import (
    block_c2s_diag, cart2sph, block_diag, contract, load_library, get_avail_mem, print_mem_info, take_last2d)
from gpu4pyscf.lib import logger
from gpu4pyscf.gto.mole import basis_seg_contraction

LMAX_ON_GPU = 8
FREE_CUPY_CACHE = True
STACK_SIZE_PER_THREAD = 8192 * 4
BLKSIZE = 128
NROOT_ON_GPU = 7

libgvhf = load_library('libgvhf')
libgint = load_library('libgint')
libcupy_helper = load_library('libcupy_helper')

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
        # use local basis_seg_contraction for efficiency
        # TODO: switch _mol and mol
        self.mol = basis_seg_contraction(mol,allow_replica=True)
        self.auxmol = basis_seg_contraction(auxmol, allow_replica=True)
        self._mol = mol
        self._auxmol = auxmol

        '''
        # Note mol._bas will be sorted in .build() method. VHFOpt should be
        # initialized after mol._bas updated.
        '''
        self.nao = self.mol.nao
        self.naux = self.auxmol.nao

        self._intor = intor
        self._prescreen = prescreen
        self._qcondname = qcondname
        self._dmcondname = dmcondname

        self.bpcache = None

        self.cart_ao_idx = None
        self.sph_ao_idx = None
        self.cart_aux_idx = None
        self.sph_aux_idx = None

        self.cart_ao_loc = []
        self.cart_aux_loc = []
        self.sph_ao_loc = []
        self.sph_aux_loc = []

        self.cart2sph = None
        self.aux_cart2sph = None

        self.angular = None
        self.aux_angular = None

        self.cp_idx = None
        self.cp_jdx = None

        self.log_qs = None
        self.aux_log_qs = None

    init_cvhf_direct = _vhf.VHFOpt.init_cvhf_direct

    def clear(self):
        _vhf.VHFOpt.__del__(self)
        if self.bpcache is not None:
            libgvhf.GINTdel_basis_prod(ctypes.byref(self.bpcache))
        return self

    def __del__(self):
        try:
            self.clear()
        except AttributeError:
            pass

    def build(self, cutoff=1e-14, group_size=None,
              group_size_aux=None, diag_block_with_triu=False, aosym=False):
        '''
        int3c2e is based on int2e with (ao,ao|aux,1)
        a tot_mol is created with concatenating [mol, fake_mol, aux_mol]
        we will pair (ao,ao) and (aux,1) separately.
        '''
        _mol = self._mol
        _auxmol = self._auxmol
        mol = self.mol
        auxmol = self.auxmol

        log = logger.new_logger(_mol, _mol.verbose)
        cput0 = log.init_timer()
        sorted_mol, sorted_idx, uniq_l_ctr, l_ctr_counts = sort_mol(mol, log=log)
        if group_size is not None :
            uniq_l_ctr, l_ctr_counts = _split_l_ctr_groups(uniq_l_ctr, l_ctr_counts, group_size)
        self.nctr = len(uniq_l_ctr)

        # sort fake mol
        fake_mol = make_fake_mol()
        _, _, fake_uniq_l_ctr, fake_l_ctr_counts = sort_mol(fake_mol, log=log)

        # sort auxiliary mol
        sorted_auxmol, sorted_aux_idx, aux_uniq_l_ctr, aux_l_ctr_counts = sort_mol(auxmol, log=log)
        if group_size_aux is not None:
            aux_uniq_l_ctr, aux_l_ctr_counts = _split_l_ctr_groups(aux_uniq_l_ctr, aux_l_ctr_counts, group_size_aux)

        tot_mol = sorted_mol + fake_mol + sorted_auxmol
        tot_mol.cart = True
        self.tot_mol = tot_mol

        # Initialize vhfopt after reordering mol._bas
        _vhf.VHFOpt.__init__(self, sorted_mol, self._intor, self._prescreen,
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
        cput1 = log.timer_debug1('Get pairing', *cput1)

        # contraction coefficient for ao basis
        cart_ao_loc = sorted_mol.ao_loc_nr(cart=True)
        sph_ao_loc = sorted_mol.ao_loc_nr(cart=False)
        self.cart_ao_loc = [cart_ao_loc[cp] for cp in l_ctr_offsets]
        self.sph_ao_loc = [sph_ao_loc[cp] for cp in l_ctr_offsets]
        self.angular = [l[0] for l in uniq_l_ctr]

        cart_ao_loc = mol.ao_loc_nr(cart=True)
        sph_ao_loc = mol.ao_loc_nr(cart=False)
        nao = sph_ao_loc[-1]
        ao_idx = np.array_split(np.arange(nao), sph_ao_loc[1:-1])
        self.sph_ao_idx = np.hstack([ao_idx[i] for i in sorted_idx])

        # cartesian ao index
        nao = cart_ao_loc[-1]
        ao_idx = np.array_split(np.arange(nao), cart_ao_loc[1:-1])
        self.cart_ao_idx = np.hstack([ao_idx[i] for i in sorted_idx])
        ncart = cart_ao_loc[-1]
        nsph = sph_ao_loc[-1]
        self.cart2sph = block_c2s_diag(ncart, nsph, self.angular, l_ctr_counts)

        if _mol.cart:
            inv_idx = np.argsort(self.cart_ao_idx, kind='stable').astype(np.int32)
            self.coeff = cupy.eye(ncart)[:,inv_idx]
        else:
            inv_idx = np.argsort(self.sph_ao_idx, kind='stable').astype(np.int32)
            self.coeff = self.cart2sph[:, inv_idx]
        cput1 = log.timer_debug1('AO cart2sph coeff', *cput1)

        # pairing auxiliary basis with fake basis set
        fake_l_ctr_offsets = np.append(0, np.cumsum(fake_l_ctr_counts))
        fake_l_ctr_offsets += l_ctr_offsets[-1]
        aux_l_ctr_offsets = np.append(0, np.cumsum(aux_l_ctr_counts))

        # contraction coefficient for auxiliary basis
        cart_aux_loc = sorted_auxmol.ao_loc_nr(cart=True)
        sph_aux_loc = sorted_auxmol.ao_loc_nr(cart=False)
        self.cart_aux_loc = [cart_aux_loc[cp] for cp in aux_l_ctr_offsets]
        self.sph_aux_loc = [sph_aux_loc[cp] for cp in aux_l_ctr_offsets]
        self.aux_angular = [l[0] for l in aux_uniq_l_ctr]

        cart_aux_loc = self.auxmol.ao_loc_nr(cart=True)
        sph_aux_loc = self.auxmol.ao_loc_nr(cart=False)
        naux = sph_aux_loc[-1]
        ao_idx = np.array_split(np.arange(naux), sph_aux_loc[1:-1])
        self.sph_aux_idx = np.hstack([ao_idx[i] for i in sorted_aux_idx])

        # cartesian aux index
        naux = cart_aux_loc[-1]
        ao_idx = np.array_split(np.arange(naux), cart_aux_loc[1:-1])
        self.cart_aux_idx = np.hstack([ao_idx[i] for i in sorted_aux_idx])
        ncart = cart_aux_loc[-1]
        nsph = sph_aux_loc[-1]
        self.aux_cart2sph = block_c2s_diag(ncart, nsph, self.aux_angular, aux_l_ctr_counts)

        if _auxmol.cart:
            inv_idx = np.argsort(self.cart_aux_idx, kind='stable').astype(np.int32)
            self.aux_coeff = cupy.eye(ncart)[:,inv_idx]
        else:
            inv_idx = np.argsort(self.sph_aux_idx, kind='stable').astype(np.int32)
            self.aux_coeff = self.aux_cart2sph[:, inv_idx]
        aux_l_ctr_offsets += fake_l_ctr_offsets[-1]
        cput1 = log.timer_debug1('aux cart2sph coeff', *cput1)

        ao_loc = sorted_mol.ao_loc_nr(cart=_mol.cart)
        self.ao_pairs_row, self.ao_pairs_col = get_ao_pairs(pair2bra, pair2ket, ao_loc)
        cderi_row = cupy.hstack(self.ao_pairs_row)
        cderi_col = cupy.hstack(self.ao_pairs_col)
        self.cderi_row = cderi_row
        self.cderi_col = cderi_col
        self.cderi_diag = cupy.argwhere(cderi_row == cderi_col)[:,0]
        cput1 = log.timer_debug1('Get AO pairs', *cput1)

        aux_pair2bra = []
        aux_pair2ket = []
        aux_log_qs = []
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
        bas_pair2shls = np.hstack(pair2bra + pair2ket).astype(np.int32).reshape(2,-1)
        bas_pairs_locs = np.append(0, np.cumsum([x.size for x in pair2bra])).astype(np.int32)
        log_qs = log_qs + aux_log_qs
        ao_loc = tot_mol.ao_loc_nr(cart=True)
        ncptype = len(log_qs)

        self.bpcache = ctypes.POINTER(BasisProdCache)()
        scale_shellpair_diag = 1.
        libgint.GINTinit_basis_prod(
            ctypes.byref(self.bpcache), ctypes.c_double(scale_shellpair_diag),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            bas_pair2shls.ctypes.data_as(ctypes.c_void_p),
            bas_pairs_locs.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ncptype),
            tot_mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(tot_mol.natm),
            tot_mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(tot_mol.nbas),
            tot_mol._env.ctypes.data_as(ctypes.c_void_p))

        cput1 = log.timer_debug1('Initialize GPU cache', *cput1)
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
            self.ao_idx = self.cart_ao_idx
        else:
            self.ao_loc = self.sph_ao_loc
            self.ao_idx = self.sph_ao_idx
        if _auxmol.cart:
            self.aux_ao_loc = self.cart_aux_loc
            self.aux_ao_idx = self.cart_aux_idx
        else:
            self.aux_ao_loc = self.sph_aux_loc
            self.aux_ao_idx = self.sph_aux_idx

        self.rev_ao_idx = np.argsort(self.ao_idx, kind='stable').astype(np.int32)
        self.ao_idx = cupy.array(self.ao_idx)
        self.cart_ao_idx = cupy.array(self.cart_ao_idx)
        self.sph_ao_idx = cupy.array(self.sph_ao_idx)
        self.aux_ao_idx = cupy.array(self.aux_ao_idx)
        self.cart_aux_idx = cupy.array(self.cart_aux_idx)
        self.sph_aux_idx = cupy.array(self.sph_aux_idx)
        self.rev_ao_idx = cupy.array(self.rev_ao_idx)

def get_int3c2e_wjk(mol, auxmol, dm0_tag, thred=1e-12, omega=None, with_k=True):
    log = logger.new_logger(mol, mol.verbose)
    intopt = VHFOpt(mol, auxmol, 'int2e')
    intopt.build(thred, diag_block_with_triu=True, aosym=True, group_size=BLKSIZE, group_size_aux=BLKSIZE)
    orbo = dm0_tag.occ_coeff
    nao = mol.nao
    naux = auxmol.nao
    nocc = orbo.shape[1]
    wj = cupy.empty([naux])
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
        rhoj_tmp = cupy.zeros([k1-k0], order='C')
        rhok_tmp = cupy.zeros([k1-k0, nao, nocc], order='C')

        for cp_ij_id, _ in enumerate(intopt.log_qs):
            cpi = intopt.cp_idx[cp_ij_id]
            cpj = intopt.cp_jdx[cp_ij_id]
            li = intopt.angular[cpi]
            lj = intopt.angular[cpj]
            int3c_blk = get_int3c2e_slice(intopt, cp_ij_id, cp_kl_id, omega=omega)
            if not intopt._mol.cart:
                int3c_blk = cart2sph(int3c_blk, axis=1, ang=lj)
                int3c_blk = cart2sph(int3c_blk, axis=2, ang=li)
            i0, i1 = intopt.ao_loc[cpi], intopt.ao_loc[cpi+1]
            j0, j1 = intopt.ao_loc[cpj], intopt.ao_loc[cpj+1]

            tmp = contract('Lji,ij->L', int3c_blk, dm0_tag[i0:i1,j0:j1])
            rhoj_tmp += tmp
            rhok_tmp[:,j0:j1] += contract('Lji,io->Ljo', int3c_blk, orbo[i0:i1])

            if cpi != cpj and intopt.aosym:
                rhoj_tmp += tmp
                rhok_tmp[:,i0:i1] += contract('Lji,jo->Lio', int3c_blk, orbo[j0:j1])
        wj[k0:k1] = rhoj_tmp
        if with_k:
            if isinstance(wk, cupy.ndarray):
                wk[k0:k1] = rhok_tmp
            else:
                rhok_tmp.get(out=wk[k0:k1])
    return wj, wk

def get_int3c2e_ip_jk(intopt, cp_aux_id, ip_type, rhoj, rhok, dm, omega=None):
    '''
    build jk with int3c2e slice (sliced in k dimension)
    '''
    fn = getattr(libgvhf, 'GINTbuild_int3c2e_' + ip_type + '_jk')
    if omega is None: omega = 0.0
    nao = intopt.mol.nao
    n_dm = 1

    cp_kl_id = cp_aux_id + len(intopt.log_qs)
    log_q_kl = intopt.aux_log_qs[cp_aux_id]

    k0, k1 = intopt.cart_aux_loc[cp_aux_id], intopt.cart_aux_loc[cp_aux_id+1]
    ao_offsets = np.array([0,0,nao+1+k0,nao], dtype=np.int32)
    nk = k1 - k0

    vj_ptr = vk_ptr = lib.c_null_ptr()
    rhoj_ptr = rhok_ptr = lib.c_null_ptr()
    vj = vk = None
    if rhoj is not None:
        assert(rhoj.flags['C_CONTIGUOUS'])
        rhoj_ptr = ctypes.cast(rhoj.data.ptr, ctypes.c_void_p)
        if ip_type == 'ip1':
            vj = cupy.zeros([3, nao], order='C')
        elif ip_type == 'ip2':
            vj = cupy.zeros([3, nk], order='C')
        vj_ptr = ctypes.cast(vj.data.ptr, ctypes.c_void_p)
    if rhok is not None:
        assert(rhok.flags['C_CONTIGUOUS'])
        rhok_ptr = ctypes.cast(rhok.data.ptr, ctypes.c_void_p)
        if ip_type == 'ip1':
            vk = cupy.zeros([3, nao], order='C')
        elif ip_type == 'ip2':
            vk = cupy.zeros([3, nk], order='C')
        vk_ptr = ctypes.cast(vk.data.ptr, ctypes.c_void_p)
    num_cp_ij = [len(log_qs) for log_qs in intopt.log_qs]
    bins_locs_ij = np.append(0, np.cumsum(num_cp_ij)).astype(np.int32)
    ntasks_kl = len(log_q_kl)
    ncp_ij = len(intopt.log_qs)
    err = fn(
        intopt.bpcache,
        vj_ptr,
        vk_ptr,
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

    return vj, vk


def loop_int3c2e_general(intopt, ip_type='', omega=None, stream=None):
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

    nao = intopt.mol.nao
    naux = intopt.auxmol.nao
    norb = nao + naux + 1
    ao_loc = intopt.ao_loc
    aux_ao_loc = intopt.aux_ao_loc
    comp = 3**order

    lmax = intopt.mol._bas[:gto.ANG_OF].max()
    aux_lmax = intopt.auxmol._bas[:gto.ANG_OF].max()
    nroots = (lmax + aux_lmax + order)//2 + 1
    if nroots > NROOT_ON_GPU:
        from pyscf.gto.moleintor import getints, make_cintopt
        pmol = intopt.tot_mol
        intor = pmol._add_suffix('int3c2e_' + ip_type)
        opt = make_cintopt(pmol._atm, pmol._bas, pmol._env, intor)

    nbins = 1

    for aux_id, log_q_kl in enumerate(intopt.aux_log_qs):
        cp_kl_id = aux_id + len(intopt.log_qs)
        lk = intopt.aux_angular[aux_id]

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
                # TODO: sph2cart in CPU?
                ishl0, ishl1 = intopt.l_ctr_offsets[cpi], intopt.l_ctr_offsets[cpi+1]
                jshl0, jshl1 = intopt.l_ctr_offsets[cpj], intopt.l_ctr_offsets[cpj+1]
                kshl0, kshl1 = intopt.l_ctr_offsets[aux_id+1+intopt.nctr], intopt.l_ctr_offsets[aux_id+1+intopt.nctr+1]
                shls_slice = np.array([ishl0, ishl1, jshl0, jshl1, kshl0, kshl1], dtype=np.int64)
                int3c_cpu = getints(intor, pmol._atm, pmol._bas, pmol._env, shls_slice, cintopt=opt).transpose([0,3,2,1])
                int3c_blk = cupy.asarray(int3c_cpu)

            if not intopt._auxmol.cart:
                int3c_blk = cart2sph(int3c_blk, axis=1, ang=lk)
            if not intopt._mol.cart:
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

    nao = len(intopt.ao_idx)
    nao_cart = intopt.mol.nao
    naux_cart = intopt.auxmol.nao
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
    ao_idx = intopt.ao_idx
    ao2atom = cupy.zeros([len(ao_idx), len(aoslices)])
    for ia, aoslice in enumerate(aoslices):
        _, _, p0, p1 = aoslice
        ao2atom[p0:p1,ia] = 1.0
    return ao2atom[ao_idx,:]

def get_aux2atom(intopt, auxslices):
    aux_ao_idx = intopt.aux_ao_idx
    aux2atom = cupy.zeros([len(aux_ao_idx), len(auxslices)])
    for ia, auxslice in enumerate(auxslices):
        _, _, p0, p1 = auxslice
        aux2atom[p0:p1,ia] = 1.0
    return aux2atom[aux_ao_idx,:]

def get_j_int3c2e_pass1(intopt, dm0):
    '''
    get rhoj pass1 for int3c2e
    '''
    n_dm = 1

    naux = len(intopt.cart_aux_idx)
    rhoj = cupy.zeros([naux])
    coeff = intopt.coeff
    dm_cart = coeff @ dm0 @ coeff.T

    num_cp_ij = [len(log_qs) for log_qs in intopt.log_qs]
    num_cp_kl = [len(log_qs) for log_qs in intopt.aux_log_qs]

    bins_locs_ij = np.append(0, np.cumsum(num_cp_ij)).astype(np.int32)
    bins_locs_kl = np.append(0, np.cumsum(num_cp_kl)).astype(np.int32)

    ncp_ij = len(intopt.log_qs)
    ncp_kl = len(intopt.aux_log_qs)
    norb = dm_cart.shape[0]
    err = libgvhf.GINTbuild_j_int3c2e_pass1(
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

    aux_coeff = intopt.aux_coeff
    rhoj = cupy.dot(rhoj, aux_coeff)
    return rhoj

def get_j_int3c2e_pass2(intopt, rhoj):
    '''
    get vj pass2 for int3c2e
    '''
    n_dm = 1
    norb = len(intopt.cart_ao_idx)
    naux = len(intopt.cart_aux_idx)
    vj = cupy.zeros([norb, norb])

    num_cp_ij = [len(log_qs) for log_qs in intopt.log_qs]
    num_cp_kl = [len(log_qs) for log_qs in intopt.aux_log_qs]

    bins_locs_ij = np.append(0, np.cumsum(num_cp_ij)).astype(np.int32)
    bins_locs_kl = np.append(0, np.cumsum(num_cp_kl)).astype(np.int32)

    ncp_ij = len(intopt.log_qs)
    ncp_kl = len(intopt.aux_log_qs)

    aux_coeff = intopt.aux_coeff
    rhoj = cupy.dot(aux_coeff, rhoj)

    err = libgvhf.GINTbuild_j_int3c2e_pass2(
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
    coeff = intopt.coeff
    vj = coeff.T @ vj @ coeff
    vj = vj + vj.T
    return vj

def get_int3c2e_jk(mol, auxmol, dm0_tag, with_k=True, omega=None):
    '''
    get rhoj and rhok for int3c2e
    '''
    intopt = VHFOpt(mol, auxmol, 'int2e')
    intopt.build(1e-14, diag_block_with_triu=True, aosym=True, group_size=BLKSIZE, group_size_aux=BLKSIZE)

    if omega is None: omega = 0.0
    naux = len(intopt.aux_ao_idx)
    orbo = cupy.asarray(dm0_tag.occ_coeff, order='C')
    nocc = orbo.shape[1]
    rhoj = cupy.empty([naux])
    rhok = cupy.empty([naux,nocc,nocc])

    for cp_kl_id, _ in enumerate(intopt.aux_log_qs):
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
            if not intopt._mol.cart:
                int3c_blk = cart2sph(int3c_blk, axis=1, ang=lj)
                int3c_blk = cart2sph(int3c_blk, axis=2, ang=li)
            i0, i1 = intopt.ao_loc[cpi], intopt.ao_loc[cpi+1]
            j0, j1 = intopt.ao_loc[cpj], intopt.ao_loc[cpj+1]
            if cpi == cpj and intopt.aosym:
                int3c_blk *= 0.5

            rhoj_tmp += contract('pji,ij->p', int3c_blk, dm0_tag[i0:i1,j0:j1])
            ints_o = contract('pji,jo->poi', int3c_blk, orbo[j0:j1])
            rhok_tmp += contract('poi,ir->por', ints_o, orbo[i0:i1])

        if intopt.aosym:
            rhoj[k0:k1] = 2.0 * rhoj_tmp
            rhok[k0:k1] = rhok_tmp + rhok_tmp.transpose([0,2,1])
        else:
            rhoj[k0:k1] = rhoj_tmp
            rhok[k0:k1] = rhok_tmp
    return rhoj, rhok

def get_int3c2e_ip1_vjk(intopt, rhoj, rhok, dm0_tag, aoslices, with_k=True, omega=None):
    '''
    # vj and vk responses (due to int3c2e_ip1) to changes in atomic positions
    '''
    ao2atom = get_ao2atom(intopt, aoslices)
    natom = len(aoslices)
    nao = len(intopt.ao_idx)
    orbo = cupy.asarray(dm0_tag.occ_coeff, order='C')
    nocc = orbo.shape[1]
    vj1_buf = cupy.zeros([3,nao,nao])
    vk1_buf = cupy.zeros([3,nao,nao])
    vj1 = cupy.zeros([natom,3,nao,nocc])
    vk1 = cupy.zeros([natom,3,nao,nocc])

    ncp_ij = len(intopt.log_qs)
    count = 0
    for i0,i1,j0,j1,k0,k1,int3c_blk in loop_int3c2e_general(intopt, ip_type='ip1', omega=omega):
        # initialize intermediate variables
        if count % ncp_ij == 0:
            rhok_tmp = cupy.asarray(rhok[k0:k1])
            if with_k:
                rhok0 = contract('pio,ir->pro', rhok_tmp, orbo)
                rhok0 = contract('pro,Jo->prJ', rhok0, orbo)
            rhoj0 = cupy.zeros([3,k1-k0,nao])
            int3c_ip1_occ = cupy.zeros([3,k1-k0,nao,nocc])

        vj1_buf[:,i0:i1,j0:j1] += contract('xpji,p->xij', int3c_blk, rhoj[k0:k1])
        rhoj0[:,:,i0:i1] += contract('xpji,ij->xpi', int3c_blk, dm0_tag[i0:i1,j0:j1])
        int3c_ip1_occ[:,:,i0:i1] += contract('xpji,jo->xpio', int3c_blk, orbo[j0:j1])

        if (count+1) % ncp_ij == 0:
            rhoj0_atom = contract('xpi,ia->xpa', rhoj0, 2.0*ao2atom)
            vj1 += contract('pJo,xpa->axJo', rhok_tmp, rhoj0_atom)
            rhoj0_atom = None
            vk1_buf += contract('xpio,plo->xil', int3c_ip1_occ, rhok_tmp)

            #vj1_ao = contract('pJo,xpi->xiJo', rhok_tmp, rhoj0)
            #vj1 += contract('xiJo,ia->axJo', vj1_ao, 2.0*ao2atom)
            #vj1_ao = None

        if with_k:
            #rhok0_slice = contract('pio,Jo->piJ', rhok_tmp, orbo[j0:j1])
            #vk1_buf[:,i0:i1] += contract('xpji,plj->xil', int3c_blk, rhok0_slice)
            #int3c_occ = contract('xpji,jo->xpio', int3c_blk, orbo[j0:j1])
            #vk1_buf[:,i0:i1] += contract('xpio,plo->xil', int3c_occ, rhok_tmp)

            vk1_ao = contract('xpji,poi->xijo', int3c_blk, rhok0[:,:,i0:i1])
            vk1[:,:,j0:j1] += contract('xijo,ia->axjo', vk1_ao, ao2atom[i0:i1])

            int3c_occ = contract('xpji,jo->xpio', int3c_blk, orbo[j0:j1])
            rhok0_slice = contract('pJr,ir->pJi', rhok_tmp, orbo[i0:i1])

            vk1_ao = contract('xpio,pJi->xiJo', int3c_occ, rhok0_slice)
            vk1 += contract('xiJo,ia->axJo', vk1_ao, ao2atom[i0:i1])
            vk1_ao = int3c_occ = None
        count += 1

    return vj1_buf, vk1_buf, vj1, vk1

def get_int3c2e_ip2_vjk(intopt, rhoj, rhok, dm0_tag, auxslices, with_k=True, omega=None):
    '''
    vj and vk responses (due to int3c2e_ip2) to changes in atomic positions
    '''
    aux2atom = get_aux2atom(intopt, auxslices)
    natom = len(auxslices)
    nao = len(intopt.ao_idx)
    orbo = cupy.asarray(dm0_tag.occ_coeff, order='C')
    nocc = orbo.shape[1]
    vj1 = cupy.zeros([natom,3,nao,nocc])
    vk1 = cupy.zeros([natom,3,nao,nocc])

    ncp_ij = len(intopt.log_qs)
    count = 0
    for i0,i1,j0,j1,k0,k1,int3c_blk in loop_int3c2e_general(intopt, ip_type='ip2', omega=omega):
        # initialize intermediate variables
        if count % ncp_ij == 0:
            wj2 = cupy.zeros([3,k1-k0])
            wk2_P__ = cupy.zeros([3,k1-k0,nao,nocc])

        # contraction
        wj2 += contract('xpji,ji->xp', int3c_blk, dm0_tag[j0:j1,i0:i1])
        wk2_P__[:,:,i0:i1] += contract('xpji,jo->xpio', int3c_blk, orbo[j0:j1])

        # reduction
        if (count+1) % ncp_ij == 0:
            rhok_tmp = cupy.asarray(rhok[k0:k1])
            vj1_tmp = -contract('pio,xp->xpio', rhok_tmp, wj2)
            vj1_tmp -= contract('xpio,p->xpio', wk2_P__, rhoj[k0:k1])

            vj1 += contract('xpio,pa->axio', vj1_tmp, aux2atom[k0:k1])
            if with_k:
                rhok0_slice = contract('pio,jo->pij', rhok_tmp, orbo)
                vk1_tmp = -contract('xpjo,pij->xpio', wk2_P__, rhok0_slice)

                rhok0_oo = contract('pio,ir->pro', rhok_tmp, orbo)
                vk1_tmp -= contract('xpio,pro->xpir', wk2_P__, rhok0_oo)

                vk1 += contract('xpir,pa->axir', vk1_tmp, aux2atom[k0:k1])
            wj2 = wk2_P__ = rhok0_slice = rhok0_oo = None
            rhok_tmp = vk1_tmp = None
        count += 1
    return vj1, vk1

def get_int3c2e_ip1_wjk(intopt, dm0_tag, with_k=True, omega=None):
    '''
    get wj and wk for int3c2e_ip1
    '''
    nao = len(intopt.ao_idx)
    naux = len(intopt.aux_ao_idx)
    orbo = cupy.asarray(dm0_tag.occ_coeff, order='C')
    nocc = orbo.shape[1]

    wj = cupy.zeros([naux,nao,3])
    avail_mem = get_avail_mem()
    use_gpu_memory = True
    if nao*naux*nocc*3*8 < 0.4*avail_mem:
        try:
            wk = cupy.empty([naux,nao,nocc,3])
        except Exception:
            use_gpu_memory = False
    else:
        use_gpu_memory = False
    
    if not use_gpu_memory:
        mem = cupy.cuda.alloc_pinned_memory(nao*naux*nocc*3*8)
        wk = np.ndarray([naux,nao,nocc,3], dtype=np.float64, order='C', buffer=mem)

    # TODO: async data transfer
    ncp_ij = len(intopt.log_qs)
    count = 0
    for i0,i1,j0,j1,k0,k1,int3c_blk in loop_int3c2e_general(intopt, ip_type='ip1', omega=omega):
        if count % ncp_ij == 0:
            wk_tmp = cupy.zeros([k1-k0,nao,nocc,3])
        wj[k0:k1,i0:i1] += contract('xpji,ij->pix', int3c_blk, dm0_tag[i0:i1,j0:j1])
        wk_tmp[:,i0:i1] += contract('xpji,jo->piox', int3c_blk, orbo[j0:j1])
        if (count+1) % ncp_ij == 0:
            if use_gpu_memory:
                wk[k0:k1] = wk_tmp
            else:
                wk_tmp.get(out=wk[k0:k1])
        count += 1
    return wj, wk

def get_int3c2e_ip2_wjk(intopt, dm0_tag, with_k=True, omega=None):
    '''
    get wj and wk for int3c2e_ip2
    '''
    naux = len(intopt.aux_ao_idx)
    orbo = cupy.asarray(dm0_tag.occ_coeff, order='C')
    nocc = orbo.shape[1]
    wj = cupy.zeros([naux,3])
    wk = cupy.zeros([naux,nocc,nocc,3])
    for i0,i1,j0,j1,k0,k1,int3c_blk in loop_int3c2e_general(intopt, ip_type='ip2', omega=omega):
        wj[k0:k1] += contract('xpji,ji->px', int3c_blk, dm0_tag[j0:j1,i0:i1])
        tmp = contract('xpji,jo->piox', int3c_blk, orbo[j0:j1])
        wk[k0:k1] += contract('piox,ir->prox', tmp, orbo[i0:i1])
    return wj, wk

def get_int3c2e_ipip1_hjk(intopt, rhoj, rhok, dm0_tag, with_k=True, omega=None):
    '''
    get hj and hk with int3c2e_ipip1
    '''
    nao_sph = dm0_tag.shape[0]
    orbo = cupy.asarray(dm0_tag.occ_coeff, order='C')
    hj = cupy.zeros([nao_sph,9])
    hk = None
    if with_k:
        hk = cupy.zeros([nao_sph,9])
    for i0,i1,j0,j1,k0,k1,int3c_blk in loop_int3c2e_general(intopt, ip_type='ipip1', omega=omega):
        tmp = contract('xpji,ij->xpi', int3c_blk, dm0_tag[i0:i1,j0:j1])
        hj[i0:i1] += contract('xpi,p->ix', tmp, rhoj[k0:k1])
        if with_k:
            rhok_tmp = contract('por,ir->pio', rhok[k0:k1], orbo[i0:i1])
            rhok_tmp = contract('pio,jo->pij', rhok_tmp, orbo[j0:j1])
            hk[i0:i1] += contract('xpji,pij->ix', int3c_blk, rhok_tmp)
    hj = hj.reshape([nao_sph,3,3])
    if with_k:
        hk = hk.reshape([nao_sph,3,3])
    return hj, hk

def get_int3c2e_ipvip1_hjk(intopt, rhoj, rhok, dm0_tag, with_k=True, omega=None):
    '''
    # get hj and hk with int3c2e_ipvip1
    '''
    nao_sph = dm0_tag.shape[0]
    orbo = cupy.asarray(dm0_tag.occ_coeff, order='C')
    hj = cupy.zeros([nao_sph,nao_sph,9])
    hk = None
    if with_k:
        hk = cupy.zeros([nao_sph,nao_sph,9])
    for i0,i1,j0,j1,k0,k1,int3c_blk in loop_int3c2e_general(intopt, ip_type='ipvip1', omega=omega):
        tmp = contract('xpji,ij->xpij', int3c_blk, dm0_tag[i0:i1,j0:j1])
        hj[i0:i1,j0:j1] += contract('xpij,p->ijx', tmp, rhoj[k0:k1])
        if with_k:
            rhok_tmp = contract('por,ir->pio', rhok[k0:k1], orbo[i0:i1])
            rhok_tmp = contract('pio,jo->pji', rhok_tmp, orbo[j0:j1])
            hk[i0:i1,j0:j1] += contract('xpji,pji->ijx', int3c_blk, rhok_tmp)
    hj = hj.reshape([nao_sph,nao_sph,3,3])
    if with_k:
        hk = hk.reshape([nao_sph,nao_sph,3,3])
    return hj, hk

def get_int3c2e_ip1ip2_hjk(intopt, rhoj, rhok, dm0_tag, with_k=True, omega=None):
    '''
    # get hj and hk with int3c2e_ip1ip2
    '''
    nao_sph = dm0_tag.shape[0]
    naux_sph = rhok.shape[0]
    orbo = cupy.asarray(dm0_tag.occ_coeff, order='C')
    hj = cupy.zeros([nao_sph,naux_sph,9])
    hk = None
    if with_k:
        hk = cupy.zeros([nao_sph,naux_sph,9])
    for i0,i1,j0,j1,k0,k1,int3c_blk in loop_int3c2e_general(intopt, ip_type='ip1ip2', omega=omega):
        tmp = contract('xpji,ij->xpi', int3c_blk, dm0_tag[i0:i1,j0:j1])
        hj[i0:i1,k0:k1] += contract('xpi,p->ipx', tmp, rhoj[k0:k1])
        if with_k:
            rhok_tmp = contract('por,ir->pio', rhok[k0:k1], orbo[i0:i1])
            rhok_tmp = contract('pio,jo->pij', rhok_tmp, orbo[j0:j1])
            hk[i0:i1,k0:k1] += contract('xpji,pij->ipx', int3c_blk, rhok_tmp)
    hj = hj.reshape([nao_sph,naux_sph,3,3])
    if with_k:
        hk = hk.reshape([nao_sph,naux_sph,3,3])
    return hj, hk

def get_int3c2e_ipip2_hjk(intopt, rhoj, rhok, dm0_tag, with_k=True, omega=None):
    '''
    # get hj and hk with int3c2e_ipip2
    '''
    naux_sph = rhok.shape[0]
    orbo = cupy.asarray(dm0_tag.occ_coeff, order='C')
    hj = cupy.zeros([naux_sph,9])
    hk = None
    if with_k:
        hk = cupy.zeros([naux_sph,9])
    for i0,i1,j0,j1,k0,k1,int3c_blk in loop_int3c2e_general(intopt, ip_type='ipip2', omega=omega):
        tmp = contract('xpji,ij->xp', int3c_blk, dm0_tag[i0:i1,j0:j1])
        hj[k0:k1] += contract('xp,p->px', tmp, rhoj[k0:k1])
        if with_k:
            rhok_tmp = contract('por,jr->pjo', rhok[k0:k1], orbo[j0:j1])
            rhok_tmp = contract('pjo,io->pji', rhok_tmp, orbo[i0:i1])
            hk[k0:k1] += contract('xpji,pji->px', int3c_blk, rhok_tmp)
    hj = hj.reshape([naux_sph,3,3])
    if with_k:
        hk = hk.reshape([naux_sph,3,3])
    return hj, hk

def get_hess_nuc_elec(mol, dm):
    '''
    calculate int1e_ipiprinv contribution
    '''
    coords = mol.atom_coords()
    charges = cupy.asarray(mol.atom_charges(), dtype=np.float64)

    fakemol = gto.fakemol_for_charges(coords)
    fakemol.output = mol.output
    fakemol.verbose = mol.verbose
    fakemol.stdout = mol.stdout
    intopt = VHFOpt(mol, fakemol, 'int2e')
    intopt.build(1e-14, diag_block_with_triu=True, aosym=False, group_size=BLKSIZE, group_size_aux=BLKSIZE)
    ao_idx = intopt.ao_idx
    dm = take_last2d(cupy.asarray(dm), ao_idx)

    natm = mol.natm
    nao = mol.nao
    hcore_diag = cupy.zeros([9,natm])
    hcore_aa = cupy.zeros([9,natm,nao])
    for i0,i1,j0,j1,k0,k1,int3c_blk in loop_int3c2e_general(intopt, ip_type='ipip1'):
        haa = contract('xpji,ij->xpi', int3c_blk, dm[i0:i1,j0:j1])
        hcore_aa[:,k0:k1,i0:i1] += haa
        hcore_diag[:,k0:k1] -= contract('xpji,ij->xp', int3c_blk, dm[i0:i1,j0:j1])

    hcore_ab = cupy.zeros([9,natm,nao])
    for i0,i1,j0,j1,k0,k1,int3c_blk in loop_int3c2e_general(intopt, ip_type='ipvip1'):
        hab = contract('xpji,ij->xpi', int3c_blk, dm[i0:i1,j0:j1])
        hcore_ab[:,k0:k1,i0:i1] += hab
        hcore_diag[:,k0:k1] -= contract('xpji,ij->xp', int3c_blk, dm[i0:i1,j0:j1])

    hcore_diag = contract('xp,p->xp', hcore_diag, charges)
    hcore_aa = contract('xpj,p->xpj', hcore_aa, charges)
    hcore_ab = contract('xpj,p->xpj', hcore_ab, charges)

    aoslices = mol.aoslice_by_atom()
    ao2atom = get_ao2atom(intopt, aoslices)

    hcore_aa = contract('xpj,jq->xpq', hcore_aa, ao2atom).reshape([3,3,natm,natm])
    hcore_ab = contract('xpj,jq->xpq', hcore_ab, ao2atom).reshape([3,3,natm,natm])
    hcore = hcore_aa + hcore_aa.transpose([1,0,3,2])
    hcore+= hcore_ab.transpose([1,0,2,3]) + hcore_ab.transpose([0,1,3,2])
    hcore_diag = hcore_diag.reshape([3,3,natm])
    idx = np.arange(natm)
    for x in range(3):
        for y in range(3):
            hcore[x,y,idx,idx] += hcore_diag[x,y]
    return hcore

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

            if not intopt._auxmol.cart:
                int3c_blk = cart2sph(int3c_blk, axis=1, ang=lk)
            if not intopt._mol.cart:
                int3c_blk = cart2sph(int3c_blk, axis=2, ang=lj)
                int3c_blk = cart2sph(int3c_blk, axis=3, ang=li)

            i0, i1 = ao_loc[cpi], ao_loc[cpi+1]
            j0, j1 = ao_loc[cpj], ao_loc[cpj+1]
            k0, k1 = aux_ao_loc[aux_id], aux_ao_loc[aux_id+1]

            int3c[:, k0:k1, j0:j1, i0:i1] = int3c_blk
    ao_idx = np.argsort(intopt.ao_idx)
    aux_idx = np.argsort(intopt.aux_ao_idx)
    int3c = int3c[cupy.ix_(np.arange(3), aux_idx, ao_idx, ao_idx)]

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

    nao = mol.nao
    naux = auxmol.nao

    intopt = VHFOpt(mol, auxmol, 'int2e')
    intopt.build(direct_scf_tol, diag_block_with_triu=True, aosym=False, group_size=BLKSIZE, group_size_aux=BLKSIZE)

    lmax = mol._bas[:gto.ANG_OF].max()
    aux_lmax = auxmol._bas[:gto.ANG_OF].max()
    nroots = (lmax + aux_lmax + order)//2 + 1
    if nroots > NROOT_ON_GPU:
        from pyscf.gto.moleintor import getints, make_cintopt
        mol = intopt.mol
        pmol = intopt.tot_mol
        intor = pmol._add_suffix('int3c2e_' + ip_type)
        opt = make_cintopt(pmol._atm, pmol._bas, pmol._env, intor)

    nao_cart = intopt.mol.nao
    naux_cart = intopt.auxmol.nao
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
                # TODO: sph2cart in CPU?
                ishl0, ishl1 = intopt.l_ctr_offsets[cpi], intopt.l_ctr_offsets[cpi+1]
                jshl0, jshl1 = intopt.l_ctr_offsets[cpj], intopt.l_ctr_offsets[cpj+1]
                kshl0, kshl1 = intopt.l_ctr_offsets[aux_id+1+intopt.nctr], intopt.l_ctr_offsets[aux_id+1+intopt.nctr+1]
                shls_slice = np.array([ishl0, ishl1, jshl0, jshl1, kshl0, kshl1], dtype=np.int64)
                int3c_cpu = getints(intor, pmol._atm, pmol._bas, pmol._env, shls_slice, cintopt=opt).transpose([0,3,2,1])
                int3c_blk = cupy.asarray(int3c_cpu)

            if not intopt._auxmol.cart:
                int3c_blk = cart2sph(int3c_blk, axis=1, ang=lk)
            if not intopt._mol.cart:
                int3c_blk = cart2sph(int3c_blk, axis=2, ang=lj)
                int3c_blk = cart2sph(int3c_blk, axis=3, ang=li)

            i0, i1 = ao_loc[cpi], ao_loc[cpi+1]
            j0, j1 = ao_loc[cpj], ao_loc[cpj+1]
            k0, k1 = aux_ao_loc[aux_id], aux_ao_loc[aux_id+1]

            int3c[:, k0:k1, j0:j1, i0:i1] = int3c_blk

    ao_idx = np.argsort(intopt.ao_idx)
    aux_idx = np.argsort(intopt.aux_ao_idx)
    int3c = int3c[cupy.ix_(np.arange(comp), aux_idx, ao_idx, ao_idx)]

    return int3c.transpose([0,3,2,1])

def get_dh1e(mol, dm0):
    '''
    contract 'int1e_iprinv', with density matrix
    xijk,ij->kx
    '''
    natm = mol.natm
    coords = mol.atom_coords()
    charges = cupy.asarray(mol.atom_charges(), dtype=np.float64)
    fakemol = gto.fakemol_for_charges(coords)
    fakemol.output = mol.output
    fakemol.verbose = mol.verbose
    fakemol.stdout = mol.stdout
    intopt = VHFOpt(mol, fakemol, 'int2e')
    intopt.build(1e-14, diag_block_with_triu=True, aosym=False, group_size=BLKSIZE, group_size_aux=BLKSIZE)
    dm0_sorted = take_last2d(dm0, intopt.ao_idx)
    dh1e = cupy.zeros([natm,3])
    for i0,i1,j0,j1,k0,k1,int3c_blk in loop_int3c2e_general(intopt, ip_type='ip1'):
        dh1e[k0:k1,:3] += contract('xkji,ij->kx', int3c_blk, dm0_sorted[i0:i1,j0:j1])
    return 2.0 * contract('kx,k->kx', dh1e, -charges)

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
    dm0_sorted = take_last2d(dm0, intopt.ao_idx)
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
    nao_cart = intopt.mol.nao
    naux_cart = intopt.auxmol.nao
    norb_cart = nao_cart + naux_cart + 1

    cpi = intopt.cp_idx[cp_ij_id]
    cpj = intopt.cp_jdx[cp_ij_id]
    cp_kl_id = cp_aux_id + len(intopt.log_qs)

    log_q_ij = intopt.log_qs[cp_ij_id]
    log_q_kl = intopt.aux_log_qs[cp_aux_id]

    nbins = 1
    bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)
    bins_locs_kl = np.array([0, len(log_q_kl)], dtype=np.int32)

    i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi+1]
    j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj+1]
    k0, k1 = intopt.cart_aux_loc[cp_aux_id], intopt.cart_aux_loc[cp_aux_id+1]

    ni = i1 - i0
    nj = j1 - j0
    nk = k1 - k0
    lk = intopt.aux_angular[cp_aux_id]

    ao_offsets = np.array([i0,j0,nao_cart+1+k0,nao_cart], dtype=np.int32)
    '''
    # if possible, write the data into the given allocated space
    # otherwise, need a temporary space for cart2sph
    '''
    if out is None or (lk > 1 and not intopt._auxmol.cart):
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
    if lk > 1 and intopt._auxmol.cart == 0:
        int3c_blk = cart2sph(int3c_blk, axis=0, ang=lk, out=out)
    return int3c_blk

def get_int3c2e(mol, auxmol=None, auxbasis='weigend+etb', direct_scf_tol=1e-13, aosym=True, omega=None):
    '''
    Generate full int3c2e tensor on GPU
    '''
    if auxmol is None:
        auxmol = df.addons.make_auxmol(mol, auxbasis)
    assert(aosym)

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
    row, col = np.tril_indices(nao)
    int3c[:, row, col] = int3c[:, col, row]
    ao_idx = np.argsort(intopt.ao_idx)
    aux_id = np.argsort(intopt.aux_ao_idx)
    int3c = int3c[np.ix_(aux_id, ao_idx, ao_idx)]

    return int3c.transpose([2,1,0])

def get_int2c2e_sorted(mol, auxmol, intopt=None, direct_scf_tol=1e-13, aosym=None, omega=None, stream=None):
    '''
    Generated int2c2e consistent with pyscf
    '''
    if omega is None: omega = 0.0
    if stream is None: stream = cupy.cuda.get_current_stream()
    if intopt is None:
        intopt = VHFOpt(mol, auxmol, 'int2e')
        intopt.build(direct_scf_tol, diag_block_with_triu=True, aosym=False)
    naux = auxmol.nao
    rows, cols = np.tril_indices(naux)

    nbins = 1

    nao_cart = intopt.mol.nao
    naux_cart = intopt.auxmol.nao
    norb_cart = nao_cart + naux_cart + 1

    int2c = cupy.zeros([naux_cart, naux_cart], order='F')
    ao_offsets = np.array([nao_cart+1, nao_cart, nao_cart+1, nao_cart], dtype=np.int32)
    strides = np.array([1, naux_cart, naux_cart, naux_cart*naux_cart], dtype=np.int32)
    for k_id, log_q_k in enumerate(intopt.aux_log_qs):
        bins_locs_k = _make_s_index_offsets(log_q_k, nbins)
        cp_k_id = k_id + len(intopt.log_qs)
        for l_id, log_q_l in enumerate(intopt.aux_log_qs):
            if k_id > l_id: continue
            bins_locs_l = _make_s_index_offsets(log_q_l, nbins)
            cp_l_id = l_id + len(intopt.log_qs)
            err = libgint.GINTfill_int2e(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                intopt.bpcache,
                ctypes.cast(int2c.data.ptr, ctypes.c_void_p),
                ctypes.c_int(norb_cart),
                strides.ctypes.data_as(ctypes.c_void_p),
                ao_offsets.ctypes.data_as(ctypes.c_void_p),
                bins_locs_k.ctypes.data_as(ctypes.c_void_p),
                bins_locs_l.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbins),
                ctypes.c_int(cp_k_id),
                ctypes.c_int(cp_l_id),
                ctypes.c_double(omega))

            if err != 0:
                raise RuntimeError("int2c2e failed\n")

    int2c[rows, cols] = int2c[cols, rows]
    if not mol.cart:
        coeff = intopt.aux_cart2sph
        int2c = coeff.T @ int2c @ coeff

    return int2c

def get_int2c2e_ip_sorted(mol, auxmol, intopt=None, direct_scf_tol=1e-13, intor=None, aosym=None, stream=None):
    '''
    TODO: WIP
    '''
    if stream is None: stream = cupy.cuda.get_current_stream()
    if intopt is None:
        intopt = VHFOpt(mol, auxmol, 'int2e')
        intopt.build(direct_scf_tol, diag_block_with_triu=True, aosym=False)

    nbins = 1

    nao_cart = intopt.mol.nao
    naux_cart = intopt.auxmol.nao
    norb_cart = nao_cart + naux_cart + 1
    rows, cols = np.tril_indices(naux_cart)

    int2c = cupy.zeros([naux_cart, naux_cart], order='F')
    ao_offsets = np.array([nao_cart+1, nao_cart, nao_cart+1, nao_cart], dtype=np.int32)
    strides = np.array([1, naux_cart, naux_cart, naux_cart*naux_cart], dtype=np.int32)
    for k_id, log_q_k in enumerate(intopt.aux_log_qs):
        bins_locs_k = _make_s_index_offsets(log_q_k, nbins)
        cp_k_id = k_id + len(intopt.log_qs)
        for l_id, log_q_l in enumerate(intopt.aux_log_qs):
            if k_id > l_id: continue
            bins_locs_l = _make_s_index_offsets(log_q_l, nbins)
            cp_l_id = l_id + len(intopt.log_qs)
            err = libgint.GINTfill_int2e(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                intopt.bpcache,
                ctypes.cast(int2c.data.ptr, ctypes.c_void_p),
                ctypes.c_int(norb_cart),
                strides.ctypes.data_as(ctypes.c_void_p),
                ao_offsets.ctypes.data_as(ctypes.c_void_p),
                bins_locs_k.ctypes.data_as(ctypes.c_void_p),
                bins_locs_l.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbins),
                ctypes.c_int(cp_k_id),
                ctypes.c_int(cp_l_id))

            if err != 0:
                raise RuntimeError("int2c2e failed\n")

    int2c[rows, cols] = int2c[cols, rows]
    if not auxmol.cart:
        coeff = intopt.aux_cart2sph
        int2c = coeff.T @ int2c @ coeff

    return int2c

def get_int2c2e(mol, auxmol, direct_scf_tol=1e-13):
    '''
    Generate int2c2e on GPU
    '''
    intopt = VHFOpt(mol, auxmol, 'int2e')
    intopt.build(direct_scf_tol, diag_block_with_triu=True, aosym=True)
    int2c = get_int2c2e_sorted(mol, auxmol, intopt=intopt)
    aux_idx = np.argsort(intopt.aux_ao_idx)
    int2c = int2c[np.ix_(aux_idx, aux_idx)]
    return int2c

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
