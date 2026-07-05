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


import ctypes
import cupy
import numpy as np
import cupy as cp
from cupyx.scipy.linalg import solve_triangular
from pyscf import lib
from pyscf.df import addons, incore
from gpu4pyscf.lib.cupy_helper import (
    cholesky, tag_array, get_avail_mem, cart2sph, copy_array,
    asarray, empty_mapped, ndarray)
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import utils
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.df import df_jk_o1
from gpu4pyscf.df import int3c2e_bdiv
from gpu4pyscf.gto.mole import SortedMole
from gpu4pyscf import __config__

libvhf_rys = int3c2e_bdiv.libvhf_rys
num_devices = multi_gpu.num_devices

MIN_BLK_SIZE = 256
ALIGNED = getattr(__config__, 'ao_aligned', 32)

LINEAR_DEP_THR = incore.LINEAR_DEP_THR

class DF(lib.StreamObject):

    use_host_memory = True

    _keys = {'intopt', 'nao', 'naux', 'cd_low', 'mol', 'auxmol', 'use_host_memory'}

    def __init__(self, mol, auxbasis=None):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory
        self._auxbasis = auxbasis

        self.auxmol = None
        self.intopt = None
        self.nao = None
        self.cd_low = None
        self._cderi = None
        self._rsh_df = {}

    __getstate__, __setstate__ = lib.generate_pickle_methods(
        excludes=('cd_low', 'intopt', '_cderi'))

    @property
    def auxbasis(self):
        return self._auxbasis
    @auxbasis.setter
    def auxbasis(self, x):
        if self._auxbasis != x:
            self.reset()
            self._auxbasis = x

    to_gpu = utils.to_gpu
    device = utils.device

    def to_cpu(self):
        from pyscf.df.df import DF
        return utils.to_cpu(self, out=DF(self.mol, auxbasis=self.auxbasis))

    def build(self, direct_scf_tol=1e-14, omega=None):
        mol = self.mol
        auxmol = self.auxmol
        self.nao = mol.nao
        log = logger.new_logger(mol, mol.verbose)
        t0 = log.init_timer()
        if auxmol is None:
            self.auxmol = auxmol = addons.make_auxmol(mol, self.auxbasis)

        self.intopt = intopt = int3c2e_bdiv.Int3c2eOpt(mol, auxmol)
        intopt.mol = SortedMole.from_mol(mol, decontract=True)
        intopt.build()
        self._cderi, self._cderi_idx = _cholesky_eri(
            intopt, omega=None, use_host_memory=self.use_host_memory)
        rows, cols = divmod(self._cderi_idx[0], self.nao)
        self.cderi_row  = intopt.cderi_row = rows
        self.cderi_col  = intopt.cderi_col = cols
        self.cderi_diag = intopt.cderi_diag = self._cderi_idx[1]
        log.timer_debug1('cholesky_eri', *t0)
        return self

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True,
               direct_scf_tol=getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13),
               omega=None):
        if omega is None:
            return df_jk_o1.get_jk(self, dm, hermi, with_j, with_k, direct_scf_tol)
        assert omega >= 0.0

        # A temporary treatment for RSH-DF integrals
        # TODO: use the range_coulomb context from pyscf
        key = '%.6f' % omega
        if key in self._rsh_df:
            rsh_df = self._rsh_df[key]
        else:
            rsh_df = self._rsh_df[key] = self.copy().reset()
            logger.info(self, 'Create RSH-DF object %s for omega=%s', rsh_df, omega)

        return df_jk_o1.get_jk(rsh_df, dm, hermi, with_j, with_k, direct_scf_tol, omega=omega)

    def get_blksize(self, extra=0, nao=None):
        '''
        extra for pre-calculated space for other variables
        '''
        if nao is None: nao = self.nao
        mem_avail = get_avail_mem()
        blksize = int(mem_avail*0.2/8/(nao*nao + extra) / ALIGNED) * ALIGNED
        blksize = min(blksize, MIN_BLK_SIZE)
        log = logger.new_logger(self.mol, self.mol.verbose)
        device_id = cupy.cuda.Device().id
        log.debug(f"{mem_avail/1e9:.3f} GB memory available on Device {device_id}, block size = {blksize}")
        assert blksize > 0
        return blksize

    def loop(self, blksize=None, unpack=True):
        ''' loop over cderi for the current device
            and unpack the CDERI in (Lij) format
        '''
        if self._cderi is None:
            self.build()

        device_id = cupy.cuda.Device().id
        cderi_sparse = self._cderi[device_id]
        nao = self.nao
        naux_slice = cderi_sparse.shape[0]
        if blksize is None:
            blksize = self.get_blksize()
        blksize = min(blksize, naux_slice)
        on_gpu = isinstance(cderi_sparse, cp.ndarray)

        if unpack:
            work = cp.zeros((nao, nao, blksize))

        pair_idx = cp.asarray(self._cderi_idx[0], dtype=np.int32)
        npairs = len(pair_idx)
        if on_gpu:
            for p0, p1 in lib.prange(0, naux_slice, blksize):
                cderi_blk = cderi_sparse[p0:p1]
                if not unpack:
                    yield None, cderi_blk
                    continue

                out = work[:,:,:p1-p0]
                if cderi_sparse.flags.c_contiguous:
                    err = libvhf_rys.decompress_and_transpose(
                        ctypes.cast(out.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(blksize),
                        ctypes.cast(cderi_sparse.data.ptr, ctypes.c_void_p),
                        ctypes.cast(pair_idx.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(npairs),
                        ctypes.c_int(nao),
                        ctypes.c_int(naux_slice),
                        ctypes.c_int(p0), ctypes.c_int(p1),
                        ctypes.c_int(1), ctypes.c_int(0))
                else:
                    err = libvhf_rys.decompress_and_fill(
                        ctypes.cast(out.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(blksize),
                        ctypes.cast(cderi_sparse.data.ptr, ctypes.c_void_p),
                        ctypes.cast(pair_idx.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(npairs),
                        ctypes.c_int(nao),
                        ctypes.c_int(naux_slice),
                        ctypes.c_int(p0), ctypes.c_int(p1))
                if err != 0:
                    raise RuntimeError('decompress cderi failed')
                yield out.transpose(2,0,1), cderi_blk

        else:
            buf = cp.empty((blksize, npairs))
            buf_prefetch = cp.empty_like(buf)

            comput_stream = cp.cuda.get_current_stream()
            compute_event = cp.cuda.Event()
            io_stream = cp.cuda.stream.Stream(non_blocking=True)
            io_event = cp.cuda.Event()

            buf_prefetch.set(cderi_sparse[:blksize], stream=io_stream)
            io_event.record(io_stream)

            for p0, p1 in lib.prange(0, naux_slice, blksize):
                buf, buf_prefetch = buf_prefetch, buf
                cderi_blk = buf[:p1-p0]
                comput_stream.wait_event(io_event)

                # prefetch the next block
                p2 = min(naux_slice, p1 + blksize)
                if p1 < p2:
                    io_stream.wait_event(compute_event)
                    buf_prefetch[:p2-p1].set(cderi_sparse[p1:p2], stream=io_stream)
                    io_event.record(io_stream)

                if not unpack:
                    yield None, cderi_blk
                    compute_event.record(comput_stream)
                    continue

                out = work[:,:,:p1-p0]
                err = libvhf_rys.decompress_and_transpose(
                    ctypes.cast(out.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(blksize),
                    ctypes.cast(cderi_blk.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_idx.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(npairs),
                    ctypes.c_int(nao),
                    ctypes.c_int(p1-p0),
                    ctypes.c_int(0), ctypes.c_int(p1-p0),
                    ctypes.c_int(1), ctypes.c_int(0))
                if err != 0:
                    raise RuntimeError('decompress cderi failed')
                yield out.transpose(2,0,1), cderi_blk
                compute_event.record(comput_stream)

    def reset(self, mol=None):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        if mol is not None:
            self.mol = mol
            self.auxmol = None
        self._cderi = None
        self._rsh_df = {}
        self.intopt = None
        self.nao = None
        self.cd_low = None
        return self

    get_ao_eri = get_eri = NotImplemented
    get_mo_eri = ao2mo = NotImplemented

def cholesky_eri_gpu(intopt, mol, auxmol, cd_low,
                     omega=None, sr_only=False, use_gpu_memory=True):
    '''
    Returns:
        2D array of (naux,nao*(nao+1)/2) in C-contiguous
    '''
    import warnings
    from concurrent.futures import ThreadPoolExecutor
    warnings.warn(
        'cholesky_eri_gpu is deprecated',
        DeprecationWarning, stacklevel=2)

    GB = 1024*1024*1024

    naux = cd_low.shape[1]
    npairs = len(intopt.cderi_row)
    log = logger.new_logger(mol, mol.verbose)

    # Available memory on Device 0.
    avail_mem = get_avail_mem()

    if use_gpu_memory:
        # CDERI will be equally distributed to the devices
        # Other devices usually have more memory available than Device 0
        # CDERI will use up to 40% of the available memory
        use_gpu_memory = naux * npairs * 8 < 0.4 * avail_mem * num_devices

    if use_gpu_memory:
        log.debug("Saving CDERI on GPU")
    else:
        log.debug("Saving CDERI on CPU")

    _cderi = {}
    aux_blksize = (naux + num_devices - 1) // num_devices
    aux_blksize = (aux_blksize + ALIGNED - 1) // ALIGNED * ALIGNED
    for device_id in range(num_devices):
        p0 = min(aux_blksize*device_id, naux)
        p1 = min(aux_blksize*(device_id+1), naux)
        #for device_id, (p0,p1) in enumerate(lib.prange(0, naux, aux_blksize)):
        if use_gpu_memory:
            with cupy.cuda.Device(device_id):
                _cderi[device_id] = cupy.empty([p1-p0, npairs])
            log.debug(f"CDERI size {_cderi[device_id].nbytes/GB:.3f} GB on Device {device_id}")
        else:
            _cderi[device_id] = empty_mapped((p1-p0, npairs), dtype=np.float64)

    npairs_per_ctr = [len(intopt.ao_pairs_row[cp_ij_id]) for cp_ij_id in range(len(intopt.log_qs))]

    npairs_per_ctr = np.array(npairs_per_ctr)
    total_task_list = np.argsort(npairs_per_ctr)
    task_list_per_device = []
    for device_id in range(num_devices):
        task_list_per_device.append(total_task_list[device_id::num_devices])

    cd_low_f = cupy.asarray(cd_low, order='F')
    cd_low_f = tag_array(cd_low_f, tag=cd_low.tag)

    cupy.cuda.get_current_stream().synchronize()
    futures = []
    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        for device_id in range(num_devices):
            task_list = task_list_per_device[device_id]
            future = executor.submit(_cderi_task, intopt, cd_low_f, task_list, _cderi, aux_blksize,
                                     omega=omega, sr_only=sr_only, device_id=device_id)
            futures.append(future)

    for future in futures:
        future.result()

    if not use_gpu_memory:
        cupy.cuda.Device().synchronize()

    return _cderi

def _cderi_task(intopt, cd_low, task_list, _cderi, aux_blksize,
                omega=None, sr_only=False, device_id=0):
    ''' Execute CDERI tasks on one device
    '''
    from gpu4pyscf.df import int3c2e
    nq = len(intopt.log_qs)
    mol = intopt.mol
    naux = cd_low.shape[1]
    naoaux = cd_low.shape[0]
    npairs = [len(intopt.ao_pairs_row[cp_ij]) for cp_ij in range(len(intopt.log_qs))]
    pairs_loc = np.append(0, np.cumsum(npairs))
    with cupy.cuda.Device(device_id):
        assert isinstance(mol.verbose, int)
        log = logger.new_logger(mol, mol.verbose)
        t1 = log.init_timer()
        cd_low_tag = cd_low.tag
        cd_low = asarray(cd_low)

        cart_ao_loc = intopt.cart_ao_loc
        aux_ao_loc = intopt.aux_ao_loc
        ao_loc = intopt.ao_loc
        for cp_ij_id in task_list:
            cpi = intopt.cp_idx[cp_ij_id]
            cpj = intopt.cp_jdx[cp_ij_id]
            li = intopt.angular[cpi]
            lj = intopt.angular[cpj]
            i0, i1 = cart_ao_loc[cpi], cart_ao_loc[cpi+1]
            j0, j1 = cart_ao_loc[cpj], cart_ao_loc[cpj+1]
            ni = i1 - i0
            nj = j1 - j0
            if sr_only:
                # TODO: in-place implementation or short-range kernel
                ints_slices = cupy.zeros([naoaux, nj, ni], order='C')
                for cp_kl_id, _ in enumerate(intopt.aux_log_qs):
                    k0 = aux_ao_loc[cp_kl_id]
                    k1 = aux_ao_loc[cp_kl_id+1]
                    int3c2e.get_int3c2e_slice(intopt, cp_ij_id, cp_kl_id, out=ints_slices[k0:k1])
                if omega is not None:
                    ints_slices_lr = cupy.zeros([naoaux, nj, ni], order='C')
                    for cp_kl_id, _ in enumerate(intopt.aux_log_qs):
                        k0 = aux_ao_loc[cp_kl_id]
                        k1 = aux_ao_loc[cp_kl_id+1]
                        int3c2e.get_int3c2e_slice(intopt, cp_ij_id, cp_kl_id, out=ints_slices[k0:k1], omega=omega)
                    ints_slices -= ints_slices_lr
            else:
                # Initialization is required due to cutensor operations later
                ints_slices = cupy.zeros([naoaux, nj, ni], order='C')
                for cp_kl_id, _ in enumerate(intopt.aux_log_qs):
                    k0 = aux_ao_loc[cp_kl_id]
                    k1 = aux_ao_loc[cp_kl_id+1]

                    int3c2e.get_int3c2e_slice(intopt, cp_ij_id, cp_kl_id, out=ints_slices[k0:k1], omega=omega)
            if lj>1 and not mol.cart: ints_slices = cart2sph(ints_slices, axis=1, ang=lj)
            if li>1 and not mol.cart: ints_slices = cart2sph(ints_slices, axis=2, ang=li)

            i0, i1 = ao_loc[cpi], ao_loc[cpi+1]
            j0, j1 = ao_loc[cpj], ao_loc[cpj+1]

            row = intopt.ao_pairs_row[cp_ij_id] - i0
            col = intopt.ao_pairs_col[cp_ij_id] - j0

            ints_slices_f= cupy.empty([naoaux,len(row)], order='F')
            ints_slices_f[:] = ints_slices[:,col,row]
            ints_slices = None
            if cd_low_tag == 'eig':
                cderi_block = cupy.dot(cd_low.T, ints_slices_f)
                ints_slices = None
            elif cd_low_tag == 'cd':
                cderi_block = solve_triangular(cd_low, ints_slices_f, lower=True, overwrite_b=True)
            else:
                raise RuntimeError('Tag is not found in lower triangular matrix.')
            t1 = log.timer_debug1(f'solve {cp_ij_id} / {nq} on Device {device_id}', *t1)

            # TODO:
            # 1) async data transfer
            # 2) auxiliary basis in the last dimension

            # if CDERI is saved on CPU
            ij0 = pairs_loc[cp_ij_id]
            ij1 = pairs_loc[cp_ij_id+1]
            if isinstance(_cderi[0], np.ndarray):
                for slice_id, (p0,p1) in enumerate(lib.prange(0, naux, aux_blksize)):
                    tmp = cupy.array(cderi_block[p0:p1], order='C', copy=True)
                    cupy.cuda.get_current_stream().synchronize()
                    copy_array(tmp, _cderi[slice_id][:p1-p0,ij0:ij1])
                    cupy.cuda.get_current_stream().synchronize()
            elif num_devices > 1:
                # Multi-GPU case, copy data to other Devices
                for dev_id, (p0,p1) in enumerate(lib.prange(0, naux, aux_blksize)):
                    # Making a copy for contiguous data transfer
                    tmp = cupy.array(cderi_block[p0:p1], order='C', copy=True)
                    with cupy.cuda.Device(dev_id):
                        tmp = copy_array(tmp)
                        _cderi[dev_id][:,ij0:ij1] = tmp
            else:
                _cderi[0][:,ij0:ij1] = cderi_block
            t1 = log.timer_debug1(f'transfer data for {cp_ij_id} / {nq} on Device {device_id}', *t1)
    return

def _cholesky_eri(intopt, omega=None, use_host_memory=True):
    assert isinstance(intopt, int3c2e_bdiv.Int3c2eOpt)
    if intopt._int3c2e_envs is None:
        intopt.build()
    sorted_mol = intopt.mol
    mol = sorted_mol.mol
    log = logger.new_logger(mol)
    auxmol = intopt.auxmol
    naux_sorted = auxmol.nao
    num_devices = multi_gpu.num_devices

    # When the basis set does not contain general contractions, sorted_mol are
    # simply an reordering of the original mol bases.
    recontract_bas = cp.asnumpy(sorted_mol.recontract_bas)
    needs_recontraction = any(recontract_bas[:,int3c2e_bdiv.NCTR_OF] != 1)

    mem_avail = get_avail_mem(exclude_memory_pool=True)
    word_avail = mem_avail // 8
    if needs_recontraction:
        batch_size = int(word_avail * .3) // naux_sorted
    else:
        batch_size = int(word_avail * .45) // naux_sorted

    current_device = cp.cuda.device.get_device_id()
    eval_j3c, aux_sorting, ao_pair_offsets, _, bas_ij_batches = intopt.int3c2e_evaluator(
        ao_pair_batch_size=batch_size, reorder_aux=True, omega=omega,
        pair_batch_by_l=needs_recontraction, return_bas_ij_batches=True)

    if needs_recontraction:
        recontract, ao_pair_counts, cderi_pair_counts, pair_addresses = \
                int3c2e_bdiv._create_pair_recontraction(sorted_mol, bas_ij_batches)
        cderi_offsets = np.append(0, np.cumsum(cderi_pair_counts))
        cderi_npairs = len(pair_addresses)
        pair_addresses = asarray(pair_addresses)
        rows, cols = divmod(pair_addresses, mol.nao)
        diag_addrs = cp.where(rows == cols)[0]
        cderi_idx = (pair_addresses, diag_addrs)
    else:
        cderi_offsets = ao_pair_offsets
        pair_addresses, diag_addrs = intopt.pair_and_diag_indices()
        cderi_npairs = len(pair_addresses)
        pair_addresses = cp.asarray(pair_addresses, dtype=np.int32)
        cderi_idx = (pair_addresses, diag_addrs)

    aux_coef, tag = _decompose_j2c(auxmol, aux_sorting)
    batch_size = int(max(ao_pair_offsets[1:] - ao_pair_offsets[:-1]))

    naux = aux_coef.shape[1]
    naux_per_device = min(naux, (naux + num_devices - 1) // num_devices)

    cp.get_default_memory_pool().free_all_blocks()
    mem_avail = get_avail_mem(exclude_memory_pool=True)
    word_avail = mem_avail // 8
    word_avail -= batch_size * naux
    if needs_recontraction:
        word_avail -= batch_size * naux_per_device
    on_gpu = True
    if cderi_npairs * naux > word_avail * 0.95 * num_devices:
        if not use_host_memory:
            cderi_size = cderi_npairs * naux / num_devices * 8e-9
            raise MemoryError(f'Not enough GPU memory. cderi size = {cderi_size:.2f} GB')
        on_gpu = False
    log.debug1('on_gpu=%s, nao_pairs=%d, naux_per_device=%d, batch_size=%d, num_batches=%d',
               on_gpu, cderi_npairs, naux_per_device, batch_size, len(bas_ij_batches))

    if not on_gpu:
        cderi_cpu = empty_mapped((naux, cderi_npairs))

        def proc(batch_iter):
            device_id = cp.cuda.device.get_device_id()
            stream = cp.cuda.get_current_stream()
            _eval_j3c = eval_j3c
            if device_id != current_device:
                _eval_j3c = intopt.int3c2e_evaluator(
                    ao_pair_batch_size=batch_size, reorder_aux=True, omega=omega,
                    pair_batch_by_l=needs_recontraction)[0]

            work = cp.empty(naux_sorted * batch_size)
            if needs_recontraction:
                work1 = cp.empty(naux_sorted * batch_size)
            work2 = cp.empty(naux * batch_size)

            for batch_id in batch_iter:
                log.debug1('processing cderi batch %d', batch_id)
                j3c = _eval_j3c(shl_pair_batch_id=batch_id, out=work)
                if needs_recontraction:
                    tmp = ndarray((j3c.shape[0], naux_sorted), buffer=work1)
                    j3c = recontract(batch_id, j3c, out=tmp)
                cderi_gpu = ndarray((j3c.shape[0], naux), buffer=work2)
                cderi_gpu = j3c.dot(aux_coef, out=cderi_gpu)
                p0, p1 = cderi_offsets[batch_id:batch_id+2]
                err = libvhf_rys.transpose_write(
                    cderi_cpu.ctypes,
                    ctypes.cast(cderi_gpu.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(naux), ctypes.c_int(cderi_npairs),
                    ctypes.c_int(p0), ctypes.c_int(p1))
                if err != 0:
                    raise RuntimeError('transpose_write cderi_cpu failed')
                stream.synchronize()

        batch_iter = range(len(bas_ij_batches))
        multi_gpu.run(proc, args=(batch_iter,), non_blocking=True)

        # Ensure data are fully written to host memory.
        multi_gpu.synchronize()
        cderi = [cderi_cpu[i*naux_per_device:(i+1)*naux_per_device]
                 for i in range(num_devices)]

    else:
        def proc():
            device_id = cp.cuda.device.get_device_id()
            aux0 = naux_per_device * device_id
            aux1 = min(int(aux_coef.shape[1]), aux0 + naux_per_device)
            c = cp.asarray(aux_coef[:,aux0:aux1])

            _eval_j3c = eval_j3c
            if device_id != current_device:
                _eval_j3c = intopt.int3c2e_evaluator(
                    ao_pair_batch_size=batch_size, reorder_aux=True, omega=omega,
                    pair_batch_by_l=needs_recontraction)[0]

            out = cp.empty((cderi_npairs, aux1-aux0))
            work = cp.empty(naux_sorted * batch_size)
            if needs_recontraction:
                work1 = cp.empty((naux_per_device * batch_size))

            for batch_id in range(len(bas_ij_batches)):
                j3c = _eval_j3c(batch_id, out=work)
                p0, p1 = cderi_offsets[batch_id:batch_id+2]
                if needs_recontraction:
                    tmp = ndarray((j3c.shape[0], aux1-aux0), buffer=work1)
                    tmp = j3c.dot(c, out=tmp)
                    recontract(batch_id, tmp, out=out[p0:p1])
                else:
                    j3c.dot(c, out=out[p0:p1])
            return out.T

        cderi = multi_gpu.run(proc, non_blocking=True)
    return cderi, cderi_idx

def _decompose_j2c(auxmol, aux_sorting=None):
    j2c = int3c2e_bdiv.int2c2e(auxmol)

    try:
        cd_low = cholesky(j2c)
        aux_coef = auxmol.ctr_coeff
        aux_coef = solve_triangular(
            cd_low, aux_coef.T, lower=True, overwrite_b=True).T
        tag = 'cd'
    except RuntimeError:
        w, v = cupy.linalg.eigh(j2c)
        idx = cp.where(w > LINEAR_DEP_THR)[0]
        logger.debug1(auxmol, 'discard %d small eigenvectors for auxiliary dimension',
                      w.size - len(idx))
        v = v[:,idx] / cupy.sqrt(w[idx])
        aux_coef = auxmol.apply_C_dot(v, axis=0)
        tag = 'ed'

    if aux_sorting is not None:
        aux_coef, tmp = cupy.empty_like(aux_coef), aux_coef
        aux_coef[aux_sorting] = tmp
    return aux_coef, tag
