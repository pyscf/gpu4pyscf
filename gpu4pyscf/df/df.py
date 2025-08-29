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


import copy
import cupy
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from cupyx.scipy.linalg import solve_triangular
from pyscf import lib
from pyscf.df import df, addons, incore
from gpu4pyscf.lib.cupy_helper import (
    cholesky, tag_array, get_avail_mem, cart2sph, p2p_transfer, copy_array,
    asarray)
from gpu4pyscf.df import int3c2e, df_jk
from gpu4pyscf.df import int3c2e_bdiv
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import utils
from gpu4pyscf import __config__
from gpu4pyscf.__config__ import _streams, num_devices

MIN_BLK_SIZE = getattr(__config__, 'min_ao_blksize', 128)
ALIGNED = getattr(__config__, 'ao_aligned', 32)
GB = 1024*1024*1024
INT3C2E_V2 = False

LINEAR_DEP_THR = incore.LINEAR_DEP_THR
GROUP_SIZE = 256

class DF(lib.StreamObject):

    _keys = {'intopt', 'nao', 'naux', 'cd_low', 'mol', 'auxmol', 'use_gpu_memory'}

    def __init__(self, mol, auxbasis=None):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory
        self.use_gpu_memory = True
        self._auxbasis = auxbasis

        self.auxmol = None
        self.intopt = None
        self.nao = None
        self.naux = None
        self.cd_low = None
        self._cderi = None
        self._vjopt = None
        self._rsh_df = {}

    __getstate__, __setstate__ = lib.generate_pickle_methods(
        excludes=('cd_low', 'intopt', '_cderi', '_vjopt'))

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

        if INT3C2E_V2:
            self.intopt = intopt = int3c2e_bdiv.Int3c2eOpt(mol, auxmol)
            self._cderi = {}
            self._cderi[0] = _cholesky_eri_bdiv(intopt, omega=omega)
            rows, cols, diags = intopt.orbital_pair_nonzero_indices()
            intopt.cderi_row = rows
            intopt.cderi_col = cols
            intopt.cderi_diag = diags
            log.timer_debug1('cholesky_eri', *t0)
            return self

        if omega and omega > 1e-10:
            with auxmol.with_range_coulomb(omega):
                j2c_cpu = auxmol.intor('int2c2e', hermi=1)
        else:
            j2c_cpu = auxmol.intor('int2c2e', hermi=1)
        j2c = asarray(j2c_cpu)
        t0 = log.timer_debug1('2c2e', *t0)
        intopt = int3c2e.VHFOpt(mol, auxmol, 'int2e')
        intopt.build(direct_scf_tol, diag_block_with_triu=False, aosym=True,
                     group_size=GROUP_SIZE, group_size_aux=GROUP_SIZE)
        log.timer_debug1('prepare intopt', *t0)
        self.j2c = j2c.copy()

        j2c = intopt.sort_orbitals(j2c, aux_axis=[0,1])
        try:
            self.cd_low = cholesky(j2c)
            self.cd_low = tag_array(self.cd_low, tag='cd')
        except Exception:
            w, v = cupy.linalg.eigh(j2c)
            idx = w > LINEAR_DEP_THR
            self.cd_low = (v[:,idx] / cupy.sqrt(w[idx]))
            self.cd_low = tag_array(self.cd_low, tag='eig')

        v = w = None
        naux = self.naux = self.cd_low.shape[1]
        log.debug('size of aux basis %d', naux)

        self._cderi = cholesky_eri_gpu(intopt, mol, auxmol, self.cd_low,
                                       omega=omega, use_gpu_memory=self.use_gpu_memory)
        log.timer_debug1('cholesky_eri', *t0)
        self.intopt = intopt
        return self

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True,
               direct_scf_tol=getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13),
               omega=None):
        if omega is None:
            return df_jk.get_jk(self, dm, hermi, with_j, with_k, direct_scf_tol)
        assert omega >= 0.0

        # A temporary treatment for RSH-DF integrals
        # TODO: use the range_coulomb context from pyscf
        key = '%.6f' % omega
        if key in self._rsh_df:
            rsh_df = self._rsh_df[key]
        else:
            rsh_df = self._rsh_df[key] = self.copy().reset()
            logger.info(self, 'Create RSH-DF object %s for omega=%s', rsh_df, omega)

        return df_jk.get_jk(rsh_df, dm, hermi, with_j, with_k, direct_scf_tol, omega=omega)

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
        device_id = cupy.cuda.Device().id
        cderi_sparse = self._cderi[device_id]
        if blksize is None:
            blksize = self.get_blksize()
        nao = self.nao
        naux_slice = cderi_sparse.shape[0]
        rows = self.intopt.cderi_row
        cols = self.intopt.cderi_col
        buf_prefetch = None
        buf_cderi = cupy.zeros([blksize,nao,nao])
        for p0, p1 in lib.prange(0, naux_slice, blksize):
            p2 = min(naux_slice, p1+blksize)
            if isinstance(cderi_sparse, cupy.ndarray):
                buf = cderi_sparse[p0:p1,:]
            if isinstance(cderi_sparse, np.ndarray):
                # first block
                if buf_prefetch is None:
                    buf = asarray(cderi_sparse[p0:p1,:])
                buf_prefetch = cupy.empty([p2-p1,cderi_sparse.shape[1]])
            if isinstance(cderi_sparse, np.ndarray) and p1 < p2:
                buf_prefetch.set(cderi_sparse[p1:p2,:])
            if unpack:
                buf_cderi[:p1-p0,rows,cols] = buf
                buf_cderi[:p1-p0,cols,rows] = buf
                buf2 = buf_cderi[:p1-p0]
            else:
                buf2 = None
            yield buf2, buf.T
            if isinstance(cderi_sparse, np.ndarray):
                cupy.cuda.Device().synchronize()

            if buf_prefetch is not None:
                buf = buf_prefetch

    def reset(self, mol=None):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        if mol is not None:
            self.mol = mol
            self.auxmol = None
        self._cderi = None
        self._vjopt = None
        self._rsh_df = {}
        self.intopt = None
        self.nao = None
        self.naux = None
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
            with cupy.cuda.Device(device_id), _streams[device_id]:
                _cderi[device_id] = cupy.empty([p1-p0, npairs])
            log.debug(f"CDERI size {_cderi[device_id].nbytes/GB:.3f} GB on Device {device_id}")
        else:
            mem = cupy.cuda.alloc_pinned_memory((p1-p0) * npairs * 8)
            cderi_blk = np.ndarray([p1-p0, npairs], dtype=np.float64, order='C', buffer=mem)
            _cderi[device_id] = cderi_blk

    npairs_per_ctr = [len(intopt.ao_pairs_row[cp_ij_id]) for cp_ij_id in range(len(intopt.log_qs))]

    npairs_per_ctr = np.array(npairs_per_ctr)
    total_task_list = np.argsort(npairs_per_ctr)
    task_list_per_device = []
    for device_id in range(num_devices):
        task_list_per_device.append(total_task_list[device_id::num_devices])

    cd_low_f = cupy.array(cd_low, order='F', copy=False)
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
    nq = len(intopt.log_qs)
    mol = intopt.mol
    naux = cd_low.shape[1]
    naoaux = cd_low.shape[0]
    npairs = [len(intopt.ao_pairs_row[cp_ij]) for cp_ij in range(len(intopt.log_qs))]
    pairs_loc = np.append(0, np.cumsum(npairs))
    with cupy.cuda.Device(device_id), _streams[device_id]:
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
                    copy_array(tmp, _cderi[slice_id][:p1-p0,ij0:ij1])
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

# Generate CDERI using the new int3c2e_bdiv algorithm
def _cholesky_eri_bdiv(intopt, omega=None):
    assert isinstance(intopt, int3c2e_bdiv.Int3c2eOpt)
    assert omega is None
    eri3c = next(intopt.int3c2e_bdiv_generator())
    if intopt.mol.cart:
        eri3c = intopt.orbital_pair_cart2sph(eri3c)
    auxmol = intopt.auxmol
    j2c = asarray(auxmol.intor('int2c2e', hermi=1), order='C')
    cd_low = cholesky(j2c)
    aux_coeff = cupy.array(intopt.aux_coeff, copy=True)
    cd_low = solve_triangular(cd_low, aux_coeff.T, lower=True, overwrite_b=True)
    cderi = cd_low.dot(eri3c.T)
    return cderi
