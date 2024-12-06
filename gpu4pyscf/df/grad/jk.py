# Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
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

from concurrent.futures import ThreadPoolExecutor
import cupy
from gpu4pyscf.lib.cupy_helper import contract, concatenate
from gpu4pyscf.lib import logger
from gpu4pyscf.__config__ import _streams, _num_devices

def _jk_task(with_df, dm, orbo, with_j=True, with_k=True, device_id=0):
    '''  # (L|ij) -> rhoj: (L), rhok: (L|oo)
    '''
    rhoj = rhok = None
    with cupy.cuda.Device(device_id), _streams[device_id]:
        log = logger.new_logger(with_df.mol, with_df.verbose)
        t0 = log.init_timer()
        dm = cupy.asarray(dm)
        orbo = cupy.asarray(orbo)
        naux_slice = with_df._cderi[device_id].shape[0]
        nocc = orbo.shape[-1]
        rows = with_df.intopt.cderi_row
        cols = with_df.intopt.cderi_col
        dm_sparse = dm[rows, cols]
        dm_sparse[with_df.intopt.cderi_diag] *= .5
        
        blksize = with_df.get_blksize()
        if with_j:
            rhoj = cupy.empty([naux_slice])
        if with_k:
            rhok = cupy.empty([naux_slice, nocc, nocc], order='C')
        p0 = p1 = 0

        for cderi, cderi_sparse in with_df.loop(blksize=blksize):
            p1 = p0 + cderi.shape[0]
            if with_j:
                rhoj[p0:p1] = 2.0*dm_sparse.dot(cderi_sparse)
            if with_k:
                tmp = contract('Lij,jk->Lki', cderi, orbo)
                contract('Lki,il->Lkl', tmp, orbo, out=rhok[p0:p1])
            p0 = p1
            cupy.cuda.get_current_stream().synchronize()
        t0 = log.timer_debug1(f'rhoj and rhok on Device {device_id}', *t0)
    return rhoj, rhok

def get_rhoj_rhok(with_df, dm, orbo, with_j=True, with_k=True):
    ''' Calculate rhoj and rhok on Multi-GPU system
    '''
    futures = []
    cupy.cuda.get_current_stream().synchronize()
    with ThreadPoolExecutor(max_workers=_num_devices) as executor:
        for device_id in range(_num_devices):
            future = executor.submit(
                _jk_task, with_df, dm, orbo,
                with_j=with_j, with_k=with_k, device_id=device_id)
            futures.append(future)
    
    rhoj_total = []
    rhok_total = []
    for future in futures:
        rhoj, rhok = future.result()
        rhoj_total.append(rhoj)
        rhok_total.append(rhok)
        
    rhoj = rhok = None
    if with_j:
        rhoj = concatenate(rhoj_total)
    if with_k:
        rhok = concatenate(rhok_total)
    
    return rhoj, rhok
