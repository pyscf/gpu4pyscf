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

import threading
import cupy
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.lib import logger
from gpu4pyscf.__config__ import _streams

def _jk_task(with_df, dm, orbo, device_id, rhoj_total, rhok_total, with_j=True, with_k=True):
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
        rhoj_total[device_id] = rhoj
        rhok_total[device_id] = rhok
        t0 = log.timer_debug1(f'rhoj and rhok on Device {device_id}', *t0)
    return

def get_rhoj_rhok(with_df, dm, orbo, with_j=True, with_k=True):
    ''' Calculate rhoj and rhok on Multi-GPU system
    '''
    num_gpus = cupy.cuda.runtime.getDeviceCount()
    rhoj_total = [None] * num_gpus
    rhok_total = [None] * num_gpus
    threads = []
    for device_id in range(num_gpus):
        thread = threading.Thread(target=_jk_task, 
                                    args=(with_df, dm, orbo, device_id, rhoj_total, rhok_total),
                                    kwargs={"with_j": with_j, "with_k": with_k})
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    for stream in _streams:
        stream.synchronize()
    
    rhoj = rhok = None
    if with_j:
        rhoj = cupy.concatenate(rhoj_total)
    if with_k:
        rhok = cupy.concatenate(rhok_total)
    
    return rhoj, rhok