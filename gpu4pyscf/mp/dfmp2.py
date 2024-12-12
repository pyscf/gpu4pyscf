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
import numpy as np
import cupy
from pyscf.mp import mp2 as mp2_pyscf
from gpu4pyscf import df
from gpu4pyscf.mp import mp2
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, tag_array, reduce_to_device
from gpu4pyscf.__config__ import _streams, _num_devices
from pyscf import __config__

WITH_T2 = getattr(__config__, 'mp_dfmp2_with_t2', True)
_einsum = cupy.einsum

def _dfmp2_tasks(mp, mo_coeff, mo_energy, device_id=0):
    with cupy.cuda.Device(device_id), _streams[device_id]:
        mo_energy = cupy.asarray(mo_energy)
        mo_coeff = cupy.asarray(mo_coeff)
        
        nocc = mp.nocc
        nvir = mp.nmo - nocc

        _cderi = mp.with_df._cderi[device_id]
        naux_slice = _cderi.shape[0]
        Lov = cupy.empty((naux_slice, nocc*nvir))
        p1 = 0
        for istep, qov in enumerate(mp.loop_ao2mo(mo_coeff, nocc)):
            logger.debug(mp, 'Load cderi step %d', istep)
            p0, p1 = p1, p1 + qov.shape[0]
            Lov[p0:p1] = qov.reshape([p1-p0,nocc*nvir])
    return Lov

def get_occ_blk(Lov_dist, i, nocc, nvir):
    occ_blk_dist = [None] * _num_devices
    for device_id in range(_num_devices):
        with cupy.cuda.Device(device_id), _streams[device_id]:
            Lov = Lov_dist[device_id]
            mat = cupy.dot(Lov[:,i*nvir:(i+1)*nvir].T,
                            Lov).reshape(nvir,nocc,nvir)
            occ_blk_dist[device_id] = mat
    occ_blk = reduce_to_device(occ_blk_dist)
    return occ_blk

def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2,
           verbose=logger.NOTE):
    if mo_energy is not None or mo_coeff is not None:
        # For backward compatibility.  In pyscf-1.4 or earlier, mp.frozen is
        # not supported when mo_energy or mo_coeff is given.
        assert (mp.frozen == 0 or mp.frozen is None)

    if eris is None:      eris = mp.ao2mo(mo_coeff)
    if mo_energy is None: mo_energy = eris.mo_energy
    if mo_coeff is None:  mo_coeff = eris.mo_coeff
    mo_energy = cupy.asarray(mo_energy)
    mo_coeff = cupy.asarray(mo_coeff)
    
    if mp.with_df.naux is None:
        mp.with_df.build()

    # Submit tasks to different devices
    futures = []
    with ThreadPoolExecutor(max_workers=_num_devices) as executor:
        for device_id in range(_num_devices):
            future = executor.submit(_dfmp2_tasks, mp, mo_coeff, mo_energy, 
                                     device_id=device_id)
            futures.append(future)

    Lov_dist = [future.result() for future in futures]

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    if with_t2:
        t2 = cupy.empty((nocc,nocc,nvir,nvir), dtype=mo_coeff.dtype)
    else:
        t2 = None
    
    emp2_ss = emp2_os = 0
    for i in range(nocc):
        buf = get_occ_blk(Lov_dist, i, nocc, nvir)
        gi = cupy.array(buf, copy=False)
        gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
        #lib.direct_sum('jb+a->jba', eia, eia[i])
        t2i = gi/(eia[:,:,None] + eia[i])
        edi = _einsum('jab,jab', t2i, gi) * 2
        exi = -_einsum('jab,jba', t2i, gi)
        emp2_ss += edi*0.5 + exi
        emp2_os += edi*0.5
        if with_t2:
            t2[i] = t2i
        buf = gi = t2i = None # free mem

    emp2_ss = emp2_ss.real
    emp2_os = emp2_os.real
    emp2 = tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    return emp2, t2


class DFMP2(mp2.MP2):
    _keys = {'with_df'}

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        mp2.MP2.__init__(self, mf, frozen, mo_coeff, mo_occ)
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

    def reset(self, mol=None):
        self.with_df.reset(mol)
        return mp2.MP2.reset(self, mol)

    def loop_ao2mo(self, mo_coeff, nocc):
        mo_coeff = cupy.asarray(mo_coeff, order='C')
        Lov = None
        with_df = self.with_df
        mo_coeff = with_df.intopt.sort_orbitals(mo_coeff, axis=[0])
        orbo = mo_coeff[:,:nocc]
        orbv = mo_coeff[:,nocc:]
        blksize = with_df.get_blksize()
        for cderi, cderi_sparse in with_df.loop(blksize=blksize):
            tmp = _einsum('Lpq,po->Loq', cderi, orbo)
            Lov = _einsum('Loq,qi->Loi', tmp, orbv)
            yield Lov

    def ao2mo(self, mo_coeff=None):
        eris = mp2_pyscf._ChemistsERIs()
        # Initialize only the mo_coeff
        if isinstance(mo_coeff, np.ndarray):
            mo_coeff = cupy.asarray(mo_coeff)
        eris._common_init_(self, mo_coeff)
        return eris

    def make_rdm1(self, t2=None, ao_repr=False):
        raise NotImplementedError

    def make_rdm2(self, t2=None, ao_repr=False):
        raise NotImplementedError

    def nuc_grad_method(self):
        raise NotImplementedError

    # For non-canonical MP2
    def update_amps(self, t2, eris):
        raise NotImplementedError

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)
