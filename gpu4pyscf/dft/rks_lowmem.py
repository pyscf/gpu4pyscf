# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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
The RKS implemented in this module prioritizes a reduced GPU memory footprint.
'''

import numpy as np
import cupy as cp
from pyscf import lib as pyscf_lib
from gpu4pyscf.lib import logger
from gpu4pyscf.dft import numint, gen_grid, rks
from gpu4pyscf.scf import hf_lowmem, jk, j_engine
from gpu4pyscf.lib.cupy_helper import (
    tag_array, pack_tril, asarray)
from pyscf import __config__

__all__ = [
    'RKS',
]

class RKS(rks.RKS):
    '''The low-memory RKS class for large systems. Not fully compatible with the
    default RKS class in rks.py.
    '''

    small_rho_cutoff = 0

    DIIS = hf_lowmem.CDIIS

    kernel = scf = hf_lowmem.kernel
    density_fit              = NotImplemented
    as_scanner               = NotImplemented
    newton                   = NotImplemented
    x2c = x2c1e = sfx2c1e    = NotImplemented
    stability                = NotImplemented
    to_hf = NotImplemented

    check_sanity = hf_lowmem.RHF.check_sanity
    get_hcore = hf_lowmem.RHF.get_hcore
    get_jk = hf_lowmem.RHF.get_jk
    get_k = hf_lowmem.RHF.get_k
    get_j = hf_lowmem.RHF.get_j
    get_fock = hf_lowmem.RHF.get_fock
    make_wfn = hf_lowmem.RHF.make_wfn
    make_rdm1 = hf_lowmem.RHF.make_rdm1
    _delta_rdm1 = hf_lowmem.RHF._delta_rdm1
    _eigh = hf_lowmem.RHF._eigh

    def __init__(self, mol, xc='LDA,VWN'):
        hf_lowmem.RHF.__init__(self, mol)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        hf_lowmem.RHF.dump_flags(self, verbose)
        return rks.KohnShamDFT.dump_flags(self, verbose)

    def _get_k_sorted_mol(self, dm_or_wfn, hermi, omega, log):
        mol = self.mol
        with mol.with_range_coulomb(omega):
            vhfopt = self._opt_gpu.get(omega)
            if vhfopt is None:
                vhfopt = self._opt_gpu[omega] = jk._VHFOpt(mol, self.direct_scf_tol).build()
            return vhfopt.get_jk(dm_or_wfn, hermi, False, True, log)[1]

    def get_veff(self, mol, dm_or_wfn, dm_last=None, vhf_last=0, hermi=1):
        '''Constructus the lower-triangular part of the Fock matrix.'''
        assert hermi == 1
        log = logger.new_logger(mol, self.verbose)
        cput0 = log.init_timer()
        if dm_or_wfn is None:
            # Avoid explicitly constructing density matrix. Use the RHFWfn
            # instance to just pass mo_coeff and mo_occ to the nr_rks function.
            dm = hf_lowmem.RHFWfn(self.mo_coeff, self.mo_occ)
        else:
            dm = dm_or_wfn
        initialize_grids(self, mol, dm)
        if log.verbose >= logger.DEBUG1:
            mem_avail = log.print_mem_info()
            log.debug1('available GPU memory for rks.get_veff: %.3f GB', mem_avail/1e9)

        ni = self._numint
        n, exc, vxc = ni.nr_rks(mol, self.grids, self.xc, dm)
        if self.do_nlc():
            if ni.libxc.is_nlc(self.xc):
                xc = self.xc
            else:
                assert ni.libxc.is_nlc(self.nlc)
                xc = self.nlc
            n, enlc, vnlc = ni.nr_nlc_vxc(mol, self.nlcgrids, xc, dm)
            exc += enlc
            vxc += vnlc
        dm = None
        vxc = pack_tril(vxc)
        log.debug('nelec by numeric integration = %s', n)
        cput1 = log.timer_debug1('vxc tot', *cput0)

        omega = mol.omega
        if omega in self._opt_gpu:
            vhfopt = self._opt_gpu[omega]
        else:
            self._opt_gpu[omega] = vhfopt = jk._VHFOpt(mol, self.direct_scf_tol).build()
        if omega in self._opt_jengine:
            jopt = self._opt_jengine[omega]
        else:
            self._opt_jengine[omega] = jopt = j_engine._VHFOpt(mol, self.direct_scf_tol).build()

        cp.get_default_memory_pool().free_all_blocks()
        if log.verbose >= logger.DEBUG1:
            mem_avail = log.print_mem_info()
            log.debug1('available GPU memory for get_jk in rks.get_veff: %.3f GB', mem_avail/1e9)

        dm = lambda: self._delta_rdm1(dm_or_wfn, dm_last, jopt)
        vj = jopt.get_j(dm, log)
        assert vj.ndim == 3
        vj = jopt.apply_coeff_CT_mat_C(vj)
        cput2 = log.timer_debug1('vj', *cput1)
        vj = pack_tril(vj[0])
        vj_last = getattr(vhf_last, 'vj', None)
        if vj_last is not None:
            if isinstance(vj_last, cp.ndarray):
                vj += vj_last
            else:
                vj += asarray(vj_last)
        vxc += vj
        vj = vj.get()

        vk = None
        if ni.libxc.is_hybrid_xc(self.xc):
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.xc, spin=mol.spin)
            dm = lambda: self._delta_rdm1(dm_or_wfn, dm_last, vhfopt)
            if omega == 0:
                vk = vhfopt.get_jk(dm, hermi, False, True, log)[1]
                vk *= hyb
            elif alpha == 0: # LR=0, only SR exchange
                vk = self._get_k_sorted_mol(dm, hermi, -omega, log)
                vk *= hyb
            elif hyb == 0: # SR=0, only LR exchange
                vk = self._get_k_sorted_mol(dm, hermi, omega, log)
                vk *= alpha
            else: # SR and LR exchange with different ratios
                vk = vhfopt.get_jk(dm, hermi, False, True, log)[1]
                vk *= hyb
                vklr = self._get_k_sorted_mol(dm, hermi, omega, log)
                vklr *= (alpha - hyb)
                vk += vklr
            vk = vhfopt.apply_coeff_CT_mat_C(vk)
            log.timer_debug1('vk', *cput2)
            vk_last = getattr(vhf_last, 'vk', None)
            vk = pack_tril(vk[0])
            vk *= .5
            if vk_last is not None:
                if isinstance(vk_last, cp.ndarray):
                    vk += vk_last
                else:
                    vk += asarray(vk_last)
            vxc -= vk
            vk = vk.get()

        vxc = vxc.get()
        log.timer_debug1('veff', *cput0)
        vxc = pyscf_lib.tag_array(vxc, exc=exc, vj=vj, vk=vk)
        return vxc

    def energy_elec(self, dm_or_wfn, h1e, vhf):
        assert h1e.ndim == vhf.ndim == 1
        if isinstance(dm_or_wfn, hf_lowmem.WaveFunction):
            dm = dm_or_wfn.make_rdm1()
        else:
            dm = dm_or_wfn
        assert dm.dtype == np.float64
        dm_tril = pack_tril(dm)
        nao = dm.shape[0]
        i = cp.arange(nao)
        diag = i*(i+1)//2 + i
        dm_tril[diag] *= .5
        dm_tril = dm_tril.get()
        e1 = float(h1e.dot(dm_tril) * 2)
        ecoul = float(vhf.vj.dot(dm_tril))
        exc = float(vhf.exc)
        if vhf.vk is not None:
            exc -= float(vhf.vk.dot(dm_tril))
        vtmp = h1e * 2
        vtmp += vhf.vj
        e_tot = float(vtmp.dot(dm_tril)) + exc
        self.scf_summary['e1'] = e1
        self.scf_summary['coul'] = ecoul
        self.scf_summary['exc'] = exc
        logger.debug(self, 'E1 = %s  Ecoul = %s  Exc = %s', e1, ecoul, exc)
        return e_tot, e_tot-e1

    def to_cpu(self):
        raise NotImplementedError

def initialize_grids(ks, mol=None, dm_or_wfn=None):
    # Initialize self.grids the first time call get_veff
    if mol is None: mol = ks.mol
    if ks.grids.coords is None:
        t0 = logger.init_timer(ks)
        ks.grids.build()
        #ks.grids.build(with_non0tab=True)
        ks.grids.weights = asarray(ks.grids.weights)
        ks.grids.coords = asarray(ks.grids.coords)
        if ks.small_rho_cutoff > 1e-20:
            # Filter grids the first time setup grids
            ks.grids = rks.prune_small_rho_grids_(ks, ks.mol, dm_or_wfn, ks.grids)
        t0 = logger.timer_debug1(ks, 'setting up grids', *t0)

    if ks.do_nlc() and ks.nlcgrids.coords is None:
        t0 = logger.init_timer(ks)
        #ks.nlcgrids.build(with_non0tab=True)
        ks.nlcgrids.build()
        ks.nlcgrids.weights = asarray(ks.nlcgrids.weights)
        ks.nlcgrids.coords = asarray(ks.nlcgrids.coords)
        if ks.small_rho_cutoff > 1e-20:
            # Filter grids the first time setup grids
            ks.nlcgrids = rks.prune_small_rho_grids_(ks, ks.mol, dm_or_wfn, ks.nlcgrids)
        t0 = logger.timer_debug1(ks, 'setting up nlc grids', *t0)
    return ks
