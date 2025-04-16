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
from importlib.util import find_spec
has_dpctl = find_spec("dpctl")
if not has_dpctl:
    import cupy as cp
    from gpu4pyscf.lib.cupy_helper import tag_array, pack_tril, get_avail_mem
else:
    import dpnp as cp
    from gpu4pyscf.lib.dpnp_helper import tag_array, pack_tril, get_avail_mem
from gpu4pyscf.lib import logger
from gpu4pyscf.dft import numint, gen_grid, rks
from gpu4pyscf.scf import hf_lowmem, jk
from pyscf import __config__

__all__ = [
    'RKS',
]

class RKS(rks.RKS):
    '''The low-memory RKS class for large systems. Not fully compatible with the
    default RKS class in rks.py.
    '''

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
    make_rdm1 = hf_lowmem.RHF.make_rdm1
    _delta_rdm1 = hf_lowmem.RHF._delta_rdm1

    def __init__(self, mol, xc='LDA,VWN'):
        hf_lowmem.RHF.__init__(self, mol)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        hf_lowmem.RHF.dump_flags(self, verbose)
        return rks.KohnShamDFT.dump_flags(self, verbose)

    def _get_k_sorted_mol(self, dm, hermi, omega, log):
        mol = self.mol
        with mol.with_range_coulomb(omega):
            vhfopt = self._opt_gpu.get(omega)
            if vhfopt is None:
                vhfopt = self._opt_gpu[omega] = jk._VHFOpt(mol, self.direct_scf_tol).build()
            return vhfopt.get_jk(dm, hermi, False, True, log)[1]

    def get_veff(self, mol, dm, dm_last=None, vhf_last=0, hermi=1):
        '''Constructus the lower-triangular part of the Fock matrix.'''
        assert hermi == 1
        log = logger.new_logger(mol, self.verbose)
        cput0 = log.init_timer()
        if dm is None:
            _dm = tag_array(self.make_rdm1(),
                            mo_coeff=self.mo_coeff, mo_occ=self.mo_occ)
        else:
            _dm = dm
        initialize_grids(self, mol, _dm)
        mem_avail = get_avail_mem()
        log.debug1('available GPU memory for rks.get_veff: %d B', mem_avail)

        ni = self._numint
        n, exc, vxc = ni.nr_rks(mol, self.grids, self.xc, _dm)
        if self.do_nlc():
            if ni.libxc.is_nlc(self.xc):
                xc = self.xc
            else:
                assert ni.libxc.is_nlc(self.nlc)
                xc = self.nlc
            n, enlc, vnlc = ni.nr_nlc_vxc(mol, self.nlcgrids, xc, _dm)
            exc += enlc
            vxc += vnlc
        _dm = None
        vxc = pack_tril(vxc)
        log.debug('nelec by numeric integration = %s', n)
        cput1 = log.timer_debug1('vxc tot', *cput0)

        omega = mol.omega
        vhfopt = self._opt_gpu.get(omega)
        if vhfopt is None:
            vhfopt = self._opt_gpu[omega] = jk._VHFOpt(mol, self.direct_scf_tol).build()
        _dm = self._delta_rdm1(dm, dm_last, vhfopt)

        if not has_dpctl:
            cp.get_default_memory_pool().free_all_blocks()
        mem_avail = get_avail_mem()
        log.debug1('available GPU memory for get_jk in rks.get_veff: %d B', mem_avail)
        vj = vk = None
        if not ni.libxc.is_hybrid_xc(self.xc):
            vj = vhfopt.get_j(_dm, log)
            _dm = None
            vj = vhfopt.apply_coeff_CT_mat_C(vj)
            vj = pack_tril(vj[0])
            vj_last = getattr(vhf_last, 'vj', None)
            if isinstance(vj_last, cp.ndarray):
                vj_last += vj # Reuse the memory of the previous Veff
                vj = vj_last
            vxc += vj
        else:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.xc, spin=mol.spin)
            if omega == 0:
                vj, vk = vhfopt.get_jk(_dm, hermi, True, True, log)
                vk *= hyb
            elif alpha == 0: # LR=0, only SR exchange
                vj = vhfopt.get_j(_dm, log)
                vk = self._get_k_sorted_mol(_dm, hermi, -omega, log)
                vk *= hyb
            elif hyb == 0: # SR=0, only LR exchange
                vj = vhfopt.get_j(_dm, log)
                vk = self._get_k_sorted_mol(_dm, hermi, omega, log)
                vk *= alpha
            else: # SR and LR exchange with different ratios
                vj, vk = vhfopt.get_jk(_dm, hermi, True, True, log)
                vk *= hyb
                vklr = self._get_k_sorted_mol(_dm, hermi, omega, log)
                vklr *= (alpha - hyb)
                vk += vklr
            _dm = None

            assert vj.ndim == 3
            vj_last = getattr(vhf_last, 'vj', None)
            vk_last = getattr(vhf_last, 'vk', None)
            vj = vhfopt.apply_coeff_CT_mat_C(vj)
            vj = pack_tril(vj[0])
            if isinstance(vj_last, cp.ndarray):
                vj_last += vj
                vj = vj_last
            vk = vhfopt.apply_coeff_CT_mat_C(vk)
            vk = pack_tril(vk[0])
            if isinstance(vk_last, cp.ndarray):
                vk_last += vk
                vk = vk_last
            vxc += vj
            vxc -= vk * .5

        log.timer_debug1('jk total', *cput1)
        vxc = tag_array(vxc, exc=exc, vj=vj, vk=vk)
        return vxc

    def energy_elec(self, dm, h1e, vhf):
        assert dm.dtype == np.float64
        assert h1e.ndim == vhf.ndim == 1
        dm_tril = pack_tril(dm)
        nao = dm.shape[0]
        i = cp.arange(nao)
        diag = i*(i+1)//2 + i
        dm_tril[diag] *= .5
        e1 = float(h1e.dot(dm_tril) * 2)
        ecoul = float(vhf.vj.dot(dm_tril))
        exc = vhf.exc
        if vhf.vk is not None:
            exc -= float(vhf.vk.dot(dm_tril)) * .5
        e2 = ecoul + exc
        self.scf_summary['e1'] = e1
        self.scf_summary['coul'] = ecoul
        self.scf_summary['exc'] = exc
        logger.debug(self, 'E1 = %s  Ecoul = %s  Exc = %s', e1, ecoul, exc)
        return e1+e2, e2

    def to_cpu(self):
        raise NotImplementedError

def initialize_grids(ks, mol=None, dm=None):
    # Initialize self.grids the first time call get_veff
    if mol is None: mol = ks.mol
    if ks.grids.coords is None:
        t0 = logger.init_timer(ks)
        ks.grids.build()
        #ks.grids.build(with_non0tab=True)
        ks.grids.weights = cp.asarray(ks.grids.weights)
        ks.grids.coords = cp.asarray(ks.grids.coords)
        if ks.small_rho_cutoff > 1e-20:
            # Filter grids the first time setup grids
            ks.grids = rks.prune_small_rho_grids_(ks, ks.mol, dm, ks.grids)
        t0 = logger.timer_debug1(ks, 'setting up grids', *t0)

        if ks.do_nlc() and ks.nlcgrids.coords is None:
            if ks.nlcgrids.coords is None:
                t0 = logger.init_timer(ks)
                #ks.nlcgrids.build(with_non0tab=True)
                ks.nlcgrids.build()
                ks.nlcgrids.weights = cp.asarray(ks.nlcgrids.weights)
                ks.nlcgrids.coords = cp.asarray(ks.nlcgrids.coords)
                if ks.small_rho_cutoff > 1e-20:
                    # Filter grids the first time setup grids
                    ks.nlcgrids = rks.prune_small_rho_grids_(ks, ks.mol, dm, ks.nlcgrids)
                t0 = logger.timer_debug1(ks, 'setting up nlc grids', *t0)
    return ks
