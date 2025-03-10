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
from gpu4pyscf.lib import logger
from gpu4pyscf.dft import numint, gen_grid, rks
from gpu4pyscf.scf import hf_lowmem, jk
from gpu4pyscf.lib.cupy_helper import tag_array, sandwich_dot, pack_tril
from pyscf import __config__

__all__ = [
    'RKS',
]

class RKS(rks.RKS):

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
    get_fock = hf_lowmem.RHF.get_j
    make_rdm1 = hf_lowmem.RHF.make_rdm1

    def __init__(self, mol, xc='LDA,VWN'):
        hf_lowmem.RHF.__init__(self, mol)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        hf_lowmem.RHF.dump_flags(self, verbose)
        return rks.KohnShamDFT.dump_flags(self, verbose)

    def get_veff(self, mol, dm, dm_last=None, vhf_last=0, hermi=1):
        '''Constructus the lower-triangular part of the Fock matrix.'''
        assert hermi == 1
        log = logger.new_logger(mol, self.verbose)
        cput0 = log.init_timer()
        rks.initialize_grids(self, mol, dm)

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
        vxc = pack_tril(vxc)
        logger.debug(self, 'nelec by numeric integration = %s', n)
        cput1 = logger.timer_debug1(self, 'vxc tot', *cput0)

        omega = mol.omega
        vhfopt = self._opt_gpu.get(omega)
        if vhfopt is None:
            vhfopt = self._opt_gpu[omega] = jk._VHFOpt(mol, self.direct_scf_tol).build()
            if isinstance(vhfopt.coeff, cp.ndarray):
                vhfopt.coeff = vhfopt.coeff.get()
        dm = self._delta_rdm1(dm, dm_last, vhfopt)

        vj = vk = None
        if not ni.libxc.is_hybrid_xc(self.xc):
            vj = vhfopt.get_j(dm, log)
            vj = sandwich_dot(vj, cp.asarray(vhfopt.coeff))
            vj = pack_tril(vj)
            vj_last = getattr(vhf_last, 'vj', None)
            if isinstance(vj_last, cp.ndarray):
                vj += vj_last
            else:
                vj = vj_last
            vxc += vj
        else:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.xc, spin=mol.spin)
            if omega == 0:
                vj, vk = vhfopt.get_jk(dm, hermi, True, True, log)
                vk *= hyb
            elif alpha == 0: # LR=0, only SR exchange
                vj = vhfopt.get_j(dm, log)
                vk = _get_k_sorted_mol(self, dm, hermi, -omega, log)
                vk *= hyb
            elif hyb == 0: # SR=0, only LR exchange
                vj = vhfopt.get_j(dm, log)
                vk = _get_k_sorted_mol(self, dm, hermi, omega, log)
                vk *= alpha
            else: # SR and LR exchange with different ratios
                vj, vk = vhfopt.get_jk(mol, dm, hermi)
                vk *= hyb
                vklr = _get_k_sorted_mol(self, dm, hermi, omega, log)
                vklr *= (alpha - hyb)
                vk += vklr

            coeff = cp.asarray(vhfopt.coeff)
            vj = sandwich_dot(vj, coeff)
            vk = sandwich_dot(vk, coeff)
            vj = pack_tril(vj)
            vk = pack_tril(vk)
            vj_last = getattr(vhf_last, 'vj', None)
            vk_last = getattr(vhf_last, 'vk', None)
            if isinstance(vj_last, cp.ndarray):
                vj += vj_last
                vk += vk_last
            else:
                vj = vj_last
                vk = vk_last
            vxc += vj
            vxc -= vk * .5

        logger.timer_debug1(self, 'jk total', *cput1)
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

def _get_k_sorted_mol(mf, dm, hermi, omega, log):
    mol = mf.mol
    vhfopt = mf._opt_gpu.get(omega)
    if vhfopt is None:
        vhfopt = mf._opt_gpu[omega] = jk._VHFOpt(mol, mf.direct_scf_tol).build()
        if isinstance(vhfopt.coeff, cp.ndarray):
            vhfopt.coeff = vhfopt.coeff.get()
    return vhfopt.get_jk(dm, hermi, False, True, log)[1]
