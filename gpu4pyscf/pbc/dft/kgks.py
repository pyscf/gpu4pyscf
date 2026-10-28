# Copyright 2026 The PySCF Developers. All Rights Reserved.
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

"""Generalized collinear Kohn-Sham for periodic systems with k-points."""

__all__ = ['KGKS']

import cupy as cp
from pyscf import lib, __config__
from pyscf.pbc.dft import kgks as kgks_cpu
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.pbc.scf import kghf
from gpu4pyscf.pbc.dft import rks, krks, gks, multigrid
from gpu4pyscf.pbc.dft.numint2c import KNumInt2C


def get_veff(ks, cell=None, dm=None, dm_last=None, vhf_last=None, hermi=1,
             kpts=None, kpts_band=None):
    '''Coulomb + XC functional for KGKS

    Args:
        ks : an instance of :class:`GKS`
            XC functional are controlled by ks.xc attribute.  Attribute
            ks.grids might be initialized.
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Returns:
        Veff : ``(nkpts, 2*nao, 2*nao)`` or ``(*, nkpts, 2*nao, 2*nao)`` ndarray
        Veff = J + Vxc.
    '''
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    dm = cp.asarray(dm)
    if kpts is None: kpts = ks.kpts
    t0 = (logger.process_clock(), logger.perf_counter())

    ni = ks._numint
    if ks.do_nlc():
        raise NotImplementedError(f'NLC functional {ks.xc} + {ks.nlc}')

    hybrid = ni.libxc.is_hybrid_xc(ks.xc)

    # TODO GKS with hybrid functional
    hybrid = ks._numint.libxc.is_hybrid_xc(ks.xc)
    if hybrid:
        raise NotImplementedError('Hybrid functionals for periodic GKS')

    # TODO GKS with multigrid method
    if isinstance(ks._numint, multigrid.MultiGridNumIntBase):
        raise NotImplementedError('Multigrid for periodic GKS')

    # ndim = 3 : dm.shape = (nkpts, nao, nao)
    ground_state = (isinstance(dm, cp.ndarray) and dm.ndim == 3 and
                    kpts_band is None)
    ks.initialize_grids(cell, dm, kpts, ground_state)

    # TODO: support non-symmetric density matrix
    assert (hermi == 1)

    max_memory = ks.max_memory - lib.current_memory()[0]
    ni = ks._numint
    n, exc, vxc = ni.get_vxc(cell, ks.grids, ks.xc, dm, hermi=hermi, kpts=kpts,
                             kpts_band=kpts_band, max_memory=max_memory)
    logger.info(ks, 'nelec by numeric integration = %s', n)
    t0 = logger.timer(ks, 'vxc', *t0)

    nkpts = len(kpts)
    weight = 1. / nkpts
    if not hybrid:
        vj = ks.get_j(cell, dm, hermi, kpts, kpts_band)
        vxc += vj
    else:
        omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=cell.spin)
        if omega == 0:
            vj, vk = ks.get_jk(cell, dm, hermi, kpts, kpts_band)
            vk *= hyb
        elif alpha == 0: # LR=0, only SR exchange
            vj = ks.get_j(cell, dm, hermi, kpts, kpts_band)
            vk = ks.get_k(cell, dm, hermi, kpts, kpts_band, omega=-omega)
            vk *= hyb
        elif hyb == 0: # SR=0, only LR exchange
            vj = ks.get_j(cell, dm, hermi, kpts, kpts_band)
            vk = ks.get_k(cell, dm, hermi, kpts, kpts_band, omega=omega)
            vk *= alpha
        else: # SR and LR exchange with different ratios
            vj, vk = ks.get_jk(cell, dm, hermi, kpts, kpts_band)
            vk *= hyb
            vklr = ks.get_k(cell, dm, hermi, kpts, kpts_band, omega=omega)
            vklr *= (alpha - hyb)
            vk += vklr
        vxc += vj - vk

        if ground_state:
            exc -= cp.einsum('Kij,Kji', dm, vk).real * .5

    if ground_state:
        ecoul = float(cp.einsum('Kij,Kji', dm, vj).real) * .5 * weight
    else:
        ecoul = None

    vxc = tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
    return vxc


class KGKS(rks.KohnShamDFT, kghf.KGHF):

    collinear = kgks_cpu.KGKS.collinear
    spin_samples = kgks_cpu.KGKS.spin_samples
    get_veff = get_veff
    energy_elec = krks.energy_elec
    get_rho = kghf.KGHF.get_rho
    density_fit = kghf.KGHF.density_fit

    def __init__(self, cell, kpts=None, xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        kghf.KGHF.__init__(self, cell, kpts, exxdiv)
        rks.KohnShamDFT.__init__(self, xc)
        self._numint = KNumInt2C(self.kpts)

    def dump_flags(self, verbose=None):
        kghf.KGHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    def to_hf(self):
        out = self._transfer_attrs_(kghf.KGHF(self.cell, self.kpts))
        # Pure functionals only build J-type integrals; HF also needs exchange.
        if (not self._numint.libxc.is_hybrid_xc(self.xc) and
            len(self.kpts) > 1 and getattr(out.with_df, '_j_only', False)):
            out.with_df._j_only = False
            out.with_df.reset()
        return out

    to_ghf = to_hf
    to_gks = gks.GKS.to_gks

    def to_cpu(self):
        return utils.to_cpu(self, out=kgks_cpu.KGKS(self.cell, self.kpts))
