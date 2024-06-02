# gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
#
# Copyright (C) 2022 Qiming Sun
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

import cupy
from pyscf.dft import uks
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.dft import numint, gen_grid, rks
from gpu4pyscf.scf import hf, uhf
from gpu4pyscf.lib.cupy_helper import tag_array


def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + XC functional for UKS.  See pyscf/dft/rks.py
    :func:`get_veff` fore more details.
    '''
    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()
    t0 = logger.init_timer(ks)
    rks.initialize_grids(ks, mol, dm)

    if hasattr(ks, 'screen_tol') and ks.screen_tol is not None:
        ks.direct_scf_tol = ks.screen_tol
    ground_state = getattr(dm, 'ndim', 0) == 3

    ni = ks._numint
    if hermi == 2:  # because rho = 0
        n, exc, vxc = (0,0), 0, 0
    else:
        max_memory = ks.max_memory - lib.current_memory()[0]
        n, exc, vxc = ni.nr_uks(mol, ks.grids, ks.xc, dm, max_memory=max_memory)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        if ks.do_nlc():
            if ni.libxc.is_nlc(ks.xc):
                xc = ks.xc
            else:
                assert ni.libxc.is_nlc(ks.nlc)
                xc = ks.nlc
            n, enlc, vnlc = ni.nr_nlc_vxc(mol, ks.nlcgrids, xc, dm[0]+dm[1],
                                          max_memory=max_memory)
            exc += enlc
            vxc += vnlc
            logger.debug(ks, 'nelec with nlc grids = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

    if not ni.libxc.is_hybrid_xc(ks.xc):
        vk = None
        if (ks._eri is None and ks.direct_scf and
            getattr(vhf_last, 'vj', None) is not None):
            ddm = cupy.asarray(dm) - cupy.asarray(dm_last)
            vj = ks.get_j(mol, ddm[0]+ddm[1], hermi)
            vj += vhf_last.vj
        else:
            vj = ks.get_j(mol, dm[0]+dm[1], hermi)
        vxc += vj
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)
        if (ks._eri is None and ks.direct_scf and
            getattr(vhf_last, 'vk', None) is not None):
            ddm = cupy.asarray(dm) - cupy.asarray(dm_last)
            vj, vk = ks.get_jk(mol, ddm, hermi)
            vk *= hyb
            if abs(omega) > 1e-10:  # For range separated Coulomb operator
                vklr = ks.get_k(mol, ddm, hermi, omega)
                vklr *= (alpha - hyb)
                vk += vklr
            vj = vj[0] + vj[1] + vhf_last.vj
            vk += vhf_last.vk
        else:
            vj, vk = ks.get_jk(mol, dm, hermi)
            vj = vj[0] + vj[1]
            vk *= hyb
            if abs(omega) > 1e-10:
                vklr = ks.get_k(mol, dm, hermi, omega=omega)
                vklr *= (alpha - hyb)
                vk += vklr
        vxc += vj - vk

        if ground_state:
            exc -=(cupy.einsum('ij,ji', dm[0], vk[0]).real +
                   cupy.einsum('ij,ji', dm[1], vk[1]).real) * .5
    if ground_state:
        ecoul = cupy.einsum('ij,ji', dm[0]+dm[1], vj).real * .5
    else:
        ecoul = None
    t0 = logger.timer_debug1(ks, 'jk total', *t0)
    vxc = tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
    return vxc


def energy_elec(ks, dm=None, h1e=None, vhf=None):
    if dm is None: dm = ks.make_rdm1()
    if h1e is None: h1e = ks.get_hcore()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = ks.get_veff(ks.mol, dm)
    if not (isinstance(dm, cupy.ndarray) and dm.ndim == 2):
        dm = dm[0] + dm[1]
    return rks.energy_elec(ks, dm, h1e, vhf)


class UKS(rks.KohnShamDFT, uhf.UHF):
    from gpu4pyscf.lib.utils import to_gpu, device
    _keys = {'disp', 'screen_tol'}

    def __init__(self, mol, xc='LDA,VWN', disp=None):
        uhf.UHF.__init__(self, mol)
        rks.KohnShamDFT.__init__(self, xc)
        self.disp = disp

    get_veff = get_veff
    get_vasp = uks.get_vsap
    energy_elec = energy_elec
    energy_tot = hf.RHF.energy_tot
    init_guess_by_vsap = uks.UKS.init_guess_by_vsap

    to_hf = NotImplemented

    def reset(self, mol=None):
        super().reset(mol)
        self.grids.reset(mol)
        self.nlcgrids.reset(mol)
        self._numint.gdftopt = None
        return self

    def nuc_grad_method(self):
        from gpu4pyscf.grad import uks as uks_grad
        return uks_grad.Gradients(self)

    def to_cpu(self):
        from gpu4pyscf.lib import utils
        mf = uks.UKS(self.mol, xc=self.xc)
        mf.disp = self.disp
        utils.to_cpu(self, mf)
        return mf