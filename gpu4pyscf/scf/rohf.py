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

from functools import reduce
import numpy as np
import cupy
from pyscf.scf import rohf as rohf_cpu
from gpu4pyscf.scf import hf, uhf
from gpu4pyscf.lib.cupy_helper import tag_array, contract


def get_roothaan_fock(focka_fockb, dma_dmb, s):
    '''Roothaan's effective fock.
    Ref. http://www-theor.ch.cam.ac.uk/people/ross/thesis/node15.html

    ======== ======== ====== =========
    space     closed   open   virtual
    ======== ======== ====== =========
    closed      Fc      Fb     Fc
    open        Fb      Fc     Fa
    virtual     Fc      Fa     Fc
    ======== ======== ====== =========

    where Fc = (Fa + Fb) / 2

    Returns:
        Roothaan effective Fock matrix
    '''
    nao = s.shape[0]
    focka, fockb = focka_fockb
    dma, dmb = dma_dmb
    fc = (focka + fockb) * .5
# Projector for core, open-shell, and virtual
    pc = cupy.dot(dmb, s)
    po = cupy.dot(dma-dmb, s)
    pv = cupy.eye(nao) - cupy.dot(dma, s)
    fock  = reduce(cupy.dot, (pc.conj().T, fc, pc)) * .5
    fock += reduce(cupy.dot, (po.conj().T, fc, po)) * .5
    fock += reduce(cupy.dot, (pv.conj().T, fc, pv)) * .5
    fock += reduce(cupy.dot, (po.conj().T, fockb, pc))
    fock += reduce(cupy.dot, (po.conj().T, focka, pv))
    fock += reduce(cupy.dot, (pv.conj().T, fc, pc))
    fock = fock + fock.conj().T
    fock = tag_array(fock, focka=focka, fockb=fockb)
    return fock

def canonicalize(mf, mo_coeff, mo_occ, fock=None):
    '''Canonicalization diagonalizes the Fock matrix within occupied, open,
    virtual subspaces separatedly (without change occupancy).
    '''
    if getattr(fock, 'focka', None) is None:
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        fock = mf.get_fock(dm=dm)
    mo_e, mo_coeff = hf.canonicalize(mf, mo_coeff, mo_occ, fock)
    fa, fb = fock.focka, fock.fockb
    mo_ea = contract('pi,pi->i', mo_coeff.conj(), fa.dot(mo_coeff)).real
    mo_eb = contract('pi,pi->i', mo_coeff.conj(), fb.dot(mo_coeff)).real
    mo_e = tag_array(mo_e, mo_ea=mo_ea, mo_eb=mo_eb)
    return mo_e, mo_coeff

class ROHF(hf.RHF):
    from gpu4pyscf.lib.utils import to_cpu, to_gpu, device

    nelec = rohf_cpu.ROHF.nelec
    check_sanity = hf.SCF.check_sanity
    get_jk = hf._get_jk
    _eigh = staticmethod(hf.eigh)
    scf = kernel = hf.RHF.kernel
    # FIXME: Needs more tests for get_fock and get_occ
    get_occ = hf.return_cupy_array(rohf_cpu.ROHF.get_occ)
    get_hcore = hf.RHF.get_hcore
    get_ovlp = hf.RHF.get_ovlp
    get_init_guess = uhf.UHF.get_init_guess
    init_guess_by_minao      = rohf_cpu.ROHF.init_guess_by_minao
    init_guess_by_atom       = rohf_cpu.ROHF.init_guess_by_atom
    init_guess_by_huckel     = rohf_cpu.ROHF.init_guess_by_huckel
    init_guess_by_mod_huckel = rohf_cpu.ROHF.init_guess_by_mod_huckel
    init_guess_by_1e         = rohf_cpu.ROHF.init_guess_by_1e
    init_guess_by_chkfile    = rohf_cpu.ROHF.init_guess_by_chkfile
    make_rdm2 = NotImplemented
    x2c = x2c1e = sfx2c1e = NotImplemented
    to_rhf = NotImplemented
    to_uhf = NotImplemented
    to_ghf = NotImplemented
    to_rks = NotImplemented
    to_uks = NotImplemented
    to_gks = NotImplemented
    to_ks = NotImplemented
    analyze = NotImplemented
    stability = NotImplemented
    mulliken_pop = NotImplemented
    mulliken_meta = NotImplemented
    nuc_grad_method = NotImplemented

    canonicalize = canonicalize

    def make_rdm1(self, mo_coeff=None, mo_occ=None, **kwargs):
        '''One-particle density matrix.  mo_occ is a 1D array, with occupancy 1 or 2.
        '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        if isinstance(mo_occ, cupy.ndarray) and mo_occ.ndim == 1:
            mo_occa = (mo_occ > 0).astype(np.double)
            mo_occb = (mo_occ ==2).astype(np.double)
        else:
            mo_occa, mo_occb = mo_occ
        dm_a = cupy.dot(mo_coeff*mo_occa, mo_coeff.conj().T)
        dm_b = cupy.dot(mo_coeff*mo_occb, mo_coeff.conj().T)
        return tag_array((dm_a, dm_b), mo_coeff=mo_coeff, mo_occ=mo_occ)

    def eig(self, fock, s):
        e, c = self._eigh(fock, s)
        if getattr(fock, 'focka', None) is not None:
            mo_ea = contract('pi,pi->i', c.conj(), fock.focka.dot(c)).real
            mo_eb = contract('pi,pi->i', c.conj(), fock.fockb.dot(c)).real
            e = tag_array(e, mo_ea=mo_ea, mo_eb=mo_eb)
        return e, c

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None: dm = self.make_rdm1()
        elif isinstance(dm, cupy.ndarray) and dm.ndim == 2:
            dm = [dm*.5, dm*.5]
        return uhf.energy_elec(self, dm, h1e, vhf)

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
                 diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
                 fock_last=None):
        '''Build fock matrix based on Roothaan's effective fock.
        See also :func:`get_roothaan_fock`
        '''
        if h1e is None: h1e = self.get_hcore()
        if s1e is None: s1e = self.get_ovlp()
        if vhf is None: vhf = self.get_veff(self.mol, dm)
        if dm is None: dm = self.make_rdm1()
        if isinstance(dm, cupy.ndarray) and dm.ndim == 2:
            dm = cupy.repeat(dm[None]*.5, 2, axis=0)
# To Get orbital energy in get_occ, we saved alpha and beta fock, because
# Roothaan effective Fock cannot provide correct orbital energy with `eig`
# TODO, check other treatment  J. Chem. Phys. 133, 141102
        focka = h1e + vhf[0]
        fockb = h1e + vhf[1]
        f = get_roothaan_fock((focka,fockb), dm, s1e)
        if cycle < 0 and diis is None:  # Not inside the SCF iteration
            return f

        if diis_start_cycle is None:
            diis_start_cycle = self.diis_start_cycle
        if level_shift_factor is None:
            level_shift_factor = self.level_shift
        if damp_factor is None:
            damp_factor = self.damp

        dm_tot = dm[0] + dm[1]
        if 0 <= cycle < diis_start_cycle-1 and abs(damp_factor) > 1e-4 and fock_last is not None:
            raise NotImplementedError('ROHF Fock-damping')
        if diis and cycle >= diis_start_cycle:
            f = diis.update(s1e, dm_tot, f, self, h1e, vhf, f_prev=fock_last)
        if abs(level_shift_factor) > 1e-4:
            f = hf.level_shift(s1e, dm_tot*.5, f, level_shift_factor)
        f = tag_array(f, focka=focka, fockb=fockb)
        return f

    def get_veff(self, mol=None, dm=None, dm_last=None, vhf_last=0, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if getattr(dm, 'ndim', 0) == 2:
            dm = cupy.stack((dm*.5,dm*.5))

        if dm_last is None or not self.direct_scf:
            if getattr(dm, 'mo_coeff', None) is not None:
                mo_coeff = dm.mo_coeff
                mo_occ_a = (dm.mo_occ > 0).astype(np.double)
                mo_occ_b = (dm.mo_occ ==2).astype(np.double)
                dm = tag_array(dm, mo_coeff=(mo_coeff,mo_coeff),
                               mo_occ=(mo_occ_a,mo_occ_b))
            vj, vk = self.get_jk(mol, dm, hermi)
            vhf = vj[0] + vj[1] - vk
        else:
            ddm = cupy.asarray(dm) - cupy.asarray(dm_last)
            vj, vk = self.get_jk(mol, ddm, hermi)
            vhf = vj[0] + vj[1] - vk
            vhf += vhf_last
        return vhf

    def get_grad(self, mo_coeff, mo_occ, fock):
        '''ROHF gradients is the off-diagonal block [co + cv + ov], where
        [ cc co cv ]
        [ oc oo ov ]
        [ vc vo vv ]
        '''
        occidxa = mo_occ > 0
        occidxb = mo_occ == 2
        viridxa = ~occidxa
        viridxb = ~occidxb
        uniq_var_a = viridxa.reshape(-1,1) & occidxa
        uniq_var_b = viridxb.reshape(-1,1) & occidxb

        if getattr(fock, 'focka', None) is not None:
            focka = fock.focka
            fockb = fock.fockb
        elif isinstance(fock, (tuple, list)) or getattr(fock, 'ndim', None) == 3:
            focka, fockb = fock
        else:
            focka = fockb = fock
        focka = mo_coeff.conj().T.dot(focka).dot(mo_coeff)
        fockb = mo_coeff.conj().T.dot(fockb).dot(mo_coeff)

        g = cupy.zeros_like(focka)
        g[uniq_var_a]  = focka[uniq_var_a]
        g[uniq_var_b] += fockb[uniq_var_b]
        return g[uniq_var_a | uniq_var_b]

    def newton(self):
        from gpu4pyscf.scf.soscf import newton
        return newton(self)
