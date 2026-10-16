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

"""Generalized Hartree-Fock for periodic systems at a single k-point."""

__all__ = ['GHF']

import numpy as np
import cupy as cp
from pyscf import lib, __config__
from pyscf.data import nist
from pyscf.pbc.scf import ghf as ghf_cpu
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import tag_array, return_cupy_array
from gpu4pyscf.scf import ghf as mol_ghf
from gpu4pyscf.scf import hf as mol_hf
from gpu4pyscf.pbc.scf import hf as pbchf
from gpu4pyscf.pbc.df import GDF


def _block_diag(a, b=None):
    if b is None: b = a
    nao = a.shape[-1]
    out = cp.zeros(a.shape[:-2] + (2*nao, 2*nao),
                   dtype=cp.result_type(a, b))
    out[..., :nao, :nao] = a
    out[..., nao:, nao:] = b
    return out


def _df_jk(mf, dm, hermi, kpts, kpts_band, with_j, with_k, omega):
    if (with_k and
        kpts.ndim == 2 and
        isinstance(mf.with_df, GDF) and mf.with_df._j_only):
        # A preceding get_j call may have built only the diagonal k-pairs.
        mf.with_df._j_only = False
        mf.with_df.reset()

    # Gamma-point GDF stores real integrals and only accepts real densities.
    # Both parts of a complex spin density must contribute to exchange.
    def build(dm, hermi=0):
        if kpts_band is not None:
            raise NotImplementedError
        vj, vk = mf.with_df.get_jk(dm, hermi, kpts, kpts_band, with_j, with_k,
                                   omega=omega, exxdiv=mf.exxdiv)
        if with_j:
            vj = vj.reshape(dm.shape)
        if with_k:
            vk = vk.reshape(dm.shape)
        return vj, vk

    if not cp.iscomplexobj(dm):
        return build(dm, hermi)

    vjR, vkR = build(cp.asarray(dm.real))
    vjI, vkI = build(cp.asarray(dm.imag))
    vj = vk = None
    if with_j:
        vj = vjR + vjI * 1j
    if with_k:
        vk = vkR + vkI * 1j
    return vj, vk

def _get_jk(mf, dm, hermi, kpts, kpts_band, with_j, with_k, omega=None):
    """Split spin blocks while retaining arbitrary leading density dimensions."""
    # Unlike the CPU DF interface, GPU Gamma GDF requires real densities and
    # FFTDF requires Hermitian charge densities. Keep this adapter separate
    # from the CPU get_jk implementation to handle those backend constraints.
    if mf.rsjk is not None:
        raise NotImplementedError('RSJK does not support GHF')
    dm = cp.asarray(dm)
    nao = dm.shape[-1] // 2
    vj = vk = None
    # Build K first so a fresh GDF cache includes all exchange k-pairs.
    if with_k:
        blocks = [dm[..., :nao, :nao], dm[..., nao:, nao:],
                  dm[..., :nao, nao:]]
        if hermi != 1:
            blocks.append(dm[..., nao:, :nao])
        k = _df_jk(mf, cp.stack(blocks), 0, kpts, kpts_band,
                   False, True, omega)[1]
        vk = _block_diag(k[0], k[1])
        vk[..., :nao, nao:] = k[2]
        vk[..., nao:, :nao] = k[2].swapaxes(-1, -2).conj() if hermi == 1 else k[3]
    if with_j:
        charge = dm[..., :nao, :nao] + dm[..., nao:, nao:]
        if hermi == 1:
            j = _df_jk(mf, charge, 1, kpts, kpts_band, True, False, omega)[0]
        else:
            # FFTDF builds real charge densities. Decompose a general density
            # into two Hermitian parts before applying this linear map.
            adj = charge.swapaxes(-1, -2).conj()
            jr = _df_jk(mf, (charge+adj)*.5, 1, kpts, kpts_band,
                        True, False, omega)[0]
            ji = _df_jk(mf, (charge-adj)/2j, 1, kpts, kpts_band,
                        True, False, omega)[0]
            j = jr + 1j*ji
        vj = _block_diag(j)
    return vj, vk


def get_jk(mf, cell=None, dm=None, hermi=0, kpt=None, kpts_band=None,
           with_j=True, with_k=True, omega=None, **kwargs):
    if dm is None: dm = mf.make_rdm1()
    if kpt is None: kpt = mf.kpt
    return _get_jk(mf, dm, hermi, kpt, kpts_band, with_j, with_k, omega)


def get_occ(mf, mo_energy=None, mo_coeff=None):
    if mo_energy is None: mo_energy = mf.mo_energy
    mo_energy = cp.asarray(mo_energy)
    e_idx = cp.argsort(mo_energy.round(9), kind='stable')
    nmo = mo_energy.size
    mo_occ = cp.zeros_like(mo_energy)
    nocc = mf.mol.nelectron
    if 0 < nocc < nmo:
        homo, lumo = mo_energy[e_idx[nocc-1:nocc+1]].get()
        gap = (lumo - homo) * nist.HARTREE2EV
        mf.scf_summary['gap'] = gap
        if mf.verbose >= logger.INFO:
            if homo+1e-3 > lumo:
                logger.warn(mf, 'HOMO %.15g == LUMO %.15g', homo, lumo)
            else:
                logger.info(mf, '  HOMO = %.15g  LUMO = %.15g  gap/eV = %.5f',
                            homo, lumo, gap)
    elif nocc > nmo:
        raise RuntimeError(f'Failed to assign mo_occ. Nocc ({nocc}) > Nmo ({nmo})')
    mo_occ[e_idx[:nocc]] = 1

    if mf.verbose >= logger.DEBUG:
        np.set_printoptions(threshold=nmo)
        logger.debug(mf, '  mo_energy =\n%s', mo_energy)
        np.set_printoptions(threshold=1000)

    if mo_coeff is not None and mf.verbose >= logger.DEBUG:
        ss, s = mf.spin_square(mo_coeff[:,mo_occ>0], mf.get_ovlp())
        logger.debug(mf, 'multiplicity <S^2> = %.8g  2S+1 = %.8g', ss, s)
    return mo_occ


class GHF(pbchf.SCF):

    with_soc = None
    _keys = {'with_soc'}

    def __init__(self, cell, kpt=None,
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        pbchf.SCF.__init__(self, cell, kpt, exxdiv)
        self.with_soc = None

    @pbchf.SCF.kpt.setter
    def kpt(self, kpt):
        self.with_df.kpts = np.reshape(kpt, (1, 3))
        if self.rsjk is not None:
            self.rsjk.kpts = self.with_df.kpts

    get_jk = get_jk
    get_occ = get_occ
    _finalize = ghf_cpu.GHF._finalize
    get_bands = ghf_cpu.GHF.get_bands
    density_fit = pbchf.RHF.density_fit
    energy_elec = mol_hf.energy_elec
    get_init_guess = pbchf.SCF.get_init_guess
    init_guess_by_huckel = return_cupy_array(ghf_cpu.GHF.init_guess_by_huckel)
    init_guess_by_mod_huckel = return_cupy_array(ghf_cpu.GHF.init_guess_by_mod_huckel)
    init_guess_by_chkfile = return_cupy_array(ghf_cpu.GHF.init_guess_by_chkfile)
    stability = gen_response = nuc_grad_method = Gradients = NotImplemented
    newton = canonicalize = NotImplemented
    smearing = NotImplemented

    def _transfer_attrs_(self, dst):
        # The destination needs its own scalar or spin-AO XC integrator.
        ni = dst._numint
        dst = mol_hf.SCF._transfer_attrs_(self, dst)
        dst._numint = ni
        return dst

    def get_hcore(self, cell=None, kpt=None):
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt
        h = _block_diag(pbchf.SCF.get_hcore(self, cell, kpt))
        if self.with_soc and cell.has_ecp_soc():
            raise NotImplementedError('ECP in PBC SCF')
        return h

    def get_ovlp(self, cell=None, kpt=None):
        return _block_diag(pbchf.SCF.get_ovlp(self, cell, kpt))

    def get_j(self, cell=None, dm=None, hermi=1, kpt=None, kpts_band=None,
              omega=None):
        return self.get_jk(cell, dm, hermi, kpt, kpts_band, True, False, omega)[0]

    def get_k(self, cell=None, dm=None, hermi=1, kpt=None, kpts_band=None,
              omega=None):
        return self.get_jk(cell, dm, hermi, kpt, kpts_band, False, True, omega)[1]

    def get_veff(self, cell=None, dm=None, dm_last=None, vhf_last=None, hermi=1,
                 kpt=None, kpts_band=None):
        if dm is None: dm = self.make_rdm1()
        vj, vk = self.get_jk(cell, dm, hermi, kpt, kpts_band)
        vhf = vj - vk
        if dm.ndim == 2 and kpts_band is None:
            ecoul = float(cp.einsum('ij,ji->', dm, vj).real) * .5
            vhf = tag_array(vhf, ecoul=ecoul)
        return vhf

    def init_guess_by_minao(self, mol=None):
        return mol_ghf._from_rhf_init_dm(mol_hf.SCF.init_guess_by_minao(self, mol))

    def init_guess_by_atom(self, mol=None):
        return mol_ghf._from_rhf_init_dm(mol_hf.SCF.init_guess_by_atom(self, mol))

    def init_guess_by_1e(self, mol=None):
        if mol is None: mol = self.mol
        logger.info(self, 'Initial guess from hcore.')
        h1e = self.get_hcore(mol)
        s1e = self.get_ovlp(mol)
        mo_energy, mo_coeff = self.eig(h1e, s1e)
        mo_occ = self.get_occ(mo_energy, mo_coeff)
        return self.make_rdm1(mo_coeff, mo_occ)

    def spin_square(self, mo_coeff=None, s=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff[:,self.mo_occ>0]
        if s is None: s = self.get_ovlp()
        return mol_ghf.GHF.spin_square(self, mo_coeff, s)

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(self.mol) + self.get_veff(self.mol, dm1)
        occidx = mo_occ > 0
        viridx = ~occidx
        g = mo_coeff[:,occidx].T.conj().dot(
            fock.dot(mo_coeff[:,viridx]))
        return g.conj().T.ravel()

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
                 diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
                 fock_last=None):
        # GHF occupations are 0/1: apply the level shift with D, not D/2.
        f = mol_hf.get_fock(self, h1e, s1e, vhf, dm, cycle, diis,
                            diis_start_cycle, 0., damp_factor, fock_last)
        shift = self.level_shift if level_shift_factor is None else level_shift_factor
        if shift and not (cycle < 0 and diis is None):
            if s1e is None: s1e = self.get_ovlp()
            if dm is None: dm = self.make_rdm1()
            f = mol_hf.level_shift(s1e, dm, f, shift)
        return f

    def get_fermi(self):
        # Follow periodic HF, with one electron per occupied spin orbital.
        nocc = int(self.mo_occ.sum().round(3))
        return float(self.mo_energy[nocc-1])

    def get_rho(self, dm=None, grids=None, kpt=None):
        if dm is None: dm = self.make_rdm1()
        nao = self.cell.nao
        charge = dm[..., :nao, :nao] + dm[..., nao:, nao:]
        # Use scalar numerical integration for the spin trace.
        from gpu4pyscf.pbc.dft import numint, gen_grid
        if grids is None:
            grids = getattr(self, 'grids', None)
        if grids is None:
            grids = gen_grid.UniformGrids(self.cell)
        if grids.coords is None:
            grids.build()
        if kpt is None: kpt = self.kpt
        return numint.NumInt().get_rho(self.cell, charge, grids, kpt)

    def x2c1e(self):
        from gpu4pyscf.pbc.x2c.x2c1e import x2c1e_gscf
        return x2c1e_gscf(self)
    x2c = sfx2c1e = x2c1e

    def to_ks(self, xc='HF'):
        from gpu4pyscf.pbc.dft.gks import GKS
        return self._transfer_attrs_(GKS(self.cell, self.kpt, xc=xc))

    def to_cpu(self):
        mf = ghf_cpu.GHF(self.cell)
        with lib.temporary_env(self, _numint=None):
            utils.to_cpu(self, out=mf)
        return mf
