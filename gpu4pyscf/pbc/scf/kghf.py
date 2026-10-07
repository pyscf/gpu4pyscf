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

"""Generalized Hartree-Fock for periodic systems with k-point sampling."""

__all__ = ['KGHF']

from functools import reduce
import numpy as np
import cupy as cp
from pyscf import gto, lib, __config__
from pyscf.data import nist
from pyscf.pbc.scf import kghf as kghf_cpu
from pyscf.pbc.scf import addons, chkfile as chkfile_cpu, kuhf as kuhf_cpu
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.scf import hf as mol_hf
from gpu4pyscf.scf import ghf as mol_ghf
from gpu4pyscf.pbc.scf import khf, ghf


def get_jk(mf, cell=None, dm_kpts=None, hermi=0, kpts=None, kpts_band=None,
           with_j=True, with_k=True, omega=None, **kwargs):
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if kpts is None: kpts = mf.kpts
    return ghf._get_jk(mf, dm_kpts, hermi, kpts, kpts_band, with_j, with_k, omega)


def get_occ(mf, mo_energy_kpts=None, mo_coeff_kpts=None):
    '''Label the occupancies for each orbital for sampled k-points.

    This is a k-point version of scf.hf.SCF.get_occ
    '''
    if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy

    mo_energy_kpts = [cp.asarray(e) for e in mo_energy_kpts]
    nkpts = len(mo_energy_kpts)
    nocc = mf.cell.nelectron * nkpts

    mo_energy = cp.sort(cp.hstack(mo_energy_kpts))
    nmo = mo_energy.size
    if nocc > nmo:
        raise RuntimeError('Failed to assign occupancies. '
                           f'Nocc ({nocc}) > Nmo ({nmo})')
    if nocc == 0:
        return cp.stack([cp.zeros_like(e) for e in mo_energy_kpts])
    fermi = mo_energy[nocc-1]
    mo_occ_kpts = []
    for mo_e in mo_energy_kpts:
        mo_occ_kpts.append((mo_e <= fermi).astype(cp.double))

    if nocc < nmo:
        homo, lumo = mo_energy[nocc-1:nocc+1].get()
        gap = (lumo - homo) * nist.HARTREE2EV
        mf.scf_summary['gap'] = gap
        if mf.verbose >= logger.INFO:
            if homo+1e-3 > lumo:
                logger.warn(mf, 'HOMO %.12g == LUMO %.12g', homo, lumo)
            else:
                logger.info(mf, '  HOMO = %.12g  LUMO = %.12g  gap/eV = %.5f',
                            homo, lumo, gap)
    else:
        logger.info(mf, 'HOMO = %.12g (no LUMO)', mo_energy[nocc-1])

    if mf.verbose >= logger.DEBUG:
        np.set_printoptions(threshold=len(mo_energy))
        logger.debug(mf, '     k-point                  mo_energy')
        for k,kpt in enumerate(mf.cell.get_scaled_kpts(mf.kpts)):
            logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s %s',
                         k, kpt[0], kpt[1], kpt[2],
                         mo_energy_kpts[k][mo_occ_kpts[k]> 0],
                         mo_energy_kpts[k][mo_occ_kpts[k]==0])
        np.set_printoptions(threshold=1000)

    # GPU KSCF stores occupations as a dense CuPy array.
    return cp.stack(mo_occ_kpts)


def _cast_mol_init_guess(fn):
    def fn_init_guess(mf, cell=None, kpts=None):
        if cell is None: cell = mf.cell
        if kpts is None: kpts = mf.kpts
        dm = mol_ghf._from_rhf_init_dm(fn(cell))
        nkpts = len(kpts)
        dm_kpts = cp.stack([dm] * nkpts)
        return dm_kpts
    fn_init_guess.__name__ = fn.__name__
    fn_init_guess.__doc__ = (
        'Generates initial guess density matrix and the orbitals of the initial '
        'guess DM ' + fn.__doc__)
    return fn_init_guess


class KGHF(khf.KSCF):

    with_soc = None
    _keys = {'with_soc'}

    # Complex spinors at -k are not simply the conjugates of spinors at k.
    time_reversal_symmetry = False

    def __init__(self, cell, kpts=None,
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        khf.KSCF.__init__(self, cell, kpts, exxdiv)
        self.with_soc = None

    get_bands = kghf_cpu.KGHF.get_bands
    get_jk = get_jk
    get_occ = get_occ
    get_init_guess = khf.KRHF.get_init_guess
    init_guess_by_1e = khf.KRHF.init_guess_by_1e
    init_guess_by_huckel = khf._cast_mol_init_guess(ghf.GHF.init_guess_by_huckel)
    init_guess_by_mod_huckel = khf._cast_mol_init_guess(ghf.GHF.init_guess_by_mod_huckel)
    density_fit = khf.KRHF.density_fit
    stability = gen_response = nuc_grad_method = Gradients = NotImplemented
    newton = canonicalize = NotImplemented
    smearing = NotImplemented
    _transfer_attrs_ = ghf.GHF._transfer_attrs_

    init_guess_by_minao = _cast_mol_init_guess(mol_hf.init_guess_by_minao)
    init_guess_by_atom = _cast_mol_init_guess(mol_hf.init_guess_by_atom)

    def init_guess_by_chkfile(self, chkfile=None, project=None, kpts=None):
        '''Read periodic HF/KS orbitals into a spin-AO density on kpts.

        project controls projection onto the current AO basis; None projects
        when the basis differs. Spinor projection normalizes both spin blocks
        together. Different k-point meshes use PySCF's density projection.
        kpts is an (nkpts, 3) array in inverse Bohr, defaulting to self.kpts.
        The returned CuPy density has shape (nkpts, 2*nao, 2*nao).
        '''
        # CPU KGHF currently binds the molecular checkpoint reader, which
        # cannot restore k-point spinors. Reuse its periodic projection tools.
        if chkfile is None: chkfile = self.chkfile
        if kpts is None: kpts = self.kpts
        kpts = np.asarray(kpts).reshape(-1, 3)
        cell = self.cell
        chk_cell, scf_rec = chkfile_cpu.load_scf(chkfile)
        mo = scf_rec['mo_coeff']
        occ = scf_rec['mo_occ']
        chk_kpts = np.asarray(scf_rec.get('kpts', scf_rec.get('kpt', np.zeros(3))))
        chk_kpts = chk_kpts.reshape(-1, 3)
        if 'kpts' not in scf_rec and np.ndim(mo) == 2:
            mo = [mo]
            occ = [occ]
        if np.ndim(mo[0]) != 2 or mo[0].shape[0] != 2*chk_cell.nao:
            # Reuse the periodic reader for restricted/unrestricted orbitals.
            dm = cp.asarray(kuhf_cpu.init_guess_by_chkfile(
                cell, chkfile, project, kpts))
            return ghf._block_diag(dm[0], dm[1])

        if project is None:
            project = not gto.same_basis_set(chk_cell, cell)
        if project:
            nao = chk_cell.nao
            moa = addons.project_mo_nr2nr(chk_cell, [c[:nao] for c in mo],
                                         cell, chk_kpts)
            mob = addons.project_mo_nr2nr(chk_cell, [c[nao:] for c in mo],
                                         cell, chk_kpts)
            ovlp = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=chk_kpts)
            mo = []
            for a, b, s in zip(moa, mob, ovlp):
                norm = (np.einsum('pi,pi->i', a.conj(), s.dot(a)) +
                        np.einsum('pi,pi->i', b.conj(), s.dot(b))).real
                mo.append(np.vstack((a, b)) / np.sqrt(norm))
        dm = cp.stack([mol_hf.make_rdm1(c, o) for c, o in zip(mo, occ)])
        if chk_kpts.shape != kpts.shape or not np.allclose(chk_kpts, kpts):
            dm = cp.asarray(addons.project_dm_k2k(
                cell, cp.asnumpy(dm), chk_kpts, kpts))
        # Spinor densities may be complex even at Gamma.
        return dm

    def get_hcore(self, cell=None, kpts=None):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        h = ghf._block_diag(khf.KSCF.get_hcore(self, cell, kpts))
        if self.with_soc and cell.has_ecp_soc():
            raise NotImplementedError('ECP in PBC SCF')
        return h

    def get_ovlp(self, cell=None, kpts=None):
        return ghf._block_diag(khf.KSCF.get_ovlp(self, cell, kpts))

    def eig(self, h_kpts, s_kpts, overwrite=False, x=None,
            time_reversal_symmetry=None):
        return khf.KSCF.eig(self, h_kpts, s_kpts, overwrite, x, False)

    def get_j(self, cell=None, dm_kpts=None, hermi=0, kpts=None, kpts_band=None,
              omega=None):
        return self.get_jk(cell, dm_kpts, hermi, kpts, kpts_band, True, False, omega)[0]

    def get_k(self, cell=None, dm_kpts=None, hermi=0, kpts=None, kpts_band=None,
              omega=None):
        return self.get_jk(cell, dm_kpts, hermi, kpts, kpts_band, False, True, omega)[1]

    def get_veff(self, cell=None, dm_kpts=None, dm_last=None, vhf_last=None, hermi=1,
                 kpts=None, kpts_band=None):
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        vj, vk = self.get_jk(cell, dm_kpts, hermi, kpts, kpts_band, True, True)
        vhf = vj - vk
        if dm_kpts.ndim == 3 and kpts_band is None:
            nkpts = len(dm_kpts)
            ecoul = float(cp.einsum('Kij,Kji->', dm_kpts, vj).real) * .5/nkpts
            vhf = tag_array(vhf, ecoul=ecoul)
        return vhf

    def get_grad(self, mo_coeff_kpts, mo_occ_kpts, fock=None):
        '''
        returns 1D array of gradients, like non K-pt version
        note that occ and virt indices of different k pts now occur
        in sequential patches of the 1D array
        '''
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
            fock = self.get_hcore(self.cell, self.kpts) + self.get_veff(self.cell, dm1)

        def grad(mo, mo_occ, fock):
            occidx = mo_occ > 0
            viridx = ~occidx
            g = reduce(cp.dot, (mo[:,viridx].conj().T, fock, mo[:,occidx]))
            return g.ravel()

        grad_kpts = [grad(mo, mo_occ_kpts[k], fock[k])
                     for k, mo in enumerate(mo_coeff_kpts)]
        return cp.hstack(grad_kpts)

    def get_fermi(self, mo_energy_kpts=None, mo_occ_kpts=None):
        '''Fermi level
        '''
        if mo_energy_kpts is None: mo_energy_kpts = self.mo_energy
        if mo_occ_kpts is None: mo_occ_kpts = self.mo_occ

        # GHF occupations count one electron per spin orbital.
        assert (mo_energy_kpts[0].ndim == 1)
        assert (mo_occ_kpts[0].ndim == 1)

        # occ array in mo_occ_kpts may have different size. See issue #250
        nocc = sum(mo_occ.sum() for mo_occ in mo_occ_kpts)
        # nocc may not be perfect integer when smearing is enabled
        nocc = int(nocc.round(3))
        fermi = cp.sort(cp.hstack(mo_energy_kpts))[nocc-1]

        for k, mo_e in enumerate(mo_energy_kpts):
            mo_occ = mo_occ_kpts[k]
            if mo_occ[mo_e > fermi].sum() > 1.:
                logger.warn(self, 'Occupied band above Fermi level: \n'
                            'k=%d, mo_e=%s, mo_occ=%s', k, mo_e, mo_occ)
        return float(fermi)

    def get_rho(self, dm=None, grids=None, kpts=None):
        if kpts is None: kpts = self.kpts
        return ghf.GHF.get_rho(self, dm, grids, kpts)

    x2c1e = x2c = ghf.GHF.x2c1e

    def to_ghf(self):
        '''Return a copy of this periodic KGHF object.'''
        return self.copy()

    def to_ks(self, xc='HF'):
        '''Convert to periodic KGKS with the requested XC functional.'''
        from gpu4pyscf.pbc.dft.kgks import KGKS
        return self._transfer_attrs_(KGKS(self.cell, self.kpts, xc=xc))

    to_gks = to_ks

    def to_cpu(self):
        mf = kghf_cpu.KGHF(self.cell, self.kpts)
        with lib.temporary_env(self, _numint=None):
            utils.to_cpu(self, out=mf)
        return mf
