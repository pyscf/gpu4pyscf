#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
DFT+U on molecules

See also the pbc.dft.krkspu and pbc.dft.kukspu module
'''

import numpy as np
import cupy as cp
from pyscf.data.nist import HARTREE2EV
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import asarray
from gpu4pyscf.dft import uks
from gpu4pyscf.dft.rkspu import (
    _set_U, _make_minao_lo, _print_U_info, reference_mol)

def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    """
    Coulomb + XC functional + (Hubbard - double counting) for UKS+U.
    """
    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()

    # J + V_xc
    vxc = uks.get_veff(ks, mol, dm, dm_last=dm_last, vhf_last=vhf_last,
                       hermi=hermi)

    # V_U
    ovlp = asarray(mol.intor('int1e_ovlp', hermi=1))
    pmol = reference_mol(mol, ks.minao_ref)
    U_idx, U_val, U_lab = _set_U(mol, pmol, ks.U_idx, ks.U_val)
    # Construct orthogonal minao local orbitals.
    assert ks.C_ao_lo is None
    C_ao_lo = _make_minao_lo(mol, pmol)

    alphas = ks.alpha
    if not hasattr(alphas, '__len__'): # not a list or tuple
        alphas = [alphas] * len(U_idx)

    E_U = 0.0
    logger.info(ks, "-" * 79)
    lab_string = " "
    with np.printoptions(precision=5, suppress=True, linewidth=1000):
        for idx, val, lab, alpha in zip(U_idx, U_val, U_lab, alphas):
            if ks.verbose >= logger.INFO:
                lab_string = " "
                for l in lab:
                    lab_string += "%9s" %(l.split()[-1])
                lab_sp = lab[0].split()
                logger.info(ks, "local rdm1 of atom %s: ",
                            " ".join(lab_sp[:2]) + " " + lab_sp[2][:2])
            C_loc = C_ao_lo[:,idx]
            SC = ovlp.dot(C_loc) # ~ C^{-1}
            for s in range(2):
                P = SC.conj().T.dot(dm[s]).dot(SC)
                loc_sites = P.shape[-1]
                vhub_loc = (cp.eye(loc_sites) - P * 2.0) * (val * 0.5)
                if alpha is not None:
                    # LR-cDFT perturbation for Hubbard U
                    E_U += alpha * P.trace()
                    vhub_loc += cp.eye(loc_sites) * alpha
                vxc[s] += SC.dot(vhub_loc).dot(SC.conj().T)
                E_U += (val * 0.5) * (P.trace() - P.dot(P).trace())
                logger.info(ks, "spin %s\n%s\n%s", s, lab_string, P)
            logger.info(ks, "-" * 79)

    E_U = E_U.real.get()[()]
    if E_U < 0.0 and all(np.asarray(U_val) > 0):
        logger.warn(ks, "E_U (%s) is negative...", E_U)
    vxc.E_U = E_U
    return vxc

def energy_elec(mf, dm=None, h1e=None, vhf=None):
    """
    Electronic energy for UKSpU.
    """
    if dm is None: dm = mf.make_rdm1()
    if h1e is None: h1e = mf.get_hcore()
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    e1 = cp.einsum('ij,nji->', h1e, dm).get()[()].real
    ecoul = vhf.ecoul.real
    exc = vhf.exc.real
    E_U = vhf.E_U
    if isinstance(ecoul, cp.ndarray):
        ecoul = ecoul.get()[()]
    if isinstance(exc, cp.ndarray):
        exc = exc.get()[()]
    e2 = ecoul + exc + E_U
    mf.scf_summary['e1'] = e1
    mf.scf_summary['coul'] = ecoul
    mf.scf_summary['exc'] = exc
    mf.scf_summary['E_U'] = E_U
    logger.debug(mf, 'E1 = %s  Ecoul = %s  Exc = %s  EU = %s', e1, ecoul, exc, E_U)
    return e1+e2, e2

class UKSpU(uks.UKS):
    """
    UKSpU class adapted for PBCs with k-point sampling.
    """

    _keys = {"U_idx", "U_val", "C_ao_lo", "U_lab", 'minao_ref', 'alpha'}

    get_veff = get_veff
    energy_elec = energy_elec
    to_hf = NotImplemented

    def __init__(self, mol, xc='LDA,VWN',
                 U_idx=[], U_val=[], C_ao_lo=None, minao_ref='MINAO'):
        """
        DFT+U args:
            U_idx: can be
                   list of list: each sublist is a set indices for AO orbitals
                                 (indcies corresponding to the large-basis-set mol).
                   list of string: each string is one kind of LO orbitals,
                                   e.g. ['Ni 3d', '1 O 2pz'].
                   or a combination of these two.
            U_val: a list of effective U [in eV], i.e. U-J in Dudarev's DFT+U.
                   each U corresponds to one kind of LO orbitals, should have
                   the same length as U_idx.
            C_ao_lo: LO coefficients, can be
                     np.array, shape ((spin,), nao, nlo),
                     string, in 'minao'.
            minao_ref: reference for minao orbitals, default is 'MINAO'.

        Attributes:
            U_idx: same as the input.
            U_val: effectiv U-J [in AU]
            C_ao_loc: np.array
            alpha: the perturbation [in AU] used to compute U in LR-cDFT.
                Refs: Cococcioni and de Gironcoli, PRB 71, 035105 (2005)
        """
        super(self.__class__, self).__init__(mol, xc=xc)
        self.U_idx = U_idx
        self.U_val = U_val
        if isinstance(C_ao_lo, str):
            assert C_ao_lo.upper() == 'MINAO'
            C_ao_lo = None # API backward compatibility
        self.C_ao_lo = C_ao_lo
        self.minao_ref = minao_ref
        self.alpha = None

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        if log.verbose >= logger.INFO:
            _print_U_info(self, log)
        return self

    def Gradients(self):
        from gpu4pyscf.grad.ukspu import Gradients
        return Gradients(self)

    def nuc_grad_method(self):
        return self.Gradients()

def linear_response_u(mf_plus_u, alphalist=(0.02, 0.05, 0.08)):
    '''
    Refs:
        [1] M. Cococcioni and S. de Gironcoli, Phys. Rev. B 71, 035105 (2005)
        [2] H. J. Kulik, M. Cococcioni, D. A. Scherlis, and N. Marzari, Phys. Rev. Lett. 97, 103001 (2006)
        [3] Heather J. Kulik, J. Chem. Phys. 142, 240901 (2015)
        [4] https://hjkgrp.mit.edu/tutorials/2011-05-31-calculating-hubbard-u/
        [5] https://hjkgrp.mit.edu/tutorials/2011-06-28-hubbard-u-multiple-sites/

    Args:
        alphalist :
            alpha parameters (in eV) are the displacements for the linear
            response calculations. For each alpha in this list, the DFT+U with
            U=u0+alpha, U=u0-alpha are evaluated. u0 is the U value from the
            reference mf_plus_u object, which will be treated as a standard DFT
            functional.
    '''
    assert isinstance(mf_plus_u, UKSpU)
    assert len(mf_plus_u.U_idx) > 0
    if not mf_plus_u.converged:
        mf_plus_u.run()
    assert mf_plus_u.converged
    # The bare density matrix without adding U
    bare_dm = mf_plus_u.make_rdm1()

    mf = mf_plus_u.copy()
    log = logger.new_logger(mf)

    alphalist = np.asarray(alphalist)
    alphalist = np.append(-alphalist[::-1], alphalist)

    mol = mf.mol
    pmol = reference_mol(mol, mf.minao_ref)
    U_idx, U_val, U_lab = _set_U(mol, pmol, mf.U_idx, mf.U_val)
    assert mf.C_ao_lo is None
    C_ao_lo = _make_minao_lo(mol, pmol)
    ovlp = asarray(mol.intor('int1e_ovlp', hermi=1))
    C_inv = []
    for idx in U_idx:
        c = C_ao_lo[:,idx]
        C_inv.append(c.conj().T.dot(ovlp))

    bare_occupancies = []
    final_occupancies = []
    for alpha in alphalist:
        mf.alpha = alpha / HARTREE2EV
        mf.kernel(dm0=bare_dm)
        local_occ = 0
        for c in C_inv:
            C_on_site = [c.dot(mf.mo_coeff[0]), c.dot(mf.mo_coeff[1])]
            rdm1_lo = mf.make_rdm1(C_on_site, mf.mo_occ)
            local_occ += rdm1_lo[0].trace() + rdm1_lo[1].trace()
        final_occupancies.append(local_occ.get())

        # The first iteration of SCF
        fock = mf.get_fock(dm=bare_dm)
        e, mo = mf.eig(fock, ovlp)
        local_occ = 0
        for c in C_inv:
            C_on_site = [c.dot(mo[0]), c.dot(mo[1])]
            rdm1_lo = mf.make_rdm1(C_on_site, mf.mo_occ)
            local_occ += rdm1_lo[0].trace() + rdm1_lo[1].trace()
        bare_occupancies.append(local_occ.get())
        log.info('alpha=%f bare_occ=%g final_occ=%g',
                 alpha, bare_occupancies[-1], final_occupancies[-1])

    chi0, occ0 = np.polyfit(alphalist, bare_occupancies, deg=1)
    chif, occf = np.polyfit(alphalist, final_occupancies, deg=1)
    log.info('Line fitting chi0 = %f x + %f', chi0, occ0)
    log.info('Line fitting chif = %f x + %f', chif, occf)
    Uresp = 1./chi0 - 1./chif
    log.note('Uresp = %f, chi0 = %f, chif = %f', Uresp, chi0, chif)
    return Uresp
