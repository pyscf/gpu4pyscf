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

"""
Restricted DFT+U with kpoint sampling.
Based on KRHF routine.

Refs: PRB, 1998, 57, 1505.
"""

import cupy as cp
import numpy as np
from pyscf.lo import vec_lowdin
from pyscf.pbc.dft import krkspu as krkspu_cpu
from pyscf.data.nist import HARTREE2EV
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.dft import krks
from gpu4pyscf.lib.cupy_helper import asarray, tag_array

def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    """
    Coulomb + XC functional + Hubbard U terms.

    .. note::
        This is a replica of pyscf.dft.rks.get_veff with kpts added.
        This function will change the ks object.

    Args:
        ks : an instance of :class:`RKS`
            XC functional are controlled by ks.xc attribute.  Attribute
            ks.grids might be initialized.
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Returns:
        Veff : ``(nkpts, nao, nao)`` or ``(*, nkpts, nao, nao)`` ndarray
        Veff = J + Vxc + V_U.
    """
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts

    # J + V_xc
    vxc = krks.get_veff(ks, cell, dm, dm_last=dm_last, vhf_last=vhf_last,
                        hermi=hermi, kpts=kpts, kpts_band=kpts_band)
    vxc = _add_Vhubbard(vxc, ks, dm, kpts)
    return vxc

def _add_Vhubbard(vxc, ks, dm, kpts):
    '''Add Hubbard U to Vxc matrix inplace.
    '''
    C_ao_lo = asarray(ks.C_ao_lo)
    ovlp = ks.get_ovlp()
    nkpts = len(kpts)
    nlo = C_ao_lo.shape[-1]

    rdm1_lo = cp.empty((nkpts, nlo, nlo), dtype=np.complex128)
    for k in range(nkpts):
        C_inv = C_ao_lo[k].conj().T.dot(ovlp[k])
        rdm1_lo[k] = C_inv.dot(dm[k]).dot(C_inv.conj().T)

    is_ibz = hasattr(kpts, "kpts_ibz")
    if is_ibz:
        rdm1_lo_0 = kpts.dm_at_ref_cell(rdm1_lo)

    alphas = ks.alpha
    if not hasattr(alphas, '__len__'): # not a list or tuple
        alphas = [alphas] * len(ks.U_idx)

    E_U = 0.0
    weight = getattr(kpts, "weights_ibz", np.repeat(1.0/nkpts, nkpts))
    logger.info(ks, "-" * 79)
    with np.printoptions(precision=5, suppress=True, linewidth=1000):
        for idx, val, lab, alpha in zip(ks.U_idx, ks.U_val, ks.U_lab, alphas):
            lab_string = " "
            for l in lab:
                lab_string += "%9s" %(l.split()[-1])
            lab_sp = lab[0].split()
            logger.info(ks, "local rdm1 of atom %s: ",
                        " ".join(lab_sp[:2]) + " " + lab_sp[2][:2])
            U_mesh = np.ix_(idx, idx)
            P_loc = 0.0
            for k in range(nkpts):
                S_k = ovlp[k]
                C_k = C_ao_lo[k][:, idx]
                P_k = rdm1_lo[k][U_mesh]
                E_U += weight[k] * (val * 0.5) * (P_k.trace() - P_k.dot(P_k).trace() * 0.5)
                vhub_loc = (cp.eye(P_k.shape[-1]) - P_k) * (val * 0.5)
                if alpha is not None:
                    # The alpha perturbation is only applied to the linear term of
                    # the local density.
                    E_U += weight[k] * alpha * P_k.trace()
                    vhub_loc += cp.eye(P_k.shape[-1]) * alpha
                SC = S_k.dot(C_k)
                vhub_loc = SC.dot(vhub_loc).dot(SC.conj().T)
                if vxc.dtype == np.float64:
                    vhub_loc = vhub_loc.real
                vxc[k] += vhub_loc
                if not is_ibz:
                    P_loc += P_k
            if is_ibz:
                P_loc = rdm1_lo_0[U_mesh].real
            else:
                P_loc = P_loc.real / nkpts
            logger.info(ks, "%s\n%s", lab_string, P_loc)
            logger.info(ks, "-" * 79)

    E_U = E_U.get()[()]
    if E_U.real < 0.0 and all(np.asarray(ks.U_val) > 0):
        logger.warn(ks, "E_U (%g) is negative...", E_U.real)
    vxc = tag_array(vxc, E_U=E_U)
    return vxc

def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf=None):
    """
    Electronic energy for KRKSpU.
    """
    if h1e_kpts is None: h1e_kpts = mf.get_hcore(mf.cell, mf.kpts)
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = mf.get_veff(mf.cell, dm_kpts)

    if hasattr(mf.kpts, "weights_ibz"):
        e1 = cp.einsum('k,kij,kji->', mf.kpts.weights_ibz.dot, h1e_kpts, dm_kpts).get()[()]
    else:
        weight = 1./len(h1e_kpts)
        e1 = weight * cp.einsum('kij,kji', h1e_kpts, dm_kpts).get()[()]
    ecoul = vhf.ecoul
    exc = vhf.exc.real
    E_U = vhf.E_U.real
    e2 = ecoul + exc + E_U
    tot_e = e1 + e2
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['coul'] = ecoul.real
    mf.scf_summary['exc'] = exc.real
    mf.scf_summary['E_U'] = E_U.real
    logger.debug(mf, 'E1 = %s  Ecoul = %s  Exc = %s  EU = %s', e1, ecoul, exc, E_U)
    if abs(ecoul.imag) > mf.cell.precision*10:
        logger.warn(mf, "Coulomb energy has imaginary part %s. "
                    "Coulomb integrals (e-e, e-N) may not converge !",
                    ecoul.imag)
    return tot_e.real, e2.real

def make_minao_lo(ks, minao_ref):
    """
    Construct minao local orbitals.
    """
    cell = ks.cell
    nao = cell.nao
    kpts = ks.kpts
    nkpts = len(kpts)
    ovlp = ks.get_ovlp()
    C_ao_minao, labels = krkspu_cpu.proj_ref_ao(cell, minao=minao_ref, kpts=kpts,
                                                return_labels=True)
    for k in range(nkpts):
        C_ao_minao[k] = vec_lowdin(C_ao_minao[k], ovlp[k].get())

    C_ao_lo = np.zeros((nkpts, nao, nao), dtype=np.complex128)
    for idx, lab in zip(ks.U_idx, ks.U_lab):
        idx_minao = [i for i, l in enumerate(labels) if l in lab]
        assert len(idx_minao) == len(idx)
        C_ao_sub = C_ao_minao[:, :, idx_minao]
        C_ao_lo[:, :, idx] = C_ao_sub
    return C_ao_lo

class KRKSpU(krks.KRKS):
    """
    RKSpU (DFT+U) class adapted for PBCs with k-point sampling.
    """

    _keys = {"U_idx", "U_val", "C_ao_lo", "U_lab", 'alpha'}

    get_veff = get_veff
    energy_elec = energy_elec
    to_hf = NotImplemented

    def __init__(self, cell, kpts=np.zeros((1,3)), xc='LDA,VWN',
                 exxdiv='ewald', U_idx=[], U_val=[], C_ao_lo='minao',
                 minao_ref='MINAO', **kwargs):
        """
        DFT+U args:
            U_idx: can be
                   list of list: each sublist is a set of LO indices to add U.
                   list of string: each string is one kind of LO orbitals,
                                   e.g. ['Ni 3d', '1 O 2pz'], in this case,
                                   LO should be aranged as ao_labels order.
                   or a combination of these two.
            U_val: a list of effective U [in eV], i.e. U-J in Dudarev's DFT+U.
                   each U corresponds to one kind of LO orbitals, should have
                   the same length as U_idx.
            C_ao_lo: LO coefficients, can be
                     np.array, shape ((spin,), nkpts, nao, nlo),
                     string, in 'minao'.
            minao_ref: reference for minao orbitals, default is 'MINAO'.

        Attributes:
            U_idx: same as the input.
            U_val: effectiv U-J [in AU]
            C_ao_loc: np.array
            alpha: the perturbation [in AU] used to compute U in LR-cDFT.
                Refs: Cococcioni and de Gironcoli, PRB 71, 035105 (2005)
        """
        super(self.__class__, self).__init__(cell, kpts, xc=xc, exxdiv=exxdiv, **kwargs)

        krkspu_cpu.set_U(self, U_idx, U_val)

        if isinstance(C_ao_lo, str):
            if C_ao_lo.upper() == 'MINAO':
                self.C_ao_lo = make_minao_lo(self, minao_ref)
            else:
                raise NotImplementedError
        else:
            self.C_ao_lo = C_ao_lo
        if self.C_ao_lo.ndim == 4:
            self.C_ao_lo = self.C_ao_lo[0]

        # The perturbation (eV) used to compute U in LR-cDFT.
        self.alpha = None

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        if log.verbose >= logger.INFO:
            _print_U_info(self, log)
        return self

    def Gradients(self):
        raise NotImplementedError

def _print_U_info(mf, log):
    from pyscf.pbc.dft.krkspu import format_idx
    alphas = mf.alpha
    if not hasattr(alphas, '__len__'): # not a list or tuple
        alphas = [alphas] * len(mf.U_idx)
    log.info("-" * 79)
    log.info('U indices and values: ')
    for idx, val, lab, alpha in zip(mf.U_idx, mf.U_val, mf.U_lab, alphas):
        log.info('%6s [%.6g eV] ==> %-100s', format_idx(idx),
                    val * HARTREE2EV, "".join(lab))
        if alpha is not None:
            log.info('alpha for LR-cDFT %s (eV)', alpha*HARTREE2EV)
    log.info("-" * 79)

def linear_response_u(mf_plus_u, alphalist=(0.02, 0.05, 0.08)):
    # LR-cDFT for Hubbard U is only available fro pyscf>2.9
    from pyscf.pbc.dft.krkspu import linear_response_u
    return linear_response_u(mf_plus_u, alphalist)
