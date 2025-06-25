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
Unrestricted DFT+U with kpoint sampling.
Based on KUHF routine.

Refs: PRB, 1998, 57, 1505.
"""

import numpy as np
import cupy as cp
from pyscf.pbc.dft import kukspu as kukspu_cpu
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.dft import kuks
from gpu4pyscf.pbc.dft.krkspu import make_minao_lo
from gpu4pyscf.lib.cupy_helper import asarray, tag_array

def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    """
    Coulomb + XC functional + (Hubbard - double counting) for KUKSpU.
    """
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts

    # J + V_xc
    vxc = kuks.get_veff(ks, cell, dm, dm_last=dm_last, vhf_last=vhf_last,
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

    rdm1_lo  = cp.zeros((2, nkpts, nlo, nlo), dtype=np.complex128)
    for s in range(2):
        for k in range(nkpts):
            C_inv = C_ao_lo[s, k].conj().T.dot(ovlp[k])
            rdm1_lo[s, k] = C_inv.dot(dm[s][k]).dot(C_inv.conj().T)

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
            for s in range(2):
                P_loc = 0.0
                for k in range(nkpts):
                    S_k = ovlp[k]
                    C_k = C_ao_lo[s, k][:, idx]
                    P_k = rdm1_lo[s, k][U_mesh]
                    E_U += weight[k] * (val * 0.5) * (P_k.trace() - cp.dot(P_k, P_k).trace())
                    vhub_loc = (cp.eye(P_k.shape[-1]) - P_k * 2.0) * (val * 0.5)
                    if alpha is not None:
                        # The alpha perturbation is only applied to the linear term of
                        # the local density.
                        E_U += weight[k] * alpha * P_k.trace()
                        vhub_loc += cp.eye(P_k.shape[-1]) * alpha
                    SC = cp.dot(S_k, C_k)
                    vxc[s,k] += SC.dot(vhub_loc).dot(SC.conj().T).astype(vxc[s,k].dtype,copy=False)
                    if not is_ibz:
                        P_loc += P_k
                if is_ibz:
                    P_loc = rdm1_lo_0[s][U_mesh].real
                else:
                    P_loc = P_loc.real / nkpts
                logger.info(ks, "spin %s\n%s\n%s", s, lab_string, P_loc)
            logger.info(ks, "-" * 79)

    E_U = E_U.get()
    if E_U.real < 0.0 and all(np.asarray(ks.U_val) > 0):
        logger.warn(ks, "E_U (%g) is negative...", E_U.real)
    vxc = tag_array(vxc, E_U=E_U.real)
    return vxc

def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf=None):
    """
    Electronic energy for KUKSpU.
    """
    if h1e_kpts is None: h1e_kpts = mf.get_hcore(mf.cell, mf.kpts)
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = mf.get_veff(mf.cell, dm_kpts)

    weight = getattr(mf.kpts, "weights_ibz",
                     np.array([1.0/len(h1e_kpts),]*len(h1e_kpts)))
    e1 = (cp.einsum('k,kij,kji', weight, h1e_kpts, dm_kpts[0]) +
          cp.einsum('k,kij,kji', weight, h1e_kpts, dm_kpts[1]))
    tot_e = e1 + vhf.ecoul + vhf.exc + vhf.E_U
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['coul'] = vhf.ecoul.real
    mf.scf_summary['exc'] = vhf.exc.real
    mf.scf_summary['E_U'] = vhf.E_U.real

    logger.debug(mf, 'E1 = %s  Ecoul = %s  Exc = %s  EU = %s',
                 e1, vhf.ecoul, vhf.exc, vhf.E_U)
    return tot_e.real, (vhf.ecoul + vhf.exc + vhf.E_U).real

class KUKSpU(kuks.KUKS):
    """
    UKSpU class adapted for PBCs with k-point sampling.
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

        kukspu_cpu.set_U(self, U_idx, U_val)

        if isinstance(C_ao_lo, str):
            if C_ao_lo.upper() == 'MINAO':
                self.C_ao_lo = make_minao_lo(self, minao_ref)
            else:
                raise NotImplementedError
        else:
            self.C_ao_lo = np.asarray(C_ao_lo)
        if self.C_ao_lo.ndim == 3:
            self.C_ao_lo = np.asarray((self.C_ao_lo, self.C_ao_lo))
        elif self.C_ao_lo.ndim == 4:
            if self.C_ao_lo.shape[0] == 1:
                self.C_ao_lo = np.asarray((self.C_ao_lo[0], self.C_ao_lo[0]))
            assert self.C_ao_lo.shape[0] == 2
        else:
            raise ValueError

        # The perturbation (eV) used to compute U in LR-cDFT.
        self.alpha = None

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        if log.verbose >= logger.INFO:
            from gpu4pyscf.pbc.dft.krkspu import _print_U_info
            _print_U_info(self, log)
        return self

    def nuc_grad_method(self):
        raise NotImplementedError

def linear_response_u(mf_plus_u, alphalist=(0.02, 0.05, 0.08)):
    # LR-cDFT for Hubbard U is only available fro pyscf>2.9
    from pyscf.pbc.dft.kukspu import linear_response_u
    return linear_response_u(mf_plus_u, alphalist)
