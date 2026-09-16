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

import numpy as np
import cupy as cp
from pyscf import lib, gto
from pyscf.pbc.gto import pseudo
from pyscf.gto.mole import ATOM_OF
from pyscf.pbc.lib.kpts_helper import gamma_point
from gpu4pyscf.gto.mole import groupby
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, asarray
from gpu4pyscf.pbc.df import ft_ao
from gpu4pyscf.pbc.df.aft import get_SI
from gpu4pyscf.pbc.gto.pseudo.pp_int import _int_vnl_gpu, _sorted_fake_cell_vnl

def vppnl_nuc_grad(cell, dm, kpts=None):
    '''Nuclear gradients of the non-local part of the GTH pseudo potential,
    contracted with the density matrix.

    Uses GPU CUDA kernels for the r^2/r^4 moment integrals at gamma point,
    with CPU fallback via pyscf _int_vnl for multi-k-point calculations.
    '''
    if kpts is None:
        kpts_lst = np.zeros((1, 3))
    else:
        kpts_lst = np.reshape(kpts, (-1, 3))
    nkpts = len(kpts_lst)

    # pattern stores the unique [hl_dim, l] combinations
    fakecell, hl_blocks, pattern, splits = _sorted_fake_cell_vnl(cell)

    intors_d = ('int1e_ipovlp', 'int1e_r2_origi_ip2', 'int1e_r4_origi_ip2')
    ppnl_half = _int_vnl_gpu(cell, fakecell, hl_blocks, kpts_lst)
    ppnl_half_ip2 = _int_vnl_gpu(cell, fakecell, hl_blocks, kpts_lst, intors_d, comp=3)
    if len(ppnl_half_ip2[0]) > 0:
        ppnl_half_ip2[0] *= -1

    nao = cell.nao
    dm = cp.asarray(dm).reshape(-1, nao, nao)
    if gamma_point(kpts_lst):
        dm = dm.real
    dm_dmH = dm + dm.transpose(0, 2, 1).conj()

    grad = np.zeros([cell.natm, 3], dtype=cp.complex128)
    dppnl = cp.zeros((nao, 3), dtype=cp.complex128)

    hl_offset = [0] * 3
    for ii, (i0, i1) in enumerate(zip(splits[:-1], splits[1:])):
        hl_dim, l = pattern[ii]
        nd = 2 * l + 1
        hl_block = cp.asarray(np.stack(hl_blocks[i0:i1]))
        n_hl = len(hl_block)

        ilp = cp.empty((nkpts, n_hl, hl_dim, nd, nao), dtype=cp.complex128)
        dilp = cp.empty((nkpts, 3, n_hl, hl_dim, nd, nao), dtype=cp.complex128)
        for i in range(hl_dim):
            p0 = hl_offset[i]
            p1 = p0 + n_hl * nd
            ilp[:,:,i] = ppnl_half[i][:,p0:p1].reshape(nkpts, n_hl, nd, nao)
            dilp[:,:,:,i] = ppnl_half_ip2[i][:,:,p0:p1].reshape(nkpts, 3, n_hl, nd, nao).conj()
            hl_offset[i] = p1

        tmp = contract('nij,knjlq->knilq', hl_block, ilp)
        ilp = contract('knilq,kqp->knilp', tmp, dm_dmH, out=ilp)

        value = contract('kdnilp,knilp->nd', dilp, ilp)
        np.add.at(grad, fakecell._bas[i0:i1, ATOM_OF], value.get())

        dppnl += contract('kdnilp,knilp->pd', dilp, ilp)

    ao_loc = cell.ao_loc
    atm_labels = np.repeat(cell._bas[:,ATOM_OF], ao_loc[1:]-ao_loc[:-1])
    grad -= groupby(atm_labels, dppnl.get(), 'sum')

    grad_max_imag = np.max(np.abs(grad.imag))
    if grad_max_imag >= 1e-8:
        logger.warn(cell, f"Large imaginary part ({grad_max_imag:e}) from pseudopotential non-local term gradient.")
    return grad.real

def _get_pp_nonloc_strain_derivatives(cell, mesh, dm_kpts, kpts=None):
    from gpu4pyscf.pbc.grad.rhf import _finite_diff_cells
    if kpts is None:
        assert dm_kpts.ndim == 2
        dm_kpts = dm_kpts[None,:,:]
        kpts = np.zeros((1, 3))
    fakemol = gto.Mole()
    fakemol._atm = np.zeros((1,gto.ATM_SLOTS), dtype=np.int32)
    fakemol._bas = np.zeros((1,gto.BAS_SLOTS), dtype=np.int32)
    ptr = gto.PTR_ENV_START
    fakemol._env = np.zeros(ptr+10)
    fakemol._bas[0,gto.NPRIM_OF ] = 1
    fakemol._bas[0,gto.NCTR_OF  ] = 1
    fakemol._bas[0,gto.PTR_EXP  ] = ptr+3
    fakemol._bas[0,gto.PTR_COEFF] = ptr+4

    ngrids = np.prod(mesh)
    buf = np.empty((48,ngrids), dtype=np.complex128)
    scaled_kpts = kpts.dot(cell.lattice_vectors().T)
    nkpts = len(kpts)

    def eval_pp_nonloc(cell):
        vol = cell.vol
        b = cell.reciprocal_vectors(norm_to=1)
        Gv = cell.get_Gv(mesh)
        SI = get_SI(cell, mesh=mesh)
        # buf for SPG_lmi upto l=0..3 and nl=3
        vppnl = 0
        for k, dm in enumerate(dm_kpts):
            kpt = scaled_kpts[k].dot(b)
            Gk = Gv + kpt
            G_rad = lib.norm(Gk, axis=1)
            aokG = ft_ao.ft_ao(cell, Gv, kpt=kpt) * (1/vol)**.5
            for ia in range(cell.natm):
                symb = cell.atom_symbol(ia)
                if symb not in cell._pseudo:
                    continue
                pp = cell._pseudo[symb]
                p1 = 0
                for l, proj in enumerate(pp[5:]):
                    rl, nl, hl = proj
                    if nl > 0:
                        fakemol._bas[0,gto.ANG_OF] = l
                        fakemol._env[ptr+3] = .5*rl**2
                        fakemol._env[ptr+4] = rl**(l+1.5)*np.pi**1.25
                        pYlm_part = fakemol.eval_gto('GTOval', Gk)

                        p0, p1 = p1, p1+nl*(l*2+1)
                        # pYlm is real, SI[ia] is complex
                        pYlm = np.ndarray((nl,l*2+1,ngrids), dtype=np.complex128, buffer=buf[p0:p1])
                        for k in range(nl):
                            qkl = pseudo.pp._qli(G_rad*rl, l, k)
                            pYlm[k] = pYlm_part.T * qkl
                if p1 > 0:
                    SPG_lmi = asarray(buf[:p1])
                    SPG_lmi *= SI[ia].conj()
                    SPG_lm_aoGs = SPG_lmi.dot(aokG)
                    rho = SPG_lm_aoGs.dot(dm).dot(SPG_lm_aoGs.conj().T).real.get()
                    p1 = 0
                    for l, proj in enumerate(pp[5:]):
                        rl, nl, hl = proj
                        if nl > 0:
                            nf = l * 2 + 1
                            p0, p1 = p1, p1+nl*nf
                            hl = np.asarray(hl)
                            rho_sub = rho[p0:p1,p0:p1].reshape(nl, nf, nl, nf)
                            vppnl += np.einsum('ij,jmim->', hl, rho_sub)
        return vppnl / (nkpts*vol)

    disp = max(1e-5, (cell.precision*.1)**.5)
    out = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp)
            e1 = eval_pp_nonloc(cell1)
            e2 = eval_pp_nonloc(cell2)
            out[i,j] = (e1 - e2) / (2*disp)
    return out
