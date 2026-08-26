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

"""GPU cross-basis integrals for GTH pseudopotentials via merged Cell + SortedGTO."""
import numpy as np
import cupy as cp
from pyscf import gto, lib
from pyscf.pbc.gto.cell import _estimate_rcut
from pyscf.pbc.gto.pseudo.pp_int import fake_cell_vnl, _int_vnl
from pyscf.pbc.lib.kpts_helper import gamma_point
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.gto.mole import most_diffuse_pgto
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.tools import k2gamma


def _int_vnl_gpu(cell, fakecell, hl_blocks, kpts, intors=None, comp=1):
    if intors is None:
        intors = ['int1e_ovlp', 'int1e_r2_origi', 'int1e_r4_origi']

    kern_map = {
        'int1e_ovlp':         ('PBCint1e_ovlp',         1, (0, 0)),
        'int1e_r2_origi':     ('PBCint1e_r2_origi',     1, (0, 2)),
        'int1e_r4_origi':     ('PBCint1e_r4_origi',     1, (0, 4)),
        'int1e_ipovlp':       ('PBCint1e_ipovlp',       3, (1, 0)),
        'int1e_r2_origi_ip2': ('PBCint1e_r2_origi_ip2', 3, (0, 3)),
        'int1e_r4_origi_ip2': ('PBCint1e_r4_origi_ip2', 3, (0, 5)),
    }

    hl_dims = np.asarray([len(hl) for hl in hl_blocks])

    bvk_kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
    pcell = fakecell.copy(deep=False)

    def int_ket(_bas_fake, intor_name):
        if len(_bas_fake) == 0:
            return []

        kern_name, expected_comp, deriv_ij = kern_map[intor_name]
        pcell._bas = _bas_fake
        return int1e.CrossInt1e(pcell, cell, bvk_kmesh).intor(
            kern_name, expected_comp, deriv_ij, kpts)

    return [int_ket(fakecell._bas[hl_dims > 0], intors[0]),
            int_ket(fakecell._bas[hl_dims > 1], intors[1]),
            int_ket(fakecell._bas[hl_dims > 2], intors[2])]

def _sorted_fake_cell_vnl(cell):
    fakecell, hl_blocks = fake_cell_vnl(cell)
    hl_dims = np.asarray([len(hl) for hl in hl_blocks])
    ls = fakecell._bas[:,gto.ANG_OF]
    # groupby [hl_dim, l]
    label = np.stack((hl_dims, ls)).T
    pattern, inv_idx, counts = np.unique(
        label, return_inverse=True, return_counts=True, axis=0)
    idx = np.argsort(inv_idx)
    fakecell._bas = fakecell._bas[idx]
    hl_blocks = [hl_blocks[i] for i in idx]
    splits = np.append(0, counts).cumsum()
    return fakecell, hl_blocks, pattern, splits

def get_pp_nl_gpu(cell, kpts=None):
    if kpts is None:
        kpts_lst = np.zeros((1, 3))
    else:
        kpts_lst = np.reshape(kpts, (-1, 3))
    nkpts = len(kpts_lst)

    # pattern stores the unique [hl_dim, l] combinations
    fakecell, hl_blocks, pattern, splits = _sorted_fake_cell_vnl(cell)

    ppnl_half = _int_vnl_gpu(cell, fakecell, hl_blocks, kpts_lst)

    nao = cell.nao
    ppnl = cp.zeros((nkpts, nao, nao), dtype=cp.complex128)

    hl_offset = [0] * 3
    for ii, (i0, i1) in enumerate(zip(splits[:-1], splits[1:])):
        hl_dim, l = pattern[ii]
        nd = 2 * l + 1
        hl_block = cp.asarray(np.stack(hl_blocks[i0:i1]))
        n_hl = len(hl_block)

        ilp = cp.empty((nkpts, n_hl, hl_dim, nd, nao), dtype=cp.complex128)
        for i in range(hl_dim):
            p0 = hl_offset[i]
            p1 = p0 + n_hl * nd
            ilp[:,:,i] = ppnl_half[i][:,p0:p1].reshape(nkpts, n_hl, nd, nao)
            hl_offset[i] = p1

        tmp = contract('nij,knjlq->knilq', hl_block, ilp)
        ilp_conj = cp.conjugate(ilp, out=ilp)
        contract('knilp,knilq->kpq', ilp_conj, tmp, beta=1, out=ppnl)

    return ppnl
