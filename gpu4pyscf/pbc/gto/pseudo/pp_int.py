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
from pyscf import gto, lib
from pyscf.pbc.gto.cell import _estimate_rcut
from gpu4pyscf.gto.mole import most_diffuse_pgto
from gpu4pyscf.pbc.gto.int1e import _Int1eOpt


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

    def int_ket(_bas_fake, intor_name):
        if len(_bas_fake) == 0:
            return []

        kern_name, expected_comp, deriv_ij = kern_map[intor_name]

        atm_m, bas_m, env_m = gto.conc_env(
            cell._atm, cell._bas, cell._env,
            fakecell._atm, _bas_fake, fakecell._env)

        merged = cell.copy(deep=False)
        merged._atm = np.asarray(atm_m, dtype=np.int32)
        merged._bas = np.asarray(bas_m, dtype=np.int32)
        merged._env = np.asarray(env_m, dtype=np.float64)
        merged._built = True

        bvk_kmesh = np.ones(3, dtype=int)
        a, c, l = most_diffuse_pgto(merged)
        precision = merged.precision * 1e-1
        rcut = _estimate_rcut(a, l, c, precision)
        with lib.temporary_env(merged, precision=precision, rcut=rcut):
            opt = _Int1eOpt(merged, hermi=0, bvk_kmesh=bvk_kmesh)

        mat = opt.intor(kern_name, expected_comp, deriv_ij, None, True).get()

        ao_loc = gto.moleintor.make_loc(bas_m, 'int1e_ovlp_sph')
        i0 = ao_loc[cell.nbas]
        j1 = ao_loc[cell.nbas]

        if expected_comp == 1:
            return mat[i0:, :j1][np.newaxis].astype(np.complex128)
        else:
            return mat[:, i0:, :j1][np.newaxis].astype(np.complex128)

    return (int_ket(fakecell._bas[hl_dims > 0], intors[0]),
            int_ket(fakecell._bas[hl_dims > 1], intors[1]),
            int_ket(fakecell._bas[hl_dims > 2], intors[2]))
