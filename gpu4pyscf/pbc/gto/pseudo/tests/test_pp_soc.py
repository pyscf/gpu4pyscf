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

"""GTH SOC integrals: CPU projector/spinor reference and symmetry checks."""
import copy
import os
import unittest
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.symm import sph
from pyscf.pbc import gto
from pyscf.pbc.gto import pseudo
from pyscf.gto.basis import parse_cp2k_pp
from pyscf.pbc.gto.pseudo import pp_int as cpu_pp
from gpu4pyscf.pbc.gto.pseudo import pp_int


def make_cell():
    pp = parse_cp2k_pp.load(
        os.path.join(os.path.dirname(pseudo.__file__), 'GTH_SOC_POTENTIALS'),
        'Zn', 'q12')
    return gto.M(atom='Zn 0.2 0.3 0.1; Zn 2.1 1.8 2.4',
                 a=np.eye(3)*8, unit='Bohr',
                 basis=[[l, [1.2, .7], [.6, .4]] for l in range(3)],
                 pseudo={'Zn': pp}, precision=1e-11, verbose=0)


def scalar_cell(cell, soc=False):
    cell = cell.copy()
    for pp in cell._pseudo.values():
        for l, proj in enumerate(pp[5:]):
            rl, nr, h = proj[:3]
            if soc:
                nr, h = (nr, proj[3]) if l and len(proj) == 4 else (0, [])
            pp[5+l] = (rl, nr, h)
        pp[4] = len(pp) - 5
    return cell


def spinor_reference(cell, kpts):
    # CPU libcint projector integrals, contracted through coupled |l,j,mj>
    # states instead of the GPU's Cartesian L matrices.
    scalar = scalar_cell(cell, soc=True)
    fake, blocks = cpu_pp.fake_cell_vnl(scalar)
    half = cpu_pp._int_vnl(scalar, fake, blocks, kpts)
    offsets = [0, 0, 0]
    nao = cell.nao
    out = np.zeros((len(kpts), 2*nao, 2*nao), dtype=complex)
    for ib, h in enumerate(blocks):
        l = fake.bas_angular(ib)
        nd = 2*l+1
        nr = len(h)
        proj = np.empty((nr, len(kpts), nd, nao), dtype=complex)
        for i in range(nr):
            p0 = offsets[i]
            offsets[i] += nd
            proj[i] = half[i][:,p0:p0+nd]
        ua, ub = sph.sph2spinor(l)
        # L.S eigenvalues for j=l-1/2 and j=l+1/2.
        eig = np.r_[np.full(2*l, -(l+1)*.5), np.full(2*l+2, l*.5)]
        spin_proj = np.concatenate([
            np.einsum('mj,ikmp->ikjp', u.conj(), proj) for u in (ua, ub)
        ], axis=-1)
        out += np.einsum('ikmp,ij,m,jkmq->kpq',
                         spin_proj.conj(), h, eig, spin_proj)
    return out


class KnownValues(unittest.TestCase):
    def check_reference(self, cell, kpts):
        before = copy.deepcopy(cell._pseudo)
        v = pp_int.get_pp_soc(cell, kpts)
        self.assertIsInstance(v, cp.ndarray)
        self.assertEqual(v.shape, (len(kpts), 3, cell.nao, cell.nao))
        v = v.get()
        if not cell.cart:
            np.testing.assert_allclose(v, cpu_pp.get_pp_soc(cell, kpts),
                                       atol=2e-10, rtol=0)
        h = np.einsum('ast,kapq->ksptq', 1j * lib.PauliMatrices, v)
        h = h.reshape(len(kpts), 2*cell.nao, 2*cell.nao)*.5
        np.testing.assert_allclose(h, spinor_reference(cell, kpts), atol=2e-10, rtol=0)
        np.testing.assert_allclose(v, -v.swapaxes(-1, -2).conj(), atol=1e-12)
        self.assertEqual(cell._pseudo, before)
        self.assertGreater(np.max(abs(v)), 1e-5)
        return v

    def test_parser_gamma(self):
        cell = make_cell()
        v = self.check_reference(cell, np.zeros((1, 3)))
        np.testing.assert_allclose(pp_int.get_pp_soc(cell).get(), v, atol=1e-13)
        np.testing.assert_allclose(v.imag, 0, atol=1e-13)
        scalar = pp_int.get_pp_nl(cell).get()
        np.testing.assert_allclose(scalar[0], cpu_pp.get_pp_nl(scalar_cell(cell)),
                                   atol=2e-10, rtol=0)

    def test_kpoints(self):
        cell = make_cell()
        kpt = np.array([.11, -.07, .03])
        v = self.check_reference(cell, np.array([kpt, -kpt]))
        np.testing.assert_allclose(v[1], v[0].conj(), atol=2e-11)
        np.testing.assert_allclose(pp_int.get_pp_soc(cell, kpt).get()[0],
                                   v[0], atol=2e-11)
        self.check_reference(cell, cell.make_kpts([2, 1, 1]))

    def test_three_radial_and_f_projectors(self):
        cell = make_cell()
        pp = cell._pseudo['Zn']
        # Synthetic channels exercise every radial kernel, off-diagonal
        # coefficients, and f projectors without a large heavy-atom AO basis.
        for l, nr in ((1, 3), (2, 2), (3, 1)):
            h = np.eye(nr).tolist()
            k = (np.eye(nr)*.03 + .01).tolist()
            proj = (.45 + .05*l, nr, h, k)
            if 5+l < len(pp):
                pp[5+l] = proj
            else:
                pp.append(proj)
        pp[4] = (4, 'SOC')
        self.check_reference(cell, np.array([[.08, .03, -.05]]))

    def test_cartesian(self):
        cell = make_cell()
        cell.basis = {'Zn': [[l, [.8, 1.]] for l in range(4)]}
        cell.build()
        kpts = np.array([[.1, .04, -.02]])
        sph_v = self.check_reference(cell, kpts)
        cell.cart = True
        cart_v = pp_int.get_pp_soc(cell, kpts).get()
        c = cell.cart2sph_coeff()
        np.testing.assert_allclose(
            np.einsum('pi,kapq,qj->kaij', c, cart_v, c), sph_v,
            atol=2e-10, rtol=0)

    def test_mixed_atoms_and_ghost(self):
        cell = make_cell()
        cell.atom = 'Zn .2 .3 .1; O 2.1 1.8 2.4; ghost-Zn 3.2 .7 1.1'
        cell.pseudo = {'Zn': cell._pseudo['Zn'], 'O': 'gth-pade'}
        cell.build()
        self.check_reference(cell, np.array([[.04, -.03, .02]]))

    def test_no_soc(self):
        cell = scalar_cell(make_cell())
        for kpts in (None, [[.1, .2, .3]]):
            np.testing.assert_array_equal(pp_int.get_pp_soc(cell, kpts).get(), 0)
        cell = make_cell()
        # The s channel has no SOC matrix (the parser stores an empty entry).
        cell._pseudo['Zn'] = cell._pseudo['Zn'][:6]
        cell._pseudo['Zn'][4] = (1, 'SOC')
        np.testing.assert_array_equal(pp_int.get_pp_soc(cell).get(), 0)
        cell._pseudo = {}
        np.testing.assert_array_equal(pp_int.get_pp_soc(cell).get(), 0)
        np.testing.assert_array_equal(pp_int.get_pp_nl(cell).get(), 0)


if __name__ == '__main__':
    unittest.main()
