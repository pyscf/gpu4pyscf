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

from types import SimpleNamespace

import cupy
import numpy
import gpu4pyscf
from pyscf import gto
from pyscf import scf
from pyscf.mp import ump2

from gpu4pyscf.mp.mp2 import MP2Base
from gpu4pyscf.mp.dfump2 import DFUMP2


def test_spin_paired_unequal_occupations_are_noncanonical():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    coeff = cupy.eye(2)
    mf = SimpleNamespace(
        mol=mol,
        mo_coeff=cupy.stack((coeff, coeff)),
        mo_occ=cupy.asarray([[1, 0], [0, 0]]),
        max_memory=4000,
    )
    mp = MP2Base(mf)

    assert not mp._reference_is_canonical()

    mp.mo_occ = cupy.asarray([[1, 0], [1, 0]])
    assert mp._reference_is_canonical()

    mp.mo_occ = cupy.asarray([[1, 0], [0, 0]])
    mp.mo_coeff = cupy.stack((coeff, coeff[:, ::-1]))
    assert mp._reference_is_canonical()


def test_rohf_orbitals_rebuild_ump2_fock_and_run_dfump2():
    mol = gto.M(
        atom='O 0 0 0; H 0 0 0.97', basis='sto-3g', spin=1, verbose=0
    )
    rohf = scf.ROHF(mol).run()

    # UMP2 uses one copy of the ROHF orbitals for each spin, with different
    # occupations in the open-shell orbitals.
    coeff = cupy.asarray(numpy.stack((rohf.mo_coeff, rohf.mo_coeff)))
    occ = cupy.asarray(numpy.stack((rohf.mo_occ == 2, rohf.mo_occ > 0)), dtype=float)
    energy = cupy.asarray(numpy.stack((rohf.mo_energy, rohf.mo_energy)))
    mf_gpu = gpu4pyscf.scf.UHF(mol)
    mf_gpu.mo_coeff = coeff
    mf_gpu.mo_occ = occ
    mf_gpu.mo_energy = energy
    mf_gpu.converged = True

    mp = DFUMP2(mf_gpu, auxbasis='def2-SVP-ri')
    assert not mp._reference_is_canonical()

    # Exercise the PySCF UMP2 ERI initializer used by the compatibility path.
    eris = ump2._ChemistsERIs()._common_init_(mp)
    dm = mf_gpu.make_rdm1(mp.mo_coeff, mp.mo_occ)
    vhf = mf_gpu.get_veff(mol, dm)
    fock_ao = mf_gpu.get_fock(vhf=vhf, dm=dm)
    expected_fock = coeff[0].conj().T @ fock_ao[0] @ coeff[0]
    assert cupy.allclose(eris.fock[0], expected_fock)
    assert not cupy.allclose(expected_fock, cupy.diag(energy[0]))

    # Also cover the complete supported GPU UMP2 energy path for this reference.
    mp.run()
    assert numpy.isfinite(mp.e_corr)
    assert numpy.isclose(mp.e_corr, mp.e_corr_os + mp.e_corr_ss)
