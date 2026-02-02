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
QM/MM helper functions that modify the QM methods.
'''

import numpy as np
import pyscf
from pyscf import gto, lib
from pyscf.qmmm import mm_mole

import gpu4pyscf
from gpu4pyscf.lib import utils, logger
from gpu4pyscf.gto.int3c1e import int1e_grids
from gpu4pyscf.gto.int3c1e_ip import int1e_grids_ip1, int1e_grids_ip2


def add_mm_charges(scf_method, atoms_or_coords, charges, radii=None, unit=None):
    ''' Refer to the comments in the corresponding function in pyscf/qmmm/itrf.py '''
    mol = scf_method.mol
    if unit is None:
        unit = mol.unit
    mm_mol = mm_mole.create_mm_mol(atoms_or_coords, charges,
                                   radii=radii, unit=unit)
    return qmmm_for_scf(scf_method, mm_mol)

mm_charge = add_mm_charges

def qmmm_for_scf(method, mm_mol):
    ''' Refer to the comments in the corresponding function in pyscf/qmmm/itrf.py '''
    assert (isinstance(method, gpu4pyscf.scf.hf.SCF))

    if isinstance(method, QMMM):
        method.mm_mol = mm_mol
        return method

    cls = QMMMSCF

    return lib.set_class(cls(method, mm_mol), (cls, method.__class__))

class QMMM:
    __name_mixin__ = 'QMMM'

_QMMM = QMMM

class QMMMSCF(QMMM):
    _keys = {'mm_mol'}

    def __init__(self, method, mm_mol=None):
        self.__dict__.update(method.__dict__)
        if mm_mol is None:
            mm_mol = gto.Mole()
        self.mm_mol = mm_mol

    def undo_qmmm(self):
        obj = lib.view(self, lib.drop_class(self.__class__, QMMM))
        del obj.mm_mol
        return obj

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        logger.info(self, '** Add background charges for %s **',
                    self.__class__.__name__)
        if self.verbose >= logger.DEBUG:
            logger.debug(self, 'Charge      Location')
            coords = self.mm_mol.atom_coords()
            charges = self.mm_mol.atom_charges()
            for i, z in enumerate(charges):
                logger.debug(self, '%.9g    %s', z, coords[i])
        return self

    def get_hcore(self, mol=None):
        if mol is None:
            mol = self.mol
        h1e = super().get_hcore(mol)

        mm_mol = self.mm_mol
        coords = mm_mol.atom_coords()
        charges = mm_mol.atom_charges()
        if mm_mol.charge_model == 'gaussian':
            expnts = mm_mol.get_zetas()
        else:
            expnts = None
        h1e -= int1e_grids(mol, coords, charges = charges, charge_exponents = expnts)
        return h1e

    def energy_nuc(self):
        # interactions between QM nuclei and MM particles
        nuc = super().energy_nuc()

        assert self.mm_mol.charge_model == 'point' # TODO: support Gaussian charge, same as the one in PCM
        coords = self.mm_mol.atom_coords()
        charges = self.mm_mol.atom_charges()
        nuclear_charges = self.mol.atom_charges()
        nuclear_coords = self.mol.atom_coords()
        r_nuc_ext = np.linalg.norm(nuclear_coords[None, :, :] - coords[:, None, :], axis = 2)
        e_nuc_ext = np.einsum("qA->", (nuclear_charges[None, :] * charges[:, None]) / r_nuc_ext)
        nuc += e_nuc_ext
        return nuc

    to_gpu = utils.to_gpu
    def to_cpu(self):
        obj = self.undo_qmmm().to_cpu()
        obj = pyscf.qmmm.itrf.qmmm_for_scf(obj, self.mm_mol)
        return utils.to_cpu(self, obj)

    def Gradients(self):
        scf_grad = super().Gradients()
        return qmmm_grad_for_scf(scf_grad)


def add_mm_charges_grad(scf_grad, atoms_or_coords, charges, radii=None, unit=None):
    ''' Refer to the comments in the corresponding function in pyscf/qmmm/itrf.py '''
    assert (isinstance(scf_grad, gpu4pyscf.grad.rhf.Gradients))
    mol = scf_grad.mol
    if unit is None:
        unit = mol.unit
    mm_mol = mm_mole.create_mm_mol(atoms_or_coords, charges,
                                   radii=radii, unit=unit)
    mm_grad = qmmm_grad_for_scf(scf_grad)
    mm_grad.base.mm_mol = mm_mol
    return mm_grad

mm_charge_grad = add_mm_charges_grad

def qmmm_grad_for_scf(scf_grad):
    ''' Refer to the comments in the corresponding function in pyscf/qmmm/itrf.py '''
    if getattr(scf_grad.base, 'with_x2c', None):
        raise NotImplementedError('X2C with QM/MM charges')

    # Avoid to initialize QMMMGrad twice
    if isinstance(scf_grad, QMMMGrad):
        return scf_grad

    assert (isinstance(scf_grad.base, gpu4pyscf.scf.hf.SCF) and
           isinstance(scf_grad.base, QMMM))

    return scf_grad.view(lib.make_class((QMMMGrad, scf_grad.__class__)))

class QMMMGrad:
    __name_mixin__ = 'QMMM'

    def __init__(self, scf_grad):
        self.__dict__.update(scf_grad.__dict__)

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        logger.info(self, '** Add background charges for %s **',
                    self.__class__.__name__)
        if self.verbose >= logger.DEBUG1:
            logger.debug1(self, 'Charge      Location')
            coords = self.base.mm_mol.atom_coords()
            charges = self.base.mm_mol.atom_charges()
            for i, z in enumerate(charges):
                logger.debug1(self, '%.9g    %s', z, coords[i])
        return self

    def get_hcore(self, mol=None, exclude_ecp=False):
        ''' (QM 1e grad) + <-d/dX i|q_mm/r_mm|j>'''
        if mol is None:
            mol = self.mol
        g_qm = super().get_hcore(mol, exclude_ecp)

        mm_mol = self.base.mm_mol
        coords = mm_mol.atom_coords()
        charges = mm_mol.atom_charges()
        if mm_mol.charge_model == 'gaussian':
            expnts = mm_mol.get_zetas()
        else:
            expnts = None

        g_qm += int1e_grids_ip1(mol, coords, charges = charges, charge_exponents = expnts)
        return g_qm

    def grad_hcore_mm(self, dm, mol=None):
        r'''Nuclear gradients of the electronic energy
        with respect to MM atoms:

        ... math::
            g = \sum_{ij} \frac{\partial hcore_{ij}}{\partial R_{I}} P_{ji},

        where I represents MM atoms.

        Args:
            dm : array
                The QM density matrix.
        '''
        if mol is None:
            mol = self.mol
        mm_mol = self.base.mm_mol

        coords = mm_mol.atom_coords()
        charges = mm_mol.atom_charges()
        expnts = mm_mol.get_zetas()

        return int1e_grids_ip2(mol, coords, dm = dm, charge_exponents = expnts).T.get() * charges[:, None]

    contract_hcore_mm = grad_hcore_mm

    def grad_nuc(self, mol=None, atmlst=None):
        if mol is None: mol = self.mol
        g_qm = super().grad_nuc(mol, atmlst)

        assert self.base.mm_mol.charge_model == 'point' # TODO: support Gaussian charge, same as the one in PCM
        coords = self.base.mm_mol.atom_coords()
        charges = self.base.mm_mol.atom_charges()

        nuclear_charges = mol.atom_charges()
        nuclear_coords = mol.atom_coords()
        r_nuc_ext = np.linalg.norm(nuclear_coords[None, :, :] - coords[:, None, :], axis = 2)
        g_qm -= np.einsum("qA,qAd->Ad",
                          (nuclear_charges[None, :] * charges[:, None]) / r_nuc_ext**3,
                          nuclear_coords[None, :, :] - coords[:, None, :])
        return g_qm

    def grad_nuc_mm(self, mol=None):
        '''Nuclear gradients of the QM-MM nuclear energy
        (in the form of point charge Coulomb interactions)
        with respect to MM atoms.
        '''
        if mol is None:
            mol = self.mol
        mm_mol = self.base.mm_mol
        coords = mm_mol.atom_coords()
        charges = mm_mol.atom_charges()

        nuclear_charges = mol.atom_charges()
        nuclear_coords = mol.atom_coords()
        r_nuc_ext = np.linalg.norm(nuclear_coords[None, :, :] - coords[:, None, :], axis = 2)
        g_mm = np.einsum("qA,qAd->qd",
                         (nuclear_charges[None, :] * charges[:, None]) / r_nuc_ext**3,
                         nuclear_coords[None, :, :] - coords[:, None, :])
        return g_mm

    to_gpu = utils.to_gpu
    to_cpu = utils.to_cpu

_QMMMGrad = QMMMGrad

# Inject QMMM interface wrapper to other modules
gpu4pyscf.scf.hf.SCF.QMMM = mm_charge
gpu4pyscf.grad.rhf.Gradients.QMMM = mm_charge_grad
