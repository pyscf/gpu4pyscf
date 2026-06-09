#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
1-electron Spin-free X2C approximation
'''

import numpy as np
import cupy as cp
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.data import nist
from pyscf.x2c import x2c as x2c_cpu
from gpu4pyscf.lib.cupy_helper import block_diag, asarray
from gpu4pyscf.x2c import x2c
from gpu4pyscf.scf import hf, ghf
from gpu4pyscf.gto.mole import SortedGTO
from gpu4pyscf.df.int3c2e_bdiv import contract_int3c2e_auxvec

__all__ = [
    'sfx2c1e'
]

def sfx2c1e(mf):
    assert isinstance(mf, hf.SCF)
    if isinstance(mf, x2c._X2C_SCF):
        if mf.with_x2c is None:
            mf.with_x2c = SpinFreeX2CHelper(mf.mol)
        else:
            assert isinstance(mf.with_x2c, SpinFreeX2CHelper)
        return mf
    return lib.set_class(SFX2C1E_SCF(mf), (SFX2C1E_SCF, mf.__class__))

class SFX2C1E_SCF(hf.SCF):
    __name_mixin__ = 'sfX2C1e'

    _keys = {'with_x2c'}

    def __init__(self, mf):
        self.__dict__.update(mf.__dict__)
        self.with_x2c = SpinFreeX2CHelper(mf.mol)

    def undo_x2c(self):
        '''Remove the X2C Mixin'''
        obj = lib.view(self, lib.drop_class(self.__class__, SFX2C1E_SCF))
        del obj.with_x2c
        return obj

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        if self.with_x2c:
            self.with_x2c.dump_flags(verbose)
        return self

    def reset(self, mol=None):
        self.with_x2c.reset(mol)
        return super().reset(mol)

    def sfx2c1e(self):
        return self
    x2c = x2c1e = sfx2c1e

    def Gradients(self):
        raise NotImplementedError

    def get_hcore(self, mol=None):
        if self.with_x2c:
            hcore = self.with_x2c.get_hcore(mol)
            if isinstance(self, ghf.GHF):
                hcore = block_diag([hcore, hcore])
            return hcore
        else:
            return super(x2c._X2C_SCF, self).get_hcore(mol)

    def dip_moment(self, mol=None, dm=None, unit='Debye', verbose=logger.NOTE,
                   picture_change=True, **kwargs):
        r''' Dipole moment calculation with picture change correction

        Args:
             mol: an instance of :class:`Mole`
             dm : a 2D ndarrays density matrices

        Kwarg:
            picture_chang (bool) : Whether to compute the dipole moment with
            picture change correction.

        Return:
            A list: the dipole moment on x, y and z component
        '''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        log = logger.new_logger(mol, verbose)

        if dm.ndim == 3: # UHF density matrices
            dm = dm[0] + dm[1]

        if isinstance(self, ghf.GHF):
            nao = mol.nao_nr()
            dm = dm[:nao,:nao] + dm[nao:,nao:]

        with mol.with_common_orig((0,0,0)):
            if picture_change:
                xmol = self.with_x2c.get_xmol()
                dm = xmol.apply_C_mat_CT(dm).get()
                nao = xmol.nao
                int1e_r = xmol.intor_symmetric('int1e_r')
                prp = xmol.intor_symmetric('int1e_sprsp').reshape(3,4,nao,nao)[:,3]
                c1 = 0.5/lib.param.LIGHT_SPEED
                ao_dip = self.with_x2c.picture_change((int1e_r, prp*c1**2))
            else:
                dm = cp.asnumpy(dm)
                ao_dip = mol.intor_symmetric('int1e_r')

        charges = mol.atom_charges()
        coords  = mol.atom_coords()
        nucl_dip = np.einsum('i,ix->x', charges, coords)
        el_dip = np.einsum('xij,ji->x', ao_dip, dm).real
        mol_dip = nucl_dip - el_dip

        if unit.upper() == 'DEBYE':
            mol_dip *= nist.AU2DEBYE
            log.note('Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *mol_dip)
        else:
            log.note('Dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *mol_dip)
        return mol_dip

    def _transfer_attrs_(self, dst):
        if self.with_x2c and not hasattr(dst, 'with_x2c'):
            logger.warn(self, 'Destination object of to_hf/to_ks method is not '
                        'an X2C object. Convert dst to X2C object.')
            dst = dst.sfx2c()
        return hf.SCF._transfer_attrs_(self, dst)

    def to_cpu(self):
        mf = self.undo_x2c().to_cpu().sfx2c1e()
        return lib.to_cpu(self, mf)


class SpinFreeX2CHelper(x2c.X2CHelperBase):
    '''1-component X2c (spin-free part only)
    '''
    def get_hcore(self, mol=None):
        from gpu4pyscf.pbc.gto import int1e
        if mol is None: mol = self.mol
        if mol.has_ecp():
            raise NotImplementedError
        assert '1E' in self.approx.upper()

        xmol = self.with_x2c.get_xmol()
        sort_ao = not self.xuncontract
        c = lib.param.LIGHT_SPEED
        t = int1e.int1e_kin(xmol, sort_output=sort_ao)
        s = int1e.int1e_ovlp(xmol, sort_output=sort_ao)
        #:v = xmol.intor_symmetric('int1e_nuc')
        nucmol = gto.mole.fakemol_for_charges(xmol.atom_coords())
        v = contract_int3c2e_auxvec(xmol, nucmol, -xmol.atom_charges(),
                                    sort_output=sort_ao)
        with lib.temporary_env(xmol, cart=mol.cart):
            w = asarray(xmol.intor_symmetric('int1e_pnucp'))
        if not mol.cart:
            envs = xmol.rys_envs
            s = x2c._orbital_pair_cart2sph(xmol, s, envs)
            t = x2c._orbital_pair_cart2sph(xmol, t, envs)
            v = x2c._orbital_pair_cart2sph(xmol, v, envs)

        if 'ATOM' in self.approx.upper():
            x = _atomic_1e_x(xmol)
            h1 = x2c._get_hcore_fw(t, v, w, s, x, c)
        else:
            h1 = x2c._x2c1e_get_hcore(t, v, w, s, c)

        h1 = x2c._recontract_matrix(xmol, h1)
        return h1

    @lib.with_doc(x2c.X2CHelperBase.picture_change.__doc__)
    def picture_change(self, even_operator=(None, None), odd_operator=None):
        mol = self.mol
        xmol = self.get_xmol(mol)
        pc_mat = self._picture_change(xmol, even_operator, odd_operator)
        if self.basis is not None:
            raise NotImplementedError
        return pc_mat

    def get_xmat(self, mol=None):
        from gpu4pyscf.pbc.gto import int1e
        if mol is None:
            xmol = self.get_xmol(mol)
        else:
            xmol = mol
        sort_ao = not self.xuncontract
        assert '1E' in self.approx.upper()

        if 'ATOM' in self.approx.upper():
            x = _atomic_1e_x(xmol)
        else:
            c = lib.param.LIGHT_SPEED
            t = int1e.int1e_kin(xmol, sort_output=sort_ao)
            s = int1e.int1e_ovlp(xmol, sort_output=sort_ao)
            #:v = xmol.intor_symmetric('int1e_nuc')
            nucmol = gto.mole.fakemol_for_charges(xmol.atom_coords())
            v = contract_int3c2e_auxvec(xmol, nucmol, -xmol.atom_charges(),
                                        sort_output=sort_ao)
            with lib.temporary_env(xmol, cart=mol.cart):
                w = asarray(xmol.intor_symmetric('int1e_pnucp'))
            if not xmol.mol.cart:
                envs = xmol.rys_envs
                s = x2c._orbital_pair_cart2sph(xmol, s, envs)
                t = x2c._orbital_pair_cart2sph(xmol, t, envs)
                v = x2c._orbital_pair_cart2sph(xmol, v, envs)
            x = x2c._x2c1e_xmatrix(t, v, w, s, c)
        return x

    def _get_rmat(self, x=None):
        '''The matrix (in AO basis) that changes metric from NESC metric to NR metric'''
        from gpu4pyscf.pbc.gto import int1e
        xmol = self.get_xmol()
        if x is None:
            x = self.get_xmat(xmol)
        sort_ao = not self.xuncontract
        c = lib.param.LIGHT_SPEED
        s = int1e.int1e_ovlp(xmol, sort_output=sort_ao)
        t = int1e.int1e_kin(xmol, sort_output=sort_ao)
        if not xmol.mol.cart:
            envs = xmol.rys_envs
            s = x2c._orbital_pair_cart2sph(xmol, s, envs)
            t = x2c._orbital_pair_cart2sph(xmol, t, envs)
        s1 = s + x.conj().T.dot(t).dot(x) * (.5/c**2)
        return x2c._get_r(s, s1)

SpinFreeX2C = SpinFreeX2CHelper

def _atomic_1e_x(xmol):
    atoms = x2c._atoms_in_mole(xmol)
    x_conf = {}
    c = lib.param.LIGHT_SPEED
    for elem, atom in atoms.items():
        with atom.with_rinv_at_nucleus(0):
            z = -atom.atom_charge(0)
            v1 = z * atom.intor_symmetric('int1e_rinv')
            w1 = z * atom.intor_symmetric('int1e_prinvp')
            t1 = atom.intor_symmetric('int1e_kin')
            s1 = atom.intor_symmetric('int1e_ovlp')
        x_conf[elem] = asarray(x2c_cpu._x2c1e_xmatrix(t1, v1, w1, s1, c))

    mol = xmol.mol
    l = xmol._bas[:,gto.ANG_OF]
    if mol.cart:
        nf = (l + 1) * (l + 2) // 2
    else:
        nf = l * 2 + 1
    ao_labels = np.repeat(xmol._bas[:,gto.ATOM_OF], nf)
    nao = nf.sum()
    x = cp.zeros((nao, nao))
    for i in range(xmol.natm):
        symb = mol.atom_symbol(i)
        idx = asarray(np.where(ao_labels == i)[0])
        x[idx[:,None],idx] = x_conf[symb]
    return x
