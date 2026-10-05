# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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

import warnings
import numpy as np
import cupy as cp
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.scf.smearing import * # noqa: F403

warnings.warn(
    'smearing functions have been moved to the gpu4pyscf.scf.smearing module.',
    DeprecationWarning
)


def convert_to_uhf(mf, out=None, remove_df=False):
    '''Convert the given mean-field object to the unrestricted HF/KS object

    Note this conversion only changes the class of the mean-field object.
    The total energy and wave-function are the same as them in the input
    mf object. If mf is a second order SCF (SOSCF) object, the SOSCF layer
    will be discarded. Its underlying SCF object mf._scf will be converted.

    Args:
        mf : SCF object

    Kwargs
        remove_df : bool
            Whether to convert the DF-SCF object to the normal SCF object.
            This conversion is not applied by default.

    Returns:
        An unrestricted SCF object
    '''
    from gpu4pyscf import scf
    from gpu4pyscf import dft
    assert isinstance(mf, scf.hf.SCF)
    mf = _without_soscf(mf, remove_df)

    logger.debug(mf, 'Converting %s to UHF', mf.__class__)

    if mf.istype('GHF'):
        raise NotImplementedError

    elif out is not None:
        assert out.istype('UHF')
        _update_mf(mf, out, remove_df)

    elif mf.istype('UHF'):
        return mf.copy()

    else:
        known_cls = {
            dft.roks.ROKS     : dft.uks.UKS,
            dft.rks.RKS       : dft.uks.UKS,
            scf.rohf.ROHF     : scf.uhf.UHF,
            scf.rohf.HF1e     : scf.uhf.UHF,
            scf.hf.RHF        : scf.uhf.UHF,
        }
        out = _object_without_soscf(mf, known_cls, remove_df)

    return _update_mo_to_uhf_(mf, out)

def _without_soscf(mf, remove_df=False):
    from gpu4pyscf.scf.soscf import _CIAH_SOSCF
    from gpu4pyscf.df.df_jk import _DFHF
    if isinstance(mf, _CIAH_SOSCF):
        mf = mf.undo_soscf()
    if remove_df and isinstance(mf, _DFHF):
        mf = mf.undo_df()
    return mf


def _update_mf(mf, out, remove_df):
    from gpu4pyscf.df.df_jk import _DFHF
    if remove_df and isinstance(out, _DFHF):
        raise ValueError('remove_df=True requires an out object without density fitting')
    out.__dict__.update(mf.__dict__)


def _object_without_soscf(mf, known_class, remove_df=False):
    '''Create a new SCF object, retaining mixins such as density fitting.'''
    mf = _without_soscf(mf, remove_df)
    for old_cls in mf.__class__.__mro__:
        if old_cls in known_class:
            break
    else:
        raise NotImplementedError(
            "Incompatible object types. Mean-field `mf` class not found in "
            "`known_class` type.\n\nmf = '%s'\n\nknown_class = '%s'" %
            (mf.__class__, known_class))

    new_cls = known_class[old_cls]
    out = new_cls(mf.mol)
    out.__dict__.update(mf.__dict__)
    out.__class__ = lib.replace_class(mf.__class__, old_cls, new_cls)
    return out

def convert_to_rhf(mf, out=None, remove_df=False):
    '''Convert the given mean-field object to the restricted HF/KS object

    Note this conversion only changes the class of the mean-field object.
    The total energy and wave-function are the same as them in the input
    mf object. If mf is a second order SCF (SOSCF) object, the SOSCF layer
    will be discarded. Its underlying SCF object mf._scf will be converted.

    Args:
        mf : SCF object

    Kwargs
        remove_df : bool
            Whether to convert the DF-SCF object to the normal SCF object.
            This conversion is not applied by default.

    Returns:
        A restricted SCF object
    '''
    from gpu4pyscf import scf
    from gpu4pyscf import dft
    assert isinstance(mf, scf.hf.SCF)
    mf = _without_soscf(mf, remove_df)

    logger.debug(mf, 'Converting %s to RHF', mf.__class__)

    if getattr(mf, 'nelec', None) is None:
        nelec = mf.mol.nelec
    else:
        nelec = mf.nelec

    if mf.istype('GHF'):
        raise NotImplementedError

    elif out is not None:
        assert out.istype('RHF')
        _update_mf(mf, out, remove_df)

    elif ((mf.istype('RHF') and not mf.istype('ROHF')) or
          (nelec[0] != nelec[1] and mf.istype('ROHF'))):
        return mf.copy()

    else:
        if nelec[0] == nelec[1]:
            known_cls = {
                dft.roks.ROKS    : dft.rks.RKS     ,
                dft.uks.UKS      : dft.rks.RKS     ,
                scf.rohf.ROHF    : scf.hf.RHF      ,
                scf.uhf.UHF      : scf.hf.RHF      ,
            }
        else:
            known_cls = {
                dft.uks.UKS      : dft.roks.ROKS    ,
                scf.uhf.UHF      : scf.rohf.ROHF    ,
            }
        out = _object_without_soscf(mf, known_cls, remove_df)

    return _update_mo_to_rhf_(mf, out)

def convert_to_ghf(mf, out=None, remove_df=False):
    '''Convert the given mean-field object to the generalized HF/KS object

    Note this conversion only changes the class of the mean-field object.
    The total energy and wave-function are the same as them in the input
    mf object. If mf is a second order SCF (SOSCF) object, the SOSCF layer
    will be discarded. Its underlying SCF object mf._scf will be converted.

    Args:
        mf : SCF object

    Kwargs
        remove_df : bool
            Whether to convert the DF-SCF object to the normal SCF object.
            This conversion is not applied by default.

    Returns:
        A generalized SCF object
    '''
    from gpu4pyscf import scf
    from gpu4pyscf import dft
    assert isinstance(mf, scf.hf.SCF)
    mf = _without_soscf(mf, remove_df)

    logger.debug(mf, 'Converting %s to GHF', mf.__class__)

    if out is not None:
        assert out.istype('GHF')
        _update_mf(mf, out, remove_df)

    elif mf.istype('GHF'):
        out = mf.copy()

    else:
        known_cls = {
            dft.roks.ROKS     : dft.gks.GKS,
            dft.rks.RKS       : dft.gks.GKS,
            dft.uks.UKS       : dft.gks.GKS,
            scf.rohf.ROHF     : scf.ghf.GHF,
            scf.rohf.HF1e     : scf.ghf.GHF,
            scf.hf.RHF        : scf.ghf.GHF,
            scf.uhf.UHF       : scf.ghf.GHF,
        }
        out = _object_without_soscf(mf, known_cls, remove_df)

    if out.istype('GKS'):
        from gpu4pyscf.dft.numint2c import NumInt2C
        if not isinstance(out._numint, NumInt2C):
            out._numint = NumInt2C()
    return _update_mo_to_ghf_(mf, out)

def _update_mo_to_uhf_(mf, mf1):
    if mf.mo_energy is None:
        return mf1

    if mf.istype('UHF'):
        mf1.mo_occ = mf.mo_occ
        mf1.mo_coeff = mf.mo_coeff
        mf1.mo_energy = mf.mo_energy
    else:  # RHF/ROHF
        if mf.istype('ROHF'):
            mf1.mo_occ = cp.stack((mf.mo_occ>0, mf.mo_occ==2)).astype(np.double)
        else:
            occ = mf.mo_occ
            mf1.mo_occ = cp.stack((occ*.5, occ*.5))
        # ROHF orbital energies, not canonical UHF orbital energies
        mo_ea = getattr(mf.mo_energy, 'mo_ea', mf.mo_energy)
        mo_eb = getattr(mf.mo_energy, 'mo_eb', mf.mo_energy)
        mf1.mo_energy = cp.stack((mo_ea, mo_eb))
        mf1.mo_coeff = cp.stack((mf.mo_coeff, mf.mo_coeff))
    return mf1

def _update_mo_to_rhf_(mf, mf1):
    if mf.mo_energy is None:
        return mf1

    if mf.istype('RHF'): # RHF/ROHF
        mf1.mo_occ = mf.mo_occ
        mf1.mo_coeff = mf.mo_coeff
        mf1.mo_energy = mf.mo_energy
    else:  # UHF
        mf1.mo_occ = mf.mo_occ[0] + mf.mo_occ[1]
        mf1.mo_energy = mf.mo_energy[0]
        mf1.mo_coeff = mf.mo_coeff[0]
        if getattr(mf.mo_coeff[0], 'orbsym', None) is not None:
            mf1.mo_coeff = tag_array(mf1.mo_coeff, orbsym=mf.mo_coeff[0].orbsym)
        mf1.converged = False
    return mf1

def _update_mo_to_ghf_(mf, mf1):
    if mf.mo_energy is None:
        return mf1

    if mf.istype('GHF'):
        return mf1
    elif mf.istype('RHF'):
        nao, nmo = mf.mo_coeff.shape
        if mf.istype('ROHF'):
            mo_occa = (mf.mo_occ>0).astype(np.double)
            mo_occb = (mf.mo_occ==2).astype(np.double)
            orbspin = get_ghf_orbspin(mf.mo_energy, mf.mo_occ, True)
        else:
            mo_occa = mo_occb = mf.mo_occ * .5
            orbspin = cp.tile(cp.array([0, 1]), nmo)

        mf1.mo_energy = cp.empty(nmo*2)
        mf1.mo_energy[orbspin==0] = mf.mo_energy
        mf1.mo_energy[orbspin==1] = mf.mo_energy
        mf1.mo_occ = cp.empty(nmo*2)
        mf1.mo_occ[orbspin==0] = mo_occa
        mf1.mo_occ[orbspin==1] = mo_occb

        mo_coeff = cp.zeros((nao*2,nmo*2), dtype=mf.mo_coeff.dtype)
        mo_coeff[:nao,orbspin==0] = mf.mo_coeff
        mo_coeff[nao:,orbspin==1] = mf.mo_coeff
        if getattr(mf.mo_coeff, 'orbsym', None) is not None:
            orbsym = cp.zeros_like(orbspin)
            orbsym[orbspin==0] = mf.mo_coeff.orbsym
            orbsym[orbspin==1] = mf.mo_coeff.orbsym
            mo_coeff = tag_array(mo_coeff, orbsym=orbsym)
        mf1.mo_coeff = tag_array(mo_coeff, orbspin=orbspin)

    else: # UHF
        nao, nmo = mf.mo_coeff[0].shape
        orbspin = get_ghf_orbspin(mf.mo_energy, mf.mo_occ, False)

        mf1.mo_energy = cp.empty(nmo*2)
        mf1.mo_energy[orbspin==0] = mf.mo_energy[0]
        mf1.mo_energy[orbspin==1] = mf.mo_energy[1]
        mf1.mo_occ = cp.empty(nmo*2)
        mf1.mo_occ[orbspin==0] = mf.mo_occ[0]
        mf1.mo_occ[orbspin==1] = mf.mo_occ[1]

        mo_coeff = cp.zeros((nao*2,nmo*2), dtype=mf.mo_coeff[0].dtype)
        mo_coeff[:nao,orbspin==0] = mf.mo_coeff[0]
        mo_coeff[nao:,orbspin==1] = mf.mo_coeff[1]
        if getattr(mf.mo_coeff[0], 'orbsym', None) is not None:
            orbsym = cp.zeros_like(orbspin)
            orbsym[orbspin==0] = mf.mo_coeff[0].orbsym
            orbsym[orbspin==1] = mf.mo_coeff[1].orbsym
            mo_coeff = tag_array(mo_coeff, orbsym=orbsym)
        mf1.mo_coeff = tag_array(mo_coeff, orbspin=orbspin)
    return mf1

def get_ghf_orbspin(mo_energy, mo_occ, is_rhf=None):
    '''Spin of each GHF orbital when the GHF orbitals are converted from
    RHF/UHF orbitals

    For RHF orbitals, the orbspin corresponds to first occupied orbitals then
    unoccupied orbitals.  In the occupied orbital space, if degenerated, first
    alpha then beta, last the (open-shell) singly occupied (alpha) orbitals. In
    the unoccupied orbital space, first the (open-shell) unoccupied (beta)
    orbitals if applicable, then alpha and beta orbitals

    For UHF orbitals, the orbspin corresponds to first occupied orbitals then
    unoccupied orbitals.

    Fractionally occupied restricted orbitals use interleaved alpha and beta
    spins with equal occupations. Fractional ROHF conversions are unsupported.
    '''
    if is_rhf is None:  # guess whether the orbitals are RHF orbitals
        is_rhf = mo_energy[0].ndim == 0

    if is_rhf:
        nmo = mo_energy.size
        nocc = int(cp.count_nonzero(mo_occ >0))
        nvir = nmo - nocc
        ndocc = int(cp.count_nonzero(mo_occ==2))
        nsocc = nocc - ndocc
        orbspin = cp.array([0,1]*ndocc + [0]*nsocc + [1]*nsocc + [0,1]*nvir)
    else:
        nmo = mo_energy[0].size
        nocca = int(cp.count_nonzero(mo_occ[0]>0))
        nvira = nmo - nocca
        noccb = int(cp.count_nonzero(mo_occ[1]>0))
        nvirb = nmo - noccb
        # round(6) to avoid numerical uncertainty in degeneracy
        es = cp.append(mo_energy[0][mo_occ[0] >0], mo_energy[1][mo_occ[1] >0])
        oidx = cp.argsort(es.round(6), kind='stable')
        es = cp.append(mo_energy[0][mo_occ[0]==0], mo_energy[1][mo_occ[1]==0])
        vidx = cp.argsort(es.round(6), kind='stable')
        orbspin = cp.append(cp.array([0]*nocca+[1]*noccb)[oidx],
                            cp.array([0]*nvira+[1]*nvirb)[vidx])
    return orbspin
