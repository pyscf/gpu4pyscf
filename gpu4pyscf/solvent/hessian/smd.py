# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
Hessian SMD solvent model
'''
# pylint: disable=C0103

import numpy as np
from pyscf import lib
from gpu4pyscf import scf
from gpu4pyscf.lib import logger
from gpu4pyscf.solvent import smd
from gpu4pyscf.solvent.hessian import pcm as pcm_hess
from gpu4pyscf.hessian.jk import _ao2mo
from gpu4pyscf.hessian.rhf import HessianBase

def get_cds(smdobj):
    mol = smdobj.mol
    solvent = smdobj.solvent
    def smd_grad_scanner(mol):
        smdobj_tmp = smd.SMD(mol)
        smdobj_tmp.solvent = solvent
        return smd.get_cds_legacy(smdobj_tmp)[1]

    log = logger.new_logger(mol, mol.verbose)
    t1 = log.init_timer()

    eps = 1e-4
    natm = mol.natm
    hess_cds = np.zeros([natm,natm,3,3])
    for ia in range(mol.natm):
        for j in range(3):
            coords = mol.atom_coords(unit='B')
            coords[ia,j] += eps
            mol.set_geom_(coords, unit='B')
            mol.build()
            grad0_cds = smd_grad_scanner(mol)

            coords[ia,j] -= 2.0*eps
            mol.set_geom_(coords, unit='B')
            mol.build()
            grad1_cds = smd_grad_scanner(mol)

            coords[ia,j] += eps
            mol.set_geom_(coords, unit='B')
            hess_cds[ia,:,j] = (grad0_cds - grad1_cds) / (2.0 * eps)
    t1 = log.timer_debug1('solvent energy', *t1)
    return hess_cds # hartree

def make_hess_object(base_method):
    '''Create nuclear hessian object with solvent contributions for the given
    solvent-attached method based on its hessian method in vaccum
    '''
    if isinstance(base_method, HessianBase):
        # For backward compatibility. In gpu4pyscf-1.4 and older, the input
        # argument is a hessian object.
        base_method = base_method.base

    # Must be a solvent-attached method
    with_solvent = base_method.with_solvent
    if with_solvent.frozen:
        raise RuntimeError('Frozen solvent model is not avialbe for Hessian')

    vac_hess = base_method.undo_solvent().Hessian()
    vac_hess.base = base_method
    name = with_solvent.__class__.__name__ + vac_hess.__class__.__name__
    return lib.set_class(WithSolventHess(vac_hess),
                         (WithSolventHess, vac_hess.__class__), name)

class WithSolventHess:
    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'de_solvent', 'de_solute', 'de_cds'}

    def __init__(self, hess_method):
        self.__dict__.update(hess_method.__dict__)
        self.de_solvent = None
        self.de_solute = None

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.base.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, WithSolventHess, name_mixin))
        del obj.de_solvent
        del obj.de_solute
        return obj

    def to_cpu(self):
        from pyscf.solvent.hessian import smd  # type: ignore
        hess_method = self.undo_solvent().to_cpu()
        return smd.make_hess_object(hess_method)

    def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        if dm is None:
            dm = self.base.make_rdm1()
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        with lib.temporary_env(self.base.with_solvent, equilibrium_solvation=True):
            logger.debug(self, 'Compute hessian from solutes')
            self.de_solute = super().kernel(*args, **kwargs)
        logger.debug(self, 'Compute hessian from solvents')
        self.de_solvent = self.base.with_solvent.hess(dm)
        self.de_cds = get_cds(self.base.with_solvent)
        self.de = self.de_solute + self.de_solvent + self.de_cds
        return self.de

    def make_h1(self, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
        if atmlst is None:
            atmlst = range(self.mol.natm)
        h1ao = super().make_h1(mo_coeff, mo_occ, atmlst=atmlst, verbose=verbose)
        if isinstance(self.base, scf.hf.RHF):
            dm = self.base.make_rdm1()
            dv = pcm_hess.analytical_grad_vmat(self.base.with_solvent, dm, mo_coeff, mo_occ, atmlst=atmlst, verbose=verbose)
            for i0, ia in enumerate(atmlst):
                h1ao[i0] += dv[i0]
            return h1ao
        elif isinstance(self.base, scf.uhf.UHF):
            h1aoa, h1aob = h1ao
            solvent = self.base.with_solvent
            dm = self.base.make_rdm1()
            dm = dm[0] + dm[1]
            dva = pcm_hess.analytical_grad_vmat(solvent, dm, mo_coeff[0], mo_occ[0], atmlst=atmlst, verbose=verbose)
            dvb = pcm_hess.analytical_grad_vmat(solvent, dm, mo_coeff[1], mo_occ[1], atmlst=atmlst, verbose=verbose)
            for i0, ia in enumerate(atmlst):
                h1aoa[i0] += dva[i0]
                h1aob[i0] += dvb[i0]
            return h1aoa, h1aob
        else:
            raise NotImplementedError('Base object is not supported')

    def get_veff_resp_mo(self, mol, dms, mo_coeff, mo_occ, hermi=1):
        v1vo = super().get_veff_resp_mo(mol, dms, mo_coeff, mo_occ, hermi=hermi)
        if not self.base.with_solvent.equilibrium_solvation:
            return v1vo
        v_solvent = self.base.with_solvent._B_dot_x(dms)
        
        if isinstance(self.base, scf.uhf.UHF):
            n_dm = dms.shape[1]
            mocca = mo_coeff[0][:,mo_occ[0]>0]
            moccb = mo_coeff[1][:,mo_occ[1]>0]
            moa, mob = mo_coeff
            nmoa = moa.shape[1]
            nocca = mocca.shape[1]
            v1vo_sol = v_solvent[0] + v_solvent[1]
            v1vo[:,:nmoa*nocca] += _ao2mo(v1vo_sol, mocca, moa).reshape(n_dm,-1)
            v1vo[:,nmoa*nocca:] += _ao2mo(v1vo_sol, moccb, mob).reshape(n_dm,-1)
        elif isinstance(self.base, scf.hf.RHF):
            n_dm = dms.shape[0]
            mocc = mo_coeff[:,mo_occ>0]
            v1vo += _ao2mo(v_solvent, mocc, mo_coeff).reshape(n_dm,-1)
        else:
            raise NotImplementedError('Base object is not supported')
        return v1vo

    def _finalize(self):
        # disable _finalize. It is called in grad_method.kernel method
        # where self.de was not yet initialized.
        pass
