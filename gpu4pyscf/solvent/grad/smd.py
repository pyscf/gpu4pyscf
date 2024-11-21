# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
Gradient of SMD solvent model
'''
# pylint: disable=C0103

import numpy as np
import cupy
#from cupyx import scipy, jit
from pyscf import lib
from pyscf.grad import rhf as rhf_grad
from gpu4pyscf.solvent import pcm, smd
from gpu4pyscf.solvent.grad import pcm as pcm_grad
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract

def get_cds(smdobj):
    return smd.get_cds_legacy(smdobj)[1]

"""
def grad_solver(smdobj, dm):
    '''
    dE = 0.5*v* d(K^-1 R) *v + q*dv
    v^T* d(K^-1 R)v = v^T*K^-1(dR - dK K^-1R)v = v^T K^-1(dR - dK q)
    '''
    mol = smdobj.mol
    log = logger.new_logger(mol, mol.verbose)
    t1 = log.init_timer()
    if not smdobj._intermediates:
        smdobj.build()
    dm_cache = smdobj._intermediates.get('dm', None)
    if dm_cache is not None and cupy.linalg.norm(dm_cache - dm) < 1e-10:
        pass
    else:
        smdobj._get_vind(dm)

    gridslice    = smdobj.surface['gslice_by_atom']
    v_grids      = smdobj._intermediates['v_grids']
    A            = smdobj._intermediates['A']
    D            = smdobj._intermediates['D']
    S            = smdobj._intermediates['S']
    K            = smdobj._intermediates['K']
    q            = smdobj._intermediates['q']

    vK_1 = cupy.linalg.solve(K.T, v_grids)

    dF, dA = pcm_grad.get_dF_dA(smdobj.surface)

    with_D = smdobj.method.upper() in ['IEF-PCM', 'IEFPCM', 'SS(V)PE', 'SMD']
    dD, dS, dSii = pcm_grad.get_dD_dS(smdobj.surface, dF, with_D=with_D, with_S=True)

    epsilon = smdobj.eps
    de = cupy.zeros([smdobj.mol.natm,3])

    def contract_bra(a, B, c):
        ''' i,xij,j->jx '''
        tmp = a.dot(B)
        return (tmp * c).T

    def contract_ket(a, B, c):
        ''' i,xij,j->ix '''
        tmp = B.dot(c)
        return (a*tmp).T

    # IEF-PCM and SS(V)PE formally are the same in gradient calculation
    # dR = f_eps/(2*pi) * (dD*A + D*dA),
    # dK = dS - f_eps/(2*pi) * (dD*A*S + D*dA*S + D*A*dS)
    f_epsilon = (epsilon - 1.0)/(epsilon + 1.0)
    fac = f_epsilon/(2.0*np.pi)

    Av = A*v_grids
    de_dR  = 0.5*fac * contract_ket(vK_1, dD, Av)
    de_dR -= 0.5*fac * contract_bra(vK_1, dD, Av)
    de_dR  = cupy.asarray([cupy.sum(de_dR[p0:p1], axis=0) for p0,p1 in gridslice])

    vK_1_D = vK_1.dot(D)
    vK_1_Dv = vK_1_D * v_grids
    de_dR += 0.5*fac * contract('j,xjn->nx', vK_1_Dv, dA)

    de_dS0  = 0.5*contract_ket(vK_1, dS, q)
    de_dS0 -= 0.5*contract_bra(vK_1, dS, q)
    de_dS0  = cupy.asarray([cupy.sum(de_dS0[p0:p1], axis=0) for p0,p1 in gridslice])

    vK_1_q = vK_1 * q
    de_dS0 += 0.5*contract('i,xin->nx', vK_1_q, dSii)

    vK_1_DA = vK_1_D*A
    de_dS1  = 0.5*contract_ket(vK_1_DA, dS, q)
    de_dS1 -= 0.5*contract_bra(vK_1_DA, dS, q)
    de_dS1  = cupy.asarray([cupy.sum(de_dS1[p0:p1], axis=0) for p0,p1 in gridslice])

    vK_1_DAq = vK_1_DA*q
    de_dS1 += 0.5*contract('j,xjn->nx', vK_1_DAq, dSii)

    Sq = cupy.dot(S,q)
    ASq = A*Sq
    de_dD  = 0.5*contract_ket(vK_1, dD, ASq)
    de_dD -= 0.5*contract_bra(vK_1, dD, ASq)
    de_dD  = cupy.asarray([cupy.sum(de_dD[p0:p1], axis=0) for p0,p1 in gridslice])

    de_dA = 0.5*contract('j,xjn->nx', vK_1_D*Sq, dA)   # 0.5*cupy.einsum('j,xjn,j->nx', vK_1_D, dA, Sq)

    de_dK = de_dS0 - fac * (de_dD + de_dA + de_dS1)
    de += de_dR - de_dK

    t1 = log.timer_debug1('grad solver', *t1)
    return de.get()
"""
grad_solver = pcm_grad.grad_solver

def make_grad_object(grad_method):
    '''For grad_method in vacuum, add nuclear gradients of solvent smdobj'''
    if grad_method.base.with_solvent.frozen:
        raise RuntimeError('Frozen solvent model is not avialbe for energy gradients')

    name = (grad_method.base.with_solvent.__class__.__name__
            + grad_method.__class__.__name__)
    return lib.set_class(WithSolventGrad(grad_method),
                         (WithSolventGrad, grad_method.__class__), name)

class WithSolventGrad:
    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'de_solvent', 'de_solute'}

    def __init__(self, grad_method):
        self.__dict__.update(grad_method.__dict__)
        self.de_solvent = None
        self.de_solute = None

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.base.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, WithSolventGrad, name_mixin))
        del obj.de_solvent
        del obj.de_solute
        return obj

    def to_cpu(self):
        from pyscf.solvent.grad import smd      # type: ignore
        grad_method = self.undo_solvent().to_cpu()
        return smd.make_grad_object(grad_method)

    def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        dm = kwargs.pop('dm', None)
        if dm is None:
            dm = self.base.make_rdm1(ao_repr=True)
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        self.de_solute  = super().kernel(*args, **kwargs)
        self.de_solvent = pcm_grad.grad_qv(self.base.with_solvent, dm)
        self.de_solvent+= grad_solver(self.base.with_solvent, dm)
        self.de_solvent+= pcm_grad.grad_nuc(self.base.with_solvent, dm)
        self.de_cds     = get_cds(self.base.with_solvent)
        self.de = self.de_solute + self.de_solvent + self.de_cds
        if self.verbose >= logger.NOTE:
            logger.note(self, '--------------- %s (+%s) gradients ---------------',
                        self.base.__class__.__name__,
                        self.base.with_solvent.__class__.__name__)
            rhf_grad._write(self, self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')
        return self.de

    def _finalize(self):
        # disable _finalize. It is called in grad_method.kernel method
        # where self.de was not yet initialized.
        pass


