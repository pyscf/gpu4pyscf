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

'''
TD of PCM family solvent model
'''

import cupy as cp
from pyscf import lib
from gpu4pyscf.solvent.pcm import PI, switch_h, libsolvent
from gpu4pyscf.solvent.grad.pcm import grad_nuc, grad_qv, grad_solver
from gpu4pyscf.solvent.grad.pcm import left_multiply_dS, right_multiply_dS, get_dF_dA, get_dSii, get_dD_dS
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.lib import logger
from gpu4pyscf import scf
from gpu4pyscf.gto import int3c1e


def make_tdscf_object(tda_method):
    '''For td_method in vacuum, add td of solvent pcmobj'''
    name = (tda_method._scf.with_solvent.__class__.__name__
            + tda_method.__class__.__name__)
    return lib.set_class(WithSolventTDSCF(tda_method),
                         (WithSolventTDSCF, tda_method.__class__), name)


def make_tdscf_gradient_object(tda_grad_method):
    '''For td_method in vacuum, add td of solvent pcmobj'''
    name = (tda_grad_method.base._scf.with_solvent.__class__.__name__
            + tda_grad_method.__class__.__name__)
    return lib.set_class(WithSolventTDSCFGradient(tda_grad_method),
                         (WithSolventTDSCFGradient, tda_grad_method.__class__), name)


class WithSolventTDSCF:
    from gpu4pyscf.lib.utils import to_gpu, device

    def __init__(self, tda_method):
        self.__dict__.update(tda_method.__dict__)

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.base.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, WithSolventTDSCF, name_mixin))
        return obj
    
    def _finalize(self):
        super()._finalize()

    def nuc_grad_method(self):
        grad_method = super().nuc_grad_method()
        return make_tdscf_gradient_object(grad_method)


def grad_solver_td(pcmobj, dm, dm1, with_nuc_v = False, with_nuc_q = False):
    '''
    dE = 0.5*v* d(K^-1 R) *v + q*dv
    v^T* d(K^-1 R)v = v^T*K^-1(dR - dK K^-1R)v = v^T K^-1(dR v - dK q)
    '''
    mol = pcmobj.mol
    log = logger.new_logger(mol, mol.verbose)
    t1 = log.init_timer()
    if not pcmobj._intermediates:
        pcmobj.build()
    dm_cache = pcmobj._intermediates.get('dm', None)
    if dm_cache is not None and cp.linalg.norm(dm_cache - dm) < 1e-10:
        pass
    else:
        pcmobj._get_vind(dm)

    gridslice    = pcmobj.surface['gslice_by_atom']
    f_epsilon    = pcmobj._intermediates['f_epsilon']
    if not pcmobj.if_method_in_CPCM_category:
        A = pcmobj._intermediates['A']
        D = pcmobj._intermediates['D']
        S = pcmobj._intermediates['S']

    v_grids = pcmobj._get_vgrids(dm, with_nuc_q)
    q = pcmobj._get_qsym(dm, with_nuc_q)[1]
    v_grids_1 = pcmobj._get_vgrids(dm1, with_nuc_v)

    vK_1 = pcmobj.left_solve_K(v_grids_1, K_transpose = True)

    def contract_bra(a, B, c):
        ''' i,xij,j->jx '''
        tmp = a.dot(B)
        return (tmp*c).T

    def contract_ket(a, B, c):
        ''' i,xij,j->ix '''
        tmp = B.dot(c)
        return (a*tmp).T

    de = cp.zeros([pcmobj.mol.natm,3])
    if pcmobj.method.upper() in ['C-PCM', 'CPCM', 'COSMO']:
        # dR = 0, dK = dS
        de_dS  = 0.5 * vK_1.reshape(-1, 1) * left_multiply_dS(pcmobj.surface, q, stream=None)
        de_dS -= 0.5 * q.reshape(-1, 1) * right_multiply_dS(pcmobj.surface, vK_1, stream=None)
        de -= cp.asarray([cp.sum(de_dS[p0:p1], axis=0) for p0,p1 in gridslice])

        dF, _ = get_dF_dA(pcmobj.surface, with_dA = False)
        dSii = get_dSii(pcmobj.surface, dF)
        de -= 0.5*contract('i,xij->jx', vK_1*q, dSii) # 0.5*cp.einsum('i,xij,i->jx', vK_1, dSii, q)

    elif pcmobj.method.upper() in ['IEF-PCM', 'IEFPCM', 'SMD']:
        dF, dA = get_dF_dA(pcmobj.surface)
        dSii = get_dSii(pcmobj.surface, dF)
        dF = None

        dD, dS = get_dD_dS(pcmobj.surface, with_D=True, with_S=True)

        # dR = f_eps/(2*pi) * (dD*A + D*dA),
        # dK = dS - f_eps/(2*pi) * (dD*A*S + D*dA*S + D*A*dS)
        fac = f_epsilon/(2.0*PI)

        Av = A*v_grids
        de_dR  = 0.5*fac * contract_ket(vK_1, dD, Av)
        de_dR -= 0.5*fac * contract_bra(vK_1, dD, Av)
        de_dR  = cp.asarray([cp.sum(de_dR[p0:p1], axis=0) for p0,p1 in gridslice])

        vK_1_D = vK_1.dot(D)
        vK_1_Dv = vK_1_D * v_grids
        de_dR += 0.5*fac * contract('j,xjn->nx', vK_1_Dv, dA)

        de_dS0  = 0.5*contract_ket(vK_1, dS, q)
        de_dS0 -= 0.5*contract_bra(vK_1, dS, q)
        de_dS0  = cp.asarray([cp.sum(de_dS0[p0:p1], axis=0) for p0,p1 in gridslice])

        vK_1_q = vK_1 * q
        de_dS0 += 0.5*contract('i,xin->nx', vK_1_q, dSii)

        vK_1_DA = vK_1_D*A
        de_dS1  = 0.5*contract_ket(vK_1_DA, dS, q)
        de_dS1 -= 0.5*contract_bra(vK_1_DA, dS, q)
        de_dS1  = cp.asarray([cp.sum(de_dS1[p0:p1], axis=0) for p0,p1 in gridslice])

        vK_1_DAq = vK_1_DA*q
        de_dS1 += 0.5*contract('j,xjn->nx', vK_1_DAq, dSii)

        Sq = cp.dot(S,q)
        ASq = A*Sq
        de_dD  = 0.5*contract_ket(vK_1, dD, ASq)
        de_dD -= 0.5*contract_bra(vK_1, dD, ASq)
        de_dD  = cp.asarray([cp.sum(de_dD[p0:p1], axis=0) for p0,p1 in gridslice])

        de_dA = 0.5*contract('j,xjn->nx', vK_1_D*Sq, dA)   # 0.5*cp.einsum('j,xjn,j->nx', vK_1_D, dA, Sq)

        de_dK = de_dS0 - fac * (de_dD + de_dA + de_dS1)
        de += de_dR - de_dK

    elif pcmobj.method.upper() in [ 'SS(V)PE' ]:
        dF, dA = get_dF_dA(pcmobj.surface)
        dSii = get_dSii(pcmobj.surface, dF)
        dF = None

        dD, dS = get_dD_dS(pcmobj.surface, with_D=True, with_S=True)

        # dR = f_eps/(2*pi) * (dD*A + D*dA),
        # dK = dS - f_eps/(2*pi) * (dD*A*S + D*dA*S + D*A*dS)
        fac = f_epsilon/(2.0*PI)

        Av = A*v_grids
        de_dR  = 0.5*fac * contract_ket(vK_1, dD, Av)
        de_dR -= 0.5*fac * contract_bra(vK_1, dD, Av)
        de_dR  = cp.asarray([cp.sum(de_dR[p0:p1], axis=0) for p0,p1 in gridslice])

        vK_1_D = vK_1.dot(D)
        vK_1_Dv = vK_1_D * v_grids
        de_dR += 0.5*fac * contract('j,xjn->nx', vK_1_Dv, dA)

        de_dS0  = 0.5*contract_ket(vK_1, dS, q)
        de_dS0 -= 0.5*contract_bra(vK_1, dS, q)
        de_dS0  = cp.asarray([cp.sum(de_dS0[p0:p1], axis=0) for p0,p1 in gridslice])

        vK_1_q = vK_1 * q
        de_dS0 += 0.5*contract('i,xin->nx', vK_1_q, dSii)

        vK_1_DA = vK_1_D*A
        de_dS1  = 0.5*contract_ket(vK_1_DA, dS, q)
        de_dS1 -= 0.5*contract_bra(vK_1_DA, dS, q)
        de_dS1  = cp.asarray([cp.sum(de_dS1[p0:p1], axis=0) for p0,p1 in gridslice])
        vK_1_DAq = vK_1_DA*q
        de_dS1 += 0.5*contract('j,xjn->nx', vK_1_DAq, dSii)

        DT_q = cp.dot(D.T, q)
        ADT_q = A * DT_q
        de_dS1_T  = 0.5*contract_ket(vK_1, dS, ADT_q)
        de_dS1_T -= 0.5*contract_bra(vK_1, dS, ADT_q)
        de_dS1_T  = cp.asarray([cp.sum(de_dS1_T[p0:p1], axis=0) for p0,p1 in gridslice])
        vK_1_ADT_q = vK_1 * ADT_q
        de_dS1_T += 0.5*contract('j,xjn->nx', vK_1_ADT_q, dSii)

        Sq = cp.dot(S,q)
        ASq = A*Sq
        de_dD  = 0.5*contract_ket(vK_1, dD, ASq)
        de_dD -= 0.5*contract_bra(vK_1, dD, ASq)
        de_dD  = cp.asarray([cp.sum(de_dD[p0:p1], axis=0) for p0,p1 in gridslice])

        vK_1_S = cp.dot(vK_1, S)
        vK_1_SA = vK_1_S * A
        de_dD_T  = 0.5*contract_ket(vK_1_SA, -dD.transpose(0,2,1), q)
        de_dD_T -= 0.5*contract_bra(vK_1_SA, -dD.transpose(0,2,1), q)
        de_dD_T  = cp.asarray([cp.sum(de_dD_T[p0:p1], axis=0) for p0,p1 in gridslice])

        de_dA = 0.5*contract('j,xjn->nx', vK_1_D*Sq, dA)   # 0.5*cp.einsum('j,xjn,j->nx', vK_1_D, dA, Sq)

        de_dA_T = 0.5*contract('j,xjn->nx', vK_1_S*DT_q, dA)

        de_dK = de_dS0 - 0.5 * fac * (de_dD + de_dA + de_dS1 + de_dD_T + de_dA_T + de_dS1_T)
        de += de_dR - de_dK

    else:
        raise RuntimeError(f"Unknown implicit solvent model: {pcmobj.method}")
    t1 = log.timer_debug1('grad solver', *t1)
    return de.get()


class WithSolventTDSCFGradient:
    from gpu4pyscf.lib.utils import to_gpu, device

    def __init__(self, tda_grad_method):
        self.__dict__.update(tda_grad_method.__dict__)
        
    def grad_elec(self, xy, singlet, atmlst=None, verbose=logger.INFO):
        de = super().grad_elec(xy, singlet, atmlst, verbose) 

        assert self.base._scf.with_solvent.equilibrium_solvation

        dm = self.base._scf.make_rdm1(ao_repr=True)
        # TODO: add unrestricted case support
        dmP = 0.5 * (self.dmz1doo + self.dmz1doo.T)
        dmxpy = 1.0 * (self.dmxpy + self.dmxpy.T)
        pcmobj = self.base._scf.with_solvent
        de += grad_qv(pcmobj, dm)
        de += grad_solver(pcmobj, dm)
        de += grad_nuc(pcmobj, dm)
        
        q_sym_dm = pcmobj._get_qsym(dm, with_nuc = True)[0]
        qE_sym_dmP = pcmobj._get_qsym(dmP)[0]
        qE_sym_dmxpy = pcmobj._get_qsym(dmxpy)[0]
        de += grad_qv(pcmobj, dm, q_sym = qE_sym_dmP)
        de += grad_nuc(pcmobj, dm, q_sym = qE_sym_dmP.get())
        de += grad_qv(pcmobj, dmP, q_sym = q_sym_dm)
        v_grids_1 = pcmobj._get_vgrids(dmP, with_nuc = False)
        de += grad_solver(pcmobj, dm, v_grids_1 = v_grids_1) * 2.0
        de += grad_qv(pcmobj, dmxpy, q_sym = qE_sym_dmxpy) * 2.0
        v_grids = pcmobj._get_vgrids(dmxpy, with_nuc = False)
        q = pcmobj._get_qsym(dmxpy, with_nuc = False)[1]
        v_grids_1 = pcmobj._get_vgrids(dmxpy, with_nuc = False)
        de += grad_solver(pcmobj, dmxpy, v_grids=v_grids, v_grids_1=v_grids_1, q=q) * 2.0
        
        return de

    def _finalize(self):
        super()._finalize()

