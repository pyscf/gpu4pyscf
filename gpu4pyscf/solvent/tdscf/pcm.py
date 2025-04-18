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
from gpu4pyscf.solvent.pcm import PCM
from gpu4pyscf.solvent.grad.pcm import grad_nuc, grad_qv, grad_solver
from gpu4pyscf.lib import logger
from gpu4pyscf import scf


class TDPCM(PCM):
    def __init__(self, mfpcmobj, eps_optical=1.78, equilium_solvation=False):
        self.__dict__.update(mfpcmobj.__dict__)
        self.equilibrium_solvation = equilium_solvation
        if not equilium_solvation:
            self.eps = eps_optical
        

def make_tdscf_object(tda_method, equilibrium_solvation=False, eps_optical=1.78):
    '''For td_method in vacuum, add td of solvent pcmobj'''
    name = (tda_method._scf.with_solvent.__class__.__name__
            + tda_method.__class__.__name__)
    return lib.set_class(WithSolventTDSCF(tda_method, eps_optical, equilibrium_solvation),
                         (WithSolventTDSCF, tda_method.__class__), name)


def make_tdscf_gradient_object(tda_grad_method):
    '''For td_method in vacuum, add td of solvent pcmobj'''
    name = (tda_grad_method.base._scf.with_solvent.__class__.__name__
            + tda_grad_method.__class__.__name__)
    return lib.set_class(WithSolventTDSCFGradient(tda_grad_method),
                         (WithSolventTDSCFGradient, tda_grad_method.__class__), name)


class WithSolventTDSCF:
    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'with_solvent'}

    def __init__(self, tda_method, eps_optical=1.78, equilibrium_solvation=False):
        self.__dict__.update(tda_method.__dict__)
        self.with_solvent = TDPCM(tda_method._scf.with_solvent, eps_optical, equilibrium_solvation)
        if not self.with_solvent.equilibrium_solvation:
            self.with_solvent.build()

    def gen_response(self, *args, **kwargs):
        pcmobj = self.with_solvent
        mf = self._scf
        vind = super().gen_response(*args, **kwargs)
        is_uhf = isinstance(mf, scf.uhf.UHF)
        # singlet=None is orbital hessian or CPHF type response function
        singlet = kwargs.get('singlet', True)
        singlet = singlet or singlet is None
        def vind_with_solvent(dm1):
            v = vind(dm1)
            if is_uhf:
                v_solvent = pcmobj._B_dot_x(dm1[0]+dm1[1])
                if not self._scf.with_solvent.equilibrium_solvation:
                    v += v_solvent
            elif singlet:
                if not self._scf.with_solvent.equilibrium_solvation:
                    v += pcmobj._B_dot_x(dm1)
            else:
                logger.warn(pcmobj, 'Singlet-Triplet excitation has no LR-PCM contribution!')    
            return v     
        return vind_with_solvent

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.base.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, WithSolventTDSCF, name_mixin))
        return obj
    
    def _finalize(self):
        super()._finalize()
        if self.with_solvent.equilibrium_solvation:
            logger.info(self.with_solvent, 'equilibrium solvation NOT suitable for vertical excitation')
        else:
            logger.info(self.with_solvent, 'Non equilibrium solvation NOT suitable for adiabatic excitation,\n\
                        eps_optical = %s', self.with_solvent.eps)

    def nuc_grad_method(self):
        grad_method = super().nuc_grad_method()
        return make_tdscf_gradient_object(grad_method)


class WithSolventTDSCFGradient:
    from gpu4pyscf.lib.utils import to_gpu, device

    def __init__(self, tda_grad_method):
        self.__dict__.update(tda_grad_method.__dict__)

    def solvent_response(self, dm):
        return self.base.with_solvent._B_dot_x(dm)*2.0 
        
    def grad_elec(self, xy, singlet=None, atmlst=None, verbose=logger.INFO):
        de = super().grad_elec(xy, singlet, atmlst, verbose) 

        assert self.base.with_solvent.equilibrium_solvation
        if self.base.with_solvent.frozen:
            raise RuntimeError('Frozen solvent model is not supported')

        dm = self.base._scf.make_rdm1(ao_repr=True)
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        dmP = 0.5 * (self.dmz1doo + self.dmz1doo.T)
        dmxpy = self.dmxpy + self.dmxpy.T
        pcmobj = self.base.with_solvent
        de += grad_qv(pcmobj, dm)
        de += grad_solver(pcmobj, dm)
        de += grad_nuc(pcmobj, dm)
        
        q_sym_dm = pcmobj._get_qsym(dm, with_nuc = True)[0]
        qE_sym_dmP = pcmobj._get_qsym(dmP)[0]
        qE_sym_dmxpy = pcmobj._get_qsym(dmxpy)[0]
        de += grad_qv(pcmobj, dm, q_sym = qE_sym_dmP)
        de += grad_nuc(pcmobj, dm, q_sym = qE_sym_dmP.get())
        de += grad_qv(pcmobj, dmP, q_sym = q_sym_dm)
        v_grids_l = pcmobj._get_vgrids(dmP, with_nuc = False)
        de += grad_solver(pcmobj, dm, v_grids_l = v_grids_l) * 2.0
        de += grad_qv(pcmobj, dmxpy, q_sym = qE_sym_dmxpy) * 2.0
        v_grids = pcmobj._get_vgrids(dmxpy, with_nuc = False)
        q = pcmobj._get_qsym(dmxpy, with_nuc = False)[1]
        de += grad_solver(pcmobj, dmxpy, v_grids=v_grids, v_grids_l=v_grids, q=q) * 2.0
        
        return de

    def _finalize(self):
        super()._finalize()

