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


def make_tdscf_object(tda_method, equilibrium_solvation=False):
    '''For td_method in vacuum, add td of solvent pcmobj'''
    assert hasattr(tda_method._scf, 'with_solvent')
    with_solvent = tda_method._scf.with_solvent.copy()
    with_solvent.equilibrium_solvation = equilibrium_solvation
    if not equilibrium_solvation:
        # The vertical excitation is a fast process, applying non-equilibrium
        # solvation with optical dielectric constant eps=1.78
        # TODO: reset() can be skipped. Most intermeidates can be reused.
        with_solvent.reset()
        with_solvent.eps = 1.78
        with_solvent.build()
    name = (tda_method._scf.with_solvent.__class__.__name__
            + tda_method.__class__.__name__)
    return lib.set_class(WithSolventTDSCF(tda_method, with_solvent),
                         (WithSolventTDSCF, tda_method.__class__), name)


def make_tdscf_gradient_object(td_base_method):
    '''For td_method in vacuum, add td of solvent pcmobj'''
    # The nuclear gradients of stable exited states should correspond to a
    # fully relaxed solvent. Strictly, the TDDFT exited states should be
    # solved using state-specific solvent model. Even if running LR-PCM for
    # the zeroth order TDDFT, the wavefunction should be comptued using the
    # same dielectric constant as the ground state (the zero-frequency eps).
    with_solvent = td_base_method.with_solvent
    if not with_solvent.equilibrium_solvation:
        raise RuntimeError(
            'When computing gradients of PCM-TDDFT, equilibrium solvation should '
            'be employed. The PCM TDDFT should be initialized as\n'
            '    mf.TDDFT(equilibrium_solvation=True)')
    td_grad = td_base_method.undo_solvent().Gradients()
    td_grad.base = td_base_method
    name = with_solvent.__class__.__name__ + td_grad.__class__.__name__
    return lib.set_class(WithSolventTDSCFGradient(td_grad),
                         (WithSolventTDSCFGradient, td_grad.__class__), name)


def make_tdscf_nac_object(td_base_method):
    '''For td_method in vacuum, add td of solvent pcmobj'''
    # The nuclear gradients of stable exited states should correspond to a
    # fully relaxed solvent. Strictly, the TDDFT exited states should be
    # solved using state-specific solvent model. Even if running LR-PCM for
    # the zeroth order TDDFT, the wavefunction should be comptued using the
    # same dielectric constant as the ground state (the zero-frequency eps).
    with_solvent = td_base_method.with_solvent
    if not with_solvent.equilibrium_solvation:
        raise RuntimeError(
            'When computing derivative couplings of PCM-TDDFT, equilibrium solvation should '
            'be employed. The PCM TDDFT should be initialized as\n'
            '    mf.TDDFT(equilibrium_solvation=True)')
    td_nac = td_base_method.undo_solvent().nac_method()
    td_nac.base = td_base_method
    name = with_solvent.__class__.__name__ + td_nac.__class__.__name__
    return lib.set_class(WithSolventTDSCFNacMethod(td_nac),
                         (WithSolventTDSCFNacMethod, td_nac.__class__), name)


class WithSolventTDSCF:
    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'with_solvent'}

    def __init__(self, tda_method, with_solvent):
        self.__dict__.update(tda_method.__dict__)
        self.with_solvent = with_solvent

    def reset(self, mol=None):
        self.with_solvent.reset(mol)
        return super().reset(mol)

    def gen_response(self, *args, **kwargs):
        # The contribution of the solvent to an excited state include the fast
        # and the slow response parts. In the process of fast vertical excitation,
        # only the fast part is able to respond to changes of the solute
        # wavefunction. This process is described by the non-equilibrium
        # solvation. In the excited Hamiltonian, the potential from the slow part is
        # omitted. Changes of the solute electron density would lead to a
        # redistribution of the surface charge (due to the fast part).
        # The redistributed surface charge is computed by solving
        #     K^{-1} R (dm_response)
        # using a different dielectric constant. The optical dielectric constant
        # (eps=1.78, see QChem manual) is a suitable choice for the excited state.
        #
        # In the case of excited state gradients, it is mostly used in the
        # geometry optimization or molecular dynamics. The excited state is
        # obtained from the adiabatic excitation. State-specific PCM is a more
        # accurate description for the solvent. When using LR-PCM, the
        # zero-frequency dielectric constant should be used.
        mol = self.mol
        if not self.with_solvent.equilibrium_solvation:
            # Solvent with optical dielectric constant, for evaluating the
            # response of the fast solvent part
            with_solvent = self.with_solvent
            logger.info(mol, 'TDDFT non-equilibrium solvation with eps=%g', with_solvent.eps)
        else:
            # Solvent with zero-frequency dielectric constant. The ground state
            # solvent is utilized to ensure the same eps are used in the
            # gradients of excited state.
            with_solvent = self._scf.with_solvent
            logger.info(mol, 'TDDFT equilibrium solvation with eps=%g', with_solvent.eps)

        # vind computes the response in gas-phase
        vind = self._scf.undo_solvent().gen_response(
            *args, with_nlc=not self.exclude_nlc, **kwargs)

        is_uhf = isinstance(self._scf, scf.uhf.UHF)
        singlet = kwargs.get('singlet', True)
        singlet = singlet or singlet is None
        def vind_with_solvent(dm1):
            v = vind(dm1)
            if is_uhf:
                v_solvent = with_solvent._B_dot_x(dm1[0]+dm1[1])
                v += v_solvent
            elif singlet:
                v_solvent = with_solvent._B_dot_x(dm1)
                v += v_solvent
            else:
                logger.warn(mol, 'Singlet-Triplet excitation has no LR-PCM contribution!')
            return v
        return vind_with_solvent

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, WithSolventTDSCF, name_mixin))
        return obj

    nuc_grad_method = make_tdscf_gradient_object
    Gradients = nuc_grad_method
    nac_method = make_tdscf_nac_object


class WithSolventTDSCFGradient:
    from gpu4pyscf.lib.utils import to_gpu, device

    def __init__(self, tda_grad_method):
        self.__dict__.update(tda_grad_method.__dict__)

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.base.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, WithSolventTDSCFGradient, name_mixin))
        del obj.with_solvent
        return obj

    def solvent_response(self, dm):
        return self.base.with_solvent._B_dot_x(dm)*2.0

    def grad_elec(self, xy, singlet=None, atmlst=None, verbose=logger.INFO,
                  with_solvent=True):
        if self.base.with_solvent.frozen:
            raise RuntimeError('Frozen solvent model is not supported')

        # self._dmz1doo and self._dmxpy are initialized in super().grad_elec
        de = super().grad_elec(xy, singlet, atmlst, verbose, with_solvent=True)

        dm = self.base._scf.make_rdm1(ao_repr=True)
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        dmP = 0.5 * (self._dmz1doo + self._dmz1doo.T)
        dmxpy = self._dmxpy + self._dmxpy.T
        pcmobj = self.base.with_solvent
        de += pcmobj.grad(dm)

        q_sym_dm = pcmobj._get_qsym(dm, with_nuc = True)[0]
        qE_sym_dmP = pcmobj._get_qsym(dmP)[0]
        qE_sym_dmxpy = pcmobj._get_qsym(dmxpy)[0]
        de += grad_qv(pcmobj, dm, q_sym = qE_sym_dmP)
        de += grad_nuc(pcmobj, dm, q_sym = qE_sym_dmP.get())
        de += grad_qv(pcmobj, dmP, q_sym = q_sym_dm)
        v_grids_l = pcmobj._get_vgrids(dmP, with_nuc = False)[0]
        de += grad_solver(pcmobj, dm, v_grids_l = v_grids_l) * 2.0
        de += grad_qv(pcmobj, dmxpy, q_sym = qE_sym_dmxpy) * 2.0
        v_grids = pcmobj._get_vgrids(dmxpy, with_nuc = False)[0]
        q = pcmobj._get_qsym(dmxpy, with_nuc = False)[1]
        de += grad_solver(pcmobj, dmxpy, v_grids=v_grids, v_grids_l=v_grids, q=q) * 2.0
        
        return de


class WithSolventTDSCFNacMethod:
    from gpu4pyscf.lib.utils import to_gpu, device

    def __init__(self, tda_grad_method):
        self.__dict__.update(tda_grad_method.__dict__)

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.base.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, WithSolventTDSCFNacMethod, name_mixin))
        del obj.with_solvent
        return obj

    def solvent_response(self, dm):
        return self.base.with_solvent._B_dot_x(dm)*2.0 
        
    def get_nacv_ge(self, xy, EI, singlet=None, atmlst=None, verbose=logger.INFO):
        if self.base.with_solvent.frozen:
            raise RuntimeError('Frozen solvent model is not supported')

        de_tuple = super().get_nacv_ge(xy, EI, singlet, atmlst, verbose) 
        de, de_scaled, de_etf, de_etf_scaled = de_tuple

        dm = self.base._scf.make_rdm1(ao_repr=True)
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        dmP = self._dmz1doo  #1.0 * (self._dmz1doo + self._dmz1doo.T)
        pcmobj = self.base.with_solvent
        assert pcmobj.equilibrium_solvation

        de_modify = 0
        q_sym_dm = pcmobj._get_qsym(dm, with_nuc = True)[0]
        qE_sym_dmP = pcmobj._get_qsym(dmP)[0]
        de_modify += grad_qv(pcmobj, dm, q_sym = qE_sym_dmP)
        de_modify += grad_nuc(pcmobj, dm, q_sym = qE_sym_dmP.get())
        de_modify += grad_qv(pcmobj, dmP, q_sym = q_sym_dm)
        v_grids_l = pcmobj._get_vgrids(dmP, with_nuc = False)[0]
        de_modify += grad_solver(pcmobj, dm, v_grids_l = v_grids_l) * 2.0

        de += de_modify
        de_scaled = de/EI
        de_etf += de_modify
        de_etf_scaled = de_etf/EI
        
        return de, de_scaled, de_etf, de_etf_scaled

    def get_nacv_ee(self, x_yI, x_yJ, EI, EJ, singlet=None, atmlst=None, verbose=logger.INFO):
        if self.base.with_solvent.frozen:
            raise RuntimeError('Frozen solvent model is not supported')

        de_tuple = super().get_nacv_ee(x_yI, x_yJ, EI, EJ, singlet, atmlst, verbose) 
        de, de_scaled, de_etf, de_etf_scaled = de_tuple

        dm = self.base._scf.make_rdm1(ao_repr=True)
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        dmP = 0.5 * (self._dmz1doo + self._dmz1doo.T)
        dmxpyI = self._dmxpyI + self._dmxpyI.T
        dmxpyJ = self._dmxpyJ + self._dmxpyJ.T
        pcmobj = self.base.with_solvent
        assert pcmobj.equilibrium_solvation

        de_modify = 0.0
        q_sym_dm = pcmobj._get_qsym(dm, with_nuc = True)[0]
        qE_sym_dmP = pcmobj._get_qsym(dmP)[0]
        qE_sym_dmxpyI = pcmobj._get_qsym(dmxpyI)[0]
        qE_sym_dmxpyJ = pcmobj._get_qsym(dmxpyJ)[0]
        de_modify += grad_qv(pcmobj, dm, q_sym = qE_sym_dmP)
        de_modify += grad_nuc(pcmobj, dm, q_sym = qE_sym_dmP.get())
        de_modify += grad_qv(pcmobj, dmP, q_sym = q_sym_dm)
        v_grids_l = pcmobj._get_vgrids(dmP, with_nuc = False)[0]
        de_modify += grad_solver(pcmobj, dm, v_grids_l = v_grids_l) * 2.0

        de_modify += grad_qv(pcmobj, dmxpyJ, q_sym = qE_sym_dmxpyI)
        de_modify += grad_qv(pcmobj, dmxpyI, q_sym = qE_sym_dmxpyJ)

        v_gridsJ = pcmobj._get_vgrids(dmxpyJ, with_nuc = False)[0]
        v_gridsI = pcmobj._get_vgrids(dmxpyI, with_nuc = False)[0]
        qI = pcmobj._get_qsym(dmxpyI, with_nuc = False)[1]
        qJ = pcmobj._get_qsym(dmxpyJ, with_nuc = False)[1]
        de_modify += grad_solver(pcmobj, dmxpyJ, v_grids=v_gridsI, v_grids_l=v_gridsJ, q=qI)
        de_modify += grad_solver(pcmobj, dmxpyI, v_grids=v_gridsJ, v_grids_l=v_gridsI, q=qJ)

        de = de + de_modify
        de_scaled = de/(EJ-EI)
        de_etf = de_etf + de_modify
        de_etf_scaled = de_etf/(EJ-EI)
        
        return de, de_scaled, de_etf, de_etf_scaled

def from_cpu(method):
    from pyscf.solvent.tdscf import pcm as pcm_cpu
    if isinstance(method, pcm_cpu.WithSolventTDSCF):
        return make_tdscf_object(method.undo_solvent().to_gpu(),
                                 equilibrium_solvation=method.equilibrium_solvation)
    elif isinstance(method, pcm_cpu.WithSolventTDSCFGradient):
        return make_tdscf_gradient_object(method.base.to_gpu())
    elif isinstance(method, pcm_cpu.WithSolventTDSCFNacMethod):
        return make_tdscf_nac_object(method.base.to_gpu())
    else:
        raise RuntimeError(f'{method} must be a PCM-TDDFT instance')
