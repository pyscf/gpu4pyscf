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

import numpy, cupy
from pyscf import lib
from pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import tag_array, pack_tril
from gpu4pyscf import scf
from gpu4pyscf.scf.hf_lowmem import WaveFunction

def _for_scf(mf, solvent_obj, dm=None):
    '''Add solvent model to SCF (HF and DFT) method.

    Kwargs:
        dm : if given, solvent does not respond to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''
    if isinstance(mf, _Solvation):
        mf.with_solvent = solvent_obj
        return mf

    if dm is not None:
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    sol_mf = SCFWithSolvent(mf, solvent_obj)
    name = solvent_obj.__class__.__name__ + mf.__class__.__name__
    return lib.set_class(sol_mf, (SCFWithSolvent, mf.__class__), name)

# 1. A tag to label the derived method class
class _Solvation:
    pass

class SCFWithSolvent(_Solvation):
    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'with_solvent'}

    def __init__(self, mf, solvent):
        self.__dict__.update(mf.__dict__)
        self.with_solvent = solvent

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, SCFWithSolvent, name_mixin))
        del obj.with_solvent
        return obj

    def to_cpu(self):
        from pyscf.solvent import _attach_solvent
        solvent_obj = self.with_solvent.to_cpu()
        obj = _attach_solvent._for_scf(self.undo_solvent().to_cpu(), solvent_obj)
        return obj

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        self.with_solvent.check_sanity()
        self.with_solvent.dump_flags(verbose)
        return self

    def reset(self, mol=None):
        self.with_solvent.reset(mol)
        return super().reset(mol)

    def get_veff(self, mol=None, dm_or_wfn=None, *args, **kwargs):
        veff = super().get_veff(mol, dm_or_wfn, *args, **kwargs)
        with_solvent = self.with_solvent
        if not with_solvent.frozen:
            if dm_or_wfn is None:
                dm = self.make_rdm1()
            elif isinstance(dm_or_wfn, WaveFunction):
                dm = dm_or_wfn.make_rdm1()
            else:
                dm = dm_or_wfn
            with_solvent.e, with_solvent.v = with_solvent.kernel(dm)
        e_solvent, v_solvent = with_solvent.e, with_solvent.v
        if veff.shape[-1] != v_solvent.shape[-1]:
            # lowmem mode, only lower triangular part of Fock matrix is stored
            nao = v_solvent.shape[-1]
            assert not isinstance(veff, cupy.ndarray)
            assert veff.ndim == 1 and veff.shape[0] == nao * (nao + 1) // 2
            v_solvent = pack_tril(v_solvent).get()
            veff = lib.tag_array(veff, e_solvent=e_solvent, v_solvent=v_solvent)
        else:
            veff = tag_array(veff, e_solvent=e_solvent, v_solvent=v_solvent)
        return veff

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm_or_wfn=None, cycle=-1,
                 diis=None, diis_start_cycle=None,
                 level_shift_factor=None, damp_factor=None, fock_last=None):
        # DIIS was called inside oldMF.get_fock. v_solvent, as a function of
        # dm, should be extrapolated as well. To enable it, v_solvent has to be
        # added to the fock matrix before DIIS was called.
        if getattr(vhf, 'v_solvent', None) is None:
            vhf = self.get_veff(self.mol, dm_or_wfn)
        return super().get_fock(h1e, s1e, vhf+vhf.v_solvent, dm_or_wfn, cycle, diis,
                                diis_start_cycle, level_shift_factor, damp_factor)

    def energy_elec(self, dm_or_wfn=None, h1e=None, vhf=None):
        if getattr(vhf, 'e_solvent', None) is None:
            vhf = self.get_veff(self.mol, dm_or_wfn)
        e_tot, e_coul = super().energy_elec(dm_or_wfn, h1e, vhf)
        e_solvent = vhf.e_solvent
        if isinstance(e_solvent, cupy.ndarray):
            e_solvent = e_solvent.get()[()]
        e_tot += e_solvent
        self.scf_summary['e_solvent'] = e_solvent

        if (hasattr(self.with_solvent, 'method') and self.with_solvent.method.upper() == 'SMD'):
            if self.with_solvent.e_cds is None:
                e_cds = self.with_solvent.get_cds()
                self.with_solvent.e_cds = e_cds
            else:
                e_cds = self.with_solvent.e_cds
            if isinstance(e_cds, cupy.ndarray):
                e_cds = e_cds.get()[()]
            e_tot += e_cds
            self.scf_summary['e_cds'] = e_cds
            logger.info(self, f'CDS correction = {e_cds:.15f}')
        logger.info(self, 'Solvent Energy = %.15g', vhf.e_solvent)

        return e_tot, e_coul

    def nuc_grad_method(self):
        # TODO: merge the two make_grad_object functions into a general one
        from gpu4pyscf.solvent.pcm import PCM
        if isinstance(self.with_solvent, PCM):
            from gpu4pyscf.solvent.grad.pcm import make_grad_object
        else:
            from gpu4pyscf.solvent.grad.smd import make_grad_object
        return make_grad_object(self)

    Gradients = nuc_grad_method

    def Hessian(self):
        from gpu4pyscf.solvent.pcm import PCM
        if isinstance(self.with_solvent, PCM):
            from gpu4pyscf.solvent.hessian.pcm import make_hess_object
        else:
            from gpu4pyscf.solvent.hessian.smd import make_hess_object
        return make_hess_object(self)

    def TDA(self, equilibrium_solvation=False, **kwargs):
        td = super().TDA()
        from gpu4pyscf.solvent.tdscf import pcm as pcm_td
        return pcm_td.make_tdscf_object(td, equilibrium_solvation=equilibrium_solvation)

    def TDDFT(self, equilibrium_solvation=False, **kwargs):
        td = super().TDDFT()
        from gpu4pyscf.solvent.tdscf import pcm as pcm_td
        return pcm_td.make_tdscf_object(td, equilibrium_solvation=equilibrium_solvation)

    def TDHF(self, equilibrium_solvation=False, **kwargs):
        td = super().TDHF()
        from gpu4pyscf.solvent.tdscf import pcm as pcm_td
        return pcm_td.make_tdscf_object(td, equilibrium_solvation=equilibrium_solvation)

    def CasidaTDDFT(self, equilibrium_solvation=False, **kwargs):
        td = super().CasidaTDDFT()
        from gpu4pyscf.solvent.tdscf import pcm as pcm_td
        return pcm_td.make_tdscf_object(td, equilibrium_solvation=equilibrium_solvation)

    def gen_response(self, *args, **kwargs):
        vind = self.undo_solvent().gen_response(*args, **kwargs)
        is_uhf = isinstance(self, scf.uhf.UHF)
        def vind_with_solvent(dm1):
            v = vind(dm1)
            if self.with_solvent.equilibrium_solvation:
                if is_uhf:
                    v += self.with_solvent._B_dot_x(dm1[0]+dm1[1])
                else:
                    v += self.with_solvent._B_dot_x(dm1)
            return v
        return vind_with_solvent

    def stability(self, *args, **kwargs):
        # When computing orbital hessian, the second order derivatives of
        # solvent energy needs to be computed. It is enabled by
        # the attribute equilibrium_solvation in gen_response method.
        # If solvent was frozen, its contribution is treated as the
        # external potential. The response of solvent does not need to
        # be considered in stability analysis.
        with lib.temporary_env(self.with_solvent,
                                equilibrium_solvation=not self.with_solvent.frozen):
            return super().stability(*args, **kwargs)
