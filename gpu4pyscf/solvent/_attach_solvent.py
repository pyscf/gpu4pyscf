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

import cupy
from pyscf import lib
from pyscf.lib import logger
from pyscf.solvent._attach_solvent import _Solvation
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf import scf

# NOTE: copied from pyscf, different from the latest version

def _for_scf(mf, solvent_obj, dm=None):
    '''Add solvent model to SCF (HF and DFT) method.

    Kwargs:
        dm : if given, solvent does not respond to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''
    if isinstance(mf, _Solvation):
        mf.with_solvent = solvent_obj
        return mf

    oldMF = mf.__class__

    if dm is not None:
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    class SCFWithSolvent(_Solvation, oldMF):
        def __init__(self, mf, solvent):
            self.__dict__.update(mf.__dict__)
            self.with_solvent = solvent
            self._keys.update(['with_solvent'])

        def dump_flags(self, verbose=None):
            oldMF.dump_flags(self, verbose)
            self.with_solvent.check_sanity()
            self.with_solvent.dump_flags(verbose)
            return self

        def reset(self, mol=None):
            self.with_solvent.reset(mol)
            return oldMF.reset(self, mol)

        # Note v_solvent should not be added to get_hcore for scf methods.
        # get_hcore is overloaded by many post-HF methods. Modifying
        # SCF.get_hcore may lead error.

        def get_veff(self, mol=None, dm=None, *args, **kwargs):
            vhf = oldMF.get_veff(self, mol, dm, *args, **kwargs)
            with_solvent = self.with_solvent
            if not with_solvent.frozen:
                with_solvent.e, with_solvent.v = with_solvent.kernel(dm)
            e_solvent, v_solvent = with_solvent.e, with_solvent.v

            # NOTE: v_solvent should not be added to vhf in this place. This is
            # because vhf is used as the reference for direct_scf in the next
            # iteration. If v_solvent is added here, it may break direct SCF.
            return tag_array(vhf, e_solvent=e_solvent, v_solvent=v_solvent)

        def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1,
                     diis=None, diis_start_cycle=None,
                     level_shift_factor=None, damp_factor=None):
            # DIIS was called inside oldMF.get_fock. v_solvent, as a function of
            # dm, should be extrapolated as well. To enable it, v_solvent has to be
            # added to the fock matrix before DIIS was called.
            if getattr(vhf, 'v_solvent', None) is None:
                vhf = self.get_veff(self.mol, dm)
            return oldMF.get_fock(self, h1e, s1e, vhf+vhf.v_solvent, dm, cycle, diis,
                                  diis_start_cycle, level_shift_factor, damp_factor)

        def energy_elec(self, dm=None, h1e=None, vhf=None):
            if dm is None:
                dm = self.make_rdm1()
            if getattr(vhf, 'e_solvent', None) is None:
                vhf = self.get_veff(self.mol, dm)

            e_tot, e_coul = oldMF.energy_elec(self, dm, h1e, vhf)
            e_solvent = vhf.e_solvent
            if isinstance(e_solvent, cupy.ndarray):
                e_solvent = e_solvent.get()[()]
            e_tot += e_solvent
            self.scf_summary['e_solvent'] = vhf.e_solvent.real

            if self.with_solvent.method.upper() == 'SMD':
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
            grad_method = oldMF.nuc_grad_method(self)
            return self.with_solvent.nuc_grad_method(grad_method)

        Gradients = nuc_grad_method

        def Hessian(self):
            hess_method = oldMF.Hessian(self)
            return self.with_solvent.Hessian(hess_method)

        def gen_response(self, *args, **kwargs):
            vind = oldMF.gen_response(self, *args, **kwargs)
            is_uhf = isinstance(self, scf.uhf.UHF)
            # singlet=None is orbital hessian or CPHF type response function
            singlet = kwargs.get('singlet', True)
            singlet = singlet or singlet is None
            def vind_with_solvent(dm1):
                v = vind(dm1)
                if self.with_solvent.equilibrium_solvation:
                    if is_uhf:
                        v_solvent = self.with_solvent._B_dot_x(dm1)
                        v += v_solvent[0] + v_solvent[1]
                    elif singlet:
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
                return oldMF.stability(self, *args, **kwargs)

    mf1 = SCFWithSolvent(mf, solvent_obj)
    return mf1