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

import numpy as np
import cupy as cp
import scipy
from pyscf import __config__, lib
from pyscf.scf import addons as cpu_addons
from pyscf.scf.addons import (_fermi_smearing_occ, _gaussian_smearing_occ,
                              _get_fermi, _smearing_optimize)
from pyscf.pbc.tools import print_mo_energy_occ
from gpu4pyscf.lib import logger

SMEARING_METHOD = getattr(__config__, "pbc_scf_addons_smearing_method", "fermi")

def _get_grad_tril(mo_coeff, mo_occ, fock):
    f_mo = mo_coeff.conj().T.dot(fock).dot(mo_coeff)
    return f_mo[cp.tril_indices_from(f_mo, -1)]

def smearing(mf, sigma=None, method=SMEARING_METHOD, mu0=None, fix_spin=False):
    """Fermi-Dirac or Gaussian smearing"""
    if isinstance(mf, _SmearingSCF):
        mf.sigma = sigma
        mf.method = method
        mf.mu0 = mu0
        mf.fix_spin = fix_spin
        return mf

    assert not mf.istype("KSCF")

    # Commenting out the complication of checking the mean-field object
    # To make linter happy.
    # if mf.istype("ROHF"):
    # Roothaan Fock matrix does not make much sense for smearing.
    # Restore the conventional RHF treatment.
    #    from pyscf import dft, scf

    # known_class = {
    #     dft.rks_symm.ROKS: dft.rks_symm.RKS,
    #     dft.roks.ROKS: dft.rks.RKS,
    #     scf.hf_symm.ROHF: scf.hf_symm.RHF,
    #     scf.rohf.ROHF: scf.hf.RHF,
    # }
    return lib.set_class(
        _SmearingSCF(mf, sigma, method, mu0, fix_spin), (_SmearingSCF, mf.__class__)
    )


def smearing_(mf, *args, **kwargs):
    mf1 = smearing(mf, *args, **kwargs)
    mf.__class__ = mf1.__class__
    mf.__dict__ = mf1.__dict__
    return mf


class _SmearingSCF:

    __name_mixin__ = "Smearing"

    _keys = {
        "sigma",
        "smearing_method",
        "mu0",
        "fix_spin",
        "entropy",
        "e_free",
        "e_zero",
    }

    def __init__(self, mf, sigma, method, mu0, fix_spin):
        self.__dict__.update(mf.__dict__)
        self.sigma = sigma
        self.smearing_method = method
        self.mu0 = mu0
        self.fix_spin = fix_spin
        self.entropy = None
        self.e_free = None
        self.e_zero = None

    def undo_smearing(self):
        obj = lib.view(self, lib.drop_class(self.__class__, _SmearingSCF))
        del obj.sigma
        del obj.smearing_method
        del obj.fix_spin
        del obj.entropy
        del obj.e_free
        del obj.e_zero
        return obj

    def get_occ(self, mo_energy=None, mo_coeff=None):
        """Label the occupancies for each orbital"""
        from gpu4pyscf import scf
        if (self.sigma == 0) or (not self.sigma) or (not self.smearing_method):
            mo_occ = super().get_occ(mo_energy, mo_coeff)
            return mo_occ

        is_uhf = self.istype("UHF")
        is_rhf = self.istype("RHF")
        if isinstance(self, scf.rohf.ROHF):
            # ROHF leads to two Fock matrices. It's not clear how to define the
            # Roothaan effective Fock matrix from the two.
            raise NotImplementedError("Smearing-ROHF")

        sigma = self.sigma
        if self.smearing_method.lower() == "fermi":
            f_occ = _fermi_smearing_occ
        else:
            f_occ = _gaussian_smearing_occ

        mo_energy = mo_energy.get()
        if self.fix_spin and is_uhf:  # spin separated fermi level
            mo_es = mo_energy
            nocc = self.nelec
            if self.mu0 is None:
                mu_a, occa = _smearing_optimize(f_occ, mo_es[0], nocc[0], sigma)
                mu_b, occb = _smearing_optimize(f_occ, mo_es[1], nocc[1], sigma)
                mu_a = mu_a[0]
                mu_b = mu_b[0]
            else:
                if np.isscalar(self.mu0):
                    mu_a = mu_b = self.mu0
                elif len(self.mu0) == 2:
                    mu_a, mu_b = self.mu0
                else:
                    raise TypeError(f"Unsupported mu0: {self.mu0}")
                occa = f_occ(mu_a, mo_es[0], sigma)
                occb = f_occ(mu_b, mo_es[1], sigma)
            mu = [mu_a, mu_b]
            mo_occs = [occa, occb]
            self.entropy = self._get_entropy(mo_es[0], mo_occs[0], mu[0])
            self.entropy += self._get_entropy(mo_es[1], mo_occs[1], mu[1])
            if self.verbose >= logger.INFO:
                fermi = (_get_fermi(mo_es[0], nocc[0]), _get_fermi(mo_es[1], nocc[1]))
                logger.debug(
                    self,
                    "    Alpha-spin Fermi level %g  Sum mo_occ = %s  should equal nelec = %s",
                    fermi[0],
                    mo_occs[0].sum(),
                    nocc[0],
                )
                logger.debug(
                    self,
                    "    Beta-spin  Fermi level %g  Sum mo_occ = %s  should equal nelec = %s",
                    fermi[1],
                    mo_occs[1].sum(),
                    nocc[1],
                )
                logger.info(
                    self,
                    "    sigma = %g  Optimized mu_alpha = %.12g  entropy = %.12g",
                    sigma,
                    mu[0],
                    self.entropy,
                )
                logger.info(
                    self,
                    "    sigma = %g  Optimized mu_beta  = %.12g  entropy = %.12g",
                    sigma,
                    mu[1],
                    self.entropy,
                )
            if self.verbose >= logger.DEBUG:
                print_mo_energy_occ(self, mo_energy, mo_occs, True)
        else:  # all orbitals treated with the same fermi level
            nelectron = self.mol.nelectron
            mo_es = mo_energy.ravel()
            if is_rhf:
                nelectron = nelectron / 2

            if self.mu0 is None:
                mu, mo_occs = _smearing_optimize(f_occ, mo_es, nelectron, sigma)
                mu = mu[0]
            else:
                # If mu0 is given, fix mu instead of electron number. XXX -Chong Sun
                mu = self.mu0
                assert np.isscalar(mu)
                mo_occs = f_occ(mu, mo_es, sigma)
            self.entropy = self._get_entropy(mo_es, mo_occs, mu)
            if is_rhf:
                mo_occs *= 2
                self.entropy *= 2

            if self.verbose >= logger.INFO:
                fermi = _get_fermi(mo_es, nelectron)
                logger.debug(
                    self,
                    "    Fermi level %g  Sum mo_occ = %s  should equal nelec = %s",
                    fermi,
                    mo_occs.sum(),
                    nelectron,
                )
                logger.info(
                    self,
                    "    sigma = %g  Optimized mu = %.12g  entropy = %.12g",
                    sigma,
                    mu,
                    self.entropy,
                )
            if is_uhf:
                mo_occs = mo_occs.reshape(2, -1)
            if self.verbose >= logger.DEBUG:
                print_mo_energy_occ(self, mo_energy, mo_occs, is_uhf)
        return cp.asarray(mo_occs)

    _get_entropy = cpu_addons._SmearingSCF._get_entropy

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        if (self.sigma == 0) or (not self.sigma) or (not self.smearing_method):
            return super().get_grad(mo_coeff, mo_occ, fock)

        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore() + self.get_veff(self.mol, dm1)
        if self.istype("UHF"):
            ga = _get_grad_tril(mo_coeff[0], mo_occ[0], fock[0])
            gb = _get_grad_tril(mo_coeff[1], mo_occ[1], fock[1])
            return cp.hstack((ga, gb))
        else:  # rhf and ghf
            return _get_grad_tril(mo_coeff, mo_occ, fock)

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        e_tot = self.energy_elec(dm, h1e, vhf)[0] + self.energy_nuc()
        if self.sigma and self.smearing_method and self.entropy is not None:
            self.e_free = e_tot - self.sigma * self.entropy
            self.e_zero = e_tot - self.sigma * self.entropy * 0.5
            logger.info(
                self,
                "    Total E(T) = %.15g  Free energy = %.15g  E0 = %.15g",
                e_tot,
                self.e_free,
                self.e_zero,
            )
        return e_tot

    def to_cpu(self):
        from pyscf.scf.addons import smearing
        return smearing(self.undo_smearing().to_cpu(), self.sigma,
                        self.smearing_method, self.mu0, self.fix_spin)

def from_cpu(method):
    from pyscf.scf.addons import _SmearingSCF
    assert isinstance(method, _SmearingSCF)
    return smearing(method.undo_smearing().to_cpu(), method.sigma,
                    method.smearing_method, method.mu0, method.fix_spin)
