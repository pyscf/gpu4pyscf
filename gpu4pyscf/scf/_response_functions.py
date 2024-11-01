# Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
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
from gpu4pyscf.lib import logger
from gpu4pyscf.scf import hf, uhf

def _gen_rhf_response(mf, mo_coeff=None, mo_occ=None,
                      singlet=None, hermi=0, max_memory=None):
    '''Generate a function to compute the product of RHF response function and
    RHF density matrices.

    Kwargs:
        singlet (None or boolean) : If singlet is None, response function for
            orbital hessian or CPHF will be generated. If singlet is boolean,
            it is used in TDDFT response kernel.
    '''
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol
    if isinstance(mf, hf.KohnShamDFT):
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if getattr(mf, 'nlc', '') != '':
            logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                        'deriviative is not available. Its contribution is '
                        'not included in the response function.')
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = abs(hyb) > 1e-10

        if singlet is None:
            # for ground state orbital hessian
            rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc,
                                                mo_coeff, mo_occ, 0)
        else:
            rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc,
                                                [mo_coeff]*2, [mo_occ*.5]*2, spin=1)
        dm0 = None  #mf.make_rdm1(mo_coeff, mo_occ)

        if singlet is None:
            # Without specify singlet, used in ground state orbital hessian
            def vind(dm1):
                # The singlet hessian
                if hermi == 2:
                    v1 = cupy.zeros_like(dm1)
                else:
                    v1 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                       rho0, vxc, fxc, max_memory=max_memory)
                if hybrid or abs(alpha) > 1e-10:
                    if hermi != 2:
                        vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                        vk *= hyb
                        if omega > 1e-10:  # For range separated Coulomb
                            vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                        v1 += vj - .5 * vk
                    else:
                        v1 -= .5 * hyb * mf.get_k(mol, dm1, hermi=hermi)
                elif hermi != 2:
                    v1 += mf.get_j(mol, dm1, hermi=hermi)
                return v1
        else:
            raise NotImplementedError('only singlet response is supported!')

    else:  # HF
        if (singlet is None or singlet) and hermi != 2:
            def vind(dm1):
                vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                return vj - .5 * vk
        else:
            def vind(dm1):
                return -.5 * mf.get_k(mol, dm1, hermi=hermi)

    return vind


def _gen_uhf_response(mf, mo_coeff=None, mo_occ=None,
                      with_j=True, hermi=0, max_memory=None):
    '''Generate a function to compute the product of UHF response function and
    UHF density matrices.
    '''
    assert isinstance(mf, uhf.UHF)
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol
    if isinstance(mf, hf.KohnShamDFT):
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if mf.do_nlc():
            logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                        'deriviative is not available. Its contribution is '
                        'not included in the response function.')
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)

        rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc,
                                            mo_coeff, mo_occ, 1)
        dm0 = None

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        def vind(dm1):
            if hermi == 2:
                v1 = cupy.zeros_like(dm1)
            else:
                v1 = ni.nr_uks_fxc(mol, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                   rho0, vxc, fxc, max_memory=max_memory)
            if not hybrid:
                if with_j:
                    vj = mf.get_j(mol, dm1, hermi=hermi)
                    v1 += vj[0] + vj[1]
            else:
                if with_j:
                    vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                    vk *= hyb
                    if omega > 1e-10:  # For range separated Coulomb
                        vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                    v1 += vj[0] + vj[1] - vk
                else:
                    vk = mf.get_k(mol, dm1, hermi=hermi)
                    vk *= hyb
                    if omega > 1e-10:  # For range separated Coulomb
                        vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                    v1 -= vk
            return v1

    elif with_j:
        def vind(dm1):
            vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
            v1 = vj[0] + vj[1] - vk
            return v1

    else:
        def vind(dm1):
            return -mf.get_k(mol, dm1, hermi=hermi)

    return vind

hf.RHF.gen_response = _gen_rhf_response
uhf.UHF.gen_response = _gen_uhf_response
