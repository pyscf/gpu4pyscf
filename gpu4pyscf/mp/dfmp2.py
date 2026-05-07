# Copyright 2026 The PySCF Developers. All Rights Reserved.
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


import pyscf
import cupy as cp
import numpy as np

import pyscf.df.addons

from pyscf import __config__
from gpu4pyscf.lib import logger
from gpu4pyscf.df import DF
from gpu4pyscf.mp import dfmp2_addons, dfmp2_drivers
from gpu4pyscf.mp.mp2 import MP2Base


def kernel(
    mp,
    mo_energy=None,
    mo_coeff=None,
    mo_occ=None,
    frozen_mask=None,
    with_t2=None,
    j2c_decomp_alg=None,
    j3c_backend=None,
    fp_type=None,
    verbose=None,
):
    mol = mp.mol
    aux = mp.auxmol
    log = logger.new_logger(mol, verbose)

    # handle default values for parameters
    frozen_mask = frozen_mask if frozen_mask is not None else mp.get_frozen_mask()
    with_t2 = with_t2 if with_t2 is not None else mp.with_t2
    j2c_decomp_alg = j2c_decomp_alg if j2c_decomp_alg is not None else mp.j2c_decomp_alg
    j3c_backend = j3c_backend if j3c_backend is not None else mp.j3c_backend
    fp_type = fp_type if fp_type is not None else mp.fp_type

    if fp_type is np.float32:
        fp_type = 'FP32'
    elif fp_type is np.float64:
        fp_type = 'FP64'
    assert fp_type in ['FP64', 'FP32']
    dtype_cderi = np.float64 if fp_type == 'FP64' else np.float32

    # obtain necessary arguments
    [_, occ_coeff, vir_coeff, _] = dfmp2_addons.split_mo_coeff_restricted(mp, mo_coeff=mo_coeff, frozen_mask=frozen_mask, mo_occ=mo_occ)
    [_, occ_energy, vir_energy, _] = dfmp2_addons.split_mo_energy_restricted(mp, mo_energy=mo_energy, frozen_mask=frozen_mask, mo_occ=mo_occ)

    # allocate t2 if needed
    if with_t2 is True:
        if mp.t2 is None:
            nocc = mol.nelectron // 2
            nmo = mo_energy.size
            nvir = nmo - nocc
            mp.t2 = cp.empty((nocc, nocc, nvir, nvir), dtype=dtype_cderi)
    else:
        mp.t2 = None

    # run driver
    result = dfmp2_drivers.dfmp2_kernel_one_gpu(
        mol,
        aux,
        occ_coeff,
        vir_coeff,
        occ_energy,
        vir_energy,
        j3c_backend=j3c_backend,
        j2c_decomp_alg=j2c_decomp_alg,
        t2=mp.t2,
        dtype_cderi=dtype_cderi,
        log=log,
    )

    # handle results
    e_corr_os = result['e_corr_os']
    e_corr_ss = result['e_corr_ss']
    e_corr = e_corr_os + e_corr_ss

    mp.e_corr_os = e_corr_os
    mp.e_corr_ss = e_corr_ss
    mp.e_corr = e_corr
    return mp.e_corr, mp.t2


class DFMP2(MP2Base):
    """Density-fitted (resolution-of-identity) MP2 implementation with GPU acceleration.

    Attributes
    ----------
    mo_energy : numpy.ndarray
        Molecular orbital energies.
    auxmol : pyscf.gto.Mole
        Auxiliary basis molecule used for density fitting.
        If not provided, it will be generated automatically based on the main molecule and the `mp2fit` auxiliary basis set.

    j3c_backend : str
        Configurable option.
        Backend to use for three-center integral computations.
        - 'bdiv': block-divergent generator (default).
        - 'vhfopt': VHFOpt generator.
    with_t2 : bool
        Configurable option.
        Whether to compute and store T2 amplitudes.
        Default is False.
    fp_type : str
        Configurable option.
        Floating-point precision to use for integral computations and T2 amplitudes.
        FP32 should be sufficiently accurate for MP2 energies if full FP32 precision is used; but using TF32 may cause loss of accuracy in some cases.
        Please check bash environment variable ``CUPY_TF32`` when using 'FP32' option.
        - 'FP64': double precision (default).
        - 'FP32': single precision.
    j2c_decomp_alg : str
        Configurable option.
        Algorithm to use for the decomposition of the two-center Coulomb matrix in density fitting.
        The decomposed matrices are used in for transformation of auxiliary index in three-center integrals.
        Cholesky decomposition is more efficient and costs less memory.
        - 'cd': Cholesky decomposition (default).
        - 'eig': eigenvalue decomposition.
    """

    mo_energy = None
    auxmol = None
    with_df = None  # placeholder for API compatibility (especially `reset` method)

    j3c_backend = dfmp2_addons.CONFIG_J3C_BACKEND
    with_t2 = dfmp2_addons.CONFIG_WITH_T2
    fp_type = dfmp2_addons.CONFIG_FP_TYPE
    j2c_decomp_alg = dfmp2_addons.CONFIG_J2C_DECOMP_ALG
    force_outcore = False  # placeholder for API compatibility (pyscf.mp.dfmp2.DFMP2 attribute)

    _keys = {
        'mo_energy',
        'auxmol',
        'with_df',
        'j3c_backend',
        'with_t2',
        'fp_type',
        'j2c_decomp_alg',
        'force_outcore',
    }

    _kernel_impl = kernel

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, auxbasis=None):
        # initialize attributes in _keys to be instance instead of class attributes (__dict__ available)
        for key in self._keys:
            setattr(self, key, getattr(self, key, None))

        super().__init__(mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)

        self.mo_energy = mf.mo_energy
        self._make_auxmol(auxbasis=auxbasis)
            
        self.with_df = DF(mf.mol, auxbasis=self.auxmol.basis)  # dummy placeholder

    def _make_auxmol(self, auxbasis=None):
        if auxbasis is not None:
            if isinstance(auxbasis, pyscf.gto.Mole):
                self.auxmol = auxbasis
            else:
                self.auxmol = pyscf.df.addons.make_auxmol(self.mol, auxbasis)
        else:
            auxbasis = pyscf.df.addons.make_auxbasis(self.mol, mp2fit=True)
            self.auxmol = pyscf.df.addons.make_auxmol(self.mol, auxbasis)

    def kernel(self, *args, **kwargs):
        kwargs.setdefault('mo_coeff', self.mo_coeff)
        kwargs.setdefault('mo_occ', self.mo_occ)
        kwargs.setdefault('mo_energy', self.mo_energy)
        kwargs.setdefault('with_t2', self.with_t2)
        kwargs.setdefault('j2c_decomp_alg', self.j2c_decomp_alg)
        kwargs.setdefault('fp_type', self.fp_type)

        log = logger.new_logger(self)
        t0 = t1 = log.init_timer()

        self.e_hf = self.get_e_hf(mo_coeff=kwargs['mo_coeff'])
        t1 = log.timer(f'ehf in {self.__class__.__name__}', *t1)

        if self.auxmol is None:
            self._make_auxmol()

        if self._scf.converged:
            self._kernel_impl(*args, **kwargs)
        else:
            raise RuntimeError('SCF not converged. Current implementation does not support iterative MP2.')

        log.timer(self.__class__.__name__, *t0)
        self._finalize()
        return self.e_corr, self.t2


DFRMP2 = DFMP2
