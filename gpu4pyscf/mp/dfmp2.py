import pyscf
import cupy as cp
import numpy as np

import pyscf.df.addons

from pyscf import __config__
from gpu4pyscf.mp import dfmp2_addons, dfmp2_drivers


def kernel(
    mp,
    mo_energy=None,
    mo_coeff=None,
    mo_occ=None,
    frozen_mask=None,
    with_t2=None,
    j2c_decomp_alg=None,
    fp_type=None,
    verbose=None,
):
    mol = mp.mol
    aux = mp.auxmol
    log = pyscf.lib.logger.new_logger(mp, verbose)

    # handle default values for parameters
    frozen_mask = frozen_mask if frozen_mask is not None else mp.get_frozen_mask()
    with_t2 = with_t2 if with_t2 is not None else mp.with_t2
    j2c_decomp_alg = j2c_decomp_alg if j2c_decomp_alg is not None else mp.j2c_decomp_alg
    fp_type = fp_type if fp_type is not None else mp.fp_type

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
    args = (mol, aux, occ_coeff, vir_coeff, occ_energy, vir_energy)
    kwargs = {
        'driver': 'bdiv',
        'j2c_decomp_alg': j2c_decomp_alg,
        't2': mp.t2,
        'dtype_cderi': dtype_cderi,
        'log': log,
    }
    result = dfmp2_drivers.dfmp2_kernel_one_gpu(*args, **kwargs)

    # handle results
    e_corr_os = result['e_corr_os']
    e_corr_ss = result['e_corr_ss']
    e_corr = e_corr_os + e_corr_ss

    mp.e_corr_os = e_corr_os
    mp.e_corr_ss = e_corr_ss
    mp.e_corr = e_corr
    return mp.e_corr


class DFMP2(pyscf.mp.mp2.MP2Base):
    mo_energy = None
    auxmol = None

    with_t2 = dfmp2_addons.CONFIG_WITH_T2
    fp_type = dfmp2_addons.CONFIG_FP_TYPE
    j2c_decomp_alg = dfmp2_addons.CONFIG_J2C_DECOMP_ALG

    _keys = {
        'mo_energy',
        'auxmol',
        'with_t2',
        'fp_type',
        'j2c_decomp_alg',
    }

    def __init__(self, mf, auxbasis=None):
        super().__init__(mf)

        self.mo_energy = mf.mo_energy

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

        return kernel(self, *args, **kwargs)
