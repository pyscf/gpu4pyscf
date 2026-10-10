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

'''
SMD solvent model
'''

import numpy as np
import cupy
from pyscf import lib, gto
from pyscf.data import radii
from pyscf.dft.gen_grid import LEBEDEV_ORDER
from gpu4pyscf.solvent import pcm, _attach_solvent
from gpu4pyscf.solvent._solvent_data import solvent_db, resolve_solvent_name
from gpu4pyscf.solvent.pcm import natm_without_ghost
from gpu4pyscf.lib import logger
from gpu4pyscf.gto import int3c1e
from cupyx.scipy.linalg import lu_factor
from gpu4pyscf.lib import utils

@lib.with_doc(_attach_solvent._for_scf.__doc__)
def smd_for_scf(mf, solvent_obj=None, dm=None, solvent='water'):
    if isinstance(solvent_obj, str):
        solvent_obj = SMD(mf.mol, solvent=solvent_obj)
    if solvent_obj is None:
        solvent_obj = SMD(mf.mol, solvent=solvent)
    return _attach_solvent._for_scf(mf, solvent_obj, dm)

# Inject PCM to SCF, TODO: add it to other methods later
from gpu4pyscf import scf
scf.hf.RHF.SMD = smd_for_scf
scf.uhf.UHF.SMD = smd_for_scf
hartree2kcal = 627.509451

def smd_radii(alpha):
    '''
    eq. (16)
    use smd radii if defined
    use Bondi radii if defined
    use 2.0 otherwise
    '''
    radii_table = radii.VDW.copy() * radii.BOHR
    radii_table[1] = 1.20
    radii_table[6] = 1.85
    radii_table[7] = 1.89
    if alpha >= 0.43:
        r = 1.52
    else:
        r = 1.52 + 1.8 * (0.43 - alpha)
    radii_table[8] = r
    radii_table[9] = 1.73
    radii_table[14] = 2.47
    radii_table[15] = 2.12
    radii_table[16] = 2.49
    radii_table[17] = 2.38
    #radii_table[35] = 3.06 # original SMD
    # following value from SMD18
    # https://chemistry-europe.onlinelibrary.wiley.com/doi/10.1002/chem.201803652
    radii_table[35] = 2.60
    radii_table[53] = 2.74
    return radii_table/radii.BOHR

import ctypes
from gpu4pyscf.lib.cupy_helper import load_library
try:
    libsolvent = load_library('libsolvent')
except OSError:
    libsolvent = None

def get_cds_legacy(smdobj):
    mol = smdobj.mol
    natm = mol.natm
    solvent_descriptors = smdobj.get_solvent_descriptors()
    soln, _, sola, solb, solg, _, solc, solh = solvent_descriptors
    #symbols = [mol.atom_s(ia) for ia in range(mol.natm)]
    charges = np.asarray(mol.atom_charges(), dtype=np.int32, order='F')
    coords = np.asarray(mol.atom_coords(unit='B'), dtype=np.float64, order='C')
    icds = 1 if smdobj.solvent.upper() == 'WATER' else 2
    dcds = np.empty([natm,3])
    mnsol_interface =  libsolvent.mnsol_interface_

    double_ndptr = np.ctypeslib.ndpointer(dtype=np.float64)
    int_ndptr = np.ctypeslib.ndpointer(dtype=np.int32)
    double_ptr = ctypes.POINTER(ctypes.c_double)
    int_ptr = ctypes.POINTER(ctypes.c_int)

    mnsol_interface.argtypes = [
        double_ndptr, int_ndptr,
        int_ptr,
        double_ptr, double_ptr, double_ptr, double_ptr, double_ptr, double_ptr,
        int_ptr,
        double_ptr, double_ptr, double_ndptr]
    natm = ctypes.byref(ctypes.c_int(natm))
    icds = ctypes.byref(ctypes.c_int(icds))
    soln = ctypes.byref(ctypes.c_double(soln))
    sola = ctypes.byref(ctypes.c_double(sola))
    solb = ctypes.byref(ctypes.c_double(solb))
    solg = ctypes.byref(ctypes.c_double(solg))
    solc = ctypes.byref(ctypes.c_double(solc))
    solh = ctypes.byref(ctypes.c_double(solh))
    gcds = ctypes.c_double()
    areacds = ctypes.c_double()

    mnsol_interface(coords, charges,
                    natm,
                    sola, solb, solc, solg, solh, soln,
                    icds,
                    ctypes.byref(gcds), ctypes.byref(areacds), dcds)
    return gcds.value / hartree2kcal, dcds

def from_cpu(method):
    out = lib.to_gpu(method, out=SMD(method.mol, method.solvent))
    out.reset()
    # Older PySCF stores the solvent name in the _solvent attribute.
    out.__dict__.pop('_solvent', None)
    return out

class SMD(lib.StreamObject):
    '''SMD with optional overrides of the selected solvent's parameters.
    '''

    eps_optical = None
    to_gpu = utils.to_gpu
    device = utils.device

    _keys = {
        'method', 'vdw_scale', 'sasa_ng',
        'mol', 'radii_table', 'lebedev_order', 'lmax', 'eta',
        'solvent', 'eps', 'eps_optical', 'max_cycle', 'conv_tol', 'state_id', 'frozen',
        'frozen_dm0_for_finite_difference_without_response',
        'equilibrium_solvation', 'solvent_descriptors',
        'surface', 'intopt', 'e', 'v', 'v_grids_n', 'e_cds',
    }

    def __init__(self, mol, solvent='water'):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory

        self.vdw_scale = 1.0
        self.sasa_ng = 590 # quadrature grids for calculating SASA
        self.method = 'SMD'
        self.solvent = solvent
        self.solvent_descriptors = None
        self.radii_table = None
        self.eps = None
        self.surface_discretization_method = "SWIG"
        self.max_cycle = 20
        self.conv_tol = 1e-7
        self.state_id = 0
        self.frozen = False
        self.frozen_dm0_for_finite_difference_without_response = None
        self.equilibrium_solvation = False

        # Following are intermediates
        self.surface = {}
        self._intermediates = {}
        self.e = None
        self.v = None
        self.v_grids_n = None
        self.e_cds = None

    def __setattr__(self, key, val):
        if key == 'solvent':
            val = self._set_solvent(val)
        elif key == 'solvent_descriptors' and val is not None:
            if len(val) != 8:
                raise ValueError('SMD solvent_descriptors must contain eight values')
            val = tuple(val)
        super().__setattr__(key, val)
        if (key in ('solvent', 'solvent_descriptors', 'eps', 'eps_optical')
                and '_intermediates' in self.__dict__):
            self.reset()

    def _set_solvent(self, solvent):
        return resolve_solvent_name(solvent)

    def get_solvent_descriptors(self):
        '''Return custom descriptors, or the selected solvent's database values.'''
        if self.solvent_descriptors is not None:
            return self.solvent_descriptors
        if not self.solvent:
            raise ValueError('SMD requires a solvent name or solvent_descriptors')
        return tuple(solvent_db[self.solvent])

    def get_eps(self):
        '''Return the static dielectric constant.'''
        if self.eps is not None:
            return self.eps
        return self.get_solvent_descriptors()[5]

    def get_eps_optical(self):
        '''The optical (high-frequency) dielectric constant of the solvent.'''
        if self.eps_optical is not None:
            return self.eps_optical
        n = self.get_solvent_descriptors()[0]
        return n**2

    @property
    def sol_desc(self):
        return self.solvent_descriptors

    @sol_desc.setter
    def sol_desc(self, values):
        '''Assign custom descriptors, or None to restore database defaults.'''
        self.solvent_descriptors = values

    @property
    def lebedev_order(self):
        for key, val in LEBEDEV_ORDER.items():
            if val == self.sasa_ng:
                return key
        raise RuntimeError(f'sasa_ng={self.sasa_ng} does not have a corresponding lebedev_order')
    @lebedev_order.setter
    def lebedev_order(self, x):
        self.sasa_ng = LEBEDEV_ORDER[x]

    def dump_flags(self, verbose=None):
        solvent_descriptors = self.get_solvent_descriptors()
        n, _, alpha, beta, gamma, _, phi, psi = solvent_descriptors
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'solvent = %s', self.solvent)
        logger.info(self, 'eps_optical = %s (%s)', self.get_eps_optical(),
                    'override' if self.eps_optical is not None else 'default')
        logger.info(self, 'sasa_ng = %s', self.sasa_ng)
        logger.info(self, 'eps = %s (%s)', self.get_eps(),
                    'override' if self.eps is not None else 'default')
        logger.info(self, 'solvent descriptors = %s',
                    'custom' if self.solvent_descriptors is not None else 'database')
        logger.info(self, 'CDS treatment = %s',
                    'water' if self.solvent == 'water' else 'non-water')
        logger.info(self, 'frozen = %s', self.frozen)
        logger.info(self, '---------- SMD solvent descriptors -------')
        logger.info(self, f'n     = {n}')
        logger.info(self, f'alpha = {alpha}')
        logger.info(self, f'beta  = {beta}')
        logger.info(self, f'gamma = {gamma}')
        logger.info(self, f'phi   = {phi}')
        logger.info(self, f'psi   = {psi}')
        logger.info(self, '--------------------- end ----------------')
        logger.info(self, 'equilibrium_solvation = %s', self.equilibrium_solvation)
        return self

    def build(self, ng=None):
        solvent_descriptors = self.get_solvent_descriptors()
        if self.radii_table is None:
            radii_table = smd_radii(solvent_descriptors[2])
        else:
            radii_table = cupy.asnumpy(self.radii_table)
        logger.debug(self, 'radii_table %s', radii_table*radii.BOHR)
        mol = self.mol
        if ng is None:
            ng = self.sasa_ng

        if natm_without_ghost(mol) != mol.natm:
            raise RuntimeError('SMD does not support ghost atoms')

        self.surface = pcm.gen_surface(mol, rad=radii_table, ng=ng)
        self._intermediates = {}
        F, A = pcm.get_F_A(self.surface)
        D, S = pcm.get_D_S(self.surface, with_S=True, with_D=True)

        epsilon = self.get_eps()
        f_epsilon = (epsilon - 1.0)/(epsilon + 1.0) if epsilon != float('inf') else 1.
        DA = D*A
        DAS = cupy.dot(DA, S)
        K = S - f_epsilon/(2.0*np.pi) * DAS
        K_LU, K_LU_pivot = lu_factor(K, overwrite_a = True, check_finite = False)
        K = None

        intermediates = {
            'S': cupy.asarray(S),
            'D': cupy.asarray(D),
            'A': cupy.asarray(A),
            'K_LU': cupy.asarray(K_LU),
            'K_LU_pivot': cupy.asarray(K_LU_pivot),
            'f_epsilon': f_epsilon
        }
        self._intermediates.update(intermediates)

        charge_exp  = self.surface['charge_exp']
        grid_coords = self.surface['grid_coords']
        atom_coords = mol.atom_coords(unit='B')
        atom_charges = mol.atom_charges()

        # Move this to GPU
        intopt = int3c1e.VHFOpt(mol)
        intopt.build(1e-14)
        self.intopt = intopt

        int2c2e = mol._add_suffix('int2c2e')
        fakemol_charge = gto.fakemol_for_charges(grid_coords.get(), expnt=charge_exp.get()**2)
        fakemol_nuc = gto.fakemol_for_charges(atom_coords)
        v_ng = gto.mole.intor_cross(int2c2e, fakemol_nuc, fakemol_charge)
        v_grids_n = np.dot(atom_charges, v_ng)
        self.v_grids_n = cupy.asarray(v_grids_n)

    kernel = pcm.PCM.kernel
    _get_vind = pcm.PCM._get_vind
    _get_qsym = pcm.PCM._get_qsym
    _get_vgrids = pcm.PCM._get_vgrids
    _get_v = pcm.PCM._get_v
    _get_vmat = pcm.PCM._get_vmat
    _B_dot_x = pcm.PCM._B_dot_x
    left_multiply_R = pcm.PCM.left_multiply_R
    left_solve_K = pcm.PCM.left_solve_K
    if_method_in_CPCM_category = pcm.PCM.if_method_in_CPCM_category

    def get_cds(self):
        if self.e_cds is None:
            self.e_cds = get_cds_legacy(self)[0]
        return self.e_cds

    def nuc_grad_method(self, grad_method):
        raise DeprecationWarning

    def grad(self, dm):
        '''This function computes intermediates for Gradients. It is intended
        for internal use only and should not be called directly by users.
        '''
        from gpu4pyscf.solvent.grad.pcm import grad_qv, grad_nuc, grad_solver
        de_solvent = grad_qv(self, dm)
        de_solvent+= grad_solver(self, dm)
        de_solvent+= grad_nuc(self, dm)
        return de_solvent

    def Hessian(self, hess_method):
        raise DeprecationWarning

    def hess(self, dm):
        '''This function computes intermediates for Hessian. It is intended
        for internal use only and should not be called directly by users.
        '''
        from gpu4pyscf.solvent.hessian.pcm import (
            analytical_hess_nuc, analytical_hess_qv, analytical_hess_solver)
        de_solvent  =    analytical_hess_nuc(self, dm, verbose=self.verbose)
        de_solvent +=     analytical_hess_qv(self, dm, verbose=self.verbose)
        de_solvent += analytical_hess_solver(self, dm, verbose=self.verbose)
        return de_solvent

    def reset(self, mol=None):
        pcm.PCM.reset(self, mol)
        self.e_cds = None
        return self

    def to_cpu(self):
        from pyscf.solvent.smd import SMD
        out = utils.to_cpu(self, SMD(self.mol))
        out.reset()
        if hasattr(out, 'lebedev_order'):
            out.lebedev_order = self.lebedev_order
        out.solvent = self.solvent
        if self.solvent_descriptors is not None:
            out.solvent_descriptors = self.solvent_descriptors
        solvent_descriptors = self.get_solvent_descriptors()
        out.eps = self.eps
        out.eps_optical = self.eps_optical
        if self.radii_table is None:
            out.radii_table = smd_radii(solvent_descriptors[2])
        else:
            out.radii_table = cupy.asnumpy(self.radii_table)
        return out
