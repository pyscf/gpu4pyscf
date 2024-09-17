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

'''
PCM family solvent model
'''
# pylint: disable=C0103
import ctypes
import numpy
import cupy
import cupyx.scipy as scipy
from pyscf import lib
from pyscf import gto
from pyscf.dft import gen_grid
from pyscf.data import radii
from pyscf.solvent import ddcosmo
from gpu4pyscf.solvent import _attach_solvent
from gpu4pyscf.df import int3c2e
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import dist_matrix, load_library

libdft = lib.load_library('libdft')
libsolvent = load_library('libsolvent')

@lib.with_doc(_attach_solvent._for_scf.__doc__)
def pcm_for_scf(mf, solvent_obj=None, dm=None):
    if solvent_obj is None:
        solvent_obj = PCM(mf.mol)
    return _attach_solvent._for_scf(mf, solvent_obj, dm)

# Inject PCM to SCF, TODO: add it to other methods later
from gpu4pyscf import scf
scf.hf.RHF.PCM = pcm_for_scf
scf.uhf.UHF.PCM = pcm_for_scf
# TABLE II,  J. Chem. Phys. 122, 194110 (2005)
XI = {
    6: 4.84566077868,
    14: 4.86458714334,
    26: 4.85478226219,
    38: 4.90105812685,
    50: 4.89250673295,
    86: 4.89741372580,
    110: 4.90101060987,
    146: 4.89825187392,
    170: 4.90685517725,
    194: 4.90337644248,
    302: 4.90498088169,
    350: 4.86879474832,
    434: 4.90567349080,
    590: 4.90624071359,
    770: 4.90656435779,
    974: 4.90685167998,
    1202: 4.90704098216,
    1454: 4.90721023869,
    1730: 4.90733270691,
    2030: 4.90744499142,
    2354: 4.90753082825,
    2702: 4.90760972766,
    3074: 4.90767282394,
    3470: 4.90773141371,
    3890: 4.90777965981,
    4334: 4.90782469526,
    4802: 4.90749125553,
    5294: 4.90762073452,
    5810: 4.90792902522,
}

modified_Bondi = radii.VDW.copy()
modified_Bondi[1] = 1.1/radii.BOHR      # modified version
#radii_table = bondi * 1.2
PI = numpy.pi

def switch_h(x):
    '''
    switching function (eq. 3.19)
    J. Chem. Phys. 133, 244111 (2010)
    notice the typo in the paper
    '''
    y = x**3 * (10.0 - 15.0*x + 6.0*x**2)
    y[x<0] = 0.0
    y[x>1] = 1.0
    return y

def gen_surface(mol, ng=302, rad=modified_Bondi, vdw_scale=1.2, r_probe=0.0):
    '''J. Phys. Chem. A 1999, 103, 11060-11079'''
    unit_sphere = numpy.empty((ng,4))
    libdft.MakeAngularGrid(unit_sphere.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ng))
    unit_sphere = cupy.asarray(unit_sphere)

    atom_coords = cupy.asarray(mol.atom_coords(unit='B'))
    charges = mol.atom_charges()
    N_J = ng * cupy.ones(mol.natm)
    R_J = cupy.asarray([rad[chg] for chg in charges])
    R_sw_J = R_J * (14.0 / N_J)**0.5
    alpha_J = 1.0/2.0 + R_J/R_sw_J - ((R_J/R_sw_J)**2 - 1.0/28)**0.5
    R_in_J = R_J - alpha_J * R_sw_J

    grid_coords = []
    weights = []
    charge_exp = []
    switch_fun = []
    R_vdw = []
    norm_vec = []
    area = []
    gslice_by_atom = []
    p0 = p1 = 0
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        chg = gto.charge(symb)
        r_vdw = rad[chg]

        atom_grid = r_vdw * unit_sphere[:,:3] + atom_coords[ia,:]
        #riJ = scipy.spatial.distance.cdist(atom_grid[:,:3], atom_coords)
        #riJ = cupy.sum((atom_grid[:,None,:] - atom_coords[None,:,:])**2, axis=2)**0.5
        riJ = dist_matrix(atom_grid, atom_coords)
        diJ = (riJ - R_in_J) / R_sw_J
        diJ[:,ia] = 1.0
        diJ[diJ<1e-8] = 0.0

        fiJ = switch_h(diJ)

        w = unit_sphere[:,3] * 4.0 * PI
        swf = cupy.prod(fiJ, axis=1)
        idx = w*swf > 1e-12

        p0, p1 = p1, p1+sum(idx).get()
        gslice_by_atom.append([p0,p1])
        grid_coords.append(atom_grid[idx,:3])
        weights.append(w[idx])
        switch_fun.append(swf[idx])
        norm_vec.append(unit_sphere[idx,:3])
        xi = XI[ng] / (r_vdw * w[idx]**0.5)
        charge_exp.append(xi)
        R_vdw.append(cupy.ones(idx.sum().get()) * r_vdw)
        area.append(w[idx]*r_vdw**2*swf[idx])

    grid_coords = cupy.vstack(grid_coords)
    norm_vec = cupy.vstack(norm_vec)
    weights = cupy.concatenate(weights)
    charge_exp = cupy.concatenate(charge_exp)
    switch_fun = cupy.concatenate(switch_fun)
    area = cupy.concatenate(area)
    R_vdw = cupy.concatenate(R_vdw)

    surface = {
        'ng': ng,
        'gslice_by_atom': gslice_by_atom,
        'grid_coords': grid_coords,
        'weights': weights,
        'charge_exp': charge_exp,
        'switch_fun': switch_fun,
        'R_vdw': R_vdw,
        'norm_vec': norm_vec,
        'area': area,
        'R_in_J': R_in_J,
        'R_sw_J': R_sw_J,
        'atom_coords': atom_coords
    }
    return surface

def get_F_A(surface):
    '''
    generate F and A matrix in  J. Chem. Phys. 133, 244111 (2010)
    '''
    R_vdw = surface['R_vdw']
    switch_fun = surface['switch_fun']
    weights = surface['weights']
    A = weights*R_vdw**2*switch_fun
    return switch_fun, A

def get_D_S_slow(surface, with_S=True, with_D=False):
    '''
    generate D and S matrix in  J. Chem. Phys. 133, 244111 (2010)
    The diagonal entries of S is not filled
    '''
    charge_exp  = surface['charge_exp']
    grid_coords = surface['grid_coords']
    switch_fun  = surface['switch_fun']
    norm_vec    = surface['norm_vec']
    R_vdw       = surface['R_vdw']

    xi_i, xi_j = cupy.meshgrid(charge_exp, charge_exp, indexing='ij')
    xi_ij = xi_i * xi_j / (xi_i**2 + xi_j**2)**0.5
    rij = dist_matrix(grid_coords, grid_coords)
    xi_r_ij = xi_ij * rij
    cupy.fill_diagonal(rij, 1)
    S = scipy.special.erf(xi_r_ij) / rij
    cupy.fill_diagonal(S, charge_exp * (2.0 / PI)**0.5 / switch_fun)

    D = None
    if with_D:
        nrij = grid_coords.dot(norm_vec.T) - cupy.sum(grid_coords * norm_vec, axis=-1)
        D = S*nrij/rij**2 -2.0*xi_r_ij/PI**0.5*cupy.exp(-xi_r_ij**2)*nrij/rij**3
        cupy.fill_diagonal(D, -charge_exp * (2.0 / PI)**0.5 / (2.0 * R_vdw))
    return D, S

def get_D_S(surface, with_S=True, with_D=False, stream=None):
    ''' Efficiently generating D matrix and S matrix in PCM models '''
    charge_exp  = surface['charge_exp']
    grid_coords = surface['grid_coords']
    switch_fun  = surface['switch_fun']
    norm_vec    = surface['norm_vec']
    R_vdw       = surface['R_vdw']
    n = charge_exp.shape[0]
    S = cupy.empty([n,n])
    D = None
    S_ptr = ctypes.cast(S.data.ptr, ctypes.c_void_p)
    D_ptr = lib.c_null_ptr()
    if with_D:
        D = cupy.empty([n,n])
        D_ptr = ctypes.cast(D.data.ptr, ctypes.c_void_p)
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    err = libsolvent.pcm_d_s(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        D_ptr, S_ptr,
        ctypes.cast(grid_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(norm_vec.data.ptr, ctypes.c_void_p),
        ctypes.cast(R_vdw.data.ptr, ctypes.c_void_p),
        ctypes.cast(charge_exp.data.ptr, ctypes.c_void_p),
        ctypes.cast(switch_fun.data.ptr, ctypes.c_void_p),
        ctypes.c_int(n)
    )
    if err != 0:
        raise RuntimeError('Failed in generating PCM D and S matrices.')
    return D, S

class PCM(lib.StreamObject):
    _keys = {
        'method', 'vdw_scale', 'surface', 'r_probe', 'intopt',
        'mol', 'radii_table', 'atom_radii', 'lebedev_order', 'lmax', 'eta',
        'eps', 'grids', 'max_cycle', 'conv_tol', 'state_id', 'frozen',
        'equilibrium_solvation', 'e', 'v',
    }
    from gpu4pyscf.lib.utils import to_gpu, device
    kernel = ddcosmo.DDCOSMO.kernel

    def __init__(self, mol):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory
        self.method = 'C-PCM'

        self.vdw_scale = 1.2 # default value in qchem
        self.surface = {}
        self.r_probe = 0.0
        self.radii_table = None
        self.atom_radii = None
        self.lebedev_order = 29
        self._intermediates = {}
        self.eps = 78.3553

        self.max_cycle = 20
        self.conv_tol = 1e-7
        self.state_id = 0

        self.frozen = False
        self.equilibrium_solvation = False

        self.e = None
        self.v = None
        self._dm = None

    def dump_flags(self, verbose=None):
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'lebedev_order = %s (%d grids per sphere)',
                    self.lebedev_order, gen_grid.LEBEDEV_ORDER[self.lebedev_order])
        logger.info(self, 'eps = %s'          , self.eps)
        logger.info(self, 'frozen = %s'       , self.frozen)
        logger.info(self, 'equilibrium_solvation = %s', self.equilibrium_solvation)
        logger.debug2(self, 'radii_table %s', self.radii_table)
        if self.atom_radii:
            logger.info(self, 'User specified atomic radii %s', str(self.atom_radii))
        return self

    def to_cpu(self):
        from gpu4pyscf.lib.utils import to_cpu
        obj = to_cpu(self)
        return obj.reset()

    def build(self, ng=None):
        if self.radii_table is None:
            vdw_scale = self.vdw_scale
            self.radii_table = vdw_scale * modified_Bondi + self.r_probe
        mol = self.mol
        if ng is None:
            ng = gen_grid.LEBEDEV_ORDER[self.lebedev_order]

        self.surface = gen_surface(mol, rad=self.radii_table, ng=ng)
        self._intermediates = {}
        F, A = get_F_A(self.surface)
        D, S = get_D_S(self.surface, with_S=True, with_D=True)

        epsilon = self.eps
        if self.method.upper() in ['C-PCM', 'CPCM']:
            f_epsilon = (epsilon-1.)/epsilon
            K = S
            R = -f_epsilon * cupy.eye(K.shape[0])
        elif self.method.upper() == 'COSMO':
            f_epsilon = (epsilon - 1.0)/(epsilon + 1.0/2.0)
            K = S
            R = -f_epsilon * cupy.eye(K.shape[0])
        elif self.method.upper() in ['IEF-PCM', 'IEFPCM']:
            f_epsilon = (epsilon - 1.0)/(epsilon + 1.0)
            DA = D*A
            DAS = cupy.dot(DA, S)
            K = S - f_epsilon/(2.0*PI) * DAS
            R = -f_epsilon * (cupy.eye(K.shape[0]) - 1.0/(2.0*PI)*DA)
        elif self.method.upper() == 'SS(V)PE':
            f_epsilon = (epsilon - 1.0)/(epsilon + 1.0)
            DA = D*A
            DAS = cupy.dot(DA, S)
            K = S - f_epsilon/(4.0*PI) * (DAS + DAS.T)
            R = -f_epsilon * (cupy.eye(K.shape[0]) - 1.0/(2.0*PI)*DA)
        else:
            raise RuntimeError(f"Unknown implicit solvent model: {self.method}")

        intermediates = {
            'S': cupy.asarray(S),
            'D': cupy.asarray(D),
            'A': cupy.asarray(A),
            'K': cupy.asarray(K),
            'R': cupy.asarray(R),
            'f_epsilon': f_epsilon
        }
        self._intermediates.update(intermediates)

        charge_exp  = self.surface['charge_exp']
        grid_coords = self.surface['grid_coords']
        atom_coords = mol.atom_coords(unit='B')
        atom_charges = mol.atom_charges()

        auxmol = gto.fakemol_for_charges(grid_coords.get(), expnt=charge_exp.get()**2)
        intopt = int3c2e.VHFOpt(mol, auxmol, 'int2e')
        intopt.build(1e-14, diag_block_with_triu=False, aosym=True, group_size=256)
        self.intopt = intopt

        int2c2e = mol._add_suffix('int2c2e')
        fakemol_nuc = gto.fakemol_for_charges(atom_coords)
        v_ng = gto.mole.intor_cross(int2c2e, fakemol_nuc, auxmol)
        v_grids_n = numpy.dot(atom_charges, v_ng)
        self.v_grids_n = cupy.asarray(v_grids_n)

    def _get_vind(self, dms):
        if not self._intermediates:
            self.build()
        nao = dms.shape[-1]
        dms = dms.reshape(-1,nao,nao)
        if dms.shape[0] == 2:
            dms = (dms[0] + dms[1]).reshape(-1,nao,nao)
        K = self._intermediates['K']
        R = self._intermediates['R']
        v_grids_e = self._get_v(dms)
        v_grids = self.v_grids_n - v_grids_e

        b = cupy.dot(R, v_grids.T)
        q = cupy.linalg.solve(K, b).T

        vK_1 = cupy.linalg.solve(K.T, v_grids.T)
        qt = cupy.dot(R.T, vK_1).T
        q_sym = (q + qt)/2.0

        vmat = self._get_vmat(q_sym)
        epcm = 0.5 * cupy.dot(v_grids[0], q_sym[0])

        self._intermediates['K'] = K
        self._intermediates['R'] = R
        self._intermediates['q'] = q[0]
        self._intermediates['q_sym'] = q_sym[0]
        self._intermediates['v_grids'] = v_grids[0]
        return epcm, vmat[0]

    def _get_v(self, dms):
        '''
        return electrostatic potential on surface
        '''
        nset = dms.shape[0]
        ngrids = self.surface['grid_coords'].shape[0]
        v_grids_e = cupy.empty([nset, ngrids])
        for i in range(nset):
            v_grids_e[i] = 2.0*int3c2e.get_j_int3c2e_pass1(self.intopt, dms[i])
        return v_grids_e

    def _get_vmat(self, q):
        assert q.ndim == 2
        nset = q.shape[0]
        nao = self.mol.nao
        vmat = cupy.empty([nset, nao, nao])
        for i in range(nset):
            vmat[i] = -int3c2e.get_j_int3c2e_pass2(self.intopt, q[i])
        return vmat

    def nuc_grad_method(self, grad_method):
        from gpu4pyscf.solvent.grad import pcm as pcm_grad
        if self.frozen:
            raise RuntimeError('Frozen solvent model is not supported')
        from gpu4pyscf import scf
        if isinstance(grad_method.base, (scf.hf.RHF, scf.uhf.UHF)):
            return pcm_grad.make_grad_object(grad_method)
        else:
            raise RuntimeError('Only SCF gradient is supported')

    def Hessian(self, hess_method):
        from gpu4pyscf.solvent.hessian import pcm as pcm_hess
        if self.frozen:
            raise RuntimeError('Frozen solvent model is not supported')
        from gpu4pyscf import scf
        if isinstance(hess_method.base, (scf.hf.RHF, scf.uhf.UHF)):
            return pcm_hess.make_hess_object(hess_method)
        else:
            raise RuntimeError('Only SCF gradient is supported')

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._intermediates = None
        self.surface = None
        self.intopt = None
        return self

    def _B_dot_x(self, dms):
        if not self._intermediates:
            self.build()
        out_shape = dms.shape
        nao = dms.shape[-1]
        dms = dms.reshape(-1,nao,nao)

        K = self._intermediates['K']
        R = self._intermediates['R']
        v_grids = -self._get_v(dms)

        b = cupy.dot(R, v_grids.T)
        q = cupy.linalg.solve(K, b).T

        vK_1 = cupy.linalg.solve(K.T, v_grids.T)
        qt = cupy.dot(R.T, vK_1).T
        q_sym = (q + qt)/2.0

        vmat = self._get_vmat(q_sym)
        return vmat.reshape(out_shape)

