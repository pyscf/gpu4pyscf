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

from functools import reduce
from pyscf.hessian import thermo
import numpy as np
import cupy
from pyscf.data import elements, nist
from scipy.constants import physical_constants
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract

LINDEP_THRESHOLD = 1e-7


def eval_ir_freq_intensity(mf, hessian_obj):
    """main function to calculate the polarizability

    Args:
        mf: mean field object
        unit (str, optional): the unit of the polarizability. Defaults to 'au'.

    Returns:
        polarizability (numpy.array): polarizability
    """
    log = logger.new_logger(hessian_obj, mf.mol.verbose)
    hessian = hessian_obj.kernel()
    hartree_kj = nist.HARTREE2J*1e3
    unit2cm = ((hartree_kj * nist.AVOGADRO)**.5 / (nist.BOHR*1e-10)
               / (2*np.pi*nist.LIGHT_SPEED_SI) * 1e-2)
    natm = mf.mol.natm
    nao = mf.mol.nao
    dm0 = mf.make_rdm1()

    atom_charges = mf.mol.atom_charges()
    mass = cupy.array([elements.MASSES[atom_charges[i]] for i in range(natm)])
    # hessian_mass = contract('ijkl,i,j->ijkl', hessian,
    #                          1/cupy.sqrt(mass), 1/cupy.sqrt(mass))
    hessian_mass = contract('ijkl,i->ijkl', cupy.array(hessian), 1/cupy.sqrt(mass))
    hessian_mass = contract('ijkl,j->ijkl', hessian_mass, 1/cupy.sqrt(mass))

    TR = thermo._get_TR(mass.get(), mf.mol.atom_coords())
    TRspace = []
    TRspace.append(TR[:3])
    
    rot_const = thermo.rotation_const(mass.get(), mf.mol.atom_coords())
    rotor_type = thermo._get_rotor_type(rot_const)
    if rotor_type == 'ATOM':
        pass
    elif rotor_type == 'LINEAR':  # linear molecule
        TRspace.append(TR[3:5])
    else:
        TRspace.append(TR[3:])

    if TRspace:
        TRspace = cupy.vstack(TRspace)
        q, r = cupy.linalg.qr(TRspace.T)
        P = cupy.eye(natm * 3) - q.dot(q.T)
        w, v = cupy.linalg.eigh(P)
        bvec = v[:,w > LINDEP_THRESHOLD]
        h = reduce(cupy.dot, (bvec.T, hessian_mass.transpose(0,2,1,3).reshape(3*natm,3*natm), bvec))
        e, mode = cupy.linalg.eigh(h)
        mode = bvec.dot(mode)
    
    c = contract('ixn,i->ixn', mode.reshape(natm, 3, -1),
                  1/np.sqrt(mass)).reshape(3*natm, -1)
    freq = cupy.sign(e)*cupy.sqrt(cupy.abs(e))*unit2cm

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    mo_coeff = cupy.array(mo_coeff)
    mo_occ = cupy.array(mo_occ)
    mo_energy = cupy.array(mo_energy)
    mocc = mo_coeff[:, mo_occ > 0]
    mocc = cupy.array(mocc)

    atmlst = range(natm)
    h1ao = hessian_obj.make_h1(mo_coeff, mo_occ, None, atmlst)
    # TODO: compact with hessian method, which can save one time cphf solve.
    # ! Different from PySCF, mo1 is all in mo!
    mo1, mo_e1 = hessian_obj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao,
                                       None, atmlst, hessian_obj.max_memory, log)  

    
    tmp = cupy.empty((3, 3, natm))  # dipole moment, x,y,z
    aoslices = mf.mol.aoslice_by_atom()
    with mf.mol.with_common_orig((0, 0, 0)):
        hmuao = mf.mol.intor('int1e_r')  # mu
        hmuao11 = -mf.mol.intor('int1e_irp').reshape(3, 3, nao, nao)
    hmuao = cupy.array(hmuao)
    hmuao11 = cupy.array(hmuao11)
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h11ao = cupy.zeros((3, 3, nao, nao))

        h11ao[:, :, :, p0:p1] += hmuao11[:, :, :, p0:p1]
        h11ao[:, :, p0:p1] += hmuao11[:, :, :, p0:p1].transpose(0, 1, 3, 2)

        tmp0 = contract('ypi,vi->ypv', mo1[ia], mocc)  # nabla
        dm1 = contract('ypv,up->yuv', tmp0, mo_coeff)
        tmp[:, :, ia] = -contract('xuv,yuv->xy', hmuao, dm1) * 4 #the minus means the density should be negative, but calculate it is positive.
        tmp[:, :, ia] -= contract('xyuv,vu->xy', h11ao, dm0)
        tmp[:, :, ia] += mf.mol.atom_charge(ia)*cupy.eye(3)

    alpha = physical_constants["fine-structure constant"][0]
    amu = physical_constants["atomic mass constant"][0]
    m_e = physical_constants["electron mass"][0]
    N_A = physical_constants["Avogadro constant"][0]
    a_0 = physical_constants["Bohr radius"][0]
    unit_kmmol = alpha**2 * (1e-3 / amu) * m_e * N_A * np.pi * a_0 / 3

    intensity = contract('xym,myn->xn', tmp, c.reshape(natm, 3, -1))
    intensity = contract('xn,xn->n', intensity, intensity)
    intensity *= unit_kmmol

    return freq, intensity
