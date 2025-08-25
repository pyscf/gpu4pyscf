#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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

import ctypes
import numpy as np
from pyscf import gto
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.dft.multigrid.multigrid_pair import _eval_rhoG
from pyscf.pbc.dft.multigrid.pp import fake_cell_vloc_part1
import cupy as cp

from gpu4pyscf.lib.cupy_helper import load_library
libpbc = load_library('libpbc')
# from pyscf import lib
# libpbc = lib.load_library('libdft')


# Henry's note 20250821:
# The functions in this file, as well as functions in gpu4pyscf/lib/pbc/grid_integrate.c,
# are direct copies of the corresponding functions in PySCF.
# The reason is to get around with a version problem:
# Interface function vpploc_part1_nuc_grad() supports non-orthogonal lattice since
# pyscf==2.10.0, however we want gpu4pyscf to be compatable with older version of pyscf,
# particularly pyscf==2.8.0, the version used by github CI.
# So, we made a copy. Ugly, nasty, TODO: suppose to be replaced with GPU implementation.


def int_gauss_charge_v_rs(
    v_rs,
    comp,
    atm,
    bas,
    env,
    mesh,
    dimension,
    a,
    b,
    max_radius,
):
    if abs(a - np.diag(a.diagonal())).max() < 1e-12:
        lattice_type = '_orth'
        orth = True
    else:
        lattice_type = '_nonorth'
        orth = False

    fn_name = 'eval_mat_lda' + lattice_type
    if comp == 3:
        fn_name += '_ip1'
    elif comp != 1:
        raise NotImplementedError

    out = np.zeros((len(atm), comp), order='C', dtype=np.double)
    v_rs = np.asarray(v_rs, order='C', dtype=np.double)
    atm = np.asarray(atm, order='C', dtype=np.int32)
    bas = np.asarray(bas, order='C', dtype=np.int32)
    env = np.asarray(env, order='C', dtype=np.double)
    mesh = np.asarray(mesh, order='C', dtype=np.int32)
    a = np.asarray(a, order='C', dtype=np.double)
    b = np.asarray(b, order='C', dtype=np.double)

    libpbc.int_gauss_charge_v_rs(
        getattr(libpbc, fn_name),
        out.ctypes.data_as(ctypes.c_void_p),
        v_rs.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(comp),
        atm.ctypes.data_as(ctypes.c_void_p),
        bas.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(len(bas)),
        env.ctypes.data_as(ctypes.c_void_p),
        mesh.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(dimension),
        a.ctypes.data_as(ctypes.c_void_p),
        b.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(max_radius),
        ctypes.c_bool(orth)
    )
    return out


def vpploc_part1_nuc_grad(mydf, dm, kpts=np.zeros((1,3)), atm_id=None, precision=None):
    if isinstance(kpts, KPoints):
        raise NotImplementedError
    t0 = (logger.process_clock(), logger.perf_counter())
    cell = mydf.cell
    fakecell, max_radius = fake_cell_vloc_part1(cell, atm_id=atm_id, precision=precision)
    atm = fakecell._atm
    bas = fakecell._bas
    env = fakecell._env

    a = cell.lattice_vectors()
    b = np.linalg.inv(a.T)

    mesh = mydf.mesh
    ngrids = np.prod(mesh)
    comp = 3

    if mydf.rhoG is None:
        rhoG = _eval_rhoG(mydf, dm, hermi=1, kpts=kpts, deriv=0)
    else:
        rhoG = mydf.rhoG
    rhoG = rhoG[...,0,:]
    rhoG = rhoG.reshape(-1,ngrids)
    if rhoG.shape[0] == 2: #unrestricted
        rhoG = rhoG[0] + rhoG[1]
    else:
        assert rhoG.shape[0] == 1
        rhoG = rhoG[0]

    coulG = tools.get_coulG(cell, mesh=mesh)
    vG = np.multiply(rhoG, coulG)
    v_rs = tools.ifft(vG, mesh).real

    grad = int_gauss_charge_v_rs(
        v_rs,
        comp,
        atm,
        bas,
        env,
        mesh,
        cell.dimension,
        a,
        b,
        max_radius,
    )
    grad *= -1
    t0 = logger.timer(mydf, 'vpploc_part1_nuc_grad', *t0)
    return grad
