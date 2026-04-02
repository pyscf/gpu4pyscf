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

from pyscf import lib, gto
from pyscf.gto.mole import ATOM_OF, ANG_OF, NPRIM_OF, PTR_EXP, PTR_COEFF, BAS_SLOTS
from pyscf.gto.mole import RADI_POWER, SO_TYPE_OF # for ECP
from pyscf.lib import logger
from pyscf.df import addons
from pyscf.dft import gen_grid, radi
from pyscf.dft.gen_grid import _default_rad, _default_ang, LEBEDEV_ORDER, LEBEDEV_NGRID
from pyscf.data.elements import charge as charge_of_element
from pyscf.solvent.pcm import XI
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.dft.gen_grid import Grids
from gpu4pyscf.dft.numint import NumInt
from gpu4pyscf.solvent.pcm import PCM

import cuest.bindings as ce

import numpy as np
import cupy as cp
import time
import types

# flake8: noqa

def cuest_check(
    title,
    return_code
    ):

    if return_code != ce.CuestStatus.CUEST_STATUS_SUCCESS:
        raise RuntimeError(f"{title} failed with code {return_code}")

class WorkspaceDescriptor:
    """
    This helper function needs to implement the following C structure,
    and define a pointer method that returns a pointer to it.

    typedef struct {
        size_t hostBufferSizeInBytes;      ///< Required size of host workspace buffer in bytes
        size_t deviceBufferSizeInBytes;    ///< Required size of device workspace buffer in bytes
    } cuestWorkspaceDescriptor_t;

    This implementation uses Numpy, as the ctypes.data member provides a
    convenient way to access a pointer to the object
    """

    def __init__(
        self,
        *,
        host_buffer_size_in_bytes = 0,
        device_buffer_size_in_bytes = 0,
        ):

        _workspace_descriptor_dtype = np.dtype([
            ("hostBufferSizeInBytes", np.uint64, ),
            ("deviceBufferSizeInBytes", np.uint64, ),
            ], align=True
            )

        self.struct = np.empty(1, dtype=_workspace_descriptor_dtype)
        self.struct['deviceBufferSizeInBytes'] = device_buffer_size_in_bytes
        self.struct['hostBufferSizeInBytes'] = host_buffer_size_in_bytes

    def __str__(
        self,
        ):

        host_size = self.struct['hostBufferSizeInBytes'].item()
        device_size = self.struct['deviceBufferSizeInBytes'].item()
        return f'host buffer size = {host_size} bytes, device buffer size = {device_size} bytes'

    @property
    def pointer(self):
        return self.struct.ctypes.data

class Workspace:
    """
    This helper function needs to implement the following C structure,
    and define a pointer method that returns a pointer to it.

    typedef struct {
        uintptr_t hostBuffer;              ///< Opaque pointer to host-side workspace buffer
        size_t hostBufferSizeInBytes;      ///< Size of host workspace in bytes
        uintptr_t deviceBuffer;            ///< Opaque pointer to device-side (GPU) workspace buffer
        size_t deviceBufferSizeInBytes;    ///< Size of device workspace in bytes
    } cuestWorkspace_t;

    This implementation uses Numpy, as the ctypes.data member provides a
    convenient way to access a pointer to the object
    """
    def __init__(
        self,
        *,
        workspaceDescriptor,
        ):

        _workspace_dtype = np.dtype([
            ("hostBuffer", np.uintp, ),
            ("hostBufferSizeInBytes", np.uint64, ),
            ("deviceBuffer", np.uintp, ),
            ("deviceBufferSizeInBytes", np.uint64, ),
            ], align=True
            )

        host_buffer_size_in_bytes = workspaceDescriptor.struct['hostBufferSizeInBytes'].item()
        device_buffer_size_in_bytes = workspaceDescriptor.struct['deviceBufferSizeInBytes'].item()

        self.struct = np.empty(1, dtype=_workspace_dtype)
        self.struct['deviceBufferSizeInBytes'] = device_buffer_size_in_bytes
        self.struct['hostBufferSizeInBytes'] = host_buffer_size_in_bytes

        if device_buffer_size_in_bytes:
            self.gpu_memory = cp.zeros(device_buffer_size_in_bytes, dtype=cp.int8)
            self.struct['deviceBuffer'] = self.gpu_memory.data
        else:
            self.struct['deviceBuffer'] = 0
            self.gpu_memory = None

        if host_buffer_size_in_bytes:
            self.cpu_memory = np.zeros(host_buffer_size_in_bytes, dtype=np.int8)
            self.struct['hostBuffer'] = self.cpu_memory.ctypes.data
        else:
            self.struct['hostBuffer'] = 0
            self.cpu_memory = None

    @property
    def pointer(self):
        return self.struct.ctypes.data

def get_cupy_maximum_free_bytes():
    # This function provides the theoretical maximum possible amount of memory that can be allocated.
    # It is very likely that cupy cannot actually allocate this amount of memory.

    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()

    pool_used = pool.used_bytes()
    pool_total = pool.total_bytes()
    pool_free = pool_total - pool_used # Memory cached in pool, which is unused and free for cupy allocation

    device_free, device_total = cp.cuda.runtime.memGetInfo()

    return pool_free + device_free

def mol_equal(mol1, mol2):
    if mol1.natm != mol2.natm:
        return False
    if np.max(np.abs(mol1.atom_coords() - mol2.atom_coords())) > 1e-14:
        return False
    if mol1.nao != mol2.nao:
        return False
    if mol1._bas.shape != mol2._bas.shape:
        return False
    if np.max(np.abs(mol1._bas - mol2._bas)) > 1e-14:
        return False
    if mol1._env.shape != mol2._env.shape:
        return False
    if np.max(np.abs(mol1._env - mol2._env)) > 1e-14:
        return False
    return True

def pyscf_mol_to_cuest_shells(mol, cuest_handle, basis_name = "AO"):
    # The returned list of shells needs to be freed outside this function
    atom_index_of_shell = mol._bas[:, ATOM_OF]
    assert np.all(atom_index_of_shell[:-1] <= atom_index_of_shell[1:]), "CuEST only supports shells sorted by atom index as leading order."

    Ls = mol._bas[:, ANG_OF]
    assert len(set(Ls)) != max(Ls), "CuEST doesn't allow basis with holes in the L table (i.e. with S and D but not p orbitals)."

    aoshell_parameters = ce.cuestAOShellParameters()
    cuest_check(f"{basis_name} Shell Parameters Create",
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_AOSHELL_PARAMETERS,
            outParameters=aoshell_parameters,
            )
        )

    shells = []
    for shell_info in mol._bas:
        L = shell_info[ANG_OF]
        n_primitive = shell_info[NPRIM_OF]
        exponents = mol._env[shell_info[PTR_EXP] : shell_info[PTR_EXP] + n_primitive]
        coefficients = mol._env[shell_info[PTR_COEFF] : shell_info[PTR_COEFF] + n_primitive]

        # Where does this crap come from?
        # During basis function normalization, there are two steps: 1. Normalize each primitive, and 2. Normalize the contracted function.
        # The second step is well-defined and the same for pyscf and cuest.
        # The first step, on the other hand, is not. A factor independent of exponents and coefficients can appear.
        # In pyscf, it is, in gamma function representation:
        # coefficients = 2**(0.5*L+1.25) / np.sqrt(scipy.special.gamma(L+1.5)) * exponents**(0.5*L+0.75) * coefficients
        # In cuest, it is, in gamma function representation (the cuest example uses double factorial representation):
        # coefficients = 2**(0.5*L+0.25) * np.sqrt(2*L+1) / np.pi**0.5 / np.sqrt(scipy.special.gamma(L+1.5)) * exponents**(0.5*L+0.75) * coefficients
        # The following factor is the difference.
        coefficients = coefficients.copy()
        coefficients *= (0.5 * np.sqrt(2*L+1) * np.pi**-0.5)

        aoshell_handle = ce.cuestAOShellHandle()
        cuest_check(f"{basis_name} Shell Create",
            ce.cuestAOShellCreate(
                handle = cuest_handle,
                isPure = not mol.cart,
                L = L,
                numPrimitive = n_primitive,
                exponents = exponents,
                coefficients = coefficients,
                parameters = aoshell_parameters,
                outShell = aoshell_handle,
                )
            )
        shells.append(aoshell_handle)

    cuest_check(f"{basis_name} Shell Parameters Destroy",
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_AOSHELL_PARAMETERS,
            parameters=aoshell_parameters,
            )
        )
    return shells

def pyscf_mol_to_cuest_basis(mol, cuest_handle, basis_name = "AO"):
    # The returned aobasis handle and persistent workspace need to be freed outside this function
    log = logger.new_logger(mol, mol.verbose)

    assert basis_name.upper() in ["AO", "AUX"]

    shells = pyscf_mol_to_cuest_shells(mol, cuest_handle)

    persistent_workspace_descriptor = WorkspaceDescriptor()
    temporary_workspace_descriptor = WorkspaceDescriptor()

    aobasis_parameters = ce.cuestAOBasisParameters()
    cuest_check(f'Create {basis_name} Basis Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_AOBASIS_PARAMETERS,
            outParameters=aobasis_parameters,
            )
        )

    n_shells_per_atom = [mol.atom_nshells(i_atom) for i_atom in range(mol.natm)]

    aobasis_handle = ce.cuestAOBasisHandle()
    cuest_check(f'Create {basis_name}Basis Workspace Query',
        ce.cuestAOBasisCreateWorkspaceQuery(
            handle=cuest_handle,
            numAtoms=mol.natm,
            numShellsPerAtom=n_shells_per_atom,
            shells=shells,
            parameters=aobasis_parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outBasis=aobasis_handle,
            )
        )

    log.debug(f"CuEST: {basis_name} Basis Persistent sizes: {persistent_workspace_descriptor}")
    log.debug(f"CuEST: {basis_name} Basis Temporary sizes: {temporary_workspace_descriptor}")

    aobasis_persistent_workspace = Workspace(workspaceDescriptor=persistent_workspace_descriptor)
    aobasis_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    cuest_check(f'Create {basis_name}Basis',
        ce.cuestAOBasisCreate(
            handle=cuest_handle,
            numAtoms=mol.natm,
            numShellsPerAtom=n_shells_per_atom,
            shells=shells,
            parameters=aobasis_parameters,
            persistentWorkspace=aobasis_persistent_workspace.pointer,
            temporaryWorkspace=aobasis_temporary_workspace.pointer,
            outBasis=aobasis_handle,
            )
        )

    del aobasis_temporary_workspace

    for i, shell in enumerate(shells):
        cuest_check(f'Destroy {basis_name} Shell {i+1}',
            ce.cuestAOShellDestroy(
                handle=shell,
                )
            )

    cuest_check(f'Destroy {basis_name} Basis Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_AOBASIS_PARAMETERS,
            parameters=aobasis_parameters,
            )
        )

    # Sanity checks

    aobasis_is_pure = ce.data_int32_t()
    cuest_check(f'Query {basis_name} Basis Is Pure',
        ce.cuestQuery(
            handle=cuest_handle,
            type=ce.CuestType.CUEST_AOBASIS,
            object=aobasis_handle,
            attribute=ce.CuestAOBasisAttributes.CUEST_AOBASIS_IS_PURE,
            attributeValue=aobasis_is_pure,
            )
        )
    assert bool(aobasis_is_pure.value) == (not mol.cart)

    aobasis_num_atom = ce.data_uint64_t()
    cuest_check(f'Query {basis_name} Basis Num Atom',
        ce.cuestQuery(
            handle=cuest_handle,
            type=ce.CuestType.CUEST_AOBASIS,
            object=aobasis_handle,
            attribute=ce.CuestAOBasisAttributes.CUEST_AOBASIS_NUM_ATOM,
            attributeValue=aobasis_num_atom,
            )
        )
    assert aobasis_num_atom.value == mol.natm

    aobasis_num_shell = ce.data_uint64_t()
    cuest_check(f'Query {basis_name} Basis Num Shell',
        ce.cuestQuery(
            handle=cuest_handle,
            type=ce.CuestType.CUEST_AOBASIS,
            object=aobasis_handle,
            attribute=ce.CuestAOBasisAttributes.CUEST_AOBASIS_NUM_SHELL,
            attributeValue=aobasis_num_shell,
            )
        )
    assert aobasis_num_shell.value == mol.nbas

    aobasis_num_ao = ce.data_uint64_t()
    cuest_check(f'Query {basis_name} Basis Num AO',
        ce.cuestQuery(
            handle=cuest_handle,
            type=ce.CuestType.CUEST_AOBASIS,
            object=aobasis_handle,
            attribute=ce.CuestAOBasisAttributes.CUEST_AOBASIS_NUM_AO,
            attributeValue=aobasis_num_ao,
            )
        )
    assert aobasis_num_ao.value == mol.nao

    aobasis_num_cart = ce.data_uint64_t()
    cuest_check(f'Query {basis_name} Basis Num Cart',
        ce.cuestQuery(
            handle=cuest_handle,
            type=ce.CuestType.CUEST_AOBASIS,
            object=aobasis_handle,
            attribute=ce.CuestAOBasisAttributes.CUEST_AOBASIS_NUM_CART,
            attributeValue=aobasis_num_cart,
            )
        )
    assert aobasis_num_cart.value == mol.nao_cart()

    aobasis_num_primitive = ce.data_uint64_t()
    cuest_check(f'Query {basis_name} Basis Num Primitive',
        ce.cuestQuery(
            handle=cuest_handle,
            type=ce.CuestType.CUEST_AOBASIS,
            object=aobasis_handle,
            attribute=ce.CuestAOBasisAttributes.CUEST_AOBASIS_NUM_PRIMITIVE,
            attributeValue=aobasis_num_primitive,
            )
        )
    assert aobasis_num_primitive.value == int(mol._bas[:,NPRIM_OF].sum())

    return aobasis_handle, aobasis_persistent_workspace

def cuest_build_pairlist(mol, cuest_handle, aobasis_handle, threshold_pq = 1e-14):
    # The returned aopairlist handle and persistent workspace need to be freed outside this function
    log = logger.new_logger(mol, mol.verbose)

    aopairlist_parameters = ce.cuestAOPairListParameters()
    cuest_check('Create AOPairList Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_AOPAIRLIST_PARAMETERS,
            outParameters=aopairlist_parameters,
            )
        )

    xyzs = mol.atom_coords(unit = "B").flatten()

    persistent_workspace_descriptor = WorkspaceDescriptor()
    temporary_workspace_descriptor = WorkspaceDescriptor()

    aopairlist_handle = ce.cuestAOPairListHandle()
    cuest_check('Create AOPairList Workspace Query',
        ce.cuestAOPairListCreateWorkspaceQuery(
            handle=cuest_handle,
            basis=aobasis_handle,
            numAtoms=mol.natm,
            xyz=xyzs,
            thresholdPQ=threshold_pq,
            parameters=aopairlist_parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outPairList=aopairlist_handle,
            )
        )

    log.debug(f"CuEST: AO Pair Persistent sizes: {persistent_workspace_descriptor}")
    log.debug(f"CuEST: AO Pair Temporary sizes: {temporary_workspace_descriptor}")

    aopairlist_persistent_workspace = Workspace(workspaceDescriptor=persistent_workspace_descriptor)
    aopairlist_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    cuest_check('Create AOPairList',
        ce.cuestAOPairListCreate(
            handle=cuest_handle,
            basis=aobasis_handle,
            numAtoms=mol.natm,
            xyz=xyzs,
            thresholdPQ=threshold_pq,
            parameters=aopairlist_parameters,
            persistentWorkspace=aopairlist_persistent_workspace.pointer,
            temporaryWorkspace=aopairlist_temporary_workspace.pointer,
            outPairList=aopairlist_handle,
            )
        )

    del aopairlist_temporary_workspace

    cuest_check('Destroy AOPairList Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_AOPAIRLIST_PARAMETERS,
            parameters=aopairlist_parameters,
            )
        )

    return aopairlist_handle, aopairlist_persistent_workspace

def cuest_build_oeintplan(mol, cuest_handle, aobasis_handle, aopairlist_handle):
    # The returned oeintplan handle and persistent workspace need to be freed outside this function
    log = logger.new_logger(mol, mol.verbose)

    oeintplan_parameters = ce.cuestOEIntPlanParameters()
    cuest_check('Create OEIntPlan Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_OEINTPLAN_PARAMETERS,
            outParameters=oeintplan_parameters,
            )
        )

    persistent_workspace_descriptor = WorkspaceDescriptor()
    temporary_workspace_descriptor = WorkspaceDescriptor()

    oeintplan_handle = ce.cuestOEIntPlanHandle()
    cuest_check('Create OEIntPlan Workspace Query',
        ce.cuestOEIntPlanCreateWorkspaceQuery(
            handle=cuest_handle,
            basis=aobasis_handle,
            pairList=aopairlist_handle,
            parameters=oeintplan_parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outPlan=oeintplan_handle,
            )
        )

    log.debug(f"CuEST: One electron integral plan Persistent sizes: {persistent_workspace_descriptor}")
    log.debug(f"CuEST: One electron integral plan Temporary sizes: {temporary_workspace_descriptor}")

    oeintplan_persistent_workspace = Workspace(workspaceDescriptor=persistent_workspace_descriptor)
    oeintplan_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    cuest_check('Create OEIntPlan',
        ce.cuestOEIntPlanCreate(
            handle=cuest_handle,
            basis=aobasis_handle,
            pairList=aopairlist_handle,
            parameters=oeintplan_parameters,
            persistentWorkspace=oeintplan_persistent_workspace.pointer,
            temporaryWorkspace=oeintplan_temporary_workspace.pointer,
            outPlan=oeintplan_handle,
            )
        )

    del oeintplan_temporary_workspace

    cuest_check('Destroy OEIntPlan Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_OEINTPLAN_PARAMETERS,
            parameters=oeintplan_parameters,
            )
        )

    return oeintplan_handle, oeintplan_persistent_workspace

def cuest_compute_overlapint(mol, cuest_handle, oeintplan_handle):
    # The returned S matrix is in cuest order
    log = logger.new_logger(mol, mol.verbose)

    overlapint_device_handle = ce.Pointer()
    compute_overlap_parameters = ce.cuestOverlapComputeParameters()
    cuest_check('Create Overlap Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_OVERLAPCOMPUTE_PARAMETERS,
            outParameters=compute_overlap_parameters,
            )
        )

    temporary_workspace_descriptor = WorkspaceDescriptor()

    overlapint_device_handle = ce.Pointer()
    cuest_check('Compute Overlap Ints Workspace Query',
        ce.cuestOverlapComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=compute_overlap_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outSMatrix=overlapint_device_handle,
            )
        )
    log.debug(f"CuEST: Overlap integral Temporary sizes: {temporary_workspace_descriptor}")

    overlapint_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    overlapint_device = cp.empty([mol.nao, mol.nao], order = "C", dtype = cp.float64)
    overlapint_device_handle.value = overlapint_device.data.ptr

    cuest_check('Compute Overlap Ints',
        ce.cuestOverlapCompute(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=compute_overlap_parameters,
            temporaryWorkspace=overlapint_temporary_workspace.pointer,
            outSMatrix=overlapint_device_handle,
            )
        )

    del overlapint_temporary_workspace

    cuest_check('Destroy Overlap Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_OVERLAPCOMPUTE_PARAMETERS,
            parameters=compute_overlap_parameters,
            )
        )

    return overlapint_device

def cuest_compute_kineticint(mol, cuest_handle, oeintplan_handle):
    # The returned K1e matrix is in cuest order
    log = logger.new_logger(mol, mol.verbose)

    kineticint_device_handle = ce.Pointer()
    compute_kinetic_parameters = ce.cuestKineticComputeParameters()
    cuest_check('Create Kinetic Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_KINETICCOMPUTE_PARAMETERS,
            outParameters=compute_kinetic_parameters,
            )
        )

    temporary_workspace_descriptor = WorkspaceDescriptor()

    kineticint_device_handle = ce.Pointer()
    cuest_check('Compute Kinetic Ints Workspace Query',
        ce.cuestKineticComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=compute_kinetic_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outTMatrix=kineticint_device_handle,
            )
        )
    log.debug(f"CuEST: Kinetic energy integral Temporary sizes: {temporary_workspace_descriptor}")

    kineticint_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    kineticint_device = cp.empty([mol.nao, mol.nao], order = "C", dtype = cp.float64)
    kineticint_device_handle.value = kineticint_device.data.ptr

    cuest_check('Compute Kinetic Ints',
        ce.cuestKineticCompute(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=compute_kinetic_parameters,
            temporaryWorkspace=kineticint_temporary_workspace.pointer,
            outTMatrix=kineticint_device_handle,
            )
        )

    del kineticint_temporary_workspace

    cuest_check('Destroy Kinetic Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_KINETICCOMPUTE_PARAMETERS,
            parameters=compute_kinetic_parameters,
            )
        )

    return kineticint_device

def cuest_compute_potentialint(mol, xyzs_device, Zs_device, cuest_handle, oeintplan_handle):
    # The input xyzs is assumed to be in x1,y1,z1,x2,y2,z2,... order, Zs is assumed to be scaled with -1 already,
    # and the returned V1e matrix is in cuest order
    log = logger.new_logger(mol, mol.verbose)

    xyzs_device_handle = ce.Pointer()
    xyzs_device_handle.value = np.intp(xyzs_device.data.ptr)

    Zs_device_handle = ce.Pointer()
    Zs_device_handle.value = np.intp(Zs_device.data.ptr)

    potential_compute_parameters = ce.cuestPotentialComputeParameters()
    cuest_check('Create Potential Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_POTENTIALCOMPUTE_PARAMETERS,
            outParameters=potential_compute_parameters,
            )
        )

    temporary_workspace_descriptor = WorkspaceDescriptor()

    potentialint_device_handle = ce.Pointer()
    cuest_check('Compute potential Ints Workspace Query',
        ce.cuestPotentialComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=potential_compute_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numCharges=Zs_device.shape[0],
            xyz=xyzs_device_handle,
            q=Zs_device_handle,
            outVMatrix=potentialint_device_handle,
            )
        )
    log.debug(f"CuEST: nuclear attraction integral Temporary sizes: {temporary_workspace_descriptor}")

    potentialint_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    potentialint_device = cp.empty([mol.nao, mol.nao], order = "C", dtype = cp.float64)
    potentialint_device_handle.value = potentialint_device.data.ptr

    cuest_check('Compute potential Ints',
        ce.cuestPotentialCompute(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=potential_compute_parameters,
            temporaryWorkspace=potentialint_temporary_workspace.pointer,
            numCharges=Zs_device.shape[0],
            xyz=xyzs_device_handle,
            q=Zs_device_handle,
            outVMatrix=potentialint_device_handle,
            )
        )

    del potentialint_temporary_workspace

    cuest_check('Destroy Potential Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_POTENTIALCOMPUTE_PARAMETERS,
            parameters=potential_compute_parameters,
            )
        )

    return potentialint_device

def cuest_build_dfintplan(mol, cuest_handle, aobasis_handle, auxbasis_handle, aopairlist_handle, fitting_cutoff = 1e-12):
    # The returned dfintplan handle and persistent workspace need to be freed outside this function
    log = logger.new_logger(mol, mol.verbose)

    dfintplan_parameters = ce.cuestDFIntPlanParameters()
    cuest_check('Create DFIntPlan Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_DFINTPLAN_PARAMETERS,
            outParameters=dfintplan_parameters,
            )
        )

    fitting_relative_conditioning_handle = ce.data_int32_t()
    cuest_check('Query DFIntPlan Params fitting relative conditioning',
        ce.cuestParametersQuery(
            parametersType=ce.CuestParametersType.CUEST_DFINTPLAN_PARAMETERS,
            parameters=dfintplan_parameters,
            attribute=ce.CuestDFIntPlanParametersAttributes.CUEST_DFINTPLAN_PARAMETERS_FITTING_RELATIVE_CONDITIONING,
            attributeValue=fitting_relative_conditioning_handle,
            )
        )

    if fitting_relative_conditioning_handle.value != 0:
        log.info("In default cases the fitting cutoff is defined in relative sense. Set it to absolute sense.")
        fitting_relative_conditioning_handle.value = 0
        cuest_check('Configure DFIntPlan Params fitting relative conditioning',
            ce.cuestParametersConfigure(
                parametersType=ce.CuestParametersType.CUEST_DFINTPLAN_PARAMETERS,
                parameters=dfintplan_parameters,
                attribute=ce.CuestDFIntPlanParametersAttributes.CUEST_DFINTPLAN_PARAMETERS_FITTING_RELATIVE_CONDITIONING,
                attributeValue=fitting_relative_conditioning_handle,
                )
            )

    fitting_cutoff_handle = ce.data_double()
    cuest_check('Query DFIntPlan Params fitting cutoff',
        ce.cuestParametersQuery(
            parametersType=ce.CuestParametersType.CUEST_DFINTPLAN_PARAMETERS,
            parameters=dfintplan_parameters,
            attribute=ce.CuestDFIntPlanParametersAttributes.CUEST_DFINTPLAN_PARAMETERS_FITTING_CUTOFF,
            attributeValue=fitting_cutoff_handle,
            )
        )

    if fitting_cutoff_handle.value != fitting_cutoff:
        log.info(f"CuEST: Default DF metric eigenvalue threshold = {fitting_cutoff_handle.value}")

        fitting_cutoff_handle.value = fitting_cutoff
        cuest_check('Configure DFIntPlan Params fitting cutoff',
            ce.cuestParametersConfigure(
                parametersType=ce.CuestParametersType.CUEST_DFINTPLAN_PARAMETERS,
                parameters=dfintplan_parameters,
                attribute=ce.CuestDFIntPlanParametersAttributes.CUEST_DFINTPLAN_PARAMETERS_FITTING_CUTOFF,
                attributeValue=fitting_cutoff_handle,
                )
            )

        log.info(f"CuEST: Set DF metric eigenvalue threshold to {fitting_cutoff_handle.value}")

    persistent_workspace_descriptor = WorkspaceDescriptor()
    temporary_workspace_descriptor = WorkspaceDescriptor()

    dfintplan_handle = ce.cuestDFIntPlanHandle()
    cuest_check('Create DFIntPlan Workspace Query',
        ce.cuestDFIntPlanCreateWorkspaceQuery(
            handle=cuest_handle,
            primaryBasis=aobasis_handle,
            auxiliaryBasis=auxbasis_handle,
            pairList=aopairlist_handle,
            parameters=dfintplan_parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outPlan=dfintplan_handle,
            )
        )

    log.debug(f"CuEST: DFIntPlan Persistent sizes: {persistent_workspace_descriptor}")
    log.debug(f"CuEST: DFIntPlan Temporary sizes: {temporary_workspace_descriptor}")

    dfintplan_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)
    dfintplan_persistent_workspace = Workspace(workspaceDescriptor=persistent_workspace_descriptor)

    cuest_check('Create DFIntPlan',
        ce.cuestDFIntPlanCreate(
            handle=cuest_handle,
            primaryBasis=aobasis_handle,
            auxiliaryBasis=auxbasis_handle,
            pairList=aopairlist_handle,
            parameters=dfintplan_parameters,
            persistentWorkspace=dfintplan_persistent_workspace.pointer,
            temporaryWorkspace=dfintplan_temporary_workspace.pointer,
            outPlan=dfintplan_handle,
            )
        )

    del dfintplan_temporary_workspace

    cuest_check('Destroy DFIntPlan Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_DFINTPLAN_PARAMETERS,
            parameters=dfintplan_parameters,
            )
        )

    return dfintplan_handle, dfintplan_persistent_workspace

def cuest_compute_coulombmatrix(mol, densitymatrix_device, cuest_handle, dfintplan_handle):
    # The input D is assumed to be in cuest order, and the returned J matrix is in cuest order
    log = logger.new_logger(mol, mol.verbose)

    assert isinstance(densitymatrix_device, cp.ndarray)
    assert densitymatrix_device.shape == (mol.nao, mol.nao)

    densitymatrix_device_handle = ce.Pointer()
    densitymatrix_device_handle.value = densitymatrix_device.data.ptr

    compute_coulomb_matrix_parameters = ce.cuestDFCoulombComputeParameters()
    cuest_check('Create Coulomb Compute Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_DFCOULOMBCOMPUTE_PARAMETERS,
            outParameters=compute_coulomb_matrix_parameters,
            )
        )

    temporary_workspace_descriptor = WorkspaceDescriptor()

    coulombmatrix_device_handle = ce.Pointer()
    cuest_check('Compute Coulomb Ints Workspace Query',
        ce.cuestDFCoulombComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=dfintplan_handle,
            parameters=compute_coulomb_matrix_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            densityMatrix=densitymatrix_device_handle,
            outCoulombMatrix=coulombmatrix_device_handle,
            )
        )

    log.debug(f"CuEST: CoulombInt Temporary sizes: {temporary_workspace_descriptor}")

    coulombint_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    coulombmatrix_device = cp.empty([mol.nao, mol.nao], order = "C", dtype = cp.float64)
    coulombmatrix_device_handle.value = coulombmatrix_device.data.ptr
    cuest_check('Compute Coulomb Ints',
        ce.cuestDFCoulombCompute(
            handle=cuest_handle,
            plan=dfintplan_handle,
            parameters=compute_coulomb_matrix_parameters,
            temporaryWorkspace=coulombint_temporary_workspace.pointer,
            densityMatrix=densitymatrix_device_handle,
            outCoulombMatrix=coulombmatrix_device_handle,
            )
        )

    del coulombint_temporary_workspace

    cuest_check('Destroy Coulomb Compute Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_DFCOULOMBCOMPUTE_PARAMETERS,
            parameters=compute_coulomb_matrix_parameters,
            )
        )

    return coulombmatrix_device

def cuest_compute_exchangematrix(mol, occorbitals_device, cuest_handle, dfintplan_handle, maximum_workspace_bytes,
                                 additional_precision_control_parameters):
    # The input mocc is assumed to be in cuest shape and order, and the returned K matrix is in cuest order
    log = logger.new_logger(mol, mol.verbose)

    assert isinstance(occorbitals_device, cp.ndarray)
    assert occorbitals_device.ndim == 2
    nocc, nao = occorbitals_device.shape
    assert nocc <= nao and nao == mol.nao

    occorbitals_device_handle = ce.Pointer()
    occorbitals_device_handle.value = occorbitals_device.data.ptr

    maximum_workspace_bytes = min(maximum_workspace_bytes, get_cupy_maximum_free_bytes())
    maximum_workspace_descriptor = WorkspaceDescriptor(
        device_buffer_size_in_bytes=maximum_workspace_bytes
        )

    compute_exchange_matrix_parameters = ce.cuestDFSymmetricExchangeComputeParameters()
    cuest_check('Create Exchange Compute Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
            outParameters=compute_exchange_matrix_parameters,
            )
        )

    dfk_int8_slice_count = ce.data_uint64_t()
    dfk_int8_slice_count.value = additional_precision_control_parameters["CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS_INT8_SLICE_COUNT"]
    cuest_check('Configure DF K Params int8 slice count',
        ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
            parameters=compute_exchange_matrix_parameters,
            attribute=ce.CuestDFSymmetricExchangeComputeParametersAttributes.CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS_INT8_SLICE_COUNT,
            attributeValue=dfk_int8_slice_count,
            )
        )
    dfk_int8_modulus_count = ce.data_uint64_t()
    dfk_int8_modulus_count.value = additional_precision_control_parameters["CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS_INT8_MODULUS_COUNT"]
    cuest_check('Configure DF K Params int8 modulus count',
        ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
            parameters=compute_exchange_matrix_parameters,
            attribute=ce.CuestDFSymmetricExchangeComputeParametersAttributes.CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS_INT8_MODULUS_COUNT,
            attributeValue=dfk_int8_modulus_count,
            )
        )

    temporary_workspace_descriptor = WorkspaceDescriptor()

    exchangematrix_device_handle = ce.Pointer()
    cuest_check('Compute Exchange Ints Workspace Query',
        ce.cuestDFSymmetricExchangeComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=dfintplan_handle,
            parameters=compute_exchange_matrix_parameters,
            variableBufferSize=maximum_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numOccupied=nocc,
            coefficientMatrix=occorbitals_device_handle,
            outExchangeMatrix=exchangematrix_device_handle,
            )
        )

    log.debug(f"CuEST: ExchangeInt Temporary sizes: {temporary_workspace_descriptor}")

    exchangeint_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    exchangematrix_device = cp.empty([mol.nao, mol.nao], order = "C", dtype = cp.float64)
    exchangematrix_device_handle.value = exchangematrix_device.data.ptr
    cuest_check('Compute Exchange Ints',
        ce.cuestDFSymmetricExchangeCompute(
            handle=cuest_handle,
            plan=dfintplan_handle,
            parameters=compute_exchange_matrix_parameters,
            variableBufferSize=maximum_workspace_descriptor.pointer,
            temporaryWorkspace=exchangeint_temporary_workspace.pointer,
            numOccupied=nocc,
            coefficientMatrix=occorbitals_device_handle,
            outExchangeMatrix=exchangematrix_device_handle,
            )
        )

    del exchangeint_temporary_workspace

    cuest_check('Destroy Exchange Compute Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
            parameters=compute_exchange_matrix_parameters,
            )
        )

    return exchangematrix_device

def gen_atomic_grids(mol, atom_grid={}, radi_method=radi.gauss_chebyshev,
                     level=3, prune=gen_grid.nwchem_prune, **kwargs):
    '''
        Modified from gpu4pyscf.dft.gen_grid.gen_atomic_grids
        Instead of returning all grid positions and weights of an element, it returns radial grid info,
        plus number of angular grids at each radial node, this is what cuEST expects.

        Returns:
        { "atom symbol" : ( radial grid position in numpy float64 array,
                            radial grid weight in numpy float64 array,
                            number of angular grid for each radial grid in list of integer ) }
    '''
    if isinstance(atom_grid, (list, tuple)):
        atom_grid = dict([(mol.atom_symbol(ia), atom_grid)
                          for ia in range(mol.natm)])
    atom_grids_tab = {}
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)

        if symb not in atom_grids_tab:
            chg = gto.charge(symb)
            if symb in atom_grid:
                n_rad, n_ang = atom_grid[symb]
                if n_ang not in LEBEDEV_NGRID:
                    if n_ang in LEBEDEV_ORDER:
                        logger.warn(mol, 'n_ang %d for atom %d %s is not '
                                    'the supported Lebedev angular grids. '
                                    'Set n_ang to %d', n_ang, ia, symb,
                                    LEBEDEV_ORDER[n_ang])
                        n_ang = LEBEDEV_ORDER[n_ang]
                    else:
                        raise ValueError('Unsupported angular grids %d' % n_ang)
            else:
                n_rad = _default_rad(chg, level)
                n_ang = _default_ang(chg, level)
            rad, dr = radi_method(n_rad, chg, ia, **kwargs)

            # rad_weight = 4*np.pi * rad**2 * dr
            rad_weight = rad**2 * dr

            if callable(prune):
                angs = prune(chg, rad, n_ang)
            else:
                angs = [n_ang] * n_rad

            atom_grids_tab[symb] = (rad, rad_weight, angs)

    return atom_grids_tab

def cuest_build_moleculargrid(mol, grids, cuest_handle):
    # The returned moleculargrid handle and persistent workspace need to be freed outside this function

    assert grids is not None
    assert isinstance(grids, CuESTExtractedGrids)

    atom_grids_tab = gen_atomic_grids(mol, grids.atom_grid, grids.radi_method, grids.level, grids.prune)
    grids = None

    atomgrid_parameters = ce.cuestAtomGridParameters()

    cuest_check('Create AtomGrid Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_ATOMGRID_PARAMETERS,
            outParameters=atomgrid_parameters,
            )
        )

    atomgrids = []
    for i_atom in range(mol.natm):
        radial_nodes, radial_weights, num_angular_points = atom_grids_tab[mol.atom_symbol(i_atom)]
        n_radial_node = len(radial_nodes)

        atomgrid_handle = ce.cuestAtomGridHandle()
        cuest_check(f'Create AtomGrid {i_atom}',
            ce.cuestAtomGridCreate(
                handle=cuest_handle,
                numRadialPoints=n_radial_node,
                radialNodes=radial_nodes,
                radialWeights=radial_weights,
                numAngularPoints=num_angular_points,
                parameters=atomgrid_parameters,
                outAtomGrid=atomgrid_handle,
                )
            )
        atomgrids.append(atomgrid_handle)

    cuest_check('Destroy AtomGrid Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_ATOMGRID_PARAMETERS,
            parameters=atomgrid_parameters,
            )
        )

    moleculargrid_parameters = ce.cuestMolecularGridParameters()

    cuest_check('Create MolecularGrid Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_MOLECULARGRID_PARAMETERS,
            outParameters=moleculargrid_parameters,
            )
        )

    xyzs = mol.atom_coords(unit = "B").flatten()

    persistent_workspace_descriptor = WorkspaceDescriptor()
    temporary_workspace_descriptor = WorkspaceDescriptor()

    moleculargrid_handle = ce.cuestMolecularGridHandle()
    cuest_check('Create MolecularGrid Workspace Query',
        ce.cuestMolecularGridCreateWorkspaceQuery(
            handle=cuest_handle,
            numAtoms=mol.natm,
            atomGrid=atomgrids,
            xyz=xyzs,
            parameters=moleculargrid_parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outGrid=moleculargrid_handle,
            )
        )

    moleculargrid_persistent_workspace = Workspace(workspaceDescriptor=persistent_workspace_descriptor)
    moleculargrid_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    cuest_check('Create MolecularGrid',
        ce.cuestMolecularGridCreate(
            handle=cuest_handle,
            numAtoms=mol.natm,
            atomGrid=atomgrids,
            xyz=xyzs,
            parameters=moleculargrid_parameters,
            persistentWorkspace=moleculargrid_persistent_workspace.pointer,
            temporaryWorkspace=moleculargrid_temporary_workspace.pointer,
            outGrid=moleculargrid_handle,
            )
        )

    del moleculargrid_temporary_workspace

    for atom,grid in enumerate(atomgrids):
        cuest_check(f'Destroy AtomGrid{atom}',
            ce.cuestAtomGridDestroy(
                atomGrid=grid,
                )
            )

    cuest_check('Destroy MolecularGrid Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_MOLECULARGRID_PARAMETERS,
            parameters=moleculargrid_parameters,
            )
        )

    return moleculargrid_handle, moleculargrid_persistent_workspace

def pyscf_xc_to_cuest_functional(xc):
    assert type(xc) is str
    xc = xc.upper()
    xc = xc.replace("-", "")

    pyscf_xc_to_cuest_functional_map = {
        "HF"     : ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_HF    ,
        "B3LYP"  : ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B3LYP1,
        "B3LYP5" : ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B3LYP5,
        "B97"    : ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B97   ,
        "BLYP"   : ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_BLYP  ,
        "M06L"   : ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_M06L  ,
        "PBE"    : ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE   ,
        "PBE0"   : ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE0  ,
        "R2SCAN" : ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_R2SCAN,
        "SVWN"   : ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_SVWN5 ,
        "B97MV"  : ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B97MV ,
    }

    if xc not in pyscf_xc_to_cuest_functional_map:
        raise NotImplementedError(f"PySCF XC functional {xc} is not supported in CuEST. "
                                  f"The supported list of functionals are: {list(pyscf_xc_to_cuest_functional_map.keys())}")

    return pyscf_xc_to_cuest_functional_map[xc]

def check_cuest_pyscf_functional_consistency(mol, xc, numint, cuest_handle, xcintplan_handle):
    omega, alpha, hyb = numint.rsh_and_hybrid_coeff(xc, spin=mol.spin)

    exchange_scale = ce.data_double()
    cuest_check('Query XCIntPlan functional parameter exchange scale',
        ce.cuestQuery(
            handle=cuest_handle,
            type=ce.CuestType.CUEST_XCINTPLAN,
            object=xcintplan_handle,
            attribute=ce.CuestXCIntPlanAttributes.CUEST_XCINTPLAN_EXCHANGE_SCALE,
            attributeValue=exchange_scale,
            )
        )
    assert exchange_scale.value == hyb

    lrc_exchange_scale = ce.data_double()
    cuest_check('Query XCIntPlan functional parameter long-range exchange scale',
        ce.cuestQuery(
            handle=cuest_handle,
            type=ce.CuestType.CUEST_XCINTPLAN,
            object=xcintplan_handle,
            attribute=ce.CuestXCIntPlanAttributes.CUEST_XCINTPLAN_LRC_EXCHANGE_SCALE,
            attributeValue=lrc_exchange_scale,
            )
        )
    assert lrc_exchange_scale.value == (alpha - hyb) # TODO: check if this is correct

    lrc_omega = ce.data_double()
    cuest_check('Query XCIntPlan functional parameter long-range exchange omega',
        ce.cuestQuery(
            handle=cuest_handle,
            type=ce.CuestType.CUEST_XCINTPLAN,
            object=xcintplan_handle,
            attribute=ce.CuestXCIntPlanAttributes.CUEST_XCINTPLAN_LRC_OMEGA,
            attributeValue=lrc_omega,
            )
        )
    if omega == 0:
        assert lrc_omega.value <= 0
    else:
        assert abs(omega) == lrc_omega.value # TODO: check if this is correct

def check_cuest_pyscf_nlc_consistency(mol, xc, numint, cuest_handle, nlc_xcintplan_handle):
    nlc_coefs = numint.nlc_coeff(xc)
    if len(nlc_coefs) != 1:
        raise NotImplementedError('Additive NLC not supported in CuEST wrapper')
    nlc_pars, fac = nlc_coefs[0]

    vv10_scale = ce.data_double()
    cuest_check('Query XCIntPlan nlc functional parameter vv10 scale',
        ce.cuestQuery(
            handle=cuest_handle,
            type=ce.CuestType.CUEST_XCINTPLAN,
            object=nlc_xcintplan_handle,
            attribute=ce.CuestXCIntPlanAttributes.CUEST_XCINTPLAN_VV10_SCALE,
            attributeValue=vv10_scale,
            )
        )
    assert fac == vv10_scale.value

    vv10_c = ce.data_double()
    cuest_check('Query XCIntPlan nlc functional parameter vv10 c parameter',
        ce.cuestQuery(
            handle=cuest_handle,
            type=ce.CuestType.CUEST_XCINTPLAN,
            object=nlc_xcintplan_handle,
            attribute=ce.CuestXCIntPlanAttributes.CUEST_XCINTPLAN_VV10_C,
            attributeValue=vv10_c,
            )
        )
    assert nlc_pars[1] == vv10_c.value

    vv10_b = ce.data_double()
    cuest_check('Query XCIntPlan nlc functional parameter vv10 b parameter',
        ce.cuestQuery(
            handle=cuest_handle,
            type=ce.CuestType.CUEST_XCINTPLAN,
            object=nlc_xcintplan_handle,
            attribute=ce.CuestXCIntPlanAttributes.CUEST_XCINTPLAN_VV10_B,
            attributeValue=vv10_b,
            )
        )
    assert nlc_pars[0] == vv10_b.value

def cuest_build_xcintplan(mol, cuest_handle, aobasis_handle, moleculargrid_handle, functional):
    # The returned xcintplan handle and persistent workspace need to be freed outside this function
    log = logger.new_logger(mol, mol.verbose)

    xcintplan_parameters = ce.cuestXCIntPlanParameters()
    cuest_check('Create XCIntPlan Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_XCINTPLAN_PARAMETERS,
            outParameters=xcintplan_parameters,
            )
        )

    persistent_workspace_descriptor = WorkspaceDescriptor()
    temporary_workspace_descriptor = WorkspaceDescriptor()

    xcintplan_handle = ce.cuestXCIntPlanHandle()
    cuest_check('Create XCIntPlan Workspace Query',
        ce.cuestXCIntPlanCreateWorkspaceQuery(
            handle=cuest_handle,
            basis=aobasis_handle,
            grid=moleculargrid_handle,
            functional=functional,
            parameters=xcintplan_parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outPlan=xcintplan_handle,
            )
        )

    log.debug(f"CuEST: XC integral plan Persistent sizes: {persistent_workspace_descriptor}")
    log.debug(f"CuEST: XC integral plan Temporary sizes: {temporary_workspace_descriptor}")

    xcintplan_persistent_workspace = Workspace(workspaceDescriptor=persistent_workspace_descriptor)
    xcintplan_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    cuest_check('Create XCIntPlan',
        ce.cuestXCIntPlanCreate(
            handle=cuest_handle,
            basis=aobasis_handle,
            grid=moleculargrid_handle,
            functional=functional,
            parameters=xcintplan_parameters,
            persistentWorkspace=xcintplan_persistent_workspace.pointer,
            temporaryWorkspace=xcintplan_temporary_workspace.pointer,
            outPlan=xcintplan_handle,
            )
        )

    del xcintplan_temporary_workspace

    cuest_check('Destroy XCIntPlan Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_XCINTPLAN_PARAMETERS,
            parameters=xcintplan_parameters,
            )
        )

    return xcintplan_handle, xcintplan_persistent_workspace

def cuest_compute_xcpotential(mol, Cocc_list, cuest_handle, xcintplan_handle, maximum_workspace_bytes):
    # The input mocc is assumed to be in cuest shape and order, and the returned vxc matrix is in cuest order
    log = logger.new_logger(mol, mol.verbose)

    maximum_workspace_bytes = min(maximum_workspace_bytes, get_cupy_maximum_free_bytes())
    maximum_workspace_descriptor = WorkspaceDescriptor(
        host_buffer_size_in_bytes=0,
        device_buffer_size_in_bytes=maximum_workspace_bytes,
        )

    if len(Cocc_list) == 1:
        Cocc_device = Cocc_list[0]
        assert isinstance(Cocc_device, cp.ndarray)
        assert Cocc_device.ndim == 2
        nocc, nao = Cocc_device.shape
        assert nocc <= nao and nao == mol.nao

        Cocc_device_handle = ce.Pointer()
        Cocc_device_handle.value = Cocc_device.data.ptr

        compute_xc_potential_parameters = ce.cuestXCPotentialRKSComputeParameters()
        cuest_check('XCPotentialRKS Parameters Create',
            ce.cuestParametersCreate(
                parametersType=ce.CuestParametersType.CUEST_XCPOTENTIALRKSCOMPUTE_PARAMETERS,
                outParameters=compute_xc_potential_parameters,
                )
            )

        temporary_workspace_descriptor = WorkspaceDescriptor()

        Vxc_device_handle = ce.Pointer()
        Exc = ce.data_double()
        cuest_check('XCPotentialRKSCompute Workspace Query',
            ce.cuestXCPotentialRKSComputeWorkspaceQuery(
                handle=cuest_handle,
                plan=xcintplan_handle,
                variableBufferSize=maximum_workspace_descriptor.pointer,
                temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
                parameters=compute_xc_potential_parameters,
                numOccupied=nocc,
                coefficientMatrix=Cocc_device_handle,
                outXCEnergy=Exc,
                outXCPotentialMatrix=Vxc_device_handle,
                )
            )

        log.debug(f"CuEST: XCPotentialRKS Temporary sizes: {temporary_workspace_descriptor}")

        Vxc_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

        Vxc_device = cp.empty([mol.nao, mol.nao], order = "C", dtype = cp.float64)
        Vxc_device_handle.value = Vxc_device.data.ptr

        cuest_check('XCPotentialRKSCompute',
            ce.cuestXCPotentialRKSCompute(
                handle=cuest_handle,
                plan=xcintplan_handle,
                variableBufferSize=maximum_workspace_descriptor.pointer,
                temporaryWorkspace=Vxc_workspace.pointer,
                parameters=compute_xc_potential_parameters,
                numOccupied=nocc,
                coefficientMatrix=Cocc_device_handle,
                outXCEnergy=Exc,
                outXCPotentialMatrix=Vxc_device_handle,
                )
            )

        del Vxc_workspace

        cuest_check('Destroy XCPotentialRKS Parameters',
            ce.cuestParametersDestroy(
                parametersType=ce.CuestParametersType.CUEST_XCPOTENTIALRKSCOMPUTE_PARAMETERS,
                parameters=compute_xc_potential_parameters,
                )
            )
    elif len(Cocc_list) == 2:
        Cocca_device = Cocc_list[0]
        assert isinstance(Cocca_device, cp.ndarray)
        assert Cocca_device.ndim == 2
        nocca, nao = Cocca_device.shape
        assert nocca <= nao and nao == mol.nao

        Coccb_device = Cocc_list[1]
        assert isinstance(Coccb_device, cp.ndarray)
        assert Coccb_device.ndim == 2
        noccb, nao = Coccb_device.shape
        assert noccb <= nao and nao == mol.nao

        Cocca_device_handle = ce.Pointer()
        Cocca_device_handle.value = Cocca_device.data.ptr
        Coccb_device_handle = ce.Pointer()
        Coccb_device_handle.value = Coccb_device.data.ptr

        compute_xc_potential_parameters = ce.cuestXCPotentialUKSComputeParameters()
        cuest_check('XCPotentialUKS Parameters Create',
            ce.cuestParametersCreate(
                parametersType=ce.CuestParametersType.CUEST_XCPOTENTIALUKSCOMPUTE_PARAMETERS,
                outParameters=compute_xc_potential_parameters,
                )
            )

        temporary_workspace_descriptor = WorkspaceDescriptor()

        Vxca_device_handle = ce.Pointer()
        Vxcb_device_handle = ce.Pointer()
        Exc = ce.data_double()
        cuest_check('XCPotentialUKSCompute Workspace Query',
            ce.cuestXCPotentialUKSComputeWorkspaceQuery(
                handle=cuest_handle,
                plan=xcintplan_handle,
                variableBufferSize=maximum_workspace_descriptor.pointer,
                temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
                parameters=compute_xc_potential_parameters,
                numOccupiedAlpha=nocca,
                numOccupiedBeta=noccb,
                coefficientMatrixAlpha=Cocca_device_handle,
                coefficientMatrixBeta=Coccb_device_handle,
                outXCEnergy=Exc,
                outXCPotentialMatrixAlpha=Vxca_device_handle,
                outXCPotentialMatrixBeta=Vxcb_device_handle,
                )
            )

        log.debug(f"CuEST: XCPotentialUKS Temporary sizes: {temporary_workspace_descriptor}")

        Vxc_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

        Vxc_device = cp.empty([2, mol.nao, mol.nao], order = "C", dtype = cp.float64)
        Vxca_device_handle.value = Vxc_device[0].data.ptr
        Vxcb_device_handle.value = Vxc_device[1].data.ptr

        cuest_check('XCPotentialUKSCompute',
            ce.cuestXCPotentialUKSCompute(
                handle=cuest_handle,
                plan=xcintplan_handle,
                variableBufferSize=maximum_workspace_descriptor.pointer,
                temporaryWorkspace=Vxc_workspace.pointer,
                parameters=compute_xc_potential_parameters,
                numOccupiedAlpha=nocca,
                numOccupiedBeta=noccb,
                coefficientMatrixAlpha=Cocca_device_handle,
                coefficientMatrixBeta=Coccb_device_handle,
                outXCEnergy=Exc,
                outXCPotentialMatrixAlpha=Vxca_device_handle,
                outXCPotentialMatrixBeta=Vxcb_device_handle,
                )
            )

        del Vxc_workspace

        cuest_check('Destroy XCPotentialUKS Parameters',
            ce.cuestParametersDestroy(
                parametersType=ce.CuestParametersType.CUEST_XCPOTENTIALUKSCOMPUTE_PARAMETERS,
                parameters=compute_xc_potential_parameters,
                )
            )
    else:
        raise ValueError("Incorrect Cocc_list size for cuest_compute_xcpotential() function")

    return Exc.value, Vxc_device

def cuest_compute_nlcpotential(mol, Cocc_list, cuest_handle, nlc_xcintplan_handle, maximum_workspace_bytes):
    # The input mocc is assumed to be in cuest shape and order, and the returned vnlc matrix is in cuest order
    log = logger.new_logger(mol, mol.verbose)

    maximum_workspace_bytes = min(maximum_workspace_bytes, get_cupy_maximum_free_bytes())
    maximum_workspace_descriptor = WorkspaceDescriptor(
        host_buffer_size_in_bytes=0,
        device_buffer_size_in_bytes=maximum_workspace_bytes,
        )

    vv10_scale = ce.data_double()
    cuest_check('Query XCIntPlan nlc functional parameter vv10 scale',
        ce.cuestQuery(
            handle=cuest_handle,
            type=ce.CuestType.CUEST_XCINTPLAN,
            object=nlc_xcintplan_handle,
            attribute=ce.CuestXCIntPlanAttributes.CUEST_XCINTPLAN_VV10_SCALE,
            attributeValue=vv10_scale,
            )
        )

    vv10_C = ce.data_double()
    cuest_check('Query XCIntPlan nlc functional parameter vv10 c parameter',
        ce.cuestQuery(
            handle=cuest_handle,
            type=ce.CuestType.CUEST_XCINTPLAN,
            object=nlc_xcintplan_handle,
            attribute=ce.CuestXCIntPlanAttributes.CUEST_XCINTPLAN_VV10_C,
            attributeValue=vv10_C,
            )
        )

    vv10_b = ce.data_double()
    cuest_check('Query XCIntPlan nlc functional parameter vv10 b parameter',
        ce.cuestQuery(
            handle=cuest_handle,
            type=ce.CuestType.CUEST_XCINTPLAN,
            object=nlc_xcintplan_handle,
            attribute=ce.CuestXCIntPlanAttributes.CUEST_XCINTPLAN_VV10_B,
            attributeValue=vv10_b,
            )
        )

    if len(Cocc_list) == 1:
        Cocc_device = Cocc_list[0]
        assert isinstance(Cocc_device, cp.ndarray)
        assert Cocc_device.ndim == 2
        nocc, nao = Cocc_device.shape
        assert nocc <= nao and nao == mol.nao

        Cocc_device_handle = ce.Pointer()
        Cocc_device_handle.value = Cocc_device.data.ptr

        nonlocal_xc_compute_parameters = ce.cuestNonlocalXCPotentialRKSComputeParameters()
        cuest_check('NonlocalXC Parameters Create',
            ce.cuestParametersCreate(
                parametersType=ce.CuestParametersType.CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS,
                outParameters=nonlocal_xc_compute_parameters,
                )
            )

        cuest_check('VV10 b Parameter Configure',
            ce.cuestParametersConfigure(
                parametersType=ce.CuestParametersType.CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS,
                parameters=nonlocal_xc_compute_parameters,
                attribute=ce.CuestNonlocalXCPotentialRKSComputeParametersAttributes.CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS_VV10_B,
                attributeValue=vv10_b,
                )
            )
        cuest_check('VV10 C Parameter Configure',
            ce.cuestParametersConfigure(
                parametersType=ce.CuestParametersType.CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS,
                parameters=nonlocal_xc_compute_parameters,
                attribute=ce.CuestNonlocalXCPotentialRKSComputeParametersAttributes.CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS_VV10_C,
                attributeValue=vv10_C,
                )
            )
        cuest_check('VV10 Scale Parameter Configure',
            ce.cuestParametersConfigure(
                parametersType=ce.CuestParametersType.CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS,
                parameters=nonlocal_xc_compute_parameters,
                attribute=ce.CuestNonlocalXCPotentialRKSComputeParametersAttributes.CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS_VV10_SCALE,
                attributeValue=vv10_scale,
                )
            )

        temporary_workspace_descriptor = WorkspaceDescriptor()

        Vxc_device_handle = ce.Pointer()
        Exc = ce.data_double()
        cuest_check('NonlocalXCPotentialRKSCompute Workspace Query',
            ce.cuestNonlocalXCPotentialRKSComputeWorkspaceQuery(
                handle=cuest_handle,
                plan=nlc_xcintplan_handle,
                variableBufferSize=maximum_workspace_descriptor.pointer,
                temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
                parameters=nonlocal_xc_compute_parameters,
                numOccupied=nocc,
                coefficientMatrix=Cocc_device_handle,
                outXCEnergy=Exc,
                outXCPotentialMatrix=Vxc_device_handle,
                )
            )

        log.debug(f"CuEST: NonlocalXCPotentialRKS Temporary sizes: {temporary_workspace_descriptor}")

        Vxc_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

        Vxc_device = cp.empty([mol.nao, mol.nao], order = "C", dtype = cp.float64)
        Vxc_device_handle.value = Vxc_device.data.ptr

        cuest_check('NonlocalXCPotentialRKSCompute',
            ce.cuestNonlocalXCPotentialRKSCompute(
                handle=cuest_handle,
                plan=nlc_xcintplan_handle,
                variableBufferSize=maximum_workspace_descriptor.pointer,
                temporaryWorkspace=Vxc_workspace.pointer,
                parameters=nonlocal_xc_compute_parameters,
                numOccupied=nocc,
                coefficientMatrix=Cocc_device_handle,
                outXCEnergy=Exc,
                outXCPotentialMatrix=Vxc_device_handle,
                )
            )

        del Vxc_workspace

        cuest_check('Destroy NonlocalXC Parameters',
            ce.cuestParametersDestroy(
                parametersType=ce.CuestParametersType.CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS,
                parameters=nonlocal_xc_compute_parameters,
                )
            )
    elif len(Cocc_list) == 2:
        Cocca_device = Cocc_list[0]
        assert isinstance(Cocca_device, cp.ndarray)
        assert Cocca_device.ndim == 2
        nocca, nao = Cocca_device.shape
        assert nocca <= nao and nao == mol.nao

        Coccb_device = Cocc_list[1]
        assert isinstance(Coccb_device, cp.ndarray)
        assert Coccb_device.ndim == 2
        noccb, nao = Coccb_device.shape
        assert noccb <= nao and nao == mol.nao

        Cocca_device_handle = ce.Pointer()
        Cocca_device_handle.value = Cocca_device.data.ptr
        Coccb_device_handle = ce.Pointer()
        Coccb_device_handle.value = Coccb_device.data.ptr

        nonlocal_xc_compute_parameters = ce.cuestNonlocalXCPotentialUKSComputeParameters()
        cuest_check('NonlocalXC UKS Parameters Create',
            ce.cuestParametersCreate(
                parametersType=ce.CuestParametersType.CUEST_NONLOCALXCPOTENTIALUKSCOMPUTE_PARAMETERS,
                outParameters=nonlocal_xc_compute_parameters,
                )
            )

        cuest_check('VV10 b UKS Parameter Configure',
            ce.cuestParametersConfigure(
                parametersType=ce.CuestParametersType.CUEST_NONLOCALXCPOTENTIALUKSCOMPUTE_PARAMETERS,
                parameters=nonlocal_xc_compute_parameters,
                attribute=ce.CuestNonlocalXCPotentialUKSComputeParametersAttributes.CUEST_NONLOCALXCPOTENTIALUKSCOMPUTE_PARAMETERS_VV10_B,
                attributeValue=vv10_b,
                )
            )
        cuest_check('VV10 C UKS Parameter Configure',
            ce.cuestParametersConfigure(
                parametersType=ce.CuestParametersType.CUEST_NONLOCALXCPOTENTIALUKSCOMPUTE_PARAMETERS,
                parameters=nonlocal_xc_compute_parameters,
                attribute=ce.CuestNonlocalXCPotentialUKSComputeParametersAttributes.CUEST_NONLOCALXCPOTENTIALUKSCOMPUTE_PARAMETERS_VV10_C,
                attributeValue=vv10_C,
                )
            )
        cuest_check('VV10 Scale UKS Parameter Configure',
            ce.cuestParametersConfigure(
                parametersType=ce.CuestParametersType.CUEST_NONLOCALXCPOTENTIALUKSCOMPUTE_PARAMETERS,
                parameters=nonlocal_xc_compute_parameters,
                attribute=ce.CuestNonlocalXCPotentialUKSComputeParametersAttributes.CUEST_NONLOCALXCPOTENTIALUKSCOMPUTE_PARAMETERS_VV10_SCALE,
                attributeValue=vv10_scale,
                )
            )

        temporary_workspace_descriptor = WorkspaceDescriptor()

        Vxc_device_handle = ce.Pointer()
        Exc = ce.data_double()
        cuest_check('NonlocalXCPotentialUKSCompute Workspace Query',
            ce.cuestNonlocalXCPotentialUKSComputeWorkspaceQuery(
                handle=cuest_handle,
                plan=nlc_xcintplan_handle,
                variableBufferSize=maximum_workspace_descriptor.pointer,
                temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
                parameters=nonlocal_xc_compute_parameters,
                numOccupiedAlpha=nocca,
                numOccupiedBeta=noccb,
                coefficientMatrixAlpha=Cocca_device_handle,
                coefficientMatrixBeta=Coccb_device_handle,
                outXCEnergy=Exc,
                outXCPotentialMatrix=Vxc_device_handle,
                )
            )

        log.debug(f"CuEST: NonlocalXCPotentialUKS Temporary sizes: {temporary_workspace_descriptor}")

        Vxc_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

        Vxc_device = cp.empty([mol.nao, mol.nao], order = "C", dtype = cp.float64)
        Vxc_device_handle.value = Vxc_device.data.ptr

        cuest_check('NonlocalXCPotentialUKSCompute',
            ce.cuestNonlocalXCPotentialUKSCompute(
                handle=cuest_handle,
                plan=nlc_xcintplan_handle,
                variableBufferSize=maximum_workspace_descriptor.pointer,
                temporaryWorkspace=Vxc_workspace.pointer,
                parameters=nonlocal_xc_compute_parameters,
                numOccupiedAlpha=nocca,
                numOccupiedBeta=noccb,
                coefficientMatrixAlpha=Cocca_device_handle,
                coefficientMatrixBeta=Coccb_device_handle,
                outXCEnergy=Exc,
                outXCPotentialMatrix=Vxc_device_handle,
                )
            )

        del Vxc_workspace

        cuest_check('NonlocalXC Parameters Destroy',
            ce.cuestParametersDestroy(
                parametersType=ce.CuestParametersType.CUEST_NONLOCALXCPOTENTIALUKSCOMPUTE_PARAMETERS,
                parameters=nonlocal_xc_compute_parameters,
                )
            )
    else:
        raise ValueError("Incorrect Cocc_list size for cuest_compute_nlcpotential() function")

    return Exc.value, Vxc_device

def cuest_build_pcmintplan(mol, cuest_handle, oeintplan_handle, with_solvent):
    # The returned pcmintplan handle and persistent workspace need to be freed outside this function
    log = logger.new_logger(mol, mol.verbose)

    assert with_solvent is not None
    assert isinstance(with_solvent, CuESTExtractedPCM)

    pcm_int_plan_parameters = ce.cuestPCMIntPlanParameters()
    cuest_check('Create PCM Int Plan Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_PCMINTPLAN_PARAMETERS,
            outParameters=pcm_int_plan_parameters,
            )
        )

    pcm_switching_function_cutoff_handle = ce.data_double()
    pcm_switching_function_cutoff_handle.value = 1e-8 # Cutoff for surface points’ iSWIG switching function, below which the point is considered inactive.
    cuest_check('Configure PCM Int Plan switching function cutoff',
        ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_PCMINTPLAN_PARAMETERS,
            parameters=pcm_int_plan_parameters,
            attribute=ce.CuestPCMIntPlanParametersAttributes.CUEST_PCMINTPLAN_PARAMETERS_CUTOFF,
            attributeValue=pcm_switching_function_cutoff_handle,
            )
        )

    if with_solvent.method.upper() in ['C-PCM', 'CPCM']:
        pcm_x_factor_in_f_epsilon = 0.0
    elif with_solvent.method.upper() == 'COSMO':
        pcm_x_factor_in_f_epsilon = 0.5
    else:
        raise NotImplementedError(f"PCM method = {with_solvent.method} not supported by CuEST yet.")

    pcm_x_factor_in_f_epsilon_handle = ce.data_double()
    pcm_x_factor_in_f_epsilon_handle.value = pcm_x_factor_in_f_epsilon # PCM prefactor f=(epsilon-1)/(epsilon+x), where x==0 gives CPCM and x==1/2 corresponds to COSMO.
    cuest_check('Configure PCM Int Plan switching function cutoff',
        ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_PCMINTPLAN_PARAMETERS,
            parameters=pcm_int_plan_parameters,
            attribute=ce.CuestPCMIntPlanParametersAttributes.CUEST_PCMINTPLAN_PARAMETERS_X_PREFACTOR,
            attributeValue=pcm_x_factor_in_f_epsilon_handle,
            )
        )

    persistent_workspace_descriptor = WorkspaceDescriptor()
    temporary_workspace_descriptor = WorkspaceDescriptor()

    n_lebedev_grid_point = LEBEDEV_ORDER[with_solvent.lebedev_order]
    n_grids_per_atom = [n_lebedev_grid_point] * mol.natm
    element_index = [charge_of_element(e) for e in mol.elements]
    atomic_radii = [with_solvent.radii_table[chg] for chg in element_index]
    nuclear_charges = np.asarray(mol.atom_charges(), dtype = np.float64)
    zeta = XI[n_lebedev_grid_point]
    zeta_per_atom = [zeta] * mol.natm
    epsilon = float(with_solvent.eps)

    pcmintplan_handle = ce.cuestPCMIntPlanHandle()
    cuest_check('Create PCMIntPlan Workspace Query',
        ce.cuestPCMIntPlanCreateWorkspaceQuery(
            handle=cuest_handle,
            intPlan=oeintplan_handle,
            parameters=pcm_int_plan_parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numAngularPointsPerAtom=n_grids_per_atom,
            epsilon=epsilon,
            zetas=zeta_per_atom,
            atomicRadii=atomic_radii,
            effectiveNuclearCharges=nuclear_charges,
            outPlan=pcmintplan_handle,
            )
        )

    log.debug(f"CuEST: PCMIntPlan Persistent sizes: {persistent_workspace_descriptor}")
    log.debug(f"CuEST: PCMIntPlan Temporary sizes: {temporary_workspace_descriptor}")

    pcmintplan_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)
    pcmintplan_persistent_workspace = Workspace(workspaceDescriptor=persistent_workspace_descriptor)

    cuest_check('Create PCMIntPlan',
        ce.cuestPCMIntPlanCreate(
            handle=cuest_handle,
            intPlan=oeintplan_handle,
            parameters=pcm_int_plan_parameters,
            persistentWorkspace=pcmintplan_persistent_workspace.pointer,
            temporaryWorkspace=pcmintplan_temporary_workspace.pointer,
            numAngularPointsPerAtom=n_grids_per_atom,
            epsilon=epsilon,
            zetas=zeta_per_atom,
            atomicRadii=atomic_radii,
            effectiveNuclearCharges=nuclear_charges,
            outPlan=pcmintplan_handle,
            )
        )

    del pcmintplan_temporary_workspace

    cuest_check('Destroy PCM Int Plan Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_PCMINTPLAN_PARAMETERS,
            parameters=pcm_int_plan_parameters,
            )
        )

    n_pcm_grid = cuest_get_n_pcm_grid(cuest_handle, pcmintplan_handle)
    assert n_pcm_grid > 0
    log.debug(f"CuEST: PCM total number of surface grid = {n_pcm_grid}")

    return pcmintplan_handle, pcmintplan_persistent_workspace

def cuest_get_n_pcm_grid(cuest_handle, pcmintplan_handle):
    n_pcm_grid_point_handle = ce.data_uint64_t()
    cuest_check('Query PCM Int Plan number of grid point',
        ce.cuestQuery(
            handle=cuest_handle,
            type=ce.CuestType.CUEST_PCMINTPLAN,
            object=pcmintplan_handle,
            attribute=ce.CuestPCMIntPlanAttributes.CUEST_PCMINTPLAN_NUM_POINT,
            attributeValue=n_pcm_grid_point_handle,
            )
        )
    return n_pcm_grid_point_handle.value

def cuest_compute_pcmpotential(mol, densitymatrix_device, q_guess, cuest_handle, pcmintplan_handle):
    # The input D is assumed to be in cuest order, and the returned V_pcm matrix is in cuest order
    log = logger.new_logger(mol, mol.verbose)

    assert isinstance(densitymatrix_device, cp.ndarray)
    assert densitymatrix_device.shape == (mol.nao, mol.nao)

    densitymatrix_device_handle = ce.Pointer()
    densitymatrix_device_handle.value = densitymatrix_device.data.ptr

    n_pcm_grid = cuest_get_n_pcm_grid(cuest_handle, pcmintplan_handle)
    assert isinstance(q_guess, cp.ndarray)
    assert q_guess.shape == (n_pcm_grid,)
    q_guess_device_handle = ce.Pointer()
    q_guess_device_handle.value = q_guess.data.ptr

    pcm_compute_parameters = ce.cuestPCMPotentialComputeParameters()
    cuest_check('Create PCM Compute Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_PCMPOTENTIALCOMPUTE_PARAMETERS,
            outParameters=pcm_compute_parameters,
            )
        )

    pcm_charge_convergence_threshold_handle = ce.data_double()
    pcm_charge_convergence_threshold_handle.value = 1e-14 # Convergence threshold (max absolute residual value) for PCM charges computation
    cuest_check('Configure PCM charge convergence threshold',
        ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_PCMPOTENTIALCOMPUTE_PARAMETERS,
            parameters=pcm_compute_parameters,
            attribute=ce.CuestPCMPotentialComputeParametersAttributes.CUEST_PCMPOTENTIALCOMPUTE_PARAMETERS_CONVERGENCE_THRESHOLD,
            attributeValue=pcm_charge_convergence_threshold_handle,
            )
        )

    pcm_charge_max_iteration_handle = ce.data_uint64_t()
    pcm_charge_max_iteration_handle.value = 100 # Maximum number of preconditioned conjugate gradient iterations for PCM charges computation
    cuest_check('Configure PCM charge max iteration',
        ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_PCMPOTENTIALCOMPUTE_PARAMETERS,
            parameters=pcm_compute_parameters,
            attribute=ce.CuestPCMPotentialComputeParametersAttributes.CUEST_PCMPOTENTIALCOMPUTE_PARAMETERS_MAX_ITERATIONS,
            attributeValue=pcm_charge_max_iteration_handle,
            )
        )

    temporary_workspace_descriptor = WorkspaceDescriptor()

    pcm_results_handle = ce.cuestPCMResultsHandle()
    cuest_check('Create PCM Result Handle',
        ce.cuestResultsCreate(
            resultsType=ce.CuestResultsType.CUEST_PCM_RESULTS,
            outResults=pcm_results_handle,
            )
        )

    q_out = cp.empty(n_pcm_grid, dtype = cp.float64)
    q_out_device_handle = ce.Pointer()
    q_out_device_handle.value = q_out.data.ptr

    pcm_V_matrix_device = cp.empty([mol.nao, mol.nao], order = "C", dtype = cp.float64)
    pcm_V_matrix_device_handle = ce.Pointer()
    pcm_V_matrix_device_handle.value = pcm_V_matrix_device.data.ptr

    cuest_check('Compute PCM Potential Workspace Query',
        ce.cuestPCMPotentialComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=pcmintplan_handle,
            parameters=pcm_compute_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            densityMatrix=densitymatrix_device_handle,
            inQ=q_guess_device_handle,
            outQ=q_out_device_handle,
            outPCMResults=pcm_results_handle,
            outPCMPotentialMatrix=pcm_V_matrix_device_handle,
            )
        )

    log.debug(f"CuEST: PCM Potential Temporary sizes: {temporary_workspace_descriptor}")

    pcm_int_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    cuest_check('Compute PCM Potential',
        ce.cuestPCMPotentialCompute(
            handle=cuest_handle,
            plan=pcmintplan_handle,
            parameters=pcm_compute_parameters,
            temporaryWorkspace=pcm_int_temporary_workspace.pointer,
            densityMatrix=densitymatrix_device_handle,
            inQ=q_guess_device_handle,
            outQ=q_out_device_handle,
            outPCMResults=pcm_results_handle,
            outPCMPotentialMatrix=pcm_V_matrix_device_handle,
            )
        )

    del pcm_int_temporary_workspace

    cuest_check('Destroy PCM Compute Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_PCMPOTENTIALCOMPUTE_PARAMETERS,
            parameters=pcm_compute_parameters,
            )
        )

    Epcm_handle = ce.data_double()
    cuest_check('Query PCM result energy',
        ce.cuestResultsQuery(
            resultsType=ce.CuestResultsType.CUEST_PCM_RESULTS,
            results=pcm_results_handle,
            attribute=ce.CuestPCMResultAttributes.CUEST_PCMRESULT_PCM_DIELECTRIC_ENERGY,
            attributeValue=Epcm_handle,
            )
        )
    Epcm = Epcm_handle.value

    converged_handle = ce.data_int32_t()
    cuest_check('Query PCM result converged',
        ce.cuestResultsQuery(
            resultsType=ce.CuestResultsType.CUEST_PCM_RESULTS,
            results=pcm_results_handle,
            attribute=ce.CuestPCMResultAttributes.CUEST_PCMRESULT_CONVERGED,
            attributeValue=converged_handle,
            )
        )
    assert converged_handle.value

    residual_handle = ce.data_double()
    cuest_check('Query PCM result residual',
        ce.cuestResultsQuery(
            resultsType=ce.CuestResultsType.CUEST_PCM_RESULTS,
            results=pcm_results_handle,
            attribute=ce.CuestPCMResultAttributes.CUEST_PCMRESULT_CONVERGED_RESIDUAL,
            attributeValue=residual_handle,
            )
        )
    assert residual_handle.value < 1e-14

    cuest_check('Destroy PCM Result Handle',
        ce.cuestResultsDestroy(
            resultsType=ce.CuestResultsType.CUEST_PCM_RESULTS,
            results=pcm_results_handle,
            )
        )

    return Epcm, pcm_V_matrix_device, q_out

def pyscf_ecpbas_to_cuest_ecpatoms(mol, cuest_handle):
    # The returned list of ecpatoms needs to be freed outside this function
    _ecpbas = mol._ecpbas
    _ecpbas = _ecpbas[_ecpbas[:, gto.SO_TYPE_OF] == 0]

    assert len(_ecpbas) > 0

    # Split according to atom index
    atom_index = _ecpbas[:, ATOM_OF]
    _ecpbas = _ecpbas[np.argsort(atom_index, kind = "stable"), :]
    atom_index = _ecpbas[:, ATOM_OF]
    atom_first_index = np.flatnonzero(atom_index[1:] != atom_index[:-1]) + 1
    atom_first_index = np.concatenate(([0], atom_first_index, [_ecpbas.shape[0]]))
    _ecpbas = [_ecpbas[atom_first_index[i] : atom_first_index[i+1], :] for i in range(len(atom_first_index) - 1)]

    ecpIndices = np.array([_ecpbas[i][0, ATOM_OF] for i in range(len(_ecpbas))])

    # Then split further according to L
    for i_ecp_atom in range(len(_ecpbas)):
        _ecpbas_atom = _ecpbas[i_ecp_atom]

        L = _ecpbas_atom[:, ANG_OF]
        _ecpbas_atom = _ecpbas_atom[np.argsort(L, kind = "stable"), :]
        L = _ecpbas_atom[:, ANG_OF]
        Lmax = max(L) + 1
        L_first_index = np.flatnonzero(L[1:] != L[:-1]) + 1
        L_first_index = np.concatenate(([0], L_first_index, [_ecpbas_atom.shape[0]]))
        _ecpbas_atom = [_ecpbas_atom[L_first_index[i] : L_first_index[i+1], :] for i in range(len(L_first_index) - 1)]

        assert np.all(_ecpbas_atom[0][:, ANG_OF] == -1)
        _ecpbas_atom[0][:, ANG_OF] = Lmax

        _ecpbas[i_ecp_atom] = _ecpbas_atom

    def extract_ecp_shell(_ecpbas_shell, _env):
        assert _ecpbas_shell.shape[1] == BAS_SLOTS
        i_atom = _ecpbas_shell[0, ATOM_OF]
        assert np.all(i_atom == _ecpbas_shell[:, ATOM_OF])
        L = _ecpbas_shell[0, ANG_OF]
        assert np.all(L == _ecpbas_shell[:, ANG_OF])

        numPrimitive = 0
        radialPowers = []
        coefficients = []
        exponents = []
        for _ecpbas_item in _ecpbas_shell:
            nprim = _ecpbas_item[NPRIM_OF]
            radi_power = _ecpbas_item[RADI_POWER]
            exp = _env[_ecpbas_item[PTR_EXP] : _ecpbas_item[PTR_EXP] + nprim]
            coeff = _env[_ecpbas_item[PTR_COEFF] : _ecpbas_item[PTR_COEFF] + nprim]

            numPrimitive += nprim
            radialPowers.extend([radi_power] * nprim)
            coefficients.extend(list(coeff))
            exponents.extend(list(exp))

        return int(i_atom), int(L), int(numPrimitive), np.array(radialPowers), np.array(coefficients), np.array(exponents)

    ecpshell_parameters = ce.cuestECPShellParameters()
    cuest_check('Create ECP Shell Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_ECPSHELL_PARAMETERS,
            outParameters=ecpshell_parameters,
            )
        )

    ecp_shells_pack = []
    ecp_top_shell_pack = []

    for _ecpbas_atom in _ecpbas:
        _ecpbas_shell = _ecpbas_atom[0]
        i_atom, L, numPrimitive, radialPowers, coefficients, exponents = extract_ecp_shell(_ecpbas_shell, mol._env)

        top_shell_handle = ce.cuestECPShellHandle()
        cuest_check("ECP Shell Create",
            ce.cuestECPShellCreate(
                handle = cuest_handle,
                L = L,
                numPrimitive = numPrimitive,
                radialPowers = radialPowers,
                coefficients = coefficients,
                exponents = exponents,
                parameters = ecpshell_parameters,
                outECPShell = top_shell_handle,
                )
            )
        ecp_top_shell_pack.append(top_shell_handle)

        ecp_shells = []
        for i_ecp_shell in range(1, len(_ecpbas_atom)):
            _ecpbas_shell = _ecpbas_atom[i_ecp_shell]
            i_atom, L, numPrimitive, radialPowers, coefficients, exponents = extract_ecp_shell(_ecpbas_shell, mol._env)

            shell_handle = ce.cuestECPShellHandle()
            cuest_check("ECP Shell Create",
                ce.cuestECPShellCreate(
                    handle = cuest_handle,
                    L = L,
                    numPrimitive = numPrimitive,
                    radialPowers = radialPowers,
                    coefficients = coefficients,
                    exponents = exponents,
                    parameters = ecpshell_parameters,
                    outECPShell = shell_handle,
                    )
                )
            ecp_shells.append(shell_handle)

        ecp_shells_pack.append(ecp_shells)

    cuest_check('Destroy ECP Shell Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_ECPSHELL_PARAMETERS,
            parameters=ecpshell_parameters,
            )
        )

    ### Shell above, atom below

    ecpatom_parameters = ce.cuestECPShellParameters()
    cuest_check('Create ECP Atom Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_ECPATOM_PARAMETERS,
            outParameters=ecpatom_parameters,
            )
        )

    ecpAtoms = []
    for i_ecp_atom in range(len(_ecpbas)):
        _ecpbas_atom = _ecpbas[i_ecp_atom]

        i_atom = _ecpbas_atom[0][0, ATOM_OF]
        element_charge = charge_of_element(mol.elements[i_atom])
        charge_with_ecp = mol.atom_charge(i_atom)
        assert np.round(charge_with_ecp) == charge_with_ecp
        numElectrons = int(element_charge - charge_with_ecp)
        assert numElectrons > 0

        ecp_atom_handle = ce.cuestECPAtomHandle()
        cuest_check("ECP Atom Create",
            ce.cuestECPAtomCreate(
                handle = cuest_handle,
                numElectrons = numElectrons,
                numShells = len(ecp_shells_pack[i_ecp_atom]),
                shells = ecp_shells_pack[i_ecp_atom],
                topShell = ecp_top_shell_pack[i_ecp_atom],
                parameters = ecpatom_parameters,
                outECPAtom = ecp_atom_handle,
                )
            )
        ecpAtoms.append(ecp_atom_handle)

    for top_shell_handle in ecp_top_shell_pack:
        cuest_check('Destroy ECP Shell',
            ce.cuestECPShellDestroy(
                handle = top_shell_handle,
                )
            )
    for ecp_shells in ecp_shells_pack:
        for shell_handle in ecp_shells:
            cuest_check('Destroy ECP Shell',
                ce.cuestECPShellDestroy(
                    handle = shell_handle,
                    )
                )

    cuest_check('Destroy ECP Atom Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_ECPATOM_PARAMETERS,
            parameters=ecpatom_parameters,
            )
        )

    return ecpAtoms, ecpIndices

def cuest_build_ecpintplan(mol, cuest_handle, aobasis_handle):
    # The returned ecpintplan handle and persistent workspace need to be freed outside this function
    log = logger.new_logger(mol, mol.verbose)

    ecpAtoms, ecpIndices = pyscf_ecpbas_to_cuest_ecpatoms(mol, cuest_handle)
    xyzs = mol.atom_coords(unit = "B").flatten()

    ecpintplan_parameters = ce.cuestOEIntPlanParameters()
    cuest_check('Create ECP Integral Plan Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_ECPINTPLAN_PARAMETERS,
            outParameters=ecpintplan_parameters,
            )
        )

    persistent_workspace_descriptor = WorkspaceDescriptor()
    temporary_workspace_descriptor = WorkspaceDescriptor()

    ecpintplan_handle = ce.cuestECPIntPlanHandle()
    cuest_check('Create ECPIntPlan Workspace Query',
        ce.cuestECPIntPlanCreateWorkspaceQuery(
            handle=cuest_handle,
            basis=aobasis_handle,
            xyz=xyzs,
            numECPAtoms=len(ecpIndices),
            activeIndices=ecpIndices,
            activeAtoms=ecpAtoms,
            parameters=ecpintplan_parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outPlan=ecpintplan_handle,
            )
        )

    log.debug(f"CuEST: ECP integral plan Persistent sizes: {persistent_workspace_descriptor}")
    log.debug(f"CuEST: ECP integral plan Temporary sizes: {temporary_workspace_descriptor}")

    ecpintplan_persistent_workspace = Workspace(workspaceDescriptor=persistent_workspace_descriptor)
    ecpintplan_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    cuest_check('Create ECPIntPlan',
        ce.cuestECPIntPlanCreate(
            handle=cuest_handle,
            basis=aobasis_handle,
            xyz=xyzs,
            numECPAtoms=len(ecpIndices),
            activeIndices=ecpIndices,
            activeAtoms=ecpAtoms,
            parameters=ecpintplan_parameters,
            persistentWorkspace=ecpintplan_persistent_workspace.pointer,
            temporaryWorkspace=ecpintplan_temporary_workspace.pointer,
            outPlan=ecpintplan_handle,
            )
        )

    for ecp_atom_handle in ecpAtoms:
        cuest_check('Destroy ECP Atom',
            ce.cuestECPAtomDestroy(
                handle=ecp_atom_handle,
                )
            )

    del ecpintplan_temporary_workspace

    cuest_check('Destroy ECP Integral Plan Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_ECPINTPLAN_PARAMETERS,
            parameters=ecpintplan_parameters,
            )
        )

    return ecpintplan_handle, ecpintplan_persistent_workspace

def cuest_compute_ecpint(mol, cuest_handle, ecpintplan_handle, maximum_workspace_bytes):
    # The returned V_ecp matrix is in cuest order
    log = logger.new_logger(mol, mol.verbose)

    ecpcompute_parameters = ce.cuestDFCoulombComputeParameters()
    cuest_check('Create ECP Compute Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_ECPCOMPUTE_PARAMETERS,
            outParameters=ecpcompute_parameters,
            )
        )

    maximum_workspace_bytes = min(maximum_workspace_bytes, get_cupy_maximum_free_bytes())
    maximum_workspace_descriptor = WorkspaceDescriptor(
        device_buffer_size_in_bytes=maximum_workspace_bytes,
        )

    temporary_workspace_descriptor = WorkspaceDescriptor()

    ecp_matrix_device_handle = ce.Pointer()
    cuest_check('Compute ECP integral Workspace Query',
        ce.cuestECPComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=ecpintplan_handle,
            parameters=ecpcompute_parameters,
            variableBufferSize=maximum_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outECPMatrix=ecp_matrix_device_handle,
            )
        )

    log.debug(f"CuEST: ECP integral Temporary sizes: {temporary_workspace_descriptor}")

    ecp_int_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    ecp_matrix_device = cp.empty([mol.nao, mol.nao], order = "C", dtype = cp.float64)
    ecp_matrix_device_handle.value = ecp_matrix_device.data.ptr

    cuest_check('Compute ECP integral',
        ce.cuestECPCompute(
            handle=cuest_handle,
            plan=ecpintplan_handle,
            parameters=ecpcompute_parameters,
            variableBufferSize=maximum_workspace_descriptor.pointer,
            temporaryWorkspace=ecp_int_temporary_workspace.pointer,
            outECPMatrix=ecp_matrix_device_handle,
            )
        )

    del ecp_int_temporary_workspace

    cuest_check('Destroy ECP Compute Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_ECPCOMPUTE_PARAMETERS,
            parameters=ecpcompute_parameters,
            )
        )

    return ecp_matrix_device

def cuest_compute_overlap_gradient(mol, densitymatrix_device, cuest_handle, oeintplan_handle):
    # The input dme is assumed to be in cuest shape and order
    log = logger.new_logger(mol, mol.verbose)

    densitymatrix_device_handle = ce.Pointer()
    densitymatrix_device_handle.value = densitymatrix_device.data.ptr

    compute_overlap_gradient_parameters = ce.cuestOverlapDerivativeComputeParameters()
    cuest_check('Create Overlap Grad Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_OVERLAPDERIVATIVECOMPUTE_PARAMETERS,
            outParameters=compute_overlap_gradient_parameters,
            )
        )

    temporary_workspace_descriptor = WorkspaceDescriptor()

    overlapgrad_device_handle = ce.Pointer()
    cuest_check('Compute Overlap Grad Workspace Query',
        ce.cuestOverlapDerivativeComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=compute_overlap_gradient_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            densityMatrix=densitymatrix_device_handle,
            outGradient=overlapgrad_device_handle,
            )
        )
    log.debug(f"CuEST: Overlap integral gradient Temporary sizes: {temporary_workspace_descriptor}")

    overlapgrad_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    overlapgrad_device = cp.empty([mol.natm, 3], order = "C", dtype = cp.float64)
    overlapgrad_device_handle.value = overlapgrad_device.data.ptr

    cuest_check('Compute Overlap Derivative',
        ce.cuestOverlapDerivativeCompute(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=compute_overlap_gradient_parameters,
            temporaryWorkspace=overlapgrad_temporary_workspace.pointer,
            densityMatrix=densitymatrix_device_handle,
            outGradient=overlapgrad_device_handle,
            )
        )

    del overlapgrad_temporary_workspace

    cuest_check('Destroy Overlap Grad Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_OVERLAPDERIVATIVECOMPUTE_PARAMETERS,
            parameters=compute_overlap_gradient_parameters,
            )
        )

    return overlapgrad_device

def cuest_compute_kinetic_gradient(mol, densitymatrix_device, cuest_handle, oeintplan_handle):
    # The input dme is assumed to be in cuest shape and order
    log = logger.new_logger(mol, mol.verbose)

    densitymatrix_device_handle = ce.Pointer()
    densitymatrix_device_handle.value = densitymatrix_device.data.ptr

    compute_kinetic_gradient_parameters = ce.cuestKineticDerivativeComputeParameters()
    cuest_check('Create Kinetic Grad Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_KINETICDERIVATIVECOMPUTE_PARAMETERS,
            outParameters=compute_kinetic_gradient_parameters,
            )
        )

    temporary_workspace_descriptor = WorkspaceDescriptor()

    kineticgrad_device_handle = ce.Pointer()
    cuest_check('Compute Kinetic Grad Workspace Query',
        ce.cuestKineticDerivativeComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=compute_kinetic_gradient_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            densityMatrix=densitymatrix_device_handle,
            outGradient=kineticgrad_device_handle,
            )
        )
    log.debug(f"CuEST: Kinetic energy integral gradient Temporary sizes: {temporary_workspace_descriptor}")

    kineticgrad_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    kineticgrad_device = cp.empty([mol.natm, 3], order = "C", dtype = cp.float64)
    kineticgrad_device_handle.value = kineticgrad_device.data.ptr

    cuest_check('Compute Kinetic Derivative',
        ce.cuestKineticDerivativeCompute(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=compute_kinetic_gradient_parameters,
            temporaryWorkspace=kineticgrad_temporary_workspace.pointer,
            densityMatrix=densitymatrix_device_handle,
            outGradient=kineticgrad_device_handle,
            )
        )

    del kineticgrad_temporary_workspace

    cuest_check('Destroy Kinetic Grad Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_KINETICDERIVATIVECOMPUTE_PARAMETERS,
            parameters=compute_kinetic_gradient_parameters,
            )
        )

    return kineticgrad_device

def cuest_compute_potential_gradeint(mol, densitymatrix_device, xyzs_device, Zs_device, cuest_handle, oeintplan_handle):
    # The input xyzs is assumed to be in x1,y1,z1,x2,y2,z2,... order, Zs is assumed to be scaled with -1 already
    log = logger.new_logger(mol, mol.verbose)

    xyzs_device_handle = ce.Pointer()
    xyzs_device_handle.value = np.intp(xyzs_device.data.ptr)

    Zs_device_handle = ce.Pointer()
    Zs_device_handle.value = np.intp(Zs_device.data.ptr)
    n_charge = Zs_device.shape[0]

    densitymatrix_device_handle = ce.Pointer()
    densitymatrix_device_handle.value = densitymatrix_device.data.ptr

    compute_potential_gradient_parameters = ce.cuestPotentialDerivativeComputeParameters()
    cuest_check('Create Potential Grad Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_POTENTIALDERIVATIVECOMPUTE_PARAMETERS,
            outParameters=compute_potential_gradient_parameters,
            )
        )

    temporary_workspace_descriptor = WorkspaceDescriptor()

    potential_orbital_gradient_device_handle = ce.Pointer()
    potential_pointcharge_gradient_device_handle = ce.Pointer()
    cuest_check('Compute potential Grad Workspace Query',
        ce.cuestPotentialDerivativeComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=compute_potential_gradient_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numCharges=n_charge,
            xyz=xyzs_device_handle,
            q=Zs_device_handle,
            densityMatrix=densitymatrix_device_handle,
            outBasisGradient=potential_orbital_gradient_device_handle,
            outChargeGradient=potential_pointcharge_gradient_device_handle,
            )
        )
    log.debug(f"CuEST: nuclear attraction integral gradient Temporary sizes: {temporary_workspace_descriptor}")

    potentialgrad_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    potential_orbital_gradient_device = cp.empty([mol.natm, 3], order = "C", dtype = cp.float64)
    potential_orbital_gradient_device_handle.value = potential_orbital_gradient_device.data.ptr
    potential_pointcharge_gradient_device = cp.empty([n_charge, 3], order = "C", dtype = cp.float64)
    potential_pointcharge_gradient_device_handle.value = potential_pointcharge_gradient_device.data.ptr

    cuest_check('Compute potential Grad',
        ce.cuestPotentialDerivativeCompute(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=compute_potential_gradient_parameters,
            temporaryWorkspace=potentialgrad_temporary_workspace.pointer,
            numCharges=n_charge,
            xyz=xyzs_device_handle,
            q=Zs_device_handle,
            densityMatrix=densitymatrix_device_handle,
            outBasisGradient=potential_orbital_gradient_device_handle,
            outChargeGradient=potential_pointcharge_gradient_device_handle,
            )
        )

    del potentialgrad_temporary_workspace

    cuest_check('Destroy Potential Grad Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_POTENTIALDERIVATIVECOMPUTE_PARAMETERS,
            parameters=compute_potential_gradient_parameters,
            )
        )

    return potential_orbital_gradient_device, potential_pointcharge_gradient_device

def cuest_compute_coulomb_exchange_gradient(mol, densitymatrix_device, occorbitals_device, nocc, k_factor, cuest_handle, dfintplan_handle, maximum_workspace_bytes):
    # The input D and mocc is assumed to be in cuest order and format
    # k_factor is multiplied by -0.5 in this function, so it should not include the factor of -0.5 at invocation
    log = logger.new_logger(mol, mol.verbose)

    # The additional 0.5 factor is to cancel the factor of 2 on dm
    n_dm = len(nocc)
    assert n_dm in (1, 2)
    j_factor = 0.5
    k_factor *= -0.25 * n_dm

    assert isinstance(densitymatrix_device, cp.ndarray)
    assert densitymatrix_device.shape == (mol.nao, mol.nao)
    assert isinstance(occorbitals_device, cp.ndarray)
    assert occorbitals_device.shape == (sum(nocc), mol.nao)

    densitymatrix_device_handle = ce.Pointer()
    densitymatrix_device_handle.value = densitymatrix_device.data.ptr

    occorbitals_device_handle = ce.Pointer()
    occorbitals_device_handle.value = occorbitals_device.data.ptr

    maximum_workspace_bytes = min(maximum_workspace_bytes, get_cupy_maximum_free_bytes())
    maximum_workspace_descriptor = WorkspaceDescriptor(
        host_buffer_size_in_bytes=0,
        device_buffer_size_in_bytes=maximum_workspace_bytes
        )

    compute_JK_gradient_parameters = ce.cuestDFSymmetricDerivativeComputeParameters()
    cuest_check('Create JK Grad Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS,
            outParameters=compute_JK_gradient_parameters,
            )
        )

    temporary_workspace_descriptor = WorkspaceDescriptor()

    JKgrad_device_handle = ce.Pointer()
    cuest_check('Compute JK Gradient Workspace Query',
        ce.cuestDFSymmetricDerivativeComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=dfintplan_handle,
            parameters=compute_JK_gradient_parameters,
            variableBufferSize=maximum_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            densityScale=j_factor, # The scale factor for the J term
            densityMatrix=densitymatrix_device_handle,
            coefficientScale=k_factor, # This should be further scaled by the HF X scale factor for hybrid DFT
            numCoefficientMatrices=n_dm,
            numOccupied=nocc,
            coefficientMatrices=occorbitals_device_handle,
            outGradient=JKgrad_device_handle,
            )
        )

    log.debug(f"CuEST: DFSymmetricDerivative Temporary sizes: {temporary_workspace_descriptor}")

    JKgrad_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    JKgrad_device = cp.empty([mol.natm, 3], order = "C", dtype = cp.float64)
    JKgrad_device_handle.value = JKgrad_device.data.ptr

    cuest_check('Compute JK Gradient',
        ce.cuestDFSymmetricDerivativeCompute(
            handle=cuest_handle,
            plan=dfintplan_handle,
            parameters=compute_JK_gradient_parameters,
            variableBufferSize=maximum_workspace_descriptor.pointer,
            temporaryWorkspace=JKgrad_temporary_workspace.pointer,
            densityScale=j_factor, # The scale factor for the J term
            densityMatrix=densitymatrix_device_handle,
            coefficientScale=k_factor, # This should be further scaled by the HF X scale factor for hybrid DFT
            numCoefficientMatrices=n_dm,
            numOccupied=nocc,
            coefficientMatrices=occorbitals_device_handle,
            outGradient=JKgrad_device_handle,
            )
        )

    del JKgrad_temporary_workspace

    cuest_check('Destroy JK Grad Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS,
            parameters=compute_JK_gradient_parameters,
            )
        )

    return JKgrad_device

def cuest_compute_xc_gradient(mol, Cocc_list, cuest_handle, xcintplan_handle, maximum_workspace_bytes):
    # The input mocc is assumed to be in cuest shape and order
    log = logger.new_logger(mol, mol.verbose)

    maximum_workspace_bytes = min(maximum_workspace_bytes, get_cupy_maximum_free_bytes())
    maximum_workspace_descriptor = WorkspaceDescriptor(
        host_buffer_size_in_bytes=0,
        device_buffer_size_in_bytes=maximum_workspace_bytes,
        )

    if len(Cocc_list) == 1:
        Cocc_device = Cocc_list[0]
        assert isinstance(Cocc_device, cp.ndarray)
        assert Cocc_device.ndim == 2
        nocc, nao = Cocc_device.shape
        assert nocc <= nao and nao == mol.nao

        Cocc_device_handle = ce.Pointer()
        Cocc_device_handle.value = Cocc_device.data.ptr

        xc_derivative_rks_compute_parameters = ce.cuestXCDerivativeRKSComputeParameters()
        cuest_check('XCDerivativeRKSCompute Parameters Create',
            ce.cuestParametersCreate(
                parametersType=ce.CuestParametersType.CUEST_XCDERIVATIVERKSCOMPUTE_PARAMETERS,
                outParameters=xc_derivative_rks_compute_parameters,
                )
            )

        temporary_workspace_descriptor = WorkspaceDescriptor()

        Vxcgrad_device_handle = ce.Pointer()
        cuest_check('XCDerivativeRKSCompute Workspace Query',
            ce.cuestXCDerivativeRKSComputeWorkspaceQuery(
                handle=cuest_handle,
                plan=xcintplan_handle,
                parameters=xc_derivative_rks_compute_parameters,
                variableBufferSize=maximum_workspace_descriptor.pointer,
                temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
                numOccupied=nocc,
                coefficientMatrix=Cocc_device_handle,
                outGradient=Vxcgrad_device_handle,
                )
            )

        log.debug(f"CuEST: XCDerivativeRKS Temporary sizes: {temporary_workspace_descriptor}")

        Vxc_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

        Vxcgrad_device = cp.empty([mol.natm, 3], order = "C", dtype = cp.float64)
        Vxcgrad_device_handle.value = Vxcgrad_device.data.ptr

        cuest_check('XCDerivativeRKSCompute',
            ce.cuestXCDerivativeRKSCompute(
                handle=cuest_handle,
                plan=xcintplan_handle,
                parameters=xc_derivative_rks_compute_parameters,
                variableBufferSize=maximum_workspace_descriptor.pointer,
                temporaryWorkspace=Vxc_workspace.pointer,
                numOccupied=nocc,
                coefficientMatrix=Cocc_device_handle,
                outGradient=Vxcgrad_device_handle,
                )
            )

        del Vxc_workspace

        cuest_check('Destroy XCDerivativeRKSCompute Parameters',
            ce.cuestParametersDestroy(
                parametersType=ce.CuestParametersType.CUEST_XCDERIVATIVERKSCOMPUTE_PARAMETERS,
                parameters=xc_derivative_rks_compute_parameters,
                )
            )
    elif len(Cocc_list) == 2:
        Cocca_device = Cocc_list[0]
        assert isinstance(Cocca_device, cp.ndarray)
        assert Cocca_device.ndim == 2
        nocca, nao = Cocca_device.shape
        assert nocca <= nao and nao == mol.nao

        Coccb_device = Cocc_list[1]
        assert isinstance(Coccb_device, cp.ndarray)
        assert Coccb_device.ndim == 2
        noccb, nao = Coccb_device.shape
        assert noccb <= nao and nao == mol.nao

        Cocca_device_handle = ce.Pointer()
        Cocca_device_handle.value = Cocca_device.data.ptr
        Coccb_device_handle = ce.Pointer()
        Coccb_device_handle.value = Coccb_device.data.ptr

        xc_derivative_uks_compute_parameters = ce.cuestXCDerivativeUKSComputeParameters()
        cuest_check('XCDerivativeUKSCompute Parameters Create',
            ce.cuestParametersCreate(
                parametersType=ce.CuestParametersType.CUEST_XCDERIVATIVEUKSCOMPUTE_PARAMETERS,
                outParameters=xc_derivative_uks_compute_parameters,
                )
            )

        temporary_workspace_descriptor = WorkspaceDescriptor()

        Vxcgrad_device_handle = ce.Pointer()
        cuest_check('XCDerivativeUKSCompute Workspace Query',
            ce.cuestXCDerivativeUKSComputeWorkspaceQuery(
                handle=cuest_handle,
                plan=xcintplan_handle,
                parameters=xc_derivative_uks_compute_parameters,
                variableBufferSize=maximum_workspace_descriptor.pointer,
                temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
                numOccupiedAlpha=nocca,
                numOccupiedBeta=noccb,
                coefficientMatrixAlpha=Cocca_device_handle,
                coefficientMatrixBeta=Coccb_device_handle,
                outGradient=Vxcgrad_device_handle,
                )
            )

        log.debug(f"CuEST: XCDerivativeUKS Temporary sizes: {temporary_workspace_descriptor}")

        Vxc_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

        Vxcgrad_device = cp.empty([mol.natm, 3], order = "C", dtype = cp.float64)
        Vxcgrad_device_handle.value = Vxcgrad_device.data.ptr

        cuest_check('XCDerivativeUKSCompute',
            ce.cuestXCDerivativeUKSCompute(
                handle=cuest_handle,
                plan=xcintplan_handle,
                parameters=xc_derivative_uks_compute_parameters,
                variableBufferSize=maximum_workspace_descriptor.pointer,
                temporaryWorkspace=Vxc_workspace.pointer,
                numOccupiedAlpha=nocca,
                numOccupiedBeta=noccb,
                coefficientMatrixAlpha=Cocca_device_handle,
                coefficientMatrixBeta=Coccb_device_handle,
                outGradient=Vxcgrad_device_handle,
                )
            )

        del Vxc_workspace

        cuest_check('Destroy XCDerivativeUKSCompute Parameters',
            ce.cuestParametersDestroy(
                parametersType=ce.CuestParametersType.CUEST_XCDERIVATIVEUKSCOMPUTE_PARAMETERS,
                parameters=xc_derivative_uks_compute_parameters,
                )
            )
    else:
        raise ValueError("Incorrect Cocc_list size for cuest_compute_xc_gradient() function")

    return Vxcgrad_device

def cuest_compute_nlc_gradient(mol, Cocc_list, cuest_handle, nlc_xcintplan_handle, maximum_workspace_bytes):
    # The input mocc is assumed to be in cuest shape and order
    log = logger.new_logger(mol, mol.verbose)

    maximum_workspace_bytes = min(maximum_workspace_bytes, get_cupy_maximum_free_bytes())
    maximum_workspace_descriptor = WorkspaceDescriptor(
        host_buffer_size_in_bytes=0,
        device_buffer_size_in_bytes=maximum_workspace_bytes,
        )

    vv10_scale = ce.data_double()
    cuest_check('Query XCIntPlan nlc functional parameter vv10 scale',
        ce.cuestQuery(
            handle=cuest_handle,
            type=ce.CuestType.CUEST_XCINTPLAN,
            object=nlc_xcintplan_handle,
            attribute=ce.CuestXCIntPlanAttributes.CUEST_XCINTPLAN_VV10_SCALE,
            attributeValue=vv10_scale,
            )
        )

    vv10_C = ce.data_double()
    cuest_check('Query XCIntPlan nlc functional parameter vv10 c parameter',
        ce.cuestQuery(
            handle=cuest_handle,
            type=ce.CuestType.CUEST_XCINTPLAN,
            object=nlc_xcintplan_handle,
            attribute=ce.CuestXCIntPlanAttributes.CUEST_XCINTPLAN_VV10_C,
            attributeValue=vv10_C,
            )
        )

    vv10_b = ce.data_double()
    cuest_check('Query XCIntPlan nlc functional parameter vv10 b parameter',
        ce.cuestQuery(
            handle=cuest_handle,
            type=ce.CuestType.CUEST_XCINTPLAN,
            object=nlc_xcintplan_handle,
            attribute=ce.CuestXCIntPlanAttributes.CUEST_XCINTPLAN_VV10_B,
            attributeValue=vv10_b,
            )
        )

    if len(Cocc_list) == 1:
        Cocc_device = Cocc_list[0]
        assert isinstance(Cocc_device, cp.ndarray)
        assert Cocc_device.ndim == 2
        nocc, nao = Cocc_device.shape
        assert nocc <= nao and nao == mol.nao

        Cocc_device_handle = ce.Pointer()
        Cocc_device_handle.value = Cocc_device.data.ptr

        nonlocal_xc_compute_parameters = ce.cuestNonlocalXCDerivativeRKSComputeParameters()
        cuest_check('NonlocalXC Gradient Parameters Create',
            ce.cuestParametersCreate(
                parametersType=ce.CuestParametersType.CUEST_NONLOCALXCDERIVATIVERKSCOMPUTE_PARAMETERS,
                outParameters=nonlocal_xc_compute_parameters,
                )
            )

        cuest_check('VV10 b Gradient Parameter Configure',
            ce.cuestParametersConfigure(
                parametersType=ce.CuestParametersType.CUEST_NONLOCALXCDERIVATIVERKSCOMPUTE_PARAMETERS,
                parameters=nonlocal_xc_compute_parameters,
                attribute=ce.CuestNonlocalXCDerivativeRKSComputeParametersAttributes.CUEST_NONLOCALXCDERIVATIVERKSCOMPUTE_PARAMETERS_VV10_B,
                attributeValue=vv10_b,
                )
            )
        cuest_check('VV10 C Gradient Parameter Configure',
            ce.cuestParametersConfigure(
                parametersType=ce.CuestParametersType.CUEST_NONLOCALXCDERIVATIVERKSCOMPUTE_PARAMETERS,
                parameters=nonlocal_xc_compute_parameters,
                attribute=ce.CuestNonlocalXCDerivativeRKSComputeParametersAttributes.CUEST_NONLOCALXCDERIVATIVERKSCOMPUTE_PARAMETERS_VV10_C,
                attributeValue=vv10_C,
                )
            )
        cuest_check('VV10 Scale Gradient Parameter Configure',
            ce.cuestParametersConfigure(
                parametersType=ce.CuestParametersType.CUEST_NONLOCALXCDERIVATIVERKSCOMPUTE_PARAMETERS,
                parameters=nonlocal_xc_compute_parameters,
                attribute=ce.CuestNonlocalXCDerivativeRKSComputeParametersAttributes.CUEST_NONLOCALXCDERIVATIVERKSCOMPUTE_PARAMETERS_VV10_SCALE,
                attributeValue=vv10_scale,
                )
            )

        temporary_workspace_descriptor = WorkspaceDescriptor()

        Vxcgrad_device_handle = ce.Pointer()
        cuest_check('NonlocalXCDerivativeRKSCompute Workspace Query',
            ce.cuestNonlocalXCDerivativeRKSComputeWorkspaceQuery(
                handle=cuest_handle,
                plan=nlc_xcintplan_handle,
                variableBufferSize=maximum_workspace_descriptor.pointer,
                temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
                parameters=nonlocal_xc_compute_parameters,
                numOccupied=nocc,
                coefficientMatrix=Cocc_device_handle,
                outGradient=Vxcgrad_device_handle,
                )
            )

        log.debug(f"CuEST: NonlocalXCDerivativeRKS Temporary sizes: {temporary_workspace_descriptor}")

        Vxc_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

        Vxcgrad_device = cp.empty([mol.natm, 3], order = "C", dtype = cp.float64)
        Vxcgrad_device_handle.value = Vxcgrad_device.data.ptr

        cuest_check('NonlocalXCDerivativeRKSCompute',
            ce.cuestNonlocalXCDerivativeRKSCompute(
                handle=cuest_handle,
                plan=nlc_xcintplan_handle,
                variableBufferSize=maximum_workspace_descriptor.pointer,
                temporaryWorkspace=Vxc_workspace.pointer,
                parameters=nonlocal_xc_compute_parameters,
                numOccupied=nocc,
                coefficientMatrix=Cocc_device_handle,
                outGradient=Vxcgrad_device_handle,
                )
            )

        del Vxc_workspace

        cuest_check('Destroy NonlocalXC Gradient Parameters',
            ce.cuestParametersDestroy(
                parametersType=ce.CuestParametersType.CUEST_NONLOCALXCDERIVATIVERKSCOMPUTE_PARAMETERS,
                parameters=nonlocal_xc_compute_parameters,
                )
            )
    elif len(Cocc_list) == 2:
        Cocca_device = Cocc_list[0]
        assert isinstance(Cocca_device, cp.ndarray)
        assert Cocca_device.ndim == 2
        nocca, nao = Cocca_device.shape
        assert nocca <= nao and nao == mol.nao

        Coccb_device = Cocc_list[1]
        assert isinstance(Coccb_device, cp.ndarray)
        assert Coccb_device.ndim == 2
        noccb, nao = Coccb_device.shape
        assert noccb <= nao and nao == mol.nao

        Cocca_device_handle = ce.Pointer()
        Cocca_device_handle.value = Cocca_device.data.ptr
        Coccb_device_handle = ce.Pointer()
        Coccb_device_handle.value = Coccb_device.data.ptr

        nonlocal_xc_compute_parameters = ce.cuestNonlocalXCDerivativeUKSComputeParameters()
        cuest_check('NonlocalXC UKS Gradient Parameters Create',
            ce.cuestParametersCreate(
                parametersType=ce.CuestParametersType.CUEST_NONLOCALXCDERIVATIVEUKSCOMPUTE_PARAMETERS,
                outParameters=nonlocal_xc_compute_parameters,
                )
            )

        cuest_check('VV10 b UKS Gradient Parameter Configure',
            ce.cuestParametersConfigure(
                parametersType=ce.CuestParametersType.CUEST_NONLOCALXCDERIVATIVEUKSCOMPUTE_PARAMETERS,
                parameters=nonlocal_xc_compute_parameters,
                attribute=ce.CuestNonlocalXCDerivativeUKSComputeParametersAttributes.CUEST_NONLOCALXCDERIVATIVEUKSCOMPUTE_PARAMETERS_VV10_B,
                attributeValue=vv10_b,
                )
            )
        cuest_check('VV10 C UKS Gradient Parameter Configure',
            ce.cuestParametersConfigure(
                parametersType=ce.CuestParametersType.CUEST_NONLOCALXCDERIVATIVEUKSCOMPUTE_PARAMETERS,
                parameters=nonlocal_xc_compute_parameters,
                attribute=ce.CuestNonlocalXCDerivativeUKSComputeParametersAttributes.CUEST_NONLOCALXCDERIVATIVEUKSCOMPUTE_PARAMETERS_VV10_C,
                attributeValue=vv10_C,
                )
            )
        cuest_check('VV10 Scale UKS Gradient Parameter Configure',
            ce.cuestParametersConfigure(
                parametersType=ce.CuestParametersType.CUEST_NONLOCALXCDERIVATIVEUKSCOMPUTE_PARAMETERS,
                parameters=nonlocal_xc_compute_parameters,
                attribute=ce.CuestNonlocalXCDerivativeUKSComputeParametersAttributes.CUEST_NONLOCALXCDERIVATIVEUKSCOMPUTE_PARAMETERS_VV10_SCALE,
                attributeValue=vv10_scale,
                )
            )

        temporary_workspace_descriptor = WorkspaceDescriptor()

        Vxcgrad_device_handle = ce.Pointer()
        cuest_check('NonlocalXCDerivativeUKSCompute Workspace Query',
            ce.cuestNonlocalXCDerivativeUKSComputeWorkspaceQuery(
                handle=cuest_handle,
                plan=nlc_xcintplan_handle,
                variableBufferSize=maximum_workspace_descriptor.pointer,
                temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
                parameters=nonlocal_xc_compute_parameters,
                numOccupiedAlpha=nocca,
                numOccupiedBeta=noccb,
                coefficientMatrixAlpha=Cocca_device_handle,
                coefficientMatrixBeta=Coccb_device_handle,
                outGradient=Vxcgrad_device_handle,
                )
            )

        log.debug(f"CuEST: NonlocalXCDerivativeUKS Temporary sizes: {temporary_workspace_descriptor}")

        Vxc_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

        Vxcgrad_device = cp.empty([mol.natm, 3], order = "C", dtype = cp.float64)
        Vxcgrad_device_handle.value = Vxcgrad_device.data.ptr

        cuest_check('NonlocalXCDerivativeUKSCompute',
            ce.cuestNonlocalXCDerivativeUKSCompute(
                handle=cuest_handle,
                plan=nlc_xcintplan_handle,
                variableBufferSize=maximum_workspace_descriptor.pointer,
                temporaryWorkspace=Vxc_workspace.pointer,
                parameters=nonlocal_xc_compute_parameters,
                numOccupiedAlpha=nocca,
                numOccupiedBeta=noccb,
                coefficientMatrixAlpha=Cocca_device_handle,
                coefficientMatrixBeta=Coccb_device_handle,
                outGradient=Vxcgrad_device_handle,
                )
            )

        del Vxc_workspace

        cuest_check('NonlocalXC Parameters Destroy',
            ce.cuestParametersDestroy(
                parametersType=ce.CuestParametersType.CUEST_NONLOCALXCDERIVATIVEUKSCOMPUTE_PARAMETERS,
                parameters=nonlocal_xc_compute_parameters,
                )
            )
    else:
        raise ValueError("Incorrect Cocc_list size for cuest_compute_nlc_gradient() function")

    return Vxcgrad_device

def cuest_compute_pcm_gradient(mol, densitymatrix_device, q_guess, cuest_handle, pcmintplan_handle):
    # The input D is assumed to be in cuest order
    log = logger.new_logger(mol, mol.verbose)

    assert isinstance(densitymatrix_device, cp.ndarray)
    assert densitymatrix_device.shape == (mol.nao, mol.nao)

    densitymatrix_device_handle = ce.Pointer()
    densitymatrix_device_handle.value = densitymatrix_device.data.ptr

    n_pcm_grid = cuest_get_n_pcm_grid(cuest_handle, pcmintplan_handle)
    assert isinstance(q_guess, cp.ndarray)
    assert q_guess.shape == (n_pcm_grid,)
    q_guess_device_handle = ce.Pointer()
    q_guess_device_handle.value = q_guess.data.ptr

    pcm_derivative_compute_parameters = ce.cuestPCMDerivativeComputeParameters()
    cuest_check('Create PCM Grad Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_PCMDERIVATIVECOMPUTE_PARAMETERS,
            outParameters=pcm_derivative_compute_parameters,
            )
        )

    pcm_charge_convergence_threshold_handle = ce.data_double()
    pcm_charge_convergence_threshold_handle.value = 1e-14 # Convergence threshold (max absolute residual value) for PCM charges computation
    cuest_check('Configure PCM charge convergence threshold for gradient',
        ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_PCMDERIVATIVECOMPUTE_PARAMETERS,
            parameters=pcm_derivative_compute_parameters,
            attribute=ce.CuestPCMDerivativeComputeParametersAttributes.CUEST_PCMDERIVATIVECOMPUTE_PARAMETERS_CONVERGENCE_THRESHOLD,
            attributeValue=pcm_charge_convergence_threshold_handle,
            )
        )

    pcm_charge_max_iteration_handle = ce.data_uint64_t()
    pcm_charge_max_iteration_handle.value = 100 # Maximum number of preconditioned conjugate gradient iterations for PCM charges computation
    cuest_check('Configure PCM charge convergence threshold for gradient',
        ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_PCMDERIVATIVECOMPUTE_PARAMETERS,
            parameters=pcm_derivative_compute_parameters,
            attribute=ce.CuestPCMDerivativeComputeParametersAttributes.CUEST_PCMDERIVATIVECOMPUTE_PARAMETERS_MAX_ITERATIONS,
            attributeValue=pcm_charge_max_iteration_handle,
            )
        )

    pcm_results_handle = ce.cuestPCMResultsHandle()
    cuest_check('Create PCM Result Handle',
        ce.cuestResultsCreate(
            resultsType=ce.CuestResultsType.CUEST_PCM_RESULTS,
            outResults=pcm_results_handle,
            )
        )

    temporary_workspace_descriptor = WorkspaceDescriptor()

    q_out = cp.empty(n_pcm_grid, dtype = cp.float64)
    q_out_device_handle = ce.Pointer()
    q_out_device_handle.value = q_out.data.ptr

    pcmgrad_device = cp.empty([mol.natm, 3], order = "C", dtype = cp.float64)
    pcmgrad_device_handle = ce.Pointer()
    pcmgrad_device_handle.value = pcmgrad_device.data.ptr

    cuest_check('Compute PCM Grad Workspace Query',
        ce.cuestPCMDerivativeComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=pcmintplan_handle,
            parameters=pcm_derivative_compute_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            densityMatrix=densitymatrix_device_handle,
            inQ=q_guess_device_handle,
            outQ=q_out_device_handle,
            outPCMResults=pcm_results_handle,
            outPCMGradient=pcmgrad_device_handle,
            )
        )
    log.debug(f"CuEST: PCM gradient Temporary sizes: {temporary_workspace_descriptor}")

    pcmgrad_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    cuest_check('Compute PCM Grad',
        ce.cuestPCMDerivativeCompute(
            handle=cuest_handle,
            plan=pcmintplan_handle,
            parameters=pcm_derivative_compute_parameters,
            temporaryWorkspace=pcmgrad_temporary_workspace.pointer,
            densityMatrix=densitymatrix_device_handle,
            inQ=q_guess_device_handle,
            outQ=q_out_device_handle,
            outPCMResults=pcm_results_handle,
            outPCMGradient=pcmgrad_device_handle,
            )
        )

    del pcmgrad_temporary_workspace

    cuest_check('Destroy PCM Grad Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_PCMDERIVATIVECOMPUTE_PARAMETERS,
            parameters=pcm_derivative_compute_parameters,
            )
        )

    converged_handle = ce.data_int32_t()
    cuest_check('Query PCM result converged',
        ce.cuestResultsQuery(
            resultsType=ce.CuestResultsType.CUEST_PCM_RESULTS,
            results=pcm_results_handle,
            attribute=ce.CuestPCMResultAttributes.CUEST_PCMRESULT_CONVERGED,
            attributeValue=converged_handle,
            )
        )
    assert converged_handle.value

    residual_handle = ce.data_double()
    cuest_check('Query PCM result residual',
        ce.cuestResultsQuery(
            resultsType=ce.CuestResultsType.CUEST_PCM_RESULTS,
            results=pcm_results_handle,
            attribute=ce.CuestPCMResultAttributes.CUEST_PCMRESULT_CONVERGED_RESIDUAL,
            attributeValue=residual_handle,
            )
        )
    assert residual_handle.value < 1e-14

    cuest_check('Destroy PCM Result Handle',
        ce.cuestResultsDestroy(
            resultsType=ce.CuestResultsType.CUEST_PCM_RESULTS,
            results=pcm_results_handle,
            )
        )

    return pcmgrad_device

def cuest_compute_ecp_gradient(mol, densitymatrix_device, cuest_handle, ecpintplan_handle, maximum_workspace_bytes):
    # The input D is assumed to be in cuest order
    log = logger.new_logger(mol, mol.verbose)

    assert isinstance(densitymatrix_device, cp.ndarray)
    assert densitymatrix_device.shape == (mol.nao, mol.nao)

    densitymatrix_device_handle = ce.Pointer()
    densitymatrix_device_handle.value = densitymatrix_device.data.ptr

    maximum_workspace_bytes = min(maximum_workspace_bytes, get_cupy_maximum_free_bytes())
    maximum_workspace_descriptor = WorkspaceDescriptor(
        host_buffer_size_in_bytes=0,
        device_buffer_size_in_bytes=maximum_workspace_bytes,
        )

    ecpderivcompute_parameters = ce.cuestECPDerivativeComputeParameters()
    cuest_check('Create ECPDerivativeCompute Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_ECPDERIVATIVECOMPUTE_PARAMETERS,
            outParameters=ecpderivcompute_parameters,
            )
        )

    temporary_workspace_descriptor = WorkspaceDescriptor()

    ecpgrad_device_handle = ce.Pointer()
    cuest_check('ECPDerivativeCompute Workspace Query',
        ce.cuestECPDerivativeComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=ecpintplan_handle,
            parameters=ecpderivcompute_parameters,
            variableBufferSize=maximum_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            densityMatrix=densitymatrix_device_handle,
            outGradient=ecpgrad_device_handle,
            )
        )
    log.debug(f"CuEST: ECP gradient Temporary sizes: {temporary_workspace_descriptor}")

    ecp_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    ecpgrad_device = cp.empty([mol.natm, 3], order = "C", dtype = cp.float64)
    ecpgrad_device_handle.value = ecpgrad_device.data.ptr

    cuest_check('ECPDerivativeCompute',
        ce.cuestECPDerivativeCompute(
            handle=cuest_handle,
            plan=ecpintplan_handle,
            parameters=ecpderivcompute_parameters,
            variableBufferSize=maximum_workspace_descriptor.pointer,
            temporaryWorkspace=ecp_temporary_workspace.pointer,
            densityMatrix=densitymatrix_device_handle,
            outGradient=ecpgrad_device_handle,
            )
        )

    del ecp_temporary_workspace

    cuest_check('Destroy ECPDerivativeCompute Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_ECPDERIVATIVECOMPUTE_PARAMETERS,
            parameters=ecpderivcompute_parameters,
            )
        )

    return ecpgrad_device

def _cuest_pyscf_reorder_spherical_slow(mol, M, axis = [0,1], cuest_to_pyscf = True):
    # Reorder matrix between cuest order 0, 1, -1, 2, -2, ..., n, -n, and pyscf order -n, ..., -2, -1, 0, 1, 2, ..., n.
    # Except for p orbitals, cuest order for p is z,x,y, pyscf order for p is x,y,z (not y,z,x).

    assert mol.cart is False

    if type(axis) is int:
        axis = [axis]
    assert type(axis) is list
    assert all([type(dim) is int for dim in axis])
    assert all([dim >= 0 for dim in axis])
    for dim in axis:
        assert M.shape[dim] == mol.nao
    assert M.ndim <= 3, "Only up to 3rd order tensor is supported for PySCF-cuEST interconversion"
    assert min(axis) >= 0 and max(axis) <= 2, "Only up to 3rd order tensor is supported for PySCF-cuEST interconversion"

    i_ao_offset = 0
    for shell_info in mol._bas:
        L = int(shell_info[ANG_OF])
        nao_per_shell = 2 * L + 1

        if L == 0:
            pass
        else:
            if cuest_to_pyscf:
                if L == 1:
                    index_map = [1, 2, 0]
                else:
                    index_map = [(2*L+2 - m*2) for m in range(1, L+1)] + [ 0 ] + [(m*2-1) for m in range(1, L+1)]
            else:
                if L == 1:
                    index_map = [2, 0, 1]
                else:
                    index_map = [ [L] ] + [ [L+m, L-m] for m in range(1, L+1) ]
                    index_map = [ m for L in index_map for m in L ]

            if 0 in axis:
                rows = M[i_ao_offset : i_ao_offset + nao_per_shell, :]
                M[i_ao_offset : i_ao_offset + nao_per_shell, :] = rows[index_map, :]
            if 1 in axis:
                cols = M[:, i_ao_offset : i_ao_offset + nao_per_shell]
                M[:, i_ao_offset : i_ao_offset + nao_per_shell] = cols[:, index_map]
            if 2 in axis:
                cols = M[:, :, i_ao_offset : i_ao_offset + nao_per_shell]
                M[:, :, i_ao_offset : i_ao_offset + nao_per_shell] = cols[:, :, index_map]

        i_ao_offset += nao_per_shell
    assert i_ao_offset == mol.nao

    return M

_cuest_pyscf_reorder_spherical_kernel_registery = {}

def _cuest_pyscf_reorder_spherical(mol, M, axis = [0,1], cuest_to_pyscf = True):
    # Reorder matrix between cuest order 0, 1, -1, 2, -2, ..., n, -n, and pyscf order -n, ..., -2, -1, 0, 1, 2, ..., n.
    # Except for p orbitals, cuest order for p is z,x,y, pyscf order for p is x,y,z (not y,z,x).

    assert mol.cart is False

    if type(axis) is int:
        axis = [axis]
    assert type(axis) is list
    assert all([type(dim) is int for dim in axis])
    assert all([dim >= 0 for dim in axis])
    for dim in axis:
        assert M.shape[dim] == mol.nao
    assert M.ndim <= 3, "Only up to 3rd order tensor is supported for PySCF-cuEST interconversion"
    assert min(axis) >= 0 and max(axis) <= 2, "Only up to 3rd order tensor is supported for PySCF-cuEST interconversion"
    assert M.flags['C_CONTIGUOUS']

    Lmax = np.max(mol._bas[:, ANG_OF])

    ao_offsets = [[] for _ in range(Lmax + 1)]
    i_ao_offset = 0
    for L in mol._bas[:, ANG_OF]:
        nao_per_shell = 2 * L + 1
        ao_offsets[L].append(i_ao_offset)
        i_ao_offset += nao_per_shell
    assert i_ao_offset == mol.nao

    for L in range(1, Lmax + 1):
        if cuest_to_pyscf:
            if L == 1:
                index_map = [1, 2, 0]
            else:
                index_map = [(2*L+2 - m*2) for m in range(1, L+1)] + [ 0 ] + [(m*2-1) for m in range(1, L+1)]
        else:
            if L == 1:
                index_map = [2, 0, 1]
            else:
                index_map = [ [L] ] + [ [L+m, L-m] for m in range(1, L+1) ]
                index_map = [ m for L in index_map for m in L ]

        kernel_function_name = f"_cuest_pyscf_reorder_spherical_L_{L}_{'backward' if cuest_to_pyscf else 'forward'}_kernel"
        if kernel_function_name not in _cuest_pyscf_reorder_spherical_kernel_registery:
            kernel_code = f'''
                extern "C" __global__
                void {kernel_function_name} (
                    double* __restrict__ M, const int* __restrict__ offsets, const int n_target,
                    const int next_term_increment, const int next_line_increment, const int next_page_increment, const int n_page
                )
                {{
                    const int i_x = blockDim.x * blockIdx.x + threadIdx.x;
                    const int i_y = blockDim.y * blockIdx.y + threadIdx.y;
                    if (i_x >= next_term_increment * n_page || i_y >= n_target)
                        return;

                    const int i_term = i_x % next_term_increment;
                    const int i_page = i_x / next_term_increment;

                    const int i_start = i_page * next_page_increment + i_term * next_line_increment + offsets[i_y] * next_term_increment;
                    double temp[{2*L+1}];
                    for (int i = 0; i < {2*L+1}; i++)
                    {{
                        temp[i] = M[i_start + i * next_term_increment];
                    }}

                    const int index_map[{2*L+1}] {{ {",".join(map(str, index_map))} }};
                    for (int i = 0; i < {2*L+1}; i++)
                    {{
                        M[i_start + i * next_term_increment] = temp[index_map[i]];
                    }}
                }}
            '''

            _cuest_pyscf_reorder_spherical_kernel_registery[kernel_function_name] = cp.RawKernel(kernel_code, kernel_function_name)

    ao_offsets = [cp.asarray(offsets_L, dtype = cp.int32) for offsets_L in ao_offsets]

    assert isinstance(M, cp.ndarray)
    assert M.dtype == cp.float64
    assert int(np.prod(M.shape)) < np.iinfo(np.int32).max

    for dim in axis:
        next_term_increment = int(np.prod(M.shape[dim + 1 : ]))
        next_line_increment = M.shape[dim] if dim == M.ndim - 1 else 1
        next_page_increment = next_term_increment * M.shape[dim]
        n_page = int(np.prod(M.shape[ : dim]))

        for L in range(1, Lmax + 1):
            kernel_function_name = f"_cuest_pyscf_reorder_spherical_L_{L}_{'backward' if cuest_to_pyscf else 'forward'}_kernel"
            kernel = _cuest_pyscf_reorder_spherical_kernel_registery[kernel_function_name]

            offsets_L = ao_offsets[L]
            n_target = len(offsets_L)
            kernel_parameters = [M, offsets_L, cp.int32(n_target),
                                 cp.int32(next_term_increment), cp.int32(next_line_increment), cp.int32(next_page_increment), cp.int32(n_page)]
            kernel(((next_term_increment * n_page + 32 - 1) // 32, (n_target + 32 - 1) // 32), (32, 32), kernel_parameters)

    return M

def cuest_to_pyscf_output_reorder_spherical(mol, M, axis = [0,1]):
    return _cuest_pyscf_reorder_spherical(mol, M, axis, cuest_to_pyscf = True)

def pyscf_to_cuest_input_reorder_spherical(mol, M, axis = [0,1]):
    return _cuest_pyscf_reorder_spherical(mol, M, axis, cuest_to_pyscf = False)

def _cuest_pyscf_scale_cartesian(mol, M, axis = [0,1]):
    # In cuest, the "superdiagonal" functions like x, xx, xxx, ... are normalized, and others are normalized as if they are superdiagonal.
    # In pyscf, same strategy is applied, but in addition, for d and higher orbitals, there's an additional factor of np.sqrt((4*np.pi)/(2*L+1)).

    assert mol.cart is True

    if type(axis) is int:
        axis = [axis]
    assert type(axis) is list
    assert all([type(dim) is int for dim in axis])
    assert all([dim >= 0 for dim in axis])
    for dim in axis:
        assert M.shape[dim] == mol.nao
    assert M.ndim <= 3, "Only up to 3rd order tensor is supported for PySCF-cuEST interconversion"
    assert min(axis) >= 0 and max(axis) <= 2, "Only up to 3rd order tensor is supported for PySCF-cuEST interconversion"

    i_ao_offset = 0
    for shell_info in mol._bas:
        L = int(shell_info[ANG_OF])
        nao_per_shell = (L + 1) * (L + 2) // 2

        if L <= 1:
            pass
        else:
            cuest_to_pyscf_prefactor = np.sqrt((4*np.pi)/(2*L+1))
            if 0 in axis:
                M[i_ao_offset : i_ao_offset + nao_per_shell, :] *= cuest_to_pyscf_prefactor
            if 1 in axis:
                M[:, i_ao_offset : i_ao_offset + nao_per_shell] *= cuest_to_pyscf_prefactor
            if 2 in axis:
                M[:, :, i_ao_offset : i_ao_offset + nao_per_shell] *= cuest_to_pyscf_prefactor

        i_ao_offset += nao_per_shell
    assert i_ao_offset == mol.nao

    return M

cuest_to_pyscf_output_scale_cartesian = _cuest_pyscf_scale_cartesian
pyscf_to_cuest_input_scale_cartesian = _cuest_pyscf_scale_cartesian

def get_mocc_list_cuest_order_for_xc(mol, dms):
    log = logger.new_logger(mol, mol.verbose)

    mo_coeffs = None
    mo_occs   = None
    if getattr(dms, 'mo_coeff', None) is not None:
        mo_coeffs = cp.asarray(dms.mo_coeff)
        mo_occs   = cp.asarray(dms.mo_occ)

    if dms.ndim == 2:
        dms = dms[None, :]
        if mo_coeffs is not None:
            mo_coeffs = mo_coeffs[None, :]
            mo_occs   =   mo_occs[None, :]
    assert dms.ndim == 3
    assert dms.shape[-2:] == (mol.nao, mol.nao)
    n_dm = dms.shape[0]
    assert n_dm in (1, 2)

    numerical_zero = 1e-12

    mocc_list = []
    for i_dm in range(n_dm):
        if mo_coeffs is not None:
            mo_coeff = cp.asarray(mo_coeffs[i_dm])
            mo_occ   = cp.asarray(mo_occs[i_dm])
            assert cp.max(cp.abs(mo_occ - cp.round(mo_occ))) < numerical_zero, "CuEST doesn't support fractional occupation yet."
            mocc = mo_coeff[:, mo_occ > 0]

            dm = cp.asarray(dms[i_dm]) * (0.5 if n_dm == 1 else 1)
            assert cp.max(cp.abs(dm - mocc @ mocc.T)) < numerical_zero, "dm and mo_coeff are not consistent. CuEST doesn't support incremental SCF now."
        else:
            log.warn("CuEST got a dm without mo_coeff tag, so it needs to recover mo_coeff using eigh, which is super slow.")

            dm = cp.asarray(dms[i_dm])
            assert cp.max(cp.abs(dm - dm.T)) < numerical_zero
            mo_occ, mo_coeff = cp.linalg.eigh(dm)
            assert all(mo_occ > -numerical_zero), f"Large negative eigenvalue ({min(mo_occ)}) found for density matrix."
            mocc = mo_coeff[:, mo_occ > numerical_zero]

            mocc = mocc * cp.sqrt(mo_occ[mo_occ > numerical_zero]) * (np.sqrt(0.5) if n_dm == 1 else 1)

        mocc_copy = cp.asarray(mocc.T, order = "C", dtype = cp.float64)
        if mol.cart:
            mocc_cuest_order = pyscf_to_cuest_input_scale_cartesian(mol, mocc_copy, axis = 1)
        else:
            mocc_cuest_order = pyscf_to_cuest_input_reorder_spherical(mol, mocc_copy, axis = 1)

        mocc_list.append(mocc_cuest_order)

    return mocc_list



class HandleBundle:
    # This object should be attached to a mean-field object, and when calling the build functions, the host mean-field object should pass in its "self".
    #
    # If you wonder why the HandleBundle class is separated from the CuESTWrapper class, it is becuase the underlying mean-field class of CuESTWrapper object
    # can be shallow copied, for example in the following call tree of mf.PCM().Gradients():
    # -> gpu4pyscf.solvent.grad.pcm.make_grad_object()
    # -> gpu4pyscf.solvent._attach_solvent.SCFWithSolvent.undo_solvent()
    # -> pyscf.lib.misc.view()
    # And the __del__() function of the mean-field object will be called multiple times. If the handles are attached directly to mean-field objects, then
    # double free will happen. So we extract all handles to one object, and make sure this one object is only copied by reference.
    # Please make sure this object is not shallow or deep copied, to avoid double free.
    cuest_handle = None
    aobasis_handle = None
    aobasis_persistent_workspace = None
    aopairlist_handle = None
    aopairlist_persistent_workspace = None
    oeintplan_handle = None
    oeintplan_persistent_workspace = None
    auxbasis_handle = None
    auxbasis_persistent_workspace = None
    dfintplan_handle = None
    dfintplan_persistent_workspace = None
    moleculargrid_handle = None
    moleculargrid_persistent_workspace = None
    xcintplan_handle = None
    xcintplan_persistent_workspace = None
    nlc_moleculargrid_handle = None
    nlc_moleculargrid_persistent_workspace = None
    nlc_xcintplan_handle = None
    nlc_xcintplan_persistent_workspace = None
    pcmintplan_handle = None
    pcmintplan_persistent_workspace = None
    ecpintplan_handle = None
    ecpintplan_persistent_workspace = None

    def __init__(self, mf):
        mol = mf.mol

        cuest_handle_parameters = ce.cuestHandleParameters()
        cuest_check('Create Handle Params',
            ce.cuestParametersCreate(
                parametersType=ce.CuestParametersType.CUEST_HANDLE_PARAMETERS,
                outParameters=cuest_handle_parameters,
                )
            )

        maxl_handle = ce.data_uint64_t()
        cuest_check('Query Handle Max L',
            ce.cuestParametersQuery(
                parametersType=ce.CuestParametersType.CUEST_HANDLE_PARAMETERS,
                parameters=cuest_handle_parameters,
                attribute=ce.CuestHandleParametersAttributes.CUEST_HANDLE_PARAMETERS_MAX_L_SOLID_HARMONIC,
                attributeValue=maxl_handle,
                )
            )
        assert mol._bas[:,ANG_OF].max() <= maxl_handle.value

        ### TODO: Support multistream and multigpu

        cuest_handle = ce.cuestHandle()
        cuest_check('Create Cuest Handle',
            ce.cuestCreate(
                parameters=cuest_handle_parameters,
                handle=cuest_handle,
                )
            )

        cuest_check('Destroy Handle Params',
            ce.cuestParametersDestroy(
                parametersType=ce.CuestParametersType.CUEST_HANDLE_PARAMETERS,
                parameters=cuest_handle_parameters,
                )
            )

        self.cuest_handle = cuest_handle

    def __del__(self):
        # Warning: This __del__() function can only be called once, otherwise it'll cause double free.
        # The simptom is one of the free_*() function will fail with the following error:
        # RuntimeError: Destroy Cuest Handle failed with code CuestStatus.CUEST_STATUS_INVALID_TYPE
        self.reset()

        cuest_handle = self.cuest_handle
        cuest_check('Destroy Cuest Handle',
            ce.cuestDestroy(
                handle=cuest_handle,
                )
            )

    def __copy__(self):
        raise TypeError(f"No copy for {type(self).__name__}, to avoid double free problem.")
    def __deepcopy__(self, memo):
        raise TypeError(f"No deepcopy for {type(self).__name__}, to avoid double free problem.")

    def build_aobasis(self, mf):
        self.free_aobasis()

        mol = mf.mol
        cuest_handle = self.cuest_handle
        self.aobasis_handle, self.aobasis_persistent_workspace = pyscf_mol_to_cuest_basis(mol, cuest_handle)

    def free_aobasis(self):
        self.free_xcintplan()
        self.free_dfintplan()
        self.free_oeintplan()
        self.free_aopairlist()
        self.free_ecpintplan()

        aobasis_handle = self.aobasis_handle
        if aobasis_handle is not None:
            cuest_check('Destroy AOBasis',
                ce.cuestAOBasisDestroy(
                    handle=aobasis_handle,
                    )
                )
            self.aobasis_persistent_workspace = None
            self.aobasis_handle = None

    def build_aopairlist(self, mf):
        self.free_aopairlist()

        mol = mf.mol
        cuest_handle = self.cuest_handle
        if self.aobasis_handle is None:
            self.build_aobasis(mf)
        aobasis_handle = self.aobasis_handle
        threshold_pq = mf.threshold_pq
        self.aopairlist_handle, self.aopairlist_persistent_workspace = cuest_build_pairlist(mol, cuest_handle, aobasis_handle, threshold_pq)

    def free_aopairlist(self):
        self.free_dfintplan()
        self.free_oeintplan()

        aopairlist_handle = self.aopairlist_handle
        if aopairlist_handle is not None:
            cuest_check('Destroy AOPairList',
                ce.cuestAOPairListDestroy(
                    handle=aopairlist_handle,
                    )
                )
            self.aopairlist_persistent_workspace = None
            self.aopairlist_handle = None

    def build_oeintplan(self, mf):
        self.free_oeintplan()

        mol = mf.mol
        cuest_handle = self.cuest_handle
        if self.aobasis_handle is None:
            self.build_aobasis(mf)
        aobasis_handle = self.aobasis_handle
        if self.aopairlist_handle is None:
            self.build_aopairlist(mf)
        aopairlist_handle = self.aopairlist_handle
        self.oeintplan_handle, self.oeintplan_persistent_workspace = cuest_build_oeintplan(mol, cuest_handle, aobasis_handle, aopairlist_handle)

    def free_oeintplan(self):
        self.free_pcmintplan()

        oeintplan_handle = self.oeintplan_handle
        if oeintplan_handle is not None:
            cuest_check('Destroy OEIntPlan',
                ce.cuestOEIntPlanDestroy(
                    handle=oeintplan_handle,
                    )
                )
            self.oeintplan_persistent_workspace = None
            self.oeintplan_handle = None

    def build_auxbasis(self, mf):
        self.free_auxbasis()

        cuest_handle = self.cuest_handle
        if mf.auxmol is None:
            mf.build_auxmol()
        auxmol = mf.auxmol
        self.auxbasis_handle, self.auxbasis_persistent_workspace = pyscf_mol_to_cuest_basis(auxmol, cuest_handle, basis_name = "Aux")

    def free_auxbasis(self):
        self.free_dfintplan()

        auxbasis_handle = self.auxbasis_handle
        if auxbasis_handle is not None:
            cuest_check('Destroy Aux AOBasis',
                ce.cuestAOBasisDestroy(
                    handle=auxbasis_handle,
                    )
                )
            self.auxbasis_persistent_workspace = None
            self.auxbasis_handle = None

    def build_dfintplan(self, mf):
        self.free_dfintplan()

        mol = mf.mol
        cuest_handle = self.cuest_handle
        if self.aobasis_handle is None:
            self.build_aobasis(mf)
        aobasis_handle = self.aobasis_handle
        if self.auxbasis_handle is None:
            self.build_auxbasis(mf)
        auxbasis_handle = self.auxbasis_handle
        if self.aopairlist_handle is None:
            self.build_aopairlist(mf)
        aopairlist_handle = self.aopairlist_handle

        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time0 = time.time()

        self.dfintplan_handle, self.dfintplan_persistent_workspace = cuest_build_dfintplan(
            mol, cuest_handle, aobasis_handle, auxbasis_handle, aopairlist_handle,
            fitting_cutoff = mf.density_fitting_cutoff
        )

        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time1 = time.time()
            logger.debug(mol, f"CuEST: time_build_dfintplan = {time1 - time0} s")

    def free_dfintplan(self):
        dfintplan_handle = self.dfintplan_handle
        if dfintplan_handle is not None:
            cuest_check('Destroy DFIntPlan',
                ce.cuestDFIntPlanDestroy(
                    handle=dfintplan_handle,
                    )
                )
            self.dfintplan_persistent_workspace = None
            self.dfintplan_handle = None

    def build_moleculargrid(self, mf):
        self.free_moleculargrid()

        mol = mf.mol
        grids = getattr(mf, "grids", None)
        assert grids is not None
        cuest_handle = self.cuest_handle
        self.moleculargrid_handle, self.moleculargrid_persistent_workspace = cuest_build_moleculargrid(mol, grids, cuest_handle)

    def free_moleculargrid(self):
        self.free_xcintplan()

        moleculargrid_handle = self.moleculargrid_handle
        if moleculargrid_handle is not None:
            cuest_check('Destroy MolecularGrid',
                ce.cuestMolecularGridDestroy(
                    grid=moleculargrid_handle,
                    )
                )
            self.moleculargrid_persistent_workspace = None
            self.moleculargrid_handle = None

    def build_xcintplan(self, mf):
        self.free_xcintplan()

        mol = mf.mol
        cuest_handle = self.cuest_handle
        if self.aobasis_handle is None:
            self.build_aobasis(mf)
        aobasis_handle = self.aobasis_handle
        if self.moleculargrid_handle is None:
            self.build_moleculargrid(mf)
        moleculargrid_handle = self.moleculargrid_handle

        functional = pyscf_xc_to_cuest_functional(mf.xc)
        self.xcintplan_handle, self.xcintplan_persistent_workspace = cuest_build_xcintplan(mol, cuest_handle, aobasis_handle, moleculargrid_handle, functional)

        numint = getattr(mf, "_numint", None)
        if numint is None:
            assert "numint" in mf.__class__.__name__.lower()
            numint = mf

        xcintplan_handle = self.xcintplan_handle
        check_cuest_pyscf_functional_consistency(mol, mf.xc, numint, cuest_handle, xcintplan_handle)

    def free_xcintplan(self):
        xcintplan_handle = self.xcintplan_handle
        if xcintplan_handle is not None:
            cuest_check('Destroy XCIntPlan',
                ce.cuestXCIntPlanDestroy(
                    handle=xcintplan_handle,
                    )
                )
            self.xcintplan_persistent_workspace = None
            self.xcintplan_handle = None

    def build_nlc_moleculargrid(self, mf):
        self.free_nlc_moleculargrid()

        mol = mf.mol
        nlcgrids = getattr(mf, "nlcgrids", None)
        assert nlcgrids is not None
        cuest_handle = self.cuest_handle
        self.nlc_moleculargrid_handle, self.nlc_moleculargrid_persistent_workspace = cuest_build_moleculargrid(mol, nlcgrids, cuest_handle)

    def free_nlc_moleculargrid(self):
        self.free_nlc_xcintplan()

        nlc_moleculargrid_handle = self.nlc_moleculargrid_handle
        if nlc_moleculargrid_handle is not None:
            cuest_check('Destroy NLC MolecularGrid',
                ce.cuestMolecularGridDestroy(
                    grid=nlc_moleculargrid_handle,
                    )
                )
            self.nlc_moleculargrid_persistent_workspace = None
            self.nlc_moleculargrid_handle = None

    def build_nlc_xcintplan(self, mf):
        self.free_nlc_xcintplan()

        mol = mf.mol
        cuest_handle = self.cuest_handle
        if self.aobasis_handle is None:
            self.build_aobasis(mf)
        aobasis_handle = self.aobasis_handle
        if self.nlc_moleculargrid_handle is None:
            self.build_nlc_moleculargrid(mf)
        nlc_moleculargrid_handle = self.nlc_moleculargrid_handle

        numint = getattr(mf, "_numint", None)
        if numint is None:
            assert "numint" in mf.__class__.__name__.lower()
            numint = mf

        assert numint.libxc.is_nlc(mf.xc)
        functional = pyscf_xc_to_cuest_functional(mf.xc)
        self.nlc_xcintplan_handle, self.nlc_xcintplan_persistent_workspace = cuest_build_xcintplan(mol, cuest_handle, aobasis_handle, nlc_moleculargrid_handle, functional)

        nlc_xcintplan_handle = self.nlc_xcintplan_handle
        check_cuest_pyscf_nlc_consistency(mol, mf.xc, numint, cuest_handle, nlc_xcintplan_handle)

    def free_nlc_xcintplan(self):
        nlc_xcintplan_handle = self.nlc_xcintplan_handle
        if nlc_xcintplan_handle is not None:
            cuest_check('Destroy NLC XCIntPlan',
                ce.cuestXCIntPlanDestroy(
                    handle=nlc_xcintplan_handle,
                    )
                )
            self.nlc_xcintplan_persistent_workspace = None
            self.nlc_xcintplan_handle = None

    def build_pcmintplan(self, mf):
        self.free_pcmintplan()

        mol = mf.mol
        cuest_handle = self.cuest_handle
        if self.oeintplan_handle is None:
            self.build_oeintplan(mf)
        oeintplan_handle = self.oeintplan_handle

        with_solvent = getattr(mf, "with_solvent", None)
        if with_solvent is None:
            assert "pcm" in mf.__class__.__name__.lower()
            with_solvent = mf

        self.pcmintplan_handle, self.pcmintplan_persistent_workspace = cuest_build_pcmintplan(mol, cuest_handle, oeintplan_handle, with_solvent)

    def free_pcmintplan(self):
        pcmintplan_handle = self.pcmintplan_handle
        if pcmintplan_handle is not None:
            cuest_check('Destroy PCMIntPlan',
                ce.cuestPCMIntPlanDestroy(
                    handle=pcmintplan_handle,
                    )
                )
            self.pcmintplan_persistent_workspace = None
            self.pcmintplan_handle = None

    def build_ecpintplan(self, mf):
        self.free_ecpintplan()

        mol = mf.mol
        cuest_handle = self.cuest_handle
        if self.aobasis_handle is None:
            self.build_aobasis(mf)
        aobasis_handle = self.aobasis_handle

        self.ecpintplan_handle, self.ecpintplan_persistent_workspace = cuest_build_ecpintplan(mol, cuest_handle, aobasis_handle)

    def free_ecpintplan(self):
        ecpintplan_handle = self.ecpintplan_handle
        if ecpintplan_handle is not None:
            cuest_check('Destroy ECPIntPlan',
                ce.cuestECPIntPlanDestroy(
                    handle=ecpintplan_handle,
                    )
                )
            self.ecpintplan_persistent_workspace = None
            self.ecpintplan_handle = None

    def reset(self):
        self.free_aopairlist()
        self.free_aobasis()
        self.free_oeintplan()
        self.free_auxbasis()
        self.free_dfintplan()
        self.free_moleculargrid()
        self.free_xcintplan()
        self.free_nlc_moleculargrid()
        self.free_nlc_xcintplan()
        self.free_pcmintplan()
        self.free_ecpintplan()



class CuESTExtractedGrids(Grids):
    _locked_keys = []

    def __init__(self, grids, log):
        assert grids is not None
        if grids.becke_scheme != gen_grid.stratmann:
            log.warn("Warning: Stratmann modified Becke scheme is used for grid partitioning, "
                    "instead of the original Becke scheme (default), or any other scheme you specified. "
                    "Because right now cuEST only supports Stratmann scheme.")
            grids.becke_scheme = gen_grid.stratmann
        if grids.radii_adjust is not None:
            log.warn("Warning: Atomic radii adjustment in Becke partitioning is not supported in cuEST. "
                    "So it is turned off.")
            grids.radii_adjust = None

        if isinstance(grids, CuESTExtractedGrids):
            self._backup_grids = grids._backup_grids
        else:
            self._backup_grids = grids

        self.atom_grid = grids.atom_grid
        self.level = grids.level
        self.prune = grids.prune
        self.radi_method = grids.radi_method
        self.becke_scheme = grids.becke_scheme
        self.atomic_radii = grids.atomic_radii
        self.radii_adjust = grids.radii_adjust

        self._locked_keys = [
            '_backup_grids',
            'atom_grid', 'level', 'prune', 'radi_method', 'becke_scheme', 'atomic_radii', 'radii_adjust',
        ]

        self.verbose = log.verbose
        self.coords = np.array([np.nan])
        self.weights = np.array([np.nan])

        # Intentionally not calling super().__init__()

    def __setattr__(self, key, val):
        if key in self._locked_keys:
            raise RuntimeError("The grids setup should not be changed after CuESTWrapper is applied.")
        super().__setattr__(key, val)

    def dump_flags(self, verbose=None):
        logger.info(self, 'radial grids: %s', self.radi_method.__doc__)
        logger.info(self, 'becke partition: %s', self.becke_scheme.__doc__)
        logger.info(self, 'pruning grids: %s', self.prune)
        logger.info(self, 'grids dens level: %d', self.level)
        if self.radii_adjust is not None:
            logger.info(self, 'atomic radii adjust function: %s', self.radii_adjust)
            logger.debug2(self, 'atomic_radii : %s', self.atomic_radii)
        if self.atom_grid:
            logger.info(self, 'User specified grid scheme %s', str(self.atom_grid))
        return self

    def _do_nothing(self, *args, **kwargs):
        return self
    build = _do_nothing
    reset = _do_nothing

    def _should_never_be_called(*args, **kwargs):
        raise RuntimeError("This function should never be called when CuESTWrapper is applied")
    to_cpu = _should_never_be_called
    to_gpu = _should_never_be_called

class CuESTExtractedNumint(NumInt):
    _locked_keys = []

    def __init__(self, numint, mol, grids, nlcgrids, xc, handles, maximum_workspace_bytes, threshold_pq, turn_on_cuest_xc, turn_on_cuest_nlc):
        assert numint is not None
        
        self.mol = mol
        self.grids = grids
        self.nlcgrids = nlcgrids
        self.xc = xc
        self.handles = handles
        self.maximum_workspace_bytes = maximum_workspace_bytes
        self.threshold_pq = threshold_pq
        self.turn_on_cuest_xc = turn_on_cuest_xc
        self.turn_on_cuest_nlc = turn_on_cuest_nlc

        if isinstance(numint, CuESTExtractedNumint):
            self._backup_numint = numint._backup_numint
        else:
            self._backup_numint = numint

        self._locked_keys = [
            "_backup_numint",
            "mol", "grids", "xc", "handles", "maximum_workspace_bytes", "threshold_pq", "turn_on_cuest_xc", "turn_on_cuest_nlc",
        ]

        self.libxc = numint.libxc
        self.rsh_and_hybrid_coeff = numint.rsh_and_hybrid_coeff
        self.nlc_coeff = numint.nlc_coeff

        # Intentionally not calling super().__init__()

    def __setattr__(self, key, val):
        if key in self._locked_keys:
            raise RuntimeError("The numint setup should not be changed after CuESTWrapper is applied.")
        super().__setattr__(key, val)

    def nr_rks(self, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=0, verbose=None):
        assert relativity == 0, "Relativistic KS is not supported by cuEST yet"

        assert mol is None or mol is self.mol or mol_equal(mol, self.mol)
        assert grids == self.grids
        assert xc_code == self.xc

        if not self.turn_on_cuest_xc:
            grids = self.grids._backup_grids
            if grids.coords is None:
                grids.build()
            return self._backup_numint.nr_rks(mol = mol, grids = grids, xc_code = xc_code, dms = dms, relativity = relativity, hermi = hermi, max_memory = max_memory, verbose = verbose)

        mol = self.mol
        log = logger.new_logger(mol, mol.verbose)

        cuest_handle = self.handles.cuest_handle
        if self.handles.xcintplan_handle is None:
            self.handles.build_xcintplan(self)
        xcintplan_handle = self.handles.xcintplan_handle

        time_pre_post = 0
        time_kernel = 0
        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time0 = time.time()

        assert dms.ndim == 2
        mocc_list = get_mocc_list_cuest_order_for_xc(mol, dms)
        assert len(mocc_list) == 1

        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time1 = time.time()
            time_pre_post += time1 - time0
            time0 = time1

        exc, vxc_cuest_order = cuest_compute_xcpotential(mol, mocc_list, cuest_handle, xcintplan_handle, self.maximum_workspace_bytes)

        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time1 = time.time()
            time_kernel += time1 - time0
            time0 = time1

        assert vxc_cuest_order.ndim == 2
        if mol.cart:
            vxc = cuest_to_pyscf_output_scale_cartesian(mol, vxc_cuest_order)
        else:
            vxc = cuest_to_pyscf_output_reorder_spherical(mol, vxc_cuest_order)

        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time1 = time.time()
            time_pre_post += time1 - time0
            time0 = time1

        log.debug(f"CuEST: time_XC_pre_and_post_processing = {time_pre_post} s, time_XC_kernel = {time_kernel} s")

        nelec = np.nan
        return nelec, exc, vxc

    def nr_uks(self, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=0, verbose=None):
        assert relativity == 0, "Relativistic KS is not supported by cuEST yet"

        assert mol is None or mol is self.mol or mol_equal(mol, self.mol)
        assert grids == self.grids
        assert xc_code == self.xc

        if not self.turn_on_cuest_xc:
            grids = self.grids._backup_grids
            if grids.coords is None:
                grids.build()
            return self._backup_numint.nr_uks(mol = mol, grids = grids, xc_code = xc_code, dms = dms, relativity = relativity, hermi = hermi, max_memory = max_memory, verbose = verbose)

        mol = self.mol
        log = logger.new_logger(mol, mol.verbose)

        cuest_handle = self.handles.cuest_handle
        if self.handles.xcintplan_handle is None:
            self.handles.build_xcintplan(self)
        xcintplan_handle = self.handles.xcintplan_handle

        time_pre_post = 0
        time_kernel = 0
        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time0 = time.time()

        assert dms.ndim == 3
        mocc_list = get_mocc_list_cuest_order_for_xc(mol, dms)
        n_dm = 2
        assert len(mocc_list) == n_dm

        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time1 = time.time()
            time_pre_post += time1 - time0
            time0 = time1

        exc, vxc_cuest_order = cuest_compute_xcpotential(mol, mocc_list, cuest_handle, xcintplan_handle, self.maximum_workspace_bytes)

        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time1 = time.time()
            time_kernel += time1 - time0
            time0 = time1

        vxc = cp.zeros_like(dms)
        for i_dm in range(n_dm):
            if mol.cart:
                vxc[i_dm] = cuest_to_pyscf_output_scale_cartesian(mol, vxc_cuest_order[i_dm])
            else:
                vxc[i_dm] = cuest_to_pyscf_output_reorder_spherical(mol, vxc_cuest_order[i_dm])

        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time1 = time.time()
            time_pre_post += time1 - time0
            time0 = time1

        log.debug(f"CuEST: time_XC_pre_and_post_processing = {time_pre_post} s, time_XC_kernel = {time_kernel} s")

        nelec = np.nan
        return nelec, exc, vxc

    def nr_nlc_vxc(self, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=0, verbose=None):
        assert relativity == 0, "Relativistic KS is not supported by cuEST yet"

        assert mol is None or mol is self.mol or mol_equal(mol, self.mol)
        assert grids == self.nlcgrids
        assert xc_code == self.xc

        if not self.turn_on_cuest_nlc:
            grids = self.nlcgrids._backup_grids
            if grids.coords is None:
                grids.build()
            return self._backup_numint.nr_nlc_vxc(mol = mol, grids = grids, xc_code = xc_code, dms = dms, relativity = relativity, hermi = hermi, max_memory = max_memory, verbose = verbose)

        mol = self.mol
        log = logger.new_logger(mol, mol.verbose)

        cuest_handle = self.handles.cuest_handle
        if self.handles.nlc_xcintplan_handle is None:
            self.handles.build_nlc_xcintplan(self)
        nlc_xcintplan_handle = self.handles.nlc_xcintplan_handle

        time_pre_post = 0
        time_kernel = 0
        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time0 = time.time()

        mocc_list = get_mocc_list_cuest_order_for_xc(mol, dms)

        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time1 = time.time()
            time_pre_post += time1 - time0
            time0 = time1

        enlc, vnlc_cuest_order = cuest_compute_nlcpotential(mol, mocc_list, cuest_handle, nlc_xcintplan_handle, self.maximum_workspace_bytes)

        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time1 = time.time()
            time_kernel += time1 - time0
            time0 = time1

        if mol.cart:
            vnlc = cuest_to_pyscf_output_scale_cartesian(mol, vnlc_cuest_order)
        else:
            vnlc = cuest_to_pyscf_output_reorder_spherical(mol, vnlc_cuest_order)

        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time1 = time.time()
            time_pre_post += time1 - time0
            time0 = time1

        log.debug(f"CuEST: time_vv10_pre_and_post_processing = {time_pre_post} s, time_vv10_kernel = {time_kernel} s")

        nelec = np.nan
        return nelec, enlc, vnlc

    def _do_nothing(self, *args, **kwargs):
        return self
    build = _do_nothing

    def _should_never_be_called(*args, **kwargs):
        raise RuntimeError("This function should never be called when CuESTWrapper is applied")
    reset = _should_never_be_called
    to_cpu = _should_never_be_called
    to_gpu = _should_never_be_called
    nr_vxc = _should_never_be_called
    nr_fxc = _should_never_be_called
    get_fxc = _should_never_be_called
    nr_sap = _should_never_be_called
    nr_sap_vxc = _should_never_be_called
    nr_rks_fxc = _should_never_be_called
    nr_uks_fxc = _should_never_be_called
    nr_rks_fxc_st = _should_never_be_called
    cache_xc_kernel = _should_never_be_called
    cache_xc_kernel1 = _should_never_be_called
    make_mask = _should_never_be_called
    eval_ao = _should_never_be_called
    eval_rho = _should_never_be_called
    eval_rho1 = _should_never_be_called
    eval_rho2 = _should_never_be_called
    get_rho = _should_never_be_called
    get_rho_with_derivatives = _should_never_be_called
    block_loop = _should_never_be_called
    _gen_rho_evaluator = _should_never_be_called

class CuESTExtractedPCM(PCM): # This inheritance is necessary, because in pyscf/gpu4pyscf some code will check isinstance(mf, PCM) to determine if the mf is PCM or SMD.
    _locked_keys = []

    def __init__(self, with_solvent, mol, handles, threshold_pq, turn_on_cuest_pcm):
        assert with_solvent is not None

        self.mol = mol
        self.handles = handles
        self.threshold_pq = threshold_pq
        self.turn_on_cuest_pcm = turn_on_cuest_pcm

        if getattr(with_solvent, "surface_discretization_method", "").upper() != "ISWIG":
            log = logger.new_logger(mol, mol.verbose)
            log.warn("Warning: ISWIG algorithm is used for PCM switching function, instead of "
                    "SWIG algorithm (default).")
            with_solvent.surface_discretization_method = "ISWIG"

        if with_solvent.radii_table is None:
            with_solvent.build() # We shouldn't need a full build here, we just need radii_table

        if isinstance(with_solvent, CuESTExtractedPCM):
            self._backup_with_solvent = with_solvent._backup_with_solvent
        else:
            self._backup_with_solvent = with_solvent

        self.method = with_solvent.method
        self.lebedev_order = with_solvent.lebedev_order
        self.radii_table = with_solvent.radii_table
        self.eps = with_solvent.eps
        self.surface_discretization_method = with_solvent.surface_discretization_method

        self._locked_keys = [
            "_backup_with_solvent",
            "mol", "handles", "threshold_pq", "turn_on_cuest_pcm",
            "method", "lebedev_order", "radii_table", "eps", "surface_discretization_method",
        ]

        self.verbose = mol.verbose
        self.frozen = False
        self._intermediates = {}

        # Intentionally not calling super().__init__()

    def __setattr__(self, key, val):
        if key in self._locked_keys:
            raise RuntimeError("The pcm setup should not be changed after CuESTWrapper is applied.")
        super().__setattr__(key, val)

    def dump_flags(self, verbose=None):
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'lebedev_order = %s (%d grids per sphere)',
                    self.lebedev_order, gen_grid.LEBEDEV_ORDER[self.lebedev_order])
        logger.info(self, 'eps = %s'          , self.eps)
        logger.info(self, 'Iterative inversion convergence tolerance for K^-1 = %s', 1e-14)
        return self

    def kernel(self, dm):
        if not self.turn_on_cuest_pcm:
            return self._backup_with_solvent.kernel(dm)

        mol = self.mol

        log = logger.new_logger(mol, mol.verbose)

        time_pre_post = 0
        time_kernel = 0
        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time0 = time.time()

        assert dm is not None
        if dm.ndim == 3:
            assert dm.shape[0] == 2
            dm = dm[0] + dm[1]
        assert dm.shape == (mol.nao, mol.nao)

        cuest_handle = self.handles.cuest_handle
        if self.handles.pcmintplan_handle is None:
            self.handles.build_pcmintplan(self)
        pcmintplan_handle = self.handles.pcmintplan_handle

        dm_copy = cp.array(dm, order = "C", dtype = cp.float64) # cp.array gaurantees it makes a copy, so it's safe to modify it. Don't use cp.asarray here.
        if mol.cart:
            dm_cuest_order = pyscf_to_cuest_input_scale_cartesian(mol, dm_copy)
        else:
            dm_cuest_order = pyscf_to_cuest_input_reorder_spherical(mol, dm_copy)

        n_pcm_grid = self.get_n_pcm_grid()
        if 'q' in self._intermediates:
            q = self._intermediates['q']
            assert q.shape == (n_pcm_grid,)
        else:
            q = cp.zeros(n_pcm_grid, dtype = cp.float64)

        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time1 = time.time()
            time_pre_post += time1 - time0
            time0 = time1

        epcm, vpcm_cuest_order, updated_q = cuest_compute_pcmpotential(mol, dm_cuest_order, q, cuest_handle, pcmintplan_handle)

        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time1 = time.time()
            time_kernel += time1 - time0
            time0 = time1

        self._intermediates['q'] = updated_q
        if mol.cart:
            vpcm = cuest_to_pyscf_output_scale_cartesian(mol, vpcm_cuest_order)
        else:
            vpcm = cuest_to_pyscf_output_reorder_spherical(mol, vpcm_cuest_order)

        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time1 = time.time()
            time_pre_post += time1 - time0
            time0 = time1

        log.debug(f"CuEST: time_pcm_pre_and_post_processing = {time_pre_post} s, time_pcm_kernel = {time_kernel} s")

        return epcm, vpcm

    def get_n_pcm_grid(self):
        assert self.turn_on_cuest_pcm

        cuest_handle = self.handles.cuest_handle
        pcmintplan_handle = self.handles.pcmintplan_handle
        if pcmintplan_handle is None:
            return -1
        else:
            return cuest_get_n_pcm_grid(cuest_handle, pcmintplan_handle)

    def grad(self, dm):
        if not self.turn_on_cuest_pcm:
            return self._backup_with_solvent.grad(dm)

        mol = self.mol

        assert dm is not None
        if dm.ndim == 3:
            assert dm.shape[0] == 2
            dm = dm[0] + dm[1]
        assert dm.shape == (mol.nao, mol.nao)

        cuest_handle = self.handles.cuest_handle
        if self.handles.pcmintplan_handle is None:
            self.handles.build_pcmintplan(self)
        pcmintplan_handle = self.handles.pcmintplan_handle

        dm_copy = cp.array(dm, order = "C", dtype = cp.float64) # cp.array gaurantees it makes a copy, so it's safe to modify it. Don't use cp.asarray here.
        if mol.cart:
            dm_cuest_order = pyscf_to_cuest_input_scale_cartesian(mol, dm_copy)
        else:
            dm_cuest_order = pyscf_to_cuest_input_reorder_spherical(mol, dm_copy)

        n_pcm_grid = self.get_n_pcm_grid()
        if 'q' in self._intermediates:
            q = self._intermediates['q']
            assert q.shape == (n_pcm_grid,)
        else:
            q = cp.zeros(n_pcm_grid, dtype = cp.float64)

        dpcm = cuest_compute_pcm_gradient(mol, dm_cuest_order, q, cuest_handle, pcmintplan_handle)

        return dpcm.get()

    def hess(self, dm):
        raise NotImplementedError("PCM Hessian not implemented in CuEST yet")

    def reset(self, mol = None):
        assert mol is None, "If you need to reset a CuEST modified PCM object with a new mol, please do that to the mean field object. The attached with_solvent will get re-constructed."
        self._intermediates = {}

    def _do_nothing(self, *args, **kwargs):
        return self
    build = _do_nothing
    check_sanity = _do_nothing

    def _should_never_be_called(*args, **kwargs):
        raise RuntimeError("This function should never be called when CuESTWrapper is applied")
    to_cpu = _should_never_be_called
    to_gpu = _should_never_be_called



class CuESTWrapper(lib.StreamObject):
    _unsafe_for_change_keys = []

    def __init__(self, method):
        self.__dict__.update(method.__dict__)

        mol = self.mol
        assert mol is not None
        self.auxbasis = getattr(method, "auxbasis", None)
        self.auxmol = None

        self.threshold_pq = 1e-14
        self.density_fitting_cutoff = 1e-12
        self.maximum_workspace_bytes = 10 * 1000**3 # 10 GB

        self.turn_on_cuest_hcore = True
        self.turn_on_cuest_jk = True
        self.turn_on_cuest_xc = True
        self.turn_on_cuest_nlc = True
        self.turn_on_cuest_pcm = True

        self.handles = HandleBundle(self)

        if getattr(self, "xc", None) is not None:
            assert getattr(self, "grids", None) is not None

            log = logger.new_logger(mol, mol.verbose)
            self.grids = CuESTExtractedGrids(self.grids, log)

            if getattr(self, "nlcgrids", None) is not None:
                self.nlcgrids = CuESTExtractedGrids(self.nlcgrids, log)

            assert getattr(self, "_numint", None) is not None

            self._numint = CuESTExtractedNumint(self._numint, self.mol, self.grids, self.nlcgrids, self.xc, self.handles, self.maximum_workspace_bytes, self.threshold_pq, self.turn_on_cuest_xc, self.turn_on_cuest_nlc)

            # TODO: deal with zero-rho small_rho_cutoff

        if getattr(self, "with_solvent", None) is not None:
            self.with_solvent = CuESTExtractedPCM(self.with_solvent, self.mol, self.handles, self.threshold_pq, self.turn_on_cuest_pcm)

        self._math_mode = "native_fp64"
        ce.cuestSetMathMode(
            handle = self.handles.cuest_handle,
            mode = ce.CuestMathMode.CUEST_NATIVE_FP64_MATH_MODE,
        )

        ### Warning: you're at your own risk if you modify the following parameters.
        self.additional_precision_control_parameters = {
            "CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS_INT8_SLICE_COUNT" : 10,
            "CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS_INT8_MODULUS_COUNT" : 10,
        }

        self._unsafe_for_change_keys = [
            "auxbasis", "threshold_pq", "density_fitting_cutoff",
            "turn_on_cuest_hcore", "turn_on_cuest_jk", "turn_on_cuest_xc", "turn_on_cuest_nlc", "turn_on_cuest_pcm",
        ]

    def __setattr__(self, key, val):
        super().__setattr__(key, val)
        if key in self._unsafe_for_change_keys:
            self.reset()

    def __deepcopy__(self, memo):
        raise TypeError(f"No deepcopy for {type(self).__name__}, to avoid double free problem.")

    @property
    def math_mode(self):
        return self._math_mode
    @math_mode.setter
    def math_mode(self, new_mode):
        cuest_math_mode_name_map = {
            "default" : ce.CuestMathMode.CUEST_DEFAULT_MATH_MODE,
            "native_fp64" : ce.CuestMathMode.CUEST_NATIVE_FP64_MATH_MODE,
        }
        new_mode = new_mode.lower().replace("-", "_")
        assert new_mode in cuest_math_mode_name_map, f"Math mode ({new_mode}) is not available. Available options are: {list(cuest_math_mode_name_map.keys())}"

        if self.math_mode != new_mode:
            # self.reset() # No need to reset
            ce.cuestSetMathMode(
                handle = self.handles.cuest_handle,
                mode = cuest_math_mode_name_map[new_mode],
            )
            self._math_mode = new_mode
            print(f"Switching math_mode to {new_mode}")

    def build_auxmol(self):
        assert self.auxbasis is not None, f"auxbasis field not set for {__class__} object"

        mol = self.mol
        if self.auxmol is None:
            self.auxmol = addons.make_auxmol(mol, self.auxbasis)

        if self.auxmol.cart:
            log = logger.new_logger(mol, mol.verbose)
            log.warn("CuEST only supports spherical auxiliary basis, and we found a Cartesian auxmol from PySCF input "
                     "(likely because your primary basis is Cartesian). We fake the auxmol to spherical. This means "
                     "PySCF still uses Cartesian auxiliary basis, and cuEST still uses spherical auxiliary basis. "
                     "The result is thus inconsistent.")
            self.auxmol.cart = False

    def free_auxmol(self):
        self.handles.free_auxbasis()

        self.auxmol = None

    def reset(self, mol = None):
        if mol is not None:
            self.mol = mol
        self.handles.reset()
        self.free_auxmol()

        if getattr(self, "xc", None) is not None:
            self._numint = CuESTExtractedNumint(self._numint, self.mol, self.grids, self.nlcgrids, self.xc, self.handles, self.maximum_workspace_bytes, self.threshold_pq, self.turn_on_cuest_xc, self.turn_on_cuest_nlc)
        if getattr(self, "with_solvent", None) is not None:
            self.with_solvent = CuESTExtractedPCM(self.with_solvent, self.mol, self.handles, self.threshold_pq, self.turn_on_cuest_pcm)

        # super().reset(mol)

    def get_ovlp(self, mol = None):
        assert mol is None or mol is self.mol or mol_equal(mol, self.mol)

        if not self.turn_on_cuest_hcore:
            return super().get_ovlp(mol = mol)

        mol = self.mol
        cuest_handle = self.handles.cuest_handle
        if self.handles.oeintplan_handle is None:
            self.handles.build_oeintplan(self)
        oeintplan_handle = self.handles.oeintplan_handle

        S_cuest_order = cuest_compute_overlapint(mol, cuest_handle, oeintplan_handle)

        if mol.cart:
            return cuest_to_pyscf_output_scale_cartesian(mol, S_cuest_order)
        else:
            return cuest_to_pyscf_output_reorder_spherical(mol, S_cuest_order)

    def get_hcore(self, mol = None):
        assert mol is None or mol is self.mol or mol_equal(mol, self.mol)

        if not self.turn_on_cuest_hcore:
            return super().get_hcore(mol = mol)

        mol = self.mol
        assert not mol._pseudo, "Pseudo potential is not supported by CuEST"

        cuest_handle = self.handles.cuest_handle
        if self.handles.oeintplan_handle is None:
            self.handles.build_oeintplan(self)
        oeintplan_handle = self.handles.oeintplan_handle

        K1e_cuest_order = cuest_compute_kineticint(mol, cuest_handle, oeintplan_handle)

        xyzs = mol.atom_coords(unit = "B")
        xyzs = cp.asarray(xyzs, order = "C", dtype = cp.float64)
        charges = -mol.atom_charges()
        charges = cp.asarray(charges, order = "C", dtype = cp.float64)
        V1e_cuest_order = cuest_compute_potentialint(mol, xyzs, charges, cuest_handle, oeintplan_handle)

        Hcore_cuest_order = K1e_cuest_order + V1e_cuest_order

        if len(mol._ecpbas) > 0:
            if self.handles.ecpintplan_handle is None:
                self.handles.build_ecpintplan(self)
            ecpintplan_handle = self.handles.ecpintplan_handle

            ECP_cuest_order = cuest_compute_ecpint(mol, cuest_handle, ecpintplan_handle, self.maximum_workspace_bytes)
            Hcore_cuest_order += ECP_cuest_order

        if mol.cart:
            return cuest_to_pyscf_output_scale_cartesian(mol, Hcore_cuest_order)
        else:
            return cuest_to_pyscf_output_reorder_spherical(mol, Hcore_cuest_order)

    def get_jk(self, mol = None, dm = None, hermi=1, with_j=True, with_k=True, direct_scf_tol=None, omega=None):
        assert direct_scf_tol is None, "Direct SCF JK is not supported by cuEST yet"
        assert omega is None, "Range separated JK is not supported by cuEST yet"
        assert mol is None or mol is self.mol or mol_equal(mol, self.mol)

        if not self.turn_on_cuest_jk:
            # Notice, get_jk() function from DF does not have direct_scf_tol input
            return super().get_jk(mol = mol, dm = dm, hermi = hermi, with_j = with_j, with_k = with_k, omega = omega)

        mol = self.mol
        dms, dm = dm, None
        mo_coeffs = None
        mo_occs   = None
        if getattr(dms, 'mo_coeff', None) is not None:
            mo_coeffs = cp.asarray(dms.mo_coeff)
            mo_occs   = cp.asarray(dms.mo_occ)

        dm_original_shape = dms.shape
        if dms.ndim == 2:
            dms = dms[None, :]
            if mo_coeffs is not None:
                mo_coeffs = mo_coeffs[None, :]
                mo_occs   =   mo_occs[None, :]
        assert dms.ndim == 3
        assert dms.shape[-2:] == (mol.nao, mol.nao)
        n_dm = dms.shape[0]

        log = logger.new_logger(mol, mol.verbose)

        cuest_handle = self.handles.cuest_handle
        if self.handles.dfintplan_handle is None:
            self.handles.build_dfintplan(self)
        dfintplan_handle = self.handles.dfintplan_handle

        time_pre_post = 0
        time_kernel = 0
        vj = None
        vk = None
        if with_j:
            vj = cp.zeros_like(dms)
            for i_dm in range(n_dm):
                if mol.verbose >= logger.DEBUG:
                    cp.cuda.runtime.deviceSynchronize()
                    time0 = time.time()

                dm_copy = cp.array(dms[i_dm], order = "C", dtype = cp.float64) # cp.array gaurantees it makes a copy, so it's safe to modify it. Don't use cp.asarray here.
                if mol.cart:
                    dm_cuest_order = pyscf_to_cuest_input_scale_cartesian(mol, dm_copy)
                else:
                    dm_cuest_order = pyscf_to_cuest_input_reorder_spherical(mol, dm_copy)

                if mol.verbose >= logger.DEBUG:
                    cp.cuda.runtime.deviceSynchronize()
                    time1 = time.time()
                    time_pre_post += time1 - time0
                    time0 = time1

                J_cuest_order = cuest_compute_coulombmatrix(mol, dm_cuest_order, cuest_handle, dfintplan_handle)

                if mol.verbose >= logger.DEBUG:
                    cp.cuda.runtime.deviceSynchronize()
                    time1 = time.time()
                    time_kernel += time1 - time0
                    time0 = time1

                if mol.cart:
                    vj[i_dm] = cuest_to_pyscf_output_scale_cartesian(mol, J_cuest_order)
                else:
                    vj[i_dm] = cuest_to_pyscf_output_reorder_spherical(mol, J_cuest_order)

                if mol.verbose >= logger.DEBUG:
                    cp.cuda.runtime.deviceSynchronize()
                    time1 = time.time()
                    time_pre_post += time1 - time0
                    time0 = time1
            vj = vj.reshape(dm_original_shape)

        if with_k:
            numerical_zero = 1e-12

            vk = cp.zeros_like(dms)
            for i_dm in range(n_dm):
                if mol.verbose >= logger.DEBUG:
                    cp.cuda.runtime.deviceSynchronize()
                    time0 = time.time()

                if mo_coeffs is not None:
                    mo_coeff = cp.asarray(mo_coeffs[i_dm])
                    mo_occ   = cp.asarray(mo_occs[i_dm])
                    assert cp.max(cp.abs(mo_occ - cp.round(mo_occ))) < numerical_zero, "CuEST doesn't support fractional occupation yet."
                    mocc = mo_coeff[:, mo_occ > 0]

                    # Cuest assumes dm = mocc @ mocc.T
                    if n_dm == 1:
                        assert np.all((mo_occs == 2.0) | (mo_occs == 0.0))
                        mocc = mocc * np.sqrt(2)
                    elif n_dm == 2:
                        assert np.all((mo_occs == 1.0) | (mo_occs == 0.0))
                    else:
                        mocc = mocc * cp.sqrt(mo_occ[mo_occ > numerical_zero])

                    dm = cp.asarray(dms[i_dm])
                    assert cp.max(cp.abs(dm - mocc @ mocc.T)) < numerical_zero, "dm and mo_coeff are not consistent. CuEST doesn't support incremental SCF now."
                else:
                    log.warn("CuEST got a dm without mo_coeff tag, so it needs to recover mo_coeff using eigh, which is super slow.")

                    dm = cp.asarray(dms[i_dm])
                    assert cp.max(cp.abs(dm - dm.T)) < numerical_zero
                    mo_occ, mo_coeff = cp.linalg.eigh(dm)
                    assert all(mo_occ > -numerical_zero), f"Large negative eigenvalue ({min(mo_occ)}) found for density matrix."
                    mocc = mo_coeff[:, mo_occ > numerical_zero]

                    # Cuest assumes dm = mocc @ mocc.T
                    mocc = mocc * cp.sqrt(mo_occ[mo_occ > numerical_zero])

                mocc_copy = cp.array(mocc.T, order = "C", dtype = cp.float64) # cp.array gaurantees it makes a copy, so it's safe to modify it. Don't use cp.asarray here.
                if mol.cart:
                    mocc_cuest_order = pyscf_to_cuest_input_scale_cartesian(mol, mocc_copy, axis = 1)
                else:
                    mocc_cuest_order = pyscf_to_cuest_input_reorder_spherical(mol, mocc_copy, axis = 1)

                if mol.verbose >= logger.DEBUG:
                    cp.cuda.runtime.deviceSynchronize()
                    time1 = time.time()
                    time_pre_post += time1 - time0
                    time0 = time1

                K_cuest_order = cuest_compute_exchangematrix(mol, mocc_cuest_order, cuest_handle, dfintplan_handle, self.maximum_workspace_bytes, self.additional_precision_control_parameters)

                if mol.verbose >= logger.DEBUG:
                    cp.cuda.runtime.deviceSynchronize()
                    time1 = time.time()
                    time_kernel += time1 - time0
                    time0 = time1

                if mol.cart:
                    vk[i_dm] = cuest_to_pyscf_output_scale_cartesian(mol, K_cuest_order)
                else:
                    vk[i_dm] = cuest_to_pyscf_output_reorder_spherical(mol, K_cuest_order)

                if mol.verbose >= logger.DEBUG:
                    cp.cuda.runtime.deviceSynchronize()
                    time1 = time.time()
                    time_pre_post += time1 - time0
                    time0 = time1
            vk = vk.reshape(dm_original_shape)

            log.debug(f"CuEST: time_JK_pre_and_post_processing = {time_pre_post} s, time_JK_kernel = {time_kernel} s")

        return vj, vk

    def get_j(self, mol = None, dm = None, hermi=1, with_j=True, with_k=True, direct_scf_tol=None, omega=None):
        return self.get_jk(mol, dm, hermi, with_j=True, with_k=False, direct_scf_tol=direct_scf_tol, omega=omega)[0]

    def get_k(self, mol = None, dm = None, hermi=1, with_j=True, with_k=True, direct_scf_tol=None, omega=None):
        return self.get_jk(mol, dm, hermi, with_j=False, with_k=True, direct_scf_tol=direct_scf_tol, omega=omega)[1]

    def Gradients(self):
        # Attention: The self object has a composed class, and its first parent class is CuESTWrapper,
        # This means if you just use super(), this function will be called twice recursively.
        scf_grad = super(CuESTWrapper, self).Gradients()

        return apply_cuest_gradient_wrapper(scf_grad)

    def Hessian(self):
        raise NotImplementedError("CuEST does not support analytical hessian yet")

    def to_gpu(self):
        return self

    def to_cpu(self):
        raise NotImplementedError("CuESTWrapper does not support to_cpu() method. Please reconstruct your object.")

    def PCM(self):
        raise RuntimeError("Please apply PCM before CuESTWrapper.")

    def density_fit(self):
        raise RuntimeError("Please apply density_fit before CuESTWrapper.")

    def newton(self):
        raise NotImplementedError("CuESTWrapper does not support newton (soscf) method yet. "
                                  "In particular, DFT XC response function requires functional second derivative, and that is not available from CuEST.")

class CuESTGradientWrapper(lib.StreamObject):
    def __init__(self, method):
        self.__dict__.update(method.__dict__)

        mol = self.mol
        assert mol is not None
        assert getattr(self.base, "handles", None) is not None

    def get_ovlp_grad(self, dme):
        mol = self.mol

        assert dme is not None
        if dme.ndim == 3:
            assert dme.shape[0] == 2
            dme = dme[0] + dme[1]
        assert dme.shape == (mol.nao, mol.nao)

        if not self.base.turn_on_cuest_hcore:
            from gpu4pyscf.grad.rhf import contract_h1e_dm

            s1 = cp.asarray(super().get_ovlp(mol))
            ds = contract_h1e_dm(mol, s1, dme, hermi=1)
            return ds

        cuest_handle = self.base.handles.cuest_handle
        if self.base.handles.oeintplan_handle is None:
            self.base.handles.build_oeintplan(self.base)
        oeintplan_handle = self.base.handles.oeintplan_handle

        dme_copy = cp.array(dme, order = "C", dtype = cp.float64) # cp.array gaurantees it makes a copy, so it's safe to modify it. Don't use cp.asarray here.
        if mol.cart:
            dme_cuest_order = pyscf_to_cuest_input_scale_cartesian(mol, dme_copy)
        else:
            dme_cuest_order = pyscf_to_cuest_input_reorder_spherical(mol, dme_copy)

        ds = cuest_compute_overlap_gradient(mol, dme_cuest_order, cuest_handle, oeintplan_handle)

        return ds.get()

    def get_hcore_grad(self, dm):
        mol = self.mol

        assert not mol._pseudo, "GTH pseudopotential not supported in CuEST"

        assert dm is not None
        if dm.ndim == 3:
            assert dm.shape[0] == 2
            dm = dm[0] + dm[1]
        assert dm.shape == (mol.nao, mol.nao)

        if not self.base.turn_on_cuest_hcore:
            from gpu4pyscf.grad.rhf import int3c2e, get_ecp_ip, contract, contract_h1e_dm, ensure_numpy

            # (\nabla i | hcore | j) - (\nabla i | j)
            h1 = cp.asarray(super().get_hcore(mol, exclude_ecp=True))
            # (i | \nabla hcore | j)
            dh1e = int3c2e.get_dh1e(mol, dm)

            # Calculate ECP contributions in (i | \nabla hcore | j) and
            # (\nabla i | hcore | j) simultaneously
            if len(mol._ecpbas) > 0:
                # TODO: slice ecp_atoms
                ecp_atoms = sorted(set(mol._ecpbas[:,gto.ATOM_OF]))
                h1_ecp = get_ecp_ip(mol, ecp_atoms=ecp_atoms)
                h1 -= h1_ecp.sum(axis=0)
                dh1e[ecp_atoms] += 2.0 * contract('nxij,ij->nx', h1_ecp, dm)

            dh = contract_h1e_dm(mol, h1, dm, hermi=1)
            dh += ensure_numpy(dh1e)

            return dh

        cuest_handle = self.base.handles.cuest_handle
        if self.base.handles.oeintplan_handle is None:
            self.base.handles.build_oeintplan(self.base)
        oeintplan_handle = self.base.handles.oeintplan_handle

        dm_copy = cp.array(dm, order = "C", dtype = cp.float64) # cp.array gaurantees it makes a copy, so it's safe to modify it. Don't use cp.asarray here.
        if mol.cart:
            dm_cuest_order = pyscf_to_cuest_input_scale_cartesian(mol, dm_copy)
        else:
            dm_cuest_order = pyscf_to_cuest_input_reorder_spherical(mol, dm_copy)

        dh = cuest_compute_kinetic_gradient(mol, dm_cuest_order, cuest_handle, oeintplan_handle)

        xyzs = mol.atom_coords(unit = "B")
        xyzs = cp.asarray(xyzs, order = "C", dtype = cp.float64)
        charges = -mol.atom_charges()
        charges = cp.asarray(charges, order = "C", dtype = cp.float64)
        dv1e_orbital, dv1e_pointcharge = cuest_compute_potential_gradeint(mol, dm_cuest_order, xyzs, charges, cuest_handle, oeintplan_handle)

        dh += dv1e_orbital + dv1e_pointcharge

        if len(mol._ecpbas) > 0:
            if self.base.handles.ecpintplan_handle is None:
                self.base.handles.build_ecpintplan(self.base)
            ecpintplan_handle = self.base.handles.ecpintplan_handle

            decp = cuest_compute_ecp_gradient(mol, dm_cuest_order, cuest_handle, ecpintplan_handle, self.base.maximum_workspace_bytes)
            dh += decp

        return dh.get()

    def get_jk_grad(self, dm, k_factor = 1.0):
        if not self.base.turn_on_cuest_jk:
            return super().jk_energy_per_atom(dm = dm, j_factor = 1.0, k_factor = k_factor, omega = None, verbose = None)

        mol = self.mol

        log = logger.new_logger(mol, mol.verbose)

        dms, dm = dm, None
        mo_coeffs = None
        mo_occs   = None
        if getattr(dms, 'mo_coeff', None) is not None:
            mo_coeffs = cp.asarray(dms.mo_coeff)
            mo_occs   = cp.asarray(dms.mo_occ)

        if dms.ndim == 2:
            dms = dms[None, :]
            if mo_coeffs is not None:
                mo_coeffs = mo_coeffs[None, :]
                mo_occs   =   mo_occs[None, :]
        assert dms.ndim == 3
        assert dms.shape[-2:] == (mol.nao, mol.nao)
        n_dm = dms.shape[0]
        assert n_dm in (1, 2)

        cuest_handle = self.base.handles.cuest_handle
        if self.base.handles.dfintplan_handle is None:
            self.base.handles.build_dfintplan(self.base)
        dfintplan_handle = self.base.handles.dfintplan_handle

        time_pre_post = 0
        time_kernel = 0
        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time0 = time.time()

        numerical_zero = 1e-12

        if k_factor != 0:
            mocc_list = []
            for i_dm in range(n_dm):
                if mo_coeffs is not None:
                    mo_coeff = cp.asarray(mo_coeffs[i_dm])
                    mo_occ   = cp.asarray(mo_occs[i_dm])
                    assert cp.max(cp.abs(mo_occ - cp.round(mo_occ))) < numerical_zero, "CuEST doesn't support fractional occupation yet."
                    mocc = mo_coeff[:, mo_occ > 0]

                    # Cuest assumes dm = mocc @ mocc.T
                    mocc = mocc * cp.sqrt(mo_occ[mo_occ > 0])

                    dm = cp.asarray(dms[i_dm])
                    assert cp.max(cp.abs(dm - mocc @ mocc.T)) < numerical_zero, "dm and mo_coeff are not consistent. CuEST doesn't support incremental SCF now."
                else:
                    log.warn("CuEST got a dm without mo_coeff tag, so it needs to recover mo_coeff using eigh, which is super slow.")

                    dm = cp.asarray(dms[i_dm])
                    assert cp.max(cp.abs(dm - dm.T)) < numerical_zero
                    mo_occ, mo_coeff = cp.linalg.eigh(dm)
                    assert all(mo_occ > -numerical_zero), f"Large negative eigenvalue ({min(mo_occ)}) found for density matrix."
                    mocc = mo_coeff[:, mo_occ > numerical_zero]

                    mocc = mocc * cp.sqrt(mo_occ[mo_occ > numerical_zero])

                mocc_copy = cp.asarray(mocc.T, order = "C", dtype = cp.float64)
                if mol.cart:
                    mocc_cuest_order = pyscf_to_cuest_input_scale_cartesian(mol, mocc_copy, axis = 1)
                else:
                    mocc_cuest_order = pyscf_to_cuest_input_reorder_spherical(mol, mocc_copy, axis = 1)

                mocc_list.append(mocc_cuest_order)

            nocc = [mocc.shape[0] for mocc in mocc_list]
            mocc = cp.vstack(mocc_list)
        else:
            nocc = [1] # cuest doesn't allow nocc==0 even if k_factor==0
            mocc = cp.zeros((1, mol.nao))

        if n_dm == 1:
            dms = dms[0]
        else:
            dms = dms[0] + dms[1]

        dm_copy = cp.array(dms, order = "C", dtype = cp.float64) # cp.array gaurantees it makes a copy, so it's safe to modify it. Don't use cp.asarray here.
        if mol.cart:
            dm_cuest_order = pyscf_to_cuest_input_scale_cartesian(mol, dm_copy)
        else:
            dm_cuest_order = pyscf_to_cuest_input_reorder_spherical(mol, dm_copy)

        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time1 = time.time()
            time_pre_post += time1 - time0
            time0 = time1

        djk = cuest_compute_coulomb_exchange_gradient(mol, dm_cuest_order, mocc, nocc, k_factor, cuest_handle, dfintplan_handle, self.base.maximum_workspace_bytes)

        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time1 = time.time()
            time_kernel += time1 - time0
            time0 = time1
        log.debug(f"CuEST: time_JK_gradient_pre_and_post_processing = {time_pre_post} s, time_JK_gradient_kernel = {time_kernel} s")

        return djk.get()

    def get_xc_grad(self, dm):
        mol = self.mol

        if not self.base.turn_on_cuest_xc:
            ni = self.base._numint._backup_numint
            grids = self.base.grids._backup_grids
            if grids.coords is None:
                grids.build()
            mf = self.base

            if dm.ndim == 3:
                assert dm.shape == (2, mol.nao, mol.nao)
                from gpu4pyscf.grad.uks import get_exc, get_exc_full_response
            else:
                from gpu4pyscf.grad.rks import get_exc, get_exc_full_response

            if self.grid_response:
                exc, exc1 = get_exc_full_response(ni = ni, mol = mol, grids = grids, xc_code = mf.xc, dms = dm, verbose = None)
                exc1 *= 2
                exc1 += exc
            else:
                exc, exc1 = get_exc(ni = ni, mol = mol, grids = grids, xc_code = mf.xc, dms = dm, verbose = None)
                exc1 *= 2
            return exc1

        log = logger.new_logger(mol, mol.verbose)

        cuest_handle = self.base.handles.cuest_handle
        if self.base.handles.xcintplan_handle is None:
            self.base.handles.build_xcintplan(self.base)
        xcintplan_handle = self.base.handles.xcintplan_handle

        time_pre_post = 0
        time_kernel = 0
        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time0 = time.time()

        mocc_list = get_mocc_list_cuest_order_for_xc(mol, dm)

        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time1 = time.time()
            time_pre_post += time1 - time0
            time0 = time1

        dxc = cuest_compute_xc_gradient(mol, mocc_list, cuest_handle, xcintplan_handle, self.base.maximum_workspace_bytes)

        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time1 = time.time()
            time_kernel += time1 - time0
            time0 = time1
        log.debug(f"CuEST: time_XC_gradient_pre_and_post_processing = {time_pre_post} s, time_XC_gradient_kernel = {time_kernel} s")

        return dxc.get()

    def get_nlc_grad(self, dm):
        mol = self.mol

        if not self.base.turn_on_cuest_nlc:
            ni = self.base._numint._backup_numint
            grids = self.base.nlcgrids._backup_grids
            if grids.coords is None:
                grids.build()
            mf = self.base

            from gpu4pyscf.grad.rks import get_nlc_exc, get_nlc_exc_full_response

            if self.grid_response:
                enlc1_grid, enlc1_per_atom = get_nlc_exc_full_response(ni = ni, mol = mol, grids = grids, xc_code = mf.xc, dms = dm, verbose = None)
            else:
                enlc1_grid, enlc1_per_atom = get_nlc_exc(ni = ni, mol = mol, grids = grids, xc_code = mf.xc, dms = dm, verbose = None)
            exc1 = enlc1_per_atom * 2
            if self.grid_response:
                exc1 += enlc1_grid
            return exc1

        log = logger.new_logger(mol, mol.verbose)

        cuest_handle = self.base.handles.cuest_handle
        if self.base.handles.nlc_xcintplan_handle is None:
            self.base.handles.build_nlc_xcintplan(self.base)
        nlc_xcintplan_handle = self.base.handles.nlc_xcintplan_handle

        time_pre_post = 0
        time_kernel = 0
        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time0 = time.time()

        mocc_list = get_mocc_list_cuest_order_for_xc(mol, dm)

        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time1 = time.time()
            time_pre_post += time1 - time0
            time0 = time1

        dxc = cuest_compute_nlc_gradient(mol, mocc_list, cuest_handle, nlc_xcintplan_handle, self.base.maximum_workspace_bytes)

        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time1 = time.time()
            time_kernel += time1 - time0
            time0 = time1
        log.debug(f"CuEST: time_vv10_gradient_pre_and_post_processing = {time_pre_post} s, time_vv10_gradient_kernel = {time_kernel} s")

        return dxc.get()

    def energy_ee(self, mol=None, dm=None, verbose=None):
        assert mol is None or mol is self.mol or mol_equal(mol, self.mol)
        mol = self.mol
        mf = self.base
        if dm is None: dm = self.base.make_rdm1()

        log = logger.new_logger(mol, mol.verbose)

        if getattr(mf, "xc", None) is None:
            de = self.get_jk_grad(dm)
        else:
            assert getattr(self, "grid_response", None) is not None
            if self.grid_response is False:
                log.warn("CuEST does not support KS gradient without grid response, so the grid response is turned on. Same for NLC grid, if applicable.")
                self.grid_response = True

            de = self.get_xc_grad(dm)

            if mf.do_nlc():
                de += self.get_nlc_grad(dm)

            omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
            with_k = mf._numint.libxc.is_hybrid_xc(mf.xc)

            k_factor = 0.0
            if with_k:
                assert omega == 0, "Range-separated functional not supported in CuEST yet"
                k_factor = hyb

            de += self.get_jk_grad(dm, k_factor)

        return de

    def grad_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        mf = self.base
        mol = self.mol
        if atmlst is None:
            atmlst = range(mol.natm)
        assert list(atmlst) == list(range(mol.natm)), "Gradient splitting not supported in CuEST wrapper"

        if mo_energy is None: mo_energy = mf.mo_energy
        if mo_occ is None:    mo_occ = mf.mo_occ
        if mo_coeff is None:  mo_coeff = mf.mo_coeff

        log = logger.new_logger(mol, mol.verbose)

        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time0 = time.time()
            time_total_begin = time0

        mo_energy = cp.asarray(mo_energy)
        mo_occ = cp.asarray(mo_occ)
        mo_coeff = cp.asarray(mo_coeff)
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        dme0 = self.make_rdm1e(mo_energy, mo_coeff, mo_occ)

        ds = self.get_ovlp_grad(dme0)
        dh = self.get_hcore_grad(dm0)

        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time1 = time.time()
            log.debug(f"Time gradient of 1e part = {time1 - time0}")
            time0 = time1

        dvhf = self.energy_ee(mol, dm0)

        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time1 = time.time()
            log.debug(f"Time gradient of 2e part = {time1 - time0}")
            time0 = time1

        de = dh - ds + dvhf

        if mol.verbose >= logger.DEBUG:
            cp.cuda.runtime.deviceSynchronize()
            time1 = time.time()
            time_total_end = time1
            log.debug(f"Time gradient of electronic part = {time_total_end - time_total_begin}")

        return de

    def reset(self, mol = None):
        if mol is not None:
            self.mol = mol
        self.base.reset(mol)

    def to_gpu(self):
        return self

    def to_cpu(self):
        raise NotImplementedError("CuESTWrapper does not support to_cpu() method. Please reconstruct your object.")

def apply_cuest_wrapper(method):
    import gpu4pyscf
    assert (isinstance(method, gpu4pyscf.scf.hf.SCF))

    if isinstance(method, CuESTWrapper):
        method.reset()
        return method

    cls = CuESTWrapper

    return lib.set_class(cls(method), (cls, method.__class__))

def apply_cuest_gradient_wrapper(grad_method):
    import gpu4pyscf
    assert (isinstance(grad_method, gpu4pyscf.grad.rhf.GradientsBase))

    if isinstance(grad_method, CuESTGradientWrapper):
        grad_method.reset()
        return grad_method

    assert (isinstance(grad_method.base, gpu4pyscf.scf.hf.SCF) and
            isinstance(grad_method.base, CuESTWrapper))

    cls = CuESTGradientWrapper

    return lib.set_class(cls(grad_method), (cls, grad_method.__class__))



if __name__ == "__main__":
    # stream = cp.cuda.get_current_stream()

    mol = gto.M(
        atom = """
            O  0.0000  0.7375 -0.0528
            O  0.0000 -0.7375 -0.1528
            H  0.8190  0.8170  0.4220
            H -0.8190 -0.8170  0.4220
        """,
        basis = "def2-svp",
        charge = 0,
        spin = 0,
        verbose = 4,
        # cart = True,
    )

    # mf = mol.RHF().density_fit(auxbasis = "def2-universal-jkfit")

    mf = mol.RKS(xc = "PBE").density_fit(auxbasis = "def2-universal-jkfit")
    # mf.grids.becke_scheme = stratmann
    # mf.grids.radii_adjust = None
    # mf.nlcgrids.becke_scheme = stratmann
    # mf.nlcgrids.radii_adjust = None

    # mf = mf.PCM()
    # mf.with_solvent.surface_discretization_method = "ISWIG"

    mf = apply_cuest_wrapper(mf.to_gpu())
    mf.kernel()

    gobj = mf.Gradients()
    # gobj.grid_response = True
    gobj.kernel()
