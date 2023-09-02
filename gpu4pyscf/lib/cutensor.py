# gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
#
# Copyright (C) 2022 Qiming Sun
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
import os
import ctypes
import numpy as np
import cupy
from cupyx import cutensor
from cupy.cuda import device
from cupy_backends.cuda.libs import cutensor as cutensor_lib
from cupy_backends.cuda.libs.cutensor import Handle

def load_library(libname):
    try:
        _loaderpath = os.path.dirname(__file__)
        return np.ctypeslib.load_library(libname, _loaderpath)
    except OSError:
        raise
libcupy_helper = load_library('libcupy_helper')

_handles = {}

class CutensorHandle(Handle):
    def __init__(self):
        Handle.__init__(self)
        cutensor_lib.init(self)
        cupy.cuda.runtime.deviceSynchronize()
        numCachelines = 1024
        libcupy_helper.create_plan_cache(
            ctypes.cast(self.ptr, ctypes.c_void_p),
            ctypes.c_int(numCachelines)
        )
    def __dealloc__(self):
        Handle.__dealloc__(self)
        libcupy_helper.delete_plan_cache(
            ctypes.cast(self.ptr, ctypes.c_void_p)
        )

def get_handle():
    dev = device.get_device_id()
    if dev not in _handles:
        handle = CutensorHandle()
        _handles[dev] = handle
        return handle
    return _handles[dev]

def create_contraction_descriptor(handle, 
                                  a, desc_a, mode_a, 
                                  b, desc_b, mode_b,
                                  c, desc_c, mode_c):
    alignment_req_A = cutensor_lib.getAlignmentRequirement(handle, a.data.ptr, desc_a)
    alignment_req_B = cutensor_lib.getAlignmentRequirement(handle, b.data.ptr, desc_b)
    alignment_req_C = cutensor_lib.getAlignmentRequirement(handle, c.data.ptr, desc_c)

    desc = cutensor_lib.ContractionDescriptor()
    cutensor_lib.initContractionDescriptor(
        handle,
        desc,
        desc_a, mode_a.data, alignment_req_A,
        desc_b, mode_b.data, alignment_req_B,
        desc_c, mode_c.data, alignment_req_C,
        desc_c, mode_c.data, alignment_req_C,
        cutensor_lib.COMPUTE_64F)
    return desc

def create_contraction_find(handle, algo=cutensor_lib.ALGO_DEFAULT):
    find = cutensor_lib.ContractionFind()
    cutensor_lib.initContractionFind(handle, find, algo)
    return find

def contraction(pattern, a, b, alpha, beta, out=None):
    pattern = pattern.replace(" ", "")
    str_a, rest = pattern.split(',')
    str_b, str_c = rest.split('->')
    key = str_a + str_b
    val = list(a.shape) + list(b.shape)
    shape = {k:v for k, v in zip(key, val)}
    
    mode_a = list(str_a)
    mode_b = list(str_b)
    mode_c = list(str_c)
    
    if(out is not None):
        c = out
    else:
        c = cupy.empty([shape[k] for k in str_c], order='C')
    
    handle = get_handle()

    desc_a = cutensor.create_tensor_descriptor(a)
    desc_b = cutensor.create_tensor_descriptor(b)
    desc_c = cutensor.create_tensor_descriptor(c)

    mode_a = cutensor.create_mode(*mode_a)
    mode_b = cutensor.create_mode(*mode_b)
    mode_c = cutensor.create_mode(*mode_c)
    
    out = c
    desc = create_contraction_descriptor(handle, a, desc_a, mode_a, b, desc_b, mode_b, c, desc_c, mode_c)
    find = create_contraction_find(handle)
    ws_size = cutensor_lib.contractionGetWorkspaceSize(handle, desc, find, cutensor_lib.WORKSPACE_RECOMMENDED)
    try:
        ws = cupy.empty(ws_size, dtype=np.int8)
    except:
        ws_size = cutensor_lib.contractionGetWorkspaceSize(handle, desc, find, cutensor_lib.WORKSPACE_MIN)
        ws = cupy.empty(ws_size, dtype=np.int8)
    
    plan = cutensor_lib.ContractionPlan()
    cutensor_lib.initContractionPlan(handle, plan, desc, find, ws_size)
    alpha = np.asarray(alpha)
    beta = np.asarray(beta)
    cutensor_lib.contraction(handle, plan,
                             alpha.ctypes.data, a.data.ptr, b.data.ptr,
                             beta.ctypes.data, c.data.ptr, out.data.ptr,
                             ws.data.ptr, ws_size)
    return out
    