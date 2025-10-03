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

import numpy as np
import cupy
from gpu4pyscf.lib import logger

try:
    import cupy_backends.cuda.libs.cutensor  # NOQA
    from cupyx import cutensor
    from cupy_backends.cuda.libs import cutensor as cutensor_backend
    ALGO_DEFAULT = cutensor_backend.ALGO_DEFAULT
    OP_IDENTITY = cutensor_backend.OP_IDENTITY
    JIT_MODE_NONE = cutensor_backend.JIT_MODE_NONE
    WORKSPACE_RECOMMENDED = cutensor_backend.WORKSPACE_MIN
    #WORKSPACE_RECOMMENDED = cutensor_backend.WORKSPACE_RECOMMENDED

    import ctypes
    libcutensor = ctypes.CDLL(cutensor_backend.__file__)
except (ImportError, AttributeError):
    cutensor = None
    ALGO_DEFAULT = None
    OP_IDENTITY = None
    JIT_MODE_NONE = None
    WORKSPACE_RECOMMENDED = None

def _auto_create_mode(array, mode):
    if not isinstance(mode, cutensor.Mode):
        mode = cutensor.create_mode(*mode)
    if array.ndim != mode.ndim:
        raise ValueError(
            'ndim mismatch: {} != {}'.format(array.ndim, mode.ndim))
    return mode

#def _create_tensor_descriptor(a):
#    handle = cutensor._get_handle()
#    key = (handle.ptr, a.dtype, tuple(a.shape), tuple(a.strides))
#    # hard coded
#    alignment_req = 8
#    if key not in _tensor_descriptors:
#        num_modes = a.ndim
#        extent = np.array(a.shape, dtype=np.int64)
#        stride = np.array(a.strides, dtype=np.int64) // a.itemsize
#        cutensor_dtype = cutensor._get_cutensor_dtype(a.dtype)
#        _tensor_descriptors[key] = cutensor.TensorDescriptor(
#            handle.ptr, num_modes, extent.ctypes.data, stride.ctypes.data,
#            cutensor_dtype, alignment_req=alignment_req)
#    return _tensor_descriptors[key]

def contraction(
    pattern, a, b, alpha, beta,
    out=None,
    op_a=OP_IDENTITY,
    op_b=OP_IDENTITY,
    op_c=OP_IDENTITY,
    algo=ALGO_DEFAULT,
    jit_mode=JIT_MODE_NONE,
    compute_desc=0,
    ws_pref=WORKSPACE_RECOMMENDED
):

    pattern = pattern.replace(" ", "")
    str_a, rest = pattern.split(',')
    str_b, str_c = rest.split('->')
    key = str_a + str_b
    val = list(a.shape) + list(b.shape)
    shape = {k:v for k, v in zip(key, val)}

    mode_a = list(str_a)
    mode_b = list(str_b)
    mode_c = list(str_c)
    if len(mode_c) != len(set(mode_c)):
        raise ValueError('Output subscripts string includes the same subscript multiple times.')

    dtype = np.result_type(a.dtype, b.dtype)
    a = cupy.asarray(a, dtype=dtype)
    b = cupy.asarray(b, dtype=dtype)
    if out is None:
        out = cupy.empty([shape[k] for k in str_c], order='C', dtype=dtype)
    c = out

    desc_a = cutensor.create_tensor_descriptor(a)
    desc_b = cutensor.create_tensor_descriptor(b)
    desc_c = cutensor.create_tensor_descriptor(c)

    mode_a = _auto_create_mode(a, mode_a)
    mode_b = _auto_create_mode(b, mode_b)
    mode_c = _auto_create_mode(c, mode_c)
    operator = cutensor.create_contraction(
        desc_a, mode_a, op_a, desc_b, mode_b, op_b, desc_c, mode_c, op_c,
        compute_desc)
    plan_pref = cutensor.create_plan_preference(algo=algo, jit_mode=jit_mode)
    ws_size = cutensor_backend.estimateWorkspaceSize(
        cutensor._get_handle().ptr, operator.ptr, plan_pref.ptr, ws_pref)
    plan = cutensor.create_plan(operator, plan_pref, ws_limit=ws_size)
    ws = cupy.empty(ws_size, dtype=np.int8)
    out = c

    alpha = np.asarray(alpha, dtype=dtype)
    beta = np.asarray(beta, dtype=dtype)

    handler = cutensor._get_handle()
    cutensor_backend.contract(handler.ptr, plan.ptr,
                             alpha.ctypes.data, a.data.ptr, b.data.ptr,
                             beta.ctypes.data, c.data.ptr, out.data.ptr,
                             ws.data.ptr, ws_size)
    return out

_contraction_operators = {}

# subclass OperationDescriptor, to overwrite the readonly attribute .ptr
class _OperationDescriptor(cutensor.OperationDescriptor):
    def __init__(self, ptr):
        assert isinstance(ptr, int)
        self.ctypes_ptr = ptr
        self._ptr = ptr

    @property
    def ptr(self):
        return self.ctypes_ptr

def _create_contraction_trinary(desc_a, mode_a, op_a, desc_b, mode_b, op_b,
                                desc_c, mode_c, op_c, desc_d, mode_d, op_d, compute_desc=0):
    handler = cutensor._get_handle()
    dtype = desc_d.cutensor_dtype
    key = (desc_a.ptr, mode_a.data,
           desc_b.ptr, mode_b.data,
           desc_c.ptr, mode_c.data,
           desc_d.ptr, mode_d.data, dtype)
    if key not in _contraction_operators:
        op_desc_ptr = ctypes.c_void_p()
        mode_a_ptr = ctypes.cast(mode_a.data, ctypes.POINTER(ctypes.c_int32))
        mode_b_ptr = ctypes.cast(mode_b.data, ctypes.POINTER(ctypes.c_int32))
        mode_c_ptr = ctypes.cast(mode_c.data, ctypes.POINTER(ctypes.c_int32))
        mode_d_ptr = ctypes.cast(mode_d.data, ctypes.POINTER(ctypes.c_int32))
        if dtype == 0 or dtype == 4:
            descCompute = ctypes.c_void_p.in_dll(libcutensor, "CUTENSOR_COMPUTE_DESC_32F")
        elif dtype == 1 or dtype == 5:
            descCompute = ctypes.c_void_p.in_dll(libcutensor, "CUTENSOR_COMPUTE_DESC_64F")
        else:
            raise RuntimeError(f'dtype {dtype} not supported')
        err = libcutensor.cutensorCreateContractionTrinary(
            ctypes.c_void_p(handler.ptr), ctypes.byref(op_desc_ptr),
            ctypes.c_void_p(desc_a.ptr), mode_a_ptr, ctypes.c_int(op_a),
            ctypes.c_void_p(desc_b.ptr), mode_b_ptr, ctypes.c_int(op_b),
            ctypes.c_void_p(desc_c.ptr), mode_c_ptr, ctypes.c_int(op_c),
            ctypes.c_void_p(desc_d.ptr), mode_d_ptr, ctypes.c_int(op_d),
            ctypes.c_void_p(desc_d.ptr), mode_d_ptr,
            descCompute)
        if err != cutensor_backend.STATUS_SUCCESS:
            raise RuntimeError(f'cutensorCreateContractionTrinary failed. err={err}')
        _contraction_operators[key] = _OperationDescriptor(op_desc_ptr.value)
    return _contraction_operators[key]

def _contract_trinary(pattern, a, b, c, alpha=1., beta=0., out=None):
    '''Three-tensor contraction
    out = alpha * A * B * C + beta * out
    '''
    pattern = pattern.replace(" ", "")
    str_ops, str_out = pattern.split('->')
    str_a, str_b, str_c = str_ops.split(',')
    key = str_a + str_b + str_c
    val = a.shape + b.shape + c.shape
    shape = {k:v for k, v in zip(key, val)}

    mode_a = list(str_a)
    mode_b = list(str_b)
    mode_c = list(str_c)
    mode_out = list(str_out)
    if len(mode_out) != len(set(mode_out)):
        raise ValueError('Output subscripts string includes the same subscript multiple times.')

    dtype = np.result_type(a.dtype, b.dtype, c.dtype)
    a = cupy.asarray(a, dtype=dtype)
    b = cupy.asarray(b, dtype=dtype)
    c = cupy.asarray(c, dtype=dtype)
    if out is None:
        out = cupy.empty([shape[k] for k in str_out], order='C', dtype=dtype)

    desc_a = cutensor.create_tensor_descriptor(a)
    desc_b = cutensor.create_tensor_descriptor(b)
    desc_c = cutensor.create_tensor_descriptor(c)
    desc_out = cutensor.create_tensor_descriptor(out)

    mode_a = _auto_create_mode(a, mode_a)
    mode_b = _auto_create_mode(b, mode_b)
    mode_c = _auto_create_mode(c, mode_c)
    mode_out = _auto_create_mode(out, mode_out)

    operator = _create_contraction_trinary(
        desc_a, mode_a, OP_IDENTITY, desc_b, mode_b, OP_IDENTITY,
        desc_c, mode_c, OP_IDENTITY, desc_out, mode_out, OP_IDENTITY)

    handler = cutensor._get_handle()
    algo = ALGO_DEFAULT
    jit_mode = JIT_MODE_NONE
    ws_pref = WORKSPACE_RECOMMENDED
    plan_pref = cutensor.create_plan_preference(algo=algo, jit_mode=jit_mode)
    ws_size = cutensor_backend.estimateWorkspaceSize(
        handler.ptr, operator.ptr, plan_pref.ptr, ws_pref)
    plan = cutensor.create_plan(operator, plan_pref, ws_limit=ws_size)
    ws = cupy.empty(ws_size, dtype=np.int8)

    alpha = np.asarray(alpha, dtype=dtype)
    beta = np.asarray(beta, dtype=dtype)
    stream = cupy.cuda.get_current_stream()
    err = libcutensor.cutensorContractTrinary(
        ctypes.c_void_p(handler.ptr), ctypes.c_void_p(plan.ptr),
        alpha.ctypes,
        ctypes.cast(a.data.ptr, ctypes.c_void_p),
        ctypes.cast(b.data.ptr, ctypes.c_void_p),
        ctypes.cast(c.data.ptr, ctypes.c_void_p),
        beta.ctypes,
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.cast(ws.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ws_size), ctypes.c_void_p(stream.ptr))
    if err != cutensor_backend.STATUS_SUCCESS:
        raise RuntimeError(f'cutensorContractTrinary failed. err={err}')
    return out

import os
contract_engine = None
if cutensor is None:
    contract_engine = 'cupy'  # default contraction engine
contract_engine = os.environ.get('CONTRACT_ENGINE', contract_engine)

# override the 'contract' function if einsum is customized or cutensor is not found
if contract_engine is not None:
    einsum = None
    if contract_engine == 'opt_einsum':
        import opt_einsum
        einsum = opt_einsum.contract
    elif contract_engine == 'cuquantum':
        from cuquantum import contract as einsum # type: ignore
    elif contract_engine == 'cupy':
        einsum = cupy.einsum
    else:
        raise RuntimeError('unknown tensor contraction engine.')

    import warnings
    warnings.warn(f'using {contract_engine} as the tensor contraction engine.')
    def contract(pattern, a, b, alpha=1.0, beta=0.0, out=None):
        try:
            if out is None:
                out = einsum(pattern, a, b)
                out *= alpha
            elif beta == 0.:
                out[:] = einsum(pattern, a, b)
                out *= alpha
            else:
                out *= beta
                tmp = einsum(pattern, a, b)
                tmp *= alpha
                out += tmp
        except cupy.cuda.memory.OutOfMemoryError:
            print('Out of memory error caused by cupy.einsum. '
                  'It is recommended to install cutensor to resolve this.')
            raise
        return cupy.asarray(out, order='C')

    def contract_trinary(pattern, a, b, c, alpha=1., beta=0., out=None):
        if out is None:
            out = einsum(pattern, a, b, c)
            out *= alpha
        elif beta == 0.:
            out[:] = einsum(pattern, a, b, c)
            out *= alpha
        else:
            out *= beta
            tmp = einsum(pattern, a, b, c)
            tmp *= alpha
            out += tmp
        return out
else:
    def contract(pattern, a, b, alpha=1.0, beta=0.0, out=None):
        '''
        a wrapper for general tensor contraction
        pattern has to be a standard einsum notation
        '''
        return contraction(pattern, a, b, alpha, beta, out=out)

    if cutensor_backend.get_version() < 20300:
        def contract_trinary(pattern, a, b, c, alpha=1., beta=0., out=None):
            raise RuntimeError('cutensor 2.3 or newer is required')
    else:
        contract_trinary = _contract_trinary
