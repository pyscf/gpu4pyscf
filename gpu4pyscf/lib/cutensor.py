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
    _tensor_descriptors = {}
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
else:
    def contract(pattern, a, b, alpha=1.0, beta=0.0, out=None):
        '''
        a wrapper for general tensor contraction
        pattern has to be a standard einsum notation
        '''
        return contraction(pattern, a, b, alpha, beta, out=out)
