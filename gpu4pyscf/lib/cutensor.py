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

import math
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

def _create_tensor_descriptor(a):
    if any(x == 0 for x in a.strides):
        strides = list(a.strides)
        if strides[0] == 0:
            strides[0] = a.nbytes
        for i, x in enumerate(strides[1:]):
            if x == 0:
                strides[i+1] = strides[i]
        a = cupy.ndarray(a.shape, a.dtype, a.data, strides)
    return cutensor.create_tensor_descriptor(a)

# Approximate limit on the elements in the operands and the output of one einsum
# call. cupy.einsum can copy its operands to the layout matmul needs and allocates
# its own output, while the memory budgets in the code assume contractions as in
# cuTENSOR.
EINSUM_MAX_SIZE = 1 << 24

def _einsum_slicing(pattern, a, b, out):
    '''Find the index to slice along when the temporaries of einsum exceed
    EINSUM_MAX_SIZE. Returns None if the contraction should be done in one call.
    '''
    if '->' not in pattern or '.' in pattern:
        return None
    if not all(isinstance(x, cupy.ndarray) for x in (a, b)):
        return None
    # One einsum call casts to out, the slices of a summed index are added to it
    if out is not None and np.result_type(a.dtype, b.dtype) != out.dtype:
        return None
    sub_a, rest = pattern.replace(' ', '').split(',')
    sub_b, sub_c = rest.split('->')
    dims = dict(zip(sub_a, a.shape))
    if any(dims.get(k, n) != n for k, n in zip(sub_b, b.shape)):
        return None # broadcasting
    dims.update(zip(sub_b, b.shape))
    out_size = math.prod(dims[k] for k in sub_c)
    tmp_size = a.size + b.size + out_size
    if tmp_size <= EINSUM_MAX_SIZE:
        return None
    if out is not None and (out.shape != tuple(dims[k] for k in sub_c) or
                            cupy.may_share_memory(out, a) or
                            cupy.may_share_memory(out, b)):
        return None

    def can_slice(k):
        return dims[k] > 1 and sub_a.count(k) < 2 and sub_b.count(k) < 2
    labels = [k for k in sub_c if can_slice(k)]
    # A summed index reduces both operands, but each slice has the full output
    summed = [k for k in sub_a if k in sub_b and k not in sub_c and can_slice(k)]
    if summed and (not labels or min(a.size, b.size) > out_size):
        labels = summed
    if not labels:
        return None
    # Prefer the index that reduces the most data
    k = max(labels, key=lambda k: (k in sub_a) * a.size + (k in sub_b) * b.size)
    step = max(dims[k] // math.ceil(tmp_size / EINSUM_MAX_SIZE), 1)
    return k, step, sub_a, sub_b, sub_c, dims

def _slice_along(a, subscripts, k, start, stop):
    if k not in subscripts:
        return a
    idx = [slice(None)] * a.ndim
    idx[subscripts.index(k)] = slice(start, stop)
    return a[tuple(idx)]

def _contract_einsum(pattern, a, b, alpha, beta, out=None, einsum=cupy.einsum):
    slicing = _einsum_slicing(pattern, a, b, out)
    if slicing is not None:
        k, step, sub_a, sub_b, sub_c, dims = slicing
        if out is None:
            out = cupy.empty([dims[x] for x in sub_c],
                             dtype=np.result_type(a.dtype, b.dtype))
            beta = 0.
        for i0 in range(0, dims[k], step):
            i1 = i0 + step
            _contract_einsum(pattern, _slice_along(a, sub_a, k, i0, i1),
                             _slice_along(b, sub_b, k, i0, i1), alpha, beta,
                             _slice_along(out, sub_c, k, i0, i1), einsum)
            if k not in sub_c:
                beta = 1
        return out

    if out is None:
        out = cupy.asarray(einsum(pattern, a, b), order='C')
        out *= alpha
    elif beta == 0.:
        out[...] = einsum(pattern, a, b)
        out *= alpha
    else:
        tmp = einsum(pattern, a, b)
        tmp *= alpha
        out *= beta
        out += tmp
    return out

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
    if a.size == 0 or b.size == 0:
        # cutensor does not support the 0-sized operands
        return _contract_einsum(pattern, a, b, alpha, beta, out)

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

    if a.size == 0 or b.size == 0 or c.size == 0:
        raise ValueError(f"cutensor contraction doesn't support zero-sized array (a.shape = {a.shape}, b.shape = {b.shape}, expected c.shape = {c.shape})")

    desc_a = _create_tensor_descriptor(a)
    desc_b = _create_tensor_descriptor(b)
    desc_c = _create_tensor_descriptor(c)

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
            return _contract_einsum(pattern, a, b, alpha, beta, out, einsum)
        except cupy.cuda.memory.OutOfMemoryError:
            print('Out of memory error caused by cupy.einsum. '
                  'It is recommended to install cutensor to resolve this.')
            raise
else:
    def contract(pattern, a, b, alpha=1.0, beta=0.0, out=None):
        '''
        a wrapper for general tensor contraction
        pattern has to be a standard einsum notation
        '''
        return contraction(pattern, a, b, alpha, beta, out=out)
