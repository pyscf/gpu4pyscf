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

__version__ = '1.9.1'


def _check_rocm_import_order():
    '''Fail early, with instructions, when CuPy cannot compile GPU kernels on
    ROCm because torch was imported first.

    torch loads its own ROCm runtime compiler (hipRTC). If torch is imported
    before gpu4pyscf, CuPy reuses that hipRTC, which may be unable to build
    CuPy's kernels. Only runs on ROCm, when torch has already been imported.
    '''
    import sys
    if 'torch' not in sys.modules:
        return
    import cupy
    if not cupy.cuda.runtime.is_hip:
        return
    try:
        with open('/proc/self/maps') as f:
            hiprtc = sorted({line.split()[-1] for line in f if 'libhiprtc' in line})
    except OSError:
        hiprtc = []
    # Name the loaded hipRTC in the kernel source, so that CuPy's kernel cache
    # cannot hide a hipRTC that is unable to compile.
    probe = cupy.ElementwiseKernel(
        'float64 x', 'float64 y', f'y = x; /* hipRTC: {", ".join(hiprtc)} */',
        'gpu4pyscf_hiprtc_probe')
    try:
        probe(cupy.zeros(1))
    except cupy.cuda.compiler.CompileException as e:
        raise ImportError(
            'CuPy cannot compile GPU kernels: torch was imported before gpu4pyscf '
            'and loaded a ROCm runtime compiler (hipRTC) that CuPy cannot use '
            f'({", ".join(hiprtc) or "path unknown"}). Import gpu4pyscf before '
            'torch and before packages that import torch (e.g. skala), or '
            'preload the hipRTC that CuPy uses, e.g. '
            'LD_PRELOAD=$ROCM_PATH/lib/libhiprtc.so.7.') from e


_check_rocm_import_order()

from . import _patch_pyscf

from . import lib, grad, hessian, solvent, scf, dft, tdscf, nac

# Overwrite the cupy memory allocator. Make memory pool manage small-sized
# arrays only.
lib.cupy_helper.set_conditional_mempool_malloc()
