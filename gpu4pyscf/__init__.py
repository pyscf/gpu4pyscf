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


def _rocm_preload_torch():
    """Work around a fatal LLVM clash between CuPy and PyTorch on ROCm/HIP.

    On ROCm, CuPy's hipRTC kernel compiler and PyTorch each load AMD's LLVM-based
    code-object manager (``libamd_comgr``). If CuPy is imported before torch, the
    two register the same LLVM command-line options twice and abort the whole
    process with::

        Option 'spirv-expand-step' registered more than once!
        LLVM ERROR: inconsistency in registered CommandLine options

    e.g. when combining classical gpu4pyscf DFT with the Skala ML functional in a
    single process. Importing torch first (when it is installed) makes its comgr
    load first; CuPy then reuses it by soname. No effect on CUDA (CuPy uses NVRTC)
    or when torch is absent. Disable with ``GPU4PYSCF_NO_TORCH_PRELOAD=1``.

    NB: this must not import CuPy -- importing CuPy before torch is itself enough
    to trigger the clash -- so ROCm is detected from the environment instead.
    """
    import os
    if os.environ.get('GPU4PYSCF_NO_TORCH_PRELOAD'):
        return
    on_rocm = (os.path.exists('/dev/kfd')                       # AMD GPU kernel iface
               or bool(os.environ.get('ROCM_PATH'))
               or bool(os.environ.get('HIP_PATH')))
    if not on_rocm:
        return
    try:
        import importlib.util
        if importlib.util.find_spec('torch') is not None:
            import torch  # noqa: F401
    except Exception:
        pass


_rocm_preload_torch()

from . import _patch_pyscf

from . import lib, grad, hessian, solvent, scf, dft, tdscf, nac

# Overwrite the cupy memory allocator. Make memory pool manage small-sized
# arrays only.
lib.cupy_helper.set_conditional_mempool_malloc()
