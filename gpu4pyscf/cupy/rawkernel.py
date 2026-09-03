"""
`cupy.RawKernel` / `cupy.RawModule` replacement for the SYCL/dpnp backend.

Background
----------
gpu4pyscf embeds literal CUDA C++ kernel sources in Python strings and
JIT-compiles them with `cupy.RawKernel` (NVRTC under the hood).  dpnp has no
equivalent, so those call sites die with

    AttributeError: module 'cupy' has no attribute 'RawKernel'

dpctl 0.23 exposes the DPC++ ``kernel_compiler`` extension through
``dpctl.program.create_kernel_bundle_from_sycl_source(q, source, ...)``, which
runtime-compiles a *SYCL* (not CUDA) source string into a kernel bundle.  The
kernels it can expose are "free function kernels": ``extern "C"`` functions
annotated with ``SYCL_EXTERNAL`` plus
``SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((sycl::ext::oneapi::experimental::nd_range_kernel<N>))``,
which obtain their work-item index from
``sycl::ext::oneapi::this_work_item::get_nd_item<N>()`` rather than from a
kernel-lambda parameter.

That is close enough to CUDA's ``__global__`` model that the *existing CUDA
kernel bodies compile unmodified* once a small compatibility prelude is
prepended: ``__global__`` becomes the annotation pair, and
``threadIdx``/``blockIdx``/``blockDim``/``gridDim`` become tiny structs that
pull from ``get_nd_item<3>()``.  So this shim does not require rewriting any
kernel source in the codebase -- it prepends the prelude and hands the result
to dpctl.

Dimension convention
--------------------
CUDA's fastest-varying dimension is ``x``; SYCL's is the *last* index of an
``nd_range``.  A 3D ``nd_item`` is therefore indexed as

    CUDA .x  ->  sycl dim 2
    CUDA .y  ->  sycl dim 1
    CUDA .z  ->  sycl dim 0

and a CUDA ``grid=(gx,gy,gz)``, ``block=(bx,by,bz)`` launch becomes a SYCL
global range ``[gz*bz, gy*by, gx*bx]`` with local range ``[bz, by, bx]``.  Both
the prelude and :func:`_launch` implement exactly this mapping; every kernel is
compiled as ``nd_range_kernel<3>`` so one prelude covers 1D/2D/3D launches.

Static shared memory
--------------------
``__shared__ T name[N];`` maps to
``static sycl::ext::oneapi::experimental::work_group_static<T[N]> name;``.  The
array bound moves *inside* the template argument, so an object-like macro
cannot express it; :func:`_rewrite_shared` does the rewrite textually instead.
Dynamic shared memory (CuPy's ``shared_mem=`` launch argument) has no
equivalent here and is rejected.
"""

import ctypes
import re
import threading

import numpy as np

import dpctl
import dpctl.memory as dpmem
import dpctl.program as dpprog


__all__ = ["RawKernel", "RawModule", "is_available"]


# ---------------------------------------------------------------------
# CUDA -> SYCL free-function-kernel compatibility prelude.
#
# Prepended verbatim to every source string.  Keeps the CUDA kernel bodies
# already in the codebase compilable as-is.
# ---------------------------------------------------------------------
_CUDA_COMPAT_PRELUDE = r'''
#include <sycl/sycl.hpp>
#include <complex>
#include <cmath>

namespace syclext = sycl::ext::oneapi::experimental;

namespace g4p_compat {

static inline sycl::nd_item<3> _it() {
    return sycl::ext::oneapi::this_work_item::get_nd_item<3>();
}

// CUDA .x is the fastest-varying dim; in SYCL that is the LAST index.
struct _ThreadIdx {
    int x, y, z;
    _ThreadIdx() { auto i = _it();
        x = (int)i.get_local_id(2); y = (int)i.get_local_id(1); z = (int)i.get_local_id(0); }
};
struct _BlockIdx {
    int x, y, z;
    _BlockIdx() { auto i = _it();
        x = (int)i.get_group(2); y = (int)i.get_group(1); z = (int)i.get_group(0); }
};
struct _BlockDim {
    int x, y, z;
    _BlockDim() { auto i = _it();
        x = (int)i.get_local_range(2); y = (int)i.get_local_range(1); z = (int)i.get_local_range(0); }
};
struct _GridDim {
    int x, y, z;
    _GridDim() { auto i = _it();
        x = (int)i.get_group_range(2); y = (int)i.get_group_range(1); z = (int)i.get_group_range(0); }
};

}  // namespace g4p_compat

#define threadIdx (g4p_compat::_ThreadIdx())
#define blockIdx  (g4p_compat::_BlockIdx())
#define blockDim  (g4p_compat::_BlockDim())
#define gridDim   (g4p_compat::_GridDim())
#define __syncthreads() sycl::group_barrier(g4p_compat::_it().get_group())
#define __syncwarp()    sycl::group_barrier(g4p_compat::_it().get_sub_group())
#define __global__ SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclext::nd_range_kernel<3>))
#define __device__ inline
#define __forceinline__ inline
#define __restrict__ __restrict__

// cupy/complex.cuh provides `complex<T>` in the global namespace; std::complex
// is device-usable under DPC++ and has the same interface for our uses.
using std::complex;
using std::exp;
using std::abs;
using std::fabs;
using std::sqrt;
using std::pow;
using std::log;
using std::sin;
using std::cos;

// CUDA-flavoured math spellings used by the embedded kernels. Defined as
// macros, not functions: glibc's <math.h> already declares a *narrowing*
// `float fsqrt(double)` (C2x TS-18661), so a `static inline double fsqrt`
// is a redeclaration error -- and picking up glibc's would silently halve
// the precision.
#define fsqrt(x)   (sycl::sqrt((double)(x)))
#define rsqrt(x)   (sycl::rsqrt((double)(x)))
#define rsqrtf(x)  (sycl::rsqrt((float)(x)))
#define __fdividef(a, b) ((float)(a) / (float)(b))
'''

# Source lines matched here are stripped before compiling: CUDA-only headers
# that have no SYCL counterpart but whose contents the prelude already covers.
_STRIP_INCLUDES = (
    "#include <cupy/complex.cuh>",
    "#include <cuda_runtime.h>",
    "#include <cuComplex.h>",
)


def is_available():
    """True if this dpctl/DPC++ build can runtime-compile SYCL source."""
    try:
        return bool(dpprog.is_sycl_source_compilation_available())
    except Exception:
        return False


# `__shared__ <type> <name>[<extent>];`
#   -> `static work_group_static<<type>[<extent>]> <name>;`
# The array bound has to move inside the template argument, so this cannot be
# a macro. <extent> may be any constant expression (e.g. TILE*TILE), and <type>
# any multi-word builtin ("unsigned short").
_SHARED_ARRAY_RE = re.compile(
    r"__shared__\s+([A-Za-z_][\w:\s]*?)\s+([A-Za-z_]\w*)\s*\[([^\]]+)\]\s*;"
)
# Scalar form: `__shared__ <type> <name>;`
_SHARED_SCALAR_RE = re.compile(
    r"__shared__\s+([A-Za-z_][\w:\s]*?)\s+([A-Za-z_]\w*)\s*;"
)


def _rewrite_shared(code):
    """Translate CUDA static __shared__ declarations to work_group_static."""
    code = _SHARED_ARRAY_RE.sub(
        r"static syclext::work_group_static<\1[\3]> \2;", code)
    code = _SHARED_SCALAR_RE.sub(
        r"static syclext::work_group_static<\1> \2;", code)
    if "__shared__" in code:
        raise NotImplementedError(
            "unrecognised __shared__ declaration form; dynamic (extern) "
            "shared memory is not supported by the SYCL "
            "free-function-kernel backend"
        )
    return code


def _preprocess(code):
    """Strip CUDA-only includes, rewrite __shared__, prepend the prelude."""
    for inc in _STRIP_INCLUDES:
        code = code.replace(inc, "")
    code = _rewrite_shared(code)
    return _CUDA_COMPAT_PRELUDE + "\n" + code


def _default_queue():
    """The queue dpnp allocations live on, so launches stay correctly ordered.

    gpu4pyscf.cupy.cuda installs a per-device singleton in-order master queue
    and injects it into every dpnp allocation. Use it when available so a
    JIT'd kernel is enqueued behind the dpnp work that produced its inputs.
    """
    try:
        from . import cuda as _cuda
        return _cuda._master_queue()
    except Exception:
        return dpctl.SyclQueue()


# ---------------------------------------------------------------------
# Kernel-argument marshaling
#
# dpctl.SyclQueue.submit accepts (see dpctl/_sycl_queue.pyx _populate_args):
#   ctypes c_char/c_uint8/c_short/c_ushort/c_int/c_uint/c_longlong/
#   c_ulonglong/c_float/c_double  -> by-value scalars
#   dpctl.memory._Memory                                 -> USM pointer
#   LocalAccessor / WorkGroupMemory / RawKernelArg       -> unused here
# Notably c_long is NOT accepted, so 64-bit ints must be c_longlong.
# ---------------------------------------------------------------------
_NP_TO_CTYPES = {
    np.dtype(np.int8):    ctypes.c_char,
    np.dtype(np.uint8):   ctypes.c_uint8,
    np.dtype(np.int16):   ctypes.c_short,
    np.dtype(np.uint16):  ctypes.c_ushort,
    np.dtype(np.int32):   ctypes.c_int,
    np.dtype(np.uint32):  ctypes.c_uint,
    np.dtype(np.int64):   ctypes.c_longlong,
    np.dtype(np.uint64):  ctypes.c_ulonglong,
    np.dtype(np.float32): ctypes.c_float,
    np.dtype(np.float64): ctypes.c_double,
}


def _as_usm_memory(arr):
    """USM buffer for a dpnp/dpctl array, honouring any view offset.

    `arr.usm_data` is the *base* allocation and ignores slicing offsets;
    `dpctl.memory.as_usm_memory` goes through __sycl_usm_array_interface__ and
    yields a buffer whose pointer is the array's own first element.
    """
    usm = arr.get_array() if hasattr(arr, "get_array") else arr
    flags = usm.flags
    if not (flags["C_CONTIGUOUS"] or flags["F_CONTIGUOUS"]):
        raise ValueError(
            "RawKernel arguments must be contiguous; got an array with "
            f"shape {usm.shape} strides {usm.strides}"
        )
    return dpmem.as_usm_memory(usm)


def _marshal(arg):
    # Already-marshaled USM buffers pass straight through. Checked FIRST:
    # _Memory also exposes __sycl_usm_array_interface__, so the array branch
    # below would otherwise claim it and then fail on the missing `.flags`.
    if isinstance(arg, (dpmem.MemoryUSMDevice, dpmem.MemoryUSMShared,
                        dpmem.MemoryUSMHost)):
        return arg
    # Arrays (dpnp.ndarray, dpctl.tensor.usm_ndarray, anything USM-aware)
    if hasattr(arg, "__sycl_usm_array_interface__") or hasattr(arg, "get_array"):
        return _as_usm_memory(arg)
    if isinstance(arg, ctypes._SimpleCData):
        return arg
    # numpy scalars and 0-d arrays carry their own C type
    if isinstance(arg, np.generic) or (isinstance(arg, np.ndarray) and arg.ndim == 0):
        dt = np.dtype(arg.dtype)
        try:
            return _NP_TO_CTYPES[dt](arg.item())
        except KeyError:
            raise TypeError(f"unsupported RawKernel scalar dtype {dt}")
    # Plain Python scalars: match CuPy, which widens int -> long long.
    if isinstance(arg, bool):
        return ctypes.c_char(int(arg))
    if isinstance(arg, int):
        return ctypes.c_longlong(arg)
    if isinstance(arg, float):
        return ctypes.c_double(arg)
    raise TypeError(f"unsupported RawKernel argument type {type(arg)}")


def _cuda_ranges_to_sycl(grid, block):
    """CUDA (gx,gy,gz)/(bx,by,bz) -> SYCL global/local range lists [z,y,x]."""
    def _pad(t):
        t = tuple(int(v) for v in (t if isinstance(t, (tuple, list)) else (t,)))
        return t + (1,) * (3 - len(t))
    gx, gy, gz = _pad(grid)
    bx, by, bz = _pad(block)
    return [gz * bz, gy * by, gx * bx], [bz, by, bx]


# ---------------------------------------------------------------------
# Compiled-bundle cache
#
# kernel_compiler invocation costs seconds, so bundles are memoised on
# (source, options, device). SYCL_CACHE_PERSISTENT additionally caches the
# device binary across processes.
# ---------------------------------------------------------------------
_bundle_cache = {}
_bundle_lock = threading.Lock()


def _get_bundle(code, options, queue):
    key = (code, tuple(options), queue.sycl_device.filter_string)
    with _bundle_lock:
        bundle = _bundle_cache.get(key)
        if bundle is not None:
            return bundle
        if not is_available():
            raise RuntimeError(
                "Runtime SYCL kernel compilation is unavailable in this "
                "DPC++/dpctl build (dpctl.program."
                "is_sycl_source_compilation_available() is False). Kernels "
                "that used cupy.RawKernel cannot be JIT-compiled; they must "
                "be built ahead of time into gpu4pyscf's native libraries."
            )
        bundle = dpprog.create_kernel_bundle_from_sycl_source(
            queue, _preprocess(code), headers=[], registered_names=[],
            copts=list(options))
        _bundle_cache[key] = bundle
        return bundle


class _KernelBase:
    """Shared launch machinery for RawKernel and RawModule.get_function."""

    def _resolve(self, queue):
        raise NotImplementedError

    def __call__(self, grid, block, args, shared_mem=0, stream=None,
                 enable_cooperative_groups=False):
        if shared_mem:
            raise NotImplementedError(
                "dynamic shared memory (shared_mem=) is not supported by the "
                "SYCL free-function-kernel backend")
        queue = None
        for a in args:
            q = getattr(a, "sycl_queue", None)
            if q is not None:
                queue = q
                break
        if queue is None:
            queue = _default_queue()

        kernel = self._resolve(queue)
        kargs = [_marshal(a) for a in args]
        n_expected = kernel.num_args
        if n_expected != len(kargs):
            raise ValueError(
                f"kernel {self.name!r} expects {n_expected} arguments, "
                f"got {len(kargs)}")
        gS, lS = _cuda_ranges_to_sycl(grid, block)
        # Blocking submit: keeps USM argument buffers alive for the duration
        # of the launch without a separate lifetime-tracking mechanism, and
        # matches the ordering the CUDA call sites assume.
        queue.submit(kernel, kargs, gS, lS)
        return None


class RawKernel(_KernelBase):
    """Drop-in replacement for `cupy.RawKernel` on the SYCL backend.

    Signature mirrors CuPy's; `backend`/`translate_cucomplex`/`jitify` and
    friends are accepted and ignored so call sites need no edits.
    """

    def __init__(self, code, name, options=(), backend="nvrtc",
                 translate_cucomplex=False, enable_cooperative_groups=False,
                 jitify=False, **kwargs):
        self.code = code
        self.name = name
        self.options = tuple(options)
        self._kernels = {}          # device filter string -> SyclKernel

    def _resolve(self, queue):
        key = queue.sycl_device.filter_string
        krn = self._kernels.get(key)
        if krn is None:
            bundle = _get_bundle(self.code, self.options, queue)
            if not bundle.has_sycl_kernel(self.name):
                raise RuntimeError(
                    f"compiled bundle has no kernel named {self.name!r}; the "
                    "kernel must be declared extern \"C\"")
            krn = bundle.get_sycl_kernel(self.name)
            self._kernels[key] = krn
        return krn

    # CuPy attribute-compatibility surface
    @property
    def kernel(self):
        return self._resolve(_default_queue())

    @property
    def max_threads_per_block(self):
        return self._resolve(_default_queue()).work_group_size

    @property
    def num_regs(self):
        return 0

    @property
    def shared_size_bytes(self):
        return 0

    @property
    def local_size_bytes(self):
        return self._resolve(_default_queue()).private_mem_size

    @property
    def attributes(self):
        k = self._resolve(_default_queue())
        return {
            "max_threads_per_block": k.work_group_size,
            "local_size_bytes": k.private_mem_size,
            "preferred_work_group_size_multiple":
                k.preferred_work_group_size_multiple,
        }


class _ModuleKernel(_KernelBase):
    def __init__(self, module, name):
        self._module = module
        self.name = name

    def _resolve(self, queue):
        key = queue.sycl_device.filter_string
        krn = self._module._kernels.get((key, self.name))
        if krn is None:
            bundle = _get_bundle(self._module.code, self._module.options, queue)
            if not bundle.has_sycl_kernel(self.name):
                raise RuntimeError(
                    f"compiled bundle has no kernel named {self.name!r}")
            krn = bundle.get_sycl_kernel(self.name)
            self._module._kernels[(key, self.name)] = krn
        return krn


class RawModule:
    """Drop-in replacement for `cupy.RawModule` on the SYCL backend.

    One source string may define several `extern "C" __global__` kernels; each
    is fetched with :meth:`get_function`, as in CuPy.
    """

    def __init__(self, code=None, path=None, options=(), backend="nvrtc",
                 translate_cucomplex=False, enable_cooperative_groups=False,
                 name_expressions=None, jitify=False, **kwargs):
        if code is None:
            if path is None:
                raise TypeError("RawModule requires either code= or path=")
            with open(path) as f:
                code = f.read()
        self.code = code
        self.options = tuple(options)
        self._kernels = {}
        self._functions = {}

    def get_function(self, name):
        fn = self._functions.get(name)
        if fn is None:
            fn = _ModuleKernel(self, name)
            self._functions[name] = fn
        return fn
