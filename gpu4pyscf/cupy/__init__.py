"""
CuPy-compatibility facade over dpnp for gpu4pyscf.

Exports a fake `cupy` module built from dpnp, patches CuPy-vs-dpnp API
differences in place, and sets up the `cupy.cuda` submodule (master
queue, Stream, Device, Event — see cupy/cuda.py).

Import order matters: `cuda.py` monkey-patches dpnp creation APIs to
inject sycl_queue=master. The aliases on `cupy_fake` must be bound
AFTER that patching or they'll reference the unwrapped originals —
see the rebind section near the bottom of this file.

Idempotency / single-load guarantee
-----------------------------------
This package can be reached through two dotted paths:
  - `cupy`              (because we install `cupy_fake` into sys.modules)
  - `gpu4pyscf.cupy`    (the real package path)
and similarly for the `.cuda` submodule. Without care, Python's import
machinery loads the file TWICE — once per name — producing two module
objects with two independent `_master_queues` registries and two
independent sets of wrappers on dpnp.

Fix: at the end of first-time init we alias BOTH names in sys.modules
to the same cupy_fake / cuda objects. A subsequent `import gpu4pyscf.cupy`
(or `import cupy`, or any variant of `.cuda`) then short-circuits in
the import cache and does not re-execute the file.

Namespace isolation
-------------------
`cupy_fake` is a *separate* `types.ModuleType` object, NOT the current
module. This matters: `setattr(cupy_fake, 'any', dpnp.any)` must only
populate the fake cupy namespace. If we used `sys.modules[__name__]`
as cupy_fake, the broad dpnp-attribute loop would overwrite Python
builtins (`any`, `max`, `sum`, `abs`, …) in this module's globals,
breaking every function defined here that calls `any(generator)` —
e.g. the numpy-einsum dispatcher. Keep cupy_fake separate.
"""
import os
import sys
import types
from abc import ABCMeta

import numpy as np
import dpnp
from dpnp.dpnp_array import dpnp_array
import dpnp.tensor as dpt


# =====================================================================
# Early short-circuit — if the facade has already been built under
# another name, just re-alias sys.modules and return it. This handles
# the rare case where Python manages to execute this file a second
# time despite the end-of-file aliasing (e.g. reload, stale finder).
# =====================================================================
_ALREADY_LOADED = None
for _candidate in ("cupy", "gpu4pyscf.cupy"):
    _m = sys.modules.get(_candidate)
    if _m is not None and getattr(_m, "__gpu4pyscf_cupy_facade__", False):
        _ALREADY_LOADED = _m
        break

if _ALREADY_LOADED is not None:
    # Redirect the current import to the pre-existing facade.
    sys.modules[__name__]            = _ALREADY_LOADED
    sys.modules["cupy"]              = _ALREADY_LOADED
    sys.modules["gpu4pyscf.cupy"]    = _ALREADY_LOADED
    _existing_cuda = getattr(_ALREADY_LOADED, "cuda", None)
    if _existing_cuda is not None:
        sys.modules["cupy.cuda"]           = _existing_cuda
        sys.modules["gpu4pyscf.cupy.cuda"] = _existing_cuda
else:
    # =================================================================
    # First-time initialization — build the fake cupy module.
    # =================================================================

    # -----------------------------------------------------------------
    # cupy.ndarray alias — callable with CuPy's memptr= kwarg,
    # isinstance-compatible with dpnp arrays.
    # -----------------------------------------------------------------
    def _resolve_dpnp_impl():
        try:
            import dpnp.dpnp_array as _mod
            return getattr(_mod, "dpnp_array", None)
        except Exception:
            return None


    _DPNP_ARRAY_IMPL = _resolve_dpnp_impl()


    class _CuPyNdarrayMeta(ABCMeta):
        """Supports `cupy.ndarray(shape, dtype=..., memptr=buf.data)` —
        the `memptr=` kwarg is CuPy-specific; dpnp uses `buffer=` and
        doesn't accept our raw CuPy MemoryPointer shim, so we unwrap it."""

        def __call__(cls, shape, dtype=np.float64, memptr=None):
            if memptr is not None and hasattr(memptr, "get_array"):
                memptr = memptr.get_array()
            if isinstance(shape, (tuple, list)):
                shape = tuple(int(s) for s in shape)
            else:
                shape = (int(shape),)
            if memptr is not None:
                return dpnp.ndarray(shape, dtype=dtype, buffer=memptr)
            return dpnp.ndarray(shape, dtype=dtype)

        def __instancecheck__(cls, obj):
            if isinstance(obj, dpnp.ndarray):
                return True
            if _DPNP_ARRAY_IMPL and isinstance(obj, _DPNP_ARRAY_IMPL):
                return True
            return False

        def __subclasscheck__(cls, sub):
            try:
                bases = [dpnp.ndarray]
                if _DPNP_ARRAY_IMPL:
                    bases.append(_DPNP_ARRAY_IMPL)
                return any(issubclass(sub, b) for b in bases)
            except TypeError:
                return False


    class _CuPyNdarray(dpnp.ndarray, metaclass=_CuPyNdarrayMeta):
        """Alias type for CuPy ndarray over dpnp arrays."""
        pass


    # -----------------------------------------------------------------
    # Build the fake cupy module — SEPARATE from the current module so
    # setattr doesn't pollute our globals. See the module docstring.
    # Give it package attributes so `from . import cuda` style imports
    # resolve correctly when the fake is looked up as `cupy`.
    # -----------------------------------------------------------------
    cupy_fake = types.ModuleType("cupy")
    cupy_fake.__package__ = "cupy"
    cupy_fake.__path__    = [os.path.dirname(os.path.abspath(__file__))]
    cupy_fake.__gpu4pyscf_cupy_facade__ = True   # marker for early short-circuit

    cupy_fake.ndarray = _CuPyNdarray
    cupy_fake.asnumpy = dpnp.asnumpy
    cupy_fake.einsum  = dpnp.einsum


    # -----------------------------------------------------------------
    # ndarray.dot(out=...) — fix a shape-mismatch edge case CuPy permits
    # but dpnp rejects. Guarded so a second execution is a no-op.
    # -----------------------------------------------------------------
    if not getattr(dpnp.ndarray.dot, "__gpu4pyscf_patched__", False):
        _original_ndarray_dot = dpnp.ndarray.dot

        def _ndarray_dot_method(self, b, out=None, _orig=_original_ndarray_dot):
            if out is None:
                return _orig(self, b, out=None)

            result = _orig(self, b, out=None)
            if result.shape != out.shape:
                if result.size == out.size:
                    result = result.squeeze()
                    if result.shape != out.shape:
                        result = result.reshape(out.shape)
                else:
                    raise ValueError(
                        f"Cannot fit result {result.shape} into {out.shape}")

            out[:] = result
            return out

        _ndarray_dot_method.__gpu4pyscf_patched__ = True
        dpnp.ndarray.dot = _ndarray_dot_method

    cupy_fake.dot = dpnp.dot


    # -----------------------------------------------------------------
    # Initial population of cupy_fake from dpnp (narrow, explicit list)
    # -----------------------------------------------------------------
    for _attr in (
        "append", "max", "linalg", "concatenate", "zeros", "ones",
        "empty", "eye", "view", "empty_like", "copyto", "cumsum", "any", "matmul",
        "vstack", "full", "arange", "stack", "expand_dims", "unique", "double",
        "sign", "argsort", "count_nonzero", "where", "split", "take", "tril", "log",
        "complex128", "uint8", "int32", "int64", "float32", "float64", "ravel",
        "random", "sum", "exp", "outer", "ix_", "pi", "square", "multiply",
        "diag_indices", "repeat", "diag", "tril_indices_from", "ceil", "newaxis",
        "ascontiguousarray", "nonzero", "array_equal", "isinf", "isnan", "dtype",
        "asfortranarray", "abs", "shape", "argmax", "trace", "prod",
    ):
        try:
            setattr(cupy_fake, _attr, getattr(dpnp, _attr))
        except AttributeError:
            pass


    # -----------------------------------------------------------------
    # cupy.cuda submodule — creates master queues, installs creation-API
    # wrappers on dpnp/dpt, patches dpctl's ContextVar cache, installs
    # in-place op drain. See cupy/cuda.py for details.
    # -----------------------------------------------------------------
    _cuda_mod = None
    try:
        from . import cuda as _cuda_mod
        cupy_fake.cuda = _cuda_mod
    except Exception as e:           # was: except ImportError
        raise ImportError(
            "gpu4pyscf.cupy.cuda failed to initialize the master SYCL queue"
        ) from e


    # -----------------------------------------------------------------
    # Rebind everything that cuda.py patched AFTER the patches.
    #
    # cuda.py's _wrap_with_master_queue() monkey-patched dpnp.asarray,
    # dpnp.zeros, etc., in place. The direct-alias loop above captured
    # the PRE-patch references and is now stale. Refresh every name on
    # dpnp so cupy.foo() reaches the patched (queue-injecting) version.
    #
    # Skip names that have custom cupy_fake shims later in this file —
    # those shims already call the patched dpnp.* internally and inherit
    # queue injection that way.
    # -----------------------------------------------------------------
    _CUSTOM_CUPY_FAKE_SHIMS = frozenset({
        "ndarray",
        "zeros", "zeros_like", "empty_like",
        "hstack", "vstack",
        "allclose", "sqrt", "tril_indices",
        "dot", "asarray", "array",
    })

    for _attr in dir(dpnp):
        if _attr.startswith("_") or _attr in _CUSTOM_CUPY_FAKE_SHIMS:
            continue
        _fn = getattr(dpnp, _attr, None)
        if callable(_fn):
            try:
                setattr(cupy_fake, _attr, _fn)
            except Exception:
                pass

    # Direct aliases for the most common creation APIs (post-patch).
    cupy_fake.asarray = dpnp.asarray
    cupy_fake.array   = dpnp.array

    # =================================================================
    # .get() / .set() — CuPy-style host <-> device transfer
    # =================================================================
    def _dpnp_set(self, host_array):
        self[...] = host_array


    def _dpnp_get(self, order='C'):
        host = self.asnumpy()
        if order == 'C':
            return np.ascontiguousarray(host)
        if order == 'F':
            return np.asfortranarray(host)
        if order == 'A':
            if host.flags['F_CONTIGUOUS'] and not host.flags['C_CONTIGUOUS']:
                return np.asfortranarray(host)
            return np.ascontiguousarray(host)
        if order == 'K':
            return np.array(host, order='K', copy=False)
        return np.ascontiguousarray(host)


    dpnp_array.set = _dpnp_set
    dpnp_array.get = _dpnp_get


    # =================================================================
    # hstack / vstack — cast numpy inputs to dpnp (CuPy does this, dpnp doesn't)
    # =================================================================
    def _to_dpnp_seq(seq):
        out = []
        for s in seq:
            if isinstance(s, (np.ndarray, np.generic)) and not isinstance(s, dpnp.ndarray):
                out.append(dpnp.asarray(s))
            else:
                out.append(s)
        return out


    def _hstack(tup, *, dtype=None, casting="same_kind"):
        return dpnp.hstack(_to_dpnp_seq(tup), dtype=dtype, casting=casting)


    def _vstack(tup, *, dtype=None, casting="same_kind"):
        return dpnp.vstack(_to_dpnp_seq(tup), dtype=dtype, casting=casting)


    cupy_fake.hstack = _hstack
    cupy_fake.vstack = _vstack


    # =================================================================
    # zeros wrapper — CuPy allows positional dtype; dpnp requires kwarg
    # =================================================================
    def _cupy_zeros(shape, dtype=None, order='C'):
        return dpnp.zeros(shape, dtype=dtype, order=order)


    cupy_fake.zeros = _cupy_zeros


    # =================================================================
    # zeros_like / empty_like — CuPy accepts np.ndarray input; dpnp doesn't
    # =================================================================
    def _norm_order(order):
        return 'C' if order in (None, 'K', 'A') else order


    def _shape_dtype_from(a, shape=None, dtype=None):
        if shape is None:
            try:
                shape = a.shape
            except Exception:
                shape = np.asarray(a).shape
        if dtype is None:
            try:
                dtype = a.dtype
            except Exception:
                dtype = np.asarray(a).dtype
        shape = tuple(int(s) for s in shape)
        return shape, np.dtype(dtype)


    def _zeros_like(a, dtype=None, order='K', subok=False, shape=None):
        if isinstance(a, np.ndarray):
            shape, dtype = _shape_dtype_from(a, shape, dtype)
            return dpnp.zeros(shape, dtype=dtype, order=_norm_order(order))
        return dpnp.zeros_like(a, dtype=dtype, order=_norm_order(order))


    def _empty_like(a, dtype=None, order='K', subok=False, shape=None):
        if isinstance(a, np.ndarray):
            shape, dtype = _shape_dtype_from(a, shape, dtype)
            return dpnp.empty(shape, dtype=dtype, order=_norm_order(order))
        return dpnp.empty_like(a, dtype=dtype, order=_norm_order(order))


    cupy_fake.zeros_like = _zeros_like
    cupy_fake.empty_like = _empty_like


    # =================================================================
    # allclose — CuPy accepts Python scalars; dpnp.allclose does not
    # Upstream: https://github.com/IntelPython/dpnp/issues/2566
    # =================================================================
    def _cupy_allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        if np.isscalar(a) and np.isscalar(b):
            return bool(np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))

        a_is_dp = isinstance(a, dpnp.ndarray)
        b_is_dp = isinstance(b, dpnp.ndarray)
        a_is_np = isinstance(a, np.ndarray)
        b_is_np = isinstance(b, np.ndarray)

        if (a_is_np or b_is_np) and not (a_is_dp and b_is_dp):
            if a_is_dp and b_is_np:
                return bool(np.allclose(a.asnumpy(), b, rtol=rtol, atol=atol, equal_nan=equal_nan))
            if a_is_np and b_is_dp:
                return bool(np.allclose(a, b.asnumpy(), rtol=rtol, atol=atol, equal_nan=equal_nan))
            return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))

        if a_is_dp and b_is_dp:
            return bool(dpnp.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))


    cupy_fake.allclose = _cupy_allclose


    # =================================================================
    # sqrt — CuPy accepts Python scalars; dpnp doesn't. Guarded.
    # =================================================================
    if not getattr(dpnp.sqrt, "__gpu4pyscf_patched__", False):
        _orig_dpnp_sqrt = dpnp.sqrt
        _SCALAR_TYPES = frozenset({int, float, complex, bool})

        def _patched_dpnp_sqrt(x, _orig=_orig_dpnp_sqrt, **kwargs):
            if type(x) in _SCALAR_TYPES:
                x = dpnp.array(x)
            return _orig(x, **kwargs)

        _patched_dpnp_sqrt.__gpu4pyscf_patched__ = True
        dpnp.sqrt = _patched_dpnp_sqrt

    cupy_fake.sqrt = dpnp.sqrt


    # =================================================================
    # numpy.einsum / numpy.dot — auto-dispatch to dpnp when any arg is
    # dpnp. Guarded. NOTE: `any` here is the Python builtin — we keep
    # cupy_fake separate from this module's globals specifically so that
    # stays true; see module docstring.
    # =================================================================
    def _convert_np_to_dpnp(arg):
        if isinstance(arg, str):
            return arg
        if isinstance(arg, np.ndarray) and not isinstance(arg, dpnp.ndarray):
            return dpnp.asarray(arg)
        return arg


    if not getattr(np.einsum, "__gpu4pyscf_patched__", False):
        _original_numpy_einsum = np.einsum

        def _numpy_einsum_with_dpnp(*args, _orig=_original_numpy_einsum, **kwargs):
            if any(isinstance(a, dpnp.ndarray) for a in args):
                return dpnp.einsum(*(_convert_np_to_dpnp(a) for a in args), **kwargs)
            return _orig(*args, **kwargs)

        _numpy_einsum_with_dpnp.__gpu4pyscf_patched__ = True
        np.einsum = _numpy_einsum_with_dpnp

    if not getattr(np.dot, "__gpu4pyscf_patched__", False):
        _original_numpy_dot = np.dot

        def _numpy_dot_with_dpnp(*args, _orig=_original_numpy_dot, **kwargs):
            if any(isinstance(a, dpnp.ndarray) for a in args):
                return dpnp.dot(*(_convert_np_to_dpnp(a) for a in args), **kwargs)
            return _orig(*args, **kwargs)

        _numpy_dot_with_dpnp.__gpu4pyscf_patched__ = True
        np.dot = _numpy_dot_with_dpnp


    # =================================================================
    # tril_indices — accept numpy.int64 etc.
    # =================================================================
    def _cupy_tril_indices(n, k=0, m=None):
        n = int(n)
        k = int(k)
        m = None if m is None else int(m)
        return dpnp.tril_indices(n, k=k, m=m)


    cupy_fake.tril_indices = _cupy_tril_indices


    # =================================================================
    # _LazyModule + cupy_backends stubs — defer loading onemkl_lapack
    # =================================================================
    class _LazyModule(types.ModuleType):
        def __init__(self, name, loader_func):
            super().__init__(name)
            self._loader_func = loader_func
            self._loaded = False
            self._real_module = None
            self.__path__ = []

        def _load(self):
            if not self._loaded:
                self._real_module = self._loader_func()
                self._loaded = True
            return self._real_module

        def __getattr__(self, name):
            if name.startswith('_'):
                raise AttributeError(name)
            real = self._load()
            if real is None:
                raise AttributeError(f"module has no attribute '{name}'")
            return getattr(real, name)

        def __dir__(self):
            real = self._load()
            return dir(real) if real is not None else []


    def _load_onemkl_lapack():
        try:
            from gpu4pyscf.lib import onemkl_lapack
            return onemkl_lapack
        except ImportError as e:
            import warnings
            warnings.warn(f"Could not import onemkl_lapack: {e}")
            return None


    def _setup_cupy_backends():
        if 'cupy_backends' in sys.modules:
            return

        cupy_backends = types.ModuleType('cupy_backends')
        cupy_backends.__path__ = []
        _cuda_submod = types.ModuleType('cupy_backends.cuda')
        _cuda_submod.__path__ = []
        cupy_backends.cuda = _cuda_submod
        libs = types.ModuleType('cupy_backends.cuda.libs')
        libs.__path__ = []
        _cuda_submod.libs = libs

        cublas = types.ModuleType('cupy_backends.cuda.libs.cublas')
        cublas.CUBLAS_FILL_MODE_LOWER = 0
        cublas.CUBLAS_FILL_MODE_UPPER = 1
        cublas.CUBLAS_OP_N = 0
        cublas.CUBLAS_OP_T = 1
        cublas.CUBLAS_OP_C = 2

        cusolver = _LazyModule('cupy_backends.cuda.libs.cusolver', _load_onemkl_lapack)
        libs.cusolver = cusolver
        libs.cublas   = cublas

        sys.modules['cupy_backends']                    = cupy_backends
        sys.modules['cupy_backends.cuda']               = _cuda_submod
        sys.modules['cupy_backends.cuda.libs']          = libs
        sys.modules['cupy_backends.cuda.libs.cusolver'] = cusolver
        sys.modules['cupy_backends.cuda.libs.cublas']   = cublas

        gpu4pyscf_cusolver = _LazyModule('gpu4pyscf.lib.cusolver', _load_onemkl_lapack)
        sys.modules['gpu4pyscf.lib.cusolver'] = gpu4pyscf_cusolver


    _setup_cupy_backends()
    del _setup_cupy_backends


    # =================================================================
    # Memory pool — reports actual SYCL device memory.
    #
    # Reference cuda through cupy_fake.cuda (closure-captured) rather
    # than re-importing — this works regardless of which name the cuda
    # module ended up registered under in sys.modules.
    # =================================================================
    _cuda_ref = _cuda_mod       # captured for the pool methods below


    class _MemoryPool:
        """Reports actual SYCL device memory usage via the cuda shim.

        used_bytes  = total - free
        free_bytes  = free memory
        total_bytes = total HBM/VRAM capacity

        All other methods are no-ops — dpnp has no user-managed memory pool.
        """

        def free_all_blocks(self): pass
        def free_all_free(self):   pass
        def set_limit(self, size=None, fraction=None): pass
        def get_limit(self):       return 0
        def n_free_blocks(self):   return 0

        def used_bytes(self):
            try:
                return _cuda_ref.get_total_memory() - _cuda_ref.get_free_memory()
            except Exception:
                return 0

        def free_bytes(self):
            try:
                return _cuda_ref.get_free_memory()
            except Exception:
                return 0

        def total_bytes(self):
            try:
                return _cuda_ref.get_total_memory()
            except Exception:
                return 0


    _memory_pool = _MemoryPool()


    def _get_default_memory_pool():
        return _memory_pool


    def _get_default_pinned_memory_pool():
        """Pinned memory has no equivalent under SYCL/dpnp; reuse the same pool."""
        return _memory_pool


    cupy_fake.get_default_memory_pool        = _get_default_memory_pool
    cupy_fake.get_default_pinned_memory_pool = _get_default_pinned_memory_pool

    if _cuda_mod is not None:
        _cuda_mod.PinnedMemoryPool = _MemoryPool


    # =================================================================
    # cupy.fuse — no-op under dpnp (CuPy kernel fusion not available)
    # =================================================================
    def fuse(*args, **kwargs):
        """No-op replacement for cupy.fuse.

        Supports:
            @cupy.fuse
            @cupy.fuse()
            @cupy.fuse(kernel_name='foo')
        """
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]                       # @cupy.fuse (no parens)
        return lambda func: func                 # @cupy.fuse(...) form


    cupy_fake.fuse = fuse


    # =================================================================
    # sys.modules aliasing — make `cupy`, `gpu4pyscf.cupy`, and their
    # `.cuda` submodules all resolve to the SAME module objects.
    #
    # This is what prevents the double-load: once both entries are set,
    # a subsequent `import gpu4pyscf.cupy` (or `import cupy`) finds the
    # facade in sys.modules and returns it without re-executing this file.
    #
    # We also redirect sys.modules[__name__] to cupy_fake so whoever is
    # currently waiting on this import gets the facade with all the
    # attributes (ndarray, asarray, cuda, ...) rather than the partially
    # populated current-module object.
    # =================================================================
    sys.modules["cupy"]           = cupy_fake
    sys.modules["gpu4pyscf.cupy"] = cupy_fake
    sys.modules[__name__]         = cupy_fake    # redirect current name too
    if _cuda_mod is not None:
        sys.modules["cupy.cuda"]           = _cuda_mod
        sys.modules["gpu4pyscf.cupy.cuda"] = _cuda_mod


# =================================================================
# cupy.fft submodule — direct aliases, dpnp signature is a superset
# =================================================================
_fft_mod = types.ModuleType("cupy.fft")
_fft_mod.__package__ = "cupy"
_fft_mod.__path__ = []

for _fname in (
    "fft",      "ifft",
    "fft2",     "ifft2",
    "fftn",     "ifftn",
    "rfft",     "irfft",
    "rfft2",    "irfft2",
    "rfftn",    "irfftn",
    "hfft",     "ihfft",
    "fftfreq",  "rfftfreq",
    "fftshift", "ifftshift",
):
    _fn = getattr(dpnp.fft, _fname, None)
    if _fn is not None:
        setattr(_fft_mod, _fname, _fn)

cupy_fake.fft = _fft_mod
sys.modules["cupy.fft"] = _fft_mod
sys.modules["gpu4pyscf.cupy.fft"] = _fft_mod
