"""
Single-queue-per-device SYCL runtime shim for gpu4pyscf.

Design invariant
----------------
Exactly ONE dpctl.SyclQueue lives per GPU for the lifetime of the process.
Every allocation (Python-side via dpnp/dpctl, C++-side via libgsycl.so)
must land on that master queue.

Enforcement layers (defence in depth)
-------------------------------------
1. Master queue registry — `_master_queue(d)` creates the singleton
   in-order queue for device `d` on first call, registers its native
   pointer with libgsycl.so, and caches it forever.

2. Global queue-cache replacement — dpctl's `_global_device_queue_cache`
   is a ContextVar, which does NOT propagate into ThreadPoolExecutor
   worker threads. We replace the ContextVar object itself with a plain
   thread-shared shim so every thread sees the master queue.

3. Creation-API wrappers — every dpnp and dpctl.tensor array-creation
   function is wrapped to inject `sycl_queue=master` unless the caller
   has explicitly placed the allocation.

Idempotency / reload-safety
---------------------------
This module can be imported under two dotted names: `cupy.cuda` (when
we're loaded through the gpu4pyscf.cupy facade that re-exports as
`cupy`) and `gpu4pyscf.cupy.cuda` (the real dotted path). Both names
are aliased in gpu4pyscf/cupy/__init__.py, but as belt-and-suspenders
this file stashes its mutable state (master queue registry, device
cache, stream cache) on the `dpnp` module — which is guaranteed to
load exactly once — so even if we execute twice we don't duplicate
the master queue or install the wrappers twice.

Verification
------------
`_verify_single_queue_invariant()` runs once at import and proves:
  - libgsycl's queue pointer matches the Python master per device,
  - main-thread dpnp allocations land on master,
  - worker-thread dpnp allocations land on master (catches regressions
    in layer 2).

The invariant is checked by native-handle equality (`addressof_ref()`),
not Python-object identity, because dpnp internals may reconstruct a
fresh SyclQueue Python wrapper around the same underlying sycl::queue.
"""
import atexit
import ctypes
import functools
import os
import threading
import time
import warnings

import dpctl
import dpctl.memory as dpmem
# import dpctl.utils
# import dpctl.utils as dputils
import dpctl._sycl_queue_manager as qmgr
import dpnp


_DEFERRED_FREE_THRESHOLD = int(
    os.environ.get("GPU4PYSCF_DEFERRED_FREE_THRESHOLD", "256")
)


# =====================================================================
# Shared, reload-safe state — stashed on dpnp (which loads once).
#
# If this file gets executed twice (two distinct module objects under
# two names), both copies share the same registry, the same device
# cache, and the same "bootstrapped" flag, so _bootstrap() runs its
# side effects exactly once.
# =====================================================================
_STATE_ATTR = "__gpu4pyscf_cuda_state__"
_state = getattr(dpnp, _STATE_ATTR, None)
if _state is None:
    _state = {
        "master_lock":        threading.Lock(),
        "master_queues":      {},      # int -> dpctl.SyclQueue
        "gpu_devices":        None,    # cached device list
        "stream_cache":       {},      # int -> Stream
        "stream_cache_lock":  threading.Lock(),
        "device_cache":       {},      # int -> Device
        "device_cache_lock":  threading.Lock(),
        "bootstrapped":       False,
        "verified":           False,
        "shutting_down":      False,
    }
    setattr(dpnp, _STATE_ATTR, _state)

_master_lock       = _state["master_lock"]
_master_queues     = _state["master_queues"]
_stream_cache      = _state["stream_cache"]
_stream_cache_lock = _state["stream_cache_lock"]
_device_cache      = _state["device_cache"]
_device_cache_lock = _state["device_cache_lock"]


# =====================================================================
# libgsycl.so — the C++ side's master-queue registry
# =====================================================================
_lib_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../lib/libgsycl.so"))
libgpu = ctypes.CDLL(_lib_path)

# Bindings must match sycl_api_python.cpp exactly.
libgpu.sycl_get_device_id.argtypes     = []
libgpu.sycl_get_device_id.restype      = ctypes.c_int
libgpu.sycl_get_queue_ptr.argtypes     = []
libgpu.sycl_get_queue_ptr.restype      = ctypes.c_void_p
libgpu.sycl_set_queue_ptr.argtypes     = [ctypes.c_int, ctypes.c_void_p]
libgpu.sycl_set_queue_ptr.restype      = None
libgpu.sycl_set_device.argtypes        = [ctypes.c_int]
libgpu.sycl_set_device.restype         = None
libgpu.sycl_get_total_memory.argtypes  = []
libgpu.sycl_get_total_memory.restype   = ctypes.c_size_t
libgpu.sycl_get_shared_memory.argtypes = []
libgpu.sycl_get_shared_memory.restype  = ctypes.c_size_t
libgpu.sycl_get_compute_units.argtypes = []
libgpu.sycl_get_compute_units.restype  = ctypes.c_int
libgpu.sycl_get_device_name.argtypes   = [ctypes.c_char_p, ctypes.c_int]
libgpu.sycl_get_device_name.restype    = None
libgpu.sycl_get_free_memory.argtypes   = []
libgpu.sycl_get_free_memory.restype    = ctypes.c_size_t
libgpu.sycl_memcpy.argtypes            = [ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_size_t]
libgpu.sycl_memcpy.restype             = ctypes.c_size_t


class classproperty:
    def __init__(self, fget):
        self.fget = fget
    def __get__(self, obj, owner):
        return self.fget(owner)


# =====================================================================
# Queue pointer helper
# =====================================================================
def _get_sycl_queue_ptr(q: dpctl.SyclQueue) -> int:
    """Return the actual sycl::queue* as an integer.

    DPCTLSyclQueueRef is a typedef for sycl::queue*, and
    SyclQueue.addressof_ref() returns its value cast to size_t —
    i.e. the sycl::queue* itself. sycl_set_queue_ptr does a direct
    static_cast<sycl::queue*>, so we pass the value as-is.

    q must remain alive for the lifetime of the stored pointer —
    _master_queues guarantees this for master queues.
    """
    return int(q.addressof_ref())


# =====================================================================
# Master-queue registry
# =====================================================================
def _gpu_devices():
    """Enumerate GPU devices once (prefer level_zero). Cached in _state."""
    if _state["gpu_devices"] is not None:
        return _state["gpu_devices"]
    try:
        devs = dpctl.get_devices(backend="level_zero", device_type="gpu")
    except Exception:
        devs = []
    if not devs:
        try:
            devs = dpctl.get_devices(device_type="gpu")
        except Exception:
            devs = dpctl.get_devices()
    _state["gpu_devices"] = devs
    return devs

def _master_queue(device_id=None):
    """Return the singleton master in-order SyclQueue for a device.

    First call creates the queue and registers its native sycl::queue*
    with libgsycl.so so low-level kernel launches run on the same
    in-order queue dpnp/dpctl use. Cached for the process lifetime, which
    keeps the pointer valid. dpctl defers USM frees on in-order queues
    (queue-ordered host task), so allocations are not released while
    kernels enqueued here -- including these C++ launches -- still use them.
    """
    if device_id is None:
        device_id = int(libgpu.sycl_get_device_id())
    with _master_lock:
        q = _master_queues.get(device_id)
        if q is not None:
            return q
        devs = _gpu_devices()
        if device_id < 0 or device_id >= len(devs):
            raise ValueError(
                f"device_id {device_id} out of range (have {len(devs)} GPUs)")
        q = dpctl.SyclQueue(devs[device_id], property="in_order")
        libgpu.sycl_set_queue_ptr(
            ctypes.c_int(device_id),
            ctypes.c_void_p(_get_sycl_queue_ptr(q)))
        _master_queues[device_id] = q   # keeps q alive -> pointer stays valid
        return q
# def _master_queue(device_id=None):
#     if device_id is None:
#         device_id = int(libgpu.sycl_get_device_id())
#     with _master_lock:
#         q = _master_queues.get(device_id)
#         if q is None:
#             devs = _gpu_devices()
#             if device_id < 0 or device_id >= len(devs):
#                 raise ValueError(
#                     f"device_id {device_id} out of range (have {len(devs)} GPUs)")
#             q = dpctl.SyclQueue(devs[device_id], property="in_order")

#             # Register before any USM allocation; idempotent.
#             dputils.register_externally_shared_queue(q)

#             libgpu.sycl_set_queue_ptr(
#                 ctypes.c_int(device_id),
#                 ctypes.c_void_p(int(q.addressof_ref())))
#             _master_queues[device_id] = q
#         else:
#             # Belt-and-suspenders: catch the case where _master_queues was
#             # populated by a path that skipped registration.
#             if not hasattr(q, '_dpctl_deferred_free_pool'):
#                 dputils.register_externally_shared_queue(q)
#         return q
    
# def _master_queue(device_id=None):
#     if device_id is None:
#         device_id = int(libgpu.sycl_get_device_id())
#     with _master_lock:
#         q = _master_queues.get(device_id)
#         if q is not None:
#             return q
#         devs = _gpu_devices()
#         if device_id < 0 or device_id >= len(devs):
#             raise ValueError(
#                 f"device_id {device_id} out of range (have {len(devs)} GPUs)")
#         q = dpctl.SyclQueue(devs[device_id], property="in_order")

#         libgpu.sycl_set_queue_ptr(
#             ctypes.c_int(device_id),
#             ctypes.c_void_p(_get_sycl_queue_ptr(q))  # ← fixed
#         )
#         _master_queues[device_id] = q   # keeps q alive → pointer stays valid
#         return q

# def _master_queue(device_id=None):
#     """Return the singleton master in-order SyclQueue for a device.

#     First call creates the queue, registers its native pointer with
#     libgsycl.so, and caches it. Subsequent calls are O(1).
#     """
#     if device_id is None:
#         device_id = int(libgpu.sycl_get_device_id())
#     with _master_lock:
#         q = _master_queues.get(device_id)
#         if q is not None:
#             return q
#         devs = _gpu_devices()
#         if device_id < 0 or device_id >= len(devs):
#             raise ValueError(
#                 f"device_id {device_id} out of range (have {len(devs)} GPUs)")
#         q = dpctl.SyclQueue(devs[device_id], property="in_order")
#         libgpu.sycl_set_queue_ptr(
#             ctypes.c_int(device_id),
#             ctypes.c_void_p(q.addressof_ref()))
#         _master_queues[device_id] = q
#         return q


def master_device(device_id=None):
    """Public accessor for the master SyclQueue of a device.

    Pass this anywhere code needs an explicit ``sycl_queue=``.
    """
    return _master_queue(device_id)


def _same_queue(q1, q2):
    if q1 is None or q2 is None:
        return False
    if q1 is q2:
        return True
    try:
        return _get_sycl_queue_ptr(q1) == _get_sycl_queue_ptr(q2)  # compare sycl::queue*
    except Exception:
        return False


# =====================================================================
# Layer 2 — replace dpctl's per-thread queue cache
# =====================================================================
class _InOrderQueueCache:
    """Stand-in for dpctl's internal _DeviceDefaultQueueCache that
    always returns the master in-order queue for any device query.

    get_or_create(device) must return (SyclQueue, is_newly_created).
    We always return (master, False) — master pre-existed this call.
    """
    __slots__ = ("_q",)

    def __init__(self, master_queue):
        self._q = master_queue

    def get_or_create(self, device):
        return (self._q, False)


class _GlobalQueueCacheShim:
    """Thread-shared stand-in for dpctl's ContextVar-based queue cache.

    A ContextVar set on the main thread is invisible to worker threads
    spawned by ThreadPoolExecutor — they start in a fresh default
    context. We replace the ContextVar object itself so every thread
    resolves `.get()` to the same in-order cache.

    Mimics the minimum ContextVar surface (get/set/reset) needed by
    dpctl internals. `set()` returns the OLD value as the token, and
    `reset()` restores it — this matches how ContextVar is used inside
    context-manager patterns (token = var.set(x); try: ...; finally:
    var.reset(token)). A no-op reset would leave the cache permanently
    overwritten on first scope exit.
    """
    __slots__ = ("_cache",)

    def __init__(self, cache):
        self._cache = cache

    def get(self, *default):
        return self._cache

    def set(self, value):
        token = self._cache     # old value IS the token
        self._cache = value
        return token

    def reset(self, token):
        self._cache = token     # LIFO restore

# =====================================================================
# Layer 3 — wrap every creation API so sycl_queue=master is injected
# =====================================================================
_DPNP_CREATION = (
    "asarray", "array", "zeros", "ones", "empty", "full",
    "zeros_like", "ones_like", "empty_like", "full_like",
    "arange", "linspace", "logspace", "geomspace",
    "eye", "identity", "tri", "frombuffer", "fromfunction",
    "copy",
)


def _wrap_with_master_queue(mod, names):
    """Inject sycl_queue=master into every creation call on `mod`.

    Idempotent: re-wrapping a wrapped function is a no-op.
    """
    for name in names:
        orig = getattr(mod, name, None)
        if orig is None or getattr(orig, "__master_q_wrapped__", False):
            continue

        @functools.wraps(orig)
        def wrapper(*args, _orig=orig, **kwargs):
            if "sycl_queue" not in kwargs and "device" not in kwargs:
                kwargs["sycl_queue"] = _master_queue()
            return _orig(*args, **kwargs)

        wrapper.__master_q_wrapped__ = True
        wrapper.__wrapped__ = orig
        setattr(mod, name, wrapper)

# =====================================================================
# Bootstrap — install layers 1-3. Guarded by _state["bootstrapped"]
# so a second execution of this file is a no-op.
# =====================================================================
def _bootstrap():
    if _state["bootstrapped"]:
        return

    # Layer 1: create master queues eagerly for every GPU.
    for d in range(len(_gpu_devices())):
        try:
            _master_queue(d)
        except Exception as e:
            warnings.warn(
                f"Failed to install master queue for device {d}: {e}",
                RuntimeWarning)

    # Layer 2: replace the ContextVar with a thread-shared shim —
    # but only if not already replaced by a previous load.
    try:
        existing = qmgr._global_device_queue_cache
        if not isinstance(existing, _GlobalQueueCacheShim):
            qmgr._global_device_queue_cache = _GlobalQueueCacheShim(
                _InOrderQueueCache(_master_queue(0))
            )
        probe = dpnp.zeros(4)
        if not _same_queue(probe.sycl_queue, _master_queue(0)):
            warnings.warn(
                "Layer 2: dpnp allocation did NOT land on the master queue. "
                "Queue-cache shim install may have failed.",
                RuntimeWarning,
            )
    except Exception as e:
        warnings.warn(
            f"Failed to replace dpctl device queue cache: {e}",
            RuntimeWarning)

    # Layer 3: wrap creation APIs.
    _wrap_with_master_queue(dpnp, _DPNP_CREATION)

    _state["bootstrapped"] = True


_bootstrap()


# import dpnp.tensor._ctors as _ctors
# from dpnp.tensor._device import normalize_queue_device as _nqd

# _orig_asarray_from_numpy = _ctors._asarray_from_numpy_ndarray

# def _q_ptr(q):
#     if q is None:
#         return None
#     ref = q.addressof_ref
#     return int(ref() if callable(ref) else ref)

# def _fmt_ptr(p):
#     """Format a pointer int or None as a hex string."""
#     return f"{p:#x}" if p is not None else "None"

# def _probe_asarray_from_numpy(ary, dtype=None, usm_type=None, sycl_queue=None, order="K"):
#     resolved = _nqd(sycl_queue=None, device=sycl_queue)
#     master   = _master_queue()

#     q_in_ptr  = _q_ptr(sycl_queue)
#     q_out_ptr = _q_ptr(resolved)
#     q_mst_ptr = _q_ptr(master)

#     match_master   = (q_out_ptr == q_mst_ptr)
#     match_supplied = (q_in_ptr == q_out_ptr) if q_in_ptr is not None else "n/a (scalar/None input)"

#     # Context-level check — most critical for your segfault hypothesis
#     try:
#         same_ctx = (resolved.sycl_context == master.sycl_context)
#     except Exception as e:
#         same_ctx = f"ERROR: {e}"

#     import traceback
#     print(
#         f"\n[probe] _asarray_from_numpy_ndarray"
#         f"\n  supplied sycl_queue ptr : {_fmt_ptr(q_in_ptr)}"
#         f"\n  normalize_queue_device  : {_fmt_ptr(q_out_ptr)}"
#         f"\n  master queue ptr        : {_fmt_ptr(q_mst_ptr)}"
#         f"\n  resolved == master?      {match_master}"
#         f"\n  resolved == supplied?    {match_supplied}"
#         f"\n  same sycl_context?       {same_ctx}"
#     )
#     traceback.print_stack(limit=7)
#     return _orig_asarray_from_numpy(
#         ary, dtype=dtype, usm_type=usm_type, sycl_queue=sycl_queue, order=order
#     )

# _ctors._asarray_from_numpy_ndarray = _probe_asarray_from_numpy

# =====================================================================
# Runtime verification — catches regressions early.
# Uses native-handle equality (not `is`) because dpnp may rewrap a
# SyclQueue Python object around the same underlying sycl::queue.
# Runs once per process (guarded by _state["verified"]).
# =====================================================================
def _verify_single_queue_invariant():
    if _state["verified"]:
        return


    # (1) libgsycl pointer parity per device.
    for d in range(len(_gpu_devices())):
        q = _master_queue(d)
        libgpu.sycl_set_device(ctypes.c_int(d))
        if int(libgpu.sycl_get_queue_ptr() or 0) != _get_sycl_queue_ptr(q):  # ← fixed
            raise RuntimeError(
                f"libgsycl queue pointer diverges from Python master on device {d}")

    # (2) main-thread dpnp allocation lands on master.
    libgpu.sycl_set_device(ctypes.c_int(0))
    if not _same_queue(dpnp.zeros(4).sycl_queue, _master_queue(0)):
        raise RuntimeError("main-thread dpnp allocation escaped master queue")

    # (3) worker-thread dpnp allocation lands on master.
    from concurrent.futures import ThreadPoolExecutor
    def _probe():
        return dpnp.zeros(4).sycl_queue
    with ThreadPoolExecutor(max_workers=1) as ex:
        worker_q = ex.submit(_probe).result()
    if not _same_queue(worker_q, _master_queue(0)):
        raise RuntimeError(
            "worker-thread dpnp allocation escaped master queue — "
            "ContextVar replacement regressed"
        )

    _state["verified"] = True


_verify_single_queue_invariant()


# =====================================================================
# Shutdown guard
# =====================================================================
@atexit.register
def _mark_shutdown():
    _state["shutting_down"] = True

def _shutting_down():
    return _state["shutting_down"]


# =====================================================================
# Stream — singleton per device, wraps master SyclQueue.
# Uses the shared _stream_cache on _state so both module copies (if any)
# hand out the same Stream instance per device.
# =====================================================================
class Stream:
    """CuPy-compatible singleton Stream wrapping the master SyclQueue.

    The constructor arguments (null, non_blocking, ptds) are accepted
    for CuPy API parity but ignored — every Stream for a given device
    returns the same object, backed by the master queue. If you need
    true stream-level concurrency you must step outside this shim and
    create a dpctl queue directly, which voids the single-queue
    invariant on that code path.
    """

    def __new__(cls, null=False, non_blocking=False, ptds=False,
                *, device_id=None):
        if device_id is None:
            device_id = int(libgpu.sycl_get_device_id())
        with _stream_cache_lock:
            s = _stream_cache.get(device_id)
            if s is not None:
                return s
            s = object.__new__(cls)
            s._device_id  = device_id
            s._sycl_queue = _master_queue(device_id)
            s._ptr = _get_sycl_queue_ptr(s._sycl_queue)
            _stream_cache[device_id] = s
            return s

    def __init__(self, *a, **kw):
        return

    @property
    def ptr(self):          return self._ptr
    @property
    def sycl_queue(self):   return self._sycl_queue

    def __int__(self):      return self._ptr

    def __enter__(self):
        libgpu.sycl_set_device(ctypes.c_int(self._device_id))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def synchronize(self):
        self._sycl_queue.wait()

    @classproperty
    def null(cls):
        return get_current_stream()


class _StreamNS:
    Stream = Stream

    @staticmethod
    def get_current_stream():
        return get_current_stream()


stream = _StreamNS()


def get_current_stream():
    return Stream()


def get_device_count():     return len(_gpu_devices())
def get_total_memory():     return libgpu.sycl_get_total_memory()
def get_shared_memory():    return libgpu.sycl_get_shared_memory()
def get_free_memory():      return libgpu.sycl_get_free_memory()

def get_compute_units():
    """Number of compute units (maps to CUDA multiProcessorCount).

    Queries the registered SYCL queue's device.
    """
    return int(libgpu.sycl_get_compute_units())

def get_device_name():
    """Device name (maps to CUDA cudaDeviceProp::name).

    Queries the registered SYCL queue's device.
    """
    buf = ctypes.create_string_buffer(256)
    libgpu.sycl_get_device_name(buf, ctypes.c_int(len(buf)))
    return buf.value.decode('utf-8', errors='replace')


# =====================================================================
# Device — singleton per id, backed by the shared _device_cache on _state.
# =====================================================================
class Device:
    """Singleton-per-id Device wrapper — CuPy Device(0) semantics."""

    def __new__(cls, device=None):
        if device is None:
            device = int(libgpu.sycl_get_device_id())
        elif not isinstance(device, int):
            raise TypeError("device must be None or an integer device ID")
        count = len(_gpu_devices())
        if device < 0 or device >= count:
            raise ValueError(
                f"Device index {device} out of range (available: {count})")
        with _device_cache_lock:
            d = _device_cache.get(device)
            if d is not None:
                return d
            d = object.__new__(cls)
            d._id = device
            _master_queue(device)   # ensure master exists
            _device_cache[device] = d
            return d

    def __init__(self, device=None):
        return

    @classmethod
    def get_device_id(cls) -> int:
        return int(libgpu.sycl_get_device_id())

    @property
    def id(self):
        return self._id

    def __enter__(self):
        libgpu.sycl_set_device(ctypes.c_int(self._id))
        return self

    def __exit__(self, exc_type, exc_value, tb):
        pass

    def synchronize(self):
        """Drain the device's master queue — superset of cudaDeviceSynchronize."""
        _master_queue(self._id).wait()

    @property
    def mem_info(self):
        return (get_free_memory(), get_total_memory())


device = Device


# =====================================================================
# Event — wall-clock timing + queue.wait() sync
#
# submit_barrier() on an idle in-order queue can return a Level Zero
# 'internal event' that cannot be .wait()'d on, so we use host-clock
# for elapsed-time math and queue.wait() as the sync primitive. On
# an in-order queue, queue.wait() is a strict superset of 'wait for
# the barrier we would have submitted'.
# =====================================================================
class Event:
    """CuPy-compatible GPU timing Event."""

    def __init__(self):
        self._queue     = None
        self._timestamp = None
        self._recorded  = False
        self._synced    = False

    def record(self, stream=None):
        if stream is not None and hasattr(stream, "sycl_queue"):
            self._queue = stream.sycl_queue
        else:
            self._queue = _master_queue()
        self._timestamp = time.perf_counter()
        self._recorded  = True
        self._synced    = False

    def synchronize(self):
        if self._recorded and not self._synced and self._queue is not None:
            try:
                self._queue.wait()
            except Exception:
                pass
            self._synced = True

    def query(self):
        if not self._recorded:
            return True
        self.synchronize()
        return True

    def __del__(self):
        # Finalizer never touches GPU work — queue may be in teardown.
        self._queue = None


def get_elapsed_time(start_event, end_event):
    """Elapsed wall-clock time between two recorded Events, in ms."""
    if not isinstance(start_event, Event) or not isinstance(end_event, Event):
        raise TypeError("Both arguments must be cuda.Event instances.")
    if not (start_event._recorded and end_event._recorded):
        raise ValueError("Both events must be recorded.")
    end_event.synchronize()
    return (end_event._timestamp - start_event._timestamp) * 1000.0


# =====================================================================
# Address helper — used by _Runtime.memcpy
# =====================================================================
def _addr_of(obj) -> int:
    if isinstance(obj, int):
        return obj
    if isinstance(obj, ctypes.c_void_p):
        return int(obj.value)
    ai = getattr(obj, "__sycl_usm_array_interface__", None)
    if isinstance(ai, dict) and "data" in ai:
        return int(ai["data"][0])
    ai = getattr(obj, "__array_interface__", None)
    if isinstance(ai, dict) and "data" in ai:
        return int(ai["data"][0])
    try:
        return int(obj)
    except Exception:
        pass
    if hasattr(obj, "ctypes") and hasattr(obj.ctypes, "data"):
        try:
            return int(obj.ctypes.data)
        except Exception:
            pass
    raise TypeError(f"Cannot obtain address from object of type {type(obj)}")


# =====================================================================
# CUDA-compat Runtime shim
# =====================================================================
class _Runtime:
    memcpyHostToHost     = 0
    memcpyHostToDevice   = 1
    memcpyDeviceToHost   = 2
    memcpyDeviceToDevice = 3
    memcpyDefault        = 4
    hostAllocMapped      = 0x02

    @staticmethod
    def getDeviceCount() -> int:
        return get_device_count()

    @staticmethod
    def memGetInfo():
        return (get_free_memory(), get_total_memory())

    @staticmethod
    def memcpy(dst, src, nbytes, kind):
        libgpu.sycl_memcpy(
            ctypes.c_void_p(_addr_of(dst)),
            ctypes.c_void_p(_addr_of(src)),
            ctypes.c_size_t(int(nbytes)))

    @staticmethod
    def getDeviceProperties(device_id: int) -> dict:
        devices = _gpu_devices()
        if not devices or device_id < 0 or device_id >= len(devices):
            compute_units = get_compute_units()
            return {
                'totalGlobalMem':         get_total_memory(),
                'sharedMemPerBlock':      get_shared_memory(),
                'sharedMemPerBlockOptin': get_shared_memory(),
                'name':                   get_device_name(),
                'maxThreadsPerBlock':     1024,
                'maxWorkGroupSize':       1024,
                'maxComputeUnits':        compute_units,
                'major': 8, 'minor': 0,
                'warpSize':               32,
                'multiProcessorCount':    compute_units,
            }
        dev = devices[device_id]
        try:
            warp_size = dev.sub_group_sizes[0] if dev.sub_group_sizes else 32
        except Exception:
            warp_size = 32
        compute_units = dev.max_compute_units
        if not compute_units or compute_units < 1:
            compute_units = get_compute_units()
        return {
            'totalGlobalMem':         dev.global_mem_size,
            'sharedMemPerBlock':      dev.local_mem_size,
            'sharedMemPerBlockOptin': dev.local_mem_size,
            'name':                   dev.name,
            'maxThreadsPerBlock':     dev.max_work_group_size,
            'maxWorkGroupSize':       dev.max_work_group_size,
            'maxComputeUnits':        compute_units,
            'major': 8, 'minor': 0,
            'warpSize':               warp_size,
            'multiProcessorCount':    compute_units,
            'localMemSize':           dev.local_mem_size,
            'globalMemSize':          dev.global_mem_size,
        }

    @staticmethod
    def deviceCanAccessPeer(src: int, dst: int) -> bool:
        return True


runtime = _Runtime()


# =====================================================================
# Pinned-memory allocator (attached to master queue)
# =====================================================================
def alloc_pinned_memory(nbytes, flags=None):
    nbytes = int(nbytes)
    q      = _master_queue()
    mapped = True
    if flags is not None:
        try:
            mapped = bool(flags & runtime.hostAllocMapped)
        except Exception:
            mapped = True
    Mem = dpmem.MemoryUSMShared if mapped else dpmem.MemoryUSMHost
    return Mem(nbytes, queue=q)


def _gpu_probe(label):
    """Probe whether the GPU context is still healthy. Prints OK or raises
    with the failure site. Drains the master queue so earlier async faults
    surface HERE instead of at some later innocent-looking call."""
    import dpnp
    try:
        x = dpnp.zeros(4, dtype=dpnp.float64)
        x.sycl_queue.wait()
        print(f"[gpu_probe {label}] OK", flush=True)
    except Exception as e:
        print(f"[gpu_probe {label}] FAILED: {e}", flush=True)
        raise
