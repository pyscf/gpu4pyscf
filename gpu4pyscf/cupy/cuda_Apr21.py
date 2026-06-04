import os, sys, traceback

if os.environ.get("GPU4PYSCF_TRACE_SYCL_QUEUE"):
    import dpctl, sys, traceback

    _OrigQ = dpctl.SyclQueue
    _ctx_n = {"q": 0}

    class _TracedSyclQueue(_OrigQ):
        def __new__(cls, *a, **kw):
            _ctx_n["q"] += 1
            n = _ctx_n["q"]
            sys.stderr.write(f"\n[SYCL-Q #{n}] args={a} kwargs={kw}\n")
            sys.stderr.writelines(traceback.format_stack()[-12:-1])
            sys.stderr.flush()
            return _OrigQ.__new__(cls, *a, **kw)

    # Swap the attribute. isinstance(some_orig_q, _TracedSyclQueue) is False,
    # but isinstance(some_orig_q, _OrigQ) is True — and most dpctl isinstance
    # checks resolve via the Cython cdef class, not the module attribute.
    # If a failure occurs, we at least see it for the specific call site.
    dpctl.SyclQueue = _TracedSyclQueue
    
    
import dpctl
import dpctl.memory as dpmem
import dpctl._sycl_queue_manager as qmgr
import dpnp
import functools
import inspect
import time
import threading
import warnings
import atexit
import ctypes
import os


class _InOrderQueueCache:
    """Replacement for _DeviceDefaultQueueCache that always returns
    the master in-order queue instead of dpctl's cached out-of-order queue.
    
    get_or_create(device) must return (SyclQueue, bool) where bool is
    is_newly_created — we always return False since master queue pre-exists.
    """
    def __init__(self, q):
        self._q = q

    def get_or_create(self, device):
        return (self._q, False)
    
lib_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../lib/libgsycl.so"))
libgpu = ctypes.CDLL(lib_path)

# ctypes bindings — MUST match sycl_api_python.cpp signatures exactly
libgpu.sycl_get_device_id.argtypes  = []
libgpu.sycl_get_device_id.restype   = ctypes.c_int

libgpu.sycl_get_queue_ptr.argtypes  = []
libgpu.sycl_get_queue_ptr.restype   = ctypes.c_void_p

libgpu.sycl_set_queue_ptr.argtypes  = [ctypes.c_int, ctypes.c_void_p]
libgpu.sycl_set_queue_ptr.restype   = None

libgpu.sycl_set_device.argtypes     = [ctypes.c_int]
libgpu.sycl_set_device.restype      = None

libgpu.sycl_get_total_memory.argtypes  = []
libgpu.sycl_get_total_memory.restype   = ctypes.c_size_t

libgpu.sycl_get_shared_memory.argtypes = []
libgpu.sycl_get_shared_memory.restype  = ctypes.c_size_t

libgpu.sycl_get_free_memory.argtypes   = []
libgpu.sycl_get_free_memory.restype    = ctypes.c_size_t

libgpu.sycl_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
libgpu.sycl_memcpy.restype  = ctypes.c_size_t


class classproperty:
    def __init__(self, fget):
        self.fget = fget
    def __get__(self, obj, owner):
        return self.fget(owner)


# =====================================================================
# Master-queue registry  — SINGLE dpctl.SyclQueue per device
#
# This dict is the singleton store. Every layer of the stack (dpnp,
# libgsycl, Stream, Device) obtains its queue from here — one queue
# per device, for the whole process lifetime.
# =====================================================================
_master_lock        = threading.Lock()
_master_queues      = {}    # int -> dpctl.SyclQueue   (THE singletons)
_cached_gpu_devices = None


def _gpu_devices():
    """Enumerate GPU devices once; prefer level_zero for parity with
    what libgsycl sees after the C++ fix (which trusts Python ordering)."""
    global _cached_gpu_devices
    if _cached_gpu_devices is not None:
        return _cached_gpu_devices
    try:
        devs = dpctl.get_devices(backend="level_zero", device_type="gpu")
    except Exception:
        devs = []
    if not devs:
        try:
            devs = dpctl.get_devices(device_type="gpu")
        except Exception:
            devs = dpctl.get_devices()
    _cached_gpu_devices = devs
    return devs


def _master_queue(device_id=None):
    """Return the singleton master in-order SyclQueue for a device.

    Creates it on first access, registers the native pointer with
    libgsycl.so so every .so in the process sees the SAME queue,
    then caches it. Subsequent calls always return the same object.
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
            ctypes.c_void_p(q.addressof_ref()))
        _master_queues[device_id] = q
        return q


def master_device(device_id=None):
    """Public accessor — returns the master dpctl.SyclQueue for a device.
    Use this anywhere code needs to pass `sycl_queue=` explicitly."""
    return _master_queue(device_id)

# =====================================================================
# Bootstrap — register master queues for every GPU at import time
# =====================================================================
class _GlobalQueueCacheShim:
    """Thread-shared stand-in for dpctl's ContextVar-based queue cache.

    A ContextVar set in the main thread is invisible to worker threads
    spawned by ThreadPoolExecutor — they start in a fresh default
    context. We replace the ContextVar object itself so every thread
    resolves `.get()` to the same in-order cache.

    Mimics just enough of ContextVar's surface (get/set/reset) to keep
    dpctl internals that poke at it happy.
    """
    __slots__ = ("_cache",)

    def __init__(self, cache):
        self._cache = cache

    def get(self, *default):          # ContextVar.get(default=MISSING)
        return self._cache

    def set(self, value):             # returns a pseudo-token
        old = self._cache
        self._cache = value
        return old

    def reset(self, token):           # no-op; we don't track tokens
        pass


def _bootstrap():
    for d in range(len(_gpu_devices())):
        try:
            _master_queue(d)
        except Exception as e:
            warnings.warn(
                f"Failed to install master queue for device {d}: {e}",
                RuntimeWarning)

    # Replace the ContextVar outright so all threads — including
    # ThreadPoolExecutor workers — see the same in-order cache.
    try:
        import dpctl._sycl_queue_manager as qmgr
        qmgr._global_device_queue_cache = _GlobalQueueCacheShim(
            _InOrderQueueCache(_master_queue(0))
        )
        # Verify — main thread.
        probe = dpnp.zeros(4)
        if not probe.sycl_queue.is_in_order:
            warnings.warn(
                "Queue-cache replacement did not redirect dpnp allocations "
                "to the master in-order queue.",
                RuntimeWarning)
    except Exception as e:
        warnings.warn(
            f"Failed to replace dpctl device queue cache: {e}",
            RuntimeWarning)

_bootstrap()


# --------------------------------------------------------------------
# Force every allocation through the master queue by wrapping creation
# APIs and injecting sycl_queue=. This does not rely on ContextVar and
# works identically on main and worker threads.
# --------------------------------------------------------------------
import dpctl.tensor as dpt

_DPNP_CREATION = (
    "asarray", "array", "zeros", "ones", "empty", "full",
    "zeros_like", "ones_like", "empty_like", "full_like",
    "arange", "linspace", "logspace", "geomspace",
    "eye", "identity", "tri", "frombuffer", "fromfunction",
    "copy",
)
_DPT_CREATION = (
    "asarray", "empty", "zeros", "ones", "full",
    "empty_like", "zeros_like", "ones_like", "full_like",
    "arange", "linspace", "eye",
)

def _wrap_with_master_queue(mod, names):
    for name in names:
        orig = getattr(mod, name, None)
        if orig is None or getattr(orig, "__master_q_wrapped__", False):
            continue

        @functools.wraps(orig)
        def wrapper(*args, _orig=orig, **kwargs):
            # Respect callers that explicitly placed the allocation.
            if ("sycl_queue" not in kwargs
                    and "device"     not in kwargs
                    and "usm_type"   not in kwargs):
                kwargs["sycl_queue"] = _master_queue()
            return _orig(*args, **kwargs)

        wrapper.__master_q_wrapped__ = True
        wrapper.__wrapped__ = orig
        setattr(mod, name, wrapper)

_wrap_with_master_queue(dpnp, _DPNP_CREATION)
_wrap_with_master_queue(dpt,  _DPT_CREATION)

# =====================================================================
# Runtime verification — prove the single-queue-per-device invariant
#
# Runs once after bootstrap. Set GPU4PYSCF_SKIP_QUEUE_VERIFY=1 to skip
# (e.g. in production hot paths where the startup check is redundant).
# =====================================================================
def _verify_single_queue_invariant():
    if os.environ.get("GPU4PYSCF_SKIP_QUEUE_VERIFY"):
        return

    # (1) pointer parity per device — unchanged
    for d in range(len(_gpu_devices())):
        q = _master_queue(d)
        libgpu.sycl_set_device(ctypes.c_int(d))
        if int(libgpu.sycl_get_queue_ptr() or 0) != int(q.addressof_ref()):
            raise RuntimeError(f"queue-pointer mismatch on device {d}")

    # (2) main-thread allocation lands on master queue — unchanged
    libgpu.sycl_set_device(ctypes.c_int(0))
    if dpnp.zeros(4).sycl_queue is not _master_queue(0):
        raise RuntimeError("main-thread dpnp allocation escaped master queue")

    # (3) NEW: worker-thread allocation must also land on master queue.
    # This is the check that would have caught the ContextVar bug.
    from concurrent.futures import ThreadPoolExecutor
    def _probe():
        return dpnp.zeros(4).sycl_queue
    with ThreadPoolExecutor(max_workers=1) as ex:
        worker_q = ex.submit(_probe).result()
    if worker_q is not _master_queue(0):
        raise RuntimeError(
            "worker-thread dpnp allocation escaped master queue — "
            "ContextVar is not propagating across threads."
        )

_verify_single_queue_invariant()


# =====================================================================
# Allocation auditor — debug-only. Set GPU4PYSCF_AUDIT_ALLOCS=1 to arm.
#
# Logs any dpnp_array whose sycl_queue does not match the master queue
# for its device, along with the Python stack that created it. Use this
# to track down the last allocation paths that escape the wrappers.
# =====================================================================
def _install_allocation_auditor():
    import os, sys, traceback, functools
    from dpnp.dpnp_array import dpnp_array
    import dpctl.tensor as dpt

    # --- Open a real file descriptor. Bypasses Python/pytest buffering. ---
    log_path = os.environ.get("GPU4PYSCF_AUDIT_LOG", "/tmp/gpu4pyscf_audit.log")
    log_fd = os.open(log_path,
                     os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                     0o644)

    def _emit(msg):
        os.write(log_fd, (msg + "\n").encode("utf-8", "replace"))
        os.fsync(log_fd)  # survive the segfault

    _emit(f"[AUDIT] installed, pid={os.getpid()}, log={log_path}")

    master_ptrs = {int(q.addressof_ref()) for q in _master_queues.values()}
    _emit(f"[AUDIT] master queue ptrs: {[hex(p) for p in master_ptrs]}")

    # Install count so we can tell 'auditor ran but nothing escaped'
    # from 'auditor never ran'.
    counters = {"dpnp": 0, "dpt": 0, "dpt-op": 0, "escapes": 0}

    def _check(arr, origin, creating_fn=None):
        counters[origin] = counters.get(origin, 0) + 1
        q = getattr(arr, "sycl_queue", None)
        if q is None:
            return
        try:
            actual = int(q.addressof_ref())
        except Exception:
            return
        if actual in master_ptrs:
            return
        counters["escapes"] += 1
        fn_note = f" fn={creating_fn}" if creating_fn else ""
        stack = "".join(traceback.format_stack()[:-2])
        _emit(
            f"[ESCAPE-{origin}]{fn_note} queue={actual:#x} "
            f"shape={getattr(arr,'shape',None)} "
            f"dtype={getattr(arr,'dtype',None)}\n"
            f"{stack}"
            f"{'-'*72}"
        )

    # Also dump the counters on interpreter exit so we know the auditor ran.
    import atexit
    @atexit.register
    def _dump():
        try:
            _emit(f"[AUDIT] final counts: {counters}")
            os.close(log_fd)
        except Exception:
            pass

    # --- dpnp_array __init__ is patchable (pure Python) ---
    _orig_dpnp = dpnp_array.__init__
    def audited_dpnp(self, *a, **kw):
        _orig_dpnp(self, *a, **kw)
        _check(self, "dpnp")
    dpnp_array.__init__ = audited_dpnp

    # --- dpt factory and op audit via wrapper ---
    _FACTORIES = {
        "asarray","array","empty","zeros","ones","full",
        "empty_like","zeros_like","ones_like","full_like",
        "arange","linspace","eye","copy",
    }
    _OPS = {
        "multiply","add","subtract","divide","matmul",
        "tensordot","concat","stack","reshape","broadcast_to",
    }

    def _wrap_audit(mod, names, origin):
        for name in names:
            fn = getattr(mod, name, None)
            if fn is None or getattr(fn, "__audit_wrapped__", False):
                continue

            @functools.wraps(fn)
            def audited(*a, _fn=fn, _name=name, _origin=origin, **kw):
                out = _fn(*a, **kw)
                if isinstance(out, tuple):
                    for o in out: _check(o, _origin, _name)
                else:
                    _check(out, _origin, _name)
                return out
            audited.__audit_wrapped__ = True
            setattr(mod, name, audited)

    _wrap_audit(dpt, _FACTORIES, "dpt")
    _wrap_audit(dpt, _OPS,       "dpt-op")

    _emit("[AUDIT] hooks installed, ready")

# IMPORTANT: call this AFTER _wrap_with_master_queue(dpt, ...) so our
# master-queue wrappers are the inner layer and the auditor wraps them.
if os.environ.get("GPU4PYSCF_AUDIT_ALLOCS"):
    _install_allocation_auditor()
    
# =====================================================================
# Shutdown guard
# =====================================================================
_shutting_down = False

@atexit.register
def _mark_shutdown():
    global _shutting_down
    _shutting_down = True


# =====================================================================
# Stream — SINGLETON per device
#
# Stream(0) always returns the same Python object. All Streams for the
# same device share the same underlying sycl::queue (the master queue).
# =====================================================================
_stream_cache      = {}     # device_id -> Stream
_stream_cache_lock = threading.Lock()


class Stream:
    """Thin singleton-per-device wrapper around the master SyclQueue.

    Signature mirrors ``cupy.cuda.Stream(null=False, non_blocking=False,
    ptds=False)`` for API parity — those flags are *accepted but ignored*
    because we have one in-order master queue per device and every Stream
    for that device wraps it.  ``device_id`` (keyword-only) is our
    extension for multi-GPU code.

    Implication: ``Stream(non_blocking=True)`` called twice on the same
    device returns the *same* object.  Work submitted "through" two such
    streams serialises through the single master queue.  If you need
    stream-level concurrency, you must step outside this wrapper and
    create a dpctl queue directly — doing so breaks the single-queue
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
            s = super().__new__(cls)
            s._device_id   = device_id
            s._sycl_queue  = _master_queue(device_id)
            s._ptr         = int(s._sycl_queue.addressof_ref())
            s._initialized = True
            _stream_cache[device_id] = s
            return s

    def __init__(self, null=False, non_blocking=False, ptds=False,
                 *, device_id=None):
        # __new__ already fully initialised the (cached) instance.
        # Skip to avoid clobbering state on repeat calls.
        return

    @property
    def ptr(self):
        return self._ptr

    @property
    def sycl_queue(self):
        return self._sycl_queue

    def __int__(self):
        return self._ptr

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


def get_device_count():
    return len(_gpu_devices())


def get_total_memory():
    return libgpu.sycl_get_total_memory()


def get_shared_memory():
    return libgpu.sycl_get_shared_memory()


def get_free_memory():
    return libgpu.sycl_get_free_memory()


# =====================================================================
# Device — SINGLETON per id
#
# Device(0) always returns the same object; all Device(0) callers
# reference the same master queue. Matches CuPy's Device(0) API.
# =====================================================================
_device_cache      = {}
_device_cache_lock = threading.Lock()


class Device:
    """Singleton-per-id Device wrapper."""

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
            d = super().__new__(cls)
            d._id = device
            _master_queue(device)           # ensure queue exists
            d._initialized = True
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

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def synchronize(self):
        """Wait for all work submitted to this device's master queue.

        Because we keep one in-order queue per device, syncing that
        queue is equivalent to ``cupy.cuda.Device().synchronize()`` —
        it waits for every op enqueued by any thread on this device.
        """
        _master_queue(self._id).wait()

    @property
    def mem_info(self):
        return (get_free_memory(), get_total_memory())

device = Device


# =====================================================================
# Event — backed by dpctl.SyclQueue.submit_barrier()
# =====================================================================

class Event:
    """CuPy-compatible GPU timing Event.

    Implementation detail: we do NOT use dpctl.SyclQueue.submit_barrier()
    directly.  On an in-order queue that happens to be quiescent at the
    moment of the call, submit_barrier() can return a Level Zero
    'internal event' — a placeholder that hasn't been promoted to a
    real, waitable ur_event_handle_t.  Calling .wait() on one aborts
    with 'urEventWait must not be called for an internal event'.
    Setting ZE_SERIALIZE=2 hides this by forcing synchronous submission,
    but that's a workaround with real performance cost.

    Instead, we rely on the host-clock timestamp for elapsed-time math
    (which is what get_elapsed_time() actually needs — sub-ms GPU
    timing is not the goal of this shim) and use queue.wait() as the
    sync primitive when the user calls .synchronize().  The queue is
    in-order per device, so queue.wait() is a strict superset of
    "wait for everything recorded by this Event".
    """

    def __init__(self):
        self._queue     = None      # dpctl.SyclQueue captured at record()
        self._timestamp = None      # perf_counter at record() time
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
        """Wait for all work submitted to this event's queue up to
        (and slightly past) record() time.  Uses queue.wait(), which
        on an in-order queue is a superset of 'wait for the barrier
        we would have submitted' — and doesn't risk the internal-event
        abort that submit_barrier() can trigger on a quiescent queue."""
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
        # Finalizer never waits on GPU work — by the time __del__ runs,
        # the queue may be mid-teardown and waiting here is an abort
        # risk.  Just drop the reference.
        self._queue = None


def get_elapsed_time(start_event, end_event):
    """Elapsed wall-clock time in milliseconds between two recorded Events.

    Uses perf_counter deltas, which is sufficient for logger.py timing
    output.  If you need true device-side sub-µs profiling, create the
    master queue with property=['in_order','enable_profiling'] and
    switch to profiling_info_start / profiling_info_end on a real
    queue event — but that's a separate change.
    """
    if not isinstance(start_event, Event) or not isinstance(end_event, Event):
        raise TypeError("Both arguments must be cuda.Event instances.")
    if not (start_event._recorded and end_event._recorded):
        raise ValueError(
            "Both events must be recorded before calling get_elapsed_time.")
    end_event.synchronize()
    return (end_event._timestamp - start_event._timestamp) * 1000.0  # ms


# =====================================================================
# Address helper
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
# Runtime shim
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
            return {
                'totalGlobalMem':         get_total_memory(),
                'sharedMemPerBlock':      get_shared_memory(),
                'sharedMemPerBlockOptin': get_shared_memory(),
                'name':                   'Unknown SYCL Device',
                'maxThreadsPerBlock':     1024,
                'maxWorkGroupSize':       1024,
                'maxComputeUnits':        1,
                'major': 8, 'minor': 0,
                'warpSize':               32,
                'multiProcessorCount':    1,
            }
        dev = devices[device_id]
        try:
            warp_size = dev.sub_group_sizes[0] if dev.sub_group_sizes else 32
        except Exception:
            warp_size = 32
        return {
            'totalGlobalMem':         dev.global_mem_size,
            'sharedMemPerBlock':      dev.local_mem_size,
            'sharedMemPerBlockOptin': dev.local_mem_size,
            'name':                   dev.name,
            'maxThreadsPerBlock':     dev.max_work_group_size,
            'maxWorkGroupSize':       dev.max_work_group_size,
            'maxComputeUnits':        dev.max_compute_units,
            'major': 8, 'minor': 0,
            'warpSize':               warp_size,
            'multiProcessorCount':    dev.max_compute_units,
            'localMemSize':           dev.local_mem_size,
            'globalMemSize':          dev.global_mem_size,
        }

    @staticmethod
    def deviceCanAccessPeer(src: int, dst: int) -> bool:
        return True

runtime = _Runtime()


# =====================================================================
# Pinned-memory allocator — attached to the master queue
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


# # in cupy/cuda.py, at the bottom
# def _drain_after(fn):
#     @functools.wraps(fn)
#     def w(*args, **kwargs):
#         r = fn(*args, **kwargs)
#         _master_queue().wait()
#         return r
#     return w

def _patch_dpnp_creation_apis():
    """Force every dpnp creation call to land on the master queue,
    regardless of ContextVar state or thread."""
    CREATION = (
        "asarray", "array", "zeros", "ones", "empty", "full",
        "arange", "linspace", "logspace", "eye", "identity",
        "frombuffer",
    )
    for name in CREATION:
        orig = getattr(dpnp, name, None)
        if orig is None or getattr(orig, "__master_q_wrapped__", False):
            continue

        @functools.wraps(orig)
        def wrapper(*args, _orig=orig, **kwargs):
            # Respect explicit placement by the caller.
            if (kwargs.get("sycl_queue") is None
                    and kwargs.get("device") is None
                    and kwargs.get("usm_type") is None):
                kwargs["sycl_queue"] = _master_queue()
            return _orig(*args, **kwargs)

        wrapper.__master_q_wrapped__ = True
        wrapper.__wrapped__ = orig
        setattr(dpnp, name, wrapper)

_patch_dpnp_creation_apis()
