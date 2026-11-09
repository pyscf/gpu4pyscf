"""
Single-queue-per-device SYCL runtime shim for gpu4pyscf.

Design invariant
----------------
Exactly ONE dpctl.SyclQueue lives per GPU for the lifetime of the process.
Every allocation (Python-side via dpnp/dpctl, C++-side via libgsycl.so)
must land on that master queue.

Enforcement layers (defence in depth)
-------------------------------------
1. Master queue registry -- `_master_queue(d)` creates the singleton
   in-order queue for device `d` on first call, registers its native
   pointer with libgsycl.so, and caches it forever.

2. Global queue-cache replacement -- on dpctl/dpnp master,
   `_global_device_queue_cache` is a plain process-global object whose
   `get_or_create(key)` returns a SyclQueue. We replace it with a cache
   that always returns the per-device master queue. Being process-global
   (not a ContextVar), it is also visible to ThreadPoolExecutor worker
   threads, so every thread sees the master queue.

3. Creation-API wrappers -- every dpnp and dpctl.tensor array-creation
   function is wrapped to inject `sycl_queue=master` unless the caller
   has explicitly placed the allocation.

Idempotency / reload-safety
---------------------------
This module can be imported under two dotted names: `cupy.cuda` (when
we're loaded through the gpu4pyscf.cupy facade that re-exports as
`cupy`) and `gpu4pyscf.cupy.cuda` (the real dotted path). Both names
are aliased in gpu4pyscf/cupy/__init__.py, but as belt-and-suspenders
this file stashes its mutable state (master queue registry, device
cache, stream cache) on the `dpnp` module -- which is guaranteed to
load exactly once -- so even if we execute twice we don't duplicate
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
import contextlib
import ctypes
import functools
import os
import threading
import time
import types
import warnings
import weakref

import dpctl
import dpctl.memory as dpmem
import dpctl._sycl_queue_manager as qmgr
import dpnp


def _pin_libtcm():
    '''Work around the oneAPI TBB/TCM process-exit race (vendor bug).

    libmkl_tbb_thread -> libtbb dlopens libtcm (its TBB-compatible backend)
    lazily and dlcloses it from an exit handler. libtcm's own
    ``__cxa_atexit`` entries (e.g. the hwloc topology loader) can remain
    pending after its text is unmapped, so the next exit-handler call jumps
    to an unmapped address -> intermittent SIGSEGV (rc=139) AFTER the whole
    test session has passed (~15-30% of runs on Aurora's 2026.1 stack).
    Pinning libtcm with an extra dlopen before libtbb's first one makes
    libtbb's dlclose only decrement the refcount: the library stays mapped
    and its handlers run normally. Debug-disable: GPU4PYSCF_PIN_TCM=0.
    '''
    if os.environ.get("GPU4PYSCF_PIN_TCM", "1") == "0":
        return
    try:
        ctypes.CDLL("libtcm.so.1", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass  # stack without TCM (e.g. non-Aurora): the race cannot occur


_pin_libtcm()

_DEFERRED_FREE_THRESHOLD = int(
    os.environ.get("GPU4PYSCF_DEFERRED_FREE_THRESHOLD", "256")
)

_DEFER_FREE_ENABLED = os.environ.get("GPU4PYSCF_DEFER_FREE", "1") != "0"
_DEFER_FREE_MIN_BYTES = int(
    os.environ.get("GPU4PYSCF_DEFER_FREE_MIN_BYTES", "0")
)
# Coalesce this many freed buffers into a single keep-alive host task to
# amortize per-free enqueue/GIL cost. 1 = submit a host task per free.
_DEFER_FREE_BATCH = max(1, int(
    os.environ.get("GPU4PYSCF_DEFER_FREE_BATCH", "64")
))
# Flush a lingering partial batch at most once per this many allocations,
# so tail buffers are not pinned indefinitely without flushing a host task
# on every single allocation.
_DEFER_FREE_FLUSH_STRIDE = max(1, int(
    os.environ.get("GPU4PYSCF_DEFER_FREE_FLUSH_STRIDE", "128")
))
_DEFER_FREE_MAIN_THREAD_ONLY = (
    os.environ.get("GPU4PYSCF_DEFER_FREE_MAIN_THREAD_ONLY", "1") != "0"
)
# Safety valve for the guard above: if frees keep arriving on non-owning
# threads and no main-thread activity flushes the batch, stop growing it
# past this many buffers (release the oldest eagerly instead). Only reachable
# in a sustained all-off-thread free burst; normal runs flush long before.
_DEFER_FREE_OFFTHREAD_CAP = max(
    _DEFER_FREE_BATCH,
    int(os.environ.get("GPU4PYSCF_DEFER_FREE_OFFTHREAD_CAP", "4096")),
)


# =====================================================================
# Shared, reload-safe state -- stashed on dpnp (which loads once).
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
        # RLock, not Lock: a GC pass can fire mid-critical-section (any
        # allocation can cross the threshold) and run a weakref finalizer
        # that calls back into _master_queue() from the same thread --
        # _deferred_release -> _flush_deferred_frees_locked -> _master_queue().
        # A plain Lock self-deadlocks there (caught with gdb on a hung
        # test_pbc_df_grad.py).
        "master_lock":        threading.RLock(),
        "master_queues":      {},      # int -> dpctl.SyclQueue
        "gpu_devices":        None,    # cached device list
        "stream_cache":       {},      # int -> Stream
        "stream_cache_lock":  threading.Lock(),
        "device_cache":       {},      # int -> Device
        "device_cache_lock":  threading.Lock(),
        "bootstrapped":       False,
        "verified":           False,
        "shutting_down":      False,
        # Layer 4 deferred-free bookkeeping. Holds keys:
        #   "batch"       -> list[_Memory] pending queue-ordered release
        #   "tagged"      -> list[(SyclEvent, list[_Memory])] awaiting completion
        #   "alloc_count" -> int, throttles partial-batch flushing
        "defer_free_lock":    threading.Lock(),
        "pending_frees":      {},
        # Thread that imported this module. Layer 4 only ever touches the
        # SYCL scheduler (submit_barrier / queue.wait) from this thread --
        # see _on_scheduler_safe_thread() for why.
        "owner_thread_id":    threading.get_ident(),
    }
    setattr(dpnp, _STATE_ATTR, _state)

# Reload-safety: a pre-existing _state (from an earlier load of this
# module under a different dotted name) may predate the Layer 4 keys.
if "defer_free_lock" not in _state:
    _state["defer_free_lock"] = threading.Lock()
if "pending_frees" not in _state:
    _state["pending_frees"] = {}
if "owner_thread_id" not in _state:
    _state["owner_thread_id"] = threading.get_ident()

_master_lock       = _state["master_lock"]
_master_queues     = _state["master_queues"]
_stream_cache      = _state["stream_cache"]
_stream_cache_lock = _state["stream_cache_lock"]
_device_cache      = _state["device_cache"]
_device_cache_lock = _state["device_cache_lock"]
_defer_free_lock   = _state["defer_free_lock"]
_pending_frees     = _state["pending_frees"]
_owner_thread_id   = _state["owner_thread_id"]


def _on_scheduler_safe_thread():
    """True only on the thread that imported this module.

    Layer 4's flush/reap paths call into the SYCL scheduler
    (`queue.submit_barrier()`, `queue.wait()`). Both take the scheduler's
    global reader/writer lock: `submit_barrier` -> `Scheduler::addCG` needs
    it EXCLUSIVELY, while a `wait` sits inside
    `Scheduler::GraphProcessor::waitForEvent` holding it SHARED for the whole
    duration of the wait.

    These calls originate from a weakref finalizer, so they run on whatever
    thread happened to drop the last reference. That is frequently NOT the
    main thread: gpu4pyscf's DFT path (`dft/numint.py`) runs XC evaluation
    inside a ThreadPoolExecutor even for num_devices == 1, and dpnp's own
    `keep_args_alive` host tasks drop their kept references from SYCL
    thread-pool worker threads. Measured on a single short RKS+newton run:
    barriers submitted from 13 distinct non-main threads, plus 15 off-main
    `queue.wait()` calls.

    That is enough for a reader/writer deadlock inside libsycl, observed
    live under gdb on the full scf/tests/test_soscf.py suite:

        Thread 1 (main): event_impl::wait()
                         -> GraphProcessor::waitForEvent(shared_lock&)
                         -> blocked in waitInternal(), STILL HOLDING the
                            shared lock

        Thread 7 (worker): our finalizer -> SyclQueue.submit_barrier()
                         -> Scheduler::addCG
                         -> pthread_rwlock_wrlock  BLOCKED behind Thread 1

    The pending writer then blocks any further readers, the queue never
    drains, Thread 1's event never signals. GPU utilization sits at 0% with
    every thread parked in futex_wait. `ZE_SERIALIZE=2` masks it only by
    changing when frees land relative to scheduler activity.

    So: off-thread frees are still COLLECTED (correctness of the deferral is
    unchanged -- the _Memory strong ref keeps the buffer alive, which is the
    entire point of Layer 4), they are just not the ones to drive a scheduler
    operation. The next main-thread allocation or free flushes them. The
    batch is bounded by _DEFER_FREE_BATCH and drained on every main-thread
    flush, so this defers reclamation slightly; it does not leak.
    """
    if not _DEFER_FREE_MAIN_THREAD_ONLY:
        return True
    return threading.get_ident() == _owner_thread_id


# =====================================================================
# libgsycl.so -- the C++ side's master-queue registry
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
    SyclQueue.addressof_ref() returns its value cast to size_t --
    i.e. the sycl::queue* itself. sycl_set_queue_ptr does a direct
    static_cast<sycl::queue*>, so we pass the value as-is.

    q must remain alive for the lifetime of the stored pointer --
    _master_queues guarantees this for master queues.
    """
    return int(q.addressof_ref())


# =====================================================================
# Master-queue registry
# =====================================================================
def _gpu_devices():
    """Enumerate compute devices once (prefer level_zero GPUs). Cached in _state.

    Each candidate is tried in turn and the first non-empty result wins. A
    query that succeeds but returns nothing has to fall through exactly like
    one that raises: on a machine with no GPU -- a CPU-only CI runner, say --
    the GPU queries return [] without error, and stopping there would leave
    the master-queue registry empty. Every later layer assumes device 0
    exists, so the shortfall would not surface here but as a std::terminate
    out of libgsycl's sycl_set_device.

    The final candidate is unfiltered, so the OpenCL CPU device is used when
    that is all there is. Where a GPU is present the first query answers and
    the rest are never reached, leaving that path unchanged.
    """
    if _state["gpu_devices"] is not None:
        return _state["gpu_devices"]

    candidates = (
        lambda: dpctl.get_devices(backend="level_zero", device_type="gpu"),
        lambda: dpctl.get_devices(device_type="gpu"),
        lambda: dpctl.get_devices(),
    )

    devs = []
    for query in candidates:
        try:
            devs = query() or []
        except Exception:
            devs = []
        if devs:
            break

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
# Layer 2 -- replace dpctl's process-global queue cache
# =====================================================================
class _MasterQueueCache:
    """Drop-in replacement for dpctl._DeviceDefaultQueueCache.

    On dpctl/dpnp master, `_global_device_queue_cache` is a plain
    process-global object (NOT a ContextVar), and
    `get_device_cached_queue(key)` calls
    `_global_device_queue_cache.get_or_create(key)` directly, expecting a
    bare dpctl.SyclQueue in return.

    We resolve every key to the per-device master in-order queue so all
    dpnp/dpctl allocations land on the singleton queue for that GPU.
    Because this object is process-global rather than a ContextVar,
    ThreadPoolExecutor worker threads observe it too -- fixing the
    worker-thread allocation escape that motivated the original shim.

    Accepted key types (per dpctl): a SyclDevice, a (SyclContext,
    SyclDevice) 2-tuple, or a oneAPI filter-selector string. Unknown key
    types or devices not present among the enumerated GPUs raise rather
    than silently falling back to device 0.
    """
    __slots__ = ("_lock",)

    def __init__(self):
        self._lock = threading.Lock()

    def _device_from_key(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            return key[1]
        if isinstance(key, str):
            return dpctl.SyclDevice(key)            # may raise -> propagate
        if isinstance(key, dpctl.SyclDevice):
            return key
        raise TypeError(
            f"_MasterQueueCache.get_or_create: unsupported key type "
            f"{type(key)!r}")

    def _device_id_for(self, dev):
        devs = _gpu_devices()
        # Exact device-object match against the same list used to build the
        # master queues.
        for i, d in enumerate(devs):
            try:
                if d == dev:
                    return i
            except Exception:
                pass
        # Backup match by oneAPI filter string.
        for i, d in enumerate(devs):
            try:
                if d.filter_string == dev.filter_string:
                    return i
            except Exception:
                pass
        raise RuntimeError(
            f"_MasterQueueCache: device {dev} not found among the "
            f"{len(devs)} enumerated GPU(s); cannot map it to a master queue")

    def get_or_create(self, key):
        with self._lock:
            return _master_queue(self._device_id_for(self._device_from_key(key)))

    # dpctl internals may copy/update the cache; keep safe stubs.
    def _update_map(self, *args, **kwargs):
        return None

    def __copy__(self):
        return self

# =====================================================================
# Layer 3 -- wrap every creation API so sycl_queue=master is injected
# =====================================================================
_DPNP_CREATION = (
    "asarray", "array", "zeros", "ones", "empty", "full",
    "zeros_like", "ones_like", "empty_like", "full_like",
    "arange", "linspace", "logspace", "geomspace",
    "eye", "identity", "tri", "frombuffer", "fromfunction",
    "copy",
)


def _flush_deferred_frees_locked():
    """Tag the pending batch with a barrier event and retain it.

    Caller must hold _defer_free_lock.

    NO host task is submitted (intel/llvm#22943 -- an in-order queue plus a
    host task that takes a lock deadlocks). This previously called
    SyclQueue._submit_keep_args_alive(), which enqueues dpctl's async_dec_ref
    host task; that task calls PyGILState_Ensure() when it runs. On the shared
    in-order master queue that deadlocks: an in-order queue serializes ENQUEUE,
    so any Python thread that later submits work while holding the GIL blocks
    inside SYCL's scheduler behind the un-run host task, which can then never
    acquire the GIL to retire. (Observed as the test_rhf_hessian hang: a worker
    thread inside oneMKL ddot holding the GIL, a SYCL thread-pool worker stuck
    in async_dec_ref -> PyGILState_Ensure, and the main thread waiting on the
    worker's future.)

    Instead we submit a barrier and retain the batch on the Python side.
    `_reap_completed_locked()` now blocks on a real `.wait()` of the barrier
    event submitted here (see that function's docstring for why the earlier
    non-blocking `execution_status` check was replaced, and for the
    busy-queue/idle-queue event evidence) before releasing any tagged batch.
    The safety guarantee comes from that wait: the master queue is in-order,
    so once this barrier completes, every kernel enqueued before it -- i.e.
    everything that could read the batch's buffers -- is provably done. The
    GIL is never touched from a SYCL worker thread by anything in this
    function.
    """
    batch = _pending_frees.get("batch")
    if not batch:
        return
    # Never submit to the SYCL scheduler off the owning thread: addCG takes
    # the scheduler's write lock and deadlocks against a main thread parked
    # inside waitForEvent with the read lock held. Leave the batch pending;
    # the next main-thread flush picks it up. See _on_scheduler_safe_thread.
    if not _on_scheduler_safe_thread():
        return
    _pending_frees["batch"] = []
    try:
        ev = _master_queue().submit_barrier()
        _pending_frees.setdefault("tagged", []).append((ev, batch))
    except Exception:
        # On failure, dropping `batch` here frees eagerly (still correct
        # if no kernel is mid-flight; worst case reproduces the original
        # eager-free behavior only for this batch).
        pass


def _reap_completed_locked():
    """Release every tagged batch, after a REAL wait for true completion.

    Caller must hold _defer_free_lock.

    HISTORY / WHY THIS CHANGED (DEFECT5 hypothesis 24, Finding 4)
    ---------------------------------------------------------------
    This previously polled `ev.execution_status == event_status_type.complete`
    -- a non-blocking status query, never a host wait -- on the theory that a
    completed barrier event proves every kernel submitted before it has
    genuinely finished touching device memory, so the real sycl::free is safe.

    That assumption was independently disproved this session on the SAME
    SYCL/PVC/Level-Zero stack, in a different code path: bisecting a
    reproducible segfault in `RYS_build_jk`'s task loop showed that a bare
    `queue.submit_barrier()` event -- checked without a host wait -- is NOT
    sufficient to guarantee true completion, while the SAME barrier `.wait()`d
    IS sufficient (3/3 clean runs each way; see
    hang_analysis_evidence/DEFECT5_free_and_device_global_audit.md, section
    5e/5f). Since this reaper used the identical primitive
    (submit_barrier() + a completion check with no host wait) to decide when
    to run `sycl::free`, it was exposed to the same gap: a status query
    reporting "complete" before the barrier has actually drained lets this
    reaper free memory a still-running kernel is reading -- a genuine
    read-after-free, producing exactly the NotPresent page fault this
    investigation was chasing. Confirmed present and firing (before this fix)
    on the exact failing test via `GPU4PYSCF_REAPER_PROBE` instrumentation
    (section 5h): two batches (64 then ~16 items) reaped via the un-waited
    status query on every single `get_jk` call, before the task loop that
    later reports the fault even starts.

    THE FIX: a real, GIL-releasing wait on each tagged barrier event
    -----------------------------------------------------------------
    Instead of draining the entire master queue, this reaper now blocks on
    the barrier event that `_flush_deferred_frees_locked` tagged each batch
    with. Because the master queue is in-order, completing the barrier
    proves every kernel enqueued before it has finished touching device
    memory, so dropping the batch afterwards runs the real `sycl::free`
    strictly after the last possible reader. Events are waited in
    submission order (oldest first): the oldest event is the long pole, and
    later waits are typically already complete.

    Requires DPC++ >= 2026.2.0. On a busy queue the barrier event is a REAL
    waitable backend event (a probe submitting a ~13 s kernel, then the
    barrier, then more work, saw `ev.wait()` block ~13 s -- exactly until
    the pre-barrier work completed, ignoring the post-barrier submissions);
    on an idle queue `.wait()` returns immediately, which is safe because
    nothing is in flight. On 2026.1.0 the Level Zero adapter aborts the
    process (`ur_die: urEventWait must not be called for an internal event`)
    on back-to-back barrier submit+wait; see ur_event_barrier_bug/REPORT.md.

    `ext_oneapi_get_last_event()` was ALSO probed as a cheaper tag
    candidate (no barrier submission at all) and REJECTED: on this stack it
    returns a marker event whose `.wait()` no-ops even while the tracked
    kernel is still running (0.005 ms wait against a 62 s kernel; status
    stuck at "submitted"). Root cause in DPC++ `queue_impl::getLastEvent`:
    the common scheduler-bypass submission path for in-order queues does
    not track a last event (`LastEvent = nullptr`), so the function falls
    back to `insertMarkerEvent()`, whose event is not waitable. Do not use
    `ext_oneapi_get_last_event` for lifetime ordering on this stack.

    Both `SyclEvent.wait()` and `SyclQueue.wait()` release the GIL while
    blocking (`with nogil: DPCTLEvent_Wait(...)` / `DPCTLQueue_Wait(...)`
    in dpctl's Cython source) -- this is NOT the
    `async_dec_ref`/PyGILState_Ensure deadlock this reaper design was
    originally built to avoid. That deadlock required a SYCL WORKER THREAD
    to need the GIL back while the main thread held it inside the driver's
    enqueue path; here, the CALLING Python thread simply releases the GIL
    itself before blocking, so any other thread (worker or otherwise) that
    needs the GIL remains free to acquire it throughout the wait.

    Benefit over the whole-queue drain: memory is released as soon as the
    flush point completes. The old `_master_queue().wait()` also waited
    for every kernel the application submitted AFTER the barrier, pinning
    the batch's buffers for work that could never read them.

    Cost: this still only runs where `_flush_deferred_frees_locked` /
    `_reap_completed_locked` were already being called (throttled to every
    `_DEFER_FREE_FLUSH_STRIDE` allocations, or when a batch reaches
    `_DEFER_FREE_BATCH`) -- NOT once per kernel launch. This keeps the fix
    entirely inside this shim layer; no call site outside this file changes.
    """
    tagged = _pending_frees.get("tagged")
    if not tagged:
        return
    # ev.wait() parks inside Scheduler::GraphProcessor::waitForEvent while
    # holding the scheduler's SHARED lock. Doing that from a worker thread
    # adds a second reader that can outlive the main thread's own wait and
    # starve a pending writer (submit_barrier from any thread). Confine the
    # wait to the owning thread. See _on_scheduler_safe_thread.
    if not _on_scheduler_safe_thread():
        return
    _defer_free_lock.release()
    try:
        for ev, _batch in tagged:
            ev.wait()
    finally:
        _defer_free_lock.acquire()
    # Every tagged event was submitted strictly after the buffers in its
    # batch were last read (the master queue is in-order and
    # _flush_deferred_frees_locked always submits its barrier before
    # returning), so completing those waits proves every tagged batch is
    # now genuinely safe to release. Nothing stays "still pending".
    _pending_frees["tagged"] = []


def _deferred_release(mem):
    """Finalizer body: queue the freed USM `_Memory` for batched,
    queue-ordered release.

    `mem` is a strong reference to the dpctl _Memory owner; holding it
    here means the eager sycl::free in _Memory.__dealloc__ has NOT run
    yet. We append it to the pending batch and flush when the batch is
    large enough.
    """
    # During interpreter shutdown, host tasks acquiring the GIL are unsafe
    # (dpctl warns). Returning drops `mem` -> eager free, which is fine at
    # exit since no new kernels are being launched.
    if _state.get("shutting_down"):
        return
    with _defer_free_lock:
        batch = _pending_frees.setdefault("batch", [])
        batch.append(mem)
        if len(batch) >= _DEFER_FREE_BATCH:
            if _on_scheduler_safe_thread():
                _flush_deferred_frees_locked()
            elif len(batch) >= _DEFER_FREE_OFFTHREAD_CAP:
                del batch[:-_DEFER_FREE_BATCH]


def _register_deferred_free(arr):
    """Register a finalizer on a freshly created dpnp array so that, when
    it is garbage collected, its USM allocation is released in a batched,
    queue-ordered manner instead of eagerly.

    No-op (returns arr unchanged) if deferral is disabled, the object is
    not a dpnp array, it is too small, or the underlying USM handles are
    unavailable.
    """
    if not _DEFER_FREE_ENABLED:
        return arr
    try:
        get_array = getattr(arr, "get_array", None)
        if get_array is None:
            return arr
        usm = get_array()                 # weak-referenceable usm_ndarray
        mem = usm.usm_data                # strong ref to _Memory owner
        nbytes = getattr(mem, "nbytes", 0)
        if nbytes < _DEFER_FREE_MIN_BYTES:
            return arr
        # Flush a lingering partial batch so buffers freed during a burst
        # of frees followed by pure compute (no more frees to trigger a
        # batch flush) do not stay pinned indefinitely. This piggybacks on
        # allocation activity and is throttled by _flush_stride so it does
        # not submit a host task on every allocation.
        cnt = _pending_frees.get("alloc_count", 0) + 1
        _pending_frees["alloc_count"] = cnt
        if (cnt % _DEFER_FREE_FLUSH_STRIDE) == 0 and _on_scheduler_safe_thread():
            with _defer_free_lock:
                _flush_deferred_frees_locked()
                _reap_completed_locked()
        # weakref.finalize on the usm_ndarray fires when it is collected;
        # `mem` captured in the finalizer keeps _Memory alive past that,
        # letting us order the real free behind queue work.
        weakref.finalize(usm, _deferred_release, mem)
    except Exception:
        # Never let lifetime-management bookkeeping break array creation.
        return arr
    return arr


_DPNP_ARRAY_PRODUCERS = (
    # binary arithmetic (and their reflected forms)
    "__add__", "__radd__", "__sub__", "__rsub__",
    "__mul__", "__rmul__", "__truediv__", "__rtruediv__",
    "__floordiv__", "__rfloordiv__", "__mod__", "__rmod__",
    "__pow__", "__rpow__", "__matmul__", "__rmatmul__",
    # unary
    "__neg__", "__pos__", "__abs__",
    # dtype / layout conversions that allocate
    "astype", "conj", "conjugate",
    "dot", "matmul", "copy", "flatten", "ravel", "repeat",
    "diagonal", "trace",
    "sum", "mean", "prod", "cumsum", "cumprod",
    "max", "min", "var", "std",
)


_DPNP_ALLOCATING_FUNCS = (
    # shape / joining -- these produce the buffers most likely to be handed
    # to a raw kernel as a borrowed pointer
    "vstack", "hstack", "dstack", "column_stack", "row_stack",
    "concatenate", "stack", "append", "repeat", "tile",
    # linear algebra / reductions producing fresh buffers
    "outer", "sum", "prod", "cumsum", "trace",
    # elementwise ufuncs (module-level forms; the operator forms are 4b)
    "exp", "log", "sqrt", "square", "abs", "sign",
    "multiply", "add", "subtract", "divide",
    # selection / construction
    "where", "take", "tril", "triu", "unique",
    # linear algebra / contractions producing fresh buffers (same # crash family as the .dot producer hole -- oneMKL/einsum outputs are
    # allocated inside dpnp C++ and escaped registration)
    "dot", "matmul", "vdot", "inner", "tensordot", "einsum", "kron",
    "cross", "ascontiguousarray",
)

# dpnp.linalg functions whose outputs are fresh device buffers. Reached via
# `cupy.linalg.*` (the facade aliases dpnp.linalg) -- e.g. cholesky in
# df.DF.build, solve/eigh in DIIS/SOSCF paths.
_DPNP_LINALG_ALLOCATING_FUNCS = (
    "cholesky", "inv", "solve", "pinv", "lstsq",
    "eigh", "eig", "eigvals", "eigvalsh",
    "qr", "svd", "matrix_power", "norm",
)


def _wrap_allocating_funcs():
    """Layer 4c -- attach the deferred-free finalizer to module-level dpnp
    functions that allocate.

    Same rationale as Layer 4b (`_wrap_array_producers`), but for functions
    reached as `dpnp.foo(...)` rather than as an operator on an ndarray.
    Deliberately does NOT inject `sycl_queue=` -- these are compute-follows-
    data operations that correctly inherit their queue from their inputs
    (verified: a global queue-identity probe over this whole code path found
    zero divergence). The ONLY thing being added is the queue-ordered free.

    Idempotent; failures are swallowed so a dpnp build missing any one of
    these names cannot break import.
    """
    return _wrap_allocating_funcs_on(dpnp, _DPNP_ALLOCATING_FUNCS)


def _wrap_allocating_funcs_on(mod, names):
    """Worker behind _wrap_allocating_funcs; also used for dpnp.linalg."""
    for name in names:
        orig = getattr(mod, name, None)
        if orig is None or getattr(orig, "__master_q_wrapped__", False):
            continue

        if isinstance(orig, types.FunctionType):
            @functools.wraps(orig)
            def wrapper(*args, _orig=orig, **kwargs):
                return _register_deferred_free(_orig(*args, **kwargs))

            wrapper.__master_q_wrapped__ = True
            wrapper.__wrapped__ = orig
        else:
            # `multiply`/`add`/`subtract`/`divide` are DPNPBinaryFunc
            # objects (ufunc-like), not plain functions -- they carry
            # callable attributes such as `.outer` that dpnp's own
            # implementations reach through the module attribute (e.g.
            # `dpnp.outer` calls `dpnp.multiply.outer(...)`). A
            # `functools.wraps` closure is a bare function and has no
            # `.outer`, so replacing the module attribute with one broke
            # any internal dpnp call that goes through it
            # (AttributeError: 'function' object has no attribute
            # 'outer', hit by dpnp_helper.krylov's QR step via
            # dpnp.outer -> dpnp.multiply.outer). Use an
            # attribute-forwarding proxy instead so calling the object
            # still hits the deferred-free path while every other
            # attribute resolves straight through to `orig`.
            class _DeferredFreeProxy:
                def __init__(self, orig):
                    self._orig = orig
                    self.__master_q_wrapped__ = True

                def __call__(self, *args, **kwargs):
                    out = kwargs.get('out', None)
                    if out is not None and any(out is arg for arg in args):
                        return self._orig(*args, **kwargs)
                    return _register_deferred_free(self._orig(*args, **kwargs))

                def __getattr__(self, attr):
                    return getattr(self._orig, attr)

            wrapper = _DeferredFreeProxy(orig)
        try:
            setattr(mod, name, wrapper)
        except (TypeError, AttributeError):
            continue


def _wrap_linalg_allocating_funcs():
    """Layer 4d -- deferred-free finalizers on dpnp.linalg outputs.

    Same rationale as Layer 4c, but the results are reached as
    `dpnp.linalg.foo(...)`, which the `cupy.linalg` facade aliases directly.
    Cholesky in df.DF.build, solve/eigh in DIIS and SOSCF paths all hand
    these buffers to raw kernels; an eager free of one of them is the
    GPU-page-fault crash family. Idempotent; missing names are
    skipped.
    """
    try:
        linalg_mod = dpnp.linalg
    except AttributeError:
        return
    _wrap_allocating_funcs_on(linalg_mod, _DPNP_LINALG_ALLOCATING_FUNCS)


def _wrap_array_producers():
    """Attach the deferred-free finalizer to arithmetic / astype results.

    Layer 3 only wraps dpnp's *creation* functions, so any array produced by
    an operator (`a * 2`) or a conversion (`a.astype(...)`) escaped Layer 4
    entirely and was released by dpctl's eager synchronous sycl::free. If such
    a buffer had been handed to a raw SYCL kernel as a borrowed pointer, that
    is a use-after-free.

    Wrapping the dunder on the *type* is required -- Python looks up operators
    on the type, not the instance. Idempotent, and failures are swallowed so a
    dpnp version without one of these names cannot break import.
    """
    try:
        arr_cls = dpnp.ndarray
    except AttributeError:
        return
    for name in _DPNP_ARRAY_PRODUCERS:
        orig = getattr(arr_cls, name, None)
        if orig is None or getattr(orig, "__master_q_wrapped__", False):
            continue

        @functools.wraps(orig)
        def wrapper(self, *args, _orig=orig, _name=name, **kwargs):
            # Skip in-place operations - they don't create new arrays
            if _name.startswith("__i") and _name.endswith("__"):
                return _orig(self, *args, **kwargs)
            return _register_deferred_free(_orig(self, *args, **kwargs))

        wrapper.__master_q_wrapped__ = True
        wrapper.__wrapped__ = orig
        try:
            setattr(arr_cls, name, wrapper)
        except (TypeError, AttributeError):
            # Immutable/extension type -- skip rather than fail import.
            continue


def _wrap_with_master_queue(mod, names):
    """Inject sycl_queue=master into every creation call on `mod`, and
    register a queue-ordered deferred-free finalizer on the result.

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
            return _register_deferred_free(_orig(*args, **kwargs))

        wrapper.__master_q_wrapped__ = True
        wrapper.__wrapped__ = orig
        setattr(mod, name, wrapper)

_DPNP_NATIVE_BLOCKING = {
    "dpnp.backend.extensions.blas._blas_impl": (
        "_dot", "_dotc", "_dotu", "_gemm", "_gemm_batch", "_gemv", "_syrk",
    ),
    "dpnp.backend.extensions.lapack._lapack_impl": (
        "_geqrf", "_geqrf_batch", "_gesv", "_gesv_batch", "_gesvd",
        "_gesvd_batch", "_getrf", "_getrf_batch", "_getri_batch", "_getrs",
        "_getrs_batch", "_heevd", "_heevd_batch", "_orgqr", "_orgqr_batch",
        "_potrf", "_potrf_batch", "_syevd", "_syevd_batch", "_ungqr",
        "_ungqr_batch",
    ),
}


def _drain_then(orig):
    def wrapper(*args, **kwargs):
        try:
            _master_queue().wait()
        except Exception:
            pass
        return orig(*args, **kwargs)

    wrapper.__name__ = getattr(orig, "__name__", "wrapped")
    wrapper.__doc__ = getattr(orig, "__doc__", None)
    wrapper.__gil_drained__ = True
    wrapper.__wrapped__ = orig
    return wrapper


def _wrap_blocking_lapack():
    import importlib

    for mod_name, fn_names in _DPNP_NATIVE_BLOCKING.items():
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            warnings.warn(
                f"Layer 5: could not import {mod_name} to guard against the "
                f"oneMKL/GIL host-task deadlock: {e}",
                RuntimeWarning)
            continue
        for name in fn_names:
            orig = getattr(mod, name, None)
            if orig is None or getattr(orig, "__gil_drained__", False):
                continue
            try:
                setattr(mod, name, _drain_then(orig))
            except Exception:
                # A pybind11 module that refuses attribute assignment would
                # leave the deadlock in place; say so rather than fail
                # silently.
                warnings.warn(
                    f"Layer 5: {mod_name}.{name} is not patchable; the "
                    "oneMKL/GIL host-task deadlock remains reachable there.",
                    RuntimeWarning)


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

    # Layer 2: replace dpctl's process-global queue cache with one that
    # always returns the per-device master queue -- but only if not already
    # replaced by a previous load.
    try:
        existing = qmgr._global_device_queue_cache
        if not isinstance(existing, _MasterQueueCache):
            qmgr._global_device_queue_cache = _MasterQueueCache()
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
    # Layer 4b: arithmetic / astype results also allocate, and Layer 3's
    # creation-API wrapping does not see them. Without this they are freed
    # eagerly regardless of size.
    _wrap_array_producers()
    # Layer 4c: module-level allocating functions (vstack/concatenate/sum/...)
    # are seen by neither Layer 3 nor 4b. This is the gap that left `dms` --
    # produced by cp.vstack() in _VHFOpt.get_jk's hermi==0 path and handed to
    # RYS_build_jk as a raw pointer -- eligible for eager, non-queue-ordered
    # sycl::free while kernels were still reading it.
    _wrap_allocating_funcs()
    # Layer 4d: dpnp.linalg outputs (cholesky/solve/eigh/...). Same gap
    # family as 4c -- oneMKL results allocated inside dpnp C++.
    _wrap_linalg_allocating_funcs()
    # Layer 4e SAFETY NET ): wrong-type (numpy-into-dpnp)
    # usage is fixed at the gpu4pyscf call sites — this shim is a dormant
    # backstop so a missed site fails soft instead of crashing a run.
    # 4e: CuPy's ndarray implements __array__ as an implicit device-to-host
    # copy; dpnp raises. Keep the CuPy behavior available.
    try:
        import numpy as _np
        _dpnp_arr_cls = dpnp.ndarray
        if not getattr(_dpnp_arr_cls, "__cupy_shim_array__", False):
            def _dpnp_shim_array(self, dtype=None, copy=None):
                host = self.asnumpy()
                return _np.array(host, dtype=dtype, copy=True) if dtype is not None else _np.array(host)
            _dpnp_shim_array.__cupy_shim_array__ = True
            _dpnp_arr_cls.__array__ = _dpnp_shim_array
    except Exception:
        pass
    try:
        import operator as _op

        def _norm_idx(v):
            # CuPy accepts integral float slice bounds (it int()-converts
            # them in its own slicer); dpnp's Cython slicer does not.
            # Floats have no __index__, so convert integral floats
            # explicitly, everything else via __index__.
            if v is None:
                return None
            if isinstance(v, float):
                return int(v) if v.is_integer() else v
            try:
                return _op.index(v)
            except (TypeError, ValueError):
                return v

        def _norm_key(k):
            if isinstance(k, slice):
                return slice(_norm_idx(k.start), _norm_idx(k.stop), _norm_idx(k.step))
            if isinstance(k, tuple):
                return tuple(
                    slice(_norm_idx(x.start), _norm_idx(x.stop), _norm_idx(x.step))
                    if isinstance(x, slice) else x
                    for x in k
                )
            return k

        import numpy as np

        def _split_advanced_key(ndim, key):
            # NumPy/CuPy accept advanced (array) indices separated by basic
            # slices, e.g. `x[i,:,j]`: the broadcast advanced dims go first,
            # then the sliced dims. dpnp raises "Advanced indexing array
            # specs may not be separated by basic slicing specs" (hit by
            # pbc/scf/rsjk.py `mask_exclude_dd[ish,:,jsh] = False`).
            # Same result: permute the advanced axes to the front, then
            # index with the advanced specs adjacent. Returns (perm, key)
            # or None when dpnp handles the key as is.
            if not isinstance(key, tuple):
                return None
            adv = []
            has_array = False
            for p, k in enumerate(key):
                if k is None or k is Ellipsis or isinstance(k, (bool, np.bool_)):
                    return None
                if isinstance(k, slice):
                    continue
                if isinstance(k, (int, np.integer)):
                    adv.append(p)
                    continue
                if isinstance(k, (list, np.ndarray)) or hasattr(k, "get_array"):
                    if getattr(k, "dtype", None) is not None and k.dtype.kind == "b":
                        return None
                    has_array = True
                    adv.append(p)
                    continue
                return None
            if not has_array or len(key) > ndim:
                return None
            if adv[-1] - adv[0] + 1 == len(adv):
                return None  # already adjacent
            rest = [p for p in range(ndim) if p not in adv]
            perm = tuple(adv + rest)
            new_key = tuple(key[p] for p in adv) + tuple(
                key[p] for p in rest if p < len(key))
            return perm, new_key

        _getitem_register = (
            os.environ.get("GPU4PYSCF_GETITEM_REGISTER", "1") != "0"
        )
        _orig_setitem = dpnp.ndarray.__setitem__
        if not getattr(_orig_setitem, "__cupy_shim_setitem__", False):
            def _dpnp_shim_setitem(self, key, value, _orig=_orig_setitem):
                split = _split_advanced_key(self.ndim, key)
                if split is not None:
                    perm, key = split
                    return _orig(self.transpose(perm), key, value)
                return _orig(self, key, value)
            _dpnp_shim_setitem.__cupy_shim_setitem__ = True
            _dpnp_shim_setitem.__wrapped__ = _orig_setitem
            dpnp.ndarray.__setitem__ = _dpnp_shim_setitem
        _orig_getitem = dpnp.ndarray.__getitem__
        if not getattr(_orig_getitem, "__cupy_shim_getitem__", False):
            def _dpnp_shim_getitem(self, key):
                key = _norm_key(key)
                split = _split_advanced_key(self.ndim, key)
                if split is not None:
                    perm, key = split
                    self = self.transpose(perm)
                res = _orig_getitem(self, key)
                # Fancy/advanced indexing COPIES into a fresh device buffer
                # created inside dpnp's Cython slicer; without registration
                # its GC ran an eager sycl::free while raw kernels reading
                # it were still in flight -> GPU "Read (NotPresent)" page
                # fault -> NEO background-thread abort (the df
                # `_jk_energy_per_atom` dm_aux[aux_sorting[:,None],
                # aux_sorting] crash family, ). Basic-slice views
                # share self's USM pointer -- registering those would stack
                # duplicate finalizers on the same usm_ndarray with no
                # lifetime benefit, so only distinct-pointer results are
                # registered.
                try:
                    if (_getitem_register and
                            res is not self and
                            getattr(res, "get_array", None) is not None and
                            res.data.ptr != self.data.ptr):
                        return _register_deferred_free(res)
                except Exception:
                    pass
                return res
            _dpnp_shim_getitem.__cupy_shim_getitem__ = True
            dpnp.ndarray.__getitem__ = _dpnp_shim_getitem
        def _dpnp_array_ufunc(self, ufunc, method, *inputs, **kwargs):
            if method != '__call__':
                return NotImplemented
            fn = getattr(dpnp, ufunc.__name__, None)
            if fn is None:
                return NotImplemented
            out = kwargs.get('out')
            if isinstance(out, tuple):
                if len(out) != 1:
                    return NotImplemented
                kwargs['out'] = out[0]
            return fn(*inputs, **kwargs)
        dpnp.ndarray.__array_ufunc__ = _dpnp_array_ufunc

        # CuPy ufuncs accept Python/NumPy scalars (cupy.sqrt(2.) is a 0-d
        # device array); dpnp's unary ufuncs, and binary ones with two
        # scalars, require an array. Promote a leading scalar only; host
        # ndarrays still raise, as in CuPy.
        from dpnp.dpnp_algo import dpnp_elementwise_common as _ewc
        _scalar = (int, float, complex, np.generic)

        def _accept_scalars(cls, nin):
            orig_call = cls.__call__
            if getattr(orig_call, '__cupy_scalar__', False):
                return
            def __call__(self, *args, **kwargs):
                if args:
                    # NumPy scalars (and 0-d host arrays, which is how numpy
                    # hands scalars to __array_ufunc__) become 0-d device
                    # arrays of the same dtype (dpnp rejects them; keeps e.g. float32 compares
                    # in float32).
                    args = tuple(dpnp.asarray(a) if i < nin and (
                                     isinstance(a, np.generic) or
                                     (isinstance(a, np.ndarray) and a.ndim == 0)) else a
                                 for i, a in enumerate(args))
                    if (isinstance(args[0], _scalar)
                            and not any(hasattr(a, 'get_array') for a in args[:nin])):
                        args = (dpnp.asarray(args[0]),) + args[1:]
                return orig_call(self, *args, **kwargs)
            __call__.__cupy_scalar__ = True
            cls.__call__ = __call__
        _accept_scalars(_ewc.DPNPUnaryFunc, 1)
        _accept_scalars(_ewc.DPNPBinaryFunc, 2)

        # ufunc.at (unbuffered in-place accumulation, repeated indices add
        # up), used as cupy.add.at / cupyx.scatter_add. dpnp has no .at.
        # A 1-D integer index on axis 0 runs on device through bincount;
        # other index forms take a host round trip (correct, slower).
        def _ufunc_at(self, a, indices, b=None):
            name = getattr(self, 'name_', getattr(self, '__name__', ''))
            if name not in ('add', 'subtract'):
                raise NotImplementedError(f'dpnp ufunc.at for {name!r}')
            vals = dpnp.asarray(b, dtype=a.dtype) if b is not None else None
            if name == 'subtract':
                vals = -vals
            idx = indices[0] if isinstance(indices, tuple) and len(indices) == 1 else indices
            if hasattr(idx, 'get_array'):
                one_d = idx.ndim == 1 and idx.dtype.kind in 'iu'
            elif isinstance(idx, (np.ndarray, list)):
                one_d = np.asarray(idx).ndim == 1 and np.asarray(idx).dtype.kind in 'iu'
            else:
                one_d = False
            if one_d:
                idx = dpnp.asarray(idx)
                n, rows = idx.shape[0], a.shape[0]
                vals = dpnp.broadcast_to(vals, (n,) + a.shape[1:]).reshape(n, -1)
                acc = dpnp.stack([dpnp.bincount(idx, weights=vals[:, k], minlength=rows)
                                  for k in range(vals.shape[1])], axis=1)
                a += acc.reshape(a.shape).astype(a.dtype, copy=False)
                return
            host = dpnp.asnumpy(a)
            np.add.at(host, tuple(dpnp.asnumpy(x) if hasattr(x, 'get_array') else x
                                  for x in (indices if isinstance(indices, tuple) else (indices,))),
                      dpnp.asnumpy(vals))
            a[...] = dpnp.asarray(host)
        _ewc.DPNPBinaryFunc.at = _ufunc_at
    except Exception:
        pass
    _state["bootstrapped"] = True


_bootstrap()


# =====================================================================
# Runtime verification -- catches regressions early.
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
        if int(libgpu.sycl_get_queue_ptr() or 0) != _get_sycl_queue_ptr(q):  # <- fixed
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
            "worker-thread dpnp allocation escaped master queue -- "
            "ContextVar replacement regressed"
        )

    _state["verified"] = True


_verify_single_queue_invariant()


# =====================================================================
# Shutdown guard
# =====================================================================
@atexit.register
def _mark_shutdown():
    """Flush and release outstanding deferred frees, then mark shutdown.

    With the polled reaper there is no host task to self-guard against
    interpreter teardown (dpctl's async_dec_ref used to skip its DECREF via
    Py_IsFinalizing()), so retained _Memory objects would otherwise survive to
    process exit. We drain explicitly: flush any partial batch, wait once on
    each master queue, then release everything. This is the only wait added by
    the reaper design and it runs at process exit only, never on a hot path.
    """
    try:
        with _defer_free_lock:
            _flush_deferred_frees_locked()
        for q in list(_master_queues.values()):
            q.wait()
        with _defer_free_lock:
            _reap_completed_locked()
    except Exception:
        pass
    _state["shutting_down"] = True

def _shutting_down():
    return _state["shutting_down"]

def rebuild_dpnp_array(host, cls, state):
    """Unpickle reconstructor for dpnp arrays -- see gpu4pyscf/cupy/__init__.py.

    Lives here rather than in the `cupy` shim package because pickle has to
    import the reconstructor by qualified name, and the shim is registered
    under a synthetic module name.
    """
    arr = dpnp.asarray(host, sycl_queue=_master_queue())
    if cls is not dpnp.ndarray:
        arr = arr.view(cls)
    if state:
        arr.__dict__.update(state)
    return arr


def release_deferred_frees():
    """Return every deferred-free allocation to the driver, now.

    Backs `cupy.get_default_memory_pool().free_all_blocks()`. The ~30 call
    sites for that in gpu4pyscf are memory-pressure relief points; with the
    deferred-free reaper they are exactly where the retained batches should be
    handed back. Same flush -> wait -> reap sequence as `_mark_shutdown()`,
    minus the shutdown flag.

    Off the owning thread this is a no-op: the flush submits a barrier on the
    master queue (scheduler WRITER lock) and the wait drains it, which is the
    exact reader/writer deadlock combination documented at
    `_on_scheduler_safe_thread` when the main thread is parked inside its own
    queue wait. Batches freed from worker threads stay retained until the next
    main-thread flush; they are bounded by `_DEFER_FREE_BATCH` /
    `_DEFER_FREE_OFFTHREAD_CAP`, so nothing leaks.
    """
    if not _on_scheduler_safe_thread():
        return
    try:
        with _defer_free_lock:
            _flush_deferred_frees_locked()
        for q in list(_master_queues.values()):
            q.wait()
        with _defer_free_lock:
            _reap_completed_locked()
    except Exception:
        pass



# =====================================================================
# Stream -- singleton per device, wraps master SyclQueue.
# Uses the shared _stream_cache on _state so both module copies (if any)
# hand out the same Stream instance per device.
# =====================================================================
class Stream:
    """CuPy-compatible singleton Stream wrapping the master SyclQueue.

    The constructor arguments (null, non_blocking, ptds) are accepted
    for CuPy API parity but ignored -- every Stream for a given device
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
    def ptr(self):
        return self._ptr

    def __eq__(self, other):
        if isinstance(other, Stream):
            return self._ptr == other._ptr
        # CUDA's default/legacy stream is the null pointer (0); the shim has a
        # single in-order master queue, which plays that role.
        if isinstance(other, int) and other == 0:
            return True
        return NotImplemented

    def __hash__(self):
        return hash(self._ptr)

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

    def wait_event(self, event):
        # Every Stream is the same in-order master queue, so any work the
        # event was recorded after is already ordered before later
        # submissions on this "stream" -- nothing to wait for.
        pass

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

def get_device_id() -> int:
    """Current SYCL device id (maps to CUDA cudaGetDevice).

    Queries the registered SYCL queue's device.
    """
    return int(libgpu.sycl_get_device_id())


# =====================================================================
# Device -- singleton per id, backed by the shared _device_cache on _state.
# =====================================================================
class Device:
    """Singleton-per-id Device wrapper -- CuPy Device(0) semantics."""

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
        """Drain the device's master queue -- superset of cudaDeviceSynchronize."""
        _master_queue(self._id).wait()

    @property
    def mem_info(self):
        return (get_free_memory(), get_total_memory())


device = Device


def _get_cublas_handle():
    # oneMKL calls take the device queue, not a handle; kept for CuPy API
    # parity (cupy.cuda.device.get_cublas_handle()).
    return 0
Device.get_cublas_handle = staticmethod(_get_cublas_handle)


# ---------------------------------------------------------------------
# cupy.cuda.cublas subset used by gpu4pyscf (cc/ccsd_incore.py): raw dgemm
# with cuBLAS argument conventions -- column-major, alpha/beta passed as
# host pointers (cuBLAS default pointer mode), device pointers for A/B/C.
# ---------------------------------------------------------------------
class _Cublas:
    CUBLAS_OP_N = 0
    CUBLAS_OP_T = 1
    CUBLAS_OP_C = 2

    _lib = None

    @classmethod
    def _onemkl(cls):
        if cls._lib is None:
            lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib'))
            lib = ctypes.CDLL(os.path.join(lib_dir, 'libonemkl_helper.so'))
            lib.onemkl_dgemm.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_long, ctypes.c_long, ctypes.c_long,
                ctypes.c_double, ctypes.c_void_p, ctypes.c_long,
                ctypes.c_void_p, ctypes.c_long,
                ctypes.c_double, ctypes.c_void_p, ctypes.c_long]
            lib.onemkl_dgemm.restype = None
            cls._lib = lib
        return cls._lib

    @classmethod
    def dgemm(cls, handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
              beta, C, ldc):
        a = ctypes.c_double.from_address(int(alpha)).value
        b = ctypes.c_double.from_address(int(beta)).value
        cls._onemkl().onemkl_dgemm(int(transa), int(transb), int(m), int(n), int(k),
                                   a, int(A), int(lda), int(B), int(ldb),
                                   b, int(C), int(ldc))


cublas = _Cublas


# =====================================================================
# Event -- wall-clock timing + queue.wait() sync
#
# We use the host clock for elapsed-time math and queue.wait() as the
# sync primitive. On an in-order queue, queue.wait() is a strict superset
# of 'wait for the barrier we would have submitted'. (For lifetime
# ordering the deferred-free reaper waits barrier events directly -- see
# _reap_completed_locked.)
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
        # A per-record submit_barrier() so query() could poll instead of
        # block aborted natively on DPC++ 2026.1.0 (the UR internal-event
        # bug, see _reap_completed_locked). Not retried on 2026.2.0:
        # record() stays a host timestamp and query() stays blocking.

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
        # Finalizer never touches GPU work -- queue may be in teardown.
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
# Address helper -- used by _Runtime.memcpy
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
    def deviceSynchronize():
        for d in range(get_device_count()):
            _master_queue(d).wait()

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
        # Always False, so gpu4pyscf.__config__._p2p_access is False and the
        # multi-GPU paths (p2p_transfer, array_reduce, concatenate) take their
        # explicit-copy fallbacks. dpnp/dpctl reject any op whose operands
        # live on different queues ("Execution queue is not compatible with
        # allocation queues", ExecutionPlacementError) even with Level Zero
        # peer access enabled; verified on PVC tiles . cudaDeviceCanAccessPeer(d, d) is 0 as well.
        return False


runtime = _Runtime()


# =====================================================================
# Pinned-memory allocator (attached to master queue)
# =====================================================================
class _PinnedMemoryPointer:
    """Minimal CuPy `MemoryPointer`-like handle for pinned host memory.

    CuPy's `cupy.cuda.alloc_pinned_memory` returns a MemoryPointer exposing
    `.ptr`; the codebase (gto/int3c1e*.py, df/int3c2e.py, cupyx, and
    test_cupy_helper) uses `.ptr` plus the buffer protocol via
    `numpy.ndarray(..., buffer=...)`. dpctl's `_Memory` already implements
    the buffer protocol, so we only add `.ptr` and keep the owner alive.
    """

    __slots__ = ("ptr", "mem")

    def __init__(self, mem):
        self.mem = mem
        self.ptr = mem._pointer

    def __buffer__(self, flags):
        # Python 3.12 buffer protocol delegation (numpy ndarray(buffer=...)).
        return memoryview(self.mem).__buffer__(flags)


def alloc_pinned_memory(nbytes, flags=None):
    nbytes = int(nbytes)
    q      = _master_queue()
    # Always USM-host, mapped or not. CUDA pinned memory (with or without
    # hostAllocMapped) is page-locked host memory; USM-host is the match.
    # USM-shared is NOT: its pages migrate to the GPU and NEO services the
    # next host touch through a SIGSEGV handler that libigc can leave
    # displaced after concurrent JIT compiles (see empty_mapped in
    # gpu4pyscf/lib/dpnp_helper.py). `flags` is accepted for API parity.
    # dpctl rejects zero-byte allocations; CuPy accepts them.
    return _PinnedMemoryPointer(dpmem.MemoryUSMHost(max(nbytes, 1), queue=q))


@contextlib.contextmanager
def using_allocator(allocator=None):
    release_deferred_frees()
    try:
        yield
    finally:
        release_deferred_frees()


# =====================================================================
# cupy.cuda.memory shim
#
# CuPy exposes allocation failures as `cupy.cuda.memory.OutOfMemoryError`.
# Code that only ever *catches* that class (e.g. lib/cutensor.py) would
# otherwise raise AttributeError while unwinding, masking whatever the
# original exception actually was. Provide the name, backed by dpctl's
# USM allocation error plus the builtin MemoryError.
# =====================================================================
import sys as _sys

_memory_mod = types.ModuleType('cupy.cuda.memory')

# A tuple is a valid `except` target, so this stays usable as
# `except cupy.cuda.memory.OutOfMemoryError:` while covering both the
# dpctl USM failure and a plain host MemoryError.
_oom_types = [MemoryError]
if hasattr(dpmem, 'USMAllocationError'):
    _oom_types.insert(0, dpmem.USMAllocationError)
OutOfMemoryError = tuple(_oom_types)

_memory_mod.OutOfMemoryError = OutOfMemoryError
memory = _memory_mod
_sys.modules.setdefault('cupy.cuda.memory', _memory_mod)
