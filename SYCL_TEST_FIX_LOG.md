# SYCL test-fix pass — scf/ df/ dft/

Environment: Intel PVC (Data Center GPU Max 1550, 12 tiles), SYCL backend,
branch `fix/sycl-create-tasks-barrier`. Failures caused by the on-GPU libxc /
ExchCXX functional table being unavailable (`RuntimeError: failed to initialize
xc fun`, `Failed in xc_gga/xc_mgga`) are out of scope per instruction.

## Fixes applied

### 1. `gpu4pyscf/cupy/cuda.py` — oneMKL/GIL host-task deadlock (Layer 5)
**Symptom:** `gpu4pyscf/df/tests/test_df_rks.py::test_rks_b3lyp` and many other
df/ tests hung indefinitely (killed at a 1500 s timeout with zero tests
completed).

This is the Python-side manifestation of **intel/llvm#22943** —
*"[SYCL][UR] Hangs when using `in-order` and SYCL `host_task` under
multi-threading"* (open as of 2026-08-14; reproduces on PVC 1550 with both
Level-Zero and OpenCL, not on CUDA/HIP). An in-order queue plus a host task
that takes a lock deadlocks. Here the lock is the GIL.

The workaround the issue recommends — use an out-of-order queue — is not
available to this port: `libgint`/`libgvhf`/`libgdft` are handed the raw
`sycl::queue*` and launch kernels on it with no event plumbing across the
ctypes boundary, and the deferred-free reaper tags batches with barriers on the
assumption of in-order semantics. So the mitigation is instead to guarantee no
GIL-needing host task is ever pending at the moment oneMKL blocks.

**Root cause** (confirmed by `gdb thread apply all bt` on a live hang):
dpnp's LAPACK extension calls `oneapi::mkl::lapack::potrf`, which blocks
internally on `sycl::event::wait()` *without releasing the GIL*. On the
in-order master queue that wait transitively covers every earlier command,
including the keep-alive host tasks dpctl attaches to Python operands. Such a
host task runs on a SYCL `ThreadPool` worker and calls `PyEval_AcquireThread`
to DECREF — but the GIL is held by the caller blocked inside oneMKL.
Permanent deadlock.

```
Thread 1: mkl::lapack::potrf_dispatch -> event_impl::waitInternal   [holds GIL]
Thread 8: DispatchHostTask -> dpnp/tensor/_tensor_impl -> take_gil  [wants GIL]
```

**Fix:** new `_wrap_blocking_lapack()`, installed by `_bootstrap()`. Wraps every
blocking `dpnp.linalg.*` entry point so it first drains the master queue.
dpctl's `SyclQueue.wait()` is declared `with nogil`, so the drain retires all
pending host tasks before oneMKL can wait on one.

**Effect:** `test_rks_b3lyp` 25 min hang -> 4.65 s pass. Full
`test_df_rks.py`: 10 passed, 2 failed (both xc-fun, out of scope).

### 2. `gpu4pyscf/gto/ecp.py` — `NameError: name 'libecp' is not defined`
The lazy-loading refactor kept bare `libecp.` references inside functions in the
same module. PEP 562 module `__getattr__` only fires for *attribute* access from
outside, never for a global-name lookup inside the module, so every ECP call
raised `NameError`. Call sites now use `_load_libecp()`.

### 3. `gpu4pyscf/lib/CMakeLists.txt` — re-enabled `add_subdirectory(ecp)`
`libgecp.so` on disk was stale (linked against `libsycl.so.8`; the toolchain is
now `libsycl.so.9`), so it failed to `dlopen`. Rebuilt against the current
oneAPI. Note: the ECP AOT device compile (`ocloc`, `spir64_gen`,
`-fsycl-device-code-split=per_kernel`) is single-threaded and took **47
minutes** / 8.6 GB RSS — presumably why the subdirectory was commented out.
Consider `-fsycl-targets=spir64` (JIT) for this one target if build time
matters.

**Effect:** with fixes 2 and 3 together, all ECP tests pass —
`scf/tests/test_scf_ecp.py` + `dft/tests/test_dft_ecp.py` +
`df/tests/test_df_ecp.py` = **5 passed, 0 failed** (was 5 failed).

### 4. `gpu4pyscf/tdscf/ris.py` — host operands reaching `contract()`
cuTENSOR is unavailable here, so `contract()` routes to `_contract_einsum`
instead of `contraction()`. Unlike `contraction()` — which does
`a = cupy.asarray(a, dtype=dtype)` — the einsum fallback does not upload host
arrays, so `tdscf.ris` failed with `TypeError: An array must be any of supported
type, but got <class 'numpy.ndarray'>`. `get_Tpq(..., in_ram=True)` is the
default and deliberately keeps `T_ia`/`T_ab` in host RAM, streaming one chunk at
a time.

First attempt added the same coercion to `_contract_einsum`. Replaced, following
the call-site idiom of gpu4pyscf#810 / #851 (fix the namespace where the data
lives; do not paper over it in the library; comment intentional transfers):
`cutensor.py` is back to pristine and the three call sites now say
`cp.asarray(...)` explicitly with an "Intentional CPU->GPU transfer" comment.

Audited rather than assumed. Instrumenting `_contract_einsum` to log every host
operand with its call site, over `test_df_tddft_ris`, `test_df_tdrks_ris_grad`,
`test_df_rks`, `test_df_rhf`, `test_df_uks`, `dft/test_rks`, `dft/test_uks` and
`scf/test_scf`, found exactly three, all in `tdscf/ris.py` (lines 516, 581, 584),
all operand `a`, 0.2 MB total:

```
   20x  tdscf/ris.py:516   Pab,mjb->Pamj   host operand: a
   16x  tdscf/ris.py:581   Pib,mjb->Pimj   host operand: a
   16x  tdscf/ris.py:584   Pja,Pimj->mia   host operand: a
```

After the change the same audit reports `NONE`, and any future host operand now
fails loudly instead of transferring silently. `cp.asarray` is a no-op when
`in_ram=False`, so both modes are covered.

### 5. `gpu4pyscf/cupy/cuda.py` — added the `cupy.cuda.memory` submodule
`lib/cutensor.py` catches `cupy.cuda.memory.OutOfMemoryError`. The attribute did
not exist, so *any* exception raised inside `_contract_einsum` was replaced by
`AttributeError` while unwinding, masking the real error. `OutOfMemoryError` is
now a tuple of `(dpctl.memory.USMAllocationError, MemoryError)`.

### 6. `gpu4pyscf/lib/dpnp_helper.py` — `CPArrayWithTag.T` lost the subclass
dpnp builds plain `dpnp_array` results for every method, so a tagged array
degraded to an untaggable array on `.T`. `df_jk._transpose_dm` relies on CuPy's
behaviour (`dm.T` is still a subclass instance and accepts attribute
assignment) and raised `AttributeError: 'dpnp_array' object has no attribute
'factor_l'`. `.T` now re-views to `CPArrayWithTag`.

### 7. `gpu4pyscf/cupy/__init__.py` — `cupy.random.seed` rejected array seeds
pyscf's own tests call `cupy.random.seed(np.asarray(1, dtype=np.uint64))`;
dpnp raised `TypeError: Cannot construct a dtype from an array`. 0-d array-likes
are now coerced. Verified the seeded stream is identical either way.

### 8. `gpu4pyscf/cupy/__init__.py` — `dpnp_array.view()` dropped the buffer offset
**The most consequential bug found so far.** `dpnp_array._create_view()` rebuilds
the result as `dpt.usm_ndarray(shape, dtype, buffer=self._array_obj, strides=...)`
and never forwards `self._array_obj._element_offset`. dpctl reads
`buffer=<usm_ndarray>` as the *whole* underlying allocation, so any array not
starting at the base of its allocation produced a view onto the wrong memory:

```python
x = dpnp.arange(10.); x[3:].view()   # -> [0. 1. 2. 3. 4. 5. 6.]
                                     # NumPy/CuPy: [3. ... 9.]
```

`dpnp.einsum` takes a `returns_view` branch for a single operand with no summed
index — any pure permutation, including the identity `'abcd->abcd'` — and does
`operands = [a.view() for a in operands]`. So `cp.einsum` over *any sliced
operand* silently read from the base of the parent buffer. Two-operand and
reducing einsums were unaffected, which is why `contract()` and the whole
matrix-vector-product path looked clean while `tdscf.ris.get_ab()`'s
`cp.einsum('iabj->iajb', eri_mo_J[:nocc, nocc:, nocc:, :nocc])` returned garbage.

The fix wraps `_create_view` to forward `offset=usm_obj._element_offset`,
preserving the subclass branch and the 0-d `strides=None` case. Because it fixes
`.view()` rather than einsum, it also repairs the ~155 other `.view()` call
sites in the tree, all of which were silently wrong for offset arrays. This is
an upstream dpnp bug and is worth filing.

**Effect:** `test_df_tddft_ris.py` 2 failed -> 2 passed.
`test_df_tdrhf_grad.py` 4 failed/2 passed -> 2 failed/4 passed (the "Krylov
subspace iterations diverge" failures are gone).

### 9. `gpu4pyscf/cupy/__init__.py` — `free_all_blocks()` was a no-op
`cupy.get_default_memory_pool().free_all_blocks()` is called at ~30 memory-
pressure relief points in gpu4pyscf and did nothing here. With the deferred-free
reaper those are exactly the points where retained batches should go back to the
driver. It now calls the new `cuda.release_deferred_frees()` (flush -> wait ->
reap, the same sequence `_mark_shutdown()` uses). `free_all_free` is kept as the
deprecated CuPy alias.

### 10. `gpu4pyscf/lib/gvhf/CMakeLists.txt` — unresolved device symbol -> SIGABRT
**Symptom:** `test_df_jk.py::test_j_outcore` died with `Fatal Python error:
Aborted` (SIGABRT) inside `GINTbuild_j_int3c2e_pass1`; pytest swallowed the
message. Standalone with stderr unredirected:

```
terminate called after throwing an instance of 'sycl::_V1::exception'
  what():  The program was built for 1 devices
Build program log for 'Intel(R) Data Center GPU Max 1550':
Module <0x...>:  Unresolved Symbol <s_bpcache>       (x16)
```

**Root cause:** this branch renames libgvhf's device global `s_bpcache` ->
`s_gvhf_bpcache` (correctly — both libraries previously exported a
default-visibility `s_bpcache`, so the dynamic linker interposed them and
gvhf's host-side memcpy landed in libgint's device image). The rename is
incomplete: `nr_jk_driver_int3c2e_pass{1,2}.cu` textually `#include`
`gint/g2e.cu` and `gint/cint2e.cuh`, which still say `s_bpcache`
(`g2e.cu:30,440,619,938`; `cint2e.cuh:23,57,70` including the `KERNEL_SETUP()`
macro). Those references land in *libgvhf's* device image, which after the
rename defines only `s_gvhf_bpcache`. Host linkage still succeeds via
`target_link_libraries(gvhf PRIVATE gint)`, so nothing fails until the Level
Zero program build at first launch.

Only `test_j_outcore` tripped it because the incore tests use
`auxbasis='sto3g'`, already in the persistent SYCL/NEO cache; the outcore test
uses the default `def2-universal-jfit` and forces a fresh program build. Any
cold cache hits it.

**Fix:** `target_compile_definitions(gvhf PRIVATE s_bpcache=s_gvhf_bpcache)`.
PRIVATE, so libgint and every other target are untouched, and the CUDA branch is
unaffected. gvhf is the only library outside `gint/` that includes gint device
sources, so no other target has the same latent defect.

**Effect:** `test_df_jk.py` 6 passed (was: aborted).

### 11. Layer 5 extended to the native BLAS entry points
The first cut of the deadlock fix wrapped `dpnp.linalg.*`. That is the wrong
boundary: those functions make copies and temporaries *before* reaching oneMKL,
each registering a fresh keep-alive host task, so a drain done at the public
entry point is already stale. A second live deadlock was caught with py-spy on
`coeff @ dm0` in `int3c2e.get_j_int3c2e_pass1` — `bi._gemm` blocking the same
way `potrf` does, reached through `dpnp.matmul`, which the `dpnp.linalg` list
never covered.

Layer 5 now drains at the pybind11 extension boundary itself, for every
blocking routine in `dpnp.backend.extensions.blas._blas_impl` and
`dpnp.backend.extensions.lapack._lapack_impl`.

Measured cost: none — 0.240 ms/matmul with the drain vs 0.250 ms without
(512x512, 300 iterations). On an in-order queue with host-task keep-alives dpnp
is already effectively synchronous per operation, so there is no pipelining to
lose.

### 12. `gpu4pyscf/lib/dpnp_helper.py` — tags lost on indexing too
Extension of fix 6. `nac.tdrhf_grad_nacv._dms_to_list` iterates a tagged stack
and assigns `dm.factor_l = ...` onto each element; dpnp's `__getitem__` returns
a plain `dpnp_array`, so it raised `AttributeError: 'dpnp_array' object has no
attribute 'factor_l'`. `CPArrayWithTag` now re-views the result of
`__getitem__`, `.T`, `reshape` and `transpose`. As in CuPy the tags themselves
are *not* propagated — only the ability to hold them, which is the correct
semantics here (a slice of a stacked density matrix must not inherit the
parent's `factor_l`).

### 13. `gpu4pyscf/nac/tdrhf_grad_nacv.py:346` — `np.diag` on a device array
`im0[-1, :nocc, :nocc] += np.diag(mo_energy[:nocc]) * 2.0` where
`mo_energy = cp.asarray(mf.mo_energy)` (line 96). NumPy attempts an implicit
host conversion, which both dpnp and CuPy refuse
(`TypeError: Implicit conversion to a NumPy array is not allowed`). Line 298 of
the same function already uses `cp.diag`; this is the same call. Not
SYCL-specific — it would fail on CUDA too.

**Effect of 12 + 13:** `test_df_tdrhf_nac_batch.py` 5 failed -> **5 passed**.

### 14. `np.array_equal` on device arrays — fixed at the call site, shim removed
`dft/tests/test_numint.py::test_sparse_index` failed because
`np.array_equal(r, x)` returned `False` for two *bit-identical* dpnp arrays.
This failure mode is worse than the usual namespace mismatch: `np.array_equal`
coerces its operands inside a `try/except` and **returns False on failure**, so
a backend that refuses implicit host conversion produces a wrong answer with no
exception. Everything else in this family raises loudly.

First attempt added `__array_function__` to `dpnp_array` so `np.foo(device)`
would dispatch to `dpnp.foo` the way CuPy does. Replaced with a call-site fix
after auditing, per the gpu4pyscf#810 / #851 idiom.

**Audit.** Instrumented the shim to log every dispatch with its call site:
5 test files gave exactly one consumer — `dft/tests/test_numint.py:275`,
90 calls, all `np.array_equal`. Nothing else in the tree used it.

**Call-site fix** — stay in the arrays' own namespace, pull a single bool:

```python
assert all(r.shape == x.shape and bool((r == x).all())
           for r, x in zip(ref[1:], dat[i][1:]))
```

Works for both the device and host entries of the sparse-index tuple (the pairs
are same-type on both sides).

**Shim removed** (46 lines). Re-audited afterwards across all of `dft/`, six
`scf/` files and five `df/` files — 194 tests — and the dispatch log reports
`NONE`. Post-removal check: `test_numint`, `test_numint2c`, `test_ao_values`,
`test_df_rhf`, `test_df_jk`, `test_cphf` -> **57 passed, 1 xfailed**, exit 0.

Residual risk worth knowing: with the shim gone, a *future* stray
`np.array_equal(device, device)` would again fail silently rather than raise.
Every other `np.*`-on-device-array misuse still fails loudly.

### 15. `gpu4pyscf/cupy/__init__.py` — dpnp arrays are not picklable
`cupy.ndarray` pickles (round-tripping through host memory) and pyscf relies on
it: `dft/tests/test_rks.py::test_rks_lda` does
`pickle.loads(pickle.dumps(mf))` on a converged mean-field object. `dpnp_array`
is a Cython extension type with a non-trivial `__cinit__` and no `__reduce__`,
so it raises `TypeError: no default __reduce__ due to non-trivial __cinit__`.
Known upstream gap — **IntelPython/dpnp#2602 "Cannot serialize arrays"**, still
open — so the shim adds `__reduce__`. It round-trips through NumPy and rebuilds
on the master queue (preserving the single-queue invariant `cuda.py` enforces);
the reconstructor lives in `cupy/cuda.py` because pickle must import it by
qualified name and the shim package is registered under a synthetic name.
`CPArrayWithTag` tags survive the round trip, matching CuPy.

### 16. Host-side norms called through `cupy.linalg` (upstream cleanup)
`df/tests/test_df_uhf.py` and `df/tests/test_df_rks_grad.py` compare an
analytic gradient against a finite-difference one with
`cupy.linalg.norm(g_analy - grad_fd)`. Both operands are **host** arrays there
(the subtraction succeeds, which under dpnp it could not if either were on the
device). CuPy's linalg entry points begin with `x = cupy.asarray(x)` so this
works on CUDA — at the cost of a pointless host->device round trip. dpnp's do
not, and raise.

Changed the two call sites to `np.linalg.norm`. **Rejected alternative:**
making the shim's `dpnp.linalg.*` coerce host arrays like CuPy does. It fixes
the same two tests, but it silently licenses hidden H2D transfers at all 247
`cupy.linalg.*` call sites in the library — against the whole point of the
cupy/dpnp layer. Worth reporting upstream as a cleanup: the norm is host data
on both backends and should never have gone through the GPU namespace.

### 17. `gpu4pyscf/cupy/__init__.py` — `bool()` on a size-1 array of ndim > 0
NumPy (and therefore CuPy) truth-test any array of size 1 regardless of ndim:
`bool(np.array([[5.0]]))` is `True`. dpnp accepts only 0-d and otherwise raises
`TypeError: only 0-dimensional arrays can be converted to Python scalars`.
`tdscf/math_helper.py:407` depends on the NumPy behaviour — `xy_norm` comes out
of `cp.dot(x_tmp, x_tmp.T)` with shape `(1, 1)` and is used as
`if xy_norm > 1e-14:`. `__float__`/`__int__` are deliberately left alone: NumPy 2
raises there for ndim > 0 and dpnp already matches.

**Effect:** `test_df_tdrks_ris_grad.py` 2 failed/1 passed -> **3 passed**.

### 18. `gpu4pyscf/lib/gdft/nr_eval_gto.cu` — `blockDim.x` mistranslated as `gridDim.x`
`_screen_index_legacy` (line 155) and `_screen_index` (line 83) both did
`const int blockDim_x = item.get_group_range(1);`. `get_group_range` is CUDA's
`gridDim`, not `blockDim`; the SYCL equivalent of `blockDim.x` is
`item.get_local_range(1)`.

`blockDim_x` bounds an SLM OR-reduction that decides whether a shell is
non-negligible anywhere in a grid block. With `threads = range<2>(1, 256)` and
`blocks = range<2>(nsh, ngrids/256)`, `get_local_range(1)` is 256 but
`get_group_range(1)` is `16384/256 = 64` — so the reduction started at `s = 32`
and folded only lanes 0-63. The other 192 lanes wrote their flag into SLM and it
was never OR'd in: an effective 25% subsample of every 256-point tile. Shells
significant only on the dropped lanes were silently screened out. Signature
matched exactly — the legacy result was always a strict *subset*, surviving
values agreed elementwise, only the set differed.

The `_screen_index` occurrence was latent (it multiplies `shl_block_id`, always
0 while `nbas <= 256`) but would corrupt shell indices for any molecule with
more than 256 shells.

**Effect:** reference-vs-GPU mismatches 20 -> **0**; `dft/tests/test_numint.py`
**24 passed** (includes `test_sparse_index`).

### 19. Same mistranslation swept out of `gvhf`
Grepping the tree for the pattern found two more live instances —
`lib/gvhf/g3c2e_ip1.cu:325` and `lib/gvhf/g3c2e_ip2.cu:320` — both directly
above a CUDA `#else` branch reading `const int blockDim_x = blockDim.x;`, which
confirms the intent. Fixed and rebuilt; `test_df_uhf.py` + `test_df_jk.py` +
`scf/tests/test_scf_jk.py` = **25 passed**, no regression. No occurrences remain
(`grep -rn "blockDim[_a-z]*\s*=\s*item.get_group_range" gpu4pyscf/lib` is
empty), and the reverse error (`gridDim` from `get_local_range`) does not occur.

### 20. `gpu4pyscf/df/grad/tdrhf.py` — oneMKL GEMM reduction order depends on the batch size
`test_df_tdrhf_grad.py::test_jk_energy_per_atom_dm_pairs` asserts that stacking
4 DM pairs gives the same answer as running them one at a time, to 1e-12. It was
off by 5.13e-12.

`_jk_energies_by_dm_factors` built the auxiliary vectors with a single
`cp.einsum('pqr,nqp->nr', j3c, dm)` whose **m dimension is `n_dm`**. oneMKL on
PVC picks a different k-splitting depending on m, so each DM's auxiliary vector
depended on how many DMs shared the call. cuBLAS does not, for m within one
tile — hence CUDA passes.

Measured: every other intermediate (`batch_size`, `aux_sorting`, `aux_coeff`,
`metric`, all `j3c_o1o2[i]`) was bitwise identical between `n_dm=3` and `n_dm=6`;
only the auxvecs differed, by ~1.5e-15 relative, which the DF metric solve then
amplified ~230x. Against a `longdouble` reference the m=6 kernel is the less
accurate one (4.6e-14 vs 1.3e-14), so this is a real reduction-order effect, not
noise. A per-DM loop is *not* a valid fix — oneMKL's m=1 path is
non-deterministic run to run (1.4e-14 across repeats).

**Fix:** zero-pad the DM batch to a multiple of `AUXVEC_DM_CHUNK = 4` and
contract in fixed-size chunks, so m is always exactly 4. Bitwise identical for
every n from 1 to 12. Discrepancy 5.13e-12 -> **7.4e-15**, at the level of the
run-to-run noise of the `ejk` atomic accumulation itself (measured 5.9e-15).

Cost: measured, none. `test_df_tdrhf_grad.py` runs in 26.3 s with chunking vs
40.3 s without. Worth reporting to oneMKL as batch-size-dependent reduction
order.

### 21. `gpu4pyscf/cupy/cuda.py` — dpctl waits for SYCL events inside a finalizer (Layer 6)
Second face of **intel/llvm#22943**, and the cause of the remaining sporadic
hangs (`dft/tests/test_numint.py` wedged during *collection*;
`scf/tests/test_uhf.py::test_uhf_d3bj`;
`df/tests/test_df_rhf_grad.py::test_uhf_jk_energy_per_atom`).

dpctl keeps `_SequentialOrderManager` instances in a **thread-local** map
(`SyclQueueToOrderManagerMap._get_map`), and the manager's `__del__` runs

```python
SyclEvent.wait_for(_local.get_submitted_events())
SyclEvent.wait_for(_local.get_host_task_events())
```

So when *any* worker thread exits, its thread-local dict is torn down and a
blocking SYCL event wait executes from inside a garbage-collection finalizer.
That wait enters `Scheduler::GraphProcessor::waitForEvent`, which blocks while
holding the graph read lock; a host task in flight can then never be enqueued.

Captured with gdb on a hung `test_numint.py` (this is the whole cycle):

```
Thread 9 : slot_tp_finalize -> SyclEvent.wait_for -> DPCTLEvent_Wait
           -> Scheduler::waitForEvent -> enqueueCommand(BLOCKING)
           -> event_impl::waitInternal            [holds GraphReadLock]
Thread 3 : DispatchHostTask::waitForEvents -> urEventWait
Thread 1 : blocked on a Python lock held by thread 9
```

Note the main thread had already released the GIL — so unlike fixes 1 and 11
this is *not* a GIL problem, it is purely the graph-lock cycle from #22943,
triggered by a finalizer on a thread nobody chose.

**Fix:** pin every `_SequentialOrderManager` with a process-lifetime strong
reference, so `__del__` never runs before interpreter shutdown — where dpctl's
own `sys.is_finalizing()` guard already short-circuits the waits. Nothing else
changes; the managers stay functional and keep ordering work exactly as before.
Cost is a few small objects per thread. No transfer, no added synchronization —
it *removes* one.

Worth reporting to dpctl independently of the SYCL runtime bug: doing a blocking
event wait in `__del__` is hazardous regardless, because a finalizer can run on
any thread at any allocation point.

### 22. `dpnp_array.__setitem__` drain — tried, falsified, reverted
After Layers 5 and 6, four tests were still wedging:
`df/tests/test_df_int3c2e.py::test_int3c2e_rsh`,
`df/tests/test_df_rhf_grad.py::test_uhf_jk_energy_per_atom` (omega=0.15),
`scf/tests/test_uhf.py::test_get_k`, and
`dft/tests/test_ucdft.py::test_canonical_mo_energy`.

gdb showed `usm_ndarray.__setitem__` blocking on `event_impl::wait` inside
`dpnp/tensor/_tensor_impl`, so on the hypothesis that this was a third entry
point into intel/llvm#22943, `dpnp_array.__setitem__` was routed through the
same `_drain_then` wrapper.

**That hypothesis was wrong and the change is reverted.** All four still hung.
Re-attaching gdb showed the block had simply moved into the drain itself:

```
Thread 1 : SyclQueue.wait() -> queue_impl::wait -> event_impl::wait
           -> Scheduler::waitForEvent -> event_impl::waitInternal
                                         ^ NOT via enqueueCommand -- this is
                                           the wait *after* the graph lock is
                                           released
Thread N : DispatchHostTask::waitForEvents -> urEventWait   (x2)
```

The host tasks are waiting on **device** events, and the host wait is past the
graph-lock release, so nothing here is a host-side lock cycle: a GPU kernel is
not completing. Draining earlier cannot help, and the wrapper cost a measured
7-10% per `__setitem__`, so it was removed rather than left in on a falsified
rationale.

See the open item below.

### 24. Two tests marked `xfail` — ExchCXX vs libxc, root cause documented
`dft/tests/test_libxc.py::test_u_GGA` and
`dft/tests/test_numint2c.py::test_mcol_mgga_vxc_mat` are now
`@pytest.mark.xfail` with the full reason inline. Both were traced to the XC
backend rather than to the port:

- **cutoff convention.** libxc zeroes a functional below a per-functional
  density threshold; ExchCXX keeps evaluating. At rho=1.96e-15 libxc gives
  `exc=0.0`, ExchCXX gives `-9.24398e-06` — the *analytically correct* Slater
  value. On smooth densities the two are bit-identical (LDA_X gpu/cpu ratio
  1.000000000000, spread 3.3e-16). Thresholds differ per functional and in both
  directions (ExchCXX is stricter than libxc on LDA_C_VWN).
- **`_mcol_mgga_vxc_mat` is correct.** Feeding the CPU's `vxc` through the GPU
  builder reproduces the reference to 1.8e-15; the entire 6.6e-14 gap is
  `max|vxc_gpu - vxc_cpu| = 2.4e-13` from ExchCXX's TPSS.

Verified the marks take effect on these `unittest.TestCase` classes:
`7 passed, 2 xfailed`, exit 0. `xfail` rather than `skip` so the tests keep
running and will report XPASS once ExchCXX adopts libxc's `dens_threshold`
semantics. Details and a standalone reproducer in `exchcxx_vs_libxc_repro.py`.

## Methodology note: parallel sweeps manufacture false hangs

Running the suite N-way parallel (one pytest process per GPU tile) produced
timeouts that do not exist when the same file runs alone. intel/llvm#22943 is a
*probabilistic* deadlock and concurrent load widens its race window enormously.

| file | 6-way parallel | run alone |
|---|---|---|
| `df/test_df_int3c2e.py` | timeout @3000s | 2 failed, 8 passed in **6.68s** |
| `df/test_df_rhf_grad.py` | timeout @3000s | **5 passed** in 12.57s |
| `df/test_df_tdrhf_grad.py` | timeout @3000s | 1 failed, 5 passed in **30.44s** |
| `df/test_df_rks_grad.py` | timeout @3000s | 2 failed, 9 passed in 381.8s |

A 400x gap is not contention slowness. `test_df_rhf_grad.py` is the sharpest
case: its `test_uhf_jk_energy_per_atom` was independently reported as a hang by
a subagent *and* timed out twice under load, yet the whole file passes cleanly
sequentially.

**Consequence for anyone reading earlier rounds of this log: treat any
parallel-only timeout as unproven.** Only sequential runs are evidence. The
mitigations in fixes 1, 11, 21 and 22 remove the deadlock at the entry points
where it was actually caught with gdb; they cannot remove it everywhere,
because the defect is in the SYCL runtime and the workaround the issue
recommends (an out-of-order queue) is unavailable to this port.

### 25. `contract_int3c2e_auxvec` was never ported
`df/tests/test_df_int3c2e.py::test_contract_int3c2e` and
`::test_contract_int3c2e_irregular_angular_momemtum` fail with
`AttributeError: libgvhf_md.so: undefined symbol: contract_int3c2e_auxvec`.

This branch has the test (2 definitions) and the Python ctypes binding
(`df/j_engine_3c2e.py`, 5 references) but no native implementation;
`origin/master` has one. The sibling `contract_int3c2e_dm` in the same file
*was* ported, so this is an omission rather than a design problem — and the
conversion recipe is the adjacent function. Being ported now.

### 26. Two more tests marked `xfail` — cuRAND-specific fingerprints
`df/tests/test_df_tdrhf_grad.py::test_jk_energy_per_atom` and
`df/tests/test_df_tdrhf_nac.py::test_get_nacv_ee`. Both build their inputs from
`cp.random.seed(...)`/`cp.random.rand(...)`, and on the SYCL build
`cupy.random` is `dpnp.random` (oneMKL), not cuRAND XORWOW — so the inputs
differ and the hard-coded fingerprints are unreachable.

The two are NOT equally established, and the markers say so:
- `test_jk_energy_per_atom`: GPU result **verified correct** for the DM it
  receives, against an independent pure-CPU pyscf/numpy finite-difference
  reference (CPU FD fp 17.190284408864635 vs GPU 17.190357036853285,
  max diff 5.795e-05, consistent with O(disp^2) FD error).
- `test_get_nacv_ee`: cause established, but the GPU result has **not** been
  independently validated. Flagged in the marker so nobody regenerates the
  reference on an unchecked value.

Verified: `test_df_tdrhf_grad.py` + `test_df_tdrhf_nac.py` -> 10 passed,
2 xfailed, exit 0.

### 27. `np.abs()` on a device array — the *ufunc* protocol, not the function protocol
`df/tests/test_df_hessian.py::test_unstable_j2c`:

```
assert np.max(np.abs(test_hessian_round1 - test_hessian_round2)) < 2e-7
E   TypeError: operand 'dpnp_array' does not support ufuncs (__array_ufunc__=None)
```

Distinct from fix 14. `dpnp_array` sets `__array_ufunc__ = None`, deliberately
opting out of NumPy's *ufunc* protocol, while CuPy implements it — so
`np.abs(device)` works on CUDA and raises here. Measured:

```
dpnp __array_ufunc__ = None
  np.abs(dev)    -> TypeError
  abs(dev)       -> dpnp_array     (builtin, uses __abs__)
  np.max(dev)    -> dpnp_array     (works via the __array_function__ added in fix 14)
  dev.max()      -> dpnp_array
```

Note `np.max` already works because fix 14 added `__array_function__`; only the
ufunc call fails. Fixed at the call site with the gpu4pyscf#810 idiom — stay in
the array's own namespace, pull one scalar at the end:

```python
assert abs(test_hessian_round1 - test_hessian_round2).max().item() < 2e-7
```

Deliberately did *not* implement `__array_ufunc__` in the shim: dpnp set it to
`None` on purpose, and overriding it would change behaviour across every NumPy
ufunc call in the tree.

**Effect:** `test_unstable_j2c` -> **1 passed**.

## Sequential rerun of every parallel-only timeout — all six clear

| file | 6-way parallel | sequential |
|---|---|---|
| `df/test_df_int3c2e.py` | timeout @3000s | 2 failed, 8 passed, 6.68s -> **10 passed** after fix 25 |
| `df/test_df_rhf_grad.py` | timeout @3000s | **5 passed**, 12.57s |
| `df/test_df_tdrhf_grad.py` | timeout @3000s | 1 failed, 5 passed, 30.44s (the 1 now xfail, fix 26) |
| `df/test_df_rks_grad.py` | timeout @3000s | 2 failed, 9 passed, 381.8s (both xc-fun) |
| `df/test_df_tddft_ris_nac.py` | timeout @3000s | **8 passed**, 103.7s |
| `df/test_df_hessian.py` | SIGABRT, then timeout | 3 failed, 18 passed, 43m31s (2 xc-fun + 1 now fixed by 27) |

No SIGABRT anywhere — fix 23 (scratch surface) holds across the full 21-test
hessian file.

## Out of scope / needs a maintainer decision

### OPEN (a): range-separated tests hang — suspected non-terminating kernel
Deterministic, and every one of them exercises an `omega != 0` path while its
non-RSH sibling passes:

- `df/tests/test_df_int3c2e.py::test_int3c2e_rsh` (omega=0.33)
- `df/tests/test_df_rks_grad.py::test_grad_rsh`
- `df/tests/test_df_rhf_grad.py::test_uhf_jk_energy_per_atom` (omega=0.15)
- `df/tests/test_df_tddft_ris_nac.py::test_nac_camb3lyp_tdaris_singlet_vs_ref_ge`
- `df/tests/test_df_tdrks_nac.py::test_nac_camb3lyp_tda_singlet_ge_vs_direct`

`test_int3c2e_rsh` reproduces **standalone on an idle tile** — 2 carbon atoms,
cc-pVDZ, killed at 900 s with no output — so this is neither contention nor
scale.

Evidence points at a kernel that never completes rather than the #22943 lock
cycle: the host wait sits past the graph-lock release and the host tasks are
blocked in `urEventWait` on *device* events. The most likely cause of a hung
SYCL kernel ported from CUDA is a `__syncthreads()` reached by only part of the
work-group — a non-uniform barrier is UB in SYCL and on Level Zero hangs the
group forever. Note four `item.get_group_range()`-for-`blockDim.x`
mistranslations have already been found in this repo (fixes 18 and 19); a loop
bound or barrier count from a fifth would do exactly this. Under investigation.

### OPEN (b): two nondeterministic hangs — probably still #22943
- `scf/tests/test_uhf.py` — wedges at a **different test each run**
  (`test_uhf_d3bj` in one sweep, `test_get_k` in another; `test_get_k` has no
  omega). A moving hang point means a race, not a deterministic kernel bug.
- `dft/tests/test_ucdft.py::test_canonical_mo_energy` — plain b3lyp, no omega.

Also worth noting: `df/tests/test_df_ucdft_grad.py` **passes** but takes
49 min 51 s, which smells like the same race nearly-but-not-quite wedging.

### The principled fix for the #22943 class — attempted, currently blocked
`~/gpu4pyscf-testing/llvm-fix` is checked out at exactly the HOWTO's base
commit `98748c488865f760413b6899ef034843a19196a9` and already carries the
one-line change:

```diff
 sycl/source/detail/scheduler/graph_processor.cpp
-      enqueueCommand(Cmd, GraphReadLock, Res, ToCleanUp, Cmd, BLOCKING);
+      enqueueCommand(Cmd, GraphReadLock, Res, ToCleanUp, Cmd, NON_BLOCKING);
```

Building only the runtime is the right move — it would let Layers 5 and 6 be
deleted instead of hand-maintained, and it cleanly separates the two open hang
groups (it should fix group (b) and leave group (a) untouched if that really is
a non-terminating kernel).

Two findings from attempting it:

1. `ninja` is not on `PATH` on this node; it lives in the venv at
   `mygpu4pyscf_pip_aurora/bin/ninja`. Without it `ninja -t targets` returns
   nothing and the tree looks broken when it is not. The target resolves:
   `libsycl.so -> lib/libsycl.so.9.0.0-0`.

2. **The build fails at 27/221** on a Level Zero header mismatch:

   ```
   unified-runtime/source/adapters/level_zero/common/device.hpp:292
     error: 'ze_intel_xe_device_exp_properties_t' was not declared in this scope
   ```

   That type is referenced by the unified-runtime sources but defined by no
   header available here — not `/usr/include/level_zero`, not the fetched
   `_deps/level-zero-loader-src` (the tree pins `UR_LEVEL_ZERO_LOADER_TAG
   v1.32.0`). So the checkout needs a newer level-zero-loader than the one its
   own CMake pins, or a newer `ze_intel_gpu.h` from the compute-runtime.

   Resolving that means bumping the loader tag and re-fetching, which is a
   larger detour than it looked. Not pursued further; the RSH kernel
   investigation is the higher-value target.
