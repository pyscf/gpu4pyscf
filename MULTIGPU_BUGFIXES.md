# Multi-GPU Bug Fixes for TDA-NACV on ≥3 GPUs

**System:** 4× NVIDIA A100-SXM4-40GB, CUDA 11.8, CuPy 13.4.1 + cuTENSOR  
**Test case:** 60-atom C/H/N/O molecule, def2-tzvp basis (~1042 AOs), PBE0/TDA-RIS, NACV  
**Symptom:** `cudaErrorLaunchFailure` on device 1 (or higher) during TDA's first `get_jk`
call when running with 3+ GPUs; SCF converges normally.

---

## Bug 1 — P2P race condition in `array_reduce` tree-reduce

### File
`gpu4pyscf/lib/multi_gpu.py`

### Root cause
`p2p_transfer(buf, src)` copies data from `src` (on device `device_id+step`) into `buf`
(on device `device_id`) using a peer-to-peer DMA. The DMA is asynchronous and can enqueue
work on **both** devices' CUDA streams simultaneously. The original code immediately
accumulated the result with `dst[p0:p1] += p2p_transfer(...)` — a fused form that started
the accumulation kernel on device `device_id` while device `device_id+step` was still
writing to `buf` via the DMA. This is a classic producer-consumer race.

The race did not manifest in 2-GPU SCF because the arrays are small (single density
matrix, (nao, nao) shape). In TDA the Davidson subspace grows each iteration, making
arrays larger (n_states × nao × nao), which widens the DMA window enough for the race
to fire — typically on the 3rd or later Davidson `get_jk` call.

### Fix
Separate `p2p_transfer` from `+=` and synchronize **both** the destination and source
devices before and after the accumulation:

```python
# Before (buggy):
dst[p0:p1] += p2p_transfer(buf[:p1-p0], src[p0:p1])

# After (fixed):
p2p_transfer(buf[:p1-p0], src[p0:p1])
Device(device_id).synchronize()        # wait for dst stream to be idle
Device(device_id+step).synchronize()   # wait for DMA push on src device to finish
dst[p0:p1] += buf[:p1-p0]
Device(device_id).synchronize()        # catch any async error from the accumulation
```

Additional defensive syncs were also added:
- `synchronize()` after `ThreadPoolExecutor.result()` in `run()` (captures deferred errors
  from all worker threads before the main thread proceeds)
- Per-device sync before and after the ravel phase in `array_reduce` (guards against
  in-flight ops when switching from nd-shape to 1D view)

---

## Bug 2 — Cross-device `cp.asarray()` race in worker-thread `proc` functions

### Files
`gpu4pyscf/scf/jk.py`, `gpu4pyscf/scf/j_engine.py`

### Root cause
Inside `multi_gpu.run`'s worker threads, `proc` copies the density-matrix arguments
from device 0 to the local device using `cp.asarray(dms)` and `cp.asarray(dm_cond)`.
CuPy's cross-device array copy can enqueue the P2P DMA on an **internal CuPy transfer
stream** that is distinct from the calling thread's per-thread CUDA stream. Subsequent
GPU operations launched on the per-thread stream (e.g. `cp.vstack`, boolean masking in
`_make_tril_pair_mappings`) therefore do not wait for the copy to land, causing those
kernels to read partially-written memory.

With 2 GPUs only one copy flies at a time and the window is too small to observe. With
4 GPUs, three copies (to devices 1, 2, 3) overlap and the race opens reliably — the
very first TDA `get_jk` call fails.

**Diagnostic signature:** `copy_to_host_async` raised inside
`_make_tril_pair_mappings` at the boolean-indexing step (which does a D2H sync to count
True elements), even though a proc-start `Device(device_id).synchronize()` showed the
device was completely clean when `proc` entered. The failure is fresh, not stale.

### Fix
One `Device(device_id).synchronize()` call from the **worker thread**, immediately after
all cross-device `cp.asarray()` calls and before the first GPU kernel that reads the
copied data:

```python
dms = cp.asarray(dms)          # D2D copy: device 0 → device i, async on internal stream
dm_cond = cp.asarray(dm_cond)  # same
if num_devices > 1:
    cp.cuda.Device(device_id).synchronize()  # flush all streams on device i
# Now safe to read dms / dm_cond in subsequent kernels
if hermi == 0:
    dms = cp.vstack([dms, dms.transpose(0,2,1)])
```

`cudaDeviceSynchronize()` (what `Device.synchronize()` calls) waits for **all** in-flight
operations on the device — including DMA transfers writing to that device initiated from
any stream or any thread — making it a sufficient barrier here.

---

## Verified timings (60-atom molecule, def2-tzvp, PBE0)

| Stage | 2× A100 | 4× A100 | Speedup |
|---|---|---|---|
| SCF | ~49 s | ~48 s | 1.02× |
| TDA (5 states) | ~423 s | ~221 s | 1.91× |
| NAC (states 1,2) | — | ~119 s | — |
| TDA-RIS | — | ~3 s | — |
| NAC-RIS | — | ~88 s | — |
| NAC-RIS (ris_zvector) | — | ~32 s | — |

TDA excitation energies agree between 2-GPU and 4-GPU runs:
4.11, 4.28, 4.42, 4.62, 4.71 eV.

---

## Summary of changed lines

| File | Change |
|---|---|
| `gpu4pyscf/lib/multi_gpu.py` | `run()`: collect futures into list, call `synchronize()` before return |
| `gpu4pyscf/lib/multi_gpu.py` | `array_reduce()`: pre- and post-ravel `Device(i).synchronize()` for all devices |
| `gpu4pyscf/lib/multi_gpu.py` | `array_reduce()` tree-reduce: split `p2p_transfer` from `+=`, add 3 syncs per block |
| `gpu4pyscf/scf/jk.py` | `VHFOpt.get_jk` proc: `Device(device_id).synchronize()` after cross-device `cp.asarray` calls |
| `gpu4pyscf/scf/j_engine.py` | `MDJOpt.get_j` proc: `Device(device_id).synchronize()` after cross-device `asarray` calls |
