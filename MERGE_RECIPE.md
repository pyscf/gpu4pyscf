# Conflict-resolution recipe: merging upstream/master into the SYCL branch

## The rule
**upstream/master is GROUND TRUTH.** This branch will eventually merge *into*
upstream, so upstream owns the logic, the algorithm, the kernel signatures and
the file layout. The SYCL port is a *transformation applied on top* of that.

For every conflict hunk:
1. Start from **theirs** (upstream) — that is the correct logic.
2. Re-apply the **SYCL adaptation** to that new logic, if the hunk needs one.
3. Never keep our version merely because it is ours. Keep it only where it
   encodes something SYCL genuinely requires.

If upstream changed a kernel signature, argument order or macro, adopt the new
one and port it — do not preserve the old signature.

## Deriving the SYCL transformation
Do not guess it. For any file, the pure transformation is:

    git diff $(git merge-base sycl upstream/master) sycl -- <path>

and for a *similar file that did not conflict*, that diff is a clean,
uncontaminated example of the idiom to copy.

## Known idioms (verified in this tree)
| CUDA | SYCL |
|---|---|
| `blockIdx.x` | `blockIdx_x` (local, from `item.get_group(1)`) |
| `blockDim.x` | `item.get_local_range(1)` |
| `gridDim.x`  | `item.get_group_range(1)` |
| `__syncthreads()` | `item.barrier(...)` — **must be uniform**, never inside divergent control flow |
| `#include <cuda.h>`, `<cuda_runtime.h>` | dropped |
| `"gvhf-rys/vhf.cuh"` | `"vhf.cuh"` (SYCL build uses a different include dir) |
| `cudaFuncSetAttribute(...)` | dropped |
| `__global__` launch | `stream.parallel_for` / `MAKE_RANGE_2D` |

**`blockDim.x` vs `gridDim.x` is the single most dangerous confusion here.**
Mistranslating it has already caused two silent-wrong-answer bugs in this repo:
it corrupted an SLM reduction so it folded only 64 of 256 lanes, which still
produced *correct results for most inputs*. Get it right every time.

## Other hazards
- SYCL requires the global range to be a multiple of the local range.
- PVC max work-group size is 1024.
- Large private arrays land in the scratch surface, sized per hardware thread
  across the device — over-allocating one costs gigabytes.
- Upstream renamed `KERNEL_ARGS`/`KERNEL_SETUP()` to
  `JKMATRIX_KERNEL_ARGS`/`JKMATRIX_KERNEL_SETUP()` in places and introduced a
  local `int nsq_per_block = _nsq_per_block;`. Adopt upstream's naming.

## Process rules
- Resolve files **on disk only**. Do **NOT** run `git add`, `git commit`,
  `git checkout --ours/--theirs`, `git merge`, `git stash` or anything that
  touches the index — several agents share this worktree and the index lock
  will race. The coordinator stages everything centrally.
- Touch **only the files assigned to you**.
- When done with a file, it must contain **zero** conflict markers:
  `grep -n '^<<<<<<<\|^=======$\|^>>>>>>>' <file>` must be empty.
- Do not try to build; the coordinator builds once at the end.
