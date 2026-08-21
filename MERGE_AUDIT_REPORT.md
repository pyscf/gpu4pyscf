# Post-merge audit of sycl-merge-upstream vs upstream/master

Audited 2026-08-21, after repair commit `f40c276`. Method: all 62 files that
upstream changed since merge-base `1740bc2` AND that still differ from
upstream/master were three-way checked (residual delta must be pure SYCL
overlay; every upstream addition must be present; `#else` CUDA branches must
match upstream). Merge rule: upstream/master is gold standard; SYCL lands on
top as `#ifdef USE_SYCL` overlay. Also checked the inverse direction (files
where the merge took upstream wholesale over SYCL-side edits).

## Verdict

62/62 files audited. 55 fully clean. All upstream regenerations
(unrolled_* kernels, up to 7.6k lines each) inherited byte-identical.
The earlier-suspected "overlay wipes" in dft/mcfun_gpu.py, dft/numint2c.py,
gto/mole.py were false alarms — upstream itself adopted equivalent dpnp-safe
fixes. pbc/decompress.cu and gint/nr_fill_ao_int3c2e_general.cu were deleted
BY upstream; the merge correctly honored the deletions (zero references left).

## Fixed on this branch (sycl-merge-audit-fixes)

1. `1f8aac1` — pbc/rys_contract_j.cu + pbc/rys_contract_k.cu CUDA path:
   upstream shrank `__constant__ c_gxyz_offset` to 256 entries and refreshes
   it per 256-tile chunk; merged code kept the merge-base single-copy design
   (up to 625 entries into a 256-entry symbol, OOB reads at OFFSET 256/512)
   → CUDA runtime failure whenever n_tiles > 256 (g-function quartets).
   The j launcher additionally omitted the `p_gxyz_offset` kernel argument
   entirely → CUDA compile error. Restored upstream's chunked-copy scheme on
   the CUDA branch; SYCL device_global scheme untouched.
   Also: pbc/CMakeLists.txt SYCL branch set `C_SRCS sorting.c` but never
   added it to the target → sorting.c symbols missing from SYCL libpbc.
2. Cosmetics: pbc/contract_int3c2e.cu `GOUT_WIDTH` restored to upstream 29 on
   CUDA (SYCL keeps 30 behind the ifdef); gpu4pyscf/__init__.py leftover
   commented-out import removed.

Both .cu fixes pass `icpx -fsyntax-only` with the production SYCL/PVC flags.
The CUDA side CANNOT be compiled on this machine — must be verified by CUDA
CI / the GH200 box before the upstream PR.

## Known intentional CUDA-path deviations (NOT fixed — decide before upstream PR)

These are deliberate SYCL-motivated edits applied unconditionally, so the
CUDA branch is no longer byte-equal to upstream. All verified semantically
safe on CUDA, but they will show up in the upstream PR diff:

- gvhf-rys/create_tasks.cu: `__syncthreads()` before early return at ~15
  sites (the barrier-divergence deadlock fix; block-uniform condition, no-op
  on CUDA); `JKEnergy` passed by value instead of by reference in
  `_fill_ejk_tasks`/`_fill_sr_ejk_tasks`.
- gvhf-rys/rys_contract_jk_ip2.cu: extra `__syncthreads()` after the
  t_id==0 init block in both kernels (race fix, uniform barrier).
- pbc/contract_int3c2e.cu auxvec kernel: upstream's `if (ish_cell0 <
  jsh_cell0) continue;` replaced by shared `fac = 0` zeroing (barrier-
  divergence workaround). Numerically identical; wasted work on skipped
  pairs → CUDA perf regression. Consider `#ifdef USE_SYCL` split.
- Various: `<cuda.h>/<cuda_runtime.h>` includes deleted unguarded (nvcc
  force-includes cuda_runtime.h — builds fine); a few dead declarations in
  CUDA #else branches; `cudaFuncSetAttribute` called for kernels upstream
  skips (legal superset); pbc/rys_contract_jk_ip1.cu dropped an upstream
  commented-out extern line.

## Latent notes

- gvhf-rys/CMakeLists.txt CUDA branch re-appends rys_roots_dat.cu (CMake
  dedupes; harmless).
- SYCL `s_rys_gxyz_offset[625]` stays over-allocated vs upstream's 256
  (harmless).

## Branch state

`sycl-merge-audit-fixes` = `f40c276` + these fixes, in worktree
/home/abagusetty/gpu4pyscf-testing/audit_fixes. Merge into
`sycl-merge-upstream` after the running scf/df/dft sweep finishes (do not
touch that worktree while the other session's sweep is live).
