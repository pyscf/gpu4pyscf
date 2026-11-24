# PySCF GPU kernels coding style

Resolve routine implementation choices using nearby code and existing APIs.
Ask when an unresolved scientific convention or requirement materially affects
the result; do not silently invent units, normalization factors, or reference
values.

## Implementation

- PySCF favors compact scientific code. Keep implementations concise while
  making algorithmic steps and numerical assumptions clear.
- Follow nearby code when conventions differ, unless changing the style is
  part of the task.
- Make the smallest complete change that solves the problem. Avoid unrelated
  cleanup, renaming, or reformatting.
- Do not substantially refactor, rewrite, or reorganize existing helper
  functions unless the task clearly requires it. If substantial refactoring is
  necessary, explain why the localized approach is insufficient.

## Portability

- Use only `gpu4pyscf.BACKEND` (`"cuda"`, `"hip"`, or `"sycl"`) to select
  backend-specific Python code. Test supported backends explicitly.
- Use `threadIdx_x`, `blockIdx_x`, and related macros defined in
  [compat/gpu_compat.h](../../gpu4pyscf/lib/compat/gpu_compat.h).
  Follow its context, launch, and shared-memory conventions and nearby patterns.
- Do not assume a warp has 32 lanes. Use backend-aware warp-size definitions,
  preserve lane-mask types, and check shuffle and reduction semantics.
- When editing CMake, respect existing backend branches, architecture settings,
  compiler flags, and dependencies. Follow nearby uses of
  `gpu4pyscf_configure_gpu_target` for GPU targets.
