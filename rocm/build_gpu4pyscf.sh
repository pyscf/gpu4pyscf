#!/usr/bin/env bash
#
# Build the gpu4pyscf native libraries for AMD ROCm/HIP.
#
# This is a thin convenience wrapper around the project's own packaging: it
# locates ROCm (so the build works from a fresh shell) and then delegates the
# actual compilation to setup.py -- the SINGLE build path, shared with
# `pip install`. Running this is equivalent to `USE_HIP=1 python setup.py
# build_py`; using setup.py here keeps the packaging path continuously exercised
# ("dogfooded") during development instead of maintaining a separate build.
#
# The AMD GPU architecture is auto-detected (rocm_agent_enumerator); override
# with GPU_ARCHITECTURES=gfxXXXX. setup.py reports actionable errors if ROCm,
# the HIP compiler, or a GPU target cannot be found.
#
# Usage:
#   source .venv/bin/activate
#   ./rocm/build_gpu4pyscf.sh
#   GPU_ARCHITECTURES=gfx942 ./rocm/build_gpu4pyscf.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- locate ROCm (convenience only; setup.py can also auto-detect) ---
if [[ -z "${ROCM_PATH:-}" ]]; then
    if command -v hipconfig >/dev/null 2>&1; then
        ROCM_PATH="$(hipconfig -p)"
    else
        echo "ERROR: ROCM_PATH not set and hipconfig not found on PATH." >&2
        echo "       Set ROCM_PATH to your ROCm root, or activate an environment" >&2
        echo "       that puts hipconfig on PATH." >&2
        exit 1
    fi
fi
export ROCM_PATH

# Force the HIP backend so the build never silently falls back to CUDA even if a
# stray nvcc is on PATH; setup.py handles the rest of the ROCm environment.
export USE_HIP=1

# Reproduce the validated release build (setup.py otherwise uses CMake's default
# RELWITHDEBINFO). setup.py forwards CMAKE_CONFIGURE_ARGS to the CMake configure.
if [[ -n "${CMAKE_CONFIGURE_ARGS:-}" ]]; then
    export CMAKE_CONFIGURE_ARGS="$CMAKE_CONFIGURE_ARGS -DCMAKE_BUILD_TYPE=Release"
else
    export CMAKE_CONFIGURE_ARGS="-DCMAKE_BUILD_TYPE=Release"
fi

echo ">>> ROCm:      $ROCM_PATH"
echo ">>> GPU arch:  ${GPU_ARCHITECTURES:-<auto-detect>}"
echo ">>> Building via setup.py (USE_HIP=1 python setup.py build_py) ..."

cd "$REPO_ROOT"
python setup.py build_py "$@"

echo ">>> Built libraries:"
ls -1 "$REPO_ROOT"/gpu4pyscf/lib/*.so
