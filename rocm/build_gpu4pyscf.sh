#!/usr/bin/env bash
#
# Reproducible build of the gpu4pyscf native libraries with AMD ROCm/HIP
# (gpu4pyscf AMD port, Phase 1). Configures the CMake project with -DUSE_HIP=ON
# and builds every libXXX.so into gpu4pyscf/lib/.
#
# The AMD GPU architecture is auto-detected via rocm_agent_enumerator; override
# with GPU_ARCHITECTURES=gfxXXXX if needed. CUTLASS (grouped GEMM) and the CUDA
# libxc fork are disabled on HIP for now (Phase 4).
#
# Usage:
#   source .venv/bin/activate
#   ROCM_PATH=/path/to/rocm ./rocm/build_gpu4pyscf.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB_SRC="$REPO_ROOT/gpu4pyscf/lib"
BUILD_DIR="${BUILD_DIR:-/tmp/gpu4pyscf_hip_build}"
JOBS="${JOBS:-$(nproc)}"

# --- locate ROCm ---
if [[ -z "${ROCM_PATH:-}" ]]; then
    if command -v hipconfig >/dev/null 2>&1; then
        ROCM_PATH="$(hipconfig -p)"
    else
        echo "ERROR: ROCM_PATH not set and hipconfig not found on PATH." >&2
        exit 1
    fi
fi
export ROCM_PATH ROCM_HOME="$ROCM_PATH" HIP_PATH="$ROCM_PATH" HIP_PLATFORM="amd"
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH:-}"

CMAKE_HIP_COMPILER="${CMAKE_HIP_COMPILER:-$ROCM_PATH/llvm/bin/clang++}"

echo ">>> ROCm:            $ROCM_PATH"
echo ">>> HIP compiler:    $CMAKE_HIP_COMPILER"
echo ">>> Build dir:       $BUILD_DIR"
echo ">>> Jobs:            $JOBS"
echo ">>> GPU arch:        ${GPU_ARCHITECTURES:-<auto-detect>}"

cmake -S "$LIB_SRC" -B "$BUILD_DIR" \
    -DUSE_HIP=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_COMPILER="$CMAKE_HIP_COMPILER" \
    ${GPU_ARCHITECTURES:+-DGPU_ARCHITECTURES=$GPU_ARCHITECTURES}

cmake --build "$BUILD_DIR" -j "$JOBS"

echo ">>> Built libraries:"
ls -1 "$LIB_SRC"/*.so
