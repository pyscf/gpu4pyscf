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
# By default the AMD GPU is auto-detected (rocm_agent_enumerator). Use GPU_TARGET
# to pick what to build for; setup.py reports actionable errors if ROCm, the HIP
# compiler, or a GPU target cannot be found.
#
# Usage:
#   source .venv/bin/activate
#   ./rocm/build_gpu4pyscf.sh                    # (1) autodetect this machine's GPU
#   GPU_TARGET=gfx942 ./rocm/build_gpu4pyscf.sh  # (2) one specific GPU
#   GPU_TARGET=all  ./rocm/build_gpu4pyscf.sh    # (3) all AMD hardware (one binary)
#   GPU_TARGET=rdna ./rocm/build_gpu4pyscf.sh    # (4) all Radeon/RDNA GPUs
#   GPU_TARGET=cdna ./rocm/build_gpu4pyscf.sh    # (5) all Instinct/CDNA GPUs
# (GPU_ARCHITECTURES=gfxXXXX still works as a raw override.)
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

# --- GPU target selection ----------------------------------------------------
# GPU_TARGET is a convenience selector for what hardware to build for (an
# explicit GPU_ARCHITECTURES always overrides it):
#   native | auto  (default) build for the GPU(s) installed in this machine
#   all            all supported AMD families in ONE fat binary (CDNA + RDNA)
#   cdna           AMD Instinct datacenter GPUs  (MI100/MI200/MI300/MI350, wave64)
#   rdna           AMD Radeon RDNA2+ GPUs / APUs (RX 6000/7000/9000, wave32)
#   gfxNNNN[;...]  an explicit arch or ;-list, e.g. gfx942  or  "gfx942;gfx1100"
#
# Curated arch lists -- edit here to add hardware. CDNA lists specific chips
# (few datacenter parts, best perf); RDNA uses per-generation "generic" targets
# (one code object per family, verified to run on the member GPUs incl. APUs).
CDNA_ARCHS="gfx908;gfx90a;gfx942;gfx950"
RDNA_ARCHS="gfx10-3-generic;gfx11-generic;gfx12-generic"

GPU_TARGET="${GPU_TARGET:-native}"
if [[ -z "${GPU_ARCHITECTURES:-}" ]]; then
    case "${GPU_TARGET,,}" in
        native|auto|"") : ;;                                    # autodetect installed GPU(s)
        all)  export GPU_ARCHITECTURES="${CDNA_ARCHS};${RDNA_ARCHS}" ;;
        cdna) export GPU_ARCHITECTURES="${CDNA_ARCHS}" ;;
        rdna) export GPU_ARCHITECTURES="${RDNA_ARCHS}" ;;
        *)    export GPU_ARCHITECTURES="${GPU_TARGET}" ;;       # explicit gfx arch(es)
    esac
fi

echo ">>> GPU target: ${GPU_TARGET}  (arch: ${GPU_ARCHITECTURES:-<auto-detect>})"
echo ">>> Building via setup.py (USE_HIP=1 python setup.py build_py) ..."

cd "$REPO_ROOT"
python setup.py build_py "$@"

echo ">>> Built libraries:"
ls -1 "$REPO_ROOT"/gpu4pyscf/lib/*.so
