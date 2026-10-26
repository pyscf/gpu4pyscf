#!/bin/bash
# Runs on a JLSE chiatta compute node, inside a GitLab CI job on the Jacamar
# `batch` runner (the runner has already obtained the node via the scheduler).
# Creates/refreshes the conda toolchain, builds gpu4pyscf's SYCL backend and
# runs pytest on the GPU.
#
# Optional environment (set as GitLab CI variables):
#   CI_ROOT      shared dir for env + caches      (default $HOME/gitlab-sycl-ci)
#   SYCL_AOT     ON | OFF | auto                  (default auto: PVC -> ON)
#   ZE_DRIVER_DIR dir with libze_intel_gpu (default: /soft intel-gpu-umd 821.36)
#   OCLOC_DIR    only if ocloc is not found in the conda env (AOT only)
#   TESTS        space-separated pytest paths     (default gpu4pyscf/scf)
#   PYTEST_WORKERS                                (default 4)
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CI_ROOT="${CI_ROOT:-$HOME/gitlab-sycl-ci}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/ci-logs}"
TESTS="${TESTS:-gpu4pyscf/scf}"
ENV_PREFIX="$CI_ROOT/env"
mkdir -p "$CI_ROOT" "$LOG_DIR"

# --- toolchain env (re-solved weekly or when env.yml changes) ---------------
MM="$CI_ROOT/bin/micromamba"
if [ ! -x "$MM" ]; then
  # micro.mamba.pm returns 503 through the JLSE proxy; the GitHub release works.
  mkdir -p "$CI_ROOT/bin"
  curl -fsSL -o "$MM" https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 &&
    chmod +x "$MM" || exit 1
fi
STAMP="$(sha256sum "$REPO_DIR/.github/jlse/env.yml" | cut -c1-12)-$(date +%G-%V)"
if [ "$(cat "$CI_ROOT/env.stamp" 2>/dev/null)" != "$STAMP" ]; then
  # Serialize concurrent jobs creating the env.
  ( flock 9
    if [ "$(cat "$CI_ROOT/env.stamp" 2>/dev/null)" != "$STAMP" ]; then
      "$MM" create -y -p "$ENV_PREFIX" -f "$REPO_DIR/.github/jlse/env.yml" &&
        echo "$STAMP" > "$CI_ROOT/env.stamp"
    fi
  ) 9>"$CI_ROOT/env.lock" || exit 1
fi

export PATH="$ENV_PREFIX/bin:$PATH"
# conda-forge's intel-compute-runtime ships only a versioned ocloc-<ver>;
# icpx looks for plain `ocloc`.
if [ ! -e "$ENV_PREFIX/bin/ocloc" ]; then
  for f in "$ENV_PREFIX"/bin/ocloc-*; do [ -x "$f" ] && ln -sf "$(basename "$f")" "$ENV_PREFIX/bin/ocloc" && break; done
fi
[ -n "${OCLOC_DIR:-}" ] && ! command -v ocloc >/dev/null && export PATH="$OCLOC_DIR:$PATH"
# Level-zero: the loader comes from the conda env (level-zero), the GPU driver
# (libze_intel_gpu) from the node's system install; conda has no package for it.
# No environment modules: just put its lib dir on the library path.
ZE_DRIVER_DIR="${ZE_DRIVER_DIR:-/soft/libraries/intel-gpu-umd/821.36/lib64}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:$ZE_DRIVER_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MKLROOT="$ENV_PREFIX" TBBROOT="$ENV_PREFIX"
export CC=icx CXX=icpx FC=ifx
export CCACHE_DIR="$CI_ROOT/ccache" CCACHE_MAXSIZE=10G
export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR="$CI_ROOT/sycl_cache"
# chiatta nodes have 8 two-tile Max 1550 cards; the tests use one device.
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}"
export OMP_NUM_THREADS=2
export PYTHONPATH="$REPO_DIR:$REPO_DIR/gpu4pyscf"

cd "$REPO_DIR"
{
  hostname; date
  icpx --version | head -1
  python -c "import dpctl, dpnp; print('dpctl', dpctl.__version__, '| dpnp', dpnp.__version__); dpctl.lsplatform()"
  python -c "import dpctl; d = dpctl.get_devices(device_type='gpu'); print(d); assert d, 'no SYCL GPU device'"
} 2>&1 | tee "$LOG_DIR/10_toolchain.log"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "toolchain check failed; diagnostics:"
  ls -l "$ZE_DRIVER_DIR"/libze_* 2>&1 | head
  ldd "$ZE_DRIVER_DIR/libze_intel_gpu.so.1" 2>&1 | grep -i "not found"
  ls /dev/dri 2>&1 | head -3
  SYCL_UR_TRACE=-1 ZE_ENABLE_LOADER_DEBUG_TRACE=1 \
    python -c "import dpctl; print(dpctl.get_devices())" 2>&1 | head -40
  sycl-ls 2>&1 | head
  exit 1
fi

# --- AOT only makes sense on PVC; elsewhere use generic spir64 JIT ----------
AOT="${SYCL_AOT:-auto}"
if [ "$AOT" = auto ]; then
  if python -c "import dpctl,sys; sys.exit(0 if any('Max' in d.name or 'Ponte' in d.name for d in dpctl.get_devices(device_type='gpu')) else 1)"; then
    AOT=ON
  else
    AOT=OFF
  fi
fi
echo "GPU4PYSCF_SYCL_AOT=$AOT" | tee "$LOG_DIR/20_aot.log"

export CMAKE_CONFIGURE_ARGS="-DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx \
  -DCMAKE_Fortran_COMPILER=ifx -DUSE_SYCL=ON -DCMAKE_BUILD_TYPE=Release \
  -DGPU4PYSCF_SYCL_AOT=$AOT \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"

python setup.py --sycl build 2>&1 | tee "$LOG_DIR/40_build.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "build failed"; exit 1; }
ccache -s || true

python -c "import gpu4pyscf; print('gpu4pyscf import OK')" 2>&1 | tee "$LOG_DIR/50_smoke.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || exit 1

# shellcheck disable=SC2086
pytest -n "${PYTEST_WORKERS:-4}" --dist loadfile -m 'not slow and not benchmark and not special' \
  --durations=50 $TESTS 2>&1 | tee "$LOG_DIR/60_pytest.log"
exit "${PIPESTATUS[0]}"
