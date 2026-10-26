#!/bin/bash
# SYCL CI on a JLSE chiatta node (Intel PVC), run inside a GitLab CI job on the
# Jacamar `batch` runner. Single GPU (level_zero:0).
#
#   ci_sycl.sh build              build the SYCL backend, publish the built tree
#                                 to $CI_ROOT/builds/$CI_COMMIT_SHA
#   ci_sycl.sh test <path>...     run pytest on that built tree
#
# Optional environment (GitLab CI/CD variables):
#   CI_ROOT        shared dir for env, caches, builds   (default $HOME/gitlab-sycl-ci)
#   SYCL_AOT       ON | OFF | auto                      (default auto: PVC -> ON)
#   ZE_DRIVER_DIR  dir with libze_intel_gpu             (default /soft intel-gpu-umd 821.36)
#   PYTEST_WORKERS                                      (default 4)
set -uo pipefail

MODE="${1:-}"; shift || true
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CI_ROOT="${CI_ROOT:-$HOME/gitlab-sycl-ci}"
LOG_DIR="${LOG_DIR:-$SRC_DIR/ci-logs}"
SHA="${CI_COMMIT_SHA:-local}"
BUILD_DIR="$CI_ROOT/builds/$SHA"
ENV_PREFIX="$CI_ROOT/env"
mkdir -p "$CI_ROOT" "$LOG_DIR"

setup_env() {
  local MM="$CI_ROOT/bin/micromamba" stamp
  if [ ! -x "$MM" ]; then
    # micro.mamba.pm returns 503 through the JLSE proxy; the GitHub release works.
    mkdir -p "$CI_ROOT/bin"
    curl -fsSL -o "$MM" https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 &&
      chmod +x "$MM" || return 1
  fi
  # Re-solved weekly or when env.yml changes; jobs running concurrently wait on the lock.
  stamp="$(sha256sum "$SRC_DIR/.github/jlse/env.yml" | cut -c1-12)-$(date +%G-%V)"
  if [ "$(cat "$CI_ROOT/env.stamp" 2>/dev/null)" != "$stamp" ]; then
    ( flock 9
      if [ "$(cat "$CI_ROOT/env.stamp" 2>/dev/null)" != "$stamp" ]; then
        "$MM" create -y -p "$ENV_PREFIX" -f "$SRC_DIR/.github/jlse/env.yml" &&
          echo "$stamp" > "$CI_ROOT/env.stamp"
      fi
    ) 9>"$CI_ROOT/env.lock" || return 1
  fi

  export PATH="$ENV_PREFIX/bin:$PATH"
  # conda-forge's intel-compute-runtime ships only a versioned ocloc-<ver>;
  # icpx looks for plain `ocloc`.
  if [ ! -e "$ENV_PREFIX/bin/ocloc" ]; then
    for f in "$ENV_PREFIX"/bin/ocloc-*; do
      [ -x "$f" ] && ln -sf "$(basename "$f")" "$ENV_PREFIX/bin/ocloc" && break
    done
  fi
  # Level-zero: loader from the conda env, GPU driver (libze_intel_gpu) from the
  # node's system install (no conda package exists). No environment modules.
  ZE_DRIVER_DIR="${ZE_DRIVER_DIR:-/soft/libraries/intel-gpu-umd/821.36/lib64}"
  export LD_LIBRARY_PATH="$ENV_PREFIX/lib:$ZE_DRIVER_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export MKLROOT="$ENV_PREFIX" TBBROOT="$ENV_PREFIX"
  export CC=icx CXX=icpx FC=ifx
  export CCACHE_DIR="$CI_ROOT/ccache" CCACHE_MAXSIZE=10G
  export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR="$CI_ROOT/sycl_cache"
  # chiatta nodes have 8 two-tile Max 1550 cards; single-GPU CI uses device 0.
  export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}"
  export OMP_NUM_THREADS=2
}

toolchain_check() {
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
    return 1
  fi
}

do_build() {
  cd "$SRC_DIR"
  export PYTHONPATH="$SRC_DIR:$SRC_DIR/gpu4pyscf"
  toolchain_check || exit 1

  # AOT only makes sense on PVC; elsewhere use generic spir64 JIT.
  local AOT="${SYCL_AOT:-auto}"
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

  # Publish the built tree for the test jobs of this pipeline (shared filesystem).
  mkdir -p "$BUILD_DIR"
  rsync -a --delete --exclude=.git --exclude=build --exclude=ci-logs "$SRC_DIR"/ "$BUILD_DIR"/ || exit 1
  touch "$BUILD_DIR/.build_ok"
  find "$CI_ROOT/builds" -mindepth 1 -maxdepth 1 -type d -mtime +7 -exec rm -rf {} + 2>/dev/null
  echo "published build to $BUILD_DIR"
}

do_test() {
  [ $# -gt 0 ] || { echo "usage: $0 test <pytest paths...>"; exit 2; }
  [ -e "$BUILD_DIR/.build_ok" ] || { echo "no build for $SHA at $BUILD_DIR (did the build job run?)"; exit 1; }
  cd "$BUILD_DIR"
  export PYTHONPATH="$BUILD_DIR:$BUILD_DIR/gpu4pyscf"
  toolchain_check || exit 1
  # Same pytest invocation as upstream unittest.yml.
  pytest -n "${PYTEST_WORKERS:-4}" --dist loadfile -m 'not slow and not benchmark and not special' \
    --durations=50 "$@" 2>&1 | tee "$LOG_DIR/60_pytest.log"
  exit "${PIPESTATUS[0]}"
}

setup_env || { echo "environment setup failed"; exit 1; }
case "$MODE" in
  build) do_build ;;
  test)  do_test "$@" ;;
  *) echo "usage: $0 build | test <paths...>"; exit 2 ;;
esac
