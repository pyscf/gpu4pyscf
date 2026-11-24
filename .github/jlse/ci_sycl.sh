#!/bin/bash
# SYCL CI on a GitLab `batch`-runner GPU node (Jacamar). The actual device is
# printed at the top of every job log (dpctl device list / sycl-ls --verbose).
# Single GPU per process (level_zero:0 via ZE_AFFINITY_MASK).
#
#   ci_sycl.sh build              build the SYCL backend, publish the built tree
#                                 to $CI_ROOT/builds/$CI_PIPELINE_ID
#   ci_sycl.sh test <path>...     run pytest on that built tree
#
# Optional environment (GitLab CI/CD variables):
#   CI_ROOT        shared dir for envs, caches, builds  (default $HOME/gitlab-sycl-ci)
#   SYCL_ARCHITECTURES  generic | pvc                   (default generic)
#   ZE_DRIVER_DIR  dir with libze_intel_gpu             (default /soft intel-gpu-umd 821.36)
#   PYTEST_WORKERS                                      (default 4)
#   PR_NUMBER, PR_REPO   set by the PR poller: build that GitHub PR's merge
#                        ref (or head, on conflicts) instead of this checkout.
#                        This script, env.yml and .gitlab-ci.yml always come
#                        from the trusted checkout, never from the PR.
set -uo pipefail

MODE="${1:-}"; shift || true
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRUSTED_DIR="$SRC_DIR"   # CI config; SRC_DIR becomes the PR worktree for PR builds
CI_ROOT="${CI_ROOT:-$HOME/gitlab-sycl-ci}"
LOG_DIR="${LOG_DIR:-$TRUSTED_DIR/ci-logs/$MODE}"
# Keyed by pipeline: PR pipelines all run from the same trusted commit.
BUILD_DIR="$CI_ROOT/builds/${CI_PIPELINE_ID:-local}"
# One fresh conda env per pipeline, like the GitHub CPU CI (which re-solves on
# every run and caches only downloaded packages): the nightly dpnp / newest
# dpctl are picked up automatically. Test jobs reuse the env their build used.
ENV_PREFIX="$CI_ROOT/envs/${CI_PIPELINE_ID:-local}"
mkdir -p "$CI_ROOT" "$LOG_DIR"

# Everything that mutates the shared micromamba root ($CI_ROOT/mroot) runs under
# one lock: the PR poller starts up to PR_MAX_TRIGGERS (2) pipelines at once, and
# one job's cache cleanup under another job's running download truncates repodata
# exactly like a dropped connection. /home (Lustre) is mounted -o flock.
with_mroot_lock() {
  (
    flock -w "${MROOT_LOCK_TIMEOUT:-3600}" 9 ||
      { echo "timed out after ${MROOT_LOCK_TIMEOUT:-3600}s waiting for $CI_ROOT/mroot.lock"; exit 1; }
    "$@"
  ) 9>"$CI_ROOT/mroot.lock"
}

# Fail early and clearly instead of with a half-written env / cache.
check_disk() {
  local need_gb="${1:-${CI_MIN_FREE_GB:-40}}" avail_gb
  avail_gb=$(df -Pk "$CI_ROOT" 2>/dev/null | awk 'NR==2 {print int($4/1048576)}')
  echo "free space on $CI_ROOT: ${avail_gb:-?} GB (need >= $need_gb GB)"
  command -v lfs >/dev/null 2>&1 && lfs quota -h -u "$(id -un)" "$CI_ROOT" 2>/dev/null | tail -n +2 | head -3
  if [ -n "$avail_gb" ] && [ "$avail_gb" -lt "$need_gb" ]; then
    echo "ERROR: only $avail_gb GB free on $CI_ROOT; clean up $CI_ROOT/{envs,builds,ccache,sycl_cache,mroot/pkgs}"
    return 1
  fi
}

fetch_micromamba() {
  local MM="$1" url
  "$MM" --version >/dev/null 2>&1 && return 0
  # micro.mamba.pm returns 503 through the JLSE proxy; the GitHub release works.
  url="https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64"
  [ -n "${MICROMAMBA_VERSION:-}" ] &&
    url="https://github.com/mamba-org/micromamba-releases/releases/download/$MICROMAMBA_VERSION/micromamba-linux-64"
  mkdir -p "$(dirname "$MM")"
  # Download to a temp name, rename only once it runs: a partial binary must
  # never sit at $MM, where every later job would reuse it.
  if curl -fsSL --retry 3 --retry-delay 10 -o "$MM.tmp.$$" "$url" && chmod +x "$MM.tmp.$$" &&
     "$MM.tmp.$$" --version; then
    mv -f "$MM.tmp.$$" "$MM"
  else
    rm -f "$MM.tmp.$$"; return 1
  fi
}

# One `micromamba create` attempt; runs under with_mroot_lock.
mm_create_attempt() {
  local MM="$1" env_file="$2" attempt="$3" rc_args=()
  if [ "$attempt" -eq 1 ]; then
    # Repodata (+ .solv and state.json, which caches the has_zst verdict) is a
    # few MB: drop it every time so an interrupted earlier download cannot
    # poison this solve. Downloaded packages are kept.
    rm -rf "$MAMBA_ROOT_PREFIX/pkgs/cache"
  else
    # "ZSTD decompression error: Unknown frame descriptor" + curl error 23 is
    # the streaming zstd decoder rejecting the body it was SERVED (proxy error
    # page / truncated transfer) -- it reproduces with an empty cache. Retry
    # with plain repodata.json (only an rc file can set this; the env var is
    # ignored), clean index cache + stale locks, and on the last try unused
    # packages and tarballs too.
    if [ "$attempt" -ge 3 ]; then "$MM" clean -y --all >/dev/null 2>&1
    else "$MM" clean -y --index-cache --locks >/dev/null 2>&1; fi
    rm -rf "$MAMBA_ROOT_PREFIX/pkgs/cache"
    printf 'repodata_use_zst: false\n' > "$CI_ROOT/mambarc-nozst.yml"
    rc_args=(--rc-file "$CI_ROOT/mambarc-nozst.yml")
  fi
  rm -rf "$ENV_PREFIX"
  "$MM" create -y ${rc_args[@]+"${rc_args[@]}"} -p "$ENV_PREFIX" -f "$env_file"
}

# Fail fast with a clear log if the solved env cannot run the port.
env_sanity() {
  "$ENV_PREFIX/bin/python" - <<'EOF' 2>&1 | tee "$LOG_DIR/06_env_sanity.log"
import sys, numpy, scipy, pyscf, dpctl, dpnp
import dpctl.compiler  # gpu4pyscf/cupy/rawkernel.py needs dpctl >= 0.23
print("python", sys.version.split()[0], "| dpctl", dpctl.__version__, "| dpnp", dpnp.__version__,
      "| numpy", numpy.__version__, "| scipy", scipy.__version__, "| pyscf", pyscf.__version__)
EOF
  return "${PIPESTATUS[0]}"
}

create_env() {
  local MM="$CI_ROOT/bin/micromamba" tries="${ENV_CREATE_TRIES:-3}" attempt rc=1
  check_disk || return 1
  # Shared package cache: re-solving per pipeline only re-downloads what changed.
  export MAMBA_ROOT_PREFIX="$CI_ROOT/mroot"
  mkdir -p "$MAMBA_ROOT_PREFIX"
  with_mroot_lock fetch_micromamba "$MM" || { echo "could not install micromamba"; return 1; }
  # Modules mode (default, USE_MODULES=1): toolchain from system modules, conda
  # env carries only python + nightly dpnp/dpctl + build tools (env.yml).
  # USE_MODULES=0: full toolchain in conda (env-conda.yml). Override the file
  # with SYCL_CI_ENV_FILE.
  local env_file="${SYCL_CI_ENV_FILE:-}"
  if [ -z "$env_file" ]; then
    if [ "${USE_MODULES:-1}" = "1" ]; then
      env_file="$TRUSTED_DIR/.github/jlse/env.yml"
    else
      env_file="$TRUSTED_DIR/.github/jlse/env-conda.yml"
    fi
  fi
  echo "conda env file: $env_file (USE_MODULES=${USE_MODULES:-1})" | tee "$LOG_DIR/04_env_file.log"
  for attempt in $(seq 1 "$tries"); do
    echo "=== micromamba create: attempt $attempt/$tries ==="
    with_mroot_lock mm_create_attempt "$MM" "$env_file" "$attempt" 2>&1 |
      tee "$LOG_DIR/05_env_create_$attempt.log"
    rc="${PIPESTATUS[0]}"
    [ "$rc" -eq 0 ] && break
    echo "micromamba create failed (rc=$rc, attempt $attempt/$tries)"
    [ "$attempt" -lt "$tries" ] && sleep $((attempt * 60))   # backoff, lock released
  done
  if [ "$rc" -ne 0 ]; then rm -rf "$ENV_PREFIX"; check_disk; return 1; fi
  "$MM" list -p "$ENV_PREFIX" 2>&1 | grep -E 'dpnp|dpctl|dpcpp|intel-sycl-rt|mkl ' | tee "$LOG_DIR/05_env_versions.log"
  # Exact solve of the moving nightly label, for reproducing a past pipeline.
  "$MM" env export -p "$ENV_PREFIX" --explicit > "$LOG_DIR/05_env_explicit.txt" 2>&1 || true
  env_sanity || { echo "env $ENV_PREFIX was created but is unusable (see 06_env_sanity.log)"; return 1; }
  touch "$ENV_PREFIX/.ci_env_ok"
  # Drop envs of old pipelines.
  find "$CI_ROOT/envs" -mindepth 1 -maxdepth 1 -type d -mtime +3 -exec rm -rf {} + 2>/dev/null
}

setup_env() {
  export PATH="$ENV_PREFIX/bin:$PATH"
  if [ "${USE_MODULES:-1}" = "1" ]; then
    # Modules mode (default): toolchain from the node's system modules —
    # icx/icpx/ifx, MKL, TBB, cmake, level-zero loader, IGC/ocloc. The conda env
    # provides python, nightly dpnp/dpctl and the small build tools.
    if ! command -v module >/dev/null 2>&1 && [ -f /etc/profile.d/modules.sh ]; then
      . /etc/profile.d/modules.sh
    fi
    module use "${SYCL_MODULE_PATH:-/soft/modulefiles}"
    # shellcheck disable=SC2086
    module load ${SYCL_MODULES:-intel/oneapi/release/2026.2.0 compiler mkl cmake}
    export CC=icx CXX=icpx FC=ifx
    # MKLROOT/TBBROOT are set by the mkl/compiler modules.
    if ! command -v ocloc >/dev/null 2>&1 && [ -e "$ENV_PREFIX"/bin/ocloc-* ]; then
      ln -sf "$(basename "$ENV_PREFIX"/bin/ocloc-*)" "$ENV_PREFIX/bin/ocloc"
    fi
  else
    # Conda toolchain mode (USE_MODULES=0): everything from the conda env.
    # conda-forge's intel-compute-runtime ships only a versioned ocloc-<ver>;
    # icpx looks for plain `ocloc`.
    if [ ! -e "$ENV_PREFIX/bin/ocloc" ]; then
      for f in "$ENV_PREFIX"/bin/ocloc-*; do
        [ -x "$f" ] && ln -sf "$(basename "$f")" "$ENV_PREFIX/bin/ocloc" && break
      done
    fi
    export MKLROOT="$ENV_PREFIX" TBBROOT="$ENV_PREFIX"
    export CC=icx CXX=icpx FC=ifx
  fi
  # Level-zero: loader from modules or the conda env, GPU driver (libze_intel_gpu)
  # from the node's system install (no conda package exists).
  ZE_DRIVER_DIR="${ZE_DRIVER_DIR:-/soft/libraries/intel-gpu-umd/821.36/lib64}"
  export LD_LIBRARY_PATH="$ENV_PREFIX/lib:$ZE_DRIVER_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export CCACHE_DIR="$CI_ROOT/ccache" CCACHE_MAXSIZE=10G
  export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR="$CI_ROOT/sycl_cache"
  # chiatta nodes have 8 two-tile Max 1550 cards. Single-GPU CI: with the FLAT
  # hierarchy every tile is a device, and mask 0 leaves exactly one (tile 0).
  # All overridable from the job environment.
  export ZE_ENABLE_SYSMAN="${ZE_ENABLE_SYSMAN:-1}"
  export ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-FLAT}"
  export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
}

toolchain_check() {
  {
    hostname; date
    env | grep -E '^(ZE_|ONEAPI_DEVICE_SELECTOR|OMP_NUM_THREADS)' | sort
    icpx --version | head -1
    python -c "import dpctl, dpnp; print('dpctl', dpctl.__version__, '| dpnp', dpnp.__version__); dpctl.lsplatform()"
    # Identify the actual device running this CI at the top of the log.
    python -c "import dpctl; d = dpctl.get_devices(device_type='gpu'); print('GPU devices:', *d, sep='\n  '); assert d, 'no SYCL GPU device'"
    sycl-ls --verbose 2>/dev/null | head -20 || true
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

checkout_pr() {
  [ -n "${PR_NUMBER:-}" ] || return 0
  local repo="${PR_REPO:-pyscf/gpu4pyscf}" dir="$CI_ROOT/pr-src/${CI_PIPELINE_ID:-local}"
  mkdir -p "$CI_ROOT/pr-src"
  cd "$TRUSTED_DIR" && git worktree prune
  find "$CI_ROOT/pr-src" -mindepth 1 -maxdepth 1 -type d -mtime +3 -exec rm -rf {} + 2>/dev/null
  rm -rf "$dir"
  # GitHub's merge ref is the PR merged into its base, like actions/checkout.
  if ! git fetch --depth=1 --no-tags "https://github.com/$repo.git" "pull/$PR_NUMBER/merge"; then
    echo "no merge ref (conflicts?): testing the PR head instead"
    git fetch --depth=1 --no-tags "https://github.com/$repo.git" "pull/$PR_NUMBER/head" || return 1
  fi
  git worktree add --detach "$dir" FETCH_HEAD || return 1
  SRC_DIR="$dir"
  echo "testing $repo PR #$PR_NUMBER at $(git -C "$dir" rev-parse HEAD) (head when triggered: ${PR_SHA:-unknown})" |
    tee "$LOG_DIR/03_pr.log"
}

# The port's CMake links libxc.so and libgdft.so against libexchcxx.so by absolute
# path, i.e. into the build job's checkout, which the runner deletes when the job
# ends. Make the published tree self-contained: relative NEEDED + $ORIGIN rpath.
relocate_libs() {
  local root="$1" f n bad=0
  for f in $(find "$root/gpu4pyscf/lib" -name '*.so'); do
    for n in $(readelf -d "$f" | sed -n 's/.*NEEDED.*\[\(\/.*\)\]/\1/p'); do
      patchelf --replace-needed "$n" "$(basename "$n")" "$f" &&
        patchelf --add-rpath '$ORIGIN:$ORIGIN/deps/lib' "$f" || bad=1
      echo "relocated $(basename "$f") -> $(basename "$n")"
    done
  done
  # Nothing may still point at an absolute path.
  for f in $(find "$root/gpu4pyscf/lib" -name '*.so'); do
    if readelf -d "$f" | grep -E 'NEEDED.*\[/' ; then echo "absolute dependency left in $f"; bad=1; fi
  done
  return $bad
}

do_build() {
  mkdir -p "$CI_ROOT/envs"
  create_env || { echo "environment creation failed"; exit 1; }
  setup_env || { echo "environment setup failed"; exit 1; }
  checkout_pr || { echo "could not fetch PR $PR_NUMBER"; exit 1; }
  cd "$SRC_DIR"
  # A CMake cache left by an earlier job would point at that job's conda env.
  rm -rf "$SRC_DIR/build"
  export PYTHONPATH="$SRC_DIR"
  toolchain_check || exit 1

  export GPU4PYSCF_BACKEND=sycl
  # Set SYCL_ARCHITECTURES=pvc for AOT compilation targeting PVC GPUs.
  # Otherwise, use generic spir64 JIT so device code is compiled at runtime.
  export SYCL_ARCHITECTURES="${SYCL_ARCHITECTURES:-generic}"
  echo "SYCL_ARCHITECTURES=$SYCL_ARCHITECTURES" | tee "$LOG_DIR/20_architecture.log"

  # ccache only if available (conda env or module); otherwise build without it.
  local launchers=""
  if command -v ccache >/dev/null 2>&1; then
    launchers="-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
  fi
  export CMAKE_CONFIGURE_ARGS="-DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx \
    -DCMAKE_Fortran_COMPILER=ifx -DCMAKE_BUILD_TYPE=Release $launchers"
  python setup.py build 2>&1 | tee "$LOG_DIR/40_build.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "build failed"; exit 1; }
  ccache -s || true

  python -c "import gpu4pyscf; print('gpu4pyscf import OK')" 2>&1 | tee "$LOG_DIR/50_smoke.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || exit 1

  # Publish the built tree for the test jobs of this pipeline (shared filesystem).
  mkdir -p "$BUILD_DIR"
  rsync -a --delete --exclude=.git --exclude=build --exclude=ci-logs "$SRC_DIR"/ "$BUILD_DIR"/ || exit 1
  relocate_libs "$BUILD_DIR" || { echo "could not relocate libraries"; exit 1; }
  echo "$ENV_PREFIX" > "$BUILD_DIR/.env_prefix"
  touch "$BUILD_DIR/.build_ok"
  find "$CI_ROOT/builds" -mindepth 1 -maxdepth 1 -type d -mtime +7 -exec rm -rf {} + 2>/dev/null
  echo "published build to $BUILD_DIR"
}

do_test() {
  [ $# -gt 0 ] || { echo "usage: $0 test <pytest paths...>"; exit 2; }
  [ -e "$BUILD_DIR/.build_ok" ] || { echo "no build for pipeline ${CI_PIPELINE_ID:-local} at $BUILD_DIR (did the build job run?)"; exit 1; }
  ENV_PREFIX="$(cat "$BUILD_DIR/.env_prefix")"
  [ -d "$ENV_PREFIX" ] || { echo "env $ENV_PREFIX from the build is gone"; exit 1; }
  [ -e "$ENV_PREFIX/.ci_env_ok" ] || { echo "env $ENV_PREFIX is incomplete (no .ci_env_ok): re-run the build job"; exit 1; }
  # Keep the 3-day env / 7-day build reapers off a pipeline whose tests are re-run.
  touch "$ENV_PREFIX" "$BUILD_DIR"
  check_disk "${CI_MIN_FREE_GB_TEST:-10}" || exit 1

  # How many GPU tiles to spread the tests over. Detected with the affinity
  # mask removed (setup_env defaults it to 0 = single tile). Each test process
  # gets exactly one tile via ZE_AFFINITY_MASK, so this stays on the supported
  # single-device-per-process code path; the parallelism is across processes.
  local tiles="${SYCL_TILES:-}"
  if [ -z "$tiles" ]; then
    tiles=$(env -u ZE_AFFINITY_MASK ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
      python3 -c 'import dpctl; print(len(dpctl.get_devices(device_type="gpu")))' 2>/dev/null || echo 1)
  fi
  [ "$tiles" -ge 1 ] 2>/dev/null || tiles=1

  setup_env || { echo "environment setup failed"; exit 1; }
  cd "$BUILD_DIR"
  export PYTHONPATH="$BUILD_DIR"
  toolchain_check || exit 1

  # Same pytest invocation as upstream unittest.yml (.github/workflows/unittest.yml),
  # including -k 'not _high_cost': the *_high_cost tests carry CUDA-tuned bounds
  # (e.g. GDM cycle counts) and are excluded from upstream CI.
  local pytest_args=(-n "${PYTEST_WORKERS:-4}" --dist loadfile
    -m 'not slow and not benchmark and not special' -k 'not _high_cost'
    --cov="$BUILD_DIR" --durations=50)

  if [ "$tiles" -le 1 ]; then
    pytest "${pytest_args[@]}" "$@" 2>&1 | tee "$LOG_DIR/60_pytest.log"
    status="${PIPESTATUS[0]}"
    rm -rf .pytest_cache
    exit "$status"
  fi

  # Multi-tile: shard the test files round-robin over the tiles and run one
  # pytest process per tile. Tile t lives on card t/2; on chiatta each of the
  # 8 cards owns one NUMA node (node n = cpus n*8..n*8+7 and 64+n*8..71+n*8),
  # so bind each tile process to its card's node with numactl (taskset
  # fallback). This also keeps oneTBB/MKL thread arenas sized to the node
  # instead of the whole 128-core machine.
  echo "running tests on $tiles GPU tiles (ZE_FLAT_DEVICE_HIERARCHY=FLAT)" | tee "$LOG_DIR/60_pytest.log"
  local shard_dir="$LOG_DIR/shards"
  rm -rf "$shard_dir"; mkdir -p "$shard_dir"
  : > "$shard_dir/all.txt"
  for p in "$@"; do
    if [ -d "$p" ]; then find "$p" -name 'test_*.py' -type f | sort >> "$shard_dir/all.txt"
    else echo "$p" >> "$shard_dir/all.txt"; fi
  done
  split -n "r/$tiles" -d "$shard_dir/all.txt" "$shard_dir/shard_"
  if [ ! -s "$shard_dir/all.txt" ]; then
    echo "no test files found under: $* -- failing instead of passing vacuously" | tee -a "$LOG_DIR/60_pytest.log"
    exit 1
  fi
  local tile_workers="${SYCL_TILE_WORKERS:-2}" tile_omp="${SYCL_TILE_OMP:-4}"
  local tile_timeout="${SYCL_TILE_TIMEOUT:-6300}"   # < 2h job timeout: logs survive a hang
  local pids=() launched=() t i trc status=0 shard
  for t in $(seq 0 $((tiles - 1))); do
    shard="$shard_dir/shard_$(printf '%02d' "$t")"
    [ -s "$shard" ] || continue
    local node=$(( t / 2 ))
    local bind=""
    if command -v numactl >/dev/null 2>&1; then
      bind="numactl --cpunodebind=$node --membind=$node"
    else
      local c1 c2
      c1=$(seq -s, $((node * 8)) $((node * 8 + 7)))
      c2=$(seq -s, $((64 + node * 8)) $((64 + node * 8 + 7)))
      bind="taskset -c $c1,$c2"
    fi
    (
      export ZE_AFFINITY_MASK="$t"
      {
        # Identify the actual device (tile) running this shard at the top of its log.
        echo "tile=$t host=$(hostname)"
        python -c "import dpctl; print('device:', dpctl.get_devices(device_type='gpu'))" 2>/dev/null || true
        $bind env OMP_NUM_THREADS="$tile_omp" timeout -k 120 "$tile_timeout" \
          pytest -n "$tile_workers" --dist loadfile \
          -m 'not slow and not benchmark and not special' -k 'not _high_cost' \
          --durations=50 $(cat "$shard")
      } > "$LOG_DIR/60_pytest_tile$t.log" 2>&1
      trc=$?
      echo "$trc" > "$shard_dir/rc_$t"
      exit "$trc"
    ) &
    pids+=($!); launched+=("$t")
  done
  for i in "${!pids[@]}"; do
    t="${launched[$i]}"
    wait "${pids[$i]}"; trc=$?
    if [ ! -s "$shard_dir/rc_$t" ]; then
      # Subshell killed before it could record a status: never count as a pass.
      echo "tile $t: no exit status recorded (wait rc=$trc) -> FAILED" | tee -a "$LOG_DIR/60_pytest.log"
      status=1
    elif [ "$(cat "$shard_dir/rc_$t")" != "0" ] || [ "$trc" -ne 0 ]; then
      [ "$(cat "$shard_dir/rc_$t")" = "124" ] &&
        echo "tile $t: TIMEOUT after ${tile_timeout}s" | tee -a "$LOG_DIR/60_pytest.log"
      status=1
    fi
  done
  for t in "${launched[@]}"; do
    printf 'tile %2d (rc=%s): %s\n' "$t" "$(cat "$shard_dir/rc_$t" 2>/dev/null || echo '?')" \
      "$(grep -E '^=+ .*(passed|failed|error|no tests ran)' "$LOG_DIR/60_pytest_tile$t.log" | tail -1)"
  done | tee -a "$LOG_DIR/60_pytest.log"
  rm -rf .pytest_cache
  [ "$status" -eq 0 ] && echo "ALL TILES PASSED" | tee -a "$LOG_DIR/60_pytest.log" \
    || echo "SOME TILES FAILED (see $LOG_DIR/60_pytest_tile*.log)" | tee -a "$LOG_DIR/60_pytest.log"
  exit "$status"
}

case "$MODE" in
  build) do_build ;;
  test)  do_test "$@" ;;
  *) echo "usage: $0 build | test <paths...>"; exit 2 ;;
esac
