# SYCL CI on JLSE chiatta (Intel PVC)

Driven by `/.gitlab-ci.yml` on gitlab.jlse.anl.gov (Jacamar `batch` runner, queue `chiatta`).

This is the SYCL CI. It replaces the former GitHub-hosted CPU/OpenCL workflow
(`sycl-cpu-ci.yml`, removed) with real PVC GPU runs.

| File | Role |
|---|---|
| `ci_sycl.sh` | `build` (fresh conda env per pipeline, PVC AOT build, tile 0) and `test <paths>` sharded over all tiles |
| `env.yml` | conda env (modules mode, default): **only** python + nightly dpnp/dpctl + test deps; toolchain from system modules |
| `env-conda.yml` | fallback full conda env incl. oneAPI 2026.2.0 toolchain (`USE_MODULES=0`) |
| `poll_prs.py` | scheduled job: finds new/updated upstream PRs and starts a pipeline for each |
| `report_pr.py` | end of a PR pipeline: publishes logs (gist) and a commit status on the PR |

## Toolchain: system modules (default) vs conda

By default (`USE_MODULES=1`) the compiler toolchain comes from the node's system
modules, loaded in `ci_sycl.sh setup_env`: `module use $SYCL_MODULE_PATH` (default
`/soft/modulefiles`) then `module load $SYCL_MODULES` (default
`intel/oneapi/release/2026.2.0 compiler mkl cmake`), which provides icx/icpx/ifx,
MKL, TBB, cmake, the level-zero loader and ocloc. The conda env (`env.yml`) then
only carries python 3.13, nightly dpnp/dpctl (`dppy/label/dev` + conda-forge), and
small build/test tools. The GPU driver (`libze_intel_gpu`) always comes from the
node via `ZE_DRIVER_DIR`. Set `USE_MODULES=0` to use the full conda toolchain from
`env-conda.yml` instead; override the env file with `SYCL_CI_ENV_FILE`.

## Multi-tile testing

A chiatta node has 8 Max 1550 cards = 16 GPU tiles (`ONEAPI_DEVICE_SELECTOR=level_zero:gpu`
+ `ZE_FLAT_DEVICE_HIERARCHY=FLAT` expose all 16 as devices). The build runs on one tile;
`ci_sycl.sh test` shards the test files round-robin over `SYCL_TILES` tiles (default: all
visible) and runs one pytest process per tile with `ZE_AFFINITY_MASK=<tile>`, so every
process stays on the supported single-device code path. Each tile process is bound to its
card's NUMA node with `numactl --cpunodebind` (taskset fallback), which also keeps the
oneTBB/MKL thread arenas node-sized. **Never more than one test process on a tile at any
time:** each tile runs a single sequential pytest (no pytest-xdist `-n`), because several
processes sharing a tile caused GPU page faults and wrong results. `SYCL_TILE_OMP` (default 4)
sets the OpenMP threads per tile process. Per-tile logs land in
`ci-logs/test/60_pytest_tile<N>.log`; the job fails if any tile fails. Test selection is
identical to the upstream single-GPU CUDA CI (`-m 'not slow and not benchmark and not
special' -k 'not _high_cost'`); only the parallelism differs (tiles, not xdist).

## Pull requests of the upstream GitHub repo

GitLab can't be called from GitHub, so GitLab polls GitHub instead:

1. **CI/CD > Schedules > New schedule**: every 5-10 minutes, variable `POLL_PRS=true`
   (the instance may enforce a minimum interval). The first poll only records the
   PRs that are already open; after that, each new or updated PR starts a pipeline.
2. The PR pipeline builds GitHub's merge ref of the PR (the PR head if it conflicts).
   The CI scripts always come from this repo, never from the PR.
3. `report-*` jobs print the result and, if tokens are set (masked CI/CD variables):
   - `GITHUB_GIST_TOKEN` (gist scope): logs go to a gist anyone with the link can read.
   - `GITHUB_STATUS_TOKEN` (commit statuses on the upstream repo): posts
     `SYCL CI` on the PR commit linking to the gist. This needs a
     token from someone with **write access** to the upstream repo (a pyscf
     maintainer) — fork-only contributors cannot mint it.
   - `GITHUB_COMMENT_TOKEN` (`public_repo` scope — any GitHub account, since the
     upstream repo is public): `report_pr.py` posts ONE PR comment per PR (a
     marker comment, updated in place on every pipeline run — never a new comment
     per run) with the result, the log/pipeline link, and per-job links
     (`sycl-build`, `sycl-test`) from the GitLab pipeline API. Use this when no
     maintainer status token is available; the result is then visible in the PR
     thread instead of as a commit status.
   Without them the logs stay in the GitLab job artifacts only.

## Full public logs (CI-only mirror) — needs instance to allow Public visibility

This GitLab project is a CI-only mirror of the (already public) GitHub repo, so
IF the instance allowed it, setting the project visibility to **public** (and
`public_jobs` on) would expose no source -- only CI pipelines, job logs and the
`ci-logs/` artifacts would become anonymously browsable at the pipeline URL
(confirm the host is reachable from the public internet; artifacts still expire
after 2 weeks). With only `GITHUB_STATUS_TOKEN` set, `report_pr.py` would then
link the `SYCL CI` commit status straight to the public pipeline
page -- full logs on the GitHub PR, no gist token needed. **gitlab.jlse.anl.gov
currently restricts visibility to Private**, so this option is unavailable there
unless an instance admin lifts the restriction; the minimum setup above (gist +
PR comment) is the working path. Trade-off if enabled: logs of every triggered PR
build (including untrusted authors' once they pass `PR_ALLOWED_ASSOCIATIONS`)
become public and may contain internal hostnames/paths.

Variables: `PR_REPO` (default `pyscf/gpu4pyscf`), `PR_MAX_TRIGGERS` (default 2 per
poll), `PR_ALLOWED_ASSOCIATIONS`, `GITLAB_TRIGGER_TOKEN` (default: the job token).

## Triggering

There is no GitHub-side workflow: GitHub cannot reach gitlab.jlse (private,
behind ANL) and fork PRs get no repository secrets, so the scheduled poller is
the single trigger path for every PR (forks and drafts included). The only
SYCL check on a PR is the `SYCL CI` commit status posted back by the pipeline.

## Minimum setup: CI logs visible on the GitHub PR

gitlab.jlse.anl.gov currently restricts project visibility to **Private** (instance
admin setting), so pipeline pages and artifacts are NOT anonymously accessible and
the project cannot be made public. The minimum setup therefore uses a link-shareable
gist + a PR comment — both tokens mintable by any GitHub account (the upstream repo
is public), no pyscf maintainer needed:

1. **Masked CI/CD variable `GITHUB_GIST_TOKEN`**: classic `gist` scope or
   fine-grained Gists:write. Log tails (400 lines / 60 KB per file) go to a secret
   gist — anyone with the link can read it.
2. **Masked CI/CD variable `GITHUB_COMMENT_TOKEN`**: fine-grained Issues:write on
   `pyscf/gpu4pyscf`, or classic `public_repo`. Commenting on a public repo needs
   no upstream write access. `report_pr.py` posts a PR comment with the result and
   the gist link.
3. **A CI/CD schedule with `POLL_PRS=true`** (every 5-10 min) so fork PRs get
   pipelines (fork PRs receive no GitHub secrets, so the GitHub-side trigger
   workflow cannot fire for them).

Optional extra: `GITHUB_STATUS_TOKEN` (`repo`-scope / Commit-statuses:write from an
account with **write** on `pyscf/gpu4pyscf`, i.e. a maintainer) adds the
`SYCL CI` commit status on the PR head linking to the gist.

## CUDA-style status-only reporting (no comments, no gists)

To report pass/fail exactly like the CUDA CI does — a single check on the PR, no
extra output — set **only** `GITHUB_STATUS_TOKEN` and leave `GITHUB_GIST_TOKEN` /
`GITHUB_COMMENT_TOKEN` unset. `report_pr.py` is fully conditional: with only the
status token it posts just the `SYCL CI` commit status on the PR head SHA
(`target_url` = the GitLab pipeline URL) and prints everything else only to the
GitLab job log. No gist is created, no PR comment is posted. Caveats: commit
statuses require push access on `pyscf/gpu4pyscf` (maintainer token), and on this
Private-only instance the status link lands on the GitLab sign-in page — the pass/
fail check itself still shows on the PR. Without a maintainer token, the PR comment
(`GITHUB_COMMENT_TOKEN`, `public_repo`) is the only no-maintainer reporting path.

## CI/CD variables to set on the GitLab project (project owner/maintainer)

Set these as **masked** variables at project Settings > CI/CD > Variables (values are
write-only once saved). The project owner/maintainer does this; the token *values* come
from the GitHub accounts noted below.

| Variable | Scope needed | Provided by | Verify with |
|---|---|---|---|
| `GITHUB_GIST_TOKEN` | `gist` (classic PAT) or Gists: write (fine-grained) | any GitHub account | `curl -H "Authorization: Bearer $T" https://api.github.com/gists -d '{"public":false,"files":{"t":{"content":"x"}}}'` |
| `GITHUB_STATUS_TOKEN` | commit-status write on `pyscf/gpu4pyscf` (`repo` classic, or fine-grained Commit statuses: write on that repo) | an account with **write** access to `pyscf/gpu4pyscf` — a pyscf maintainer; NOT mintable by fork-only contributors | `curl -H "Authorization: Bearer $T" https://api.github.com/repos/pyscf/gpu4pyscf -q .permissions` → `push: true` |
| `GITHUB_COMMENT_TOKEN` | `public_repo` (classic) or Issues: write (fine-grained) on the public repo | **any GitHub account** — commenting on a public repo needs no upstream write access. This is the fallback when no maintainer token is available: `report_pr.py` posts a PR comment with the result + log link | `curl -H "Authorization: Bearer $T" -X POST https://api.github.com/repos/pyscf/gpu4pyscf/issues/1/comments -d '{"body":"test"}' (then delete it) |
| `GITHUB_READ_TOKEN` (optional) | public repo read | any account | raises the poller's anonymous API rate limit |
| `GITLAB_TRIGGER_TOKEN` (optional) | project trigger token | create at project Settings > CI/CD > Pipeline trigger tokens; defaults to the job token | used by `poll_prs.py` |
| `POLL_PRS=true` | schedule variable | set on a CI/CD schedule (every 5-10 min) | poller runs, prints held/triggered PRs |

Optional: `PR_REPO` (default `pyscf/gpu4pyscf`), `PR_ALLOWED_ASSOCIATIONS` (default
OWNER,MEMBER,COLLABORATOR,CONTRIBUTOR), `PR_MAX_TRIGGERS` (default 2), `PR_SKIP_DRAFTS`
(default off: draft PRs are built too, like the CUDA CI), `SYCL_TESTS`,
`SYCL_TILES`, `SYCL_TILE_OMP`, `SYCL_ARCHITECTURES`, `USE_MODULES`,
`SYCL_MODULES`, `SYCL_MODULE_PATH`, `SYCL_CI_ENV_FILE`, `CI_ROOT`,
`ZE_DRIVER_DIR`.


## Security

PR code runs on JLSE as the account that owns the pipelines. By default only PRs
whose author is OWNER, MEMBER, COLLABORATOR or CONTRIBUTOR run automatically;
first-time contributors are listed in the poller log as "held". Widen
`PR_ALLOWED_ASSOCIATIONS` only if upstream accepts that risk.

**CI/CD variables are visible to every job of a pipeline, including PR-triggered
build/test jobs that execute untrusted PR code.** So:

- Prefer low-blast-radius tokens as project variables: `GITHUB_GIST_TOKEN` (gist
  scope) and `GITHUB_COMMENT_TOKEN` (`public_repo`) can only write gists/comments
  — a leaked one cannot touch repo code or statuses. Never put a write-capable
  token (`GITHUB_STATUS_TOKEN`, `repo` scope) in project variables while untrusted
  PR pipelines run; if you need the commit status, have a maintainer install a
  narrowly-scoped **GitHub App** (Commit statuses: write on `pyscf/gpu4pyscf`)
  instead of a personal PAT, and keep it out of PR-reachable env.
- The CI scripts always come from the trusted checkout (`ci_sycl.sh` header), never
  from the PR worktree; PR code only runs inside pytest against the published build.
- Log tails go to an unlisted gist (`public: false`) — anyone with the URL can read
  it, so keep tokens out of the logs (masked variables are redacted in job logs).

## Reporting behavior

**Log visibility for PR viewers:** gitlab.jlse is Private-only, so GitLab pipeline/job
links need a login. The public copy is the gist that the `SYCL CI` status's **Details**
link opens: `00_SUMMARY.md` first -- **where it failed** (job + stage + reason, written by
`ci_sycl.sh`'s `fail` into `ci-logs/<job>/00_failure.log`), device, per-tile test totals,
crashes/timeouts, every failing test with **how it failed** (raising location + `E` lines),
build errors with context, env-solve and import/sanity errors -- then each log. pytest logs keep their **full FAILURES section** (all tracebacks plus
the short summary); other logs keep their last 400 lines. The status description itself says
e.g. `SYCL CI failed: 3 failed tests`. Full logs stay in the GitLab job artifacts (2 weeks).

One GitHub check per PR head: the poller posts a `pending` `SYCL CI` status when a
pipeline starts (`poll_prs.py`), and the `report-*` jobs post the final
success/failure under the same context and `PR_SHA`, so GitHub groups them into a
single check entry (no separate build/test checks). The PR comment (marker-based,
updated in place) additionally lists per-job links (`sycl-build`, `sycl-test`). The
check name stays generic ("SYCL CI"); the actual GPU device is printed at the top of
every job log (dpctl / `sycl-ls --verbose`).

## Not tested end to end

A full PR pipeline needs the SYCL backend in the PR's base branch; upstream/master
now carries it, so PR pipelines run for real. Tested: the poller against the live
upstream API (dry run), the PR worktree checkout, and the build/test flow on a branch.
