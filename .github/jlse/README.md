# SYCL CI on JLSE chiatta (Intel PVC)

Driven by `/.gitlab-ci.yml` on gitlab.jlse.anl.gov (Jacamar `batch` runner, queue `chiatta`).

| File | Role |
|---|---|
| `ci_sycl.sh` | `build` (fresh conda env per pipeline, PVC AOT build) and `test <paths>` on one tile |
| `env.yml` | conda env: nightly dpnp, dpctl, oneAPI 2026.2.0, test deps as in the CUDA CI |
| `poll_prs.py` | scheduled job: finds new/updated upstream PRs and starts a pipeline for each |
| `report_pr.py` | end of a PR pipeline: publishes logs (gist) and a commit status on the PR |

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
     `SYCL / chiatta (PVC)` on the PR commit linking to the gist. This needs a
     token from someone with write access to the upstream repo.
   Without them the logs stay in the GitLab job artifacts only.

Variables: `PR_REPO` (default `pyscf/gpu4pyscf`), `PR_MAX_TRIGGERS` (default 2 per
poll), `PR_ALLOWED_ASSOCIATIONS`, `GITLAB_TRIGGER_TOKEN` (default: the job token).

## Security

PR code runs on JLSE as the account that owns the pipelines. By default only PRs
whose author is OWNER, MEMBER, COLLABORATOR or CONTRIBUTOR run automatically;
first-time contributors are listed in the poller log as "held". Widen
`PR_ALLOWED_ASSOCIATIONS` only if upstream accepts that risk.

## Not tested end to end

A full PR pipeline needs the SYCL backend in the PR's base branch, so it can only
run for real after this branch is merged. Tested: the poller against the live upstream
API (dry run), the PR worktree checkout, and the build/test flow on a branch.
