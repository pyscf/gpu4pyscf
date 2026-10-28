#!/usr/bin/env python3
"""Start a SYCL CI pipeline for every new/updated open PR of the upstream repo.

Runs from a scheduled GitLab pipeline on the JLSE shell runner (login node).
GitHub's public API is read without a token (set GITHUB_READ_TOKEN to raise
the rate limit). State (PR number -> last head SHA triggered) lives in
$CI_ROOT/pr-state.json so a PR is built once per pushed commit.

Environment:
  PR_REPO                 upstream repo to watch        (default pyscf/gpu4pyscf)
  PR_ALLOWED_ASSOCIATIONS author_association values that run automatically
                          (default OWNER,MEMBER,COLLABORATOR,CONTRIBUTOR; PRs
                          from first-time contributors are held and listed)
  PR_MAX_TRIGGERS         pipelines started per poll    (default 2)
  GITLAB_TRIGGER_TOKEN    project trigger token (default: the job token)
  PR_POLL_BACKLOG=1       on the very first poll, also build the PRs already open
                          (default: just record them, so only later pushes build)
  PR_POLL_DRY_RUN=1       print what would happen, trigger nothing
"""
import json
import os
import sys
import urllib.parse
import urllib.request

REPO = os.environ.get("PR_REPO", "pyscf/gpu4pyscf")
ALLOWED = set(os.environ.get(
    "PR_ALLOWED_ASSOCIATIONS", "OWNER,MEMBER,COLLABORATOR,CONTRIBUTOR").split(","))
MAX_TRIGGERS = int(os.environ.get("PR_MAX_TRIGGERS", "2"))
DRY = os.environ.get("PR_POLL_DRY_RUN") == "1"
STATE = os.path.join(os.environ.get("CI_ROOT", os.path.expanduser("~/gitlab-sycl-ci")),
                     "pr-state.json")


def github(path):
    req = urllib.request.Request("https://api.github.com" + path)
    req.add_header("Accept", "application/vnd.github+json")
    tok = os.environ.get("GITHUB_READ_TOKEN")
    if tok:
        req.add_header("Authorization", "Bearer " + tok)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def trigger(pr, sha):
    api = os.environ["CI_API_V4_URL"]
    data = urllib.parse.urlencode({
        "token": os.environ.get("GITLAB_TRIGGER_TOKEN") or os.environ["CI_JOB_TOKEN"],
        "ref": os.environ.get("CI_COMMIT_REF_NAME", os.environ.get("CI_DEFAULT_BRANCH", "main")),
        "variables[PR_NUMBER]": str(pr),
        "variables[PR_SHA]": sha,
        "variables[PR_REPO]": REPO,
    }).encode()
    url = "%s/projects/%s/trigger/pipeline" % (api, os.environ["CI_PROJECT_ID"])
    with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=30) as r:
        return json.load(r).get("web_url", "?")


def main():
    first_poll = not os.path.exists(STATE)
    try:
        with open(STATE) as f:
            state = json.load(f)
    except (IOError, ValueError):
        state = {}

    prs = github("/repos/%s/pulls?state=open&per_page=100" % REPO)
    if first_poll and os.environ.get("PR_POLL_BACKLOG") != "1":
        state = {str(p["number"]): p["head"]["sha"] for p in prs}
        print("first poll: recorded %d open PRs, none built (PR_POLL_BACKLOG=1 to build them)"
              % len(prs))
        if not DRY:
            os.makedirs(os.path.dirname(STATE), exist_ok=True)
            with open(STATE, "w") as f:
                json.dump(state, f)
        return
    started = 0
    for pr in prs:
        num, sha = str(pr["number"]), pr["head"]["sha"]
        assoc = pr.get("author_association", "NONE")
        if pr.get("draft"):
            continue
        if state.get(num) == sha:
            continue
        if assoc not in ALLOWED:
            print("PR #%s held: author_association=%s (allowed: %s)"
                  % (num, assoc, ",".join(sorted(ALLOWED))))
            continue
        if started >= MAX_TRIGGERS:
            print("PR #%s queued for the next poll (limit %d per poll)" % (num, MAX_TRIGGERS))
            continue
        if DRY:
            print("[dry-run] would trigger PR #%s at %s (%s)" % (num, sha[:10], assoc))
        else:
            print("PR #%s at %s -> %s" % (num, sha[:10], trigger(num, sha)))
            state[num] = sha
        started += 1

    if not DRY:
        # Forget PRs that are no longer open.
        open_nums = {str(p["number"]) for p in prs}
        state = {k: v for k, v in state.items() if k in open_nums}
        os.makedirs(os.path.dirname(STATE), exist_ok=True)
        with open(STATE, "w") as f:
            json.dump(state, f)
    print("polled %d open PRs of %s, started %d" % (len(prs), REPO, started))


if __name__ == "__main__":
    sys.exit(main())
