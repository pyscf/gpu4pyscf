#!/usr/bin/env python3
"""Publish a SYCL CI result for a PR so contributors can read it without GitLab.

usage: report_pr.py success|failure     (run at the end of a PR pipeline)

Always prints a summary to the job log. Each output channel is independent and
only activates when its token is set, so setting ONLY GITHUB_STATUS_TOKEN gives
CUDA-style status-only reporting (one commit status on the PR head, nothing
else posted to GitHub). Optionally, with credentials set as
masked GitLab CI/CD variables:
  GITHUB_GIST_TOKEN    token with the gist scope; logs are uploaded to a gist
                       owned by that account (readable by anyone with the link)
  GITHUB_STATUS_TOKEN  token allowed to write commit statuses on $PR_REPO
                       (needs write access to the upstream repo -- a maintainer);
                       posts the 'SYCL CI' status on the PR commit
                       with the gist (or this pipeline) as the details link.
  GITHUB_COMMENT_TOKEN token with public_repo scope; because $PR_REPO is a
                       PUBLIC repo, any GitHub account can mint this -- no write
                       access to the upstream repo needed. Posts ONE PR comment
                       (a marker comment updated on every pipeline run, not a new
                       comment each time) with the result, the log/pipeline link,
                       and per-job links (sycl-build, sycl-test).
"""
import glob
import json
import os
import sys
import urllib.request

TAIL_LINES = 400
MAX_FILE = 60000
COMMENT_MARKER = "<!-- gpu4pyscf-sycl-ci-report -->"


def request(method, url, payload, token):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data, method=method)
    req.add_header("Authorization", "Bearer " + token)
    req.add_header("Accept", "application/vnd.github+json")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def post(url, payload, token):
    return request("POST", url, payload, token)


def job_links():
    """Markdown links to the sycl-build/sycl-test jobs of this pipeline."""
    api = os.environ.get("CI_API_V4_URL", "")
    proj = os.environ.get("CI_PROJECT_ID", "")
    pipe = os.environ.get("CI_PIPELINE_ID", "")
    if not (api and proj and pipe):
        return ""
    token = os.environ.get("GITLAB_REPORT_TOKEN") or os.environ.get("CI_JOB_TOKEN", "")
    if not token:
        return ""
    try:
        req = urllib.request.Request(
            "%s/projects/%s/pipelines/%s/jobs?per_page=100" % (api, proj, pipe))
        req.add_header("Authorization", "Bearer " + token)
        with urllib.request.urlopen(req, timeout=30) as r:
            jobs = json.load(r)
    except Exception as e:
        print("could not list pipeline jobs: %s" % e)
        return ""
    lines = ["- [%s](%s) — %s" % (j["name"], j.get("web_url", ""), j.get("status", "?"))
             for j in jobs if j.get("name", "").startswith(("sycl-build", "sycl-test"))]
    return "\n".join(lines)


def main():
    result = sys.argv[1] if len(sys.argv) > 1 else "failure"
    repo, pr, sha = os.environ.get("PR_REPO"), os.environ.get("PR_NUMBER"), os.environ.get("PR_SHA")
    print("SYCL CI %s for %s PR #%s (%s)" % (result, repo, pr, (sha or "")[:10]))

    files = {}
    for path in sorted(glob.glob("ci-logs/**/*.log", recursive=True)):
        with open(path, errors="replace") as f:
            tail = "".join(f.readlines()[-TAIL_LINES:])[-MAX_FILE:]
        files[path.replace("/", "__")] = {"content": tail or "(empty)"}
    print("log files: %s" % ", ".join(files) if files else "no log files found")

    details = os.environ.get("CI_PIPELINE_URL", "")
    gist_token = os.environ.get("GITHUB_GIST_TOKEN")
    if gist_token and files:
        # Independent channel: a gist failure must not block status/comment.
        try:
            g = post("https://api.github.com/gists", {
                "description": "SYCL CI %s PR #%s %s" % (repo, pr, (sha or "")[:10]),
                "public": False, "files": files}, gist_token)
            details = g["html_url"]
            print("logs: " + details)
        except Exception as e:
            print("gist upload failed (status/comment still posted): %s" % e)

    status_token = os.environ.get("GITHUB_STATUS_TOKEN")
    if status_token and repo and sha:
        post("https://api.github.com/repos/%s/statuses/%s" % (repo, sha), {
            "state": {"success": "success", "canceled": "error"}.get(result, "failure"),
            "context": "SYCL CI",
            "description": "SYCL CI %s" % {"success": "passed", "canceled": "canceled"}.get(result, "failed"),
            "target_url": details}, status_token)
        print("posted commit status on %s@%s" % (repo, sha[:10]))
    elif not status_token:
        print("GITHUB_STATUS_TOKEN not set: no status posted to GitHub")

    comment_token = os.environ.get("GITHUB_COMMENT_TOKEN")
    if comment_token and repo and pr:
        jobs = job_links()
        body = ("%s\n**SYCL CI: %s** — PR #%s%s\n\nLogs: %s%s"
                % (COMMENT_MARKER, result, pr, (" at `%s`" % sha[:10]) if sha else "",
                   details or "(no public link; see the GitLab pipeline)",
                   ("\n\nJobs:\n" + jobs) if jobs else ""))
        try:
            comments = request("GET",
                "https://api.github.com/repos/%s/issues/%s/comments?per_page=100" % (repo, pr),
                None, comment_token)
            existing = next((c for c in comments
                             if c.get("body", "").startswith(COMMENT_MARKER)), None)
            if existing:
                request("PATCH",
                        "https://api.github.com/repos/%s/issues/comments/%s" % (repo, existing["id"]),
                        {"body": body}, comment_token)
                print("updated PR comment on %s#%s" % (repo, pr))
            else:
                post("https://api.github.com/repos/%s/issues/%s/comments" % (repo, pr),
                     {"body": body}, comment_token)
                print("posted PR comment on %s#%s" % (repo, pr))
        except Exception as e:
            print("PR comment failed (needs public_repo scope on public %s): %s" % (repo, e))
    elif not comment_token:
        print("GITHUB_COMMENT_TOKEN not set: no PR comment posted")


if __name__ == "__main__":
    main()
