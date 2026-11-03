#!/usr/bin/env python3
"""Publish a SYCL CI result for a PR so contributors can read it without GitLab.

usage: report_pr.py success|failure     (run at the end of a PR pipeline)

Always prints a summary to the job log. Optionally, with credentials set as
masked GitLab CI/CD variables:
  GITHUB_GIST_TOKEN    token with the gist scope; logs are uploaded to a gist
                       owned by that account (readable by anyone with the link)
  GITHUB_STATUS_TOKEN  token allowed to write commit statuses on $PR_REPO;
                       posts the 'SYCL / chiatta (PVC)' status on the PR commit
                       with the gist (or this pipeline) as the details link.
"""
import glob
import json
import os
import sys
import urllib.request

TAIL_LINES = 400
MAX_FILE = 60000


def post(url, payload, token):
    req = urllib.request.Request(url, json.dumps(payload).encode(), method="POST")
    req.add_header("Authorization", "Bearer " + token)
    req.add_header("Accept", "application/vnd.github+json")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


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
        g = post("https://api.github.com/gists", {
            "description": "SYCL CI (JLSE chiatta) %s PR #%s %s" % (repo, pr, (sha or "")[:10]),
            "public": False, "files": files}, gist_token)
        details = g["html_url"]
        print("logs: " + details)

    status_token = os.environ.get("GITHUB_STATUS_TOKEN")
    if status_token and repo and sha:
        post("https://api.github.com/repos/%s/statuses/%s" % (repo, sha), {
            "state": "success" if result == "success" else "failure",
            "context": "SYCL / chiatta (PVC)",
            "description": "SYCL CI %s" % ("passed" if result == "success" else "failed"),
            "target_url": details}, status_token)
        print("posted commit status on %s@%s" % (repo, sha[:10]))
    elif not status_token:
        print("GITHUB_STATUS_TOKEN not set: no status posted to GitHub")


if __name__ == "__main__":
    main()
