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
import re
import sys
import urllib.error
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


PYTEST_FILE_MAX = 300000   # pytest logs: keep the whole FAILURES section
FAILED_RE = re.compile(r"^(FAILED|ERROR) \S+")
COUNTS_RE = re.compile(r"=+ (.*\b(passed|failed|error|errors|skipped)\b.*) in [0-9.]+s")


def _pytest_excerpt(lines):
    """FAILURES/ERRORS section through the end (tracebacks + short summary),
    or the last TAIL_LINES lines when the run had no failures."""
    for i, ln in enumerate(lines):
        if re.match(r"^=+ (FAILURES|ERRORS) =+\s*$", ln):
            return "".join(lines[i:])[-PYTEST_FILE_MAX:]
    return "".join(lines[-TAIL_LINES:])[-MAX_FILE:]


CRASH_RE = re.compile(
    r"Segmentation fault from GPU|Fatal Python error|Aborted|core dumped|\bKilled\b|"
    r"TIMEOUT after|no exit status recorded|OUT_OF_DEVICE_MEMORY|out of memory|"
    r"UR_RESULT_ERROR|longjmp causes uninitialized stack frame")
BUILD_ERR_RE = re.compile(
    r"\berror:|CMake Error|make(\[\d+\])?: \*\*\*|ninja: build stopped|^FAILED: |"
    r"undefined reference to|fatal error|Traceback \(most recent call last\)")
ENV_ERR_RE = re.compile(
    r"\b(error|critical)\b|Could not solve|nothing provides|cannot install|conflict",
    re.I)


def _context(lines, pattern, before=2, after=3, max_lines=80):
    """Lines matching `pattern` with a little context, merged and capped."""
    keep = set()
    for i, ln in enumerate(lines):
        if pattern.search(ln):
            keep.update(range(max(0, i - before), min(len(lines), i + after + 1)))
    out, prev = [], None
    for i in sorted(keep):
        if prev is not None and i != prev + 1:
            out.append("   ...")
        out.append(lines[i].rstrip())
        prev = i
    return out[:max_lines]


def _test_errors(lines, max_tests=40, max_e=8):
    """Per failing test (FAILURES/ERRORS blocks): the raising location and E-lines."""
    res, cur, block, in_fail = [], None, [], False
    def flush():
        if cur:
            locs = [l.rstrip() for l in block if re.match(r"^\S+\.py:\d+: ", l)]
            errs = [l.rstrip() for l in block if l.startswith("E ")][:max_e]
            res.append((cur, locs[-1] if locs else "", errs))
    for ln in lines:
        if re.match(r"^=+ (FAILURES|ERRORS) =+\s*$", ln):
            in_fail = True; continue
        if in_fail and re.match(r"^=+ .* =+\s*$", ln):      # next section: summary
            flush(); cur, block = None, []; in_fail = False; continue
        m = re.match(r"^_{3,} (.+?) _{3,}\s*$", ln) if in_fail else None
        if m:
            flush(); cur, block = m.group(1), []
        elif cur:
            block.append(ln)
    flush()
    return res[:max_tests]


def collect_logs(result, repo, pr, sha):
    """Gist files: 00_SUMMARY.md first (where/how it failed), then each log.

    GitLab here is Private-only, so this gist is the ONLY place PR viewers can
    read the logs: the summary says which stage failed and why (from
    ci_sycl.sh's 00_failure.log), lists crashes/timeouts, each failing test's
    error lines, and build/env errors with context; pytest logs keep their full
    FAILURES section, other logs their last TAIL_LINES lines.
    """
    files, failed, counts, device = {}, [], [], ""
    stages, crashes, test_errs, build_ctx, env_ctx, py_tb = [], [], [], [], [], []
    for path in sorted(glob.glob("ci-logs/**/*.log", recursive=True)):
        with open(path, errors="replace") as f:
            lines = f.readlines()
        name = os.path.basename(path)
        job = os.path.basename(os.path.dirname(path))       # build | test
        if name == "00_failure.log":
            st = rs = ""
            for ln in lines:
                if ln.startswith("stage:"):
                    st = ln.split(":", 1)[1].strip()
                elif ln.startswith("reason:"):
                    rs = ln.split(":", 1)[1].strip()
                    stages.append((job, st, rs))
            content = "".join(lines)
        elif name.startswith("60_pytest"):
            content = _pytest_excerpt(lines)
            failed += [ln.rstrip() for ln in lines if FAILED_RE.match(ln)]
            counts += ["%s: %s" % (name, m.group(1)) for m in
                       (COUNTS_RE.search(ln) for ln in lines[-50:]) if m]
            crashes += ["%s: %s" % (name, ln.strip()) for ln in lines if CRASH_RE.search(ln)]
            test_errs += [(name,) + t for t in _test_errors(lines)]
        else:
            content = "".join(lines[-TAIL_LINES:])[-MAX_FILE:]
            if name.startswith("40_build"):
                build_ctx = _context(lines, BUILD_ERR_RE)
            elif name.startswith("05_env_create"):
                ctx = _context(lines, ENV_ERR_RE, before=1, after=2, max_lines=25)
                if ctx:
                    env_ctx = ["[%s]" % name] + ctx          # last attempt wins
            elif name.startswith(("06_env_sanity", "50_smoke")) and \
                    any("Traceback" in ln or "Error" in ln for ln in lines):
                py_tb += ["[%s]" % name] + [ln.rstrip() for ln in lines[-15:]]
            if name.startswith("10_toolchain"):
                if not device:
                    device = "".join(ln for ln in lines if "Data Center GPU" in ln or
                                     "GPU devices" in ln)[:600]
                if any("AssertionError" in ln or "no SYCL GPU device" in ln for ln in lines):
                    py_tb += ["[%s]" % name] + [ln.rstrip() for ln in lines[-15:]]
        files[path.replace("/", "__")] = {"content": content or "(empty)"}

    # multi-tile: 60_pytest.log only echoes the per-tile result lines -- count
    # (and list failures) from the tile logs; single-tile: 60_pytest.log is it
    if any(c.startswith("60_pytest_tile") for c in counts):
        counts = [c for c in counts if not c.startswith("60_pytest.log:")]
    md = ["# SYCL CI: %s" % result, "",
          "- PR: %s#%s at `%s`" % (repo, pr, (sha or "")[:10]),
          "- pipeline: %s (requires a gitlab.jlse login; this gist is the public copy)"
          % os.environ.get("CI_PIPELINE_URL", "?"), ""]
    if stages:
        md += ["## Where it failed"] + ["- **%s job, %s stage**: %s" % s_ for s_ in stages] + [""]
    if device:
        md += ["## Device", "```", device.strip(), "```", ""]
    if counts:
        md += ["## Test totals", "```"] + counts + ["```", ""]
    if crashes:
        md += ["## Crashes / timeouts", "```"] + crashes[:30] + ["```", ""]
    if failed:
        md += ["## Failed tests (%d)" % len(set(failed)), "```"] + sorted(set(failed))[:200] + ["```", ""]
    if test_errs:
        md += ["## How the tests failed"]
        for fname, test, loc, errs in test_errs:
            md += ["### %s  (%s)" % (test, fname), "```"] + ([loc] if loc else []) + \
                  (errs or ["(no E-lines; see the traceback in %s)" % fname]) + ["```"]
        md += ["", "Full tracebacks: the `ci-logs__test__60_pytest*.log` files below.", ""]
    if build_ctx:
        md += ["## Build errors (with context)", "```"] + build_ctx + ["```", ""]
    if env_ctx:
        md += ["## Environment errors", "```"] + env_ctx + ["```", ""]
    if py_tb:
        md += ["## Import / sanity / device-check errors", "```"] + py_tb + ["```", ""]
    if result != "success" and not (stages or failed or crashes or build_ctx or env_ctx or py_tb):
        md += ["## No failure details recorded",
               "The job died before writing its logs (runner/scheduler problem?). "
               "Check the GitLab pipeline."]
    files["00_SUMMARY.md"] = {"content": "\n".join(md) + "\n"}
    first_stage = stages[0][1] if stages else ""
    return files, (len(set(failed)), counts, first_stage)


def status_description(result, summary_counts):
    """<=140 chars (GitHub limit), e.g. 'SYCL CI failed: 3 failed tests (Details: logs)'."""
    nfailed, _, stage = summary_counts
    if result == "success":
        d = "SYCL CI passed"
    elif result == "canceled":
        d = "SYCL CI canceled"
    elif nfailed:
        d = "SYCL CI failed: %d failed test%s" % (nfailed, "" if nfailed == 1 else "s")
    elif stage:
        d = "SYCL CI failed at the %s stage" % stage
    else:
        d = "SYCL CI failed"
    return (d + " (Details: logs)")[:140]


def main():
    result = sys.argv[1] if len(sys.argv) > 1 else "failure"
    repo, pr, sha = os.environ.get("PR_REPO"), os.environ.get("PR_NUMBER"), os.environ.get("PR_SHA")
    print("SYCL CI %s for %s PR #%s (%s)" % (result, repo, pr, (sha or "")[:10]))

    files, summary_counts = collect_logs(result, repo, pr, sha)
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
    print("tokens present: status=%s gist=%s comment=%s (ref %s, protected=%s)" % (
        bool(status_token), bool(gist_token), bool(os.environ.get("GITHUB_COMMENT_TOKEN")),
        os.environ.get("CI_COMMIT_REF_NAME"), os.environ.get("CI_COMMIT_REF_PROTECTED")))
    status_failed = False
    if status_token and repo and sha:
        try:
            post("https://api.github.com/repos/%s/statuses/%s" % (repo, sha), {
                "state": {"success": "success", "canceled": "error"}.get(result, "failure"),
                "context": "SYCL CI",
                "description": status_description(result, summary_counts),
                "target_url": details}, status_token)
            print("posted commit status on %s@%s" % (repo, sha[:10]))
        except urllib.error.HTTPError as e:
            status_failed = True
            print("ERROR: commit status POST failed: HTTP %s %s -- token needs push access on %s"
                  % (e.code, e.read()[:300], repo))
    elif not status_token:
        print("GITHUB_STATUS_TOKEN not set: no status posted to GitHub "
              "(Protected variable on unprotected ref %s?)" % os.environ.get("CI_COMMIT_REF_NAME"))
        status_failed = bool(pr) and os.environ.get("SYCL_CI_REQUIRE_STATUS", "1") == "1"

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

    if status_failed and result != "canceled":
        sys.exit("SYCL CI result was NOT published as a commit status (see above)")


if __name__ == "__main__":
    main()
