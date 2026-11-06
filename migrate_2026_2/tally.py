import re,sys,glob
t=dict(passed=0,failed=0,skipped=0,xfailed=0,xpassed=0,errors=0,deselected=0)
for f in glob.glob(sys.argv[1]+"/*.log"):
    last=[l for l in open(f,errors="replace").read().splitlines() if l.strip()][-1:]
    for n,k in re.findall(r"(\d+) (passed|failed|skipped|xfailed|xpassed|errors?|deselected)",last[0] if last else ""):
        t["errors" if k.startswith("error") else k]+=int(n)
print(sys.argv[1].split("/")[-1], t)
