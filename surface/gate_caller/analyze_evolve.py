"""Turn the two evolution histories into the paper's result table.

The claim under test: identical loops, identical episode budget, differing ONLY in whether
the incumbent-vs-candidate comparison is paired on byte-identically restored state. If the
gates reach different verdicts, the acceptance rule changes what the search finds. If they
agree throughout, pairing tightens the estimate without changing the outcome at this scale --
a reportable negative, and the honest one.
"""
from __future__ import annotations
import json, os, re, sys
import numpy as np
MCP = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
W = os.path.join(MCP, "work", "evolve3")
LOG = os.path.join(MCP, "logs")

DEC = re.compile(r"r(\d+)c(\d+) tools=(\d+) obj=([\d.]+) d=([-+][\d.]+) t=([-+][\d.]+) "
                 r"n=(\d+) (ACCEPT|reject)")

def decisions(gate):
    p = os.path.join(LOG, f"evolve3_{gate}.log")
    out = []
    if not os.path.exists(p):
        return out
    for line in open(p, errors="ignore"):
        m = DEC.search(line)
        if m:
            out.append(dict(rd=int(m.group(1)), c=int(m.group(2)), tools=int(m.group(3)),
                            obj=float(m.group(4)), delta=float(m.group(5)),
                            t=float(m.group(6)), n=int(m.group(7)),
                            accept=m.group(8) == "ACCEPT"))
    return out

def main() -> int:
    D = {g: decisions(g) for g in ("paired", "unpaired")}
    for g, d in D.items():
        print(f"  {g:9s} {len(d):2d} decisions, {sum(x['accept'] for x in d)} accepted")
    if not all(D.values()):
        print("\n  [!] incomplete -- loops still running")
        return 1
    print(f"\n  {'rd':>3s} {'cand':>4s} | {'paired':>26s} | {'unpaired':>26s} | agree")
    print("  " + "-" * 74)
    n = min(len(D["paired"]), len(D["unpaired"]))
    agree = 0
    for i in range(n):
        p, u = D["paired"][i], D["unpaired"][i]
        same = p["accept"] == u["accept"]
        agree += same
        f = lambda x: f"d={x['delta']:+.3f} t={x['t']:+5.2f} {'ACC' if x['accept'] else 'rej'}"
        print(f"  {p['rd']:3d} {p['c']:4d} | {f(p):>26s} | {f(u):>26s} | {'yes' if same else 'NO'}")
    print("  " + "-" * 74)
    print(f"  gates agree on {agree}/{n} decisions")
    # effect-estimate dispersion: the quantity pairing is supposed to shrink
    dp = np.array([x["delta"] for x in D["paired"]])
    du = np.array([x["delta"] for x in D["unpaired"]])
    print(f"\n  |delta| mean   paired {np.abs(dp).mean():.4f}   unpaired {np.abs(du).mean():.4f}")
    print(f"  |delta| sd     paired {dp.std(ddof=1):.4f}   unpaired {du.std(ddof=1):.4f}"
          f"   ratio {dp.std(ddof=1)/max(du.std(ddof=1),1e-9):.3f}")
    print(f"  |t| mean       paired {np.abs([x['t'] for x in D['paired']]).mean():.3f}"
          f"   unpaired {np.abs([x['t'] for x in D['unpaired']]).mean():.3f}")
    for g in ("paired", "unpaired"):
        h = os.path.join(W, f"history_{g}.json")
        if os.path.exists(h):
            hh = json.load(open(h))["history"]
            traj = " -> ".join(f"{r['obj']:.3f}" for r in hh)
            print(f"  {g:9s} objective trajectory: {traj}")
    out = {g: D[g] for g in D}
    json.dump(out, open(os.path.join(MCP, "surface", "receipts", "evolve_result.json"), "w"),
              indent=2)
    print(f"\n  wrote receipts/evolve_result.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())
