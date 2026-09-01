"""Measure rho, the correlation pairing exploits -- from banked episodes, no new rollouts.

Relative efficiency of a paired (crossover / common-random-number) comparison versus an
unpaired one is exactly 1 - rho, where rho = Corr(X_A, X_B) over shared (task, seed) units.
So the variance ratio, and hence the episode budget needed for equal power, follows directly.

Var(unpaired delta) = 2 sigma^2 / n
Var(paired   delta) = 2 sigma^2 (1 - rho) / n

Estimated here two ways on the G-LANDSCAPE cube (4 capability-equivalent interfaces x 4
callers x 39 tasks x 8 seeds): directly as the correlation of paired outcomes, and via the
variance ratio of the paired vs unpaired difference estimator. They should agree.
"""
import json, collections, itertools, os
import numpy as np
W = os.path.join(os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "work", "landscape")
IF = ["RAW", "SELECTED", "SPECIALIZED", "MACRO"]
CAL = ["qwen", "mistral", "granite", "falcon3"]

print(f"  {'caller':9s} {'rho(direct)':>12s} {'var ratio':>10s} {'1-rho':>7s} "
      f"{'episodes saved':>15s}")
print("  " + "-" * 58)
rows = []
for c in CAL:
    p = os.path.join(W, f"{c}_sampled.jsonl")
    if not os.path.exists(p):
        continue
    rec = collections.defaultdict(dict)
    for r in (json.loads(l) for l in open(p)):
        rec[(r["scenario"], r["task_idx"], r["seed"])][r["interface"]] = int(r["reward"])
    units = [v for v in rec.values() if len(v) == len(IF)]
    if len(units) < 50:
        continue
    rhos, ratios = [], []
    for a, b in itertools.combinations(IF, 2):
        xa = np.array([u[a] for u in units], float)
        xb = np.array([u[b] for u in units], float)
        if xa.std() < 1e-9 or xb.std() < 1e-9:
            continue
        rhos.append(float(np.corrcoef(xa, xb)[0, 1]))
        # paired: variance of the per-unit difference. unpaired: sum of marginal variances.
        ratios.append(float(np.var(xa - xb, ddof=1) / (np.var(xa, ddof=1) + np.var(xb, ddof=1))))
    if not rhos:
        continue
    rho, ratio = float(np.mean(rhos)), float(np.mean(ratios))
    print(f"  {c:9s} {rho:12.3f} {ratio:10.3f} {1-rho:7.3f} {1/max(ratio,1e-6):14.2f}x")
    rows.append((rho, ratio))
if rows:
    mr = float(np.mean([r for r, _ in rows])); mv = float(np.mean([v for _, v in rows]))
    print("  " + "-" * 58)
    print(f"  {'MEAN':9s} {mr:12.3f} {mv:10.3f} {1-mr:7.3f} {1/max(mv,1e-6):14.2f}x")
    print(f"\n  Pairing removes {100*mr:.0f}% of the difference-estimator variance,")
    print(f"  so equal statistical power needs ~{1/max(mv,1e-6):.1f}x fewer episodes.")
    print(f"  (pre-registered prediction from the positioning brief was rho 0.4-0.8)")
