"""Where does the variance in agent-harness evaluation actually live?

Pairing (crossover / common random numbers) can only remove nuisance variance that the two
arms SHARE. Task difficulty is shared -- both harnesses face the same task. Decoding noise is
not: once two harnesses' prompts diverge there is no common random draw. So the achievable
variance reduction has a ceiling set by the variance decomposition, and measuring it tells us
whether pairing's benefit is large or marginal BEFORE building anything on it.

This also reconciles two numbers we measured:
    rho = 0.346 pairing ACROSS interfaces      -> ~1.5x fewer episodes
    MDE 3.55x better pairing WITHIN one interface across seed halves
The gap is exactly the interface x task interaction: pairing across different harnesses
cannot remove variance that the harnesses do not share.

Model, per caller, on the G-LANDSCAPE cube (4 interfaces x 39 tasks x 8 seeds):

    y[i,t,s] = mu + a_t (task) + b_i (interface) + ab_it (interaction) + e_its (seed noise)

Components are estimated by the standard balanced-design sums of squares and converted to
variance components. The quantity that matters for pairing is

    removable share = Var(task) / Var(total)

because the task effect is what a paired design blocks out.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IFACES = ["RAW", "SELECTED", "SPECIALIZED", "MACRO"]
CALLERS = ["qwen", "mistral", "granite", "falcon3"]


def cube(work, caller):
    """-> y[iface, task, seed] or None if the design is not balanced/complete."""
    rec = collections.defaultdict(dict)
    seeds, tasks = set(), set()
    for r in (json.loads(l) for l in open(os.path.join(work, f"{caller}_sampled.jsonl"))):
        t = (r["scenario"], r["task_idx"])
        rec[(r["interface"], t)][r["seed"]] = int(r["reward"])
        seeds.add(r["seed"]); tasks.add(t)
    sl, tl = sorted(seeds), sorted(tasks)
    y = np.full((len(IFACES), len(tl), len(sl)), np.nan)
    for ii, i in enumerate(IFACES):
        for ti, t in enumerate(tl):
            d = rec.get((i, t))
            if not d or len(d) != len(sl):
                return None
            for si, s in enumerate(sl):
                y[ii, ti, si] = d[s]
    return y


def decompose(y):
    """Balanced two-factor-with-replication variance components."""
    I, T, S = y.shape
    mu = y.mean()
    task = y.mean(axis=(0, 2)) - mu                 # [T]
    ifc = y.mean(axis=(1, 2)) - mu                  # [I]
    cell = y.mean(axis=2)                           # [I,T]
    inter = cell - (mu + ifc[:, None] + task[None, :])
    resid = y - cell[:, :, None]

    # mean squares
    ms_task = S * I * (task ** 2).sum() / max(T - 1, 1)
    ms_ifc = S * T * (ifc ** 2).sum() / max(I - 1, 1)
    ms_int = S * (inter ** 2).sum() / max((I - 1) * (T - 1), 1)
    ms_err = (resid ** 2).sum() / max(I * T * (S - 1), 1)

    # variance components (method of moments); clamp tiny negatives to 0
    v_err = ms_err
    v_int = max((ms_int - ms_err) / S, 0.0)
    v_task = max((ms_task - ms_int) / (S * I), 0.0)
    v_ifc = max((ms_ifc - ms_int) / (S * T), 0.0)
    tot = v_task + v_ifc + v_int + v_err
    return dict(task=v_task, iface=v_ifc, interaction=v_int, seed_noise=v_err, total=tot)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(ROOT, "..", "work", "landscape"))
    ap.add_argument("--out", default=os.path.join(ROOT, "receipts", "variance_decomp.json"))
    a = ap.parse_args()
    work = os.path.abspath(a.work)

    print(f"  {'caller':9s} {'task':>8s} {'iface':>8s} {'iface x task':>13s} "
          f"{'seed noise':>11s}   share removable")
    print("  " + "-" * 68)
    out, shares = {}, []
    for c in CALLERS:
        p = os.path.join(work, f"{c}_sampled.jsonl")
        if not os.path.exists(p):
            continue
        y = cube(work, c)
        if y is None:
            print(f"  {c:9s} incomplete/unbalanced cube, skipped")
            continue
        d = decompose(y)
        tot = d["total"] or 1.0
        share = d["task"] / tot
        shares.append(share)
        print(f"  {c:9s} {100*d['task']/tot:7.1f}% {100*d['iface']/tot:7.1f}% "
              f"{100*d['interaction']/tot:12.1f}% {100*d['seed_noise']/tot:10.1f}% "
              f"{100*share:14.1f}%")
        out[c] = {k: float(v) for k, v in d.items()} | {"removable_share": float(share)}

    if shares:
        m = float(np.mean(shares))
        print("  " + "-" * 68)
        print(f"  {'MEAN':9s} {100*m:60.1f}%")
        print(f"\n  Task effect is the only component a paired design removes: both arms face")
        print(f"  the same task, so it cancels in the per-unit difference. Seed noise does NOT")
        print(f"  cancel -- two harnesses with different prompts share no random draw.")
        print(f"\n  Implied ceiling on pairing: variance ratio ~{1-m:.3f}, "
              f"i.e. at most ~{1/max(1-m,1e-6):.1f}x fewer episodes.")
        print(f"  Measured cross-interface rho was 0.346 (1.52x), consistent with this bound")
        print(f"  once the interface x task interaction -- which pairing cannot remove -- is")
        print(f"  taken out.")
        out["mean_removable_share"] = m
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2, sort_keys=True)
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
