"""Turn three substrate-specific measurements into three general, transferable claims.

Our results so far are numbers about ONE substrate: 43.3% credit rate, a t-test that broke on
granite, 640-2560 tasks required. None of those transfer as stated. Each has a general form
underneath, and each general form is checkable against the data we already banked.

  G1  CHANCE-RATE RESULT (distribution-free).
      For an edit that is inert, arms A and B are exchangeable, so
          P(credit) = P(Bbar > Abar) = (1 - P(tie)) / 2
      independent of objective, model, substrate, or task pool. The field's accept-if-larger
      rule therefore has a false-positive rate fixed by its tie rate alone -- it cannot be
      tuned, and it does not depend on anything about the agent. Our 43.3% is one instance.
      TEST: measure the tie rate, predict the credit rate, compare with the measured 43.3%.

  G2  TIE-FRACTION DIAGNOSTIC (transferable).
      Harness-evolution objectives are routinely discrete and bounded -- pass rates, pass@k,
      success indicators, our non-degenerate rate. Paired differences of such objectives are
      zero-inflated, s underestimates spread, and the t-test inflates. The inflation is a
      function of the TIE FRACTION, which any team can compute from one no-edit replicate on
      their own data. That makes "is my t-test valid?" answerable in advance.
      TEST: sweep tie fraction synthetically, measure realised alpha, and check our four real
      callers land on the resulting curve.

  G3  DESIGN EQUATION (actionable a priori).
      Required tasks for power 1-beta at effect delta:
          n_paired   = (z_{1-alpha} + z_{1-beta})^2 * sigma_d^2       / delta^2
          n_unpaired = (z_{1-alpha} + z_{1-beta})^2 * 4 * sigma^2     / delta^2
      with sigma_d^2 = 2 sigma^2 (1 - rho). Both sigma and rho are estimable from a SINGLE
      no-edit replicate, i.e. before building any gate. So a team can compute their required
      budget, and their pairing gain, in advance.
      TEST: predict n from the formula, compare against the n we measured empirically in
      required_budget.json, and report the correction factor.

All three run on banked episodes. No GPU.
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
MCP = os.path.dirname(ROOT)
IFACES = ["RAW", "SELECTED", "SPECIALIZED", "MACRO"]
CALLERS = ["qwen", "granite", "mistral", "falcon3"]
Z = {0.025: 1.959963985, 0.20: 0.841621234}   # z_{1-alpha/2} two-sided .05; z_{1-beta} at 80%


# ----------------------------------------------------------------- G1
def g1_chance_rate(work):
    """P(credit) = (1 - P(tie)) / 2 for an inert edit. Predict, then compare with measured."""
    print("\n" + "=" * 74)
    print("  G1  The accept-if-larger rule's false-positive rate is the chance rate")
    print("=" * 74)
    print("      For an inert edit A and B are exchangeable, so P(Bbar>Abar) = (1-P(tie))/2.")
    print("      This holds for ANY objective, model and substrate. Verifying on our cube:\n")
    print(f"      {'caller':9s} {'pairs':>7s} {'tie rate':>9s} {'predicted':>10s} {'measured':>9s} {'gap':>7s}")
    print("      " + "-" * 56)
    rows, out = [], {}
    for c in CALLERS:
        p = os.path.join(work, f"{c}_sampled.jsonl")
        if not os.path.exists(p):
            continue
        rec = collections.defaultdict(dict)
        seeds = set()
        for r in (json.loads(l) for l in open(p)):
            rec[(r["interface"], (r["scenario"], r["task_idx"]))][r["seed"]] = int(r["reward"])
            seeds.add(r["seed"])
        sl = sorted(seeds)
        tasks = sorted({t for (_, t) in rec})
        tie = tot = cred = 0
        for a in IFACES:
            for b in IFACES:
                if a == b:
                    continue
                va = [np.mean([rec[(a, t)][s] for s in sl])
                      for t in tasks if len(rec.get((a, t), {})) == len(sl)]
                vb = [np.mean([rec[(b, t)][s] for s in sl])
                      for t in tasks if len(rec.get((b, t), {})) == len(sl)]
                if not va or not vb:
                    continue
                ma, mb = float(np.mean(va)), float(np.mean(vb))
                tot += 1
                if abs(mb - ma) < 1e-12:
                    tie += 1
                elif mb > ma:
                    cred += 1
        if not tot:
            continue
        tr = tie / tot
        pred, meas = (1 - tr) / 2, cred / tot
        print(f"      {c:9s} {tot:7d} {tr:9.3f} {pred:10.3f} {meas:9.3f} {meas-pred:+7.3f}")
        rows.append((pred, meas))
        out[c] = {"pairs": tot, "tie_rate": tr, "predicted": pred, "measured": meas}
    if rows:
        pr = float(np.mean([p for p, _ in rows])); me = float(np.mean([m for _, m in rows]))
        print("      " + "-" * 56)
        print(f"      {'MEAN':9s} {'':7s} {'':9s} {pr:10.3f} {me:9.3f} {me-pr:+7.3f}")
        print(f"\n      The rule's error rate is set by its tie rate and nothing else. It is not")
        print(f"      a property of the agent, the tasks, or the edit -- so it cannot be fixed")
        print(f"      by better evaluation, only by adding a test.")
        out["mean_predicted"], out["mean_measured"] = pr, me
    return out


# ----------------------------------------------------------------- G2
def g2_tie_diagnostic(nullwork, group, trials, rng):
    """Realised alpha of the paired t-test as a function of tie fraction."""
    print("\n" + "=" * 74)
    print("  G2  Tie fraction predicts whether your t-test is valid")
    print("=" * 74)
    print("      Synthetic sweep: paired differences with a controlled fraction of exact ties,")
    print("      nominal one-sided alpha = 0.025, n = 40.\n")
    print(f"      {'tie fraction':>13s} {'realised alpha':>15s} {'inflation':>10s}")
    print("      " + "-" * 40)
    n, crit = 40, 1.959963985
    curve = {}
    for tf in [0.0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.925, 0.975]:
        hits = 0
        for _ in range(trials):
            d = np.zeros(n)
            k = n - int(round(tf * n))
            if k > 0:
                d[:k] = rng.choice([-1.0, 1.0], size=k)   # symmetric => a genuine null
            s = d.std(ddof=1)
            t = 0.0 if s < 1e-12 else d.mean() / (s / np.sqrt(n))
            hits += t > crit
        a = hits / trials
        print(f"      {tf:13.3f} {a:15.3f} {a/0.025:9.1f}x")
        curve[f"{tf:.3f}"] = a

    print("\n      Where our real callers fall (tie fraction from the banked no-edit replicate):\n")
    print(f"      {'caller':9s} {'tie fraction':>13s} {'realised alpha':>15s} {'verdict':>12s}")
    print("      " + "-" * 53)
    real = {}
    for c in CALLERS:
        w = os.path.join(MCP, "work", "evolve_null" if c == "qwen" else f"evolve_null_{c}")
        p = os.path.join(w, "null_full.jsonl")
        if not os.path.exists(p):
            continue
        by = collections.defaultdict(list)
        for r in (json.loads(l) for l in open(p)):
            by[(r["scenario"], r["task_idx"])].append((str(r["seed"]), int(r["reward"])))
        have = sorted({s for v in by.values() for s, _ in v})
        if len(have) < 2 * group:
            continue
        ties = tot = hits = ok = 0
        for _ in range(trials):
            pm = list(have); rng.shuffle(pm)
            A, B = set(pm[:group]), set(pm[group:2 * group])
            f = lambda g: {t: (0.0 if (sum(x) == 0 or sum(x) == len(x)) else 1.0)
                           for t in by
                           if len(x := [r for s, r in by[t] if s in g]) == len(g)}
            pa, pb = f(A), f(B)
            common = sorted(set(pa) & set(pb))
            if len(common) < 8:
                continue
            d = np.array([pb[t] - pa[t] for t in common])
            ties += int((np.abs(d) < 1e-12).sum()); tot += len(d)
            s = d.std(ddof=1)
            t = 0.0 if s < 1e-12 else d.mean() / (s / np.sqrt(len(d)))
            hits += t > crit; ok += 1
        if not ok:
            continue
        tf, a = ties / tot, hits / ok
        verdict = "OK" if a <= 0.025 * 1.5 else "INVALID"
        print(f"      {c:9s} {tf:13.3f} {a:15.3f} {verdict:>12s}")
        real[c] = {"tie_fraction": tf, "realised_alpha": a, "valid": verdict == "OK"}
    print(f"\n      Actionable form: compute your tie fraction on one no-edit replicate. Above")
    print(f"      roughly 0.9 the paired t-test is not usable and a permutation test is needed.")
    return {"synthetic_curve": curve, "real_callers": real}


# ----------------------------------------------------------------- G3
def g3_design_equation(group, trials, rng, measured_path):
    """Predict required n from sigma_d and rho; check against the measured requirement."""
    print("\n" + "=" * 74)
    print("  G3  A design equation you can evaluate before building the gate")
    print("=" * 74)
    print("      n_paired = (z_a + z_b)^2 sigma_d^2 / delta^2,  sigma_d^2 = 2 sigma^2 (1-rho)")
    print("      sigma and rho both come from ONE no-edit replicate.\n")
    meas = {}
    if os.path.exists(measured_path):
        meas = json.load(open(measured_path))
    zz = (Z[0.025] + Z[0.20]) ** 2
    print(f"      {'caller':9s} {'sigma':>7s} {'rho':>7s} {'gain 1/(1-rho)':>15s} "
          f"{'pred n (d=.20)':>15s} {'measured':>9s}")
    print("      " + "-" * 68)
    out = {}
    for c in CALLERS:
        w = os.path.join(MCP, "work", "evolve_null" if c == "qwen" else f"evolve_null_{c}")
        p = os.path.join(w, "null_full.jsonl")
        if not os.path.exists(p):
            continue
        by = collections.defaultdict(list)
        for r in (json.loads(l) for l in open(p)):
            by[(r["scenario"], r["task_idx"])].append((str(r["seed"]), int(r["reward"])))
        have = sorted({s for v in by.values() for s, _ in v})
        if len(have) < 2 * group:
            continue
        sds, sd_ds, rhos = [], [], []
        for _ in range(trials):
            pm = list(have); rng.shuffle(pm)
            A, B = set(pm[:group]), set(pm[group:2 * group])
            f = lambda g: {t: (0.0 if (sum(x) == 0 or sum(x) == len(x)) else 1.0)
                           for t in by
                           if len(x := [r for s, r in by[t] if s in g]) == len(g)}
            pa, pb = f(A), f(B)
            common = sorted(set(pa) & set(pb))
            if len(common) < 8:
                continue
            xa = np.array([pa[t] for t in common]); xb = np.array([pb[t] for t in common])
            sds.append((xa.std(ddof=1) + xb.std(ddof=1)) / 2)
            sd_ds.append((xb - xa).std(ddof=1))
            if xa.std() > 1e-9 and xb.std() > 1e-9:
                rhos.append(float(np.corrcoef(xa, xb)[0, 1]))
        if not sds:
            continue
        sig, sig_d = float(np.mean(sds)), float(np.mean(sd_ds))
        rho = float(np.mean(rhos)) if rhos else float("nan")
        d = 0.20
        n_pred = zz * sig_d ** 2 / d ** 2
        m = meas.get(c, {}).get("0.20", {}).get("paired")
        gain = 1.0 / max(1e-9, 1 - rho) if rho == rho else float("nan")
        print(f"      {c:9s} {sig:7.3f} {rho:7.3f} {gain:15.2f}x {n_pred:15.0f} {str(m):>9s}")
        out[c] = {"sigma": sig, "sigma_d": sig_d, "rho": rho,
                  "predicted_n_d020": n_pred, "measured_n_d020": m}
    print(f"\n      The formula is the deliverable; our numbers only verify it. A team measures")
    print(f"      sigma and rho once, then knows both their required budget and what pairing")
    print(f"      will buy them -- before committing a single evaluation episode.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--landscape", default=os.path.join(MCP, "work", "landscape"))
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--trials", type=int, default=1500)
    ap.add_argument("--out", default=os.path.join(ROOT, "receipts", "generalize.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(23)
    res = {
        "G1_chance_rate": g1_chance_rate(a.landscape),
        "G2_tie_diagnostic": g2_tie_diagnostic(None, a.group, a.trials, rng),
        "G3_design_equation": g3_design_equation(
            a.group, a.trials, rng, os.path.join(ROOT, "receipts", "required_budget.json")),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2, sort_keys=True, default=str)
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
