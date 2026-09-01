"""In-loop false-positive rate: how often does each acceptance gate credit a PROVABLY INERT edit?

The live paired-vs-unpaired loop (receipts/evolve_result.json) had both gates agree on 8/8
decisions and accept nothing. That is the easy direction: every candidate there was a single
tool removal that was neutral-to-harmful, and both gates correctly reject harmful edits. The
claim actually under test is about ACCEPTING NOISE, and that loop contained no null edit to
accept.

This measures it directly. Evaluate ONE fixed advertised tool set across many seeds, then form
comparisons in which the two arms differ only by which seeds they drew. The "edit" is inert by
construction -- same tools, same tasks, same model -- so every accept is a false positive.
Both gates see the identical episodes; only the estimator differs.

This is the retrospective 43.1% redeal-null number (receipts/harness_attribution.json) restated
in the loop's own currency: same objective (non-degenerate group rate), same 4-seed groups,
same one-sided t > 1.96 acceptance rule as evolve_inproc.py.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MCP = os.path.dirname(ROOT)
SURF = os.path.join(ROOT, "gate_surface", "work", "surfaces")
sys.path.insert(0, os.path.join(ROOT, "gate_surface"))
sys.path.insert(0, HERE)


def scenario_tools(scen):
    return [t["name"] for t in json.load(open(os.path.join(SURF, f"{scen}.json"))).get("tools", [])]


def per_task_nondegenerate(by, tasks, seed_group):
    """Loop's objective, restricted to one seed group: 1.0 if the group can yield gradient."""
    out = {}
    for t in tasks:
        v = [r for s, r in by.get(t, []) if s in seed_group]
        if len(v) == len(seed_group):
            out[t] = 0.0 if (sum(v) == 0 or sum(v) == len(v)) else 1.0
    return out


def paired_t(base, cand):
    common = sorted(set(base) & set(cand))
    if len(common) < 8:
        return None
    d = np.array([cand[t] - base[t] for t in common], float)
    s = d.std(ddof=1)
    return float(d.mean()), (0.0 if s < 1e-12 else float(d.mean() / (s / np.sqrt(len(d))))), len(common)


def unpaired_t(base, cand, rng):
    common = sorted(set(base) & set(cand))
    if len(common) < 16:
        return None
    sh = list(common); rng.shuffle(sh); h = len(sh) // 2
    a = np.array([base[t] for t in sh[:h]], float)
    b = np.array([cand[t] for t in sh[h:]], float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    den = np.sqrt(va + vb)
    return float(b.mean() - a.mean()), (0.0 if den < 1e-12 else float((b.mean() - a.mean()) / den)), len(common)


def perm_paired(base, cand, rng, B=999):
    """Exact-style sign-flip test on per-task differences. One-sided upper-tail p.

    The t-test assumes the differences are roughly normal. Ours are binary-derived and live in
    {-1,0,+1}, mostly 0: a task that is non-degenerate under both arms contributes an exact
    zero. With n~40 and only a handful of non-zero entries, s underestimates the spread and t
    inflates. Measured consequence: on granite the paired t-test's false-positive rate is 0.045
    against a 0.025 nominal rate -- WORSE than the unpaired t it was supposed to improve on.
    Under the null the differences are exchangeable in sign, so flipping signs at random gives
    a reference distribution that needs no normality assumption and is valid for any n.
    """
    common = sorted(set(base) & set(cand))
    if len(common) < 8:
        return None
    d = np.array([cand[t] - base[t] for t in common], float)
    obs = d.mean()
    signs = rng.choice([-1.0, 1.0], size=(B, len(d)))
    null = (signs * d).mean(axis=1)
    return float(obs), float(((null >= obs).sum() + 1) / (B + 1)), len(common)


def perm_unpaired(base, cand, rng, B=999):
    """Label-permutation two-sample test on disjoint task halves. One-sided upper-tail p."""
    common = sorted(set(base) & set(cand))
    if len(common) < 16:
        return None
    sh = list(common); rng.shuffle(sh); h = len(sh) // 2
    a = np.array([base[t] for t in sh[:h]], float)
    b = np.array([cand[t] for t in sh[h:]], float)
    obs = b.mean() - a.mean()
    pool = np.concatenate([a, b])
    idx = np.argsort(rng.random((B, len(pool))), axis=1)
    sh_pool = pool[idx]
    null = sh_pool[:, len(a):].mean(axis=1) - sh_pool[:, :len(a)].mean(axis=1)
    return float(obs), float(((null >= obs).sum() + 1) / (B + 1)), len(common)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=os.path.join(HERE, "pools", "pool_bandenriched.json"))
    ap.add_argument("--seeds", type=int, default=16, help="total seeds to evaluate")
    ap.add_argument("--group", type=int, default=4, help="seeds per arm, matching the loop")
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--perms", type=int, default=999)
    ap.add_argument("--crit", type=float, default=1.96)
    ap.add_argument("--mem-fraction", type=float, default=0.63)
    ap.add_argument("--work", default=os.path.join(MCP, "work", "evolve_null"))
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "receipts", "evolve_null.json"))
    a = ap.parse_args()

    os.makedirs(a.work, exist_ok=True)
    pool = json.load(open(a.pool))
    tasks = sorted({(p["scenario"], p["task_idx"]) for p in pool})
    by_scen = collections.defaultdict(list)
    for s, i in tasks:
        by_scen[s].append(i)
    full = sorted({t for s in by_scen for t in scenario_tools(s)})
    seeds = [str(700 + i) for i in range(a.seeds)]
    out_jsonl = os.path.join(a.work, "null_full.jsonl")

    if not a.analyze_only:
        import parse_multi, policy_vllm as policy_mod, run_gate
        fam = policy_mod.family()
        if run_gate.MAX_TURNS < 20 or run_gate.MAX_NEW_TOKENS < 1024:
            raise SystemExit("caps not raised: need GATE_MAX_TURNS=20 GATE_MAX_NEW_TOKENS=1024")
        run_gate.SYSTEM = parse_multi.system_for(fam)
        _f = run_gate.parse_call
        run_gate.parse_call = lambda t, _g=_f: (_g(t) if _g(t) is not None else parse_multi._fallback(t))
        _r = policy_mod.render
        policy_mod.render = lambda m, t, _q=_r, _fm=fam: _q(parse_multi.normalise_history(m, _fm), t)
        sys.modules["policy"] = policy_mod
        run_gate.TEMPERATURE = 0.7
        os.environ["GATE_TOOL_SUBSET"] = ",".join(sorted(full))
        print(f"[null] {len(tasks)} tasks, {len(by_scen)} scenarios, {len(full)} tools, "
              f"{len(seeds)} seeds -> {len(tasks)*len(seeds)} episodes", flush=True)
        banked = collections.Counter()
        if os.path.exists(out_jsonl):
            for r in (json.loads(l) for l in open(out_jsonl)):
                banked[(r["scenario"], str(r["task_idx"]), str(r["seed"]))] += 1
        for sc, idxs in by_scen.items():
            idxs = [i for i in idxs if any(banked[(sc, str(i), s)] == 0 for s in seeds)]
            if not idxs:
                continue
            argv = ["run_gate", "--scenarios", sc, "--tasks", ",".join(str(i) for i in sorted(idxs)),
                    "--seeds", *seeds, "--interfaces", "RAW", "--gpu", "0",
                    "--slots", str(int(os.environ.get("NULL_SLOTS", "16"))), "--micro", str(int(os.environ.get("NULL_SLOTS", "16"))), "--mem-fraction", str(a.mem_fraction),
                    "--out", out_jsonl, "--resume-from", out_jsonl]
            old = sys.argv; sys.argv = argv
            try:
                run_gate.main()
            except SystemExit:
                pass
            except Exception as e:
                print(f"[null] {sc} FAILED {type(e).__name__}: {e}", flush=True)
            finally:
                sys.argv = old

    by = collections.defaultdict(list)
    for r in (json.loads(l) for l in open(out_jsonl)):
        by[(r["scenario"], r["task_idx"])].append((str(r["seed"]), int(r["reward"])))
    have = sorted({s for v in by.values() for s, _ in v})
    print(f"[null] {sum(len(v) for v in by.values())} episodes, {len(by)} tasks, "
          f"{len(have)} seeds present", flush=True)
    if len(have) < 2 * a.group:
        print(f"[null] need >= {2*a.group} seeds, have {len(have)}", flush=True)
        return 1

    rng = np.random.default_rng(11)
    ARMS = ("paired", "unpaired", "paired_perm", "unpaired_perm")
    acc = {k: 0 for k in ARMS}
    two = {k: 0 for k in ARMS}
    dl = {k: [] for k in ARMS}
    n = 0
    for _ in range(a.trials):
        perm = list(have); rng.shuffle(perm)
        A, B = set(perm[:a.group]), set(perm[a.group:2 * a.group])
        pa = per_task_nondegenerate(by, tasks, A)
        pb = per_task_nondegenerate(by, tasks, B)
        rp, ru = paired_t(pa, pb), unpaired_t(pa, pb, rng)
        qp, qu = perm_paired(pa, pb, rng, a.perms), perm_unpaired(pa, pb, rng, a.perms)
        if rp is None or ru is None or qp is None or qu is None:
            continue
        n += 1
        for k, (d, t, _) in (("paired", rp), ("unpaired", ru)):
            dl[k].append(abs(d))
            acc[k] += t > a.crit           # the loop's one-sided rule
            two[k] += abs(t) > a.crit      # two-sided, for calibration
        # permutation arms: reject when the one-sided p falls below the same nominal rate the
        # t-rule targets (0.025 one-sided / 0.05 two-sided), so the columns are comparable.
        for k, (d, p, _) in (("paired_perm", qp), ("unpaired_perm", qu)):
            dl[k].append(abs(d))
            acc[k] += p <= 0.025
            two[k] += p <= 0.05

    print(f"\n  in-loop NULL: the two arms differ only by which seeds they drew.")
    print(f"  Every accept is a false positive by construction. {n} trials, "
          f"{a.perms} permutations.\n")
    print(f"  {'gate':16s} {'accepts @.025':>14s} {'@.05':>8s} {'mean |delta|':>13s}")
    print("  " + "-" * 55)
    res = {"trials": n, "group": a.group, "seeds_present": len(have), "crit": a.crit,
           "perms": a.perms}
    for k in ARMS:
        fp, fp2, md = acc[k] / n, two[k] / n, float(np.mean(dl[k]))
        flag = "" if fp <= 0.025 else "   <- anti-conservative"
        print(f"  {k:16s} {fp:14.3f} {fp2:8.3f} {md:13.4f}{flag}")
        res[k] = {"fp_one_sided": fp, "fp_two_sided": fp2, "mean_abs_delta": md}
    print(f"\n  Nominal: 0.025 one-sided / 0.05 two-sided.")
    print(f"  The t-test assumes normal differences; ours are binary-derived, in {{-1,0,+1}} and")
    print(f"  mostly 0, so s underestimates spread and t inflates. The permutation arms make no")
    print(f"  distributional assumption and are the calibrated comparison.")
    print(f"  Estimator tightness (mean |delta|, paired/unpaired): "
          f"{res['paired']['mean_abs_delta']/max(res['unpaired']['mean_abs_delta'],1e-9):.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2, sort_keys=True)
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
