"""Every number the co-adaptation section reports, regenerated from banked episodes.

WHY A SCRIPT AND NOT A NOTE. These numbers move as cells finish and arms advance, and a table
hand-copied at 4am is a table that silently goes stale. One command rebuilds all of it, so the
paper always quotes what is actually on disk.

Sections:
  1. the held-out 2x2            policy x harness, with exact conditional tests
  2. dose response               held-out tasks and novel scenarios
  3. handicap-seed replicates    is the ceiling an artifact of one withheld draw
  4. gate decisions              what each arm's gate accepted, and at what cost
  5. gradient availability       why the policy half is starved, and whether editing feeds it
"""

from __future__ import annotations

import collections
import glob
import json
import math
import os

import numpy as np

R = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
E = f"{R}/work/coadapt_eval_coadapt"


def load(path):
    """Per-task mean reward, deduplicated by (scenario, task, seed).

    Concurrent writers produced duplicate and truncated lines once; first-write-wins plus a
    tolerant parser keeps a corrupted line from taking a whole cell down.
    """
    seen, bad = {}, 0
    if not os.path.exists(path):
        return {}, 0
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            bad += 1
            continue
        seen.setdefault((r["scenario"], str(r["task_idx"]), str(r["seed"])), int(r["reward"]))
    by = collections.defaultdict(list)
    for (s, t, _), v in seen.items():
        by[(s, t)].append(v)
    return {k: float(np.mean(v)) for k, v in by.items()}, bad


def p_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(b, n + 1)) / (2 ** n)


def contrast(x, y, label):
    both = sorted(set(x) & set(y))
    if not both:
        return f"    {label}: no shared tasks"
    d = np.array([y[t] - x[t] for t in both])
    b = int((d > 0.5).sum())
    c = int((d < -0.5).sum())
    return (f"    {label}: {d.mean():+.4f}  n={len(both)}  "
            f"({b} better / {c} worse, exact p={p_exact(b, c):.3g})")


def section_2x2():
    print("\n1. HELD-OUT 2x2  (295 tasks, disjoint from the 320 trained on, same 158 scenarios)")
    cells = {k: load(f"{E}/cell_{k}.jsonl")[0] for k in "ABCD"}
    if not all(cells.values()):
        print("    incomplete:", {k: len(v) for k, v in cells.items()})
        return
    names = {"A": "base    x handicapped", "B": "base    x certified",
             "C": "trained x handicapped", "D": "trained x certified"}
    for k in "ABCD":
        print(f"    {names[k]}: {np.mean(list(cells[k].values())):.4f}  ({len(cells[k])} tasks)")
    print(contrast(cells["A"], cells["B"], "harness effect, base policy    (B-A)"))
    print(contrast(cells["C"], cells["D"], "harness effect, trained policy (D-C)"))
    print(contrast(cells["A"], cells["C"], "policy  effect, handicapped    (C-A)"))
    print(contrast(cells["B"], cells["D"], "policy  effect, certified      (D-B)"))
    both = sorted(set.intersection(*[set(c) for c in cells.values()]))
    if both:
        inter = np.mean([(cells["D"][t] - cells["C"][t]) - (cells["B"][t] - cells["A"][t])
                         for t in both])
        print(f"    interaction (D-C)-(B-A): {inter:+.4f} over {len(both)} tasks")


def section_dose():
    for title, base, pts, ceil in (
        ("2a. DOSE RESPONSE, held-out tasks", "cell_A",
         [("39%", "cell_DOSE700"), ("78%", "cell_DOSE1400")], "cell_CEIL"),
        ("2b. DOSE RESPONSE, novel scenarios (60 scenarios, disjoint tool universe)",
         "cell_NOVEL_INIT",
         [("39%", "cell_NOVEL_DOSE39"), ("60%", "cell_NOVEL_DOSE60"),
          ("78%", "cell_NOVEL_DOSE78")], "cell_NOVEL_FULL"),
    ):
        print(f"\n{title}")
        a, _ = load(f"{E}/{base}.jsonl")
        top, _ = load(f"{E}/{ceil}.jsonl")
        if not a or not top:
            print("    incomplete")
            continue
        common = sorted(set(a) & set(top))
        full = np.mean([top[t] - a[t] for t in common])
        rows = [("0%", a)] + [(lbl, load(f"{E}/{f}.jsonl")[0]) for lbl, f in pts] + [("100%", top)]
        for lbl, cell in rows:
            if not cell:
                print(f"    {lbl:>5}: (missing)")
                continue
            sub = sorted(set(common) & set(cell))
            d = np.array([cell[t] - a[t] for t in sub])
            b = int((d > 0.5).sum())
            c = int((d < -0.5).sum())
            pct = 100 * d.mean() / full if full else float("nan")
            tail = "" if lbl == "0%" else f"  p={p_exact(b, c):.3g}"
            print(f"    {lbl:>5}: {np.mean([cell[t] for t in sub]):.4f}  "
                  f"gain {d.mean():+.4f}  ({pct:3.0f}% of ceiling){tail}")


def section_seeds():
    print("\n3. HANDICAP-SEED REPLICATES  (does the ceiling depend on which 40% was withheld)")
    top, _ = load(f"{E}/cell_CEIL.jsonl")
    for lbl, f in (("held-out seed 555", "cell_A"), ("held-out seed 999", "cell_HELDOUT_S999")):
        a, _ = load(f"{E}/{f}.jsonl")
        if a and top:
            print(contrast(a, top, lbl))
    nf, _ = load(f"{E}/cell_NOVEL_FULL.jsonl")
    for lbl, f in (("novel    seed 555", "cell_NOVEL_INIT"), ("novel    seed 777", "cell_NOVEL_S777")):
        a, _ = load(f"{E}/{f}.jsonl")
        if a and nf:
            print(contrast(a, nf, lbl))


def section_gates():
    print("\n4. GATE DECISIONS IN-LOOP  (what each arm certified, and from how many tasks)")
    for tag in ("coadapt", "coadapt2", "coadapt3", "coadaptn"):
        log = f"{R}/logs/rl_{tag}.log"
        if not os.path.exists(log):
            continue
        hits = []
        for line in open(log, errors="ignore"):
            if "] r" in line and ("ACCEPT" in line or "reject" in line):
                hits.append(line.strip()[-120:])
        surf = f"{R}/work/coadapt_{tag}/advertised.txt"
        n = sum(1 for _ in open(surf)) if os.path.exists(surf) else 0
        print(f"    {tag}: surface now {n} names, {len(hits)} decisions")
        for h in hits[-3:]:
            print(f"       {h}")


def section_gradient():
    print("\n5. GRADIENT AVAILABILITY  (fraction of GRPO groups that are not degenerate)")
    print("    A group whose rollouts all share one reward yields no gradient. This is why the")
    print("    policy half is starved on a handicapped surface, and what harness edits should fix.")
    for tag in ("coadapt", "coadapt2", "coadapt3", "coadaptn", "fixbase"):
        rows = []
        for f in glob.glob(f"{R}/work/verl/run_{tag}/*.jsonl"):
            for line in open(f, errors="ignore"):
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        if not rows:
            continue
        by = collections.defaultdict(list)
        for r in rows:
            by[(r["scenario"], r["task_idx"])].append(int(r["reward"]))
        multi = [v for v in by.values() if len(v) >= 2]
        mixed = [v for v in multi if len(set(v)) > 1]
        solved = [v for v in by.values() if max(v) > 0]
        avail = 100 * len(mixed) / len(multi) if multi else float("nan")
        print(f"    {tag:9s} n={len(rows):5d} tasks={len(by):4d}  availability={avail:5.1f}%  "
              f"ever-solved={100*len(solved)/max(len(by),1):5.1f}%  "
              f"mean reward={np.mean([x for v in by.values() for x in v]):.3f}")


def rl_availability():
    """The RL comparison: gradient availability per arm, with the task count that qualifies it.

    Availability needs only enough DISTINCT tasks, not finished arms -- ~100 tasks is ~13 steps.
    Arms below 40 tasks are reported but flagged, because at n=8 a single task flips the number by
    12 points, which is how an earlier 8-task reading nearly became a result.
    """
    print("\n7. RL ARMS: gradient availability (the method's mechanism)")
    import glob as _g
    for tag, lab in (("atscbase", "baseline   level 0.0"), ("atscfix", "static     level 0.5"),
                     ("atsck8", "ATSC       k=8"), ("atsc", "ATSC       adaptive"),
                     ("atsc2", "ATSC       adaptive cold"), ("dapo", "DAPO       filter groups"),
                     ("temp12", "temp 1.2   level 0.0")):
        rows = []
        for f in _g.glob(f"{R}/work/verl/run_{tag}/*.jsonl"):
            for line in open(f, errors="ignore"):
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        if not rows:
            continue
        by = collections.defaultdict(list)
        for r in rows:
            by[(r["scenario"], r["task_idx"])].append(int(r["reward"]))
        multi = [v for v in by.values() if len(v) >= 2]
        mixed = [v for v in multi if len(set(v)) > 1]
        av = 100 * len(mixed) / max(len(multi), 1)
        note = "" if len(multi) >= 40 else "   (too few tasks to quote)"
        print(f"    {lab:24s} tasks={len(multi):4d}  availability={av:5.1f}%  "
              f"reward={np.mean([x for v in by.values() for x in v]):.3f}{note}")
    print("    NOTE: DAPO's figure is tautological -- it discards degenerate groups, so what")
    print("    survives is non-degenerate by construction. Its real cost is ~12x generation.")


def availability_test():
    """Pairwise availability comparisons with a two-proportion z-test.

    Percentages alone misled twice: a 110-task reading of 9.8% vs 15.4% became 7.5% vs 10.9% at
    310 tasks. Availability is a proportion over tasks, so the comparison needs the task count in
    it. This reports the difference with a z-test and refuses to compare arms under 40 tasks.
    """
    import glob as _g
    import math as _m

    def avail(tag):
        rows = []
        for f in _g.glob(f"{R}/work/verl/run_{tag}/*.jsonl"):
            for line in open(f, errors="ignore"):
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        by = collections.defaultdict(list)
        for r in rows:
            by[(r["scenario"], r["task_idx"])].append(int(r["reward"]))
        multi = [v for v in by.values() if len(v) >= 2]
        mixed = [v for v in multi if len(set(v)) > 1]
        return len(mixed), len(multi)

    print("\n8. AVAILABILITY COMPARISONS (two-proportion z-test; the decisive one is first)")
    pairs = [("atsck8", "atsc", "k=8 vs k=5, BOTH controlled     <- clean k test"),
             ("atsc", "atscfix", "adaptive control vs static 0.5  <- THE ABLATION"),
             ("atscfix", "atscbase", "static 0.5 vs handicapped       (surface mechanism)"),
             ("gates", "atscbase", "dense reward vs handicapped     (reward mechanism)"),
             ("gatesatsc", "gates", "dense+control vs dense alone    (do they compose)"),
             ("atsck8", "atscbase", "k=8+control vs handicapped  (CONFOUNDED: two changes)")]
    for a, b, label in pairs:
        xa, na = avail(a)
        xb, nb = avail(b)
        if na < 40 or nb < 40:
            print(f"    {label}: not yet ({a} n={na}, {b} n={nb}; need 40+ each)")
            continue
        pa, pb = xa / na, xb / nb
        pool = (xa + xb) / (na + nb)
        se = _m.sqrt(pool * (1 - pool) * (1 / na + 1 / nb))
        z = (pa - pb) / se if se > 0 else 0.0
        # two-sided normal tail
        pv = _m.erfc(abs(z) / _m.sqrt(2))
        verdict = "significant" if pv < 0.05 else "not significant"
        print(f"    {label}: {100*pa:.1f}% (n={na}) vs {100*pb:.1f}% (n={nb})  "
              f"diff={100*(pa-pb):+.1f}pp  z={z:+.2f}  p={pv:.3f}  {verdict}")


def arm_table():
    """The head-to-head, priced by the measured dose curve.

    The method's OUTPUT is the tool surface its gate certified, so an arm can be scored the
    moment its surface is known -- no need to wait for the policy to finish. Interpolating the
    measured dose curve turns "reached 4087 names" into an expected held-out gain, which the
    direct cells (D, D2, D3) then confirm.
    """
    print("\n6. ARM-LEVEL OUTCOME  (surface reached -> held-out gain, priced by the dose curve)")
    # measured dose points: fraction of the 1791 withheld names restored -> held-out gain
    dose = [(0.0, 0.0), (0.39, 0.0729), (0.78, 0.1085), (1.0, 0.1356)]
    base, total = 2687, 4478
    withheld = total - base
    rows = []
    for tag, gate in (("coadapt", "DISCORD"), ("coadapt3", "DISCORD"), ("coadapt2", "DISCORD"),
                      ("coadaptn", "naive"), ("fixbase", "frozen")):
        f = f"{R}/work/coadapt_{tag}/advertised.txt"
        if not os.path.exists(f):
            continue
        n = sum(1 for _ in open(f))
        frac = max(0.0, min(1.0, (n - base) / withheld))
        pred = np.interp(frac, [d[0] for d in dose], [d[1] for d in dose])
        step = 0
        cks = glob.glob(f"{R}/work/verl/ckpt_{tag}/global_step_*")
        if cks:
            step = max(int(c.split("_")[-1]) for c in cks)
        rows.append((tag, gate, n, 100 * frac, pred, step))
    for tag, gate, n, pct, pred, step in rows:
        print(f"    {tag:9s} {gate:8s} surface={n:4d}/{total} ({pct:5.1f}% restored)  "
              f"predicted held-out gain {pred:+.4f}   step={step}")
    d = {r[0]: r for r in rows}
    if "coadapt" in d and "fixbase" in d:
        print(f"    -> best DISCORD arm over the frozen-harness baseline: "
              f"{d['coadapt'][4] - d['fixbase'][4]:+.4f} held-out")
    if "coadapt" in d and "coadaptn" in d:
        print(f"    -> best DISCORD arm over the naive-gated arm:          "
              f"{d['coadapt'][4] - d['coadaptn'][4]:+.4f} held-out")


def main():
    print("=" * 78)
    print("CO-ADAPTATION RESULTS   regenerated from banked episodes")
    print("=" * 78)
    section_2x2()
    section_dose()
    section_seeds()
    section_gates()
    section_gradient()
    arm_table()
    rl_availability()
    availability_test()
    print()


if __name__ == "__main__":
    main()
