"""The mechanism reading TRIAGE lives or dies by: what fraction of GRPO groups carry no gradient.

A group is the k rollouts of one (scenario, task_idx) inside one optimizer step's batch. It is
DEGENERATE when every rollout in it got the same binary reward -- all solved or all failed --
because GRPO's advantage is the within-group centred reward, so such a group contributes exactly
zero to the update no matter how the rollouts differed otherwise. Measured across every arm ever
run here that fraction is 87.3%, which is the number TRIAGE exists to move.

WHY THE EPISODE LOG AND NOT THE TRAINER METRICS. verl reports mean reward and grad_norm, neither
of which separates "no gradient because groups are degenerate" from "no gradient because the
policy is at an optimum". The episode log carries (scenario, task_idx, reward, split) per rollout
in generation order, and a step generates exactly train_batch_size * n prompts-worth of them, so
the groups can be reconstructed exactly by chunking the TRAIN episodes in file order.

Validation episodes are dropped first: they are a different pool on a different surface at
temperature 0, and leaving them in would shift every subsequent chunk boundary by their count --
which silently scrambles every group in the run after the first validation pass.

Usage:
    triage_report.py --tag a8T                  # the arm, at its own batch/group size
    triage_report.py --tag a8F --bsz 32 --nroll 5 --window 20    # the baseline to beat
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import Counter, defaultdict

R = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load(path, since=0):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, errors="ignore") as fh:
        for i, line in enumerate(fh):
            if i < since:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("split") == "val":
                continue
            if "scenario" not in d or "reward" not in d:
                continue
            rows.append((d["scenario"], d.get("task_idx"), 1 if float(d["reward"]) > 0 else 0))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--episodes", default=None, help="defaults to work/verl/run_<tag>/episodes.jsonl")
    ap.add_argument("--bsz", type=int, default=32, help="train_batch_size (prompts per step)")
    # 5, not 8. This default WAS 8, and it is the whole reason the fleet carried two different
    # beliefs about k: every arm here runs verl_awm_train.sh's NROLL=${NROLL:-5} -> rollout.n=5
    # (confirmed in the composed Hydra config of rl_a8T.log / rl_a8Te0.log), so a default of 8
    # chunked a8Te0's log into 32x8=256-episode "steps" that no optimizer step ever was, splitting
    # groups across chunk boundaries -- and a split group can only look degenerate. Measured
    # difference on a8Te0 at 4000 episodes: 0.8982 degenerate / 20.83 live per 1k at nroll=8
    # against 0.8950 / 21.00 at the true 5. Small, and wrong for a reason worth not repeating.
    ap.add_argument("--nroll", type=int, default=5, help="rollouts per prompt (group size k); must "
                    "equal the arm's NROLL, which is 5 everywhere except b8k16/b8k16v (16)")
    ap.add_argument("--since", type=int, default=None,
                    help="skip this many leading lines; defaults to run_<tag>/.arm_since so a "
                         "resumed arm is not judged on the run it replaced")
    ap.add_argument("--window", type=int, default=0, help="report only the last N steps")
    ap.add_argument("--min-group", type=int, default=2,
                    help="a group of one rollout cannot be non-degenerate; it is not evidence "
                         "either way and is excluded rather than counted as degenerate")
    ap.add_argument("--json", action="store_true", help="machine-readable summary only")
    a = ap.parse_args()

    ep = a.episodes or f"{R}/work/verl/run_{a.tag}/episodes.jsonl"
    since = a.since
    if since is None:
        try:
            since = int(open(f"{R}/work/verl/run_{a.tag}/.arm_since").read().strip() or 0)
        except Exception:
            since = 0
    rows = load(ep, since)
    per_step = a.bsz * a.nroll
    nsteps = len(rows) // per_step
    if nsteps == 0:
        print(json.dumps({"tag": a.tag, "steps": 0, "episodes": len(rows),
                          "note": f"fewer than one full step ({per_step} train episodes)"}))
        return 0

    out, tot_g, tot_d, tot_roll = [], 0, 0, 0
    for s in range(nsteps):
        chunk = rows[s * per_step:(s + 1) * per_step]
        g = defaultdict(list)
        for sc, ti, r in chunk:
            g[(sc, ti)].append(r)
        groups = [v for v in g.values() if len(v) >= a.min_group]
        deg = sum(1 for v in groups if len(set(v)) == 1)
        out.append({"step": s + 1, "groups": len(groups), "degenerate": deg,
                    "deg_frac": round(deg / max(len(groups), 1), 4),
                    "live_groups": len(groups) - deg,
                    "distinct_tasks": len(g), "rollouts": len(chunk)})
        tot_g += len(groups); tot_d += deg; tot_roll += len(chunk)

    shown = out[-a.window:] if a.window else out
    summary = {
        "tag": a.tag, "episodes": len(rows), "steps": nsteps,
        "bsz": a.bsz, "nroll": a.nroll,
        "degenerate_frac_all": round(tot_d / max(tot_g, 1), 4),
        "degenerate_frac_last8": round(
            sum(d["degenerate"] for d in out[-8:]) / max(sum(d["groups"] for d in out[-8:]), 1), 4),
        "live_groups_per_1k_rollouts": round(1000.0 * (tot_g - tot_d) / max(tot_roll, 1), 2),
    }
    if a.json:
        print(json.dumps(summary))
        return 0
    print(f"# {a.tag}: {len(rows)} train episodes -> {nsteps} steps "
          f"({a.bsz} prompts x {a.nroll} rollouts each), skipped {since} leading lines")
    print(f"{'step':>5} {'groups':>7} {'degen':>6} {'deg_frac':>9} {'live':>5} {'tasks':>6}")
    for d in shown:
        print(f"{d['step']:>5} {d['groups']:>7} {d['degenerate']:>6} {d['deg_frac']:>9.3f} "
              f"{d['live_groups']:>5} {d['distinct_tasks']:>6}")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
