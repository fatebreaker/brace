"""Availability-Weighted Sampling: reweight the training pool from the run's own rollouts.

MOTIVATION. Gradient availability is the binding constraint in this setting, and it decays as the
policy learns: tasks that were mid-difficulty become reliably solved and stop producing
within-group advantage. Replacing the pool at the midpoint corrects the decay but does not pay,
because it discards the policy-task fit accumulated on the original tasks and costs a fresh
screen. The pool was never the problem; the sampling distribution over it was.

ALGORITHM. Keep every task, and change how often each is sampled. After each chunk of training,
read the rollouts the run has already produced, form a Beta posterior over each task's solve
rate, and set that task's sampling weight to its expected availability at the configured group
size,

    w_i = E[ 1 - p_i^k - (1-p_i)^k ],   p_i ~ Beta(alpha_0 + x_i, beta_0 + n_i - x_i),

evaluated in closed form through E[p^k] = prod_{j<k} (a+j)/(a+b+j). Tasks are then repeated in
the training file in proportion to w_i, so the sampler draws them at that rate. The prior is the
pool-level Beta fitted once, which keeps the estimate stable for tasks with few rollouts.

COST. Zero additional rollouts. The estimate uses episodes the trainer generated anyway, so the
procedure adds only the cost of rewriting a parquet file, unlike online rejection sampling, which
pays 1/availability in extra generation at every step, and unlike re-screening, which pays a full
offline screen at every refresh.
"""
from __future__ import annotations
import argparse, collections, json, math, os, subprocess, sys
import numpy as np

R = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Interpreter locations, resolved from the environment so no absolute path is baked in.
# BRACE_ENVS is the parent directory of the three project environments (see README.md);
# VERL_PY / AWM_PY / VLLM_PY override an individual interpreter.
ENVS = os.environ.get("BRACE_ENVS", os.path.join(R, "envs"))
VERL_PY = os.environ.get("VERL_PY", os.path.join(ENVS, "mcp_verl", "bin", "python"))
AWM_PY = os.environ.get("AWM_PY", os.path.join(ENVS, "mcp_awm", "bin", "python"))
VLLM_PY = os.environ.get("VLLM_PY", os.path.join(ENVS, "mcp_vllm", "bin", "python"))


def beta_moment(a, b, k):
    r = 1.0
    for i in range(k): r *= (a + i) / (a + b + i)
    return r

def availability(a, b, k):
    return 1.0 - beta_moment(a, b, k) - beta_moment(b, a, k)

def fit_prior(counts, k):
    """Pool-level Beta by moments, used as the prior for per-task posteriors."""
    ph = np.array([x / max(n, 1) for x, n in counts], float)
    nbar = max(1.0, float(np.mean([n for _, n in counts])))
    m, v = float(ph.mean()), float(ph.var(ddof=1)) if len(ph) > 1 else 0.0
    vb = m * (1 - m) / nbar
    if m * (1 - m) <= vb or v <= vb:
        return 0.5, 0.5
    rho = min(max((v - vb) / (m * (1 - m) - vb + 1e-12), 1e-4), 0.99)
    s = (1 - rho) / rho
    return max(m * s, 0.05), max((1 - m) * s, 0.05)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True, help="the run's own episodes.jsonl")
    ap.add_argument("--base-pool", required=True, help="the task pool being trained on")
    ap.add_argument("--out-pool", required=True)
    ap.add_argument("--out-data", required=True, help="parquet directory to rebuild")
    ap.add_argument("--k", type=int, default=5, help="group size in use")
    ap.add_argument("--window", type=int, default=1200, help="most recent episodes to trust")
    ap.add_argument("--slots", type=int, default=200, help="rows in the rebuilt pool")
    ap.add_argument("--floor", type=float, default=0.02, help="minimum share per task")
    ap.add_argument("--weighting", default="availability",
                    choices=["availability", "filter", "curriculum", "variance", "external"],
                    help="availability: ours. filter: keep only tasks seen both solved and unsolved, "
                         "the offline analogue of discarding degenerate groups. curriculum: target "
                         "an annealing difficulty. variance: weight by observed reward variance. "
                         "external: take the weights from --weights-file, used by ELSA, which "
                         "prices a task by the availability it will have AFTER its scenario's "
                         "surface is steered and so cannot be computed from rollouts alone.")
    ap.add_argument("--weights-file", default=None,
                    help="JSON {\"scenario::task_idx\": weight} required by --weighting external")
    ap.add_argument("--progress", type=float, default=0.0,
                    help="fraction of training elapsed, used by the curriculum baseline")
    a = ap.parse_args()

    base = json.load(open(a.base_pool))
    tasks = [(p["scenario"], p["task_idx"]) for p in base]
    # CYCLE 1 HAS NO EPISODES. The file is created by the first rollout, so on a fresh arm this
    # open() raised FileNotFoundError and took the whole reweight down -- observed on a8A3's first
    # cycle. It was harmless there only because cycle 1's weights are uniform regardless; at any
    # later cycle the same crash would silently leave the pool un-reweighted while the arm kept
    # its method's name. Missing or unreadable episodes mean "no evidence yet", not "abort".
    #
    # External weighting does not read episodes at all: ELSA prices every task itself, including
    # the unobserved ones. Reading them here was pure coupling, and it is what made a path that
    # needs no evidence fail for lack of it.
    rows = []
    if a.weighting != "external" and os.path.exists(a.episodes):
        with open(a.episodes, errors="ignore") as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        rows = rows[-a.window:]
    obs = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        try:
            t = (r["scenario"], r["task_idx"])
        except Exception:
            continue
        obs[t][0] += int(r["reward"]); obs[t][1] += 1

    counts = [(obs[t][0], obs[t][1]) for t in tasks if obs[t][1] > 0]
    a0, b0 = fit_prior(counts, a.k) if len(counts) > 2 else (0.5, 0.5)
    # Optimism under uncertainty. A task with no observations has no posterior worth trusting,
    # and on a large pool most tasks are unobserved early: giving them the prior's availability
    # would let the sampler lock onto whichever tasks it happened to draw first. Unobserved tasks
    # instead receive the best availability any observed task has, so they are sampled at least
    # once and then judged on their own evidence.
    def weight_of(x, n):
        p = (a0 + x) / (a0 + b0 + n)
        if a.weighting == "availability":
            return availability(a0 + x, b0 + (n - x), a.k)
        if a.weighting == "filter":
            # keep a task only while it has been seen both solved and unsolved
            return 1.0 if (0 < x < n) else 0.0
        if a.weighting == "curriculum":
            # target difficulty anneals from easy to hard as training proceeds
            target = 0.8 - 0.6 * a.progress
            return float(np.exp(-((p - target) ** 2) / (2 * 0.15 ** 2)))
        if a.weighting == "variance":
            return float(p * (1 - p))
        return 1.0

    w = {}
    if a.weighting == "external":
        # ELSA has already priced every task, including the unobserved ones, in the same pass that
        # set the levels. Recomputing anything here would silently overwrite a price that was
        # derived from the surface with one that ignores it. Fail loudly instead of falling back:
        # a reweighting arm that quietly reverts to availability is an unlabelled duplicate of the
        # AWS baseline, which is the failure mode this file's docstring already records once.
        if not a.weights_file or not os.path.exists(a.weights_file):
            print(f"[aws] FATAL: --weighting external needs --weights-file; "
                  f"{a.weights_file!r} is missing", flush=True)
            return 1
        ext = json.load(open(a.weights_file))
        miss = [t for t in tasks if f"{t[0]}::{t[1]}" not in ext]
        if miss:
            print(f"[aws] FATAL: weights file covers {len(ext)} keys but misses {len(miss)} of "
                  f"{len(tasks)} pool tasks, e.g. {miss[:3]}", flush=True)
            return 1
        w = {t: float(ext[f"{t[0]}::{t[1]}"]) for t in tasks}
        print(f"[aws] external weights for {len(w)} tasks, "
              f"range {min(w.values()):.4f}-{max(w.values()):.4f}", flush=True)
        n_unseen = 0
    else:
        for t in tasks:
            x, n = obs[t]
            if n > 0:
                w[t] = weight_of(x, n)
        best = max(w.values()) if w else 0.5
        n_unseen = 0
        for t in tasks:
            if obs[t][1] == 0:
                w[t] = best
                n_unseen += 1
    if n_unseen:
        print(f"[aws] {n_unseen}/{len(tasks)} tasks unobserved so far; given optimistic weight "
              f"{best:.3f} so they are explored", flush=True)
    tot = sum(w.values()) or 1.0
    share = {t: max(w[t] / tot, a.floor / len(tasks)) for t in tasks}
    tot2 = sum(share.values())
    reps = {t: max(1, int(round(a.slots * share[t] / tot2))) for t in tasks}

    out = []
    by = {(p["scenario"], p["task_idx"]): p for p in base}
    for t in tasks:
        out += [by[t]] * reps[t]
    json.dump(out, open(a.out_pool, "w"))
    top = sorted(tasks, key=lambda t: -w[t])[:3]
    bot = sorted(tasks, key=lambda t: w[t])[:3]
    print(f"[aws] prior Beta({a0:.3f},{b0:.3f}) from {len(counts)} observed tasks, k={a.k}", flush=True)
    print(f"[aws] mean availability {np.mean(list(w.values())):.4f}; "
          f"pool {len(tasks)} tasks -> {len(out)} rows", flush=True)
    print(f"[aws] most sampled: " + ", ".join(f"{t[1]}@{reps[t]}(w={w[t]:.2f})" for t in top), flush=True)
    print(f"[aws] least sampled: " + ", ".join(f"{t[1]}@{reps[t]}(w={w[t]:.2f})" for t in bot), flush=True)
    json.dump({"prior": [a0, b0], "mean_availability": float(np.mean(list(w.values()))),
               "rows": len(out), "n_tasks": len(tasks),
               "weights": {f"{t[0]}::{t[1]}": w[t] for t in tasks}},
              open(os.path.join(os.path.dirname(a.out_pool), "aws_state.json"), "w"), indent=2)

    cmd = [VERL_PY,
           f"{R}/surface/verl_rl/prep_awm.py", "--pool", a.out_pool, "--out", a.out_data]
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join([f"{R}/surface/gate_surface", f"{R}/surface/gate_caller",
                                  env.get("PYTHONPATH", "")])
    env.setdefault("AWM_PY", AWM_PY)
    env.setdefault("MCP_ROOT", R)
    env.setdefault("GATE_MAX_TURNS", "20")
    env.setdefault("GATE_MAX_NEW_TOKENS", "1024")
    env.setdefault("CALLER_FAMILY", "qwen")
    print(f"[aws] rebuilding parquet", flush=True)
    subprocess.run(cmd, check=True, env=env)
    return 0

if __name__ == "__main__":
    sys.exit(main())
