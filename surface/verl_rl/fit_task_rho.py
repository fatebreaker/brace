"""TRIAGE-rho: fit a PER-TASK within-group correlation rho_i from the banked rollout groups.

WHY THIS EXISTS. v3g scores every task with the SAME within-group correlation (rho = 0.78, nu =
0.285, fitted once over a8T2's first two cycles) and a per-task Beta posterior over p. That signal
has run out of resolution: with the warm caps (s <= 20, f <= 40) the 1023 pool tasks carry only 51
DISTINCT weight values, and the 256th selection slot sits inside a tie class of 354 tasks that all
read exactly w = 0.202011 -- 225 of the 256 rows are handed out arbitrarily among ties. The
p-posterior has nothing left to say about which of those 354 to buy.

THE HYPOTHESIS THIS FILE TESTS AND THEN SHIPS. Within-group correlation is TASK-SPECIFIC. Two
tasks at the same p can be worth very different amounts: one whose k rollouts almost always agree
(high rho_i) yields no gradient even at p ~ 0.5, while one whose rollouts disagree (low rho_i) is
worth far more at the same p. Measured on this bank (24,877 groups of k=5 over 1127 tasks, every
8B arm at BSZ=32/NROLL=5): inside that 354-task tie class, where p_hat is IDENTICAL by
construction (0.3387), the fitted rho_i has standard deviation 0.1999 and interquartile range
0.343. It is a second, real dimension, not a re-reading of the first.

THE MODEL. A group of k rollouts on task i is exchangeable Beta-Binomial:

    s_ig ~ BetaBin(k, alpha_i, beta_i),   p_i = alpha_i/(alpha_i+beta_i),
                                          rho_i = 1/(alpha_i+beta_i+1)

which is the SAME object v3's --warm-shrink group already uses -- v3 simply pins alpha_i+beta_i =
nu for every task. This file estimates the concentration per task instead.

THE ESTIMATOR, and why it is a posterior mean and not a maximum. For a task whose observed groups
are all all-fail, the likelihood is maximised at p -> 0, where EVERY rho fits equally well: the
concentration is not identified at all, and an MLE would return an arbitrary point. That is not a
corner case here -- 530 of 1127 tasks have never been solved and 73 are always solved, so 53% of
the pool carries no identifying evidence about rho. The estimator therefore has to shrink, and
shrink by how much the task's own groups actually say:

    1.  L_i(p, rho)  -- the group likelihood on a fixed (p, rho) grid.
    2.  m_i(rho)     -- integrate p out under the flat Beta(1,1) prior the method already uses.
                        A task with no identifying evidence gets a FLAT m_i, by construction.
    3.  pi(rho)      -- the population distribution of rho over tasks, fitted by nonparametric
                        maximum likelihood (EM over the grid) on {m_i}. This is the hierarchical
                        layer: it is estimated from the tasks, not assumed.
    4.  rho_i        -- the posterior mean of rho under pi(rho) * m_i(rho).

Shrinkage toward the population is then automatic and monotone in the amount of evidence: a task
with no groups returns pi's mean exactly, a task with many informative groups returns its own
value, and everything in between interpolates. There is no tuned shrinkage constant.

WHY A PINNED FILE AND NOT A LIVE UPDATE. Same argument as --warm-bank: the bank is a snapshot, not
a subscription. rho is a property of the harness -- the k rollouts of a group share a policy
checkpoint AND a byte-identical prompt -- not of the moment, and the bank holds 24,877 groups
against the 256 a cycle adds, so a live update would be swamped for many cycles and would keep
re-injecting foreign evidence that gamma could never age out. The live signal that DOES move every
cycle is p, and that is exactly what triage.py's own count machinery already tracks. Recorded as a
limitation, not hidden: rho_i does not adapt within a run.

Output: JSONL, one row per task with any group evidence, read by triage.py --task-rho.
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict

import numpy as np
from scipy.special import betaln, comb

R = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# THE ARMS, LISTED EXPLICITLY AND NOT GLOBBED. Group reconstruction chunks an episode log into
# BSZ*NROLL blocks, so an arm whose batch shape differs is silently mis-chunked by a glob -- and a
# split group can only ever look degenerate, which biases rho UP. Excluded on purpose:
#   a8T4            BSZ=80/NROLL=2   (different batch shape; k=2 groups are not k=5 groups)
#   b8k16, b8k16v   NROLL=16
#   q2b*, q4b*      2B/4B models -- a different policy's correlation is not this policy's
#   dapo            dynamic sampling rejects groups after generating them, so its episode log is
#                   not a record of what a fixed batch produced
#   atsc*, adaptive, a8A*, a8C*, b8*   different pool or different surface regime
DEFAULT_ARMS = ("a8F a8Fr a8T a8Tg1 a8Te0 a8T2 a8T2b a8T2g1 a8T2n a8T2r a8T2s a8T2w a8T3 a8T3g "
                "a8T3g2 a8T3gr a8T5 a8T5k a8T5kr a8Tlp a8Tpe a8Tvip")


def read_groups(tag: str, bsz: int, k: int, since: int | None = None):
    """(task_key, step, n_success) per GROUP, reconstructed exactly the way triage_report.py does.

    A step generates BSZ*k train episodes in file order; inside one step a task appears once, so
    its k rollouts there are exactly one GRPO group (verl stamps one uuid per parquet row and
    groups the advantage on it). Groups that do not come back at full size are dropped rather than
    counted: a partial group cannot be non-degenerate and would bias the concentration upward.
    """
    p = f"{R}/work/verl/run_{tag}/episodes.jsonl"
    if since is None:
        try:
            since = int(open(f"{R}/work/verl/run_{tag}/.arm_since").read().strip() or 0)
        except Exception:
            since = 0
    rows = []
    if not os.path.exists(p):
        return []
    # A MARKER AT OR PAST EOF IS A DESTROYED BOUNDARY, NOT AN EMPTY SEGMENT (2026-08-19). Same
    # defect and same fix as paper_numbers.train_rows: run_a8Tvip/.arm_since equals that log's own
    # line count, so this reader returned zero groups for the arm and it vanished from the 8B
    # correlation fit without a word. Read the whole log and say so on stderr. Markers that select
    # a real sub-segment (run_q2bN at 13,622 of 14,509) are untouched.
    if since:
        with open(p, errors="ignore") as fh:
            n_lines = sum(1 for _ in fh)
        if since >= n_lines:
            print("[fit_task_rho] %s: .arm_since=%d >= %d lines -- marker at EOF, reading the "
                  "whole log" % (tag, since, n_lines), file=sys.stderr)
            since = 0
    with open(p, errors="ignore") as fh:
        for i, line in enumerate(fh):
            if i < since:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            # validation episodes are a different pool on a different surface at temperature 0;
            # leaving them in would also shift every later chunk boundary by their count
            if d.get("split") == "val":
                continue
            if "scenario" not in d or "reward" not in d:
                continue
            rows.append((d["scenario"], str(d.get("task_idx")), 1 if float(d["reward"]) > 0 else 0))
    per, out = bsz * k, []
    for s in range(len(rows) // per):
        g = defaultdict(list)
        for sc, ti, r in rows[s * per:(s + 1) * per]:
            g[f"{sc}::{ti}"].append(r)
        for kk, v in g.items():
            if len(v) == k:
                out.append((kk, s + 1, sum(v)))
    return out


def make_grid(k: int, n_p: int = 241, n_rho: int = 81, rho_lo: float = 0.02, rho_hi: float = 0.985):
    """(p grid, rho grid, log pmf table of shape (n_p, n_rho, k+1)).

    p is LOGIT-spaced: a task sitting at p = 0.01 has to be distinguishable from one at p = 0 or
    the concentration is not identifiable in the tail at all, and the tail is half this pool.
    """
    S = np.arange(k + 1)
    p = 1.0 / (1.0 + np.exp(-np.linspace(-9.0, 9.0, n_p)))
    lo, hi = np.log(rho_lo / (1 - rho_lo)), np.log(rho_hi / (1 - rho_hi))
    rho = 1.0 / (1.0 + np.exp(-np.linspace(lo, hi, n_rho)))
    nu = (1.0 - rho) / rho
    a = p[:, None] * nu[None, :]
    b = (1.0 - p)[:, None] * nu[None, :]
    lp = (np.log(comb(k, S))[None, None, :]
          + betaln(a[:, :, None] + S[None, None, :], b[:, :, None] + (k - S)[None, None, :])
          - betaln(a, b)[:, :, None])
    return p, rho, lp


def fit_rho(counts: np.ndarray, k: int, iters: int = 4000, tol: float = 1e-11):
    """counts (n_task, k+1) of groups by success count -> (rho_i posterior means, pi, rho grid).

    Step 2-4 of the docstring. `pi` is the NPMLE of the population distribution of rho over the
    grid; a task whose m_i is flat (no identifying evidence) comes back at pi's mean exactly.
    """
    p, rho, lp = make_grid(k)
    LL = (counts @ lp.reshape(-1, k + 1).T).reshape(-1, len(p), len(rho))
    dw = np.gradient(p)
    dw = dw / dw.sum()
    mx = LL.max(axis=(1, 2), keepdims=True)
    M = (np.exp(LL - mx) * dw[None, :, None]).sum(1)
    M = M / np.clip(M.sum(1, keepdims=True), 1e-300, None)
    pi = np.ones(len(rho)) / len(rho)
    for _ in range(iters):
        post = M * pi[None, :]
        post /= np.clip(post.sum(1, keepdims=True), 1e-300, None)
        new = post.mean(0)
        if np.abs(new - pi).max() < tol:
            pi = new
            break
        pi = new
    post = M * pi[None, :]
    post /= np.clip(post.sum(1, keepdims=True), 1e-300, None)
    return post @ rho, post, pi, rho


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=DEFAULT_ARMS,
                    help="whitespace/comma separated tags, all at --bsz x --k. See DEFAULT_ARMS "
                         "for why this is an explicit list and not a glob.")
    ap.add_argument("--bsz", type=int, default=32)
    ap.add_argument("--k", type=int, required=True, help="group size the arms generated at")
    ap.add_argument("--pool", default=f"{R}/surface/gate_caller/pools/pool_max.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-groups", type=int, default=1,
                    help="tasks with fewer groups than this are left out of the file entirely, so "
                         "triage.py falls back to the global rho for them")
    a = ap.parse_args()

    tags = [t for t in a.arms.replace(",", " ").split() if t]
    pool = {f"{q['scenario']}::{q['task_idx']}": (q["scenario"], q["task_idx"])
            for q in json.load(open(a.pool))}
    per_task, n_g = defaultdict(list), 0
    for t in tags:
        g = read_groups(t, a.bsz, a.k)
        for kk, st, s in g:
            if kk in pool:
                per_task[kk].append(s)
        n_g += len(g)
        print(f"[fit_task_rho] {t}: {len(g)} groups of k={a.k}", flush=True)
    keys = sorted(per_task)
    C = np.zeros((len(keys), a.k + 1))
    for i, kk in enumerate(keys):
        for s in per_task[kk]:
            C[i, s] += 1
    G = C.sum(1)
    print(f"[fit_task_rho] {n_g} groups over {len(keys)} pool tasks "
          f"(median {np.median(G):.0f} groups/task)", flush=True)

    rho_i, post, pi, rho = fit_rho(C, a.k)
    # a credible interval per task, so a consumer can see WHICH rows are informative rather than
    # having to infer it from the group count
    cdf = np.cumsum(post, axis=1)
    lo = rho[np.argmax(cdf >= 0.05, axis=1)]
    hi = rho[np.argmax(cdf >= 0.95, axis=1)]
    pbar = (C * np.arange(a.k + 1)).sum(1) / (a.k * np.maximum(G, 1))
    q1, q2, q3 = np.percentile(rho_i, [25, 50, 75])
    print(f"[fit_task_rho] rho_i: min {rho_i.min():.4f} Q1 {q1:.4f} med {q2:.4f} Q3 {q3:.4f} "
          f"max {rho_i.max():.4f} IQR {q3-q1:.4f}", flush=True)
    print(f"[fit_task_rho] population mean rho = {float(pi @ rho):.4f} "
          f"(nu = {(1-float(pi @ rho))/float(pi @ rho):.4f})", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n = 0
    with open(a.out, "w") as fh:
        for i, kk in enumerate(keys):
            if G[i] < a.min_groups:
                continue
            sc, ti = pool[kk]
            fh.write(json.dumps({"scenario": sc, "task_idx": ti, "rho": round(float(rho_i[i]), 6),
                                 "n_groups": int(G[i]), "p_bank": round(float(pbar[i]), 6),
                                 "rho_lo": round(float(lo[i]), 6),
                                 "rho_hi": round(float(hi[i]), 6)}) + "\n")
            n += 1
    print(f"[fit_task_rho] wrote {n} rows to {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
