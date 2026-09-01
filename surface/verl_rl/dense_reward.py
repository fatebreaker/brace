"""Partial-credit reward shaping for the AWM/MCP substrate.

WHY. The reward is binary and ~79% of tasks are never solved, so all k rollouts of such a task
score 0, the within-group advantage is identically zero, and the group contributes no gradient.
Measured on 14,897 banked episodes from the atsc/coadapt arms: availability is 10.0% at k=5.
Dense reward is the family of fixes that changes WHAT THE REWARD SAYS rather than which rollouts
are trained on -- "all failed" becomes "failed differently".

TWO CANDIDATE SIGNALS, AND WHY ONLY ONE IS DEFENSIBLE.

  "telemetry"  Shapes from the episode's own process counters (tool calls, errors, parse
               failures, whether a final answer was emitted). This RAISES AVAILABILITY 10.0% ->
               70.6% at k=5 (x7.1) on the banked episodes -- and that is exactly the trap. The
               within-task AUC of each counter for predicting actual success, over 555,923
               solved/unsolved pairs drawn from the same task, is:

                   emitted a final answer   0.590
                   error rate               0.540
                   response tokens          0.525
                   fewer turns              0.523
                   fewer errors             0.518
                   fewer parse failures     0.485   <- below chance
                   more tool calls          0.479   <- below chance

               0.50 is noise. The best single counter is 0.59 and two are anti-correlated with
               success, so the extra 60 points of availability are gradient pointing in a
               direction that does not track solving the task. Any near-continuous nuisance
               variable would do the same: response length alone takes 6,004 distinct values
               over the failures, so it could drive availability to ~100% while teaching the
               policy only to change its output length. This mode exists to MEASURE that
               failure, not because it is expected to work. Keep the binary reward for
               evaluation and expect availability to rise while the binary metric does not.

  "gates"      Shapes from how far the episode got through the task's own verifier. The AWM
               verifiers are sequential guard clauses -- a chain of `if <precondition fails>:
               return {"result": "others"}` ending in `return {"result": "complete"}` -- so the
               index of the guard that fired is a task-grounded measure of progress, produced by
               the same oracle that defines success rather than by a proxy. Over 6,000 verifiers
               the chain has a median of 7 guards (mean 7.6, p90 13, max 32) and 90.8% have at
               least 3, so there is real granularity to extract. 16,727 of 16,730 result-returns
               are literal dicts, so the AST rewrite below reaches essentially all of them.

               ASSUMPTION, stated because it is the weak point: guard index in SOURCE ORDER is
               taken as progress order. That holds for straight-line guard clauses, which is how
               these verifiers are written, but a verifier that checks the expensive condition
               first would be scored backwards. The reward is therefore capped strictly below
               any success (see PARTIAL_CAP) so a mis-ordered verifier can never make failing
               outrank solving.

WHAT THIS MODULE DOES NOT DO. It never changes the binary reward. `awm_agent_loop` keeps writing
the true 0/1 outcome to episodes.jsonl under `reward`; only `AgentLoopOutput.reward_score`, the
training signal, is shaped. Every existing analysis and the whole eval path read `reward` and are
unaffected.

HOW TO ACTUALLY TURN THIS ON -- both halves are required, and each fails SILENTLY on its own.
This module was written on 2026-08-07 and not exercised until the b8dense arm on 2026-08-09;
both traps were live in the repo the whole time.

  1. THE VALUE MUST BE `telemetry` OR `gates`. `enabled()` treats anything else as off, so the
     retired `gatesd` arm's `AWM_DENSE_REWARD=1` was a no-op: it trained on the plain binary
     reward for its entire run while its tag and its supervisor entry said "dense".
  2. IT MUST REACH THE RAY WORKERS. The agent loop runs inside them, MODE is read at import, and
     verl_awm_train.sh forwards a FIXED list of variables into ray_init.runtime_env precisely
     because it does not trust inheritance through the raylet -- AWM_DENSE_REWARD is not on that
     list. Only slurm/verl_awm_train_agent.sh pushes it in, so an arm must run with
     TRAIN_SCRIPT=.../verl_awm_train_agent.sh (DAPO_FILTER_GROUPS=0 to keep dynamic sampling off).
     Confirm it landed by grepping the arm's log for AWM_DENSE_REWARD inside the printed
     "ray init kwargs" dict; a shell export alone will not appear there.

Evidence it works, from b8dense's first episodes (mode `gates`, surface fixed 0.5, pool_max):
every episode carries `train_reward`/`gate_progress`/`dense_mode`, 100% of failed rollouts get
strictly positive partial credit spread over 26 distinct values, and 32% of all-failed k=5 groups
-- which are exactly the groups that contribute nothing under the binary reward -- become
gradient-bearing. The matched binary arm a8F shows 0% by construction.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from functools import lru_cache

__all__ = ["MODE", "enabled", "run_verifier_graded", "shaped_reward"]

# Failures are confined to [0, PARTIAL_CAP) and success is 1.0, so no amount of partial credit
# can ever rank a failed rollout above a solved one -- the ordering the binary reward defines is
# preserved exactly, and shaping only breaks ties inside the failures.
PARTIAL_CAP = 0.5

MODE = os.environ.get("AWM_DENSE_REWARD", "none").strip().lower()


def enabled() -> bool:
    return MODE in ("telemetry", "gates")


# -------------------------------------------------------------------------- ACCORD certificate
# Shaping is applied only where accord.py has certified that verifier progress actually ranks
# solvable tasks above dead ones (see accord.py). Uncertified scenarios keep the binary reward.
# Without this gate the mode is all-or-nothing, and the measured failure is severe: shaping on
# process telemetry raises availability 10.0% -> 70.6% while its counters predict success at
# AUC 0.479-0.590, i.e. noise. The certificate is re-read cheaply (mtime-checked) because it is
# rewritten every cycle as evidence accumulates.
_CERT_PATH = (os.environ.get("AWM_ACCORD_CERT")
              or os.environ.get("AWM_CARVE_CERT", "")).strip()
_cert_cache = {"mtime": None, "allow": None}


def accord_allows(scenario: str) -> bool:
    """True if shaping is permitted for this scenario. No certificate -> shape everywhere,
    which preserves the plain dense-reward arm's behaviour exactly."""
    if not _CERT_PATH:
        return True
    try:
        m = os.path.getmtime(_CERT_PATH)
    except OSError:
        return False          # certificate configured but absent: refuse, never silently shape
    if _cert_cache["mtime"] != m:
        try:
            with open(_CERT_PATH) as fh:
                _cert_cache["allow"] = set(json.load(fh).get("allow") or [])
            _cert_cache["mtime"] = m
        except Exception:
            return False
    return scenario in (_cert_cache["allow"] or set())


# --------------------------------------------------------------------------- verifier grading

class _GateTagger(ast.NodeTransformer):
    """Append a `__gate__` index to every literal `{"result": ...}` return.

    Indices are assigned by source position, not by visit order, so the numbering matches the
    order a reader would call "how far through the checks this got".
    """

    def __init__(self, order: dict[tuple[int, int], int], total: int):
        self._order = order
        self._total = total

    def visit_Return(self, node):  # noqa: N802 (ast API)
        self.generic_visit(node)
        if not isinstance(node.value, ast.Dict):
            return node
        for key, val in zip(node.value.keys, node.value.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "result"
                and isinstance(val, ast.Constant)
            ):
                if val.value == "others":
                    idx = self._order.get((node.lineno, node.col_offset))
                    if idx is not None:
                        node.value.keys.append(ast.Constant("__gate__"))
                        node.value.values.append(ast.Constant(idx))
                elif val.value == "complete":
                    node.value.keys.append(ast.Constant("__gate__"))
                    node.value.values.append(ast.Constant(self._total))
                break
        return node


@lru_cache(maxsize=4096)
def _instrument(code: str):
    """(compiled_code_object, func_name, n_gates) or None when the source cannot be tagged."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    others = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            for key, val in zip(node.value.keys, node.value.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "result"
                    and isinstance(val, ast.Constant)
                    and val.value == "others"
                ):
                    others.append((node.lineno, node.col_offset))
                    break
    others.sort()
    order = {pos: i + 1 for i, pos in enumerate(others)}
    total = len(others) + 1  # reaching "complete" is one step past the last guard
    tree = _GateTagger(order, total).visit(tree)
    ast.fix_missing_locations(tree)
    func_name = "verify_task_completion"
    for line in code.split("\n"):
        s = line.strip()
        if s.startswith("def verify_") and "(" in s:
            func_name = s.split("(")[0].replace("def ", "").strip()
            break
    try:
        return compile(tree, "<verifier:graded>", "exec"), func_name, total
    except (SyntaxError, ValueError):
        return None


def run_verifier_graded(code: str, initial_db: str, final_db: str, final_answer):
    """Like awm_env.run_verifier, but also returns progress in [0,1].

    Returns (result, status, progress). `progress` is gate_index / n_gates, i.e. the fraction of
    the verifier's guard chain the episode cleared. A verifier that cannot be instrumented falls
    back to progress 0.0 for a failure, which degrades to the binary reward rather than erroring.
    """
    if not code or len(code.strip()) < 10:
        return "others", "no_code", 0.0
    inst = _instrument(code)
    if inst is None:
        return "others", "uninstrumentable", 0.0
    compiled, func_name, total = inst
    ns: dict = {}
    try:
        exec(compiled, ns)  # noqa: S102 - same trust model as awm_env.run_verifier
        res = ns[func_name](initial_db, final_db, final_answer)
    except Exception as exc:  # noqa: BLE001 - mirror run_verifier's contract
        return "others", f"error:{type(exc).__name__}:{exc}"[:200], 0.0
    if not isinstance(res, dict) or "result" not in res:
        return "others", "bad_return", 0.0
    v = res.get("result", "others")
    v = v if v in ("complete", "others") else "others"
    gate = res.get("__gate__")
    if not isinstance(gate, int) or total <= 0:
        # a dynamically-built return dict: correct result, no progress information
        return v, "success", (1.0 if v == "complete" else 0.0)
    return v, "success", max(0.0, min(1.0, gate / total))


# --------------------------------------------------------------------------- shaping

def _telemetry_score(n_backend: int, n_err: int, n_parse_fail: int, stop_reason: str) -> float:
    """Process-counter partial credit. Measured to be near-noise; see the module docstring."""
    s = 0.20 if stop_reason == "final_answer" else 0.0
    s += 0.15 * min(n_backend, 10) / 10.0
    s += 0.10 * (1.0 - min(n_err, 10) / 10.0)
    s += 0.05 * (1.0 - min(n_parse_fail, 5) / 5.0)
    return min(s, PARTIAL_CAP)


def shaped_reward(binary: float, progress: float, n_backend: int, n_err: int,
                  n_parse_fail: int, stop_reason: str, scenario: str = "") -> float:
    """Training signal. Identical to `binary` unless AWM_DENSE_REWARD selects a mode AND -- when an
    ACCORD certificate is configured -- this scenario's verifier has been certified."""
    if binary >= 1.0 or not enabled():
        return binary
    if scenario and not accord_allows(scenario):
        return binary
    if MODE == "gates":
        return min(PARTIAL_CAP * progress, PARTIAL_CAP)
    return _telemetry_score(n_backend, n_err, n_parse_fail, stop_reason)


def fingerprint(code: str) -> str:
    return hashlib.sha1(code.encode("utf-8", "ignore")).hexdigest()[:8]
