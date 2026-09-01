"""Tool RETRIEVAL: advertise the tools most RELEVANT to the task, not the ones that teach best.

WHY THIS IS THE BASELINE THE PAPER CANNOT SKIP. ATSC's claim is not "a smaller tool surface helps"
-- it is that the surface should be chosen for GRADIENT AVAILABILITY rather than for task fit. The
obvious objection is that nobody would deploy an MCP agent with a randomly handicapped surface;
they would put a retriever in front of the registry and advertise the top-k tools for the request,
which is what every production tool-use stack (and every "too many tools" paper) actually does.
That rule also shrinks the surface, also changes p, and costs nothing to run. If it lands in the
same place as the availability controller, the controller is not the contribution.

So this arm answers exactly one question: at the SAME surface size, does choosing tools for
RELEVANCE do what choosing them for AVAILABILITY does? The two rules are expected to disagree in a
specific, checkable way. Relevance is monotone in solve rate -- it strictly helps the model find
the tool it needs -- so it pushes each scenario toward p=1, which is the availability-collapsing
end of the dial (measured: g(p,5) is 3.5% at full restoration against 6.4% at 39%). Availability
control deliberately withholds tools the task DOES need, once the policy has learned to use them.
A retrieval surface therefore should look better on reward and worse on gradient, and the arm
exists to show that trade rather than assert it.

BUDGET MATCHING IS THE ENTIRE EXPERIMENT. Compared against a8F (fixed level 0.5) this advertises
the SAME NUMBER OF NAMES IN EVERY SCENARIO -- not the same number on average, the same number
scenario by scenario, computed by --match-level from the same handicapped init surface and the
same tool json a8F's controller uses. Verified: 4380 name-slots over 158 scenarios of pool_max,
0 per-scenario mismatches against work/coadapt_a8F/surface_map.json. Without that, any difference
in reward is a difference in surface SIZE and the arm measures nothing.

THE SCORER IS LEXICAL ON PURPOSE. BM25 over (tool name + description) against the task text. An
embedding retriever would be stronger, but it would also add a model download, a second GPU
tenant, and an unshared failure mode to a comparison whose only job is to be a fair, cheap,
reproducible stand-in for "a sensible deployed retriever". BM25 is the standard non-neural
baseline in exactly that literature, and heuristic_surface.py already showed plain word overlap at
this budget doubles the fraction of tasks whose full required tool set survives (0.527 vs 0.223).
The point of the arm is not to build the best retriever; it is to show that even a good one
selects for the wrong thing.

GRANULARITY. The advertised set is resolved once per SCENARIO by the rollout path
(awm_agent_loop._scenario_assets), never per episode, so a per-task retrieval surface is not
representable no matter how it is scored. A tool's score is therefore the MAX of its BM25 scores
over that scenario's pooled tasks -- max, not the score against the concatenated text, because
concatenation lets one long task's vocabulary dominate the shorter ones and quietly turns a
per-scenario retriever into a retriever for whichever task happens to be wordiest.

TIE-BREAKING. Most tools score zero against most tasks, so the cut usually falls inside a large
tie. Breaking it alphabetically would advertise tools beginning with "a" and is a real confound
(names are not random with respect to function -- `add_*`, `create_*` cluster). Ties break on a
per-scenario seeded shuffle instead, i.e. exactly the handicap's own random draw among tools the
scorer cannot separate. The seed is a stable CRC of the scenario name, NOT the builtin hash():
str hashing is salted per process, so `random.Random(hash(sc))` -- what surface_control.py uses --
draws a DIFFERENT order in every invocation. See the report; that is a live bug there, and this
module does not reproduce it.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import re
import sys
import zlib

R = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Interpreter locations, resolved from the environment so no absolute path is baked in.
# BRACE_ENVS is the parent directory of the three project environments (see README.md);
# VERL_PY / AWM_PY / VLLM_PY override an individual interpreter.
ENVS = os.environ.get("BRACE_ENVS", os.path.join(R, "envs"))
VERL_PY = os.environ.get("VERL_PY", os.path.join(ENVS, "mcp_verl", "bin", "python"))
AWM_PY = os.environ.get("AWM_PY", os.path.join(ENVS, "mcp_awm", "bin", "python"))
VLLM_PY = os.environ.get("VLLM_PY", os.path.join(ENVS, "mcp_vllm", "bin", "python"))

SURF = f"{R}/surface/gate_surface/work/surfaces"

# Function words plus the boilerplate that saturates AWM tool docs. A term that appears in every
# tool description carries no ranking signal; BM25's idf already discounts it, but dropping it
# outright keeps the document lengths honest, which idf does not fix.
STOP = set(
    "the a an of for for the to and or in on at by from with as is are be been was were this that "
    "these those it its i me my we our you your they them their there here if then else when "
    "where which who what how all any each both more most other some such only own same so than "
    "too very can will just should now return returns returning get gets list lists please using "
    "use used value values name names id ids field fields object objects string integer boolean "
    "number optional required parameter parameters argument arguments example examples response "
    "responses request requests data info information details detail result results".split()
)

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")


def toks(s: str) -> list[str]:
    """camelCase and snake_case both carry the signal, so split on both before lowercasing."""
    out = []
    for w in _WORD.findall(str(s or "").replace("_", " ")):
        parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+", w) or [w]
        for p in parts:
            p = p.lower()
            if len(p) > 2 and p not in STOP:
                out.append(p)
    return out


def scenario_tools(scen: str) -> list[dict]:
    p = os.path.join(SURF, f"{scen}.json")
    if not os.path.exists(p):
        return []
    return json.load(open(p)).get("tools", [])


# --------------------------------------------------------------------------------------- BM25

class BM25:
    """Okapi BM25 over one scenario's tool registry.

    The corpus is the scenario's OWN tools, which is the set a deployed retriever would search,
    so idf measures "how much does this term distinguish tools within this server" rather than
    something global and unrelated to the choice being made.
    """

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = max(len(docs), 1)
        self.tf = [collections.Counter(d) for d in docs]
        self.dl = [len(d) for d in docs]
        self.avgdl = (sum(self.dl) / self.N) or 1.0
        df: collections.Counter = collections.Counter()
        for t in self.tf:
            df.update(t.keys())
        # Robertson/Sparck-Jones idf with the +0.5 smoothing, floored at 0: the unfloored form
        # goes NEGATIVE for a term present in more than half the corpus, which would make a tool
        # score worse for matching a common word than for not matching it at all.
        self.idf = {
            w: max(0.0, math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)) for w, n in df.items()
        }

    def score(self, query: list[str], i: int) -> float:
        tf, dl, s = self.tf[i], self.dl[i], 0.0
        for w in query:
            f = tf.get(w, 0)
            if not f:
                continue
            s += self.idf.get(w, 0.0) * f * (self.k1 + 1.0) / (
                f + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
            )
        return s


def budget_for(full: list[str], init: set[str], level: float) -> int:
    """How many names a --surface-fixed LEVEL arm advertises for this scenario.

    Copied from surface_control.main()'s realisation step, deliberately, so the retrieval arm and
    the fixed-surface arm it is compared against are budget-identical BY CONSTRUCTION and not by a
    number someone typed in once. Only the COUNT is taken from there -- which names fill it is the
    thing under test.
    """
    held_back = sorted(set(full) - init)
    return len(set(full) & init) + int(round(level * len(held_back)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=f"{R}/surface/gate_caller/pools/pool_max.json")
    ap.add_argument("--init-surface", required=True,
                    help="the handicapped starting surface; used ONLY to compute the budget")
    ap.add_argument("--out-map", required=True, help="scenario -> advertised tool names")
    ap.add_argument("--match-level", type=float, default=0.5,
                    help="advertise the same per-scenario name count a --surface-fixed arm at "
                         "this level would advertise")
    ap.add_argument("--budget", type=float, default=None,
                    help="override: advertise this FRACTION of each scenario's tools instead of "
                         "matching a level. Only for diagnostics; it breaks the budget match.")
    ap.add_argument("--desc-chars", type=int, default=600,
                    help="tool descriptions embed long response examples; truncate so the tail of "
                         "a verbose doc cannot outrank a short exact-match name")
    ap.add_argument("--levels", default=None,
                    help="optional: write a levels.json so the run dir looks like every other "
                         "arm's. Retrieval has no level, so these are the realised name "
                         "fractions, recorded for analysis only -- nothing reads them back.")
    a = ap.parse_args()

    sys.path.insert(0, f"{R}/surface/gate_surface")
    os.environ.setdefault("AWM_PY", AWM_PY)
    from awm_env import load_tasks

    pool = json.load(open(a.pool))
    init = {ln.strip() for ln in open(a.init_surface) if ln.strip()}
    by_scen: dict[str, list[int]] = collections.defaultdict(list)
    for p in pool:
        by_scen[p["scenario"]].append(int(p["task_idx"]))
    scens = sorted(by_scen)

    amap, levels = {}, {}
    tot_adv = tot_all = tot_budget = 0
    n_scored = 0
    for sc in scens:
        tools = scenario_tools(sc)
        if not tools:
            continue
        names = [t["name"] for t in tools]
        docs = [toks(t["name"]) + toks(str(t.get("description", ""))[: a.desc_chars])
                for t in tools]
        bm = BM25(docs)

        try:
            tasks = load_tasks(sc)
        except Exception:
            tasks = []
        queries = [toks(str(tasks[i])) for i in sorted(set(by_scen[sc])) if i < len(tasks)]
        score = [0.0] * len(names)
        for q in queries:
            if not q:
                continue
            for i in range(len(names)):
                s = bm.score(q, i)
                if s > score[i]:
                    score[i] = s
        n_scored += sum(1 for s in score if s > 0)

        k = (max(1, int(round(a.budget * len(names)))) if a.budget is not None
             else budget_for(names, init, a.match_level))
        k = max(1, min(k, len(names)))

        order = list(range(len(names)))
        random.Random(zlib.crc32(sc.encode()) & 0xFFFFFFFF).shuffle(order)   # tie-break, stable
        order.sort(key=lambda i: -score[i])                                  # stable => ties keep
        amap[sc] = sorted(names[i] for i in order[:k])
        levels[sc] = round(k / max(len(names), 1), 4)
        tot_adv += len(amap[sc])
        tot_all += len(names)
        tot_budget += k

    json.dump(amap, open(a.out_map, "w"))
    if a.levels:
        json.dump(levels, open(a.levels, "w"), indent=1)
    print(f"[retrieval] {len(amap)} scenarios | BM25 over tool name+description vs task text | "
          f"advertising {tot_adv}/{tot_all} names (budget {tot_budget}, "
          f"match-level {a.match_level}) | {n_scored} tools scored non-zero", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
