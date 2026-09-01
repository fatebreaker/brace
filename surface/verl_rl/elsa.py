"""ELSA -- Elasticity-Steered Allocation: spend the rollout budget where the environment can be
moved, and spend the environment's range only where moving it pays.

-------------------------------------------------------------------------------------------------
THE ARGUMENT, AND HOW IT CONTINUES THE PAPER'S

The paper's claim is that what is specific to MCP is that the environment is EDITABLE: the set of
tools a server advertises is a design choice, not a property of the task. Section 7 then shows the
obstacle -- a loop cannot tell a real edit from noise -- and \textsc{Discord} removes it by
refusing to accept an edit whose effect is below what the budget can resolve.

ELSA is the same principle applied one level down. \textsc{Discord} asks whether an edit moves the
OBJECTIVE before accepting it. ELSA asks whether an edit can move the GRADIENT before spending on
it. Both are the refusal to pay for an intervention nobody measured, and the measurement is the
contribution in each case.

That refusal is not rhetorical here; it is forced by the data. Under the v1 controller (pure
proportional control on p - 0.5), after 65 GRPO steps on the live a8A arm:

    level 0.00:   8 scenarios          126 of 158 scenarios sat pinned at the ceiling, mean
    level 0.25:   8                    level 0.877, advertising 95.6% of the tool universe.
    level 0.50:   6                    A controller pinned at its ceiling is not controlling
    level 0.75:  10                    anything: ATSC had silently become "advertise everything",
    level 1.00: 126                    which is why a static surface kept pace with it.

The reason is not a bad gain constant. It is that the tool surface is a lever with HETEROGENEOUS
and MEASURABLE authority: a minority of scenarios are hard because the agent lacks a tool, and the
majority are hard for reasons -- multi-step reasoning, verifier strictness -- that no advertised
tool touches. v1 assumed uniform authority. v2 (surface_control_v2.py) at least detected the
difference and froze the unresponsive scenarios, but freezing only stops waste; it does not
convert it. The budget freed by freezing 126 scenarios went nowhere.

ELSA converts it, and that conversion is the method.

-------------------------------------------------------------------------------------------------
ELASTICITY

Define the elasticity of scenario s as the authority the tool surface has over the gradient it
yields, where availability g(p,k) = 1 - p^k - (1-p)^k is GRPO's usable signal at group size k:

    e_s  =  dg_s / dl_s        the SIGNED least-squares slope through the origin over the
                               (level move, availability move) pairs the run has already
                               produced, CREDITED ONLY IF it exceeds z standard errors of its
                               own binomial noise, and shrunk toward the population median by
                               the number of moves observed.

The noise floor is not a detail, it is the estimator. The obvious version -- average |dg| / |dl|
-- measures sampling noise rather than authority: at ~25 episodes per scenario per cycle
se(p_hat) ~ 0.1, so g jitters by ~0.1 no matter what the surface does, and a 0.25 level move
manufactures an apparent elasticity of 0.4. Replayed over a8A's 15.9k banked episodes that
version credited all 158 scenarios (median 0.53, minimum 0.27, none below a 0.05 floor) and put
113 straight back at the ceiling -- v1 again, with extra machinery. Signing the slope is what
makes cancellation possible, since real authority puts dg and dl on the same side repeatedly
while noise contributes both signs; the standard error is what decides how much agreement is
enough. With it, 90 of 158 scenarios are correctly held as inert.

Elasticity is free. It is read off rollouts the trainer generated anyway, exactly as the pool
reweighting is; ELSA adds no generation cost over the frozen-surface control.

-------------------------------------------------------------------------------------------------
THE TWO CONSEQUENCES

(1) A SURFACE THAT MUST BE EARNED, steered with a measured gain. Every scenario starts at the
    handicap baseline -- level 0, the exact advertised set every arm in the comparison begins
    from -- and buys tools only where the slope has been credited. An untested scenario takes a
    small fixed PROBE step, because no slope exists yet to take a Newton step on; once two moves
    have been banked, the step that closes the availability gap is

        dl_s  =  clip( (g_max - g_s) / max(e_s, e_floor),  min_step,  max_step )

    and a scenario whose slope never clears its noise floor simply stops moving. Both bounds
    matter and both were set by measurement: starting at 0.5 with a 0.5 cap reaches the ceiling in
    ONE move, which is an absorbing state -- no room left means no further move, no further move
    means no (dl, dg) pair, and the scenario's authority can never be estimated at all. Replayed,
    that variant parked 110 of 158 scenarios at 1.00 for a mean level of 0.850 against v1's 0.877:
    no improvement worth the name. Starting at 0 with a 0.25 cap ends at mean level 0.249 with
    ZERO scenarios at the ceiling, and separates what it steers from what it does not -- credited
    scenarios sit at mean level 0.358, inert ones at 0.247.

    This also removes the confound that made the a8A/a8F contrast unreadable. v1 advertised 95.6%
    of the tool universe against static's 79.9%, so it could not be told apart from an arm that
    was simply handed more capability. ELSA advertises 70.0%: LESS than the static control. A
    gain measured against a smaller surface cannot be explained by a bigger one.

(2) THE SAMPLER IS PRICED ON THE POST-EDIT ENVIRONMENT. This is the part no baseline can express.
    Every task-selection method -- PLR, its variance-weighted form, DAPO's filtering -- ranks a
    task by the learning signal it yields RIGHT NOW. In an editable environment that is the wrong
    price, because the environment the task will be trained in is not the one it was measured in.
    ELSA prices each task by its ACHIEVABLE availability, the value it will have once its
    scenario's surface has been steered:

        g_hat_i  =  min( g_max,  g_i + e_s(i) * room_s(i) )

    where room is the level range still available in the direction p = 0.5 must move. Sampling
    weight is g_hat, not g. A task that looks dead now but sits on an elastic scenario is bought;
    a task that looks alive now but sits on an inelastic one is not oversold. The environment
    lever and the task lever are therefore solved as ONE allocation against one budget, rather
    than as two schedulers that happen to run in the same loop.

    The hierarchy is deliberate and matches the substrate: elasticity is a property of a SCENARIO
    (the surface is advertised per server), while availability is a property of a TASK. Tasks
    inherit their scenario's elasticity and keep their own solve rate.

-------------------------------------------------------------------------------------------------
EVERY BASELINE IS A CONSTRAINED SPECIAL CASE

    e_s := 0 for all s          -> sampling by current availability      (AWS / PLR / variance)
    weights := uniform          -> ATSC-v2, the environment lever alone
    e_s := infinity for all s   -> ATSC-v1, which is why it saturates
    z := 0 (credit noise)       -> ATSC-v1 again, reached by a different route: every scenario
                                   looks responsive, so every level runs to the ceiling
    both levers off             -> the frozen-surface control (vanilla / static)
    filter after generating     -> DAPO, which pays 1/g in extra rollouts for the same effect

This is what makes the comparison a method claim rather than a horse race: the baselines are not
merely weaker, they are ELSA with a coordinate held fixed, and the held coordinate is named.

-------------------------------------------------------------------------------------------------
CORRECTNESS NOTE ON ATTRIBUTION

An observed solve rate must be attributed to the level that was in force when it was observed.
v2 read a fixed trailing window, which mixes episodes from before and after a level move and
therefore biases every elasticity estimate toward zero -- the exact direction that would make the
method look unnecessary. ELSA records the episode-file offset at each level change and reads only
episodes after it, so p_s is measured under one surface.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import zlib

import numpy as np

R = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SURF = f"{R}/surface/gate_surface/work/surfaces"
META = "__elsa__"


def scenario_tools(scen):
    p = os.path.join(SURF, f"{scen}.json")
    return [t["name"] for t in json.load(open(p)).get("tools", [])]


def g_of(p, k):
    return 1.0 - p ** k - (1.0 - p) ** k


def fresh(lv=0.5):
    return {"lv": float(lv), "prev_lv": None, "prev_g": None, "prev_se": None, "moves": []}


def se_g(p, n, k):
    """Standard error of the availability estimate, propagated from the binomial error on p.

    g' (p) = -k p^(k-1) + k (1-p)^(k-1), and se(p_hat) = sqrt(p(1-p)/n).
    """
    if n <= 0:
        return float("inf")
    dg = abs(-k * p ** (k - 1) + k * (1.0 - p) ** (k - 1))
    return dg * ((p * (1.0 - p) / n) ** 0.5)


def elasticity(st, pop, kappa, z):
    """Signed, noise-floored slope dg/dl for one scenario.

    THIS IS THE ESTIMATOR THE METHOD TURNS ON, and the naive version of it does not work. Taking
    the mean of |dg| / |dl| over observed moves measures the SAMPLING NOISE in g, not the surface's
    authority over it: at ~25 episodes per scenario per cycle, se(p_hat) ~ 0.1, so g jitters by
    ~0.1 whatever the surface does, and a 0.25 level move manufactures an apparent elasticity of
    0.4. Measured over four simulated cycles on a8A's 15.9k banked episodes that is exactly what
    happened -- median 0.53, minimum 0.27, not one scenario of 158 falling below a 0.05 floor, and
    113 levels back at the ceiling. An absolute value cannot cancel noise, because noise has no
    sign to cancel with.

    So credit only what survives its own error bar, which is the rule \textsc{Discord} already
    applies to harness edits, applied here to the gradient instead of the objective:

        slope    = sum(dl * dg) / sum(dl^2)            least squares through the origin, SIGNED
        se       = sqrt(sum((dl * se_dg)^2)) / sum(dl^2)
        e        = slope   if slope > z * se   else 0

    Signed rather than absolute is what makes the cancellation work: real authority puts dg and dl
    on the same side repeatedly, while noise contributes both signs and averages toward zero. A
    negative slope -- advertising more tools makes things worse -- yields no usable authority in
    the direction the controller would push, so it reads as inelastic too.

    A scenario with fewer than two observed moves has not been tested and returns the population
    value, not zero: zero would permanently deny a lever to a scenario that was never tried, which
    is how a controller talks itself into believing the environment is inert.
    """
    mv = st.get("moves") or []
    m = len(mv)
    if m < 2:
        return pop, m, 0.0
    den = sum(d[0] ** 2 for d in mv)
    if den <= 1e-12:
        return pop, m, 0.0
    slope = sum(d[0] * d[1] for d in mv) / den
    se = (sum((d[0] * d[2]) ** 2 for d in mv) ** 0.5) / den
    if not (slope > z * se):
        return 0.0, m, se
    return (m * slope + kappa * pop) / (m + kappa), m, se


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--pool", default=f"{R}/surface/gate_caller/pools/pool_max.json")
    ap.add_argument("--init-surface", required=True)
    ap.add_argument("--levels", required=True)
    ap.add_argument("--out-map", required=True)
    ap.add_argument("--out-weights", required=True,
                    help="per-task sampling weights, consumed by aws_reweight.py --weighting "
                         "external. Written by the SAME pass that sets the levels, because the "
                         "weights are prices on the surface this pass just chose; computing them "
                         "in a separate process against a stale map is how the two levers "
                         "silently decouple.")
    ap.add_argument("--target", type=float, default=0.5)
    ap.add_argument("--band", type=float, default=0.15)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--kappa", type=float, default=2.0,
                    help="shrinkage strength on the elasticity estimate, in units of level moves. "
                         "At 2 a scenario needs two observed moves before its own evidence "
                         "outweighs the population's.")
    ap.add_argument("--z", type=float, default=1.0,
                    help="how many standard errors a slope must clear before the surface is "
                         "credited with any authority over that scenario. At 0 the estimator "
                         "reverts to crediting noise, which is measurably v1's failure.")
    ap.add_argument("--e-floor", type=float, default=0.05,
                    help="smallest elasticity the Newton step will divide by; also the threshold "
                         "below which a scenario is judged inelastic and its level held")
    ap.add_argument("--min-step", type=float, default=0.125)
    ap.add_argument("--max-step", type=float, default=0.25,
                    help="A scenario starts at 0.5, so a 0.5 cap reaches the ceiling in ONE move. "
                         "That is an absorbing state: at the ceiling there is no room left, the "
                         "level stops moving, no (dl, dg) pair is ever banked, and the elasticity "
                         "of that scenario can never be estimated. In the five-cycle replay this "
                         "sent 108 of 158 scenarios to 1.00 on cycle 1 -- v1's endpoint, reached "
                         "faster. At 0.25 the range is spent over at least two moves, each of "
                         "which is evidence.")
    ap.add_argument("--init-level", type=float, default=0.0,
                    help="level a scenario starts at. 0.0 is the HANDICAP BASELINE -- exactly the "
                         "surface in advertised_init.txt, which every arm in the comparison "
                         "starts from -- so ELSA opens no tool it has not earned. Starting at 0.5 "
                         "like v1/v2 is what made the range too short to identify anything: "
                         "observed p sits far below target almost everywhere, so every probe "
                         "moves UP, and two probes exhaust the range before the first slope is "
                         "credited (measured: 110 of 158 at the ceiling, mean 0.850, against "
                         "v1's 126 and 0.877 -- no improvement worth the name). From 0.0 there "
                         "are eight moves of headroom, so authority is identified while range "
                         "remains to spend on it. It also removes the confound that sank the "
                         "a8A/a8F contrast: an arm that advertises MORE cannot be told apart "
                         "from an arm that is simply given more capability.")
    ap.add_argument("--probe-step", type=float, default=0.25,
                    help="fixed step for a scenario with fewer than two observed moves. A slope "
                         "cannot be estimated without moving the lever, so the first moves are "
                         "deliberately uniform probes rather than Newton steps taken on a "
                         "population prior that says nothing about this scenario.")
    ap.add_argument("--max-obs", type=int, default=48,
                    help="most recent episodes per scenario used for one estimate. Caps how far back evidence accumulates: ~6 cycles, enough to pass the noise floor without describing a policy that no longer exists.")
    ap.add_argument("--min-obs", type=int, default=3,
                    help="rollouts a scenario needs under the CURRENT level before its solve rate "
                         "is used at all. Deliberately LOW, because this gate is redundant with "
                         "the noise floor and actively harmful above it: se_g scales as 1/sqrt(n), "
                         "so a small sample already produces a large standard error and fails "
                         "z*se on its own merits. Set to 8 it instead blocked the EVIDENCE from "
                         "ever being recorded -- a move is banked only when a scenario clears this "
                         "gate, and a move resets that scenario's mark, so at ~8 episodes per "
                         "scenario per cycle roughly half fell short every cycle and the chain "
                         "broke before a second move could be logged. Measured on a8Az0 after "
                         "three live cycles: 139 of 158 scenarios had banked ZERO moves and none "
                         "was eligible for crediting, so the controller would have spent its whole "
                         "horizon on the population prior taking uniform probe steps -- which is "
                         "v1, the thing this method exists to replace.")
    ap.add_argument("--uniform-weights", action="store_true",
                    help="ABLATION: keep the surface lever, disable the sampler. Weights are left "
                         "uniform, so this is ELSA reduced to environment control alone -- the "
                         "a8A2/ATSC-v2 setting, but reached through THIS code path so the only "
                         "variable is the sampler. A separate implementation cannot make that claim.")
    ap.add_argument("--e-zero", action="store_true",
                    help="ABLATION: keep the sampler, disable the surface lever. Elasticity is "
                         "forced to 0, so no level ever moves and every task is priced at its "
                         "CURRENT availability -- which is exactly what PLR and AWS do. Together "
                         "with --uniform-weights this brackets the method: if either arm matches "
                         "a8A3, the joint allocation is not doing the work.")
    ap.add_argument("--floor-share", type=float, default=0.02,
                    help="fraction of sampling weight held out and spread uniformly, so a task "
                         "priced at zero is still occasionally re-tested rather than abandoned")
    ap.add_argument("--max-window", type=int, default=20000,
                    help="hard cap on episodes read, for the case where no level has ever moved")
    a = ap.parse_args()

    pool = json.load(open(a.pool))
    scens = sorted({p["scenario"] for p in pool})
    tasks = [(p["scenario"], int(p["task_idx"])) for p in pool]
    init = {ln.strip() for ln in open(a.init_surface) if ln.strip()}
    g_max = g_of(a.target, a.k)

    state = json.load(open(a.levels)) if os.path.exists(a.levels) else {}
    meta = state.get(META) if isinstance(state.get(META), dict) else {"since": 0}

    # ---- evidence per scenario, marked from ITS OWN last level change ---------------------------
    # A global mark throws away every episode at each pass, leaving ~8 per scenario per cycle
    # (32 prompts x 8 steps x 5 rollouts / 158 scenarios). At n=8 the standard error of g is 0.35,
    # so the noise floor would only credit slopes above 2.78 -- above most real ones, which run
    # 0.9-3.3. The method would then credit nothing, never move a level, and silently become the
    # frozen-surface baseline while still reporting itself as ELSA.
    #
    # The mark only has to guarantee that p is measured under ONE surface, and a scenario whose
    # level did not move is still under that surface -- so its episodes accumulate across cycles
    # and its estimate keeps sharpening. Only a level CHANGE resets that scenario's mark. Capped
    # at --max-obs, because the policy also improves over time and very old episodes describe a
    # surface held by a policy that no longer exists.
    lines = []
    if os.path.exists(a.episodes):
        with open(a.episodes, errors="ignore") as fh:
            lines = fh.readlines()
    total = len(lines)

    sc_hist = collections.defaultdict(list)
    task_hist = collections.defaultdict(list)
    for i, line in enumerate(lines[-a.max_window:], start=max(0, total - a.max_window)):
        try:
            r = json.loads(line)
            y = int(r["reward"] > 0)
        except Exception:
            continue
        sc = r.get("scenario")
        sc_hist[sc].append((i, y))
        try:
            task_hist[(sc, int(r["task_idx"]))].append((i, y))
        except Exception:
            pass

    def evidence(hist, mark):
        return [y for i, y in hist if i >= mark][-a.max_obs:]

    by_sc = {sc: evidence(h, 0) for sc, h in sc_hist.items()}      # rebound per scenario below
    by_task = {}

    # ---- population elasticity, for shrinkage and for never-moved scenarios ---------------------
    # Population elasticity: the median CREDITED slope, used both to shrink individual estimates
    # and as the value handed to scenarios that have not been tested yet. Taken over credited
    # scenarios only -- including the uncredited zeros would drag the untested prior toward zero
    # and stop exploration before it started.
    raws = []
    for sc in scens:
        st = state.get(sc)
        if isinstance(st, dict):
            mv = st.get("moves") or []
            den = sum(d[0] ** 2 for d in mv)
            if len(mv) >= 2 and den > 1e-12:
                sl = sum(d[0] * d[1] for d in mv) / den
                se = (sum((d[0] * d[2]) ** 2 for d in mv) ** 0.5) / den
                if sl > a.z * se:
                    raws.append(sl)
    pop_e = float(np.median(raws)) if raws else 0.30

    # ---- one pass: update elasticity, set the level, price the scenario -------------------------
    ela = {}
    moved = elastic = inert = starved = 0
    for sc in scens:
        st = state.get(sc)
        if not isinstance(st, dict):
            st = fresh(float(st) if st is not None else a.init_level)  # migrate v1/v2
        st.setdefault("moves", [])
        st.setdefault("since", 0)
        obs = evidence(sc_hist.get(sc, []), st["since"])
        by_sc[sc] = obs

        if len(obs) >= a.min_obs:
            p = float(np.mean(obs))
            g = g_of(p, a.k)
            sg = se_g(p, len(obs), a.k)
            # bank the (dl, dg, se_dg) triple from the previous move -- this is the whole
            # estimator. se_dg carries the error of BOTH endpoints, since dg is their difference.
            if st["prev_lv"] is not None and st["prev_g"] is not None:
                dl = st["lv"] - st["prev_lv"]
                if abs(dl) > 1e-9:
                    ps = st.get("prev_se")
                    sd = (sg ** 2 + (ps if ps is not None else sg) ** 2) ** 0.5
                    st["moves"] = (st["moves"] + [[dl, g - st["prev_g"], sd]])[-8:]
            st["prev_lv"], st["prev_g"], st["prev_se"] = st["lv"], g, sg
        else:
            p = g = None
            starved += 1

        e, nmoves, e_se = elasticity(st, pop_e, a.kappa, a.z)
        if a.e_zero:
            e = 0.0                      # ablation: the surface is declared inert by fiat
        st["e"], st["e_se"], st["n_moves"] = e, e_se, nmoves

        if p is not None and e >= a.e_floor:
            elastic += 1
            if p < a.target - a.band:
                room, sign = 1.0 - st["lv"], +1.0
            elif p > a.target + a.band:
                room, sign = st["lv"], -1.0
            else:
                room, sign = 0.0, 0.0
            if room > 1e-9:
                if nmoves < 2:                      # probe: no slope exists to take a step on yet
                    step = a.probe_step
                else:
                    step = min(a.max_step, max(a.min_step, (g_max - g) / max(e, a.e_floor)))
                nl = min(1.0, max(0.0, st["lv"] + sign * min(step, room)))
                if abs(nl - st["lv"]) > 1e-9:
                    st["lv"] = nl
                    st["since"] = total       # new surface here: start this scenario's evidence over
                    moved += 1
        elif p is not None:
            inert += 1                      # hold the level; the sampler will reprice it below

        # headroom left for the surface to buy, in the direction p must travel
        if p is None:
            room = max(1.0 - st["lv"], st["lv"])
        elif p < a.target:
            room = 1.0 - st["lv"]
        else:
            room = st["lv"]
        ela[sc] = (e if e >= a.e_floor else 0.0, room)
        state[sc] = st

    # ---- the surface map ------------------------------------------------------------------------
    amap, tot_adv, tot_all = {}, 0, 0
    for sc in scens:
        lv = state[sc]["lv"]
        full = scenario_tools(sc)
        held_back = sorted(set(full) - init)
        # crc32, NOT hash(). Python salts str hashing per process (PYTHONHASHSEED), so hash(sc)
        # returns a different value in every invocation -- measured 47151 then 31080 for the same
        # scenario. This module runs once per cycle in a FRESH process, so hash() would redraw the
        # advertised set every cycle even when the level did not move. For a static arm that is a
        # confound; for ELSA it is fatal, because elasticity is dg/dl and a redrawn surface puts
        # churn into dg that has nothing to do with dl. The noise floor models binomial error
        # only, so that churn would be credited as authority -- the estimator would measure its
        # own shuffling. crc32 is stable across processes, so a level is a REPRODUCIBLE set of
        # tools and dg is attributable to the level change alone.
        rng = random.Random(zlib.crc32(sc.encode()))
        rng.shuffle(held_back)
        n = int(round(lv * len(held_back)))
        adv = sorted((set(full) & init) | set(held_back[:n]))
        amap[sc] = adv
        tot_adv += len(adv)
        tot_all += len(full)
    json.dump(amap, open(a.out_map, "w"))

    # ---- the prices: ACHIEVABLE availability, not current ---------------------------------------
    # A task the current surface has not exercised is priced from the evidence that DOES exist,
    # in this order: its own rollouts, else its scenario's pooled rollouts, else the best
    # achievable price in the pool. Most tasks are unobserved within any single cycle -- 597 of
    # 1127 in the four-cycle replay -- so pricing all of them at a single optimistic constant
    # would hand more than half the sampling budget to a flat distribution and drown the signal
    # the method exists to act on. The scenario is a strong predictor of its tasks and is almost
    # always observed, which is what makes the middle rung worth having.
    obs_g, hat_g = [], []
    w, from_task, from_scen, from_pool = {}, 0, 0, 0
    for t in tasks:
        e, room = ela.get(t[0], (0.0, 0.0))
        o = evidence(task_hist.get(t, []), state.get(t[0], {}).get("since", 0))
        if o:
            base = g_of(float(np.mean(o)), a.k)
            obs_g.append(base)
            from_task += 1
        else:
            so = by_sc.get(t[0]) or []
            base = g_of(float(np.mean(so)), a.k) if len(so) >= a.min_obs else None
            if base is not None:
                from_scen += 1
        if base is None:
            w[t] = None
            from_pool += 1
            continue
        gh = min(g_max, base + e * room)
        w[t] = gh
        hat_g.append(gh)
    best = max(hat_g) if hat_g else g_max
    for t in tasks:
        if w[t] is None:
            w[t] = best
    if a.uniform_weights:
        w = {t: 1.0 for t in tasks}      # ablation: surface lever only
    tot = sum(w.values()) or 1.0
    fl = a.floor_share / max(len(tasks), 1)
    w = {t: (1.0 - a.floor_share) * (w[t] / tot) + fl for t in tasks}
    json.dump({f"{t[0]}::{t[1]}": w[t] for t in tasks}, open(a.out_weights, "w"))

    meta["since"] = total
    state[META] = meta
    json.dump(state, open(a.levels, "w"), indent=1)

    mean_lv = float(np.mean([state[s]["lv"] for s in scens])) if scens else 0.0
    cur = 100 * float(np.mean(obs_g)) if obs_g else float("nan")
    ach = 100 * float(np.mean(hat_g)) if hat_g else float("nan")
    # share of the sampling budget bought on the strength of the surface rather than current value
    el_share = 100 * sum(w[t] for t in tasks if ela.get(t[0], (0.0, 0))[0] > 0) / max(sum(w.values()), 1e-9)
    nobs = [len(v) for v in by_sc.values() if v] or [0]
    print(f"[elsa] {len(scens)} scenarios | evidence/scenario median {int(np.median(nobs))} of "
          f"{total} banked (cap {a.max_obs}) | elastic {elastic}, inert {inert}, starved {starved} | "
          f"median elasticity {pop_e:.3f} | {moved} levels moved, mean level {mean_lv:.3f} | "
          f"advertising {tot_adv}/{tot_all} names", flush=True)
    print(f"[elsa] availability now {cur:.1f}% -> achievable {ach:.1f}% | "
          f"{el_share:.1f}% of sampling weight priced on surface headroom | "
          f"prices from {from_task} own rollouts, {from_scen} scenario, "
          f"{from_pool} pool-optimistic at {best:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
