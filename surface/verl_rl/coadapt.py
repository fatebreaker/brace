"""Co-adaptation: train the policy and certify harness edits in the same loop.

WHY THIS IS THE METHOD. Selecting which tasks a policy trains on is curriculum learning by
another name, and at this scale policy learning did not transfer to held-out tasks at all. What
is specific to the Model Context Protocol is that the environment is editable: the set of tools a
server advertises is a design choice, not a fixed property of the task. A harness improvement
does not need to generalise the way a policy improvement does, because it changes the environment
for every task that touches those tools, including tasks never trained on.

The obstacle to editing a harness during training has always been that the loop cannot tell a
real improvement from noise; Section 7 measures a deployed rule accepting a harmful edit and then
misreporting its own objective by 32 to 57 per cent. \textsc{Discord} removes that obstacle: it
certifies an edit from a median of 150 evaluated tasks, and refuses edits whose effect is below
what any budget can resolve.

THE LOOP
    for each cycle:
        train the policy for K steps on the current harness
        propose an edit to the advertised tool surface
        evaluate incumbent and candidate on the same tasks, curtailed
        accept only if the exact conditional test clears alpha
        continue training on the surviving harness

Both halves are already validated separately. This measures whether together they produce a gain
on held-out scenarios, which neither produced alone.
"""
from __future__ import annotations
import argparse, json, os, random, re, shlex, subprocess, sys

R = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Interpreter locations, resolved from the environment so no absolute path is baked in.
# BRACE_ENVS is the parent directory of the three project environments (see README.md);
# VERL_PY / AWM_PY / VLLM_PY override an individual interpreter.
ENVS = os.environ.get("BRACE_ENVS", os.path.join(R, "envs"))
VERL_PY = os.environ.get("VERL_PY", os.path.join(ENVS, "mcp_verl", "bin", "python"))
AWM_PY = os.environ.get("AWM_PY", os.path.join(ENVS, "mcp_awm", "bin", "python"))
VLLM_PY = os.environ.get("VLLM_PY", os.path.join(ENVS, "mcp_vllm", "bin", "python"))
SURF = f"{R}/surface/gate_surface/work/surfaces"

def group_size() -> int:
    """The rollouts per prompt the TRAINER will actually generate with, k.

    ONE reader for a value three different rules are functions of (TRIAGE's g(p,k), AWS's
    availability, ELSA's prices). slurm/verl_awm_train.sh does `NROLL=${NROLL:-5}` and hands it to
    actor_rollout_ref.rollout.n, and the supervisor exports NROLL into this process's environment,
    so this default MUST stay equal to that one: if they ever drift, every weight rule here prices
    a group size the generator does not use and nothing crashes to say so.
    """
    return int(os.environ.get("NROLL", "5"))

def latest_step(ckpt):
    if not os.path.isdir(ckpt): return 0
    xs = [int(d.split("_")[-1]) for d in os.listdir(ckpt) if d.startswith("global_step_")]
    return max(xs) if xs else 0

def handicapped_surface(pool_path, frac, seed):
    """The advertised set both halves start from, computed exactly as awm_discord.py computes it.

    Training and gating MUST run on one pool. The tool-name universe is the union over the pool's
    scenarios, so a surface derived from one pool would strip every tool from scenarios the other
    pool contains -- silently making those tasks unsolvable rather than failing loudly.
    """
    pool = json.load(open(pool_path))
    scens = sorted({p["scenario"] for p in pool})
    full = sorted({t["name"] for s in scens
                   for t in json.load(open(f"{SURF}/{s}.json")).get("tools", [])})
    withheld = set(random.Random(seed).sample(full, int(round(frac * len(full)))))
    return full, sorted(set(full) - withheld)

def vark_trainer_agrees(cycle: int, steps: int, stats_path: str, log_path: str):
    """GATE C: does the histogram the TRAINER realized match the one the ALLOCATOR wrote?

    The allocator records vark_hist over the cycle's whole pool; the patched trainer prints one
    '[vark] repeat: N rows -> T rollouts; k histogram {...}' line per optimizer step, each over
    that step's block. Summing the cycle's step histograms must reproduce the allocation exactly.

    Returns (ok, reason). A MISSING trainer line is a failure, not a pass: it is precisely what a
    silently unarmed monkeypatch looks like.
    """
    want = None
    try:
        for ln in open(stats_path):
            rec = json.loads(ln)
            if rec.get("cycle") == cycle and rec.get("vark_hist"):
                want = {int(k): int(v) for k, v in rec["vark_hist"].items()}
    except Exception as exc:
        return False, f"could not read the allocator's histogram from {stats_path}: {exc}"
    if not want:
        return False, f"the allocator recorded no vark_hist for cycle {cycle}"
    pat = re.compile(r"\[vark\] repeat: .*k histogram \{([^}]*)\}")
    seen = []
    try:
        with open(log_path, errors="ignore") as fh:
            for ln in fh:
                m = pat.search(ln)
                if m:
                    d = {}
                    for part in m.group(1).split(","):
                        if ":" in part:
                            kk, vv = part.split(":")
                            d[int(kk.strip())] = int(vv.strip())
                    seen.append(d)
    except Exception as exc:
        return False, f"could not read the trainer log {log_path}: {exc}"
    if not seen:
        return False, ("the trainer printed NO '[vark] repeat' line -- the monkeypatch never "
                       "armed in the Ray actor, so every task trained at the uniform rollout.n")
    # TWO LINES PER STEP, NOT ONE. The patch deliberately makes BOTH repeat sites per-row -- the
    # generation-side repeat and the batch-side repeat that must mirror it -- and each prints. An
    # earlier version of this check summed the last `steps` lines, which on an 8-step cycle summed
    # the last FOUR steps twice: it produced a histogram with the right row count and the wrong
    # composition, and aborted a healthy q2bK whose trainer had in fact honoured the allocation
    # exactly. The pairing is structural, so it is asserted rather than guessed at.
    per_step = 2
    # RESUMED CYCLES RUN FEWER STEPS THAN THE CYCLE TARGET, and the full-cycle comparison below is
    # then arithmetically impossible: a trainer that resumes at step 5 of an 8-step cycle consumes
    # only the remaining blocks, so its summed histogram is a FRACTION of the allocation and can
    # never equal it. The 2026-09-06 disk-full restart hit exactly this and Gate C killed a healthy
    # q2bK3 -- the SECOND false positive from this gate (the first, 2026-08-24, was the double-count
    # of paired lines). The gate's purpose is to catch a silently unarmed monkeypatch, i.e. a run
    # that trains at uniform k while every artifact says variable k. That purpose is preserved on a
    # partial cycle by the structural checks plus a SUBSET check; only the exact sum is skipped, and
    # skipping it is announced rather than silent.
    if len(seen) == 0:
        return False, ("the trainer printed NO '[vark] repeat' line -- the monkeypatch never armed")
    if len(seen) % per_step != 0:
        return False, (f"{len(seen)} trainer histogram lines is not a whole number of "
                       f"{per_step}-line steps: a repeat site did not print")
    pairs = len(seen) // per_step
    partial = pairs < steps
    if partial:
        for i in range(0, len(seen), per_step):
            if seen[i] != seen[i + 1]:
                return False, (f"the two repeat sites disagree within a step: {seen[i]} vs "
                               f"{seen[i + 1]} -- the batch-side repeat did not mirror the "
                               f"generation-side one")
        gotp = {}
        for d in seen[0::per_step]:
            for kk, vv in d.items():
                gotp[kk] = gotp.get(kk, 0) + vv
        stray = sorted(set(gotp) - set(want))
        if stray:
            return False, (f"trainer used group sizes {stray} that the allocator never assigned "
                           f"(trainer {gotp}, allocator {want})")
        # NO uniform-k check here. It was tried on 2026-09-06 and false-positived within minutes:
        # the allocator spreads its non-default sizes thinly (14 of 256 rows at 2 or 8), and the
        # allocation is per 32-row block, so a block that is entirely k=5 is COMMON and legitimate.
        # An unarmed patch is already caught above by printing no lines at all; a patch that prints
        # a per-row histogram is by definition armed. That check added nothing and killed q2bK3 a
        # second time in one evening.
        return True, (f"PARTIAL CYCLE ({pairs} of {steps} steps ran, so this cycle was resumed): "
                      f"exact-sum check SKIPPED as arithmetically impossible; verified instead that "
                      f"the patch armed, both repeat sites mirrored, and every realized group size "
                      f"{sorted(gotp)} is one the allocator assigned {sorted(want)}")
    tail = seen[-per_step * steps:]
    for i in range(0, len(tail), per_step):
        if tail[i] != tail[i + 1]:
            return False, (f"the two repeat sites disagree within a step: {tail[i]} vs "
                           f"{tail[i + 1]} -- the batch-side repeat did not mirror the "
                           f"generation-side one, so rollouts are paired with the wrong uid")
    got = {}
    for d in tail[0::per_step]:
        for kk, vv in d.items():
            got[kk] = got.get(kk, 0) + vv
    if got != want:
        return False, f"trainer realized {got} but the allocator assigned {want}"
    return True, f"trainer realized {got}, matching the allocation"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--jobid", required=True)
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--steps-per-cycle", type=int, default=25)
    ap.add_argument("--handicap-frac", type=float, default=0.40)
    ap.add_argument("--handicap-seed", type=int, default=555)
    ap.add_argument("--edit-block", type=int, default=700)
    ap.add_argument("--pool", default=f"{R}/surface/gate_caller/pools/pool_big320.json",
                    help="ONE pool drives both halves; see handicapped_surface()")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="shifts the edit-proposal stream; a replicate run uses a different "
                         "offset so its edits are independent draws, not the same edits again")
    ap.add_argument("--surface-fixed", type=float, default=None,
                    help="hold every scenario at this restoration level and never re-tune. This "
                         "is the ablation that decides whether ATSC's CONTROL matters or whether "
                         "a mid-level surface alone explains any gain.")
    ap.add_argument("--surface-control", action="store_true",
                    help="tune the per-scenario tool surface toward the solve rate that maximises "
                         "gradient availability, instead of holding one global surface")
    ap.add_argument("--surface-elsa", action="store_true",
                    help="ELSA: estimate each scenario's ELASTICITY -- the authority the tool "
                         "surface has over the gradient it yields -- then steer the surface with "
                         "that measured gain AND price the task sampler by the availability each "
                         "task will have once its surface is steered. This is the only setting in "
                         "which the two levers are one allocation: it implies its own reweighting, "
                         "so --reweight is ignored and must not be passed. See elsa.py.")
    ap.add_argument("--accord", "--carve", dest="accord", action="store_true",
                    help="ACCORD: before each cycle, re-certify from the run's own rollouts whether "
                         "verifier progress actually ranks solvable tasks above dead ones, and let "
                         "dense_reward shape ONLY the scenarios that pass. Attacks the measured "
                         "binding constraint -- ~72%% of tasks are never solved, so ~88%% of GRPO "
                         "groups carry no gradient at all and grad_norm sits at 0.01 -- which no "
                         "amount of tool-surface steering can reach. The AUC it tests is the concordance "
                         "index -- the other half of Discord's discordant-pair test. See accord.py.")
    ap.add_argument("--elsa-extra", default="",
                    help="extra flags forwarded verbatim to elsa.py. This is how the ABLATIONS are "
                         "run -- '--z 0' to credit noise, '--uniform-weights' for the surface "
                         "lever alone, '--e-zero' for the sampler alone. They go through the same "
                         "module as the method arm, so an ablation differs from a8A3 in exactly "
                         "one flag rather than in a whole reimplementation.")
    ap.add_argument("--surface-control-v2", action="store_true",
                    help="ATSC-v2: steer only scenarios whose gradient availability RESPONDS to the "
                         "level, freezing tool-insensitive ones. v1 pinned 126 of 158 scenarios at "
                         "the ceiling, which is not control; see surface_control_v2.py")
    # THE TWO SURFACE-RULE BASELINES. Both emit exactly the surface_map.json that
    # --surface-control emits and hand it to the rollout through AWM_ADVERTISED_MAP, so the
    # rollout path, the parquet, the pool and the eval cells are all bit-identical to the method
    # arm and the ONLY thing that differs is which names the map contains. A baseline that needed
    # its own launch path would differ in ways nobody enumerated.
    ap.add_argument("--surface-retrieval", action="store_true",
                    help="advertise the tools most RELEVANT to each scenario's tasks (BM25 over "
                         "tool name+description), budget-matched per scenario to a "
                         "--surface-fixed arm. The deployed-agent rule; see surface_retrieval.py")
    ap.add_argument("--surface-accel", action="store_true",
                    help="evolve the per-scenario level by ACCEL: regret-prioritised archive plus "
                         "undirected mutation, instead of proportional control on (p - target). "
                         "See accel_surface.py for the regret estimator")
    ap.add_argument("--retrieval-match-level", type=float, default=0.5,
                    help="the --surface-fixed level whose per-scenario NAME COUNT the retrieval "
                         "surface reproduces. Must equal the comparison arm's level or the arm "
                         "measures surface size instead of surface choice.")
    ap.add_argument("--reweight",
                    choices=["none", "availability", "filter", "variance", "curriculum"],
                    default="none",
                    help="between cycles, resample the pool toward tasks that can actually "
                         "produce a gradient. ~90%% of GRPO groups here are degenerate, so the "
                         "policy half is starved; this is the lever that feeds it. This list MUST "
                         "stay in step with aws_reweight.py --weighting: 'variance' is the "
                         "PLR-style learning-potential baseline and 'curriculum' the annealing "
                         "one, and leaving them out of these choices killed those arms at argv "
                         "parse before a single rollout was generated.")
    ap.add_argument("--gate", choices=["discord", "naive", "none"], default="discord",
                    help="acceptance rule for harness edits; the naive arm is the contrast that "
                         "shows the gate, not the editing, is what makes co-adaptation work")
    ap.add_argument("--task-alloc",
                    choices=["none", "gradmass", "bandfilter", "vip", "progress", "budget"],
                    default="none",
                    help="TRIAGE: choose the cycle's task list BEFORE generating anything, by the "
                         "posterior expected probability that a group of k rollouts carries "
                         "gradient (see triage.py). 'none' is the untouched path every existing "
                         "arm runs; this flag adds a step and changes nothing else, so a8T is a8F "
                         "plus exactly one difference. It is the allocation lever, orthogonal to "
                         "--reweight (which repeats rows in an oversized pool between cycles) and "
                         "usable with any surface rule. 'bandfilter' is the NAIVE difficulty-band "
                         "baseline (a8T2n): same allocator, same warm bank, same gamma, same "
                         "budget, same everything -- and a point-estimate keep-if-p_hat-in-band "
                         "rule with a uniform draw instead of the posterior g(p,k) weights. It "
                         "exists to say whether the posterior machinery is load-bearing. "
                         "'vip' and 'progress' are the two PUBLISHED-METHOD baselines (VIP, ICLR "
                         "2026, arXiv:2602.01601: weight = the posterior form E[p(1-p)] of the "
                         "only task-dependent factor in its gradient-variance objective; "
                         "learning-progress curricula, Graves 2017 / TSCL: weight = |delta p^| "
                         "over the two most recent evidence windows). Same allocator, same warm "
                         "bank, same gamma, eps, budget, seed and stats line as the method arm -- "
                         "each differs from it in the weight rule and in nothing else, which is "
                         "the only way a baseline comparison is about the rule. "
                         "'budget' is TRIAGE-B: the same posterior allocating the ROLLOUT BUDGET "
                         "rather than the task list -- task i takes m_i parquet ROWS (= m_i "
                         "independent groups of k, because verl stamps one uuid per row) with "
                         "sum_i m_i equal to the same row budget every other arm uses, so total "
                         "generation is matched to the rollout. Tuned by TRIAGE_BUDGET_OBJECTIVE "
                         "/ TRIAGE_BUDGET_DRAW / TRIAGE_KMAX; see triage.py's TRIAGE-B block, "
                         "which records that at the matched budget on this pool the optimal "
                         "allocation is provably fixed-k selection.")
    # TRIAGE v2. Both are forwarded verbatim to triage.py and both default to v1 behaviour, so an
    # arm carrying neither is the arm a8T runs. See the v2 block in triage.py for what they fix.
    ap.add_argument("--warm-bank", default=None, metavar="GLOB",
                    help="TRIAGE v2: warm-start the per-task Beta priors from the OTHER arms' "
                         "banked episodes ('default' = every work/verl/run_*/episodes.jsonl except "
                         "this arm's). Kills the explore tax: without it the first ~5 cycles are "
                         "spent probing 1127 tasks under flat-prior optimism and the mechanism "
                         "never acts inside the measured window.")
    ap.add_argument("--temp", type=float, default=None, metavar="TAU",
                    help="TRIAGE v2: sample proportional to w^(1/tau) instead of w. 0.5 is the v2 "
                         "value; unset = 1.0 = v1.")
    # TRIAGE v3. Same contract as the v2 pair: forwarded verbatim, default to v2 behaviour, and
    # each has an env fallback so an ablation is one supervisor variable away from the method.
    ap.add_argument("--exclude-tasks", default=None, metavar="JSONL",
                    help="TRIAGE v3: certified-defective tasks the TRAINING allocation refuses "
                         "(w=0 and out of the eps-uniform floor). See triage.py --exclude-tasks. "
                         "The eval and validation pools are not touched by this and must not be.")
    ap.add_argument("--warm-shrink", choices=["none", "capability", "group"], default=None,
                    help="TRIAGE v3: 'group' scores the posterior against the measured "
                         "within-group correlation instead of assuming k independent rollouts "
                         "(what a8T3 runs); 'capability' rescales the warm prior mean by "
                         "base/bank and is the measured-negative ablation. Unset = v2.")
    # TRIAGE v4. The estimator axis. Same contract again: forwarded verbatim, defaults to v3.
    ap.add_argument("--estimator", choices=["posterior", "point", "taskrho"], default=None,
                    help="TRIAGE v4: 'point' computes the gradmass weight from the PLUG-IN "
                         "g(p_hat,k) at the posterior mean instead of the posterior expectation "
                         "E[g(p,k)] -- TRACE's value-net estimator on our posterior's evidence. "
                         "It is the estimator comparison the novelty claim rests on, and it "
                         "applies to --task-alloc gradmass only. Unset = v3 = posterior. "
                         "TRIAGE-rho: 'taskrho' is the same closed form at a PER-TASK within-group "
                         "correlation read from --task-rho (which it requires).")
    ap.add_argument("--task-rho", default=None, metavar="JSONL",
                    help="TRIAGE-rho: the per-task correlation file built by fit_task_rho.py. "
                         "Only read under --estimator taskrho. Env fallback TRIAGE_TASK_RHO.")
    a = ap.parse_args()

    # A v2/v3 flag that silently does nothing is how an arm ends up wearing a name it is not
    # running.
    if ((a.warm_bank or a.temp is not None or a.exclude_tasks or a.warm_shrink)
            and a.task_alloc not in ("gradmass", "bandfilter", "vip", "progress", "budget")):
        print("[coadapt] FATAL: --warm-bank/--temp/--exclude-tasks/--warm-shrink only mean "
              "anything to the TRIAGE allocator; pass --task-alloc gradmass or drop them.",
              flush=True)
        return 1
    # The estimator is the gradmass weight; under any other rule the sampler reads that rule's own
    # score and the flag would be a name the arm is not running. triage.py refuses it too, but
    # refusing at argv here means the pod is never held for it.
    if a.estimator in ("point", "taskrho") and a.task_alloc != "gradmass":
        print(f"[coadapt] FATAL: --estimator {a.estimator} changes the gradmass weight; "
              f"--task-alloc {a.task_alloc} samples on its own score.", flush=True)
        return 1
    # TRIAGE-rho: the per-task correlation file is what makes the arm TRIAGE-rho rather than the
    # v4 point-estimator ablation. A missing file would leave every task on the global rho and the
    # arm would run a8Tpe's method under a8T8c's name -- the same failure mode as a missing
    # exclusion list. Refuse at argv, before the pod is held.
    _trho = a.task_rho or os.environ.get("TRIAGE_TASK_RHO") or ""
    if a.estimator == "taskrho" and not _trho:
        print("[coadapt] FATAL: --estimator taskrho needs --task-rho (or TRIAGE_TASK_RHO); "
              "without it every task falls back to the global rho and the arm is --estimator "
              "point wearing the TRIAGE-rho name.", flush=True)
        return 1
    if _trho and not os.path.exists(_trho):
        print(f"[coadapt] FATAL: --task-rho {_trho} does not exist", flush=True)
        return 1
    if _trho and a.estimator != "taskrho":
        print(f"[coadapt] FATAL: --task-rho is only read by --estimator taskrho; under "
              f"--estimator {a.estimator or 'posterior'} it would not change the allocation.",
              flush=True)
        return 1
    # VIP cannot skip a prompt: its feasible set is L <= n_q <= U with L >= 3. A VIP baseline
    # carrying an exclusion list would be stronger than the paper it is named after.
    if a.task_alloc == "vip" and (a.exclude_tasks or os.environ.get("TRIAGE_EXCLUDE")):
        print("[coadapt] FATAL: --task-alloc vip cannot take an exclusion list (VIP's L>=3 "
              "structurally forbids giving a prompt zero rollouts).", flush=True)
        return 1
    # TRIAGE-B multiplies parquet ROWS, and the decorrelation permutation is keyed on a counter
    # over consecutive rollouts of one (scenario, task_idx) -- m_i > 1 rows of a task in one batch
    # runs that counter past the permutation family's size. triage.py refuses it too; refusing at
    # argv here means the pod is never held for it.
    if a.task_alloc == "budget" and os.environ.get("TRIAGE_DECOR"):
        print("[coadapt] FATAL: --task-alloc budget is not composable with TRIAGE_DECOR "
              f"({os.environ['TRIAGE_DECOR']}): row multiplicity breaks the rollout-index counter "
              "the permutation family is sized against.", flush=True)
        return 1
    # An exclusion list that is not there is not an empty exclusion list: it is a v3 arm silently
    # training as a v2 arm under the v3 name. Refuse at argv, before the pod is held.
    if a.exclude_tasks and not os.path.exists(a.exclude_tasks):
        print(f"[coadapt] FATAL: --exclude-tasks {a.exclude_tasks} does not exist", flush=True)
        return 1

    # Two rules that both rewrite the same parquet cannot both be in force. Whichever ran last
    # would win, and the arm would carry the name of the other -- so refuse at argv rather than
    # produce a result nobody can attribute. (--surface-elsa implies its own reweighting.)
    if a.task_alloc != "none" and (a.reweight != "none" or a.surface_elsa):
        print("[coadapt] FATAL: --task-alloc rebuilds the training parquet, and so do --reweight "
              "and --surface-elsa. Pass exactly one.", flush=True)
        return 1

    ckpt = f"{R}/work/verl/ckpt_{a.tag}"
    surf = f"{R}/work/coadapt_{a.tag}"
    os.makedirs(surf, exist_ok=True)

    # Inside its own allocation the driver already holds the GPU, and a nested srun step is
    # refused by the scheduler without ever running the cell -- which is exactly how the first
    # launch of this arm produced no output at all. Run the cell directly in that case.
    inside = os.environ.get("SLURM_JOB_ID") == a.jobid

    def cell(script: str) -> None:
        if inside:
            subprocess.run(["bash", "-c", script], check=False)
        else:
            # the pod holds 128G/8cpu; a 64G step cap killed the vLLM worker mid memory-probe
            subprocess.run(["srun", "--jobid", a.jobid, "--ntasks=1", "--overlap", "--exact",
                            "--gres=gpu:1", "--cpus-per-task=8", "--mem=120G",
                            "bash", "-c", script], check=False)

    # The surface in force: rewritten after every accepted edit, read by the trainer through
    # AWM_ADVERTISED and by the next gate through --init-advertised. It is the one piece of state
    # the two halves share, and it is what makes this co-adaptation rather than two loops run
    # side by side.
    adv = os.path.join(surf, "advertised.txt")
    if not os.path.exists(adv):
        full, start = handicapped_surface(a.pool, a.handicap_frac, a.handicap_seed)
        for path in (adv, os.path.join(surf, "advertised_init.txt")):
            with open(path, "w") as fh:      # the init copy is never rewritten: it is the
                fh.write("\n".join(start) + "\n")   # baseline arm of the final 2x2 comparison
        print(f"[coadapt] handicap: cycle 1 advertises {len(start)}/{len(full)} names, "
              f"{len(full) - len(start)} withheld for the gate to earn back", flush=True)

    data = f"{R}/work/verl/awm_{a.tag}"
    if not os.path.exists(f"{data}/train.parquet"):
        print("[coadapt] building the training parquet from the shared pool", flush=True)
        cell(f". {R}/env.sh; "
             f"export PYTHONPATH={R}/surface/gate_surface:{R}/surface/gate_caller "
             # HF_HUB_OFFLINE=1 for the same reason triage.py sets it on the rebuild path and
             # verl_awm_train.sh:15 sets it for the trainer: prep_awm.py needs only the cached
             # tokenizer, but transformers reaches the Hub anyway for any non-local model id, and
             # a 429 there kills this build. This is the identical hole to the one that took out
             # q2bLp at cycle 1, on the FIRST-boot path rather than the rebuild path.
             f"GATE_MAX_TURNS=20 GATE_MAX_NEW_TOKENS=1024 CALLER_FAMILY=qwen HF_HUB_OFFLINE=1; "
             f"{VERL_PY} "
             f"{R}/surface/verl_rl/prep_awm.py --pool {a.pool} --out {data}")
        if not os.path.exists(f"{data}/train.parquet"):
            print("[coadapt] FATAL: no training parquet", flush=True)
            return 1

    rw_done: set[int] = set()

    def reweight_pool(cycle: int) -> None:
        """Resample the training pool toward tasks that can still produce a gradient.

        This must be reachable from BOTH exits of a cycle. It used to sit only after the gate,
        below the `--gate none` early-continue, so every reweighting baseline -- which by
        construction runs with --gate none, as the supervisor launches all queued arms that way
        -- trained on the untouched pool. Those arms were exact duplicates of the frozen-surface
        control while printing nothing to say so, which is the worst failure mode available: a
        baseline that looks enabled and silently does nothing.

        k must be the group size actually in force, since availability g(p,k)=1-p^k-(1-p)^k is a
        function of it; an arm that raises NROLL and reweights on k=5 optimises the wrong target.
        """
        if a.reweight == "none" and not a.surface_elsa:
            return
        # ELSA reweights BEFORE training, from inside the levels branch, because its prices are
        # conditioned on the map that pass just wrote. The tail call every other arm relies on
        # would then rebuild the parquet a second time from the same weights -- same result, one
        # wasted parquet build per cycle. Idempotent per cycle, whichever call arrives first.
        if cycle in rw_done:
            return
        rw_done.add(cycle)
        k = group_size()
        mode, extra = a.reweight, ""
        if a.surface_elsa:
            # ELSA's prices were written by the same pass that chose this cycle's surface. If that
            # file is absent the levels pass did not run, and reweighting by anything else here
            # would turn the method arm into the AWS baseline under the method's name.
            wf = os.path.join(surf, "elsa_weights.json")
            if not os.path.exists(wf):
                print(f"[coadapt] cycle {cycle}: no ELSA weights at {wf}; skipping the reweight "
                      f"rather than substituting a different rule", flush=True)
                return
            mode, extra = "external", f" --weights-file {wf}"
        # aws_reweight expresses a weight as the number of times a task is repeated in the
        # training file, reps = max(1, round(slots * share)). With the default slots=200 against
        # a 320-task pool every share rounds to 1 and the floor pins it there, so the "reweighted"
        # pool comes back as an exact permutation of the base pool -- measured: rows=320,
        # rep histogram {1: 320}, identical for availability and variance. The weights need room
        # to separate, so give the multiset several rows per task; at 8x, a top-weighted task
        # draws 8 rows against 1 for a task that never produces a gradient.
        slots = 8 * len(json.load(open(a.pool)))
        print(f"[coadapt] cycle {cycle}: reweighting the pool by {mode} "
              f"(k={k}, slots={slots})", flush=True)
        cell(f". {R}/env.sh; "
             f"export PYTHONPATH={R}/surface/gate_surface:{R}/surface/gate_caller; "
             f"{VERL_PY} "
             f"{R}/surface/verl_rl/aws_reweight.py "
             f"--episodes {R}/work/verl/run_{a.tag}/episodes.jsonl "
             f"--base-pool {a.pool} "
             f"--out-pool {R}/work/coadapt_{a.tag}/pool_cycle{cycle}.json "
             f"--out-data {data} --k {k} --slots {slots} --weighting {mode}{extra}")

    alloc_done: set[int] = set()

    def alloc_pool(cycle: int) -> bool:
        """TRIAGE: replace the parquet with THIS cycle's sampled task list, before generation.

        Runs BEFORE the training cell, unlike --reweight, which runs after the cycle it belongs to
        and therefore prices the NEXT one. That ordering is not a detail: the entire claim is that
        the allocation is decided from free statistics before any rollout is paid for, which is
        what separates it from DAPO's filter-after-generating.

        The budget is the cycle's generation budget exactly -- steps_per_cycle * train_batch_size
        rows -- so verl's dataloader consumes the sampled list once and nothing else. BSZ comes
        from the environment because that is where the supervisor sets it; a mismatch would make
        the parquet either run out mid-cycle or leave part of the allocation ungenerated.
        """
        if a.task_alloc not in ("gradmass", "bandfilter", "vip", "progress", "budget"):
            return True
        if cycle in alloc_done:
            return True
        bsz = int(os.environ.get("BSZ", "8"))
        budget = a.steps_per_cycle * bsz
        # k IS THE TRAINER'S GROUP SIZE, FROM THE ONE PLACE THAT SETS IT. verl_awm_train.sh reads
        # NROLL=${NROLL:-5} and passes it to actor_rollout_ref.rollout.n, and the supervisor exports
        # NROLL into this same environment, so reading it here with the SAME default is the only
        # way the allocator and the generator cannot disagree. Verified against the composed Hydra
        # config in logs/rl_a8T.log and logs/rl_a8Te0.log: rollout n = 5 on every TRIAGE arm.
        # triage.py's --k is required precisely so this cannot be left to a default at the far end.
        k = group_size()
        stats = os.path.join(surf, "triage_stats.jsonl")
        state = os.path.join(surf, "triage_state.json")
        # Env is the ablation channel (one supervisor env var = one arm); an explicit flag is the
        # arm-definition channel. The flag wins, so a8T2's --temp 0.5 cannot be shadowed by a stray
        # TRIAGE_TAU left in an environment.
        warm = a.warm_bank or os.environ.get("TRIAGE_WARM_BANK") or ""
        tau = a.temp if a.temp is not None else os.environ.get("TRIAGE_TAU")
        excl = a.exclude_tasks or os.environ.get("TRIAGE_EXCLUDE") or ""
        shrink = a.warm_shrink or os.environ.get("TRIAGE_WARM_SHRINK") or ""
        est = a.estimator or os.environ.get("TRIAGE_ESTIMATOR") or ""
        # TRIAGE-rho. Forwarded only under the estimator that reads it: triage.py refuses the flag
        # under any other estimator, so an unconditional forward would turn a stray supervisor
        # variable into a dead arm at argv.
        trho = (a.task_rho or os.environ.get("TRIAGE_TASK_RHO") or "") if est == "taskrho" else ""
        if est == "taskrho" and not os.path.exists(trho):
            print(f"[coadapt] FATAL: TRIAGE_TASK_RHO/{trho} does not exist; refusing to allocate "
                  f"without the per-task correlation file this arm is defined by", flush=True)
            return False
        # TRIAGE-B knobs. Env-only, like every other ablation knob, and read ONLY under
        # --task-alloc budget: triage.py refuses them under any other rule, so forwarding them
        # unconditionally would turn a stray supervisor variable into a dead arm at argv.
        b_obj = (os.environ.get("TRIAGE_BUDGET_OBJECTIVE") or "") if a.task_alloc == "budget" else ""
        b_draw = (os.environ.get("TRIAGE_BUDGET_DRAW") or "") if a.task_alloc == "budget" else ""
        b_kmax = (os.environ.get("TRIAGE_KMAX") or "") if a.task_alloc == "budget" else ""
        # The env fallback can point at a file that was moved since the supervisor set it; the
        # flag path is checked at argv, so check this one where it is resolved. Same reason:
        # a missing list makes a v3 arm run as v2 under the v3 name.
        if excl and not os.path.exists(excl):
            print(f"[coadapt] FATAL: TRIAGE_EXCLUDE/{excl} does not exist; refusing to allocate "
                  f"without the exclusion list this arm is defined by", flush=True)
            return False
        print(f"[coadapt] cycle {cycle}: TRIAGE allocating {budget} rows "
              f"(rule={a.task_alloc}, k={k}, {a.steps_per_cycle} steps x BSZ {bsz}"
              + (f", warm-bank {warm}" if warm else "") + (f", tau {tau}" if tau else "")
              + (f", exclude {excl}" if excl else "")
              + (f", warm-shrink {shrink}" if shrink else "")
              + (f", estimator {est}" if est else "")
              + (f", task-rho {trho}" if trho else "")
              + (f", budget-objective {b_obj}" if b_obj else "")
              + (f", budget-draw {b_draw}" if b_draw else "")
              + (f", k-budget-max {b_kmax}" if b_kmax else "")
              + ")", flush=True)
        cell(f". {R}/env.sh; "
             f"export PYTHONPATH={R}/surface/gate_surface:{R}/surface/gate_caller; "
             f"{VERL_PY} "
             f"{R}/surface/verl_rl/triage.py "
             f"--episodes {R}/work/verl/run_{a.tag}/episodes.jsonl "
             f"--base-pool {a.pool} "
             f"--out-pool {surf}/pool_cycle{cycle}.json "
             f"--out-data {data} --state {state} --stats {stats} "
             f"--cycle {cycle} --budget {budget} --k {k} --seed {a.seed_offset}"
             # Ablation knobs, env-driven so an arm differs from a8T by ONE supervisor env var
             # and nothing else. Unset = triage.py's own defaults = a8T exactly.
             + (f" --decay {os.environ['TRIAGE_DECAY']}" if os.environ.get("TRIAGE_DECAY") else "")
             + (f" --eps {os.environ['TRIAGE_EPS']}" if os.environ.get("TRIAGE_EPS") else "")
             + (f" --warm-bank {warm}" if warm else "")
             + (f" --temp {tau}" if tau else "")
             + (f" --exclude-tasks {excl}" if excl else "")
             + (f" --warm-shrink {shrink}" if shrink else "")
             # 'gradmass' is triage.py's own default and is deliberately NOT spelled out: the
             # command line the arms in flight build has to stay the one they have always built.
             + (f" --rule {a.task_alloc}" if a.task_alloc != "gradmass" else "")
             + (f" --estimator {est}" if est else "")
             + (f" --task-rho {trho}" if trho else "")
             + (f" --band-lo {os.environ['TRIAGE_BAND_LO']}"
                if os.environ.get("TRIAGE_BAND_LO") else "")
             + (f" --band-hi {os.environ['TRIAGE_BAND_HI']}"
                if os.environ.get("TRIAGE_BAND_HI") else "")
             + (f" --shrink-nu {os.environ['TRIAGE_NU']}" if os.environ.get("TRIAGE_NU") else "")
             + (f" --shrink-lambda {os.environ['TRIAGE_LAMBDA']}"
                if os.environ.get("TRIAGE_LAMBDA") else "")
             + (f" --budget-objective {b_obj}" if b_obj else "")
             + (f" --budget-draw {b_draw}" if b_draw else "")
             + (f" --k-budget-max {b_kmax}" if b_kmax else "")
             + (" --vark" if os.environ.get("VARK", "") == "1" else ""))
        # cell() cannot fail loudly -- it is check=False by design, because a gate cell that dies
        # must not take the driver with it. So verify the allocation actually landed. An arm that
        # silently trains on the previous cycle's parquet, or on the full pool, is an unlabelled
        # duplicate of a8F wearing the method's name; that failure has already cost this project
        # two arms (a8C's overwritten envmap, the reweighting baselines below the early-continue).
        try:
            built = json.load(open(state)).get("cycles_built", [])
        except Exception:
            built = []
        if cycle not in built:
            print(f"[coadapt] FATAL: TRIAGE produced no allocation for cycle {cycle}; refusing to "
                  f"train on a stale pool and mislabel it as the method arm", flush=True)
            return False
        alloc_done.add(cycle)
        return True

    c = 1
    while c <= a.cycles:
        target = c * a.steps_per_cycle
        print(f"[coadapt] cycle {c}: training to step {target} on the current harness "
              f"({'carried surface' if os.path.exists(adv) else 'unedited substrate'})", flush=True)
        if not alloc_pool(c):
            return 1
        envmap = ""
        accord_cert = ""
        if a.accord:
            # Re-certify BEFORE training, so this cycle's rollouts are shaped by a certificate
            # derived from every episode banked so far. Cycle 1 has no episodes and no certificate,
            # which dense_reward treats as "refuse": the arm trains on the plain binary reward
            # until there is evidence to justify shaping, rather than shaping on faith.
            cert = os.path.join(surf, "accord_cert.json")
            cell(f"{VERL_PY} "
                 f"{R}/surface/verl_rl/accord.py "
                 f"--episodes {R}/work/verl/run_{a.tag}/episodes.jsonl --out {cert}")
            # NOTE: do NOT touch envmap here. Every surface branch below ASSIGNS to it
            # (envmap = " AWM_ADVERTISED_MAP=..."), so anything appended first is silently
            # destroyed. That is exactly what happened to a8C: the certificate was written and
            # certified on disk, the export was overwritten by --surface-fixed, the workers saw
            # no AWM_ACCORD_CERT, and dense_reward's "no certificate configured" path shaped
            # EVERY scenario -- making the method arm bit-identical to its own uncertified
            # ablation while its log said ACCORD. Appended once, after the branches, instead.
            accord_cert = cert if os.path.exists(cert) else ""
        if a.surface_fixed is not None:
            lv = os.path.join(surf, "levels.json")
            mp = os.path.join(surf, "surface_map.json")
            pool_scens = sorted({p["scenario"] for p in json.load(open(a.pool))})
            json.dump({sc: a.surface_fixed for sc in pool_scens}, open(lv, "w"))
            # band 2.0 makes every observation fall inside the deadband, so levels never move
            cell(f"{VERL_PY} "
                 f"{R}/surface/verl_rl/surface_control.py "
                 f"--episodes /dev/null --pool {a.pool} "
                 f"--init-surface {os.path.join(surf, 'advertised_init.txt')} "
                 f"--levels {lv} --out-map {mp} --band 2.0")
            envmap = f" AWM_ADVERTISED_MAP={mp}"
        elif a.surface_elsa:
            # Runs BEFORE training, like every other surface rule, so this cycle rolls out against
            # the surface it just chose AND samples at the prices that surface implies. One pass
            # writes both: the levels are what the prices are conditioned on, so a second process
            # reading a stale map would decouple the two levers while still printing "elsa".
            lv = os.path.join(surf, "levels.json")
            mp = os.path.join(surf, "surface_map.json")
            wf = os.path.join(surf, "elsa_weights.json")
            if not os.path.exists(lv):
                open(lv, "w").write("{}")
            cell(f"{VERL_PY} "
                 f"{R}/surface/verl_rl/elsa.py "
                 f"--episodes {R}/work/verl/run_{a.tag}/episodes.jsonl --pool {a.pool} "
                 f"--init-surface {os.path.join(surf, 'advertised_init.txt')} "
                 f"--levels {lv} --out-map {mp} --out-weights {wf} "
                 f"--k {group_size()} {a.elsa_extra}")
            if not os.path.exists(mp):
                print("[coadapt] FATAL: ELSA produced no surface map; refusing to train on the "
                      "flat surface and mislabel it as the method arm", flush=True)
                return 1
            reweight_pool(c)                  # apply the prices to THIS cycle, not the next one
            envmap = f" AWM_ADVERTISED_MAP={mp}"
        elif a.surface_control_v2:
            lv = os.path.join(surf, "levels.json")
            mp = os.path.join(surf, "surface_map.json")
            if not os.path.exists(lv):
                open(lv, "w").write("{}")
            cell(f"{VERL_PY} "
                 f"{R}/surface/verl_rl/surface_control_v2.py "
                 f"--episodes {R}/work/verl/run_{a.tag}/episodes.jsonl --pool {a.pool} "
                 f"--init-surface {os.path.join(surf, 'advertised_init.txt')} "
                 f"--levels {lv} --out-map {mp}")
            envmap = f" AWM_ADVERTISED_MAP={mp}"
        elif a.surface_control:
            # Re-tune BEFORE training so this cycle rolls out at the frontier. Uses only episodes
            # already produced, so it costs a parquet-sized rewrite and no extra generation.
            lv = os.path.join(surf, "levels.json")
            mp = os.path.join(surf, "surface_map.json")
            if not os.path.exists(lv):
                open(lv, "w").write("{}")
            cell(f"{VERL_PY} "
                 f"{R}/surface/verl_rl/surface_control.py "
                 f"--episodes {R}/work/verl/run_{a.tag}/episodes.jsonl --pool {a.pool} "
                 f"--init-surface {os.path.join(surf, 'advertised_init.txt')} "
                 f"--levels {lv} --out-map {mp}")
            envmap = f" AWM_ADVERTISED_MAP={mp}"
        elif a.surface_retrieval:
            # Retrieval is a STATIC rule: it reads task text and tool docs, never rollouts, so the
            # map is the same every cycle. It is still rebuilt each cycle rather than once,
            # because "written once at cycle 1" is exactly the state a restarted arm loses -- and
            # an arm whose AWM_ADVERTISED_MAP points at a file that does not exist falls back to
            # the flat surface silently and becomes an unlabelled copy of another arm.
            lv = os.path.join(surf, "levels.json")
            mp = os.path.join(surf, "surface_map.json")
            cell(f". {R}/env.sh; "
                 f"{VERL_PY} "
                 f"{R}/surface/verl_rl/surface_retrieval.py --pool {a.pool} "
                 f"--init-surface {os.path.join(surf, 'advertised_init.txt')} "
                 f"--levels {lv} --out-map {mp} "
                 f"--match-level {a.retrieval_match_level}")
            if not os.path.exists(mp):
                print("[coadapt] FATAL: retrieval produced no surface map; refusing to train on "
                      "the flat surface and mislabel it as the retrieval arm", flush=True)
                return 1
            envmap = f" AWM_ADVERTISED_MAP={mp}"
        elif a.surface_accel:
            # Re-evolve BEFORE training, from the rollouts already banked, so this cycle rolls out
            # against the mutated surface -- the same ordering --surface-control uses, so the two
            # arms see a surface of the same age and the comparison is not a one-cycle lag.
            lv = os.path.join(surf, "levels.json")
            mp = os.path.join(surf, "surface_map.json")
            stt = os.path.join(surf, "accel_state.json")
            cell(f"{VERL_PY} "
                 f"{R}/surface/verl_rl/accel_surface.py "
                 f"--episodes {R}/work/verl/run_{a.tag}/episodes.jsonl --pool {a.pool} "
                 f"--init-surface {os.path.join(surf, 'advertised_init.txt')} "
                 f"--levels {lv} --out-map {mp} --state {stt} --seed {a.seed_offset}")
            if not os.path.exists(mp):
                print("[coadapt] FATAL: accel produced no surface map", flush=True)
                return 1
            envmap = f" AWM_ADVERTISED_MAP={mp}"
        # AWM_RL_MAX_SLOTS was pinned to 1 here, overriding the trainer default of 6. A median
        # episode takes 14.2s to generate ~554 tokens over 6 turns, so ~9s of it is per-turn
        # round-trip latency (uvicorn/FastAPI, MCP, SQLAlchemy) that faster storage does not
        # remove -- node-local NVMe measured 630 eps/hr against 603 on NFS. Latency like that is
        # hidden by concurrency, and with 1 slot and 2 agent workers only ~2 episodes are ever in
        # flight. The old 6-slot canary was slower (180 vs 274 eps/hr) because each extra slot
        # then leaked a whole server pool into a 128G cgroup; PR_SET_PDEATHSIG in server.py fixed
        # that, so the finding no longer applies. Default stays 1 until the A/B says otherwise.
        if a.accord and accord_cert:
            envmap += f" AWM_ACCORD_CERT={accord_cert}"
        slots = os.environ.get("AWM_SLOTS_TRAIN", "1")
        # GPU_RELEASE_KEEP: DEFAULTED HERE SO THE 2026-08-24 07:30 INCIDENT CANNOT RECUR.
        # gpu_release.sh kills EVERY compute-app pid in the job cgroup unless this regex matches
        # the pid or one of its ancestors, and it shipped unset for every arm. On 2026-08-24 that
        # swept q4bTa's own pod 63549959 sixty seconds after a clean boot: it killed the torch
        # keepalive HOLDING THE ALLOCATION (pid 4088867) and, worse, the crossworld project's
        # dreamer.py (pid 768485) -- another group's job, on a shared node. gpu_release.sh's own
        # header predicts exactly this ("killing the keepalive is how the pod itself gets reaped
        # by the idle guard"); the guard existed and nothing set it.
        #
        # The default spares allocation keepalives and foreign workloads while still sweeping OUR
        # trainer, which is the only thing the release is for. Overridable by env for the case
        # where a stranded engine really does need the broader sweep -- but the safe value is now
        # the one you get by doing nothing, which is the opposite of the arrangement that failed.
        keep = os.environ.get(
            "GPU_RELEASE_KEEP",
            r"gpu_keepalive|crossworld|dreamer\.py|venv_track6dreamer|import torch, time",
        )
        cell(f"export GPU_RELEASE_KEEP={shlex.quote(keep)}; bash {R}/slurm/gpu_release.sh; "
             f"export TAG={a.tag} STEPS={target} AWM_RL_MAX_SLOTS={slots} AWM_ADVERTISED={adv}{envmap}; "
             f"bash {R}/slurm/h200_arm.sh {a.tag}")
        if latest_step(ckpt) < target:
            print(f"[coadapt] training did not reach {target} (got {latest_step(ckpt)}); "
                  f"retrying this cycle", flush=True)
            continue
        # ---- GATE C: the trainer must PROVE it honoured the per-row group sizes --------------
        # Checked HERE, after the cycle's steps actually ran -- at allocation time the trainer has
        # not yet consumed the parquet, so there is nothing to check. The allocator can emit a
        # perfect variable-k pool and the trainer can still ignore it: the monkeypatch lives in a
        # Ray actor, and if sitecustomize failed to arm there the run would train every task at
        # the uniform rollout.n while every artifact -- composed command, pool, stats -- said
        # variable k. Same treatment as the allocation guard: refuse to continue.
        if os.environ.get("VARK", "") == "1":
            ok, why = vark_trainer_agrees(
                c, a.steps_per_cycle,
                os.path.join(surf, "triage_stats.jsonl"), f"{R}/logs/rl_{a.tag}.log")
            if not ok:
                print(f"[coadapt] FATAL: VARK=1 but the trainer did not honour the per-row group "
                      f"sizes for cycle {c} ({why}). Refusing to continue: training at uniform k "
                      f"while labelled the variable-k arm is the failure this gate exists for.",
                      flush=True)
                return 1
        # A restart must not redo a gate that already returned a verdict. Without this the
        # driver re-runs cycle 1's full-pool evaluation (640 episodes) on every relaunch, and an
        # arm that gets restarted a few times never advances past its first cycle at all.
        done_marker = os.path.join(surf, f"cycle{c}", "accepted_surface.json")
        hist_marker = os.path.join(surf, f"cycle{c}",
                                   f"history_{'naive' if a.gate == 'naive' else 'discord'}_awm.json")
        if a.gate != "none" and (os.path.exists(done_marker) or os.path.exists(hist_marker)):
            print(f"[coadapt] cycle {c}: gate already decided, skipping to the next cycle",
                  flush=True)
            c += 1
            continue
        if a.gate == "none":
            # The matched baseline: same driver, same pool, same surface, same horizon -- the
            # harness simply never changes. Running it through this code path rather than a
            # separate script keeps the baseline one flag away from the method instead of one
            # script away, which is both cleaner scientifically and avoids maintaining a second
            # launch path that fails in its own ways.
            print(f"[coadapt] cycle {c}: frozen-harness baseline, no edit proposed", flush=True)
            reweight_pool(c)
            c += 1
            continue
        print(f"[coadapt] cycle {c}: proposing a harness edit and gating it with {a.gate}",
              flush=True)
        cell(f"cd /tmp && bash {R}/slurm/coadapt_gate_cell.sh {a.tag} {c} "
             f"{a.handicap_frac} {a.edit_block} {adv} {a.gate} {a.seed_offset}")
        # accepted_surface.json is written at the instant of acceptance; the history file only
        # lands if the whole gate run finishes. Prefer the former or a crash after the decision
        # throws the decision away.
        cyc = os.path.join(surf, f"cycle{c}")
        acc = os.path.join(cyc, "accepted_surface.json")
        hist = os.path.join(cyc, f"history_{'naive' if a.gate == 'naive' else 'discord'}_awm.json")
        src = acc if os.path.exists(acc) else hist
        if os.path.exists(src):
            names = json.load(open(src)).get("advertised_names") or []
            if names:
                with open(adv, "w") as fh:
                    fh.write("\n".join(names) + "\n")
                print(f"[coadapt] cycle {c}: surface now advertises {len(names)} names", flush=True)
        else:
            print(f"[coadapt] cycle {c}: gate produced no history; surface unchanged", flush=True)
        # Reweight AFTER the surface update, so the next cycle samples for availability under
        # the harness the gate just certified. The two halves compound here: an edit makes
        # more tasks solvable, and this concentrates training on the ones that became
        # gradient-bearing rather than on the ones that are still hopeless.
        reweight_pool(c)
        c += 1
    print("[coadapt] DONE", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
