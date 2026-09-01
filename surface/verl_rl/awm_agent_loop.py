"""verl agent loop over our existing AWM/MCP environment.

WHY AN AGENT LOOP AND NOT A `BaseTool`
--------------------------------------
The obvious route -- `verl.tools.base_tool.BaseTool` + a tools YAML +
`multi_turn.enable=True` -- does not fit this environment in verl 0.8.0. Three
things were read out of the installed source, not assumed:

1. `ToolAgentLoop._call_tool` runs `create()` -> `execute()` -> `release()`
   around EVERY SINGLE tool call. There is no per-trajectory lifecycle, so a
   `BaseTool` cannot hold a live AWM server across the 20 turns of an episode,
   and `release()` cannot mean "episode over, score it".
2. `BaseTool.calc_reward` is dead code in 0.8.0: nothing in the package calls
   it. Terminal reward from tool state has no supported path.
3. Tool schemas are registered GLOBALLY and keyed by name
   (`self.tools = {tool.name: tool for tool in tool_list}`). Per-sample
   selection (`extra_info.tool_selection`) only FILTERS that global dict. Our
   surfaces collide: across the 19 scenarios of pool_smoke20 there are 650 tools,
   and 23 names appear in more than one scenario WITH DIFFERENT SCHEMAS
   (e.g. `create_booking` in booking_scheduling_10 and marketplace_10). A global
   registry would silently serve the wrong schema for those, and for a 300-task
   run it would mean registering ~10k tools up front.

`AgentLoopBase` has none of those problems: one `run()` call == one episode, we
own the turn loop, tool schemas are built PER SAMPLE from that scenario's own
surface, and `AgentLoopOutput.reward_score` is written straight into `rm_scores`
(agent_loop.py `as_dict`), which bypasses the reward manager entirely.

WHAT IS REUSED UNCHANGED
------------------------
`AwmServer` (boot/restore/stop), `ToolRuntime` (persistent MCP sessions),
`interfaces.build` (the RAW surface + dispatch), `run_gate.parse_call` +
`parse_multi._fallback` (the frozen parser), and `run_verifier`. No MCP client
is reimplemented here and the verifier is the same pure-Python state-based one
the screening gates use. The turn loop mirrors `run_gate.py`.

SLOT POOL
---------
Booting an AWM server costs ~6 s; restoring its .db costs ~0.4 s (measured,
gate_surface/probe_awm_pool.py). So servers are POOLED per scenario for the whole
run and only the .db is restored between episodes -- exactly the slot model
run_gate.py already uses. Pool size is capped and evicted LRU so a many-scenario
pool cannot exhaust the 128 G cgroup at ~336 MiB per server.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import shutil
import sys
import threading
import time
import uuid
from typing import Any

# --- our harness ---------------------------------------------------------
_MCP_ROOT = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_SURFACE = os.path.join(_MCP_ROOT, "surface")
for _p in (os.path.join(_SURFACE, "gate_surface"), os.path.join(_SURFACE, "gate_caller")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import interfaces as IF                                          # noqa: E402
import parse_multi                                               # noqa: E402
import run_gate                                                  # noqa: E402
from awm_env import (AwmServer, db_content_digest, free_port,     # noqa: E402
                     load_tasks, load_verifiers, run_verifier)
from tool_runtime import ToolRuntime                             # noqa: E402

# Reward shaping, selected by AWM_DENSE_REWARD. Unset (the default, and the state of every
# existing arm) means _DENSE.enabled() is False and every path below is the original one.
# This module is imported by eight live arms, so an ImportError here would take all of them
# down for a feature none of them use: fall back to a pass-through rather than raise.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dense_reward as _DENSE                                # noqa: E402
except Exception as _exc:                                        # pragma: no cover
    class _DenseDisabled:
        MODE = "none"

        @staticmethod
        def enabled():
            return False

        @staticmethod
        def shaped_reward(binary, *_a, **_k):
            return binary

    _DENSE = _DenseDisabled()
    import logging as _logging                                   # noqa: E402

    _logging.getLogger(__file__).warning(
        "dense_reward unavailable (%s); training on the binary reward", _exc)

# --- verl ----------------------------------------------------------------
from verl.experimental.agent_loop.agent_loop import (            # noqa: E402
    AgentLoopBase, AgentLoopOutput, register)

import logging                                                   # noqa: E402
logger = logging.getLogger(__file__)
logger.setLevel(os.environ.get("VERL_LOGGING_LEVEL", "INFO"))

WORK = os.path.join(_SURFACE, "gate_surface", "work")
DBS = os.path.join(WORK, "dbs")
SURF = os.path.join(WORK, "surfaces")
RUNDIR = os.environ.get("AWM_RL_RUNDIR", os.path.join(_MCP_ROOT, "work", "verl", "awm_run"))

# Where the per-scenario MCP servers keep their SQLite working copies. Defaults to RUNDIR so
# nothing changes unless asked, but RUNDIR is on network scratch and every tool call does random
# reads and writes against those databases: a median episode takes 14.2s to generate 554 tokens
# (~39 tok/s effective), so the model is idle for most of it. Pointing this at node-local disk
# keeps the hot, disposable I/O off NFS while episodes.jsonl below still lands on RUNDIR, where
# it persists past the job.
SRVDIR = os.environ.get("AWM_RL_SRVDIR", RUNDIR)

# The caps must match the screening protocol. run_gate reads these at import; if the
# Ray worker did not inherit them we would silently re-measure at 8/256 and every
# number would be incomparable to the banked gates. Fail loudly instead.
_WANT_TURNS = int(os.environ.get("GATE_MAX_TURNS", "0"))
_WANT_TOKENS = int(os.environ.get("GATE_MAX_NEW_TOKENS", "0"))
if run_gate.MAX_TURNS < 20 or run_gate.MAX_NEW_TOKENS < 1024:
    raise RuntimeError(
        f"caps not raised in this process: MAX_TURNS={run_gate.MAX_TURNS} "
        f"MAX_NEW_TOKENS={run_gate.MAX_NEW_TOKENS}; need 20/1024. Export "
        f"GATE_MAX_TURNS=20 GATE_MAX_NEW_TOKENS=1024 and make sure Ray propagates them "
        f"(trainer ray_init.env_vars or ray_kwargs.runtime_env)."
    )

MAX_TURNS = run_gate.MAX_TURNS
MAX_NEW_TOKENS = run_gate.MAX_NEW_TOKENS
FAMILY = os.environ.get("CALLER_FAMILY", "qwen")
MAX_LIVE_SLOTS = int(os.environ.get("AWM_RL_MAX_SLOTS", "24"))


# =========================================================================
# slot pool -- one per AgentLoopWorker process
# =========================================================================
class _Slot:
    __slots__ = ("scenario", "srv", "key", "busy", "last")

    def __init__(self, scenario: str, srv: AwmServer, key: str):
        self.scenario, self.srv, self.key = scenario, srv, key
        self.busy = False
        self.last = time.time()


class AwmSlotPool:
    """Pooled AWM servers keyed by scenario, shared by all episodes in this process."""

    def __init__(self):
        self.rt = ToolRuntime()
        self.slots: list[_Slot] = []
        self.lock = asyncio.Lock()
        self.workdir = os.path.join(SRVDIR, f"pool_{os.getpid()}")
        os.makedirs(self.workdir, exist_ok=True)
        self.n_boot = 0
        self.n_evict = 0

    async def acquire(self, scenario: str) -> _Slot:
        while True:
            async with self.lock:
                for s in self.slots:
                    if s.scenario == scenario and not s.busy:
                        s.busy = True
                        s.last = time.time()
                        return s
                if len(self.slots) < MAX_LIVE_SLOTS:
                    return await self._boot(scenario)
                # at cap: evict the least-recently-used idle slot of another scenario
                idle = [s for s in self.slots if not s.busy]
                if idle:
                    victim = min(idle, key=lambda s: s.last)
                    self._drop(victim)
                    self.n_evict += 1
                    return await self._boot(scenario)
            # everything busy -- wait for a release
            await asyncio.sleep(0.2)

    async def _boot(self, scenario: str) -> _Slot:
        """Caller must hold self.lock."""
        initial = os.path.join(DBS, f"{scenario}.initial.db")
        db = os.path.join(self.workdir, f"{scenario}_{uuid.uuid4().hex[:8]}.db")
        await asyncio.to_thread(shutil.copy, initial, db)
        srv = AwmServer(scenario, os.path.join(self.workdir, "_srv"), db, port=free_port())
        await asyncio.to_thread(srv.start)
        ok = await asyncio.to_thread(srv.wait_ready, 240)
        if not ok:
            raise RuntimeError(f"AWM server for {scenario} failed to boot: {srv.tail(800)}")
        key = f"{scenario}:{srv.port}"
        await asyncio.to_thread(self.rt.open, key, srv.url, 120)
        slot = _Slot(scenario, srv, key)
        slot.busy = True
        self.slots.append(slot)
        self.n_boot += 1
        logger.info("[awm] booted %s port=%d (live=%d boots=%d evicts=%d)",
                    scenario, srv.port, len(self.slots), self.n_boot, self.n_evict)
        return slot

    def _drop(self, slot: _Slot):
        try:
            self.rt.close(slot.key)
        except Exception:
            pass
        try:
            slot.srv.stop()
        except Exception:
            pass
        for suffix in ("", "-wal", "-shm", "-journal"):
            try:
                os.remove(slot.srv.db_path + suffix)
            except OSError:
                pass
        self.slots = [s for s in self.slots if s is not slot]

    def release(self, slot: _Slot):
        slot.busy = False
        slot.last = time.time()

    def shutdown(self):
        for s in list(self.slots):
            self._drop(s)
        try:
            self.rt.shutdown()
        except Exception:
            pass


_POOL: AwmSlotPool | None = None
_POOL_LOCK = asyncio.Lock()

# ---- module-level, process-wide caches ----------------------------------
# The frozen parser with the family-aware fallback layered on exactly the way
# run_headroom2.py does it: delegate to run_gate.parse_call first, fall back only on
# inputs it rejected. Identical to screening by construction, not by test corpus.
def _parse_call(t):
    return run_gate.parse_call(t) or parse_multi._fallback(t)


_SYSTEM = parse_multi.system_for(FAMILY)
_ASSETS: dict[str, tuple] = {}


def _advertised_map():
    """Per-scenario advertised sets, when the trainer is controlling difficulty per task.

    A single global surface can only trade reward against reward. Gradient availability
    g(p,k)=1-p^k-(1-p)^k peaks at p=0.5, and restoring tools drives p from 0 toward 1, so the
    surface that maximises reward is not the one that maximises learning -- measured here:
    availability is 6.4% at 39% restoration and 3.5% at full restoration. This lets a controller
    hold each scenario near the frontier instead of pushing every task to trivially-solved.
    """
    p = os.environ.get("AWM_ADVERTISED_MAP")
    if not p or not os.path.exists(p):
        return None
    try:
        return {k: set(v) for k, v in json.load(open(p)).items()}
    except Exception:
        return None


_ADV_MAP = _advertised_map()


def _advertised():
    """Tool names the harness currently advertises, or None when unrestricted.

    Co-adaptation edits the tool surface between training cycles, so the trainer has to roll out
    against the same surface the gate certified; without this the policy would keep training on
    the unedited substrate and the two halves would never actually meet. The set is read from a
    file rather than the environment so the driver can rewrite it between cycles, and leaving the
    variable unset preserves every other arm's behaviour exactly.
    """
    p = os.environ.get("AWM_ADVERTISED")
    if not p or not os.path.exists(p):
        return None
    names = {ln.strip() for ln in open(p) if ln.strip()}
    return names or None


_ADVERTISED = _advertised()


def _val_advertised():
    """Surface used for VALIDATION rollouts, held identical across every arm.

    A validation curve is only comparable between arms if they are scored in the same
    environment. Training surfaces differ by construction -- b8van sits at level 0.0, the ELSA
    arms drift to ~0.47, the reward-axis arms are pinned at 0.5 -- so validating each arm on its
    own surface would rank them by how many tools they advertise rather than by what the policy
    learned. This is the same fixed handicap surface the eval cells use (advertised_init.txt),
    which is what makes the in-training curve and the held-out table measure the same thing.
    """
    p = os.environ.get("AWM_VAL_ADVERTISED")
    if not p or not os.path.exists(p):
        return None
    return {ln.strip() for ln in open(p) if ln.strip()}


_VAL_ADVERTISED = _val_advertised()
# verl applies rollout.val_kwargs on validation batches; we set its temperature to 0 so a
# validation rollout is identifiable from the sampling params the loop already receives. That
# avoids patching installed verl, which does not forward its `validate` flag into run()'s kwargs.
_VAL_TEMP_EPS = float(os.environ.get("AWM_VAL_TEMP_EPS", "1e-6"))


def _is_validation(sampling_params) -> bool:
    """True when verl handed us a VALIDATION batch's sampling params.

    verl builds these in agent_loop.py::generate_sequences: it starts from rollout.temperature /
    top_p / top_k and, when `validate` is set, overwrites all three from rollout.val_kwargs. Our
    VAL_ARGS pin val_kwargs to temperature=0.0 top_p=1.0, and training runs at ROLLOUT_TEMP
    (0.7 by default) with top_p=0.8 -- so the pair separates the two batches exactly.

    Both halves are required because temperature alone has one false positive: an arm launched
    with ROLLOUT_TEMP=0 would greedily decode its TRAINING rollouts and every one of them would
    be filed as validation -- a curve made of training episodes on the validation surface, which
    reads as a result rather than as a failure. top_p defaults to 1.0 when absent so a params
    dict without the key still behaves exactly as the temperature-only test did.

    (verl's other greedy path, apply_greedy_sampling_params for per-sample `__do_sample__=False`,
    sets the identical pair -- but prep_awm.py does not emit a __do_sample__ column, so
    per_sample_do_sample is None and that path never fires here.)
    """
    if _VAL_ADVERTISED is None:
        return False
    try:
        sp = sampling_params or {}
        return (float(sp.get("temperature", 1.0)) <= _VAL_TEMP_EPS
                and float(sp.get("top_p", 1.0)) >= 1.0 - _VAL_TEMP_EPS)
    except Exception:
        return False


# =========================================================================
# TRIAGE_DECOR -- per-rollout perturbation of the ADVERTISED SURFACE PRESENTATION
# =========================================================================
# WHY. Every allocator this project has built is capped by the same measured quantity: the k
# rollouts of one GRPO group share a checkpoint AND a byte-identical prompt, so they agree far
# more often than k iid draws. Fitted within-group correlation rho = 0.78 (nu = 0.285, PLAN
# section, v3 block), and at that correlation NO allocation of the budget can push the degenerate
# fraction below ~0.79-0.92 on this pool -- the ceiling is in the sampling, not in the choice of
# tasks. This flag attacks the correlation directly instead: the k rollouts of a group are shown
# DIFFERENT-BUT-EQUIVALENT presentations of the SAME task.
#
# WHAT `shuffle` DOES, AND WHY IT IS SEMANTICALLY FREE. A tool surface is a SET. MCP's tools/list
# carries no ordering contract, the chat template renders whatever order the list arrives in, and
# `interfaces.Interface` resolves calls through `by_name` -- a dict -- so no dispatch, no verifier
# and no reward depends on the order. Permuting the list changes the token sequence of the turn-0
# prompt and NOTHING about the task, the action space, or what counts as success. n_tools is
# invariant by construction (a permutation is a bijection), so the val_curve.py comparability
# audit is untouched.
#
# THE FINAL-ANSWER TOOL IS NOT PERMUTED. specs() appends FINAL_SPEC last, after the advertised
# tools; the permutation is applied to the tool LIST before Interface is built, so `final_answer`
# keeps its canonical last position on every rollout. Moving it would confound "presentation
# noise" with "how easy is the stop action to find".
#
# DETERMINISM. The permutation is a pure function of (salt, scenario, task_idx, rollout_index),
# hashed with blake2b -- not builtins.hash, which is salted per process by PYTHONHASHSEED and
# would make two workers of the same run disagree. The k permutations of one task are computed as
# a FAMILY and de-duplicated, so "the k rollouts of a group see k different orders" is true by
# construction rather than by probability (it can only fail when n! < k, i.e. a 1- or 2-tool
# scenario, which the bump cap handles by falling through).
#
# THE ROLLOUT INDEX. verl computes exactly this quantity (agent_loop.py::get_trajectory_info ->
# trajectory["rollout_n"]) and then does NOT forward it: _run_agent_loop keeps it for its own
# trace attrs and calls `agent_loop.run(sampling_params, **kwargs)` with only the row's
# non_tensor_batch columns, which are IDENTICAL across the k copies of a sample. Rather than patch
# installed verl for one integer, the index is recovered here the same way verl derives it -- a
# per-process counter over consecutive rollouts of the same (scenario, task_idx). That is exact
# for this fleet: generate_sequences creates the k tasks with asyncio.create_task in batch order
# and the loop runs them FIFO, and everything in run() before the first `await` is synchronous, so
# the counter is claimed in batch order; and verl chunks the batch across agent.num_workers=2
# while BSZ=32, so a group of k=5 is never split across worker processes (32/2 = 16 whole groups
# per chunk). Both facts are asserted offline, not assumed. If a future config DID split a group,
# the failure mode is benign -- two rollouts share a presentation, the intervention weakens, the
# reward and the audit stay correct -- and `decor_idx`/`decor_sig` in the episode record measure
# it directly from the log.
#
# VALIDATION IS UNTOUCHED, AND THIS IS THE LOAD-BEARING GUARANTEE. Validation is what makes the
# curve comparable across every arm in the fleet; a val rollout that saw a permuted surface would
# be scored in a different environment from a8F's and a8T3g's. The permutation is applied ONLY on
# the `not is_val` branch, the counter is not even touched on the val path, and val_kwargs pins
# n=1 so there is no group to decorrelate there anyway.
#
# ABSENT = BYTE-IDENTICAL. With TRIAGE_DECOR unset (every arm in flight) _DECOR_MODE is "" and
# _decorate_tools returns the caller's own list OBJECT, unmodified -- not a copy, not a reordering
# -- so the specs, the rendered prompt and the episode record are the pre-flag ones exactly.
_DECOR_MODE = os.environ.get("TRIAGE_DECOR", "").strip().lower()
if _DECOR_MODE in ("none", "off", "0"):
    _DECOR_MODE = ""
if _DECOR_MODE not in ("", "shuffle"):
    # Fail loudly. A typo here would run a plain a8T3g clone under the decorrelation arm's name
    # and be reported as a measured negative for a hypothesis that was never tested -- the same
    # class of silent-no-op failure that made the retired gatesd arm meaningless.
    raise RuntimeError(f"TRIAGE_DECOR={_DECOR_MODE!r} is not a known mode; expected 'shuffle' "
                       f"(or unset for the unmodified presentation)")

# Salt for the permutation family. Separates seed replicates of the same arm: two seeds that
# differ only in --seed-offset would otherwise see the identical k presentations per task.
_DECOR_SALT = os.environ.get("TRIAGE_DECOR_SEED", "0")

# k, from the one place that sets it. verl_awm_train.sh does NROLL=${NROLL:-5} and hands that
# same value to actor_rollout_ref.rollout.n, and forwards it here as TRIAGE_DECOR_K when the flag
# is on; the fallback chain keeps the default identical to the trainer's.
_DECOR_K = max(1, int(os.environ.get("TRIAGE_DECOR_K", os.environ.get("NROLL", "5"))))

_DECOR_COUNTS: dict[tuple, int] = {}
_DECOR_FAMILIES: dict[tuple, list] = {}
_DECOR_LOCK = threading.Lock()


def _perm(n: int, key: str, idx: int, bump: int) -> list:
    h = hashlib.blake2b(f"{_DECOR_SALT}|{key}|{idx}|{bump}".encode(), digest_size=16).digest()
    order = list(range(n))
    random.Random(int.from_bytes(h, "big")).shuffle(order)
    return order


def _perm_family(n: int, key: str, k: int) -> list:
    """The k permutations of range(n) this task's k rollouts get. Distinct by construction."""
    fam, seen = [], set()
    for i in range(k):
        order = _perm(n, key, i, 0)
        bump = 0
        while tuple(order) in seen and bump < 64:
            bump += 1
            order = _perm(n, key, i, bump)
        seen.add(tuple(order))
        fam.append(order)
    return fam


def _decor_index(key: tuple) -> int:
    """This rollout's position within its group, 0..k-1.

    Claimed BEFORE the first await in run(), so the order matches the order verl created the
    tasks in, which is batch order -- the same thing verl's own get_trajectory_info counts.
    """
    with _DECOR_LOCK:
        i = _DECOR_COUNTS.get(key, 0)
        _DECOR_COUNTS[key] = (i + 1) % _DECOR_K
    return i


def _decorate_tools(tools: list, scenario: str, task_idx: int, idx: int) -> list:
    """Return this rollout's presentation of the advertised tool list.

    Flag absent -> the caller's own object, untouched, so the downstream is bit-identical.
    """
    if not _DECOR_MODE or idx < 0 or len(tools) < 2:
        return tools
    key = (scenario, int(task_idx), len(tools))
    fam = _DECOR_FAMILIES.get(key)
    if fam is None:
        fam = _perm_family(len(tools), f"{scenario}#{task_idx}", _DECOR_K)
        _DECOR_FAMILIES[key] = fam
    order = fam[idx % len(fam)]
    return [tools[j] for j in order]


def _decor_sig(tools: list) -> str:
    """Short fingerprint of the ORDER actually advertised, so distinctness is auditable from
    episodes.jsonl alone -- no access to the process that produced it, the same standard n_tools
    already meets for the validation surface."""
    return hashlib.blake2b("\x00".join(t["name"] for t in tools).encode(),
                           digest_size=6).hexdigest()


def _scenario_assets(sc: str, val: bool = False):
    # Cache key carries `val`: the two surfaces differ, and a single-key cache would let
    # whichever rollout arrived first decide the tool set for every later one.
    key = (sc, bool(val))
    if key not in _ASSETS:
        tools = json.load(open(os.path.join(SURF, f"{sc}.json")))["tools"]
        if val and _VAL_ADVERTISED is not None:
            tools = [t for t in tools if t["name"] in _VAL_ADVERTISED]
        elif _ADV_MAP is not None and sc in _ADV_MAP:
            tools = [t for t in tools if t["name"] in _ADV_MAP[sc]]
        elif _ADVERTISED is not None:
            tools = [t for t in tools if t["name"] in _ADVERTISED]
        _ASSETS[key] = (tools, load_tasks(sc), load_verifiers(sc))
    return _ASSETS[key]


async def get_pool() -> AwmSlotPool:
    global _POOL
    async with _POOL_LOCK:
        if _POOL is None:
            _POOL = AwmSlotPool()
        return _POOL


# =========================================================================
# the agent loop
# =========================================================================
@register("awm_agent")
class AwmAgentLoop(AgentLoopBase):
    """One run() == one AWM episode == one GRPO sample."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("tools", None)          # we build schemas per sample, not from the registry
        super().__init__(*args, **kwargs)
        # NOTE: verl instantiates one AgentLoop object PER SAMPLE
        # (agent_loop.py `_run_agent_loop` -> hydra.utils.instantiate), so nothing
        # expensive may live on `self`. Scenario assets are cached at MODULE level:
        # load_tasks/load_verifiers linearly scan multi-MB jsonl files, and doing that
        # per episode would dominate the step time.
        self._parse = _parse_call
        self._system = _SYSTEM

    # ---- scenario assets (cached per PROCESS, not per instance) ----------
    def _scenario(self, sc: str, val: bool = False):
        return _scenario_assets(sc, val)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra = kwargs.get("extra_info") or {}
        scenario = extra.get("scenario")
        task_idx = int(extra.get("task_idx", 0))
        if not scenario:
            raise ValueError(f"row has no extra_info.scenario; got {extra!r}")

        is_val = _is_validation(sampling_params)
        # Claimed here, before ANY await in this coroutine, so the k rollouts of a group take
        # 0..k-1 in the order verl created their tasks. Never claimed on the validation path:
        # validation must stay the one fixed presentation every arm is scored on.
        decor_idx = _decor_index((scenario, task_idx)) if (_DECOR_MODE and not is_val) else -1
        tools, tasks, verifiers = self._scenario(scenario, is_val)
        task = tasks[task_idx]
        tools = _decorate_tools(tools, scenario, task_idx, decor_idx)
        iface = IF.build("RAW", tools, task)
        specs = iface.specs()

        metrics: dict[str, Any] = {}
        pool = await get_pool()
        slot = await pool.acquire(scenario)

        t_ep = time.time()
        n_backend = n_err = n_parse_fail = 0
        consecutive_pf = 0
        stop_reason = "max_turns"
        final_answer = None
        reward = 0.0
        vstatus = "not_run"
        gate_progress = 0.0

        try:
            initial_db = os.path.join(DBS, f"{scenario}.initial.db")
            await asyncio.to_thread(slot.srv.restore, initial_db)

            messages = [{"role": "system", "content": self._system},
                        {"role": "user", "content": task}]

            # Turn 0 prompt carries the whole RAW tool surface in the template's tools= slot,
            # same as policy_vllm.render does for screening.
            prompt_ids = await self.apply_chat_template(messages, tools=specs)
            all_ids = list(prompt_ids)
            response_mask: list[int] = []

            sp = {**sampling_params, "max_tokens": MAX_NEW_TOKENS}
            sp.pop("max_new_tokens", None)

            turns = 0
            for turns in range(1, MAX_TURNS + 1):
                out = await self.server_manager.generate(
                    request_id=uuid.uuid4().hex,
                    prompt_ids=all_ids,
                    sampling_params=sp,
                )
                gen_ids = out.token_ids
                all_ids += gen_ids
                response_mask += [1] * len(gen_ids)

                text = await asyncio.to_thread(
                    self.tokenizer.decode, gen_ids, skip_special_tokens=True)
                messages.append({"role": "assistant", "content": text})

                if len(response_mask) >= self.rollout_config.response_length:
                    stop_reason = "length"
                    break

                pc = self._parse(text)
                if pc is None:
                    n_parse_fail += 1
                    consecutive_pf += 1
                    if consecutive_pf >= 2 or turns >= MAX_TURNS:
                        final_answer, stop_reason = text, "no_tool_call"
                        break
                    reply = "Respond with exactly one <tool_call> block."
                    add = [{"role": "user", "content": reply}]
                else:
                    consecutive_pf = 0
                    name, args = pc
                    if name == IF.FINAL_TOOL:
                        final_answer = str(args.get("answer", ""))
                        stop_reason = "final_answer"
                        break
                    # dispatch on the ToolRuntime's own loop, awaited from ours without
                    # blocking this thread (ToolRuntime.submit() would block it).
                    call = pool.rt.caller(slot.key)
                    fut = asyncio.run_coroutine_threadsafe(
                        iface.dispatch(call, name, args), pool.rt.loop)
                    try:
                        txt, nb, err = await asyncio.wait_for(
                            asyncio.wrap_future(fut), timeout=120)
                    except Exception as exc:
                        txt, nb, err = f"error: {type(exc).__name__}: {exc}"[:300], 0, True
                    n_backend += nb
                    n_err += int(bool(err))
                    add = [{"role": "tool", "content": txt}]

                messages.extend(add)
                add_ids = await self.apply_chat_template(add, remove_system_prompt=True)
                if len(response_mask) + len(add_ids) >= self.rollout_config.response_length:
                    stop_reason = "length"
                    break
                all_ids += add_ids
                response_mask += [0] * len(add_ids)      # tool tokens are NOT trained on

            if final_answer is None:
                final_answer = messages[-1].get("content", "") if messages else ""

            # ---- verify: same pure-python, state-based verifier as screening ----
            final_db = os.path.join(pool.workdir, f"{scenario}_{slot.srv.port}.final.db")
            await asyncio.to_thread(shutil.copy, slot.srv.db_path, final_db)
            # `reward` stays the true binary outcome and is what lands in episodes.jsonl and in
            # every evaluation; only the TRAINING signal below is ever shaped. AWM_DENSE_REWARD
            # is unset for every existing arm, so this is the original code path verbatim.
            if _DENSE.enabled():
                r, vstatus, gate_progress = await asyncio.to_thread(
                    _DENSE.run_verifier_graded,
                    verifiers.get(task_idx, ""), initial_db, final_db, final_answer)
            else:
                r, vstatus = await asyncio.to_thread(
                    run_verifier, verifiers.get(task_idx, ""), initial_db, final_db, final_answer)
            reward = 1.0 if r == "complete" else 0.0
            try:
                os.remove(final_db)
            except OSError:
                pass
        finally:
            pool.release(slot)

        wall = time.time() - t_ep
        metrics["tool_calls"] = float(n_backend)

        train_reward = _DENSE.shaped_reward(reward, gate_progress, n_backend, n_err,
                                            n_parse_fail, stop_reason, scenario)

        # n_tools is the AUDIT for the validation curve. `split` alone only says which branch the
        # loop believed it was on; the number of tools actually advertised to the policy is what
        # decides whether two arms were scored in the same environment. A val episode must carry
        # the fixed surface's count for its scenario on EVERY arm, and it must differ from that
        # arm's own train count whenever the arm's level is not the handicap level -- so a curve
        # can be checked for comparability from the episode log alone, with no access to the run
        # that produced it. val_curve.py refuses to emit a point that fails this check.
        rec = dict(scenario=scenario, task_idx=task_idx, reward=reward, split=("val" if is_val else "train"),
                   n_tools=len(tools),
                   verifier_status=vstatus, n_turns=turns, n_backend_calls=n_backend,
                   n_errors=n_err, n_parse_fail=n_parse_fail, stop_reason=stop_reason,
                   wall_s=round(wall, 2), resp_tokens=len(response_mask),
                   final_answer=(final_answer or "")[:300])
        if _DECOR_MODE and decor_idx >= 0:
            # Added, never substituted: reward, n_tools, split and every other field above are the
            # unmodified ones, so an arm running this flag stays readable by every existing report
            # script. decor_sig is the audit that the k rollouts of a group really did see k
            # different orders -- the mechanism claim is checkable from the log, not asserted.
            rec["decor"] = _DECOR_MODE
            rec["decor_idx"] = decor_idx
            rec["decor_sig"] = _decor_sig(tools)
        if _DENSE.enabled():
            # recorded alongside, never instead of, the binary outcome
            rec["train_reward"] = round(train_reward, 4)
            rec["gate_progress"] = round(gate_progress, 4)
            rec["dense_mode"] = _DENSE.MODE
        _append_episode(rec)
        logger.info("[awm] %s/%d reward=%.0f train=%.3f gate=%.2f turns=%d calls=%d stop=%s %.1fs",
                    scenario, task_idx, reward, train_reward, gate_progress,
                    turns, n_backend, stop_reason, wall)

        n_prompt = len(all_ids) - len(response_mask)
        # Rejection-sampling SFT / STaR needs the trajectory itself, and episodes.jsonl keeps
        # only summary counters plus 300 characters of the final answer -- so no SFT corpus can
        # be rebuilt from a completed run. Capturing the exact three tensors the RL arm trains on
        # (prompt, response, loss mask) makes the SFT baseline consume identical data to GRPO:
        # same tokenizer, same multi-turn layout, same tool-result tokens masked out of the loss.
        if _SFT_CAPTURE and (reward >= 1.0 or _SFT_CAPTURE == "all"):
            _append_traj({
                "scenario": scenario, "task_idx": task_idx, "reward": reward,
                "prompt_ids": all_ids[:n_prompt],
                "response_ids": all_ids[n_prompt:][: self.rollout_config.response_length],
                "response_mask": response_mask[: self.rollout_config.response_length],
            })
        return AgentLoopOutput(
            prompt_ids=all_ids[:n_prompt],
            response_ids=all_ids[n_prompt:][: self.rollout_config.response_length],
            response_mask=response_mask[: self.rollout_config.response_length],
            reward_score=train_reward,         # -> rm_scores, bypasses the reward manager
            num_turns=turns,
            metrics=metrics,
            extra_fields={},
        )


_EP_PATH = os.path.join(RUNDIR, "episodes.jsonl")

# "" (default, every existing arm) captures nothing. "1"/"success" keeps only solved
# trajectories, which is exactly the STaR/rejection-sampling corpus. "all" keeps everything and
# is much larger -- a solved episode is ~100KB of token ids, so "all" costs ~5GB per 50k episodes.
_SFT_CAPTURE = os.environ.get("AWM_SFT_CAPTURE", "").strip().lower()
if _SFT_CAPTURE == "1":
    _SFT_CAPTURE = "success"
_TRAJ_PATH = os.path.join(RUNDIR, "sft_traj.jsonl")
_TRAJ_LOCK = threading.Lock()


def _append_traj(rec: dict):
    """Append one trajectory. Several agent loops finish concurrently in this process, and a
    partial interleaved line would silently corrupt the SFT corpus, so serialise the write."""
    try:
        os.makedirs(os.path.dirname(_TRAJ_PATH), exist_ok=True)
        line = json.dumps(rec)
        with _TRAJ_LOCK:
            with open(_TRAJ_PATH, "a") as f:
                f.write(line + "\n")
                f.flush()
    except Exception:
        pass


def _append_episode(rec: dict):
    """Per-episode receipt. Verifying a long run by an artifact that grows is the
    repo rule; pgrep self-match has burned this project eight times."""
    try:
        os.makedirs(os.path.dirname(_EP_PATH), exist_ok=True)
        with open(_EP_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
    except Exception:
        pass
