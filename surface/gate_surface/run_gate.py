"""G_surface measurement loop: every interface on every task, from identical
cloned backend state, multiple seeds.

Slot model: for one scenario we boot `--slots` AWM servers, each bound to its own
working .db. A slot runs one episode at a time; between episodes the working .db
is restored by file copy from the canonical initial.db (0.09 ms) and the restored
content digest is ASSERTED equal to the canonical digest. Generation is batched
across slots, so the GPU batch size equals the number of live slots.

Nothing about the policy changes across conditions. Only `interfaces.py` does.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import interfaces as IF                                   # noqa: E402
from awm_env import (AwmServer, db_content_digest, free_port,  # noqa: E402
                     load_tasks, load_verifiers, run_verifier)
from tool_runtime import ToolRuntime                      # noqa: E402

WORK = os.path.join(HERE, "work")
DBS = os.path.join(WORK, "dbs")
SURF = os.path.join(WORK, "surfaces")

# Defaults are the values every banked gate ran under and MUST NOT change -- G_surface
# re-derives byte-identically from committed episodes and that property is load-bearing.
# Both are env-overridable so a new gate can raise them without touching banked results.
#
# Measured 2026-07-31 on 33,296 turns: MAX_NEW_TOKENS=256 truncates tool calls mid-JSON on
# 15.2% of all turns (qwen 2.3%, granite 9.7%, mistral 18.3%, falcon3 25.7%), and 60-100%
# of json-bearing parse-fail samples have unbalanced braces. MAX_TURNS=8 ends 28.5% of
# episodes. Both inflate the never-solved class. See GATE_HEADROOM2_FROZEN.md.
MAX_TURNS = int(os.environ.get("GATE_MAX_TURNS", "8"))
MAX_NEW_TOKENS = int(os.environ.get("GATE_MAX_NEW_TOKENS", "256"))
TEMPERATURE = 0.7
TOP_P = 0.8

SYSTEM = (
    "You are an autonomous agent operating a real application through its tool API.\n"
    "Complete the user's task by calling the tools that are available to you.\n"
    "Rules:\n"
    "- Call exactly ONE tool per turn, using the <tool_call> format.\n"
    "- Inspect the environment with read tools before you change anything; do not\n"
    "  invent identifiers, look them up.\n"
    "- When the task is complete, or you are certain it cannot be completed, call\n"
    f"  {IF.FINAL_TOOL} with a short summary plus any information the user asked for.\n"
    f"- You have at most {MAX_TURNS} turns."
)

_TC = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_BARE = re.compile(r"\{\s*\"name\"\s*:.*?\}\s*$", re.S)


def parse_call(text: str):
    m = _TC.search(text)
    raw = m.group(1) if m else None
    if raw is None:
        m2 = _BARE.search(text.strip())
        raw = m2.group(0) if m2 else None
    if raw is None:
        return None
    try:
        obj = json.loads(raw)
    except Exception:
        try:
            obj = json.loads(raw.replace("\n", " "))
        except Exception:
            return None
    if not isinstance(obj, dict) or "name" not in obj:
        return None
    args = obj.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    return obj["name"], (args if isinstance(args, dict) else {})


class Episode:
    def __init__(self, scenario, task_idx, task, kind, seed, tools):
        self.scenario, self.task_idx, self.task = scenario, task_idx, task
        self.kind, self.seed = kind, seed
        self.iface = IF.build(kind, tools, task)
        self.msgs = [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": task}]
        self.turns = 0
        self.backend_calls = 0
        self.errors = 0
        self.parse_fail = 0
        self.consecutive_parse_fail = 0
        self.prompt_tokens = []
        self.gen_tokens = []
        self.schema_tokens = []
        self.final_answer = None
        self.parse_fail_sample = None
        self.calls = []              # advertised tool name the policy asked for
        self.unknown_calls = 0       # names not on the surface at all
        self.done = False
        self.stop_reason = None
        self.t0 = time.time()
        self.wall = None
        self.last_text = ""

    def record(self):
        return dict(scenario=self.scenario, task_idx=self.task_idx, interface=self.kind,
                    seed=self.seed, n_turns=self.turns, n_backend_calls=self.backend_calls,
                    n_errors=self.errors, n_parse_fail=self.parse_fail,
                    tool_calls=self.calls, n_unknown_tool=self.unknown_calls,
                    prompt_tokens=self.prompt_tokens, gen_tokens=self.gen_tokens,
                    parse_fail_sample=(self.parse_fail_sample or "")[:400],
                    schema_tokens=self.schema_tokens, stop_reason=self.stop_reason,
                    final_answer=(self.final_answer or "")[:2000], wall_s=self.wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("--tasks", default="all", help="'all' or comma list of indices")
    ap.add_argument("--interfaces", nargs="+", default=list(IF.KINDS))
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--mem-fraction", type=float, default=0.30)
    ap.add_argument("--micro", type=int, default=4, help="max sequences per GPU micro-batch")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume-from", nargs="*", default=[],
                    help="extra episode files whose records also count as done "
                         "(so a runner on a NEW output file does not redo work "
                         "already banked in another file)")
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap episodes per scenario")
    a = ap.parse_args()

    import policy
    policy.load(a.gpu, a.mem_fraction)

    done_keys = set()
    for src in [a.out] + list(a.resume_from):
        if not os.path.exists(src):
            continue
        n0 = len(done_keys)
        bad = 0
        for line in open(src):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                bad += 1
                continue
            done_keys.add((r["scenario"], r["task_idx"], r["interface"], r["seed"]))
        print(f"[resume] {len(done_keys)-n0} episodes from {src}"
              + (f" ({bad} unparseable lines skipped)" if bad else ""), flush=True)
    print(f"[resume] {len(done_keys)} episodes banked in total", flush=True)

    # A kill mid-write leaves a truncated final line with no newline; without
    # this, the next append would concatenate onto it and corrupt BOTH records.
    if os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        with open(a.out, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                with open(a.out, "a") as f2:
                    f2.write("\n")
                print("[resume] repaired a truncated final line", flush=True)

    rt = ToolRuntime()
    outf = open(a.out, "a")
    n_written = 0
    t_start = time.time()

    for scenario in a.scenarios:
        surf = json.load(open(os.path.join(SURF, f"{scenario}.json")))
        tools = surf["tools"]
        tasks = load_tasks(scenario)
        verifiers = load_verifiers(scenario)
        idxs = (list(range(len(tasks))) if a.tasks == "all"
                else [int(x) for x in a.tasks.split(",")])
        initial_db = os.path.join(DBS, f"{scenario}.initial.db")
        init_digest = db_content_digest(initial_db)

        queue = []
        for ti in idxs:
            for kind in a.interfaces:
                for sd in a.seeds:
                    queue.append((ti, kind, sd))
        # Complete one seed at a time across ALL tasks and interfaces, so that
        # stopping early still leaves a COMPLETE task x interface cube for every
        # seed finished. Within a seed, group by interface so batched prompts
        # have similar lengths.
        queue = [q for q in queue
                 if (scenario, q[0], q[1], q[2]) not in done_keys]
        queue.sort(key=lambda x: (x[2], x[1], x[0]))
        if a.limit:
            queue = queue[:a.limit]

        if not queue:
            print(f"[{scenario}] nothing to do (all episodes already present)", flush=True)
            continue

        # ---- boot slots -------------------------------------------------
        slots = []
        for i in range(a.slots):
            db = os.path.join(WORK, "_run", f"{scenario}_slot{i}.db")
            os.makedirs(os.path.dirname(db), exist_ok=True)
            shutil.copy(initial_db, db)
            srv = AwmServer(scenario, os.path.join(WORK, "_srv"), db, port=free_port())
            slots.append(srv)
        t0 = time.time()
        for s in slots:
            s.start()
        ready = [s.wait_ready() for s in slots]
        print(f"[{scenario}] {sum(ready)}/{len(slots)} slots ready in {time.time()-t0:.1f}s, "
              f"{len(queue)} episodes queued", flush=True)
        live = [s for s, r in zip(slots, ready) if r]
        for s, r in zip(slots, ready):
            if not r:
                print(f"  slot boot FAILED: {s.tail(600)}", flush=True)
        for i, s in enumerate(live):
            rt.open(f"{scenario}:{i}", s.url)

        active = [None] * len(live)
        qi = 0
        while qi < len(queue) or any(active):
            # fill free slots
            for i, ep in enumerate(active):
                if ep is not None or qi >= len(queue):
                    continue
                ti, kind, sd = queue[qi]
                qi += 1
                live[i].restore(initial_db)
                d = db_content_digest(live[i].db_path)
                if d != init_digest:
                    print(f"  [WARN] slot {i} restore digest {d} != {init_digest}", flush=True)
                active[i] = Episode(scenario, ti, tasks[ti], kind, sd, tools)

            idx = [i for i, e in enumerate(active) if e is not None]
            if not idx:
                break
            eps = [active[i] for i in idx]
            specs = [e.iface.specs() for e in eps]
            prompts = [policy.render(e.msgs, sp) for e, sp in zip(eps, specs)]
            seeds = [e.seed * 1000 + e.turns for e in eps]
            plen = [policy.count_tokens(p) for p in prompts]
            texts, n_pad, n_gen = policy.generate_batch(
                prompts, seeds, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, a.micro)

            pending = []
            for j, (i, e) in enumerate(zip(idx, eps)):
                e.turns += 1
                e.prompt_tokens.append(plen[j])
                e.gen_tokens.append(n_gen[j])
                e.schema_tokens.append(policy.tools_token_cost(SYSTEM, e.task, specs[j]))
                e.last_text = texts[j]
                e.msgs.append({"role": "assistant", "content": texts[j]})
                pc = parse_call(texts[j])
                if pc is None:
                    e.parse_fail += 1
                    if e.parse_fail_sample is None:
                        e.parse_fail_sample = texts[j]
                    e.consecutive_parse_fail += 1
                    if e.consecutive_parse_fail >= 2 or e.turns >= MAX_TURNS:
                        e.final_answer = texts[j]
                        e.done, e.stop_reason = True, "no_tool_call"
                    else:
                        e.msgs.append({"role": "user",
                                       "content": "Respond with exactly one <tool_call> block."})
                    continue
                e.consecutive_parse_fail = 0
                name, args = pc
                if name == IF.FINAL_TOOL:
                    e.final_answer = str(args.get("answer", ""))
                    e.done, e.stop_reason = True, "final_answer"
                    continue
                e.calls.append(name)
                if not (name in e.iface.by_name or name in e.iface.macros
                        or name in ("expand_toolset", "call_primitive")):
                    e.unknown_calls += 1
                call = rt.caller(f"{scenario}:{i}")
                pending.append((e, e.iface.dispatch(call, name, args)))

            if pending:
                results = rt.run_many([c for _, c in pending])
                for (e, _), res in zip(pending, results):
                    if isinstance(res, Exception):
                        txt, nb, err = f"error: {type(res).__name__}: {res}"[:300], 0, True
                    else:
                        txt, nb, err = res
                    e.backend_calls += nb
                    e.errors += int(bool(err))
                    e.msgs.append({"role": "tool", "content": txt})

            for i, e in zip(idx, eps):
                if not e.done and e.turns >= MAX_TURNS:
                    e.final_answer = e.last_text
                    e.done, e.stop_reason = True, "max_turns"
                if e.done:
                    e.wall = round(time.time() - e.t0, 2)
                    final_db = os.path.join(WORK, "_run", f"{scenario}_slot{i}.final.db")
                    shutil.copy(live[i].db_path, final_db)
                    wal = live[i].db_path + "-wal"
                    r, st = run_verifier(verifiers.get(e.task_idx, ""),
                                         initial_db, final_db, e.final_answer)
                    rec = e.record()
                    rec.update(reward=int(r == "complete"), verifier_status=st,
                               final_digest=db_content_digest(final_db),
                               db_changed=db_content_digest(final_db) != init_digest,
                               wal_present=os.path.exists(wal))
                    outf.write(json.dumps(rec) + "\n")
                    outf.flush()
                    os.fsync(outf.fileno())      # durable per-episode checkpoint
                    n_written += 1
                    active[i] = None
            if n_written and n_written % 20 == 0:
                el = time.time() - t_start
                print(f"  [{scenario}] {n_written} episodes, {el/60:.1f} min, "
                      f"{el/max(n_written,1):.1f} s/episode", flush=True)

        for i in range(len(live)):
            rt.close(f"{scenario}:{i}")
        for s in slots:
            s.stop()
        print(f"[{scenario}] done, {n_written} episodes total, "
              f"{(time.time()-t_start)/60:.1f} min", flush=True)

    outf.close()
    rt.shutdown()
    print(f"RESULT wrote {n_written} episodes to {a.out} "
          f"in {(time.time()-t_start)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
