"""Evaluate one advertised tool subset over a task set. Thin wrapper on the frozen runner."""
from __future__ import annotations
import argparse, collections, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "gate_surface")); sys.path.insert(0, HERE)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True); ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--mem-fraction", type=float, default=0.63)
    a = ap.parse_args()
    import parse_multi, policy_vllm as policy_mod, run_gate
    fam = policy_mod.family()
    if run_gate.MAX_TURNS < 20 or run_gate.MAX_NEW_TOKENS < 1024:
        raise SystemExit("caps not raised; set GATE_MAX_TURNS=20 GATE_MAX_NEW_TOKENS=1024")
    import difflib
    fl, sl = run_gate.SYSTEM.splitlines(True), parse_multi.system_for(fam).splitlines(True)
    if len([o for o in difflib.SequenceMatcher(None, fl, sl).get_opcodes() if o[0] != "equal"]) > 1:
        raise RuntimeError("system_for differs in more than one place")
    run_gate.SYSTEM = parse_multi.system_for(fam)
    _f = run_gate.parse_call
    run_gate.parse_call = lambda t, _g=_f: (_g(t) if _g(t) is not None else parse_multi._fallback(t))
    _r = policy_mod.render
    policy_mod.render = lambda m, t, _q=_r, _fam=fam: _q(parse_multi.normalise_history(m, _fam), t)
    sys.modules["policy"] = policy_mod
    run_gate.TEMPERATURE = 0.7
    by = json.loads(a.tasks); seeds = a.seeds.split(",")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    print(f"[eval] subset={len((os.environ.get('GATE_TOOL_SUBSET') or '').split(','))} tools "
          f"scenarios={len(by)} seeds={seeds}", flush=True)
    for sc, idxs in by.items():
        argv = ["run_gate", "--scenarios", sc, "--tasks", ",".join(str(i) for i in sorted(idxs)),
                "--seeds", *seeds, "--interfaces", "RAW", "--gpu", "0", "--slots", "16",
                "--micro", "16", "--mem-fraction", str(a.mem_fraction),
                "--out", a.out, "--resume-from", a.out]
        old = sys.argv; sys.argv = argv
        try: run_gate.main()
        except SystemExit: pass
        except Exception as e: print(f"[eval] {sc} FAILED: {type(e).__name__}: {e}", flush=True)
        finally: sys.argv = old
    n = sum(1 for _ in open(a.out)) if os.path.exists(a.out) else 0
    print(f"[eval] episodes={n}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
