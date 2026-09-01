"""Auto-apply the TRIAGE v4 variable-k patch inside the TRAINER and its RAY WORKERS.

WHY THIS FILE EXISTS. fit() does not run in the process that launches training: it runs inside a
Ray actor (the "TaskRunner pid=..." lines in every rl_*.log). A monkeypatch applied in the
launching process therefore never reaches the code that repeats the batch. `sitecustomize` is
imported automatically by `site` at interpreter startup for every process whose sys.path contains
this directory -- and `slurm/verl_awm_train.sh` already puts `surface/verl_rl` on PYTHONPATH
(line 31) AND already forwards PYTHONPATH into the Ray runtime_env (line 200). So the actor picks
this up with no new plumbing, and site-packages is never touched.

INERT BY DEFAULT. vark_patch.apply() returns immediately unless VARK=1, so for every other process
that happens to have this directory on its path -- coadapt.py, triage.py, prep_awm.py, the eval
scripts -- this costs one function call and does nothing.

FAILURES ARE LOUD BUT NON-FATAL HERE. A raise inside sitecustomize would kill unrelated processes
for a reason they have nothing to do with. Instead the failure is printed unmissably, and the hard
backstop lives where it belongs: the trainer prints a '[vark] repeat: ...' line every step, and
the arm's own first-cycle check aborts if the trainer's realized group-size histogram does not
match the allocation (Gate C). Silence there means the patch did not take.
"""

import os
import sys

if os.environ.get("VARK_AUTOPATCH", "") == "1":
    try:
        import vark_patch

        if vark_patch.apply():
            print("[vark] sitecustomize: variable-k patch applied", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(
            f"!!! [vark] sitecustomize FAILED to apply the variable-k patch: "
            f"{type(exc).__name__}: {exc}. Training would run at UNIFORM k while claiming "
            f"variable k -- the first-cycle histogram check must abort this arm.",
            file=sys.stderr,
            flush=True,
        )
