"""AWM environment process management for the G_surface gate.

One AWM scenario = one FastAPI+fastapi-mcp server process over one SQLite file.
Adapted from mcp_probe/probe_06_awm.py (which verified all of this by execution);
the two frictions recorded there are carried over:

  * conda exports HOST=<build triplet>  -> force HOST=127.0.0.1 in the child env
  * --output_dir without --db_path makes AWM copy final.db onto itself at exit
    -> always pass --db_path explicitly

Difference from the probe: the server process here is *long-lived* and the
backing .db is restored underneath it between episodes (verified in the probe as
checks 6f/6g, and re-asserted per episode here by digest).
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.environ.get("AWM_REPO") or os.path.join(ROOT, "mcp_probe", "repos", "agent-world-model")
ENVS = os.path.join(REPO, "outputs", "gen_envs.jsonl")

# The AWM server interpreter. An earlier revision hard-coded one absolute path, which
# is why nothing here ran on a second machine. Read it from the environment
# (see env.sh at the repository root), and fail loudly
# at import rather than handing subprocess a path that does not exist -- a missing
# interpreter otherwise surfaces as an opaque server-start timeout per episode.
AWM_PY = os.environ.get("AWM_PY", "")
if not AWM_PY:
    raise RuntimeError(
        "AWM_PY is unset. Source the project env first:\n"
        "    . $BRACE_ROOT/env.sh\n"
        "It must point at the mcp_awm interpreter (MCP client + AWM server, no transformers)."
    )
if not os.path.exists(AWM_PY):
    raise RuntimeError(f"AWM_PY={AWM_PY!r} does not exist; build it with setup_envs.sh")


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def db_digest(path: str) -> dict:
    """Row counts per table -- coarse but honest fingerprint of backend state."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    out = {}
    for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' "
                            "AND name NOT LIKE 'sqlite_%' ORDER BY name"):
        out[t] = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    con.close()
    return out


_TS_RE = None


def strip_timestamps(s: str) -> str:
    """Replace wall-clock timestamps with a placeholder.

    AWM's generated models default `created_at`/`added_at`/... to
    `datetime.utcnow`, so two executions of the SAME call at different instants
    produce different bytes. That is environment non-determinism, not interface
    behaviour, and must not be mistaken for a capability difference.
    """
    global _TS_RE
    if _TS_RE is None:
        import re
        _TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?")
    return _TS_RE.sub("<TS>", s)


def db_content_digest(path: str, ignore_time: bool = False) -> str:
    """Stronger fingerprint: sha256 over every row of every table, sorted.

    `ignore_time=True` blanks columns whose name marks them as wall-clock
    timestamps, for comparisons across executions at different instants.
    """
    import hashlib
    h = hashlib.sha256()
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    tables = [t for (t,) in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    for t in tables:
        h.update(t.encode())
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
        drop = {i for i, c in enumerate(cols)
                if ignore_time and (c.endswith("_at") or c in ("timestamp", "created", "updated"))}
        rows = []
        for r in con.execute(f"SELECT * FROM {t}"):
            v = tuple("<TS>" if i in drop else x for i, x in enumerate(r))
            rows.append(strip_timestamps(repr(v)) if ignore_time else repr(v))
        for r in sorted(rows):
            h.update(r.encode())
    con.close()
    return h.hexdigest()[:16]


class AwmServer:
    """A long-lived AWM environment server bound to one working .db path."""

    def __init__(self, scenario: str, workdir: str, db_path: str, port: int | None = None):
        self.scenario = scenario
        self.db_path = db_path
        self.port = port or free_port()
        self.outdir = os.path.join(workdir, f"{scenario}_{self.port}")
        os.makedirs(self.outdir, exist_ok=True)
        self.url = f"http://127.0.0.1:{self.port}/mcp"
        self.proc = None
        self.log = None

    def start(self):
        cmd = [AWM_PY, "-m", "awm.core.server", "--scenario", self.scenario,
               "--envs_load_path", ENVS, "--port", str(self.port),
               "--output_dir", self.outdir, "--db_path", self.db_path,
               "--temp_server_path", os.path.join(self.outdir, "server.py")]
        env = dict(os.environ, HOST="127.0.0.1", PORT=str(self.port))
        env.pop("CUDA_VISIBLE_DEVICES", None)
        self.log = open(os.path.join(self.outdir, "run.log"), "w")
        self.proc = subprocess.Popen(cmd, cwd=REPO, stdout=self.log,
                                     stderr=subprocess.STDOUT, env=env)
        return self

    def wait_ready(self, timeout=240) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                return False
            try:
                s = socket.create_connection(("127.0.0.1", self.port), 1)
                s.close()
                return True
            except OSError:
                time.sleep(0.25)
        return False

    def restore(self, source_db: str):
        """Restore the backing state underneath the live server (file copy)."""
        for suffix in ("-wal", "-shm", "-journal"):
            p = self.db_path + suffix
            if os.path.exists(p):
                os.remove(p)
        shutil.copy(source_db, self.db_path)

    def tail(self, n=3000) -> str:
        try:
            return open(os.path.join(self.outdir, "run.log")).read()[-n:]
        except OSError:
            return ""

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        try:
            self.log.close()
        except Exception:
            pass
        # Remove the per-server slot dir. outdir is "{scenario}_{random_port}", so every
        # server start created a permanent new entry and nothing ever deleted one: _srv
        # reached 9,846 entries by 2026-07-31 and 400 more within 90 min of the next run.
        # Lookups in a directory that large cost ~1 s (machine B measured ~900 ms at 18,483
        # entries on a different subtree the same night), which is what made scenario prep
        # collapse while the rest of the filesystem was fine.
        # Set GATE_KEEP_SRV=1 to retain slot dirs when debugging a server that failed.
        if os.environ.get("GATE_KEEP_SRV") != "1":
            try:
                shutil.rmtree(self.outdir, ignore_errors=True)
            except Exception:
                pass    # cleanup must never break a run


def scenario_list() -> list:
    return [json.loads(l)["scenario"] for l in open(ENVS)]


def load_tasks(scenario: str) -> list:
    p = os.path.join(REPO, "outputs", "gen_tasks.jsonl")
    for line in open(p):
        d = json.loads(line)
        if d["scenario"] == scenario:
            return d["tasks"]
    raise KeyError(scenario)


def load_verifiers(scenario: str) -> dict:
    """task_idx -> verifier python source."""
    p = os.path.join(REPO, "outputs", "gen_verifier.pure_code.jsonl")
    out = {}
    for line in open(p):
        d = json.loads(line)
        if d["scenario"] == scenario:
            out[d["task_idx"]] = d["verification"].get("code", "")
    return out


def run_verifier(code: str, initial_db: str, final_db: str, final_answer: str | None):
    """Execute an AWM code verifier. Returns (result, status).

    result in {"complete","others"}; status in {"success","error"}.
    Field names discovered from awm/core/verify.py, not assumed.
    """
    if not code or len(code.strip()) < 10:
        return "others", "no_code"
    func_name = "verify_task_completion"
    for line in code.split("\n"):
        line = line.strip()
        if line.startswith("def verify_") and "(" in line:
            func_name = line.split("(")[0].replace("def ", "").strip()
            break
    ns = {}
    try:
        exec(compile(code, "<verifier>", "exec"), ns)
        fn = ns[func_name]
        res = fn(initial_db, final_db, final_answer)
    except Exception as exc:
        return "others", f"error:{type(exc).__name__}:{exc}"[:200]
    if not isinstance(res, dict) or "result" not in res:
        return "others", "bad_return"
    v = res.get("result", "others")
    return (v if v in ("complete", "others") else "others"), "success"


if __name__ == "__main__":
    print(f"scenarios: {len(scenario_list())}")
    print(f"awm python: {AWM_PY} exists={os.path.exists(AWM_PY)}")
    print(f"envs jsonl: {ENVS} exists={os.path.exists(ENVS)}")
    sys.exit(0)
