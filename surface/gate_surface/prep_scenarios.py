"""Materialise per-scenario initial .db files and dump the RAW tool surface.

Run with the AWM conda python (it owns `awm.core.reset`):
    $AWM_PY gate_surface/prep_scenarios.py

Outputs (all under gate_surface/work/):
    dbs/<scenario>.initial.db      canonical start state, byte-identical per reset
    surfaces/<scenario>.json       MCP tool list exactly as advertised by the server
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from awm_env import REPO, AwmServer, db_content_digest, db_digest, free_port  # noqa: E402

from mcp import ClientSession  # noqa: E402
from mcp.client.streamable_http import streamablehttp_client  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# GATE_WORK lets a run materialise into a scratch directory instead of overwriting
# the committed corpus in gate_surface/work/. That is what makes it possible to
# REBUILD the substrate on a new box and diff the result against the committed
# dbs/surfaces, rather than clobbering the reference and then having nothing to
# compare against. Default is unchanged.
WORK = os.environ.get("GATE_WORK") or os.path.join(HERE, "work")
DBS = os.path.join(WORK, "dbs")
SURF = os.path.join(WORK, "surfaces")


def build_db(scenario: str) -> str:
    from awm.core.reset import reset_single_database
    tmp = os.path.join(WORK, "_reset")
    os.makedirs(tmp, exist_ok=True)
    path = reset_single_database(
        input_db=os.path.join(REPO, "outputs", "gen_db.jsonl"),
        input_sample=os.path.join(REPO, "outputs", "gen_sample.jsonl"),
        scenario=scenario, database_dir=tmp)
    dst = os.path.join(DBS, f"{scenario}.initial.db")
    shutil.copy(path, dst)
    return dst


async def dump_surface(scenario: str, db: str) -> dict:
    work_db = os.path.join(WORK, "_serve", f"{scenario}.db")
    os.makedirs(os.path.dirname(work_db), exist_ok=True)
    shutil.copy(db, work_db)
    srv = AwmServer(scenario, os.path.join(WORK, "_srv"), work_db, port=free_port()).start()
    if not srv.wait_ready():
        print(f"  [FAIL] {scenario} server did not start")
        print(srv.tail(1500))
        srv.stop()
        return {}
    try:
        async with streamablehttp_client(srv.url) as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize()
                tools = (await s.list_tools()).tools
                out = [{"name": t.name,
                        "description": t.description or "",
                        "inputSchema": t.inputSchema} for t in tools]
    finally:
        srv.stop()
    return {"scenario": scenario, "n_tools": len(out), "tools": out}


async def main():
    scenarios = sys.argv[1:]
    os.makedirs(DBS, exist_ok=True)
    os.makedirs(SURF, exist_ok=True)
    ok = 0
    for sc in scenarios:
        try:
            db = build_db(sc)
            # determinism check: rebuild and compare content digest
            d1 = db_content_digest(db)
            db2 = build_db(sc)
            d2 = db_content_digest(db2)
            surf = await dump_surface(sc, db)
            if not surf:
                print(f"[FAIL] {sc}")
                continue
            with open(os.path.join(SURF, f"{sc}.json"), "w") as f:
                json.dump(surf, f, indent=1)
            tables = db_digest(db)
            print(f"[OK] {sc}: {surf['n_tools']} tools, "
                  f"{sum(tables.values())} rows / {len(tables)} tables, "
                  f"reset_deterministic={d1 == d2} digest={d1}")
            ok += 1
        except Exception as exc:
            print(f"[FAIL] {sc}: {type(exc).__name__}: {exc}")
    print(f"RESULT {ok}/{len(scenarios)} scenarios prepared")
    return 0 if ok == len(scenarios) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
