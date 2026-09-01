"""Bank the admission-rate identity cells that the paper's Figure 1 plots.

WHY THIS EXISTS. make_figures.py promises in its own docstring that "every number is read from the
receipts, never hard-coded, so the figures cannot drift from the banked results". fig_identity()
was the one function that broke that promise: its nine (tie, predicted, observed) triples were
literals, and six of the nine had no banked counterpart anywhere in the repo to drift *from*. The
two producer scripts print to stdout and write nothing, so the summary statistics existed only
inside a .tex table and a figure literal. This script runs those two producers and banks what they
print, so the figure and the appendix table have one source.

TWO ESTIMATORS, ON PURPOSE, AND NOT INTERCHANGEABLE. The MCP rows come from
checks/g1_chance_rate.py, whose unit is (ordered interface pair, one seed draw) -- 12 pairs x 500
draws = 6,000, which is the granularity the 43.3% attribution headline was computed at. The BFCL
and API-Bank rows come from substrate_report.py, whose unit is a group-of-4 resample over 1,500
trials. Both estimate the same quantity and both verify the same identity, but they are NOT the
same code path, and substrate_report.py run on the MCP pool gives a visibly different tie rate
(0.084 against 0.088). This script therefore records `estimator` on every cell rather than
presenting the nine as one homogeneous measurement.

Run:  python bank_identity_cells.py            # writes surface/receipts/identity_cells.json
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

R = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PY = sys.executable
OUT = f"{R}/surface/receipts/identity_cells.json"

CALLERS = ["qwen", "granite", "mistral", "falcon3"]
# (substrate label, variant label, episode file template). A variant is a tool-surface setting of
# one substrate, NOT a different model -- API-Bank is measured under the published (gold) surface
# and again under the full 49-tool surface, and THIRD_SUBSTRATE.md says to treat the two as a pair.
SUBSTRATES = [
    ("BFCL", "live_multiple", f"{R}/work/bfcl/runs/{{c}}_BFCL_v3_live_multiple.jsonl"),
    ("API-Bank", "gold", f"{R}/work/apibank/runs/{{c}}_level1_gold.jsonl"),
    ("API-Bank", "49 tools", f"{R}/work/apibank/runs/{{c}}_level1_all.jsonl"),
]

_MCP_ROW = re.compile(r"^\s+(\w+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([-+][\d.]+)\s*$")
_G1 = re.compile(r"G1\s+tie rate ([\d.]+) -> predicted chance credit ([\d.]+) \| measured ([\d.]+)")
# re.M matters: substrate_report.py prints this on the THIRD line of its stdout, and `^` without
# re.MULTILINE anchors to the start of the whole string, so this silently matched nothing and every
# banked cell recorded tasks/seeds/episodes as null -- the provenance the receipt exists to carry.
_N = re.compile(r"^\s+(\d+) tasks, (\d+) seeds, (\d+) episodes", re.M)


def mcp_cells():
    """The four MCP cells, from the estimator the 43.3% headline was computed at."""
    p = subprocess.run([PY, f"{R}/surface/gate_caller/checks/g1_chance_rate.py"],
                       capture_output=True, text=True, check=True)
    out = []
    for line in p.stdout.splitlines():
        m = _MCP_ROW.match(line)
        if m and m.group(1) in CALLERS:
            out.append(dict(substrate="MCP", variant="4 interfaces", caller=m.group(1),
                            estimator="g1_chance_rate", draws=int(m.group(2)),
                            tie=float(m.group(3)), predicted=float(m.group(4)),
                            observed=float(m.group(5))))
    return out


def substrate_cell(sub, variant, path, caller):
    """One BFCL / API-Bank cell, or None when that model was never run on that substrate."""
    if not os.path.exists(path):
        return None
    p = subprocess.run([PY, f"{R}/surface/gate_caller/substrate_report.py",
                        "--episodes", path], capture_output=True, text=True, check=True)
    g1 = _G1.search(p.stdout)
    n = _N.search(p.stdout)
    if not g1:
        return None
    return dict(substrate=sub, variant=variant, caller=caller, estimator="substrate_report",
                tasks=int(n.group(1)) if n else None,
                seeds=int(n.group(2)) if n else None,
                episodes=int(n.group(3)) if n else None,
                tie=float(g1.group(1)), predicted=float(g1.group(2)),
                observed=float(g1.group(3)))


def main():
    cells = mcp_cells()
    missing = []
    for sub, variant, tmpl in SUBSTRATES:
        for c in CALLERS:
            got = substrate_cell(sub, variant, tmpl.format(c=c), c)
            if got:
                cells.append(got)
            else:
                missing.append(dict(substrate=sub, variant=variant, caller=c,
                                    reason="no episode file: this model was never run here"))
    for x in cells:
        x["gap"] = round(x["observed"] - x["predicted"], 4)
    doc = dict(
        note="Cells behind Figure 1. Two estimators; see `estimator` on each cell and the "
             "module docstring of bank_identity_cells.py. Never merge the two into one mean.",
        cells=cells, missing=missing)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(doc, open(OUT, "w"), indent=2)
    print(f"banked {len(cells)} cells, {len(missing)} known-missing -> {OUT}")
    for x in cells:
        print("  %-9s %-14s %-8s tie %.3f  pred %.3f  obs %.3f  gap %+.3f"
              % (x["substrate"], x["variant"], x["caller"], x["tie"], x["predicted"],
                 x["observed"], x["gap"]))
    for x in missing:
        print("  MISSING  %-9s %-14s %s" % (x["substrate"], x["variant"], x["caller"]))


if __name__ == "__main__":
    main()
