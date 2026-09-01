"""Consolidate every deep-regime DISCORD decision across callers and proposal streams."""
import json, os, re
R = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ARMS = [("qwen",   "stream 1", "logs/awm_discord_deep.log"),
        ("qwen",   "stream 2", "logs/awm_discord_deep_seed2.log"),
        ("qwen",   "stream 3", "logs/awm_discord_deep_seed3.log"),
        ("qwen",   "stream 4", "logs/awm_discord_deep_seed4.log"),
        ("granite","stream 1", "logs/awm_discord_deepg.log")]
rows, out, acc_all, rej_all = [], {}, [], []
for fam, stream, log in ARMS:
    p = os.path.join(R, log)
    if not os.path.exists(p): continue
    seen, dec = set(), []
    for line in open(p, errors="ignore"):
        m = re.search(r"\[discord-awm\] (r\d+c\d+) advertised=(\d+) b=(\d+) c=(\d+) p=([\d.]+) "
                      r"tasks=(\d+)/\d+ (ACCEPT|reject)", line)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            dec.append({"id":m.group(1),"b":int(m.group(3)),"c":int(m.group(4)),
                        "p":float(m.group(5)),"tasks":int(m.group(6)),
                        "accept":m.group(7)=="ACCEPT"})
    if not dec: continue
    acc = [d for d in dec if d["accept"]]
    rej = [d for d in dec if not d["accept"]]
    acc_all += acc; rej_all += rej
    out[f"{fam}/{stream}"] = dec
    at = ", ".join(f"$b{{=}}{d['b']}$/$c{{=}}{d['c']}$ in {d['tasks']}" for d in acc) or "--"
    rows.append(f"{fam} & {stream} & {len(dec)} & {len(acc)} & {at} \\\\")
open(f"{R}/surface/paper/tex/replicates.tex","w").write(
 "\\begin{tabular}{@{}llccl@{}}\n\\toprule\ncaller & proposal stream & decisions & accepts & "
 "evidence at acceptance (tasks) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
summary = {"decisions": len(acc_all)+len(rej_all), "accepts": len(acc_all),
           "accept_tasks": [d["tasks"] for d in acc_all],
           "accept_p": [d["p"] for d in acc_all],
           "reject_tasks": [d["tasks"] for d in rej_all], "by_arm": out}
if acc_all:
    summary["median_tasks_to_accept"] = sorted(d["tasks"] for d in acc_all)[len(acc_all)//2]
    summary["max_p_at_accept"] = max(d["p"] for d in acc_all)
json.dump(summary, open(f"{R}/surface/receipts/deep_replicates.json","w"), indent=2)
for r in rows: print("  " + r)
if acc_all:
    print(f"\n  {len(acc_all)} accepts over {len(acc_all)+len(rej_all)} decisions; "
          f"median {summary['median_tasks_to_accept']} tasks to accept; "
          f"max p at acceptance {summary['max_p_at_accept']:.4f}")
    print(f"  rejects consumed a median of "
          f"{sorted(d['tasks'] for d in rej_all)[len(rej_all)//2]} tasks")
