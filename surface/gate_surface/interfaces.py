"""The four capability-equivalent interfaces for gate G_surface.

All four are CLIENT-SIDE adapters in front of one unmodified AWM MCP server.
The backend, the database and the verifier are identical across interfaces;
only the *presented* tool surface and the call-to-primitive mapping differ.

Capability-equivalence is a design invariant, enforced by construction and then
verified by execution in equivalence.py:

  * every interface's dispatcher can reach EVERY primitive of the scenario with
    ARBITRARY arguments (RAW/MACRO directly; SELECTED via the always-accepted
    hidden-primitive path plus `expand_toolset` for discovery; SPECIALIZED via
    `call_primitive`),
  * three of the four surfaces (RAW, SPECIALIZED, MACRO) are functions of the
    tool schemas ALONE and are byte-identical for every task in a scenario, so
    they cannot encode task-specific solution content,
  * SELECTED is the one task-conditioned interface (that is what makes it the
    incumbent baseline); its only task input is the task string, ranked by BM25
    over tool name+description. It never sees the verifier or a gold solution.

`build_macros()` takes ONLY the tool list -- it has no parameter through which a
task could reach it. That is the structural guarantee against macro cheating.
"""

from __future__ import annotations

import json
import math
import re

# ---------------------------------------------------------------- constants
SELECTED_K = 8            # primitives advertised by SELECTED
EXPAND_K = 5              # primitives revealed per expand_toolset call
MACRO_CAP = 12            # max schema-derived macros advertised by MACRO
GENERIC_BOUND_PARAMS = ("limit", "offset", "page", "skip", "per_page")

FINAL_TOOL = "submit_final_answer"
FINAL_SPEC = {
    "type": "function",
    "function": {
        "name": FINAL_TOOL,
        "description": "End the episode and report the final answer / summary of what you did.",
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "string",
                                      "description": "Final answer or summary."}},
            "required": ["answer"],
        },
    },
}


# ---------------------------------------------------------------- utilities
def _clean_schema(schema: dict) -> dict:
    """Strip pydantic/fastapi presentation noise ('title') from a JSON schema."""
    if not isinstance(schema, dict):
        return schema
    out = {}
    for k, v in schema.items():
        if k == "title":
            continue
        if isinstance(v, dict):
            out[k] = _clean_schema(v)
        elif isinstance(v, list):
            out[k] = [_clean_schema(x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    return out


def _short_description(desc: str) -> str:
    """SPECIALIZED's narrowing: keep the human summary, drop the auto-generated
    '### Responses' example-payload block.  Presentation only -- the tool still
    returns exactly the same payload."""
    cut = desc.split("### Responses")[0]
    lines = [l.strip() for l in cut.split("\n") if l.strip()]
    return " ".join(lines[:2])[:220] if lines else desc[:220]


def _fn_spec(name: str, description: str, schema: dict) -> dict:
    return {"type": "function",
            "function": {"name": name, "description": description,
                         "parameters": _clean_schema(schema)}}


# ---------------------------------------------------------------- BM25
_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(s: str) -> list:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return _TOKEN.findall(s.lower().replace("_", " "))


def bm25_rank(query: str, docs: list, k1=1.5, b=0.75) -> list:
    """Return indices of `docs` sorted by descending BM25 against `query`.
    Ties broken by original index so the ranking is deterministic."""
    dt = [_tok(d) for d in docs]
    N = len(dt)
    avgdl = sum(len(d) for d in dt) / max(N, 1)
    df = {}
    for d in dt:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    q = _tok(query)
    scores = []
    for i, d in enumerate(dt):
        tf = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t in q:
            if t not in tf:
                continue
            idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / max(avgdl, 1)))
        scores.append((-s, i))
    scores.sort()
    return [i for _, i in scores]


# ---------------------------------------------------------------- macros
_READ_VERBS = ("search", "list", "get", "find", "lookup", "query", "create_or_get")


def _singular(tok: str) -> str:
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("ses") or tok.endswith("xes") or tok.endswith("ches"):
        return tok[:-2]
    if tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def build_macros(tools: list) -> list:
    """Derive composite tools from the TOOL SCHEMAS ALONE.

    Signature takes only `tools`: there is no channel by which task text, the
    verifier, or a gold solution could enter a macro definition.

    One mechanical rule:
        RESOLVER  R = a tool with exactly one required parameter, of type
                      string, whose name starts with a read verb
                      (search/list/get/find/lookup/query/create_or_get).
        TARGET    T = a tool with exactly one required parameter named
                      '<E>_id' (integer or string).
        MATCH     singular(last token of E) occurs as a singularised token in
                  R's name, and R != T.
        EMIT      '<T>__by_name'  taking {'<E>_name': string} + T's other
                  required params, implemented as
                      R(<E>_name) -> pick matching id -> T(<E>_id=id, ...rest).

    Deterministic resolver preference among matches: names containing 'by_name',
    then 'search', then shortest name, then lexicographic.

    This is the generic "resolve-by-name then act" composition -- the same shape
    as `probe_06 --fork`'s by_name path, which was verified to reach the same
    terminal backend state as the by_id path.
    """
    resolvers = {}
    for t in tools:
        sch = t["inputSchema"]
        req = sch.get("required") or []
        if len(req) != 1:
            continue
        p = sch.get("properties", {}).get(req[0], {})
        if p.get("type") != "string":
            continue
        if not any(t["name"].startswith(v) for v in _READ_VERBS):
            continue
        toks = {_singular(x) for x in t["name"].split("_")}
        resolvers[t["name"]] = (t, req[0], toks)

    def _pref(rname):
        return (0 if "by_name" in rname else 1,
                0 if rname.startswith("search") else 1,
                len(rname), rname)

    macros = []
    for t in sorted(tools, key=lambda x: x["name"]):
        sch = t["inputSchema"]
        req = list(sch.get("required") or [])
        idparams = [r for r in req if r.endswith("_id")
                    and sch["properties"].get(r, {}).get("type") in ("integer", "string")]
        if len(idparams) != 1:
            continue
        idp = idparams[0]
        ent = idp[:-3]                       # 'playlist_id' -> 'playlist'
        ent_tok = _singular(ent.split("_")[-1])
        cands = [r for r, (_, _, toks) in resolvers.items()
                 if ent_tok in toks and r != t["name"]]
        if not cands:
            continue
        rname = sorted(cands, key=_pref)[0]
        rtool, rparam, _ = resolvers[rname]
        rest = [r for r in req if r != idp]
        props = {f"{ent}_name": {"type": "string",
                                 "description": f"Name / search text identifying the {ent}."}}
        for r in rest:
            props[r] = _clean_schema(sch["properties"][r])
        macros.append({
            "name": f"{t['name']}__by_name",
            "description": (f"Composite: resolve a {ent} by name with `{rname}`, then call "
                            f"`{t['name']}` with the resolved {idp}. "
                            f"{_short_description(t['description'])}"),
            "parameters": {"type": "object", "properties": props,
                           "required": [f"{ent}_name"] + rest},
            "_impl": {"resolver": rname, "resolver_param": rparam,
                      "target": t["name"], "id_param": idp, "entity": ent,
                      "rest": rest},
        })
    macros.sort(key=lambda m: m["name"])
    return macros[:MACRO_CAP]


def _extract_id(payload_text: str, wanted: str, entity: str):
    """Find the id of the entity whose name best matches `wanted` in a JSON reply.
    Deterministic: exact (case-insensitive) match first, then substring, then the
    first object carrying an 'id'."""
    try:
        obj = json.loads(payload_text)
    except Exception:
        m = re.search(r'"id"\s*:\s*(\d+)', payload_text)
        return int(m.group(1)) if m else None
    cands = []

    def walk(o):
        if isinstance(o, dict):
            if "id" in o and isinstance(o["id"], (int, str)) and not isinstance(o["id"], bool):
                nm = ""
                for key in ("name", "title", "username", "full_name", "subject", "label"):
                    if isinstance(o.get(key), str):
                        nm = o[key]
                        break
                cands.append((o["id"], nm))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    if not cands:
        return None
    w = (wanted or "").strip().lower()
    for i, nm in cands:
        if nm.lower() == w:
            return i
    for i, nm in cands:
        if w and (w in nm.lower() or nm.lower() in w):
            return i
    return cands[0][0]


# ---------------------------------------------------------------- interfaces
class Interface:
    """One interface instance for ONE episode (holds per-episode surface state)."""

    kind = "RAW"

    def __init__(self, kind: str, tools: list, task: str):
        self.kind = kind
        self.tools = tools
        self.by_name = {t["name"]: t for t in tools}
        self.task = task
        self.macros = {}
        self.visible = [t["name"] for t in tools]
        self.bound = {}          # tool -> {param: bound value}  (SPECIALIZED)
        self.extra_specs = []
        self._build()

    # ---- surface construction -------------------------------------------
    def _build(self):
        k = self.kind
        if k == "RAW":
            pass
        elif k == "SELECTED":
            docs = [f"{t['name']} {_short_description(t['description'])}" for t in self.tools]
            order = bm25_rank(self.task, docs)
            self.visible = [self.tools[i]["name"] for i in order[:SELECTED_K]]
            self._rank_order = [self.tools[i]["name"] for i in order]
            self.extra_specs = [{
                "type": "function",
                "function": {
                    "name": "expand_toolset",
                    "description": ("Reveal additional tools from this environment that are not "
                                    "currently listed. Describe what you need; the matching tool "
                                    "definitions are returned and become available to call."),
                    "parameters": {"type": "object",
                                   "properties": {"query": {"type": "string",
                                                            "description": "What you need to do."}},
                                   "required": ["query"]},
                },
            }]
        elif k == "SPECIALIZED":
            for t in self.tools:
                sch = t["inputSchema"]
                req = set(sch.get("required") or [])
                b = {}
                for p in GENERIC_BOUND_PARAMS:
                    if p in sch.get("properties", {}) and p not in req:
                        d = sch["properties"][p].get("default")
                        if d is not None:
                            b[p] = d
                if b:
                    self.bound[t["name"]] = b
            self.extra_specs = [{
                "type": "function",
                "function": {
                    "name": "call_primitive",
                    "description": ("Escape hatch: call any tool of this environment by its raw "
                                    "name with an arbitrary JSON argument object, including "
                                    "parameters not shown in the narrowed signatures "
                                    "(e.g. limit/offset)."),
                    "parameters": {"type": "object",
                                   "properties": {
                                       "tool_name": {"type": "string"},
                                       "arguments": {"type": "object",
                                                     "description": "JSON arguments object."}},
                                   "required": ["tool_name", "arguments"]},
                },
            }]
        elif k == "MACRO":
            for m in build_macros(self.tools):
                self.macros[m["name"]] = m
                self.extra_specs.append({
                    "type": "function",
                    "function": {"name": m["name"], "description": m["description"],
                                 "parameters": m["parameters"]},
                })
        else:
            raise ValueError(k)

    # ---- what the policy sees -------------------------------------------
    def specs(self) -> list:
        out = []
        for name in self.visible:
            t = self.by_name[name]
            if self.kind == "SPECIALIZED":
                sch = _clean_schema(t["inputSchema"])
                for p in self.bound.get(name, {}):
                    sch.get("properties", {}).pop(p, None)
                out.append(_fn_spec(name, _short_description(t["description"]), sch))
            else:
                out.append(_fn_spec(name, t["description"], t["inputSchema"]))
        out.extend(self.extra_specs)
        out.append(FINAL_SPEC)
        return out

    # ---- dispatch --------------------------------------------------------
    async def dispatch(self, call, name: str, args: dict):
        """`call(tool_name, arguments) -> text` is the raw MCP primitive call.

        Returns (reply_text, n_backend_calls, is_error).
        """
        if name == "expand_toolset" and self.kind == "SELECTED":
            q = args.get("query", "") if isinstance(args, dict) else ""
            hidden = [n for n in self._rank_order if n not in self.visible]
            docs = [f"{n} {_short_description(self.by_name[n]['description'])}" for n in hidden]
            if not hidden:
                return "No further tools available; every tool is already listed.", 0, False
            order = bm25_rank(q or self.task, docs)
            newly = [hidden[i] for i in order[:EXPAND_K]]
            self.visible = self.visible + newly
            revealed = [_fn_spec(n, self.by_name[n]["description"],
                                 self.by_name[n]["inputSchema"]) for n in newly]
            return ("Revealed and now callable:\n"
                    + json.dumps(revealed)[:4000]), 0, False

        if name == "call_primitive" and self.kind == "SPECIALIZED":
            tn = args.get("tool_name") if isinstance(args, dict) else None
            ta = args.get("arguments") if isinstance(args, dict) else None
            if isinstance(ta, str):
                try:
                    ta = json.loads(ta)
                except Exception:
                    ta = {}
            if tn not in self.by_name:
                return f"error: unknown tool '{tn}'", 0, True
            txt = await call(tn, ta or {})
            return txt, 1, _is_err(txt)

        if name in self.macros:
            m = self.macros[name]["_impl"]
            ent_key = f"{m['entity']}_name"
            wanted = args.get(ent_key, "") if isinstance(args, dict) else ""
            r_txt = await call(m["resolver"], {m["resolver_param"]: wanted})
            if _is_err(r_txt):
                return f"[macro {name}] resolver {m['resolver']} failed: {r_txt}", 1, True
            rid = _extract_id(r_txt, wanted, m["entity"])
            if rid is None:
                return (f"[macro {name}] no {m['entity']} matching '{wanted}'. "
                        f"resolver said: {r_txt[:500]}"), 1, True
            payload = {m["id_param"]: rid}
            for r in m["rest"]:
                if isinstance(args, dict) and r in args:
                    payload[r] = args[r]
            t_txt = await call(m["target"], payload)
            return (f"[macro {name}] resolved {m['entity']} -> id={rid}; "
                    f"{m['target']} returned: {t_txt}"), 2, _is_err(t_txt)

        if name in self.by_name:
            a = dict(args) if isinstance(args, dict) else {}
            for p, v in self.bound.get(name, {}).items():
                a.setdefault(p, v)
            txt = await call(name, a)
            return txt, 1, _is_err(txt)

        return (f"error: no tool named '{name}' is available. "
                f"Available: {', '.join(self.visible[:12])}..."), 0, True


def _is_err(txt: str) -> bool:
    if not txt:
        return True
    t = txt[:400].lower()
    return ("error" in t and ("detail" in t or "422" in t or "500" in t or "404" in t)) \
        or t.startswith("error") or "internal server error" in t


def _apply_subset(tools: list) -> list:
    """GATE_TOOL_SUBSET restricts the ADVERTISED tool set to a named subset.

    This is the edit space for harness evolution over an MCP surface: the server chooses
    what appears in tools/list, which is a protocol-level control surface (cf. SafeMCP,
    ACL 2026, which learns server-side tool filtering for safety). Unset -> no filtering,
    so every banked gate re-derives byte-identically.
    """
    import os as _os
    raw = _os.environ.get("GATE_TOOL_SUBSET")
    if not raw:
        return tools
    keep = {t.strip() for t in raw.split(",") if t.strip()}
    sub = [t for t in tools if t.get("name") in keep]
    if not sub:
        raise ValueError(f"GATE_TOOL_SUBSET matched 0 of {len(tools)} tools; refusing to "
                         f"run an agent with an empty action space")
    return sub


def build(kind: str, tools: list, task: str) -> Interface:
    tools = _apply_subset(tools)
    return Interface(kind, tools, task)


KINDS = ("RAW", "SELECTED", "SPECIALIZED", "MACRO")
