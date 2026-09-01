"""Family-aware tool-call parsing and format instructions for G-CALLER.

Why this file exists
--------------------
`run_gate.parse_call` is Qwen/Hermes-shaped: `<tool_call>{...}</tool_call>` plus a
bare-JSON fallback anchored at end of string. C-TOOLPARSE measured what the other
three admitted families actually emit, read off their rendered prompts:

    qwen     <tool_call>{"name":..,"arguments":..}</tool_call>   single object
    mistral  [TOOL_CALLS][{...}]              no format instruction in its template
    granite  <|tool_call|>[{...}]             no format instruction when a system
                                              message is supplied
    falcon3  <tool_call>[{...}]</tool_call>   JSON LIST, so `_TC`'s `\\{` misses it

Three of four defeat the frozen parser. Unfixed, those three parse-fail every turn,
terminate `no_tool_call`, and score ~0 on every interface — manufacturing exactly
the caller heterogeneity B1 exists to measure.

The instrument is load-bearing
------------------------------
`G_surface`'s banked numbers were produced by the frozen parser, so this extension
is **strictly additive and delegation-first**: `parse_call` calls the frozen
`run_gate.parse_call` and returns its result unchanged whenever it is not None.
Fallbacks run only on inputs the frozen parser rejected (returned None on). So for
every input the frozen parser handled, output is identical *by construction* — a
property of the control flow, not a sample of a test corpus. `selftest()` checks it
anyway, because "verify by execution" is the repo's rule.

Byte-identity is asserted by construction rather than by replaying real Qwen
episode texts, because those no longer exist: `gate_surface/work/` is gitignored and
the box that held it is down. Stated rather than glossed.

Single source of truth
----------------------
This module owns BOTH the instruction we give a family AND the syntax we parse back.
They live together so they cannot drift apart, which is how a "model is bad at this
interface" artifact gets manufactured.
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_GATE = os.path.join(os.path.dirname(_HERE), "gate_surface")
if _GATE not in sys.path:
    sys.path.insert(0, _GATE)

import run_gate  # noqa: E402  the FROZEN instrument, imported never copied

# The one line of run_gate.SYSTEM that names a wire format. Everything else in the
# system prompt is family-independent and must stay byte-identical.
_QWEN_LINE = "- Call exactly ONE tool per turn, using the <tool_call> format.\n"

# Native convention per family, phrased as closely to the frozen line as possible.
# Only the *format* clause differs; "exactly ONE tool per turn" is preserved for all.
_FORMAT_LINE = {
    "qwen": _QWEN_LINE,
    "falcon3": (
        "- Call exactly ONE tool per turn, using the <tool_call> format.\n"
    ),
    "mistral": (
        "- Call exactly ONE tool per turn. Emit the call as [TOOL_CALLS] followed by\n"
        '  a JSON list of one object: [TOOL_CALLS] [{"name": ..., "arguments": {...}}]\n'
    ),
    "granite": (
        "- Call exactly ONE tool per turn. Emit the call as <|tool_call|> followed by\n"
        '  a JSON list of one object: <|tool_call|> [{"name": ..., "arguments": {...}}]\n'
    ),
    # Added 2026-08-01 for the G-INVARIANCE matrix, which needs >=5 distinct families.
    # Both cleared C-TOOLTMPL on their own templates (tool_block_cost 129 and 67, against
    # qwen 129 / granite 86 as controls). GLM-4, Gemma-3 and InternLM2.5 were rejected:
    # their templates accept tools= and render NOTHING, cost 0.
    "llama31": (
        "- Call exactly ONE tool per turn. Emit the call as a single JSON object:\n"
        '  {"name": ..., "parameters": {...}}\n'
    ),
    "ministral": (
        "- Call exactly ONE tool per turn. Emit the call as [TOOL_CALLS] followed by\n"
        '  a JSON list of one object: [TOOL_CALLS] [{"name": ..., "arguments": {...}}]\n'
    ),
}


def system_for(family: str) -> str:
    """`run_gate.SYSTEM` with only its wire-format line swapped for `family`'s.

    For `qwen` the return value is byte-identical to `run_gate.SYSTEM`; asserted in
    `selftest`. Falcon3 is also byte-identical: its native convention already IS
    `<tool_call>` tags, and its own template supplies the list-vs-object detail.
    """
    if family not in _FORMAT_LINE:
        raise KeyError(f"unknown family {family!r}; known: {sorted(_FORMAT_LINE)}")
    line = _FORMAT_LINE[family]
    if line == _QWEN_LINE:
        return run_gate.SYSTEM
    if _QWEN_LINE not in run_gate.SYSTEM:
        raise RuntimeError(
            "run_gate.SYSTEM no longer contains the expected wire-format line; "
            "gate_caller/parse_multi.py must be updated deliberately, not silently."
        )
    return run_gate.SYSTEM.replace(_QWEN_LINE, line)


# --------------------------------------------------------------------------
# JSON extraction
# --------------------------------------------------------------------------

_OPEN = {"{": "}", "[": "]"}


def _first_json_value(s: str):
    """First complete balanced JSON object/array in `s`, or None.

    A balanced scanner rather than a regex: generations are truncated at
    `MAX_NEW_TOKENS`, and non-greedy `\\{.*?\\}` stops at the first inner `}`, which
    silently yields invalid JSON on any nested payload. String-aware so braces
    inside string literals do not confuse the depth count.
    """
    for i, ch in enumerate(s):
        if ch not in _OPEN:
            continue
        depth, in_str, esc = 0, False, False
        for j in range(i, len(s)):
            c = s[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c in _OPEN:
                depth += 1
            elif c in ("}", "]"):
                depth -= 1
                if depth == 0:
                    frag = s[i:j + 1]
                    try:
                        return json.loads(frag)
                    except Exception:
                        break  # not valid JSON from here; try next opener
        # fall through to next candidate opener
    return None


def _normalise(obj):
    """Coerce a parsed tool call into `(name, args)`, or None.

    Handles: a bare `{"name","arguments"}`; a `{"type":"function","function":{...}}`
    envelope (Falcon3's assistant branch, Mistral's tool list); a JSON list of
    either. Only the FIRST element of a list is used — the gate's contract is
    "exactly ONE tool per turn", so a multi-call emission is truncated to its first
    call rather than silently dropped.
    """
    if isinstance(obj, list):
        for item in obj:
            got = _normalise(item)
            if got is not None:
                return got
        return None
    if not isinstance(obj, dict):
        return None
    if "function" in obj and isinstance(obj["function"], dict):
        obj = obj["function"]
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    return name, (args if isinstance(args, dict) else {})


_MARKERS = ("<|tool_call|>", "[TOOL_CALLS]", "<tool_call>")


def _fallback(text: str):
    # Marker-led: take the first complete JSON value after the marker.
    for marker in _MARKERS:
        idx = text.find(marker)
        if idx != -1:
            got = _normalise(_first_json_value(text[idx + len(marker):]))
            if got is not None:
                return got
    # Unmarked: some checkpoints emit a bare JSON list with no marker at all.
    return _normalise(_first_json_value(text))


def parse_call(text: str, family: str | None = None):
    """`(name, args)` or None. Frozen parser first, always."""
    frozen = run_gate.parse_call(text)
    if frozen is not None:
        return frozen
    return _fallback(text)


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------

_FROZEN_OK = [
    '<tool_call>\n{"name": "list_orders", "arguments": {"customer_id": "c_42"}}\n</tool_call>',
    '<tool_call>{"name": "f", "arguments": {}}</tool_call>',
    'thinking...\n<tool_call>{"name": "g", "arguments": {"a": 1, "b": {"c": 2}}}</tool_call>',
    '{"name": "bare_tail", "arguments": {"x": "y"}}',
    '<tool_call>{"name": "s", "arguments": "{\\"k\\": 1}"}</tool_call>',
]

_FAMILY_CASES = [
    ("falcon3",
     '<tool_call>\n[\n  {"name": "list_orders", "arguments": {"customer_id": "c_42"}}\n]\n</tool_call>',
     "list_orders", {"customer_id": "c_42"}),
    ("falcon3",
     '<tool_call>\n[{"type":"function","function":{"name":"lo","arguments":{"i":"1"}}}]',
     "lo", {"i": "1"}),
    ("mistral",
     '[TOOL_CALLS] [{"name": "list_orders", "arguments": {"customer_id": "c_42"}, "id": "abcdefghi"}]',
     "list_orders", {"customer_id": "c_42"}),
    ("granite",
     '<|tool_call|>[{"name": "list_orders", "arguments": {"customer_id": "c_42"}}]',
     "list_orders", {"customer_id": "c_42"}),
    ("granite",
     '<|tool_call|>[{"name": "nested", "arguments": {"a": {"b": [1, 2]}}}]<|end_of_text|>',
     "nested", {"a": {"b": [1, 2]}}),
]

_MUST_FAIL = [
    "I will now list the orders for you.",
    "",
    "<tool_call>not json at all</tool_call>",
    '<tool_call>{"arguments": {"a": 1}}</tool_call>',      # no name
    '[TOOL_CALLS] [{"arguments": {}}]',                     # no name
    '<tool_call>{"name": "trunc", "arguments": {"a": ',      # truncated mid-JSON
]


def selftest() -> int:
    fails = []

    # 1. Delegation: identical to the frozen parser wherever it returns non-None.
    for s in _FROZEN_OK:
        f, m = run_gate.parse_call(s), parse_call(s)
        if f is None:
            fails.append(f"corpus case not handled by FROZEN parser: {s[:60]!r}")
        elif f != m:
            fails.append(f"DELEGATION BROKEN frozen={f!r} multi={m!r} on {s[:60]!r}")

    # 2. Every family's real emitted syntax resolves correctly.
    for fam, s, want_name, want_args in _FAMILY_CASES:
        got = parse_call(s, fam)
        if got != (want_name, want_args):
            fails.append(f"{fam}: got {got!r}, want {(want_name, want_args)!r} "
                         f"on {s[:70]!r}")

    # 3. Negative controls — additive must not become permissive.
    for s in _MUST_FAIL:
        got = parse_call(s)
        if got is not None:
            fails.append(f"FALSE POSITIVE {got!r} on {s[:60]!r}")

    # 4. Prompt deviation is minimal and qwen is untouched.
    if system_for("qwen") != run_gate.SYSTEM:
        fails.append("system_for('qwen') is not byte-identical to run_gate.SYSTEM")
    if system_for("falcon3") != run_gate.SYSTEM:
        fails.append("system_for('falcon3') should be byte-identical (native <tool_call>)")
    for fam in ("mistral", "granite"):
        s = system_for(fam)
        if _QWEN_LINE in s:
            fails.append(f"system_for({fam!r}) still contains the qwen format line")
        if "exactly ONE tool per turn" not in s:
            fails.append(f"system_for({fam!r}) dropped the one-call-per-turn rule")
        # every other line must be preserved verbatim
        base = set(run_gate.SYSTEM.split("\n")) - set(_QWEN_LINE.strip("\n").split("\n"))
        missing = [ln for ln in base if ln and ln not in s]
        if missing:
            fails.append(f"system_for({fam!r}) dropped lines: {missing[:2]}")

    for f in fails:
        print(f"[parse_multi] FAIL {f}", flush=True)
    n_checks = len(_FROZEN_OK) + len(_FAMILY_CASES) + len(_MUST_FAIL) + 4
    if fails:
        print(f"[parse_multi] {len(fails)} FAILURES of ~{n_checks} checks", flush=True)
        return 1
    print(f"[parse_multi] all ~{n_checks} checks PASS "
          f"(delegation-first, {len(_FAMILY_CASES)} family syntaxes, "
          f"{len(_MUST_FAIL)} negative controls)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(selftest())


# --------------------------------------------------------------------------
# History normalisation -- the INPUT-side mirror of `system_for`
# --------------------------------------------------------------------------
#
# `run_gate` replays an assistant turn as raw text (`{"role":"assistant",
# "content": <generated text>}`) and then appends `{"role":"tool", ...}`. Mistral's
# chat template refuses that shape: any assistant turn preceding a tool result must
# carry structured `tool_calls`, each with a 9-character ALPHANUMERIC id, or it
# raises
#     TemplateError: Tool call IDs should be alphanumeric strings with length 9!
# at render time -- on turn 2 of every episode, killing the run.
#
# This was missed because C-TOOLTMPL, C-TOOLPARSE and C-TOOLPARSE-FIX all probe a
# SINGLE synthetic turn. The fault only appears once a tool result enters the
# conversation. Worse, the earlier "fix" added an id to the PROBE, which silenced the
# detector while leaving the runtime path broken.
#
# STRICT NO-OP for qwen / falcon3 / granite: their templates accept the raw-text
# shape, and rewriting their history would change the rendered prompt and therefore
# their results. Only families in _NEEDS_STRUCTURED are touched.

# Ministral is Mistral-family and inherits the same template constraint: replayed history
# is rejected unless assistant turns carry a 9-char tool_call.id AND the following tool
# result carries a matching tool_call_id. Mistral-7B-v0.3 produced 0/480 episodes before
# normalise_history existed; assume the same here until a smoke test says otherwise.
_NEEDS_STRUCTURED = {"mistral", "ministral"}


def _call_id(name: str, args, i: int) -> str:
    """Deterministic 9-char alphanumeric id. Deterministic so a re-run of the same
    episode renders the same prompt; Mistral only validates the shape, not the value."""
    import hashlib
    h = hashlib.sha1(f"{i}:{name}:{json.dumps(args, sort_keys=True)}".encode()).hexdigest()
    return "".join(c for c in h if c.isalnum())[:9]


def normalise_history(msgs, family: str):
    """Return `msgs` in the shape `family`'s chat template will accept.

    For every family except those in `_NEEDS_STRUCTURED` this returns the input list
    unchanged (identity, same object), so no other family's prompt can shift.
    """
    if family not in _NEEDS_STRUCTURED:
        return msgs
    # The template raises in TWO places, both checked with |length != 9:
    #   tool_call.id        on the assistant turn that made the call
    #   message.tool_call_id on the tool-result turn that answers it
    # They must be paired, so the id is carried forward to the next tool message.
    out, n, last_id = [], 0, None
    for m in msgs:
        role = m.get("role")
        if role == "assistant" and not m.get("tool_calls"):
            parsed = parse_call(m.get("content") or "", family=family)
            if parsed is not None:
                name, args = parsed[0], parsed[1]
                last_id = _call_id(name, args, n)
                out.append({"role": "assistant", "content": "",
                            "tool_calls": [{"type": "function", "id": last_id,
                                            "function": {"name": name,
                                                         "arguments": args}}]})
                n += 1
                continue
            last_id = None
        elif role == "tool" and not m.get("tool_call_id"):
            mm = dict(m)
            mm["tool_call_id"] = last_id or _call_id("orphan", {}, n)
            out.append(mm)
            continue
        out.append(m)
    return out
