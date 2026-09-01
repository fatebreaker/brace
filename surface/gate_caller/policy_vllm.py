"""vLLM-backed caller policy — a drop-in for `policy_multi`, same public API.

WHY. The cube is prefill-bound, measured not assumed: 47,228 prompt tokens per
episode against 381 generated, and 44,303 of those prompt tokens are the tool
schema block, re-encoded on every one of ~6 turns. The schema is byte-identical
across turns, across the 8 seeds of a task, and across every episode on the same
(scenario, interface). `transformers.generate()` re-prefills it every time.

vLLM addresses exactly that, in three ways this workload happens to need:
  * automatic prefix caching   -> the 94% of the prompt that never changes is
                                  prefilled once and reused
  * continuous batching        -> removes the per-seed batching ceiling.
                                  `policy_multi._gen_chunk` takes ONE seed per
                                  chunk, so episodes with different seeds cannot
                                  share a forward pass; raising --micro to 32
                                  pushed the card to 100% util and left throughput
                                  at 1.40 ep/min, unchanged. vLLM carries a
                                  per-request seed, so the ceiling disappears.
  * paged KV                   -> far more concurrent sequences per card

WHAT THIS IS NOT. This is an INSTRUMENT SWAP inside a frozen gate. Different
kernels, a different sampler, different seed semantics. It must not be adopted on
the strength of being faster: `gate_caller/vllm_equivalence.py` is the gate, and
until it passes, `policy_multi` remains the instrument of record. G_surface's
committed episodes came from `transformers.generate()` and re-derive byte-for-byte
under it; that property is the baseline being compared against, which is why this
lives in a separate module and a separate env rather than editing policy_multi.

Public API mirrors policy_multi exactly, so `run_cube`'s module injection works
unchanged: family, load, render, count_tokens, tools_token_cost, generate_batch,
describe.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "gate_surface"))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

FAMILIES = {
    "qwen":    "Qwen/Qwen3-VL-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "granite": "ibm-granite/granite-3.3-8b-instruct",
    "falcon3": "tiiuae/Falcon3-7B-Instruct",
    # G-INVARIANCE matrix additions (2026-08-01). Weights live in the scratch workspace
    # a shared model cache -- set HF_HOME to reach them.
    "llama31":   "meta-llama/Llama-3.1-8B-Instruct",
    "ministral": "mistralai/Ministral-8B-Instruct-2410",
}
LABS = {"qwen": "Alibaba", "mistral": "Mistral AI", "granite": "IBM", "falcon3": "TII",
        "llama31": "Meta", "ministral": "Mistral AI"}

MEM_FRACTION = float(os.environ.get("CALLER_MEM_FRACTION", "0.40"))
MAX_MODEL_LEN = int(os.environ.get("CALLER_MAX_LEN", "32768"))

# C-TOOLTMPL probe, IMPORTED from policy_multi rather than transcribed.
#
# I first hand-copied a probe schema here and it silently differed from the real one,
# producing tool_block_cost=146 against policy_multi's 151 -- which I nearly read as a
# transformers-version template difference. It was my own constant. That is exactly the
# failure "identity by import, not by copy" exists to prevent, and the repo applies that
# rule everywhere else (run_cube imports run_gate; toolparse_probe imports the frozen
# parser). Importing makes the two engines' admitted costs comparable by construction
# instead of by my accuracy at retyping JSON.
from policy_multi import _PROBE_TOOL as _PROBE_TOOLS   # noqa: E402
_PROBE_MARKER = "zzq_probe_marker_tool"

_S = {}


def family() -> str:
    fam = os.environ.get("CALLER_FAMILY", "qwen")
    if fam not in FAMILIES:
        raise KeyError(f"CALLER_FAMILY={fam!r} unknown; known: {sorted(FAMILIES)}")
    return fam


def _check_tooltmpl(tok, fam: str) -> int:
    """C-TOOLTMPL, enforced here too. A template that silently ignores tools= costs
    0 tokens, never puts the tool name in the prompt, scores ~0 on every interface
    and manufactures exactly the caller heterogeneity B1 measures. Two of five
    some tested models failed this. The control must live in the loader, not in
    a script someone can forget to run -- and swapping the engine must not drop it.
    """
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    with_t = tok.apply_chat_template(msgs, tools=_PROBE_TOOLS, tokenize=False,
                                     add_generation_prompt=True)
    without = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    cost = len(tok(with_t)["input_ids"]) - len(tok(without)["input_ids"])
    if cost <= 0 or _PROBE_MARKER not in with_t:
        raise RuntimeError(
            f"C-TOOLTMPL FAIL for {fam}: tool block costs {cost} tokens and "
            f"name_in_prompt={_PROBE_MARKER in with_t}. Refusing to load.")
    print(f"[policy_vllm] C-TOOLTMPL PASS tool_block_cost={cost} tokens", flush=True)
    return cost


def load(gpu: str = None, mem_fraction: float = None):
    if _S:
        return _S
    fam = family()
    model_id = FAMILIES[fam]
    # Screening with a mid-training checkpoint: point at a merged model directory without
    # touching the family map. Used by the decay-aware re-screening experiment only.
    _override = os.environ.get("CALLER_MODEL_PATH")
    if _override:
        model_id = _override
    mem_fraction = MEM_FRACTION if mem_fraction is None else mem_fraction

    from transformers import AutoTokenizer
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cost = _check_tooltmpl(tok, fam)          # before any weight is loaded

    # The engine-core startup handshake can fail spuriously under host load (poller wakes
    # on a sentinel while the child is alive and mid-capture; observed on shared nodes).
    # A failed boot leaves _S empty, so without a retry every subsequent scenario re-runs
    # the boot into the same conditions. Retry with backoff here instead.
    llm = None
    for _try in range(5):
        try:
            llm = LLM(
                model=model_id,
                dtype="bfloat16",             # match policy_multi exactly
                gpu_memory_utilization=mem_fraction,
                max_model_len=MAX_MODEL_LEN,
                # Prefix caching is the reason this module exists (94% of the prompt is a
                # constant tool block) -- but a cached prefix and a freshly computed one can
                # take different kernel paths, and at temperature 0 that flips near-tie
                # tokens. C-GREEDY measured 2 of 39 turn-1 disagreements with it on. The
                # greedy arm must be able to switch it off; the sampled arm keeps it.
                enable_prefix_caching=(os.environ.get("CALLER_PREFIX_CACHE", "1") != "0"),
                trust_remote_code=False,
                disable_log_stats=True,
            )
            break
        except RuntimeError as e:
            if _try == 4:
                raise
            print(f"[policy_vllm] engine boot failed (try {_try + 1}/5): {e}; "
                  f"retrying in 30s", flush=True)
            import gc, time
            gc.collect()
            time.sleep(30)
    _S.update(llm=llm, tok=tok, fam=fam, model_id=model_id, tool_cost=cost)
    print(f"[policy_vllm] family={fam} {model_id} prefix_caching=ON "
          f"gpu_mem_util={mem_fraction}", flush=True)
    return _S


def render(messages, tools) -> str:
    tok = _S["tok"]
    return tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                   add_generation_prompt=True)


def count_tokens(text: str) -> int:
    return len(_S["tok"](text)["input_ids"])


def tools_token_cost(system: str, task: str, tools) -> int:
    """Identical in method to policy_multi's, so per-family schema costs stay
    comparable across engines. Measured at 35 ms/call = ~1% of an episode, so it is
    deliberately left uncached rather than optimised on a hunch."""
    tok = _S["tok"]
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    with_t = tok.apply_chat_template(msgs, tools=tools, tokenize=False,
                                     add_generation_prompt=True)
    without = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return count_tokens(with_t) - count_tokens(without)


def generate_batch(prompts, seeds, max_new_tokens=256, temperature=0.7, top_p=0.8,
                   micro=4):
    """`(texts, n_pad_max, gen_counts)` -- the exact contract run_gate unpacks.

    `micro` is accepted for signature compatibility and deliberately IGNORED: vLLM
    batches continuously, and the per-seed chunking that `micro` exists to control is
    the very constraint this module removes.

    SEEDING DIFFERENCE, recorded rather than hidden. `policy_multi` seeds each chunk
    with `sum(seeds) + i` -- one batch-level seed shared by every episode in the
    chunk. Here each request carries its own seed. Neither is more "correct" and the
    two engines could not produce identical tokens regardless, since they seed
    different RNGs; but per-request seeding means an episode's sample no longer
    depends on which other episodes happened to be batched beside it, which is a
    property the frozen instrument does not have. C-ENGINE tests the distributions,
    not the tokens, precisely because token identity was never on the table.
    """
    from vllm import SamplingParams
    llm, tok = _S["llm"], _S["tok"]
    prompts = list(prompts)

    # ---- C-CTXFIT (GATE_CALLER_FROZEN sec.5b) -------------------------------
    # A prompt over the shared budget must terminate ITS OWN episode, not the cell.
    # Submitting it raises VLLMValidationError for the whole batch: that is how
    # `mistral x screened` died after 3 of 1,248 episodes. Over-length prompts are
    # withheld from the batch and returned empty, which run_gate treats as a turn
    # with no parseable call; the episode ends and the rest of the cell continues.
    budget = MAX_MODEL_LEN - max_new_tokens
    lens = [len(tok(p)["input_ids"]) for p in prompts]
    fits = [i for i, n in enumerate(lens) if n <= budget]
    over = [i for i, n in enumerate(lens) if n > budget]
    if over:
        _S["ctx_overflow"] = _S.get("ctx_overflow", 0) + len(over)
        print(f"[policy_vllm] C-CTXFIT: {len(over)} prompt(s) over the shared "
              f"{MAX_MODEL_LEN}-token budget (max {max(lens)}); episode(s) terminated, "
              f"cell continues. Running total {_S['ctx_overflow']}.", flush=True)

    texts = [""] * len(prompts)
    gens = [0] * len(prompts)
    if fits:
        params = [SamplingParams(temperature=temperature, top_p=top_p, top_k=20,
                                 max_tokens=max_new_tokens, seed=int(seeds[i]))
                  for i in fits]
        outs = llm.generate([prompts[i] for i in fits], params, use_tqdm=False)
        # vLLM may complete out of order; request_id encodes submission index.
        ordered = sorted(outs, key=lambda o: int(o.request_id))
        for slot, o in zip(fits, ordered):
            texts[slot] = o.outputs[0].text
            gens[slot] = len(o.outputs[0].token_ids)
    return texts, (max(lens) if lens else 0), gens


def ctx_overflow_count() -> int:
    """Cells withheld for exceeding the shared context budget. Read into the receipt
    so an inadmissible cell is reported, never silently scored 0."""
    return _S.get("ctx_overflow", 0)


def describe() -> dict:
    return {"engine": "vllm", "family": _S.get("fam"), "model_id": _S.get("model_id"),
            "tool_block_cost": _S.get("tool_cost"), "prefix_caching": True,
            "max_model_len": MAX_MODEL_LEN}
