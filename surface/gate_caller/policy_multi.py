"""Multi-family policy loader for G-CALLER. Drop-in replacement for `gate_surface.policy`.

Exposes the exact surface `run_gate.main()` uses — `load`, `render`, `count_tokens`,
`tools_token_cost`, `generate_batch` — so `run_gate.py` runs unmodified against any
of the four admitted callers. Which caller is selected by `CALLER_FAMILY`.

Residency
---------
One model per process, and one process at a time. The previous box had 3x H200 with
~108 GB free per card and could co-locate; this box has ONE permitted card (GPU 3,
A100 80 GB) with a co-tenant holding ~22 GB, so 4 x ~16 GB of bf16 weights cannot be
resident together. Running one family per process and exiting between families frees
the allocator completely, which is stronger than an in-process `unload()` (no
fragmentation carried across families) and mirrors the detached-process pattern the
gate already uses.

C-TOOLTMPL is enforced here, at load, not merely upstream
--------------------------------------------------------
Two of six candidates on the previous box had chat templates that silently ignore
`tools=`; such a model sees no tools, scores ~0 everywhere, and manufactures exactly
the caller heterogeneity B1 measures. `load()` therefore re-measures the tool-block
cost of a real probe schema and REFUSES to return a model whose cost is 0 or whose
tool name never reaches the prompt. A control that only runs in a separate script is
a control someone can forget to run.

Shared-box discipline
---------------------
Pinned to one whitelisted GPU (3/4/5, NEVER 2) and capped with
`torch.cuda.set_per_process_memory_fraction`, same as `gate_surface/policy.py`.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", os.path.join(
    os.environ.get("BRACE_ROOT", os.path.expanduser("~")), "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

FAMILIES = {
    "qwen":    "Qwen/Qwen3-VL-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "granite": "ibm-granite/granite-3.3-8b-instruct",
    "falcon3": "tiiuae/Falcon3-7B-Instruct",
}

DEFAULT_GPU = "3"
# 0.30 of a 141 GB H200 was ~42 GB. 0.30 of this 80 GB A100 is 24 GB, which fits one
# 7-8B bf16 model (~16 GB) plus activations at micro=4 but leaves little slack, so the
# default is raised. 0.40 x 80 GB = 32 GB, and 22 (co-tenant) + 32 = 54 GB of 80 GB,
# so the co-tenant keeps its allocation and ~26 GB stays free on the card.
MEM_FRACTION = float(os.environ.get("CALLER_MEM_FRACTION", "0.40"))

# §8.2 admits "published open-weight 4-8B models from different families". Recorded
# and checked, so an out-of-band swap cannot happen quietly.
BAND_MIN_B, BAND_MAX_B = 3.5, 9.0

_PROBE_TOOL = [{
    "type": "function",
    "function": {
        "name": "zzq_probe_marker_tool",
        "description": "Probe tool for the C-TOOLTMPL admissibility control.",
        "parameters": {
            "type": "object",
            "properties": {"zzq_probe_arg": {"type": "string"}},
            "required": ["zzq_probe_arg"],
        },
    },
}]

_S = {}


def family() -> str:
    fam = os.environ.get("CALLER_FAMILY", "qwen")
    if fam not in FAMILIES:
        raise KeyError(f"CALLER_FAMILY={fam!r} unknown; known: {sorted(FAMILIES)}")
    return fam


def _model_class(fam: str, torch):
    if fam == "qwen":
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM


def _check_tooltmpl(tok, fam: str) -> int:
    """C-TOOLTMPL, inline. Returns measured tool-block cost; raises if inadmissible."""
    msgs = [{"role": "system", "content": "You are an agent with access to tools."},
            {"role": "user", "content": "Do the thing."}]
    with_t = tok.apply_chat_template(msgs, tools=_PROBE_TOOL, tokenize=False,
                                     add_generation_prompt=True)
    without = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    cost = len(tok(with_t)["input_ids"]) - len(tok(without)["input_ids"])
    name_present = "zzq_probe_marker_tool" in with_t
    if cost <= 0 or not name_present:
        raise RuntimeError(
            f"C-TOOLTMPL FAILED for {fam} ({FAMILIES[fam]}): tool-block cost={cost}, "
            f"tool name in prompt={name_present}. This model's chat template ignores "
            f"tools=; it would score ~0 on every interface and fake caller "
            f"heterogeneity. Refusing to load it."
        )
    return cost


def load(gpu: str = None, mem_fraction: float = None):
    if _S:
        return _S
    fam = family()
    model_id = FAMILIES[fam]
    gpu = str(gpu or os.environ.get("GATE_GPU", DEFAULT_GPU))
    # The reference host was a bare shared 8-GPU box where picking the wrong index meant landing on
    # another user's card, so the index whitelist WAS the safety mechanism. Under Slurm
    # it is neither correct nor needed: the cgroup already restricts us to the allocated
    # device, which is always visible as index 0. Enforcing 3/4/5 there refuses to run on
    # a card we exclusively hold.
    #
    # Keep the guard exactly where it still guards something. GATE_GPU_WHITELIST may be
    # set to a comma list to restore box-A behaviour on any future bare host; on Slurm it
    # is left unset and the scheduler is the mechanism.
    _wl = os.environ.get("GATE_GPU_WHITELIST")
    if _wl:
        allowed = [s.strip() for s in _wl.split(",") if s.strip()]
        assert gpu in allowed, f"GPU {gpu} is not on GATE_GPU_WHITELIST={allowed}"
    elif not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "Refusing to pick a GPU on a bare host with no GATE_GPU_WHITELIST set. "
            "Either run under Slurm (which isolates the allocation) or set "
            "GATE_GPU_WHITELIST to the cards you are permitted to touch.")
    mem_fraction = MEM_FRACTION if mem_fraction is None else mem_fraction

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import torch
    from transformers import AutoTokenizer

    torch.cuda.set_per_process_memory_fraction(mem_fraction, 0)

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        # Several non-Qwen families ship no pad token. Falling back to eos is standard
        # and safe here because generation is left-padded and pads are masked out.
        tok.pad_token = tok.eos_token

    cost = _check_tooltmpl(tok, fam)   # raises before any weight touches the GPU

    cls = _model_class(fam, torch)
    model = cls.from_pretrained(model_id, dtype=torch.bfloat16,
                                device_map={"": 0}, attn_implementation="sdpa")
    model.eval()
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tok.pad_token_id

    n_params = sum(p.numel() for p in model.parameters())
    n_b = n_params / 1e9
    band = BAND_MIN_B <= n_b <= BAND_MAX_B

    _S.update(model=model, tok=tok, torch=torch, gpu=gpu, family=fam,
              model_id=model_id, tool_block_cost=cost, n_params=n_params,
              in_band=band, mem_fraction=mem_fraction)
    print(f"[policy_multi] family={fam} {model_id}", flush=True)
    print(f"[policy_multi] physical GPU {gpu}, "
          f"{torch.cuda.memory_allocated(0)/2**30:.1f} GiB weights, "
          f"mem_fraction={mem_fraction}", flush=True)
    print(f"[policy_multi] C-TOOLTMPL PASS tool_block_cost={cost} tokens", flush=True)
    print(f"[policy_multi] params={n_b:.2f}B in_band[{BAND_MIN_B},{BAND_MAX_B}]={band}",
          flush=True)
    if not band:
        print(f"[policy_multi] WARNING {n_b:.2f}B is outside the §8.2 4-8B band; "
              f"recorded, not silently accepted.", flush=True)
    return _S


def render(messages, tools) -> str:
    tok = _S["tok"]
    return tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                   add_generation_prompt=True)


def count_tokens(text: str) -> int:
    return len(_S["tok"](text)["input_ids"])


def tools_token_cost(system: str, task: str, tools) -> int:
    """Schema cost through the REAL tokenizer, identical in method to
    `gate_surface.policy.tools_token_cost` so per-family costs stay comparable."""
    tok = _S["tok"]
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    with_t = tok.apply_chat_template(msgs, tools=tools, tokenize=False,
                                     add_generation_prompt=True)
    without = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return count_tokens(with_t) - count_tokens(without)


def _gen_chunk(prompts, seed, max_new_tokens, temperature, top_p):
    S = load()
    tok, model, torch = S["tok"], S["model"], S["torch"]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)
    torch.manual_seed(int(seed) & 0x7FFFFFFF)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=temperature > 0, temperature=temperature,
                             top_p=top_p, top_k=20)
    n_in = enc["input_ids"].shape[1]
    texts, n_gen = [], []
    for row in out:
        new = row[n_in:]
        keep = new[new != tok.pad_token_id]
        texts.append(tok.decode(new, skip_special_tokens=True))
        n_gen.append(int(keep.numel()))
    del enc, out
    return texts, n_gen, int(n_in)


def generate_batch(prompts, seeds, max_new_tokens=256, temperature=0.7, top_p=0.8,
                   micro=4):
    """Identical batching/OOM policy to `gate_surface.policy.generate_batch`.

    Kept byte-for-byte equivalent in behaviour so a per-family difference in outcome
    cannot be an artifact of a different decoding path.
    """
    S = load()
    torch = S["torch"]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    texts = [None] * len(prompts)
    gens = [0] * len(prompts)
    n_pad_max = 0
    i = 0
    while i < len(order):
        chunk = order[i:i + micro]
        size = len(chunk)
        while True:
            try:
                t, g, n_in = _gen_chunk([prompts[j] for j in chunk],
                                        sum(seeds) + i, max_new_tokens,
                                        temperature, top_p)
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if size == 1:
                    raise
                size = max(1, size // 2)
                chunk = chunk[:size]
                print(f"[policy_multi] CUDA OOM -> retrying micro-batch at size {size}",
                      flush=True)
        for k, j in enumerate(chunk):
            texts[j], gens[j] = t[k], g[k]
        n_pad_max = max(n_pad_max, n_in)
        i += len(chunk)
    return texts, n_pad_max, gens


def describe() -> dict:
    """Provenance for the receipt: which caller, admitted how, at what size."""
    S = load()
    return {k: S[k] for k in ("family", "model_id", "tool_block_cost", "n_params",
                              "in_band", "gpu", "mem_fraction")}
