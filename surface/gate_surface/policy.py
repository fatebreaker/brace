"""Frozen policy for gate G_surface: Qwen3-VL-8B-Instruct, batched, text-only.

THE POLICY IS FROZEN. No training, no adapters, no bilevel loop. The only thing
that varies across conditions is the tool surface put in front of it.

Shared-box discipline: pinned to one whitelisted GPU (3/4/5, NEVER 2) and capped
with torch.cuda.set_per_process_memory_fraction.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_GPU = "3"
MEM_FRACTION = 0.30

_S = {}


def load(gpu: str = None, mem_fraction: float = MEM_FRACTION):
    if _S:
        return _S
    gpu = str(gpu or os.environ.get("GATE_GPU", DEFAULT_GPU))
    assert gpu in ("3", "4", "5"), f"GPU {gpu} is not on the whitelist (3/4/5, never 2)"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import torch
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    torch.cuda.set_per_process_memory_fraction(mem_fraction, 0)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    model.eval()
    model.generation_config.pad_token_id = tok.pad_token_id
    _S.update(model=model, tok=tok, torch=torch, gpu=gpu)
    print(f"[policy] {MODEL_ID} on physical GPU {gpu}, "
          f"{torch.cuda.memory_allocated(0)/2**30:.1f} GiB weights, "
          f"mem_fraction={mem_fraction}", flush=True)
    return _S


def render(messages, tools) -> str:
    tok = _S["tok"]
    return tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                   add_generation_prompt=True)


def count_tokens(text: str) -> int:
    return len(_S["tok"](text)["input_ids"])


def tools_token_cost(system: str, task: str, tools) -> int:
    """Schema cost measured through the REAL tokenizer: tokens added to the
    prompt by the tool block (rendered prompt with tools minus without)."""
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
    """Decoding step for a batch of episodes.

    The GPU is shared: another user holds ~110 GB per card. Rather than taking a
    bigger memory fraction, the batch is split into micro-batches of at most
    `micro` sequences, and on CUDA OOM the micro-batch is halved and retried.
    Sequences are grouped by length so padding waste stays small. Order of the
    returned list matches the order of `prompts`.
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
                print(f"[policy] CUDA OOM -> retrying micro-batch at size {size}",
                      flush=True)
        for k, j in enumerate(chunk):
            texts[j], gens[j] = t[k], g[k]
        n_pad_max = max(n_pad_max, n_in)
        i += len(chunk)
    return texts, n_pad_max, gens
