"""Merge a verl FSDP LoRA checkpoint into a standalone HF model.

WHY THIS EXISTS. `verl.model_merger merge --backend fsdp` writes out the 750 base tensors of the
actor shard and silently DROPS the 736 LoRA tensors beside them. The result is bit-identical to
the base model: comparing merged_atscfix_step5 against Qwen3-VL-8B-Instruct gave maxdiff 0.000e+00
on every tensor sampled from layers 0 and 10, including q_proj, v_proj and up_proj.

Every held-out evaluation run before 2026-08-09 therefore compared the base model against itself.
That is why cells read exactly +0.000 with b=3, c=3 out of 436 -- not a weak effect, just the
residual noise of scoring one model twice -- and why nothing downstream ever moved: availability
3.7%->93%, batch 8->32, pools 320->1127, lr 1e-5->1e-4. None of them could show an effect, because
the artifact was after training. The trainer telemetry was healthy throughout (ppo_kl 5e-4..8e-3,
advantages +/-1.79, grad_norm 0.008..0.040): the policy was training, we were never evaluating it.

The checkpoint stores PEFT-style keys:
    base_model.model.<path>.lora_A.default.weight   [r, in]
    base_model.model.<path>.lora_B.default.weight   [out, r]
and the scale comes from lora_train_meta.json (alpha / r). The merged weight is W + (alpha/r) B A.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file


def strip(k: str) -> str:
    """PEFT key -> plain HF key."""
    return k.replace("base_model.model.", "", 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="global_step_N/actor directory")
    ap.add_argument("--base", required=True, help="base HF model snapshot dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    meta_p = os.path.join(a.ckpt, "lora_train_meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    r = float(meta.get("r", 64))
    alpha = float(meta.get("lora_alpha", r))
    scale = alpha / r
    print(f"[merge] r={r:g} alpha={alpha:g} scale={scale:g}", flush=True)

    shard = sorted(glob.glob(os.path.join(a.ckpt, "model_world_size_*_rank_*.pt")))
    if not shard:
        print("[merge] no actor shard found")
        return 1
    # MULTI-RANK CHECKPOINTS. A 1-GPU arm writes model_world_size_1_rank_0.pt holding plain
    # tensors, and `update` over that one file is correct. A 2-GPU arm (every train2 slot: a8C,
    # a8A3, b8van) writes rank_0 AND rank_1 holding DTENSOR SHARDS, and `update` silently let
    # rank_1 overwrite rank_0 -- half the weights, and any later op on a DTensor raised
    # "Could not resolve the process group registered under the name 0" because a single merge
    # process has no process group. That killed both eval cards the moment they were handed an
    # a8A3 cell, and it would have made the METHOD ARM unevaluable: a8C is a 2-GPU arm too.
    #
    # Reassemble instead of overwrite. .placements and ._local_tensor are plain attribute reads
    # that need no collective, so the full tensor can be rebuilt in one process: concatenate the
    # per-rank local shards along the sharded dimension, and take a single copy of anything
    # replicated. Shapes are checked against the base weights below, which catches a wrong
    # concat axis loudly rather than producing a subtly corrupt model.
    def local_of(t):
        return getattr(t, "_local_tensor", t)

    def shard_dim(t):
        for pl in getattr(t, "placements", ()) or ():
            d = getattr(pl, "dim", None)
            if d is not None:
                return int(d)
        return None            # replicated, or a plain tensor

    per_rank = []
    for s in shard:
        # mmap: the shard is ~16G and a plain load OOM-killed the login node (exit 137)
        per_rank.append(torch.load(s, map_location="cpu", weights_only=False, mmap=True))
    sd = {}
    if len(per_rank) == 1:
        sd = {k: local_of(v) for k, v in per_rank[0].items()}
    else:
        rebuilt = replicated = 0
        for k in per_rank[0]:
            vals = [r[k] for r in per_rank if k in r]
            d = shard_dim(vals[0])
            if d is None or len(vals) == 1:
                sd[k] = local_of(vals[0]); replicated += 1
            else:
                sd[k] = torch.cat([local_of(v) for v in vals], dim=d); rebuilt += 1
        print(f"[merge] {len(per_rank)} rank shards -> reassembled {rebuilt} sharded "
              f"tensors, {replicated} replicated", flush=True)
    lora = {k: v for k, v in sd.items() if "lora_A" in k or "lora_B" in k}
    print(f"[merge] shard entries={len(sd)} lora tensors={len(lora)}", flush=True)
    if not lora:
        print("[merge] refusing: no LoRA tensors, this would just copy the base model")
        return 1

    # pair A and B by their shared prefix
    pairs = {}
    for k in lora:
        stem = k.split(".lora_")[0]
        pairs.setdefault(stem, {})["A" if ".lora_A" in k else "B"] = sd[k]
    pairs = {k: v for k, v in pairs.items() if "A" in v and "B" in v}
    print(f"[merge] adapter pairs={len(pairs)}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    applied = missing = 0
    for f in sorted(glob.glob(os.path.join(a.base, "*.safetensors"))):
        tens = load_file(f)
        for stem, ab in pairs.items():
            key = strip(stem) + ".weight"
            if key not in tens:
                continue
            A = ab["A"].to(torch.float32)          # [r, in]
            B = ab["B"].to(torch.float32)          # [out, r]
            W = tens[key].to(torch.float32)
            delta = (B @ A) * scale
            if delta.shape != W.shape:
                missing += 1
                continue
            tens[key] = (W + delta).to(tens[key].dtype)
            applied += 1
        save_file(tens, os.path.join(a.out, os.path.basename(f)), metadata={"format": "pt"})
    # config/tokenizer come from the checkpoint's own huggingface dir when present, else base
    src = os.path.join(a.ckpt, "huggingface")
    src = src if os.path.isdir(src) and os.path.exists(os.path.join(src, "config.json")) else a.base
    for fn in os.listdir(src):
        if fn.endswith(".safetensors") or fn.endswith(".bin"):
            continue
        s = os.path.join(src, fn)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(a.out, fn))
    print(f"[merge] applied={applied} shape_mismatch={missing} -> {a.out}", flush=True)
    return 0 if applied else 1


if __name__ == "__main__":
    sys.exit(main())
