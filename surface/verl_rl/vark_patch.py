"""TRIAGE v4 — variable-k group sizes, applied to verl by monkeypatch from OUR repo.

WHY A MONKEYPATCH AND NOT A VENDORED FILE ON PYTHONPATH. `verl.trainer.ppo.ray_trainer` resolves
from the first `verl` package found on the path; a lone vendored copy of one module inside that
package is never imported, so PYTHONPATH cannot shadow it. This module instead imports the
installed classes and replaces three bound methods plus one DataProto method, in-process, and only
when VARK=1.

DEFAULT OFF, STRUCTURALLY. coadapt.py imports this module only when the flag is set, so with the
flag absent the module is never loaded and there is nothing to inspect: flag-off is a no-op by
construction rather than by review.

WHAT THE MECHANISM ACTUALLY NEEDS (the scope was cut down twice by reading the installed code):

  verl already assigns one fresh uid PER PARQUET ROW and GRPO groups on exactly that column
  (compute_advantage -> compute_grpo_outcome_advantage(index=uid)). One parquet row is one task,
  and the prompt-side repeat copies that row's uid onto each replica, so group identity is ALREADY
  per-task. The one thing forcing a single group size is that the repeat factor is the scalar
  rollout.n applied uniformly (DataProto.repeat(repeat_times: int)).

  So the uid assignment is NOT patched. (It blocks only the "emit k_i duplicate parquet rows"
  design, where duplicates get k_i DIFFERENT uids and yield k_i groups of size n instead of one
  group of size k_i. Under per-row repeat the unconditional per-row uuid is exactly right.)

PATCHED (4):
  1. DataProto.repeat        -> per-row repeat when the proto carries the k column.
  2. RayPPOTrainer._get_gen_batch -> keep the k column on BOTH sides of the split. Upstream POPS
     every non-tensor key except {data_source, reward_model, extra_info, uid} out of `batch` and
     into `gen_batch`, so without this the generation-side repeat would be per-row while the
     batch-side repeat (which must mirror it exactly, because `batch.union(gen_batch_output)`
     joins them POSITIONALLY) would still be uniform. That misalignment would pair every rollout
     with the wrong prompt row's uid, silently.
  3. RayPPOTrainer.fit       -> a thin WRAPPER (guards + banner), not a copy. Copying a ~400-line
     fit() that already carries two project patches would make this repo own a chunk of verl and
     invite transcription error on every env change.
  4. (assertion only) the per-step trajectory total, checked inside the repeat.

NOT PATCHED, EACH PROVEN INERT OR GUARDED RATHER THAN ASSUMED:
  * compute_advantage(num_repeat=...): compute_grpo_outcome_advantage's signature is
    (token_level_rewards, response_mask, index, epsilon, norm_adv_by_std_in_grpo, config) -- it
    takes no num_repeat, so the argument is dead on the GRPO path these arms run.
  * critic ppo_mini_batch_size (* rollout.n): unreachable with adv_estimator=grpo (use_critic
    False). GUARDED: fit() refuses if a critic is live.
  * DAPO filter_groups traj_bsz (prompt_bsz * rollout.n): reachable only when
    algorithm.filter_groups.enable is set. GUARDED: fit() refuses.
  * actor ppo_mini_batch_size (* rollout.n): with SUM_K_PER_STEP pinned to train_batch_size *
    NROLL (160 = 32*5), the trajectory count per step is unchanged, so MINIBSZ*NROLL = 40 remains
    exactly the right trajectory-unit minibatch and divides 160 into 4. This is ASSERTED at every
    step, not assumed -- if the pin is ever violated the run aborts rather than silently training
    on a partial minibatch.

VERSION PIN. The reasoning above is tied to the exact installed ray_trainer.py. apply() refuses
unless that file's md5 matches PINNED_MD5, so an env upgrade cannot silently mis-patch.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

VARK_COL = "vark_k"
PINNED_MD5 = "1d0deed455be8a9e791b304fc319dd92"   # verl ray_trainer.py as vendored-against
K_CHOICES = (2, 3, 5, 8)
_APPLIED = False


def vark_enabled() -> bool:
    """True only when the launcher composed VARK=1."""
    return os.environ.get("VARK", "") == "1"


def _sum_k_per_step() -> int:
    """Pinned trajectory budget per optimizer step (BSZ * NROLL), the invariant that keeps the
    minibatch arithmetic exact without patching _update_actor."""
    return int(os.environ.get("BSZ", "32")) * int(os.environ.get("NROLL", "5"))


def counts_from(proto, expected_rows: int) -> np.ndarray:
    """Per-row k_i off the proto. Raises rather than falling back to uniform k -- a silent
    fallback is the one failure this whole mechanism exists to prevent, since it would yield a run
    whose composed command reads variable-k and whose gradients are not."""
    src = None
    if getattr(proto, "non_tensor_batch", None) and VARK_COL in proto.non_tensor_batch:
        src = np.asarray(proto.non_tensor_batch[VARK_COL])
    if src is None:
        raise ValueError(f"VARK=1 but the proto carries no '{VARK_COL}' column")
    k = np.asarray(src, dtype=np.int64).reshape(-1)
    if k.shape[0] != expected_rows:
        raise ValueError(f"'{VARK_COL}' has {k.shape[0]} entries for {expected_rows} rows")
    bad = sorted(set(int(x) for x in k) - set(K_CHOICES))
    if bad:
        # k=1 in particular is not a small group but a degenerate one: one rollout has zero
        # within-group variance, so its GRPO advantage is identically zero and the budget spent
        # on it cannot carry gradient by construction.
        raise ValueError(f"k_i values {bad} outside the registered choice set {K_CHOICES}")
    return k


def index_from(k: np.ndarray) -> np.ndarray:
    """Row indices reproducing repeat(interleave=True) semantics with per-row counts:
    repeat(n) yields [row0]*n + [row1]*n + ...; per-row it is row i repeated k_i times,
    contiguous, in row order."""
    return np.repeat(np.arange(k.shape[0], dtype=np.int64), k)


def apply() -> bool:
    """Install the patches. Idempotent. Returns False (and patches nothing) when VARK is off."""
    global _APPLIED
    if not vark_enabled():
        return False
    if _APPLIED:
        return True

    from verl.protocol import DataProto
    from verl.trainer.ppo import ray_trainer as rt

    # --- VERSION PIN ---------------------------------------------------------------------
    src = rt.__file__
    with open(src, "rb") as fh:
        got = hashlib.md5(fh.read()).hexdigest()
    if got != PINNED_MD5:
        raise RuntimeError(
            f"VARK refuses to patch: {src} md5 {got} != pinned {PINNED_MD5}. The patch scope was "
            f"derived by reading that exact file (uid semantics, _get_gen_batch's pop list, the "
            f"unreachability of the critic/DAPO/num_repeat sites). Re-verify before repinning."
        )

    # --- 1. per-row repeat ---------------------------------------------------------------
    _orig_repeat = DataProto.repeat

    def _vark_repeat(self, repeat_times=2, interleave=True):
        ntb = getattr(self, "non_tensor_batch", None)
        if not ntb or VARK_COL not in ntb:
            return _orig_repeat(self, repeat_times=repeat_times, interleave=interleave)
        if not interleave:
            raise ValueError("VARK per-row repeat is defined only for interleave=True")
        n_rows = len(self)
        k = counts_from(self, n_rows)
        total = int(k.sum())
        want = _sum_k_per_step()
        if total != want:
            # The pin is what lets the actor minibatch arithmetic stay untouched. If it is ever
            # violated, MINIBSZ*NROLL stops dividing the trajectory count and the step would train
            # on a partial minibatch. Abort instead.
            raise ValueError(
                f"VARK budget pin violated: sum(k_i)={total} for {n_rows} rows, expected {want} "
                f"(BSZ*NROLL). The allocator must pack each step's rows to exactly {want}."
            )
        hist = {int(c): int((k == c).sum()) for c in K_CHOICES if (k == c).any()}
        print(f"[vark] repeat: {n_rows} rows -> {total} rollouts; k histogram {hist}", flush=True)
        return self[index_from(k)]

    DataProto.repeat = _vark_repeat

    # --- 2. keep the k column on BOTH sides of the gen/batch split -------------------------
    _orig_get_gen = rt.RayPPOTrainer._get_gen_batch

    def _vark_get_gen_batch(self, batch):
        keep = VARK_COL in batch.non_tensor_batch
        saved = np.asarray(batch.non_tensor_batch[VARK_COL]).copy() if keep else None
        gen_batch = _orig_get_gen(self, batch)
        if keep:
            # upstream POPPED it out of `batch`; put it back so the batch-side repeat is per-row
            # too and mirrors the generation-side repeat exactly.
            batch.non_tensor_batch[VARK_COL] = saved
        return gen_batch

    rt.RayPPOTrainer._get_gen_batch = _vark_get_gen_batch

    # --- 3. guards, as a thin wrapper around fit() ----------------------------------------
    _orig_fit = rt.RayPPOTrainer.fit

    def _vark_fit(self):
        try:
            fg = self.config.algorithm.get("filter_groups", None)
        except Exception:
            fg = None
        if fg is not None and fg.get("enable", False):
            raise ValueError(
                "VARK=1 is not composable with algorithm.filter_groups: the DAPO path sizes its "
                "buffer as prompt_bsz * rollout.n, which is not sum(k_i) under variable k."
            )
        if getattr(self, "use_critic", False):
            raise ValueError(
                "VARK=1 assumes the critic is disabled (adv_estimator=grpo): the critic's "
                "ppo_mini_batch_size scales by rollout.n, which is not sum(k_i) under variable k."
            )
        print(
            f"[vark] ENABLED: per-row k from '{VARK_COL}', choices {K_CHOICES}, "
            f"sum(k_i)/step pinned to {_sum_k_per_step()}",
            flush=True,
        )
        return _orig_fit(self)

    rt.RayPPOTrainer.fit = _vark_fit

    _APPLIED = True
    return True
