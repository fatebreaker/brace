#!/bin/bash
# Queue learning-curve points for the depth arms.
#
# WHY A CURVE. The held-out policy effect measured ~0 for every arm (best +0.010, p=0.146) while
# gradient availability rose 7.5% -> 20%. Those arms had 5-72 GRPO steps. A single endpoint cannot
# distinguish "the method does not work" from "the policy was never trained far enough to show it",
# and those two conclusions call for opposite next moves. A curve over steps separates them: a
# rising effect says keep training, a flat one at 496 steps says the mechanism does not convert.
#
# Every point uses the FIXED initial surface, so the curve is pure policy. Its step-0 anchor is
# the existing base-policy cell A -- advertised_init.txt is byte-identical across arms (all copied
# from coadapt's), so the base measurement is arm-independent and does not need redoing per arm.
set -u
R=${BRACE_ROOT:?BRACE_ROOT is unset; source env.sh first}
Q=$R/work/eval_queue.txt
STRIDE=${STRIDE:-5}        # one point per ~5 steps. At 15 no batch-32 checkpoint qualified until step 15,
                           # so three A100s idled 2h with an empty queue while arms sat at step 10.
                           # for the same point while atscfix/dapo/atscL2 checkpoints went unmeasured.
# The *_bs4 arms are the ARCHIVED batch-8 runs. Their curve is the negative control for the
# paper: it shows a gradient-starved configuration (0.6-1.6 usable GRPO groups per step) does not
# improve with training steps, which is what makes the batch-32 comparison meaningful rather than
# a bare assertion. They are also the only checkpoints that exist until batch-32 arms save.
# LIVE arms only. The *_bs4 (batch-8), atscL2/atscfixL2 (frozen lr=1e-5) and atsc_* runs are
# retired: their policies either never moved or were measured through the broken merge.
# EVERY LIVE ARM. This list was written before the ELSA family existed and was never updated,
# so a8A3 (the method), its three ablations, and b8k16 had NEVER had a checkpoint
# evaluated -- 45 steps of the method arm with zero held-out measurements. Training reward
# cannot answer whether the method beats the baselines; transfer is the claim.
# b8k16v is the validation-curve arm. It has to be in this list or its checkpoints are never
# queued -- the arm that supplies the ordering key would be the one arm the ordering never reaches.
ARMS=${ARMS:-"q4bF q4bT q2bT q2bD q2bP q2bR q2bN q4bD q4bP q4bR a8T6d a8T5kr a8T5 a8T5k a8T3g2 a8T3gr a8Fr a8T3 a8T3g a8T4 a8Tvip a8Tvipr a8Tlp a8Tpe a8T2n a8T2nr a8T2 a8T2r a8T2w a8T2g1 a8T2b a8T2s a8T a8Te0 a8Tg1 a8C a8Cu a8A3 a8Az0 a8Auni a8Ae0 a8A a8F b8plr b8van b8ret b8m25 dapo atscfix"}

# PRUNE STALE entries first: a STEP<n>@<arm> whose checkpoint was deleted (arm relaunched,
# retention rolled it off) makes the eval card fail three engine boots and exit 7. That cost
# a100b seven strikes on 2026-08-08. Cheap to check, expensive to miss.
if [ -s "$Q" ]; then
    tmpq=$(mktemp)
    while IFS= read -r line; do
        case "$line" in
            STEP*@*)
                st=${line%@*}; st=${st#STEP}; arm=${line#*@}
                [ -f "$R/work/verl/ckpt_$arm/global_step_$st/actor/huggingface/config.json" ] || continue ;;
        esac
        printf '%s\n' "$line"
    done < "$Q" > "$tmpq"
    mv "$tmpq" "$Q"
fi

# VAL-GATED EVAL (2026-08-11): the validation curve selects, cells certify. Only arms in the
# gate get held-out cells at all -- the finalists and required baselines. Completed rows
# (a8F, a8A, b8plr, b8ret, a8Cu, a8A3, ...) stay measured; retired/non-contender arms stop
# consuming cards. Override with EVAL_GATE="..." if the finalist set changes.
EVAL_GATE=${EVAL_GATE:-"q4bF q4bT q2bT q2bD q2bP q2bR q2bN q4bD q4bP q4bR a8T6d a8T5kr a8T5 a8T5k a8T3g2 a8T3gr a8Fr a8T3 a8T3g a8Tvip a8Tvipr a8Tlp a8Tpe a8T2n a8T2nr a8T2 a8T2r a8T2w a8T2g1 a8T2g1r a8T2b a8T a8Te0 dapo a8T3 a8T3g a8T2n a8T4"}
queued=0
for t in $ARMS; do
    case " $EVAL_GATE " in *" $t "*) ;; *) continue;; esac
    d=$R/work/verl/ckpt_$t
    [ -d "$d" ] || continue
    last=0
    for s in $(ls -d $d/global_step_* 2>/dev/null | sed 's/.*_//' | sort -n); do
        # keep points roughly STRIDE apart so the curve is readable without merging every step
        [ $((s - last)) -ge $STRIDE ] || continue
        # Skip INCOMPLETE checkpoints. A restart that lands mid-save leaves global_step_N/actor/
        # with an empty huggingface/ and no shards; step 55 of atsc was exactly that, and the
        # merge died on it with "Can't load the configuration", burning an eval card per attempt.
        # Require both the HF config and at least one model shard before queueing the point.
        [ -f "$d/global_step_$s/actor/huggingface/config.json" ] || continue
        ls "$d/global_step_$s/actor/"model_world_size_*.pt >/dev/null 2>&1 || continue
        cell=$R/work/coadapt_eval_$t/cell_STEP$s.jsonl
        # Count DISTINCT PARSABLE tasks, not lines. Two eval cards briefly shared STEP20@a8A on
        # 2026-08-09 and appended to one file concurrently: 410 lines held 162 distinct tasks,
        # 146 of them repeated up to 4x, plus 40 torn records from interleaved writes. By line
        # count that cell reads as measured; by task count it is a third of a cell. The paired
        # test compares arms on the SAME task set, so a cell that is full of repeats rather than
        # coverage does not just add noise, it silently changes what is being compared.
        n=0
        [ -f "$cell" ] && n=$($BRACE_ENVS/mcp_verl/bin/python -c "
import json,sys
seen=set()
for l in open(sys.argv[1],errors='ignore'):
    try: r=json.loads(l)
    except Exception: continue
    seen.add((r.get('scenario'),r.get('task_idx')))
print(len(seen))" "$cell" 2>/dev/null || echo 0)
        [ "$n" -ge 200 ] && { last=$s; continue; }          # already measured
        # Skip points already IN FLIGHT. This runs every supervisor cycle, and a curve point takes
        # far longer than one cycle (a 26G merge plus 590 evaluations), so re-queueing on "cell not
        # full yet" handed the same point to a second card. Both merged into one directory at once
        # and corrupted the model. The merge dir is the in-flight marker.
        M=$BRACE_WORK/big/merged_${t}_step${s}
        { [ -d "$M" ] || ls -d $M.tmp.* >/dev/null 2>&1; } && { last=$s; continue; }
        # ...and skip anything an eval slot is ALREADY assigned. The merge dir alone is not enough:
        # the supervisor pops a cell, the queue empties, refill runs and re-adds the same point
        # before the merge has created its directory, and a second card takes it. That raced twice
        # (18:44 and 19:13) and corrupted a model the first time. The slot state records the cell
        # each card holds the moment it is launched, which closes the window.
        grep -q " STEP$s@$t\$" "$R/logs/.supervisor_state" 2>/dev/null && { last=$s; continue; }
        grep -q "^STEP$s@$t\$" "$Q" 2>/dev/null || { echo "STEP$s@$t" >> "$Q"; queued=$((queued + 1)); }
        last=$s
    done
done

# ---------------------------------------------------------------- order: best validation first
#
# The order of this file is the order the eval cards measure in, and cards are the scarce
# resource -- a point that reaches the front three hours earlier is a point in the paper three
# hours earlier. Until now the order was the order of $ARMS, a hand-set list, so which arm got
# measured first was decided by where someone typed it. That is not a priority, it is an
# accident, and it systematically starved whichever arms were added to the end of the line.
#
# Rank by the arm's VALIDATION reward at that step instead: held-out pool, one fixed surface for
# every arm, greedy decoding (surface/verl_rl/val_curve.py). Training reward cannot serve here --
# arms train on different surfaces, so ordering by it would put the arm with the most advertised
# tools first regardless of what any policy learned.
#
# val_curve.py --rank is total and order-preserving on its input: arms with a real validation
# measurement come first, arms with none keep working behind them ordered by their own training
# tail (explicitly a fallback, never a cross-arm comparison), and unparsable ids are left at the
# end untouched. The whole queue is re-ranked, not just the new points, because a point queued
# before its arm had validation data would otherwise keep its arbitrary position forever.
#
# THE REWRITE MUST NOT LOSE A CELL. next_cell pops with `head -1` + `sed -i 1d`, so this file is
# written under the same race the stale-entry prune above already accepts -- one atomic mv, and
# only if the ranked output still holds every line that went in. A ranker that dropped an entry
# would silently delete queued work; falling back to the unranked order costs nothing but time.
if [ -s "$Q" ]; then
    # Temp files beside $Q, not in $TMPDIR: a cross-filesystem `mv` is copy-then-unlink, which
    # leaves a window where a card popping the queue can read a half-written or empty file.
    # Same directory makes the replacement a rename, which is atomic.
    ranked=$(mktemp "$Q.rank.XXXXXX"); kept=$(mktemp "$Q.keep.XXXXXX")
    trap 'rm -f "$ranked" "$kept"' EXIT
    # Non-STEP entries are hand-queued specials (base cells, SELF cells) with their own reasons
    # for being where they are; they keep their order and stay ahead of the curve points.
    grep -v '^STEP[0-9]*@' "$Q" > "$kept" 2>/dev/null
    # TWO TIERS: cells the MAIN TABLE is missing first, validation-ranked within each tier.
    #
    # Validation rank alone answers "which checkpoint of this arm is best", which is the right
    # question for a learning curve and the wrong one for deciding what to measure next. On
    # 2026-08-10 it put a8C -- the method arm, with zero cells and therefore no row in the main
    # table at all -- at positions 45-47 of 55, behind nine points of an ELSA ablation, purely
    # because a8C had just adopted VAL_FREQ and had no validation number yet to rank on. Three
    # cards at hours per cell makes that days. A table with no row for the method cannot be
    # written no matter how well every other arm is measured.
    #
    # MATCHED_STEPS mirrors main_table.py's default window (the peak window; later steps rank
    # arms by rate of decay). A point inside it is load-bearing for the paper's headline
    # comparison; a point outside it refines a curve that already exists. Ranking is unchanged
    # WITHIN each tier, so the "best validation first" ordering still decides what runs next
    # among the cells that matter equally.
    MATCHED_STEPS=${MATCHED_STEPS:-"15 20 25 30"}
    mre=$(echo "$MATCHED_STEPS" | tr ' ' '|')
    ranktmp=$(mktemp "$Q.rt.XXXXXX")
    grep '^STEP[0-9]*@' "$Q" 2>/dev/null \
      | $BRACE_ENVS/mcp_verl/bin/python \
          "$R/surface/verl_rl/val_curve.py" --rank 2>/dev/null \
      | cut -f1 > "$ranktmp"
    # Within tier 1, COVERAGE BEFORE REFINEMENT: an arm with no measured matched-step cell has no
    # row in the main table at all and cannot be compared to anything; an arm missing one of its
    # four steps still has a row and a mean. So order tier 1 by how many matched cells the arm
    # already has, fewest first, and let the validation rank break ties as before (sort -s keeps
    # the ranker's order within equal counts). This is what actually gets the method into the
    # table: a8C sat behind nine cells belonging to arms that were already in it.
    awk -v mre="$mre" -v R="$R" '
      { split($0, p, "@"); arm = p[2]; st = p[1]; sub(/^STEP/, "", st)
        if (!(arm in have)) {                       # count this arm s measured matched cells once
          n = 0; split(msteps, ms, " ")
          for (i in ms) { f = R "/work/coadapt_eval_" arm "/cell_STEP" ms[i] ".jsonl"
                          c = 0
                          while ((getline line < f) > 0) { if (++c >= 300) break }
                          close(f)
                          # 300 matches main_table.py --min-pairs: below it the cell is reported
                          # as insufficient and contributes no row, so it is not "measured" yet.
                          if (c >= 300) n++ }
          have[arm] = n }
        # The METHOD arm and its ablation outrank other zero-coverage arms. This is a real
        # asymmetry, not a thumb on the scale: with no a8C row the main table has nothing to
        # compare, while a missing baseline row only makes the comparison narrower. Everything
        # else about the ordering is untouched, and this key does nothing once they are measured.
        prio = 1
        if (index(" " parms " ", " " arm " ") > 0 && have[arm] == 0) prio = 0
        printf "%d\t%d\t%s\n", prio, have[arm], $0 }
    ' msteps="$MATCHED_STEPS" parms="${METHOD_ARMS:-a8T6d a8T5kr a8T5 a8T5k a8T3g2 a8T3gr a8T3 a8T3g a8Tvip a8Tvipr a8Tlp a8Tpe a8T2n a8T2nr a8T2 a8T2w a8T2s a8T2r a8T2g1 a8T2g1r a8T2b a8T a8Te0 a8C a8Cu a8T4}" \
      <(grep -E "^STEP($mre)@" "$ranktmp" 2>/dev/null) \
      | sort -s -n -k1,1 -k2,2 | cut -f3- >> "$kept"              # tier 1: main-table cells
    grep -vE "^STEP($mre)@" "$ranktmp" >> "$kept" 2>/dev/null     # tier 2: curve refinement
    rm -f "$ranktmp"
    sort "$kept" > "$ranked"
    if sort "$Q" | cmp -s - "$ranked" && [ -s "$kept" ]; then
        chmod --reference="$Q" "$kept" 2>/dev/null
        mv "$kept" "$Q"
        echo "curve queue re-ranked by validation reward, best first"
    else
        echo "curve queue left as-is: ranker did not return every entry"
    fi
fi
echo "queued $queued curve point(s)"

# DECISION CELLS jump the whole ordering. The two-tier rank optimizes coverage, which is right
# in steady state and wrong at a selection moment: the cells that decide the method were being
# perpetually outranked by zero-coverage arms. Listed cells (space-separated, already-queued)
# are moved to the absolute front after every re-rank; same set, order only.
EVAL_DECIDE=${EVAL_DECIDE:-"STEP25@a8T2nr STEP15@a8T2n STEP20@a8T2n STEP15@q2bN STEP20@q2bN STEP25@q2bN STEP30@q2bN"}
if [ -n "$EVAL_DECIDE" ] && [ -s "$Q" ]; then
    dtmp=$(mktemp "$Q.dec.XXXXXX")
    for d in $EVAL_DECIDE; do grep -x "$d" "$Q" >> "$dtmp" 2>/dev/null; done
    grep -vxF -f "$dtmp" "$Q" >> "$dtmp" 2>/dev/null || cat "$Q" >> "$dtmp"
    if [ "$(sort "$dtmp" | md5sum)" = "$(sort "$Q" | md5sum)" ]; then mv "$dtmp" "$Q"; else rm -f "$dtmp"; fi
fi
echo "decision cells pinned: $EVAL_DECIDE"

