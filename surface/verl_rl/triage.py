"""TRIAGE: allocate the cycle's generation budget by posterior expected GRADIENT MASS.

WHY THIS EXISTS. Measured over 293,972 rollouts on this environment: 46.4% of tasks are never
solved once, 18.6% are always solved, and 87.3% of GRPO groups therefore have zero within-group
variance -- zero advantage, zero gradient. 36.6% of all rollout compute went to tasks that have
never produced a gradient in any arm. Neither reward shaping (ACCORD, -0.0068) nor tool-surface
steering (ELSA, +0.0110) can reach that, because the binding constraint is not reward density or
advertised capability: it is WHERE THE ROLLOUTS GO.

THE QUANTITY. For a task solved with probability p, a group of k rollouts carries gradient with
probability g(p,k) = 1 - p^k - (1-p)^k. That is exactly the object the diagnosis is written in, so
the method is to allocate by it -- with p unknown, under its posterior:

    w_i = E[ 1 - p^k - (1-p)^k ],   p ~ Beta(1 + s_i, 1 + f_i)
    E[p^k] = prod_{j<k} (a+j)/(a+b+j)                      (closed form, no sampling)

s_i and f_i are DISCOUNTED counts of solves and failures from THIS ARM'S OWN banked episodes.
Nothing is seeded from another arm: the first cycle has no evidence, every w_i is the flat-prior
value (k-1)/(k+1) ~ 0.778, and the allocation is uniform. The arm is honest and self-contained.

WHY A FLAT PRIOR AND NO EXPLORATION BONUS. Posterior width does the exploring for free. An
unseen task scores 0.778, a task failed 20 times scores 0.276, one failed 200 times scores 0.038.
There is no UCB term to tune and no optimism hack -- contrast aws_reweight.py, which has to hand
unobserved tasks "the best weight any observed task has" because its point estimate cannot express
ignorance.

WHY DECAY AND EPSILON. gamma=0.9 per cycle rebuild (half-life ~6-7 cycles) means a task written
off under an old policy is re-probed as the policy moves; eps=0.10 of the budget drawn uniformly
means no task's probability can ever reach zero, whatever the weights say. Distribution narrowing
is the failure mode this arm is most exposed to, and these two are what bound it.

WHAT IS ALLOCATED. The cycle's task list, sampled WITHOUT REPLACEMENT, is the parquet -- exactly
budget = steps_per_cycle * train_batch_size rows, so verl consumes it in precisely the cycle's
steps and the sampled multiset IS the generated multiset. This is the difference from
aws_reweight.py, which expresses a weight as row REPETITIONS in an 8x-oversized pool and lets the
shuffler draw from it: repetition can only approximate a target distribution, and it cannot
express "each chosen task appears once", which is what makes the budget accounting exact here.

WHAT IT COSTS. Nothing. The statistics are episodes the trainer already generated, read before
the cycle starts. DAPO -- the direct competitor -- filters degenerate groups AFTER paying to
generate them; this decides before generating anything.

STATE AND RESTARTS. Counts, the decay clock and the episode read offset live in triage_state.json
and a cycle already built is never rebuilt. The driver re-enters cycle 1 on every relaunch, so a
stateless implementation would re-apply gamma once per replayed cycle and quietly reset the
posterior every time the arm was restarted.

--------------------------------------------------------------------------------------------
V2 (2026-08-11): TWO MEASURED FAILURES OF THE ABOVE, AND THE TWO FLAGS THAT FIX THEM.

Both are failures of the FLAT PRIOR + LINEAR SAMPLING combination, measured on a8T/a8Te0/a8Tg1:

1. THE EXPLORE TAX. Cold start plus optimism means the first cycles are a probe of the 1127-task
   pool, not an allocation: unexplored tasks all sit at (k-1)/(k+1) and outnumber the evidenced
   ones, so cycle 4 of a8Te0 spent 137/256 slots on tasks with under 4 observations. Over steps
   1-15 the arms measured 89.5% degenerate groups against the uniform baseline's 87.4% -- the
   mechanism never got to act inside the window the paper measures.
   FIX: --warm-bank. 293k rollouts of per-task evidence already exist in the OTHER arms' episode
   logs, and task solvability is task-intrinsic (measured: tool-count vs solve-rate concordance
   0.453, z=-0.97; 75.6% of ever-solved tasks were solved under several different surfaces), so
   a foreign arm's counts are informative about THIS arm's p even though they are not its own
   evidence. They enter as a PRIOR, capped at (20 solves, 40 failures) so ~2 cycles of the arm's
   own rollouts outweigh them, and gamma ages them exactly like live counts, so the foreign
   evidence is gone within a handful of cycles rather than pinned there forever.

2. WEAK SUPPRESSION. w is bounded and flat in the tail: at k=5 an evidenced-dead task (s=0,f=20)
   still scores 0.19 against a mid-band 0.88 -- only 4.6x -- and there are ~500 evidenced-dead
   tasks against ~200 mid-band ones, so linear sampling still hands the dead ones ~80 of 256
   slots (a8Te0 cycle 4, measured: 79).
   FIX: --temp. Sample proportional to w^(1/tau); tau=0.5 squares the ratio (4.6x -> 21x), which
   is what turns a 3:1 population disadvantage into a minority of the slots. eps is untouched, so
   the coverage floor that bounds distribution narrowing is exactly the one v1 runs.

Both default OFF: with neither flag this file is v1 exactly, which is what the three arms already
in flight keep running (verified byte-identical on the same state/seed/inputs).

--------------------------------------------------------------------------------------------
V3 (2026-08-12): ALLOCATION REFUSES DEFECTIVE TASKS, AND THE POSTERIOR STOPS PRETENDING THE k
ROLLOUTS OF A GROUP ARE INDEPENDENT.

v2's mechanism works and its ordering matches the mechanism (step-15 held-out: cold a8T -0.0153,
a8T2 +0.0186, a8T2g1 +0.0255) but none of it beats plain a8F (+0.0271 @15). Two measured levers
were left on the table; v3 is both, and again nothing else.

3. CERTIFIED POOL EXCLUSION -- allocation refusing tasks that are DEFECTIVE, not merely hard.
   The failure taxonomy (work/analysis/failure_taxonomy.{jsonl,md}) certifies two sets, and
   "certified" is literal: each is a property of the verifier's source, decided before any
   rollout, not an inference from failure counts.
     verifier-unsatisfiable  a guard whose predicate reads ONLY initial_db-derived state fires,
                             so the verifier returns 'others' for every reachable final state.
                             38 of pool_max's 1127 tasks. MEASURED over 173 live groups on
                             a8T2/a8T2g1/a8T2r/a8T2w/a8F: degenerate fraction 1.0000, exactly
                             zero live groups, ever, in 11,641 banked rollouts (3.34% of all
                             generation on this pool). The detector fires on 0 of 809
                             ever-solved tasks.
     pre-solved              run_verifier_graded(code, initial_db, final_db := initial_db, "")
                             returns 'complete': the verifier passes on a NO-OP, so the task
                             tests nothing. 66 of 1127. These are NOT excluded for gradient
                             yield -- measured, they are 0.8526 degenerate against the kept
                             pool's 0.8570, i.e. very slightly BETTER -- they are excluded
                             because the reward on them is decoupled from behaviour: 25.5% of
                             their solved episodes end in no_tool_call against 4.0% on the rest,
                             and they pay 1.0 anyway. That is reward noise pointed at the exact
                             failure mode the base policy already has, and it costs 2.1x the
                             response tokens per solved episode (median 2032 vs 960).
   --exclude-tasks removes them from the candidate list ENTIRELY: w is never computed, and they
   are not in the set the eps-uniform floor draws from either. eps is a hedge against the
   POSTERIOR being wrong about a task; it is not a hedge against a verifier that cannot pass, so
   applying it here would be spending the coverage budget on the one thing that is certain.
   THE EVALUATION AND VALIDATION POOLS ARE NOT TOUCHED. This is the training allocation refusing
   to buy rollouts it can prove are wasted, which is what an allocator is for; the held-out
   comparison against a8F stays exactly as valid as it was.

4. GROUP-CORRELATED POSTERIOR (--warm-shrink group). v2's w assumes the k rollouts of a group are
   iid Bernoulli(p). They are not, and the size of the error was measured rather than assumed.
   a8T2 cycles 1-2, 512 groups of k=5, against the iid prediction at the same per-task p_i:

       n_solved      0      1      2      3      4      5
       observed    302     17     13     14      8    158
       iid  says   101    147    142     79     30     13

   The batch MEAN is right -- observed solve rate 0.3281 against the prior's 0.3273 -- but the
   within-group spread is nearly absent: rollouts of one group share a checkpoint and a prompt,
   so they agree far more often than k independent draws at the same p. Fitted intra-class
   correlation rho = 0.78 (concentration nu = (1-rho)/rho = 0.285), pooled over the four warm
   arms' first two cycles.
   THE FIX IS THE SAME CLOSED FORM, ONE PARAMETER LOWER. Model the group as exchangeable
   Beta-Binomial with mean p and concentration nu, so P(all k solve | p) = prod_{j<k} (nu*p+j) /
   prod_{j<k} (nu+j), a degree-k polynomial in p whose expectation under the Beta posterior is a
   finite sum of the beta_moment terms already in this file. nu -> infinity returns v2 exactly
   (verified to 1e-12), so this is one dial, not a second method.
   WHAT IT BUYS THE ALLOCATION, not just the report: it re-prices the ALWAYS-SOLVED band. At
   k=5, p=0.95 scores 0.226 under iid -- more than an evidenced-dead task's 0.096, so v2 bought
   it -- and 0.039 under the correlated model, because a task the policy almost always solves
   comes back all-solve almost every time. mid:dead widens 9.75x -> 11.5x, and at tau=0.5 that
   is 95x -> 132x. It is the same argument as the pre-solved exclusion, arrived at from the
   statistics instead of from the verifier source, and the two do not double-count: exclusion
   removes 66 tasks the verifier certifies, the posterior demotes the rest of the easy band.

   WHY NOT THE CAPABILITY SHRINK IT REPLACES. The obvious story for "predicted 0.24, measured
   0.88" is that the bank's rates come from trained policies (pooled 0.2465) while a fresh arm
   starts at base capability (greedy validation 0.1193), so p_prior should be scaled by
   lambda = 0.1193/0.2465 = 0.484. --warm-shrink capability implements exactly that and it is
   NOT what these arms run, because the premise is measurably false. On the identical instrument
   -- each arm's own cycle-1 batch, same pool, same surface, same temperature -- the observed
   solve rate over the prior's prediction is:
       a8T2 1.002   a8T2r 1.175   a8T2g1 1.201   a8T2w 1.027   a8Te0 0.950   a8T 0.938
       a8F (uniform allocation, 4513 groups) 1.051
   lambda is 1, not 0.48. The 0.1193 is a greedy pass on a DISJOINT 296-task held-out pool at
   temperature 0; the 0.2465 is temperature-1 training rollouts on pool_max. Their ratio measures
   two pools and two decoding regimes, not two capability levels. Applied as a prior correction
   it would halve every p_i in the wrong direction. It closes 34% of the calibration gap
   (0.2308 -> 0.4546 against a measured 0.8828); the correlated posterior closes 90% (0.8204),
   and closes it out of sample too -- a8F 0.8989 predicted against 0.8741 measured, a8Te0 0.8988
   against 0.9062, on a nu fitted without either arm.

All three v3 flags default OFF, so with none of them this file is v2, and with none of the v2
flags either it is v1 -- the arms in flight are unaffected (verified byte-identical: same chosen
pool, same counts, same offset, same cycles_built, on a cold cycle 1 and on a8T2's real cycle-5
state).

--------------------------------------------------------------------------------------------
V4 (2026-08-12): THE PUBLISHED-METHOD BASELINES, RUN INSIDE THIS ALLOCATOR RATHER THAN BESIDE IT.

The literature sweep (work/analysis/related_work_sweep.md) found that neither the objective nor
before-generation allocation is ours: TRACE (2606.11119) allocates a pre-generation budget by
1-v^m-(1-v)^m from a learned value net, and VIP (2602.01601, ICLR 2026, code public) allocates it
by minimising gradient variance from a GP. The surviving claim is the ESTIMATOR. A claim about an
estimator is only testable against the other estimators, so all three go in HERE -- same sampler,
same warm bank, same gamma, same eps, same budget, same seed, same stats line -- and differ from
a8T3g in exactly one named thing each. A baseline that needed its own launch path would differ in
ways nobody enumerated.

5. --rule vip. VIP'S OBJECTIVE, ADAPTED TO A FIXED GROUP SIZE.
   VERIFIED AGAINST THE PAPER PDF (arXiv:2602.01601v1, "Published as a conference paper at ICLR
   2026"), not a summary. What it says, literally:
     Prop 4.2 (Dr. GRPO): Var(G~) = ((n-1)/n^2) * 4 sigma_Z^2 * p(1-p)
     Prop 4.3 (RLOO):     Var(G~) = (1/(n-1))   * 4 sigma_Z^2 * p(1-p)
     Eq 5:  min { sum_q Var(G~_q) : sum_q n_q = C, n_q in {L,L+1,...,U} }, "we require L >= 3"
     a_q := 4 sigma_Zq^2 p^_q(1-p^_q); Thm 5.2 solves the RLOO relaxation as
            n*_q(lambda) = 1 + sqrt(a_q/lambda) clipped to [L,U], lambda by bisection
            (Thm 5.1 is the same shape for Dr. GRPO: lambda = a_q (n-2)/n^3).
     p^_q = sigmoid(m_t(x_q)) from a GP over MiniLM prompt embeddings, RBF kernel, median-heuristic
            bandwidth, updated from the clipped batch mean reward.
   THE ONE FACT THAT MAKES THE ADAPTATION EXACT RATHER THAN APPROXIMATE: in both theorems n*_q is
   a strictly INCREASING function of a_q at the common lambda*, and sigma_Zq^2 is assumed equal
   across prompts by the paper itself ("we also assume that Z~_q has the same variances over all q
   when we compute the allocation", Sec. 6, supported by their App. B.3 test). So VIP's optimal
   allocation ORDERS prompts by p^_q(1-p^_q) and by nothing else. At a fixed group size the
   ordering is all that survives, and it survives intact.

   DEVIATIONS, all forced by this framework, none discretionary:
     (i)   DECISION VARIABLE. VIP chooses n_q per prompt; verl generates a fixed
           actor_rollout_ref.rollout.n = k for every prompt in the parquet, so n_q is not ours to
           choose. The budget lever we have is WHICH tasks are in the batch. VIP hands prompt q a
           share n_q/C of the rollout budget; this hands it a probability proportional to a_q of
           being in the batch, which is the same ordering over the same object.
     (ii)  FEASIBILITY. VIP requires B*L <= C <= B*U with L >= 3. Our pool is B = 1127 (1023 after
           exclusion) against a cycle budget of C = 256*5 = 1280 rollouts, so B*L = 3381 > C:
           VIP's feasible set on this pool is EMPTY. Its optimisation cannot be run here at all
           without first choosing a sub-batch -- which is the selection VIP does not do (its B_t
           is the ordinary uniformly-drawn minibatch). Reported, not hidden: it is the structural
           reason a fixed-k framework cannot run VIP unchanged.
     (iii) ESTIMATOR. VIP predicts p^_q with a GP on prompt embeddings; this uses the same
           discounted Beta posterior every other rule in this file uses, from the same warm bank.
           Deliberate: swapping their predictor in as well would confound the objective comparison
           with a predictor comparison, and the predictor is the thing the paper claims. The GP is
           a different way to SHARE information across prompts (their kernel is over MiniLM
           embeddings of the prompt text); ours is per-task with no sharing. Stated as a limitation
           -- this is VIP's objective on our estimator, not VIP end to end.
     (iv)  UNCERTAINTY. VIP plugs a point p^ into p(1-p). This scores the posterior EXPECTATION
           E[p(1-p)] = E[p] - E[p^2], the same substitution TRIAGE makes on TRACE's functional, so
           the two rules differ in their objective and not in how they treat uncertainty. The two
           are not equal: E[p(1-p)] = p_bar(1-p_bar) - Var(p), i.e. VIP-at-the-posterior-mean minus
           the posterior variance, so a wide posterior is scored LOWER here than by a plug-in --
           the opposite of the optimism the gradmass rule gets for free. That is VIP's property,
           not a bug in the port, and --estimator point exposes the same gap on the gradmass side.
     (v)   NO EXCLUSION, BY CONSTRUCTION. L >= 3 means VIP structurally cannot give a prompt zero
           rollouts. The mapped property is enforced, not merely observed: --rule vip REFUSES
           --exclude-tasks (fatal), and its weight E[p(1-p)] is strictly positive for every Beta
           posterior, so no task is ever zero-weighted and the eps floor still covers all of them.
     (vi)  SAMPLING. VIP's allocation is deterministic; this draws without replacement with
           probability proportional to the score, in the same Efraimidis-Spirakis pass as every
           other rule, plus the same eps floor. Comparability is the reason: an arm that also
           swapped the sampler would not isolate the objective.
   NOT A DEVIATION, WORTH KNOWING: at any k >= 2, g(p,k) = 1 - p^k - (1-p)^k is a strictly
   increasing function of u = p(1-p) alone (P_n = P_{n-1} - u P_{n-2} with P_0=2, P_1=1 gives
   p^k+(1-p)^k as a polynomial in u; k=2,3 are exactly 2u and 3u). So on POINT estimates VIP's
   variance criterion and TRACE's availability criterion are the SAME RANKING. Everything that
   separates them at fixed k comes from the posterior, from tau, and from exclusion -- which is
   precisely the paper's position that the estimator is what is left to claim.

6. --rule progress. LEARNING PROGRESS (Graves et al. 2017; Matiisen et al., TSCL).
   w_i = |p^_i(recent) - p^_i(older)|, the absolute change in success rate between the two most
   recent evidence windows for the task. Graves scores by a learning-progress gain signal and TSCL
   by "the slope of the learning curve", taking the ABSOLUTE slope so that tasks the student is
   FORGETTING are re-selected too ("the Teacher algorithms address the problem of forgetting by
   also choosing tasks where the Student's performance is getting worse"). The signal here is
   PROGRESS, not availability: a task stuck at p=0.5 forever scores 0 under this rule and ~0.94 (at
   k=5) under gradmass, which is exactly the contrast the baseline exists to draw.
   THE TWO WINDOWS ARE THE ONES THIS ALLOCATOR ALREADY KEEPS. Counts are decayed by gamma and then
   this cycle's episodes are added, so at the moment of the rebuild the state already holds
   "everything before this cycle, gamma-aged" and "this cycle's own rollouts" separately. The older
   window is therefore a gamma-EWMA rather than a fixed-width sliding window (deviation from TSCL's
   fixed window, forced by the state design and consistent with the decay every other rule uses).
   COLD START IS UNIFORM, AND HAS TO BE: a task with fewer than two windows of evidence has no
   progress signal at all, and the classic fix -- optimism -- is the thing under test on the other
   arms. Those tasks get the MEAN of the measured |delta| (explore-neutral: they neither outrank
   nor underrank a task that was actually measured). At cycle 1 nothing has a live window, so every
   task gets the mean and the draw is exactly uniform, warm bank or not.

7. --estimator point. THE TRACE-STYLE ESTIMATOR ABLATION, AND THE ONE THE NOVELTY CLAIM RESTS ON.
   Same rule (gradmass), same posterior, same everything: only w changes, from the posterior
   EXPECTATION E[g(p,k)] to the PLUG-IN g(p^,k) at the posterior mean p^ = a/(a+b) -- TRACE's
   "V_root(x_i,m) = 1 - v_i^m - (1-v_i)^m" with v_i a point prediction of the root success
   probability. Under --warm-shrink group the plug-in is taken under the same group model
   (degen_at_mean_group), so the ablation isolates the Jensen gap and not the correlation model.
   THE GAP, STATED WHERE IT BITES. A plug-in cannot express ignorance, because g(p^,k) is a
   function of the posterior's location and of nothing else. At k=5 under the iid model, an
   UNEXPLORED task (flat prior) and a task MEASURED at exactly p=0.5 (s=f=10) have the same
   posterior mean 0.5, so the plug-in scores both 0.9375 and cannot tell them apart; the posterior
   scores them 0.6667 and 0.9087 -- it knows one of them is a guess. That is the whole optimism
   story of v1 (a flat-prior task is scored BELOW a measured mid-band task, so evidence is worth
   something) and it is unavailable to a point estimate at any width. In the tail the two nearly
   agree (s=0,f=20: 0.1923 posterior vs 0.2076 plug-in), which is why this ablation is expected to
   act mainly through the no/low-evidence tasks and through re-probing, not through the dead band.
   --estimator posterior is the default and is byte-identical to v3.

--------------------------------------------------------------------------------------------
TRIAGE-B (2026-08-14): ALLOCATING THE ROLLOUT BUDGET INSTEAD OF THE TASK LIST -- AND THE PROOF
THAT AT MATCHED COMPUTE ON THIS POOL IT CANNOT DIFFER FROM FIXED-k SELECTION.

8. --rule budget. Every rule above chooses WHICH tasks enter generation at a fixed k; this one
   chooses HOW MANY rollouts each task gets. Total budget B = k*N is unchanged (N = the parquet's
   row count = steps_per_cycle * train_batch_size), so compute is matched to every existing arm to
   the rollout; task i receives k_i = k*m_i rollouts with m_i in {0..--k-budget-max} and
   sum_i m_i = N exactly.

   THE ENGINEERING CONSTRAINT DECIDES THE OBJECTIVE, so it is stated first. verl assigns the GRPO
   grouping key itself -- ray_trainer.py::fit does
       batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))])
   UNCONDITIONALLY, one fresh uuid per parquet row, before gen_batch.repeat(rollout.n), and
   compute_advantage groups on exactly that column. A parquet-supplied uid is overwritten; and
   data.shuffle=True means duplicate rows of one task do not even land in the same optimizer step.
   So m_i rows of a task are m_i INDEPENDENT groups of k, never one group of k*m_i. Every field in
   the stats line keeps its exact meaning under that reading -- a row is still one group of k --
   which is the second reason to take it.

   Therefore the value of m units to task i is the probability it yields AT LEAST ONE
   gradient-carrying group, not the availability of a single larger group:

       V_i(m) = 1 - E[ q_k(p)^m ],   q_k(p) = P(a group of k is degenerate | p)

   q_k is the same degree-k polynomial the group model already uses (all-solve + all-fail under
   the exchangeable Beta-Binomial at --shrink-nu, or p^k + (1-p)^k under iid), so q_k^m is a
   polynomial of degree k*m and E[q_k^m] is a finite sum of the SAME beta_moment terms. Zero new
   estimation machinery, and V_i(1) is grad_mass_group / grad_mass EXACTLY (verified to 1e-16), so
   at m_max = 1 this rule is v3g's weight.

   GREEDY IS EXACTLY OPTIMAL, and here it is a theorem rather than a numerical finding.
   Delta_i(m) = V_i(m+1) - V_i(m) = E[ q^m (1-q) ] and q in [0,1], so q^{m+1}(1-q) <= q^m(1-q)
   pointwise: the marginals are non-increasing in m for EVERY posterior and every nu. A separable
   concave maximisation under one budget constraint is solved exactly by giving each unit to the
   largest remaining marginal (the classic incremental/water-filling result). Verified anyway:
   0 increasing-marginal violations over the bank's 52 distinct posteriors and over an s in [0,40]
   x f in [0,60] grid at k in [1,21], for this objective AND for the one-group w(k*m) form; and
   greedy == brute-force optimum on 400 random instances with N <= 8, m_max = 3.

   THE MEASURED FINDING, AND IT IS THE REASON THIS RULE IS AN EXPLORATION AND NOT THE METHOD.
   A second unit is bought only when some Delta_j(1) exceeds the FIRST-unit value V_i(1) of the
   best still-unfunded task. On the pinned a8T3g cycle-1 bank (1023 tasks after certified
   exclusion, 9,582 solves / 38,532 failures, nu = 0.285):
       max_j Delta_j(1)                                   0.17361
       # tasks with V(1) above that                         417
       matched budget N                                     256
   417 > 256, so the greedy allocation at matched compute is 767/256/0/0 over m = 0/1/2/3 --
   IDENTICAL to fixed-k selection of the top 256 tasks. The first N with any task at m >= 2 is
   N = 418 (1.63x the matched budget); under the one-group objective it is N = 492 (1.92x), and
   under the iid posterior 518 tasks sit above the top marginal. The degeneracy is NOT marginal
   collapse -- Delta(1)/V(1) is 0.78-0.96, because rho = 0.78 leaves V(1) at most 0.224, nowhere
   near saturation, so a second independent group really is worth ~0.8 of the first. It is SUPPLY:
   the pool holds more tasks worth a first look than the cycle has rows to give. This is the same
   measurement the a8T5k sweep hit from the other side (440 mid-band tasks vs a budget of 256).
   A related artefact of the warm caps (s<=20, f<=40): the 1023 tasks carry only 51 distinct
   V(1) values, and the 256th slot sits inside a tie class of 354, so a DETERMINISTIC argmax over
   this pool is arbitrary among ties in a way v3g's stochastic draw is not.

   --budget-draw sample is the non-degenerate variant, kept because the greedy one is provably
   inert here: it draws the N units sequentially with probability proportional to the current
   marginal Delta_i(m_i)^(1/tau), capped at m_max, with the same eps floor. It reduces to v3g's
   draw exactly when second-unit marginals vanish, and on this bank at tau=0.5 it allocates
   807/178/36/2 over m = 0/1/2/3. It is a DRAW, not an optimiser: its objective value (44.57) is
   below the greedy's (52.13) by construction.

   NOT COMPOSABLE WITH TRIAGE_DECOR=shuffle. awm_agent_loop.py recovers verl's rollout index as a
   per-process counter over CONSECUTIVE rollouts of one (scenario, task_idx); with m_i > 1 rows of
   a task in one batch that counter runs 0..k*m_i-1 against a permutation family sized k. Refused
   at argv rather than left to be discovered.

--------------------------------------------------------------------------------------------
TRIAGE-rho (2026-08-14): THE CORRELATION IS TASK-SPECIFIC, AND THAT IS WHAT IS LEFT OF THE RANKING.

9. --estimator taskrho (with --task-rho FILE). v3g calibrates the group model with ONE global
   correlation -- rho = 0.78, nu = 0.285, fitted once over a8T2's first two cycles -- and lets the
   per-task Beta posterior over p do all the ranking. THE MEASURED PROBLEM IS THAT THE p-POSTERIOR
   HAS RUN OUT OF RANKING SIGNAL. On the pinned a8T3g cycle-1 bank the warm caps (s <= 20, f <= 40)
   collapse 1023 tasks onto 51 DISTINCT weight values; only 31 tasks score strictly above the 256th
   slot, and that slot sits inside a TIE CLASS OF 354 tasks that all read w = 0.202011 because they
   all sit at the cap (20 solves, 40 failures). 225 of the cycle's 256 rows are therefore handed
   out arbitrarily among ties, and no amount of extra p-evidence can break them -- the cap is what
   destroyed the resolution, and raising it does not help either (measured: see below).

   THE MODEL, one parameter wider than v3's and no new closed form. A group on task i is
   exchangeable Beta-Binomial with its OWN concentration:

       s_ig ~ BetaBin(k, alpha_i, beta_i),   p_i = alpha_i/(alpha_i+beta_i),
                                             rho_i = 1/(alpha_i+beta_i+1)

       w_i = 1 - prod_{j<k} (alpha_i+j)/(alpha_i+beta_i+j)
               - prod_{j<k} (beta_i+j)/(alpha_i+beta_i+j)

   which is degen_at_mean_group's complement evaluated at nu_i = (1-rho_i)/rho_i instead of the
   global nu -- i.e. grad_mass_point(s, f, k, nu_i). v3 is the special case rho_i == rho for all i.
   alpha_i and beta_i are built from the SAME p_hat the rest of this file uses (the posterior mean
   of Beta(1+s, 1+f) on the same decayed counts and the same warm bank) and from rho_i, which is
   fitted OFFLINE from the task's own historical GROUPS by fit_task_rho.py and read here as a
   pinned file. A task absent from the file falls back to the global rho exactly, so this is a
   strict generalisation and not a second method.

   WHY rho_i IS PINNED AND NOT UPDATED LIVE. Same argument as --warm-bank: the bank is a snapshot,
   not a subscription. rho is a property of the harness -- the k rollouts of a group share a policy
   checkpoint AND a byte-identical prompt -- and the bank holds 24,877 groups against the 256 a
   cycle adds, so a live update would be swamped for many cycles while re-injecting foreign
   evidence gamma could never age out. What DOES move every cycle is p, and the existing count
   machinery already tracks exactly that. Stated as a limitation: rho_i does not adapt within a run.

   THE FOUR THINGS THAT WERE MEASURED BEFORE THIS SHIPPED (all on 24,877 groups of k=5 over 1127
   tasks, every 8B arm at BSZ=32/NROLL=5; full numbers in PLAN_TRIAGE.md section 5d):
     HETEROGENEITY. Pooled (one rho, p_i free) against hierarchical (rho_i free) beta-binomial:
       LR = 3724.9, against a parametric-bootstrap null of mean 550.5, sd 32.9, max 624.4 over 300
       replications drawn under the pooled MLE -- p <= 0.0033, and 96 null sd above the null mean.
       Shrunk rho_i IQR 0.166 over the pool, 0.310 over the 524 tasks with any identifying
       evidence, 0.322 over the mid band.
     DECISION RELEVANCE. Replaying a8T3g's real cycle-1 draw (reproduced byte-for-byte first) with
       this weight instead: 168/256 overlap, against a same-weights-different-seed floor of
       116/256. Inside the 354-task tie class the weight takes 353 distinct values spanning
       0.018-0.587 and replaces 65 of v3g's 185 tie-class picks.
     RETRODICTION. rho_i fitted with the tested arm HELD OUT entirely, then applied to that arm's
       own cycles: the low-rho half of the tie class produced live groups at 0.41 against the high
       half's 0.15, mean paired difference +0.246 over 778 (arm, cycle) pairs (t = 29.3), and
       20/20 arms in the predicted direction (arm-level Wilcoxon p = 9.5e-07).
     NOT AN ARTEFACT OF p. corr(rho_i, p_hat(1-p_hat)) = -0.376 Pearson / -0.271 Spearman over the
       pool; inside the tie class p_hat is IDENTICAL for all 354 tasks (0.3387) and rho_i still has
       sd 0.1999. And it is not the information the cap threw away either: predicting held-out
       degeneracy from the UNCAPPED bank rate at the global rho gives AUC 0.504 (no discrimination
       at all, Brier 0.194 against a base-rate 0.189), while rho_i gives AUC 0.562 / Brier 0.163;
       adding log nu_i on top of the uncapped-p prediction is LR 2109.6 on 1 df.

   AND THE ONE THING IT BUYS THAT NO EARLIER ARM COULD. Every rule in v1-v7 is a function of p
   alone, so VIP's variance criterion, TRACE's availability criterion and this file's gradmass are
   the SAME RANKING at fixed k (Spearman 1.000000 on this pool, measured again here). w(alpha_i,
   beta_i) is not: Spearman against VIP's E[p(1-p)] falls to 0.816, with 86,015 discordant pairs
   of 522,753. This is the first TRIAGE rule that leaves the rank-equivalence class, which is
   exactly what the budget axis (TRIAGE-B, section 8) could not do.
"""
from __future__ import annotations
import argparse, glob, json, math, os, subprocess, sys

import numpy as np

R = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Interpreter locations, resolved from the environment so no absolute path is baked in.
# BRACE_ENVS is the parent directory of the three project environments (see README.md);
# VERL_PY / AWM_PY / VLLM_PY override an individual interpreter.
ENVS = os.environ.get("BRACE_ENVS", os.path.join(R, "envs"))
VERL_PY = os.environ.get("VERL_PY", os.path.join(ENVS, "mcp_verl", "bin", "python"))
AWM_PY = os.environ.get("AWM_PY", os.path.join(ENVS, "mcp_awm", "bin", "python"))
VLLM_PY = os.environ.get("VLLM_PY", os.path.join(ENVS, "mcp_vllm", "bin", "python"))

# Every arm's episode log. The arm's OWN file is always removed from this list: it is the live
# evidence stream, read incrementally with an offset, and counting it twice would double the
# weight of exactly the observations the method is supposed to be driven by.
DEFAULT_BANK = f"{R}/work/verl/run_*/episodes.jsonl"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aws_reweight import beta_moment            # noqa: E402  the same closed form the AWS arm uses

BANDS = (("p_lt_005", 0.00, 0.05), ("p_005_02", 0.05, 0.20), ("p_02_08", 0.20, 0.80),
         ("p_08_095", 0.80, 0.95), ("p_gt_095", 0.95, 1.01))


def grad_mass(s: float, f: float, k: int) -> float:
    """w = 1 - E[p^k] - E[(1-p)^k] under Beta(1+s, 1+f). Prior Beta(1,1) is the ONLY prior here:
    fitting a pool-level prior (what aws_reweight does) would import the pool's bimodality into
    every unseen task's posterior and destroy exactly the optimism this method runs on."""
    a, b = 1.0 + s, 1.0 + f
    return 1.0 - beta_moment(a, b, k) - beta_moment(b, a, k)


def group_poly(k: int, nu: float):
    """Coefficients c[m] of prod_{j<k} (nu*p + j), and the normaliser prod_{j<k} (nu + j).

    P(all k rollouts agree | p) for an exchangeable Beta-Binomial group with mean p and
    concentration nu is prod_{j<k}(nu*p + j) / prod_{j<k}(nu + j) -- the standard beta-binomial
    all-successes probability. It is a polynomial of degree k in p, so its expectation under the
    task's Beta posterior is sum_m c[m] * E[p^m], and E[p^m] is beta_moment: the SAME closed form
    v1 uses, evaluated at k+1 orders instead of one. No sampling, no new machinery.
    """
    c = [1.0]
    for j in range(k):
        # multiply the running polynomial by (nu*p + j)
        nxt = [0.0] * (len(c) + 1)
        for m, cm in enumerate(c):
            nxt[m + 1] += cm * nu
            nxt[m] += cm * j
        c = nxt
    d = 1.0
    for j in range(k):
        d *= (nu + j)
    return c, d


def grad_mass_group(s: float, f: float, k: int, nu: float) -> float:
    """w under the group-correlated model: 1 - E[P(all solve|p)] - E[P(all fail|p)], p ~ Beta(1+s,1+f).

    nu = (1 - rho)/rho is the within-group concentration; nu -> inf recovers grad_mass exactly
    (P(all solve|p) -> p^k), which is why v2 is the limit of this and not a different rule.
    """
    a, b = 1.0 + s, 1.0 + f
    c, d = group_poly(k, nu)
    e_sol = sum(cm * beta_moment(a, b, m) for m, cm in enumerate(c)) / d
    e_fal = sum(cm * beta_moment(b, a, m) for m, cm in enumerate(c)) / d
    return 1.0 - e_sol - e_fal


def degen_at_mean_group(p: float, k: int, nu: float) -> float:
    """The plug-in degenerate probability at a POINT p under the same group model -- the
    correlated analogue of p^k + (1-p)^k, for the pred_degenerate_cal stats field."""
    c, d = group_poly(k, nu)
    return (sum(cm * p ** m for m, cm in enumerate(c))
            + sum(cm * (1.0 - p) ** m for m, cm in enumerate(c))) / d


VARK_CHOICES = (2, 3, 5, 8)


def _vark_dp(vals: list[list[float]], choices: tuple[int, ...], budget: int) -> list[int]:
    """EXACT best (k_1..k_m) with sum(k_j) == budget, by DP over (task, budget-used).

    The brief allowed a greedy marginal-gain assignment. A DP is used instead because at this size
    it is both cheaper and strictly better: m=32 tasks x budget=160 x 4 choices is ~20k relaxations
    per block, it returns the OPTIMUM rather than a heuristic, and -- the property that actually
    matters -- it lands on the budget EXACTLY. A ratio-greedy has to be repaired to hit an exact
    total (upgrade steps cost 1, 2 and 3, so it can strand a remainder it cannot spend), and that
    repair is where an off-by-one silently becomes a short batch. Feasibility is guaranteed here
    because all-k=5 already sums to exactly 32*5 = 160.
    """
    m = len(vals)
    NEG = float("-inf")
    dp = [[NEG] * (budget + 1) for _ in range(m + 1)]
    back = [[-1] * (budget + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0
    for j in range(m):
        row = vals[j]
        for b in range(budget + 1):
            cur = dp[j][b]
            if cur == NEG:
                continue
            for ci, kk in enumerate(choices):
                nb = b + kk
                if nb > budget:
                    continue
                cand = cur + row[ci]
                if cand > dp[j + 1][nb]:
                    dp[j + 1][nb] = cand
                    back[j + 1][nb] = ci
    if dp[m][budget] == NEG:
        raise ValueError(f"[triage] vark: no exact allocation of {budget} over {m} tasks")
    ks, b = [0] * m, budget
    for j in range(m, 0, -1):
        ci = back[j][b]
        ks[j - 1] = choices[ci]
        b -= choices[ci]
    return ks


def vark_allocate(chosen, counts, key, w, nu, k_nominal, rows_per_block, rng):
    """TRIAGE v4: assign a per-task group size k_i under a FIXED rollout budget.

    Fixed-k allocation is rank-equivalent family-wide (Prop 2), so it cannot express a preference
    the ordering does not already contain; variable k can, and the shipped correlation calibration
    is what makes the objective computable -- P(non-degenerate | p_i, k_i) is grad_mass_group's
    g_c(s,f,k,nu_hat), the paper's Eq. 4. Maximising sum_i g_c(s_i,f_i,k_i,nu) subject to
    sum_i k_i = budget is exactly "buy the most gradient-carrying groups the rollout budget can
    afford", which is the claim v4 makes.

    ALLOCATED PER BLOCK OF `rows_per_block` ROWS, EACH AT `rows_per_block * k_nominal` ROLLOUTS.
    The trainer consumes one block per optimizer step, so pinning each block to the fixed-k
    trajectory total (32*5 = 160) keeps the per-step batch shape, the memory footprint and the
    minibatch arithmetic identical to q2bT -- which is what lets the actor's ppo_mini_batch_size
    stay unpatched. Block membership is a fresh seeded permutation each cycle: `data.shuffle` is
    off under VARK (blocks must survive to the step), so the randomness the dataloader used to
    supply is RELOCATED here rather than removed.

    Ties are broken by the posterior-expectation weight w[t], as registered.
    """
    n = len(chosen)
    if rows_per_block <= 0 or n % rows_per_block:
        raise ValueError(f"[triage] vark: {n} rows is not a whole number of {rows_per_block}-row blocks")
    n_blocks = n // rows_per_block
    per_block = rows_per_block * k_nominal
    order = list(rng.permutation(n))
    k_by_orig = [0] * n
    for b in range(n_blocks):
        idx = order[b * rows_per_block:(b + 1) * rows_per_block]
        vals = []
        for i in idx:
            t = chosen[i]
            s, f = counts.get(key[t], (0.0, 0.0))
            vals.append([
                (grad_mass_group(s, f, kk, nu) if nu is not None else grad_mass(s, f, kk))
                + 1e-9 * float(w.get(t, 0.0))
                for kk in VARK_CHOICES
            ])
        ks = _vark_dp(vals, VARK_CHOICES, per_block)
        for j, i in enumerate(idx):
            k_by_orig[i] = ks[j]
    packed = [chosen[i] for i in order]
    packed_k = [k_by_orig[i] for i in order]
    return packed, packed_k, per_block, n_blocks


def grad_mass_point(s: float, f: float, k: int, nu: float | None = None) -> float:
    """TRACE-style PLUG-IN: g at the posterior MEAN p^ = a/(a+b), not the posterior expectation.

    v = 1 - p^^k - (1-p^)^k, which is TRACE's V_root(x_i,m) = 1 - v_i^m - (1-v_i)^m with v_i a
    point prediction. Under the group model (nu given) the same plug-in is taken under the same
    exchangeable Beta-Binomial P(all agree), so --estimator point isolates the Jensen gap and does
    not silently also turn the correlation model off.

    p^ here is the posterior mean of the SAME Beta(1+s, 1+f) the posterior estimator uses, so the
    two estimators are fed identical evidence and differ only in E[g(p)] vs g(E[p]).
    """
    p = (1.0 + s) / (2.0 + s + f)
    if nu is None:
        return 1.0 - p ** k - (1.0 - p) ** k
    return 1.0 - degen_at_mean_group(p, k, nu)


def vip_variance(s: float, f: float) -> float:
    """VIP's per-prompt gradient-variance criterion under the posterior: E[p(1-p)].

    VIP (arXiv:2602.01601, ICLR 2026) minimises sum_q Var(G~_q) under a budget; Propositions 4.2
    and 4.3 give Var(G~_q) = c(n_q) * 4 sigma_Zq^2 * p_q(1-p_q) for Dr. GRPO and RLOO, and its
    Theorems 5.1/5.2 make the optimal n*_q a strictly increasing function of a_q = 4 sigma_Zq^2
    p^_q(1-p^_q) at the common multiplier. sigma_Zq^2 is taken equal across prompts by the paper
    itself, and c(n_q) is a constant here because n_q = k is fixed, so the ENTIRE task-dependent
    part of VIP's objective is p_q(1-p_q) and that is what this returns.

    E[p(1-p)] = E[p] - E[p^2] under Beta(1+s, 1+f), both beta_moment calls -- the same closed form
    as everything else in this file. It is NOT k-aware, which is the real difference from
    g(p,k): VIP's criterion does not know the group size (see the sweep's "not k-aware in the
    degeneracy sense"). Strictly positive for every Beta posterior, so no task is ever zeroed --
    the mapped form of VIP's L >= 3, which structurally forbids skipping a prompt.
    """
    a, b = 1.0 + s, 1.0 + f
    return beta_moment(a, b, 1) - beta_moment(a, b, 2)


def learning_progress(prev: tuple, cur: tuple, min_obs: float):
    """|delta p^| between two evidence windows, or None when there is no signal yet.

    Graves et al. 2017 / TSCL: sample tasks by the ABSOLUTE slope of the learning curve, so that
    tasks getting worse are re-selected as well as tasks improving. Both windows must carry at
    least min_obs observations; otherwise there is no second point to take a slope through and the
    caller substitutes the batch mean rather than inventing one.
    """
    ps, pf = float(prev[0]), float(prev[1])
    cs, cf = float(cur[0]), float(cur[1])
    if ps + pf < min_obs or cs + cf < min_obs:
        return None
    return abs(cs / (cs + cf) - ps / (ps + pf))


def _polymul(a: list, b: list) -> list:
    out = [0.0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x == 0.0:
            continue
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def degen_poly(k: int, nu: float | None):
    """Coefficients of q_k(p) = P(a group of k rollouts is DEGENERATE | p), as a polynomial in p.

    nu given  -> exchangeable Beta-Binomial: (prod_{j<k}(nu p + j) + prod_{j<k}(nu(1-p) + j))
                 / prod_{j<k}(nu + j)   -- the same object grad_mass_group takes the expectation of.
    nu None   -> iid: p^k + (1-p)^k.
    Returning COEFFICIENTS rather than a value is what makes q^m closed-form: the m-group
    availability below is 1 - E[q^m] and q^m is just this polynomial raised to a power, whose
    expectation is a finite sum of beta_moment terms.
    """
    if nu is None:
        q = [((-1.0) ** j) * float(math.comb(k, j)) for j in range(k + 1)]
        q[k] += 1.0
        return q
    c, d = group_poly(k, nu)
    q = [x / d for x in c]                                   # all-solve part, sum c[m] p^m / d
    for m, cm in enumerate(c):                               # all-fail part, sum c[m] (1-p)^m / d
        for j in range(m + 1):
            q[j] += (cm / d) * ((-1.0) ** j) * float(math.comb(m, j))
    return q


def avail_any(s: float, f: float, k: int, m: int, nu: float | None = None) -> float:
    """TRIAGE-B's value curve: P(task yields AT LEAST ONE gradient-carrying group | m groups of k).

        V(m) = 1 - E[ q_k(p)^m ],   p ~ Beta(1+s, 1+f)

    This and not w(k*m) is the right objective for what the trainer actually runs: verl stamps one
    fresh uuid per parquet row and groups advantages on it, so m rows of a task are m INDEPENDENT
    groups of k, not one group of k*m (see the TRIAGE-B block).

    V(1) == grad_mass_group(s,f,k,nu) exactly (and grad_mass(s,f,k) when nu is None), so m_max = 1
    is v3g's weight and this rule is a strict generalisation of it rather than a second method.

    Marginals are provably non-increasing: V(m+1) - V(m) = E[q^m (1-q)] and q in [0,1], so greedy
    water-filling over m is exactly optimal. The polynomial is degree k*m; the alternating
    coefficients start to cancel badly past k*m ~ 30, which is why --k-budget-max is capped.
    """
    if m <= 0:
        return 0.0
    q = degen_poly(k, nu)
    pw = [1.0]
    for _ in range(m):
        pw = _polymul(pw, q)
    a, b = 1.0 + s, 1.0 + f
    return 1.0 - sum(cm * beta_moment(a, b, j) for j, cm in enumerate(pw))


def budget_curve(s: float, f: float, k: int, m_max: int, nu: float | None, objective: str):
    """[V(0), V(1), ..., V(m_max)] for one task under the chosen objective.

    'any'   -- 1 - E[q_k^m]: m independent groups of k, which is what Path B (row multiplicity with
               distinct uids) actually generates. THE DEFAULT, because it is the only one of the
               three that matches this verl checkout.
    'group' -- w(k*m): ONE group of k*m rollouts. Correct only under a verl that groups m rows on a
               shared uid; kept because it is the objective the design was written in and because
               its allocation is reported alongside. Concave in m as well (verified numerically).
    'count' -- m * w(k): the EXPECTED NUMBER of live groups. Exactly linear in m, so the greedy
               degenerates to giving m_max to the top N/m_max tasks -- the concentration extreme,
               kept so the two poles of the objective choice are both runnable and reported.
    """
    if objective == "count":
        w1 = grad_mass(s, f, k) if nu is None else grad_mass_group(s, f, k, nu)
        return [m * w1 for m in range(m_max + 1)]
    if objective == "group":
        return [0.0] + [grad_mass(s, f, k * m) if nu is None else grad_mass_group(s, f, k * m, nu)
                        for m in range(1, m_max + 1)]
    return [avail_any(s, f, k, m, nu) for m in range(m_max + 1)]


def allocate_budget(tasks, curves, budget: int, m_max: int, eps: float, draw: str, tau: float,
                    rng):
    """Return (m per task index, n_eps_rows). sum(m) == budget EXACTLY.

    The weighted share (budget - n_eps rows) is allocated over units of k rollouts; the eps share
    is spread uniformly over tasks the weighted pass left unfunded, so the coverage floor that
    bounds distribution narrowing means exactly what it means on every other rule: no task's
    probability of appearing is ever zero.

    draw 'greedy'  -- water-filling: each unit goes to the largest remaining marginal. Exactly
                      optimal for the concave objectives (proof in the TRIAGE-B block), and
                      deterministic.
    draw 'sample'  -- the same marginals used as SAMPLING weights at temperature tau, one unit at
                      a time. Reduces to the v1-v4 without-replacement draw when second-unit
                      marginals are negligible, and is the variant that is not inert when the pool
                      holds more first-look-worthy tasks than the cycle has rows.
    """
    n = len(tasks)
    m = [0] * n
    n_eps = int(round(eps * budget))
    n_w = budget - n_eps
    if draw == "greedy":
        import heapq
        h = [(-(curves[i][1] - curves[i][0]), i) for i in range(n)]
        heapq.heapify(h)
        for _ in range(n_w):
            if not h:
                break
            _, i = heapq.heappop(h)
            m[i] += 1
            if m[i] < m_max:
                heapq.heappush(h, (-(curves[i][m[i] + 1] - curves[i][m[i]]), i))
    else:
        cur = np.array([curves[i][1] - curves[i][0] for i in range(n)], float)
        mm = np.zeros(n, int)
        for _ in range(n_w):
            sv = np.clip(cur, 1e-12, None) ** (1.0 / max(tau, 1e-6))
            sv[mm >= m_max] = 0.0
            tot = sv.sum()
            if tot <= 0:
                break
            i = int(rng.choice(n, p=sv / tot))
            mm[i] += 1
            cur[i] = (curves[i][mm[i] + 1] - curves[i][mm[i]]) if mm[i] < m_max else 0.0
        m = [int(x) for x in mm]
    # eps floor: uniform coverage over tasks the weighted pass did not fund at all
    rest = [i for i in range(n) if m[i] == 0]
    take = min(n_eps, len(rest))
    for i in rng.permutation(len(rest))[:take]:
        m[rest[i]] += 1
    # BUDGET CONSERVATION IS AN INVARIANT, NOT AN APPROXIMATION. sum(m) must equal the row count
    # the trainer is configured to consume; a short parquet silently shortens the cycle. Top up on
    # the largest remaining marginals until it balances, and fail loudly if the pool cannot hold
    # the budget at all (n * m_max < budget). Cannot happen at 256 rows over 1023 tasks.
    while sum(m) < budget:
        cand = [i for i in range(n) if m[i] < m_max]
        if not cand:
            raise SystemExit(f"[triage] FATAL: pool of {n} tasks at m_max {m_max} cannot hold a "
                             f"budget of {budget} rows")
        for i in sorted(cand, key=lambda i: -(curves[i][m[i] + 1] - curves[i][m[i]])):
            m[i] += 1
            if sum(m) >= budget:
                break
    return m, take


def load_exclusions(path: str):
    """Read a JSONL of {scenario, task_idx, reason} -> (set of keys, Counter of reasons).

    Built by the failure-taxonomy certificates; see work/analysis/excluded_tasks.jsonl. Rows the
    parser cannot read are a build error in that file, not something to silently skip: an
    exclusion list that quietly loses half its rows produces an arm that is neither v2 nor v3, so
    this raises instead.
    """
    keys, reasons = set(), {}
    with open(path) as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)          # deliberately unguarded, see docstring
            keys.add((r["scenario"], int(r["task_idx"])))
            reasons[r.get("reason", "?")] = reasons.get(r.get("reason", "?"), 0) + 1
    return keys, reasons


def load_task_rho(path: str):
    """Read fit_task_rho.py's JSONL -> ({scenario::task_idx: rho}, summary dict).

    Same discipline as load_exclusions: a row this parser cannot read is a build error in that
    file, not something to skip. A rho file that quietly loses half its rows produces an arm that
    is part v3g and part TRIAGE-rho, wearing one name. rho is validated into (0,1) because
    nu = (1-rho)/rho has to be finite and positive for the closed form to mean anything.
    """
    rho, ng = {}, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)              # deliberately unguarded, see docstring
            v = float(r["rho"])
            if not (0.0 < v < 1.0):
                raise SystemExit(f"[triage] FATAL: rho {v} out of (0,1) in {path}")
            rho[f"{r['scenario']}::{int(r['task_idx'])}"] = v
            ng.append(int(r.get("n_groups", 0)))
    a = np.array(sorted(rho.values())) if rho else np.zeros(1)
    return rho, {"n_tasks": len(rho), "n_groups_total": int(sum(ng)),
                 "rho_min": round(float(a.min()), 5), "rho_max": round(float(a.max()), 5),
                 "rho_med": round(float(np.median(a)), 5),
                 "rho_iqr": round(float(np.percentile(a, 75) - np.percentile(a, 25)), 5)}


def band_of(s: float, f: float):
    n = s + f
    if n <= 0:
        return None
    p = s / n
    for name, lo, hi in BANDS:
        if lo <= p < hi:
            return name
    return BANDS[-1][0]


def read_new_episodes(path: str, offset: int):
    """Return (rows, new_offset). Only lines past `offset` are read: the file grows to hundreds of
    thousands of lines and every cycle would otherwise re-parse all of it. A file that SHRANK was
    rotated or replaced, so the offset is meaningless and everything is re-read."""
    if not os.path.exists(path):
        return [], offset
    rows, n = [], 0
    with open(path, errors="ignore") as fh:
        for i, line in enumerate(fh):
            n = i + 1
            if i < offset:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if n < offset:                      # truncated/rotated: re-read from the top next call
        return read_new_episodes(path, 0)
    return rows, n


def bank_counts(pattern: str, own_path: str, pool_keys: set):
    """(solves, failures) per task from every OTHER arm's episodes: the warm-start prior.

    Restricted to the pool's own tasks -- the bank spans several pools, and carrying counts for
    tasks this arm can never sample would bloat the state file for nothing. Validation episodes
    are dropped for the same reason they are dropped from live evidence: different pool, different
    surface, temperature 0.
    """
    own = os.path.realpath(own_path)          # run_<tag> is a symlink into scratch on every arm,
    files = [p for p in sorted(glob.glob(pattern))   # so compare resolved paths, not the globbed
             if os.path.realpath(p) != own]          # ones, or the arm's own file survives the cut
    c, n_used, n_val = {}, 0, 0
    # val solves/attempts are counted, not just skipped, because --warm-shrink capability derives
    # its lambda from them (base capability over bank capability) and both halves of that ratio
    # have to come from the same pass over the same files or the number is unattributable.
    val = [0.0, 0.0]
    for p in files:
        try:
            fh = open(p, errors="ignore")
        except Exception:
            continue
        with fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("split") == "val":
                    n_val += 1
                    try:
                        val[0 if float(r["reward"]) > 0 else 1] += 1.0
                    except Exception:
                        pass
                    continue
                try:
                    kk = f"{r['scenario']}::{r['task_idx']}"
                    good = float(r["reward"]) > 0
                except Exception:
                    continue
                if kk not in pool_keys:
                    continue
                v = c.setdefault(kk, [0.0, 0.0])
                v[0 if good else 1] += 1.0
                n_used += 1
    return c, files, n_used, n_val, val


def sample_without_replacement(keys, w, m, rng):
    """Efraimidis-Spirakis: successive sampling proportional to w, without replacement, in one
    pass. key_i = -log(u_i)/w_i and take the m smallest -- exact, and it does not degrade as the
    weights spread out the way repeated normalise-and-draw does."""
    if m <= 0:
        return []
    w = np.asarray(w, float)
    w = np.clip(w, 1e-12, None)
    u = rng.random(len(w))
    order = np.argsort(-np.log(np.clip(u, 1e-300, 1.0)) / w)
    return [keys[i] for i in order[:m]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True, help="THIS ARM'S OWN episodes.jsonl; never another arm's")
    ap.add_argument("--base-pool", required=True)
    ap.add_argument("--out-pool", required=True)
    ap.add_argument("--out-data", required=True, help="parquet directory to rebuild")
    ap.add_argument("--state", required=True, help="triage_state.json: counts, decay clock, read offset")
    ap.add_argument("--stats", required=True, help="triage_stats.jsonl: one mechanism line per cycle")
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--budget", type=int, required=True,
                    help="rows in the rebuilt parquet = steps_per_cycle * train_batch_size")
    # REQUIRED, and it used to default to 8. Nothing in production ever took that default --
    # coadapt.py has always passed --k from NROLL, and the composed verl config on every TRIAGE arm
    # reads actor_rollout_ref.rollout.n=5, so a8T/a8Te0/a8Tg1 all allocated at the k they generate
    # at. But the 8 sat in the file as the documented value of k and was copied into the offline
    # tooling (triage_report.py --nroll defaulted to 8 and chunked a8Te0's episode log into
    # 32x8=256-episode "steps" that no optimizer step ever was). A default that is wrong and
    # unreachable is worse than no default: it is only ever read by a human. There is exactly one
    # correct value here, the trainer knows it, so it must be passed.
    ap.add_argument("--vark", action="store_true",
                    help="TRIAGE v4: allocate a PER-TASK group size k_i in {2,3,5,8} under the "
                         "same rollout budget, instead of the uniform --k. Requires the trainer-"
                         "side VARK=1 patch; the arm aborts at cycle 1 if the trainer did not "
                         "honour the per-row k (see coadapt.py's histogram cross-check).")
    ap.add_argument("--k", type=int, required=True, help="GROUP SIZE IN FORCE = the trainer's "
                    "actor_rollout_ref.rollout.n (NROLL). g(p,k) is a function of it, so an arm "
                    "that raises NROLL and allocates on the old k optimises the wrong target")
    ap.add_argument("--eps", type=float, default=0.10, help="uniform mass; the floor that keeps "
                    "every task's probability strictly positive no matter what the weights say")
    ap.add_argument("--temp", type=float, default=1.0, help="tau: sample proportional to w^(1/tau). "
                    "1.0 = v1 (linear in the posterior gradient mass). w is bounded and flat in "
                    "the tail -- at k=5 a task with 20 straight failures still scores 0.19 against "
                    "a mid-band 0.88 -- so with ~500 evidenced-dead tasks against ~200 mid-band "
                    "ones, linear sampling gives the dead ones ~30%% of the budget. tau=0.5 "
                    "squares every ratio (4.6x -> 21x) and inverts that; eps is untouched, so the "
                    "coverage floor is unchanged.")
    ap.add_argument("--warm-bank", default=None, metavar="GLOB",
                    help="warm-start the per-task priors from the OTHER arms' episode logs. "
                         "'default' means " + DEFAULT_BANK + ". The arm's own file is always "
                         "excluded and stays the live-evidence source. Counts enter capped at "
                         "--warm-cap-s/--warm-cap-f so the arm's own rollouts overtake them in "
                         "~2 cycles, are read ONCE (the bank is a snapshot, not a subscription: "
                         "re-reading it every cycle would keep injecting fresh foreign evidence "
                         "and gamma could never age it out), and decay with gamma exactly like "
                         "live counts. Tasks absent from the bank keep Beta(1,1) and its optimism.")
    ap.add_argument("--warm-cap-s", type=float, default=20.0, help="cap on warm SOLVES per task")
    ap.add_argument("--warm-cap-f", type=float, default=40.0, help="cap on warm FAILURES per task. "
                    "Higher than the solve cap because the pool is failure-heavy and a task the "
                    "bank has never solved is the one claim the bank makes most reliably")
    ap.add_argument("--decay", type=float, default=0.9, help="gamma, applied to the counts at each "
                    "cycle rebuild BEFORE the new episodes are added")
    ap.add_argument("--min-obs", type=float, default=4.0, help="evidence needed before a task is "
                    "counted in a solve-rate band")
    # ---------------------------------------------------------------------------------- v3
    ap.add_argument("--exclude-tasks", default=None, metavar="JSONL",
                    help="TRIAGE v3: a JSONL of {scenario, task_idx, reason} whose tasks are "
                         "removed from the TRAINING allocation entirely -- w is never computed "
                         "for them and they are not in the set the eps-uniform floor draws from. "
                         "For certificates only (a verifier that cannot pass; a verifier that "
                         "passes on a no-op), never for difficulty: eps exists to hedge a wrong "
                         "posterior, and a wrong posterior is not what these are. The evaluation "
                         "and validation pools are untouched, so every held-out comparison keeps "
                         "the pool it always had.")
    ap.add_argument("--warm-shrink", choices=["none", "capability", "group"], default="none",
                    help="TRIAGE v3: how much of the bank's claim to believe. "
                         "'none' = v2. "
                         "'group' = keep the mean, drop the pretence that the k rollouts of a "
                         "group are independent: score with the exchangeable Beta-Binomial "
                         "P(all agree) at concentration --shrink-nu. This is what a8T3 runs; the "
                         "measured within-group correlation is rho=0.78 and ignoring it is what "
                         "made pred_degenerate read 0.24 against a measured 0.88. "
                         "'capability' = scale each task's WARM prior mean by lambda, keeping its "
                         "evidence count, on the theory that bank rates come from trained "
                         "policies and a fresh arm is weaker. Implemented, measured, and NOT "
                         "selected: on each arm's own cycle-1 batch the observed solve rate over "
                         "the prior's prediction is 0.94-1.20, so lambda is 1 and the 0.484 that "
                         "the val/bank ratio suggests is a comparison of two different pools at "
                         "two different temperatures. Kept as the ablation that shows it.")
    ap.add_argument("--shrink-nu", type=float, default=0.285,
                    help="nu = (1-rho)/rho for --warm-shrink group. 0.285 is rho=0.78, fitted by "
                         "matching the degenerate fraction over 1920 groups (a8T2/a8T2r/a8T2g1/"
                         "a8T2w, cycles 1-2). Out of sample it predicts a8F 0.8989 against a "
                         "measured 0.8741 and a8Te0 0.8988 against 0.9062. Larger nu = weaker "
                         "correlation; nu -> inf is exactly v2.")
    ap.add_argument("--rule", choices=["gradmass", "bandfilter", "vip", "progress", "budget"],
                    default="gradmass",
                    help="THE WEIGHT RULE, and the only thing a8T2n changes about a8T2. "
                         "'gradmass' is TRIAGE: a Beta posterior per task, weight = the posterior "
                         "expected probability that a group of k carries gradient, sampled at "
                         "temperature tau. 'bandfilter' is the naive difficulty-band baseline: "
                         "point estimate p_hat = s/(s+f) on the SAME decayed counts from the SAME "
                         "warm bank, keep --band-lo <= p_hat <= --band-hi, include every task "
                         "with no evidence at all, and sample UNIFORMLY inside the kept set. No "
                         "posterior, no g(p,k), no temperature, no optimism -- which is the "
                         "point: if a8T2n matches a8T2 then none of that machinery is "
                         "load-bearing and the paper has to say so. "
                         "'vip' is the PUBLISHED VIP baseline (arXiv:2602.01601, ICLR 2026): "
                         "weight = E[p(1-p)], the posterior form of the only task-dependent factor "
                         "in its Var(G~_q), whose ordering its own Theorems 5.1/5.2 make identical "
                         "to the ordering of its optimal n*_q. Refuses --exclude-tasks, because "
                         "L>=3 means VIP cannot skip a prompt. 'progress' is the LEARNING-PROGRESS "
                         "curriculum (Graves 2017 / TSCL): weight = |delta p^| across the two most "
                         "recent evidence windows, mean weight where there is no second window "
                         "yet. See the v4 block for every deviation from the published forms. "
                         "'budget' is TRIAGE-B: the SAME posterior, allocating the ROLLOUT BUDGET "
                         "rather than the task list -- task i gets k_i = k*m_i rollouts with "
                         "sum_i m_i = the row budget, so total compute is unchanged. See the "
                         "TRIAGE-B block, including the measurement that at the matched budget on "
                         "this pool the optimal allocation IS fixed-k selection.")
    # ------------------------------------------------------------------------------ TRIAGE-B
    ap.add_argument("--k-budget-max", type=int, default=3, metavar="M",
                    help="TRIAGE-B: the largest number of ROWS (groups of k) one task may take, so "
                         "k_max = k * this. 3 gives k_i in {0,5,10,15} at k=5. Concentration "
                         "overfit is a measured failure channel on this pool "
                         "(work/analysis/concentration_overfit.md), and the closed form's "
                         "alternating coefficients start to cancel past k*m ~ 30, so this is "
                         "capped rather than free.")
    ap.add_argument("--budget-objective", choices=["any", "group", "count"], default="any",
                    help="TRIAGE-B: what m units are WORTH to a task. 'any' = 1 - E[q_k^m], the "
                         "probability of at least one gradient-carrying group out of m independent "
                         "groups of k -- the only one of the three that matches what this verl "
                         "checkout generates (one fresh uuid per row, so m rows are m groups). "
                         "'group' = w(k*m), one group of k*m: the form the design was written in, "
                         "correct only under a verl that groups m rows on a shared uid. 'count' = "
                         "m*w(k), the expected NUMBER of live groups: exactly linear, so its "
                         "optimum is the concentration extreme (m_max to the top N/m_max tasks). "
                         "All three are the same beta_moment closed form.")
    ap.add_argument("--budget-draw", choices=["greedy", "sample"], default="greedy",
                    help="TRIAGE-B: 'greedy' is water-filling on the marginals -- exactly optimal "
                         "for the concave objectives, and deterministic. 'sample' uses the same "
                         "marginals as sampling weights at --temp, one unit at a time; it is the "
                         "variant that still moves when the pool holds more first-look-worthy "
                         "tasks than the cycle has rows (measured: greedy is inert at N=256 on "
                         "this pool, sample allocates 36 tasks to k=10 and 2 to k=15).")
    ap.add_argument("--band-lo", type=float, default=0.20, help="bandfilter: keep p_hat >= this")
    ap.add_argument("--band-hi", type=float, default=0.80, help="bandfilter: keep p_hat <= this")
    ap.add_argument("--lp-min-obs", type=float, default=1.0,
                    help="rule progress: observations a window needs before it counts as a point "
                         "on the learning curve. 1.0 = any evidence at all, which is the noisiest "
                         "and most faithful reading of TSCL's slope (it takes the slope through "
                         "whatever it just observed); raise it to trade responsiveness for noise.")
    ap.add_argument("--task-rho", default=None, metavar="JSONL",
                    help="TRIAGE-rho: a JSONL of {scenario, task_idx, rho} built by "
                         "fit_task_rho.py from the banked GROUPS, giving each task its own "
                         "within-group correlation. Only meaningful with --estimator taskrho, "
                         "which is refused without it. Tasks absent from the file fall back to "
                         "the global --shrink-nu exactly, so the file is a refinement of v3g and "
                         "not a replacement for it.")
    ap.add_argument("--estimator", choices=["posterior", "point", "taskrho"], default="posterior",
                    help="HOW w IS COMPUTED FROM THE SAME POSTERIOR, and the comparison the "
                         "novelty claim rests on. 'posterior' = E[g(p,k)] under Beta(1+s,1+f), "
                         "which is TRIAGE and is byte-identical to v1/v2/v3. 'point' = the "
                         "TRACE-style plug-in g(p^,k) at the posterior mean p^ = a/(a+b) -- their "
                         "value net predicts a point v_i and raises it to the k-th power. "
                         "'taskrho' is TRIAGE-rho: the SAME closed form at a PER-TASK "
                         "concentration nu_i = (1-rho_i)/rho_i read from --task-rho, i.e. "
                         "w = 1 - P(all k solve) - P(all k fail) under "
                         "BetaBin(k, nu_i*p_hat, nu_i*(1-p_hat)). It is the only rule in this "
                         "file that is not a function of p alone, and therefore the only one "
                         "that is not rank-equivalent to VIP at fixed k. Applies "
                         "to --rule gradmass only: under the other rules the sampler is that "
                         "rule's own score and this would silently change nothing that matters.")
    ap.add_argument("--shrink-lambda", type=float, default=0.0,
                    help="lambda for --warm-shrink capability. 0 = derive it from the same bank "
                         "pass: (pooled val solve rate) / (pooled train solve rate on pool "
                         "tasks), which reads 0.4887 on the current bank.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-parquet", action="store_true", help="offline simulation: do everything "
                    "except invoke prep_awm")
    a = ap.parse_args()

    # A flag that silently does nothing is how an arm ends up wearing a name it is not running.
    if a.warm_shrink == "capability" and not a.warm_bank:
        print("[triage] FATAL: --warm-shrink capability rescales the WARM prior; without "
              "--warm-bank there is no warm prior to rescale.", flush=True)
        return 1
    # VIP'S CAN'T-EXCLUDE PROPERTY, ENFORCED RATHER THAN DOCUMENTED. Its feasible set is
    # n_q in {L,...,U} with L >= 3, so no prompt can be given zero rollouts; a VIP arm carrying an
    # exclusion list would be a strictly stronger method than the paper it is named after, which is
    # the one direction a baseline must never be wrong in.
    if a.rule == "vip" and a.exclude_tasks:
        print("[triage] FATAL: --rule vip cannot take --exclude-tasks. VIP's constraint is "
              "L <= n_q <= U with L >= 3, so it structurally cannot skip a prompt; a VIP baseline "
              "that refuses tasks is not VIP.", flush=True)
        return 1
    # A flag that silently does nothing is how an arm ends up wearing a name it is not running: the
    # estimator changes w, and w is what the SAMPLER reads only under rule gradmass.
    if a.estimator != "posterior" and a.rule != "gradmass":
        print(f"[triage] FATAL: --estimator {a.estimator} changes the gradmass weight, but "
              f"--rule {a.rule} samples on its own score; the flag would not change the "
              f"allocation.", flush=True)
        return 1
    # TRIAGE-rho guards. Same discipline as everywhere above: a flag that silently does nothing,
    # and a composition whose halves quietly disagree, are refused at argv rather than discovered
    # in a log four hours into a run.
    if a.estimator == "taskrho":
        if not a.task_rho:
            print("[triage] FATAL: --estimator taskrho needs --task-rho FILE. Without it every "
                  "task would fall back to the global --shrink-nu and the arm would be running "
                  "--estimator point under the TRIAGE-rho name.", flush=True)
            return 1
        if a.warm_shrink != "group":
            print(f"[triage] FATAL: --estimator taskrho is the GROUP model at a per-task "
                  f"concentration; --warm-shrink {a.warm_shrink} switches the group model off, so "
                  f"there is no concentration for rho_i to replace. Use --warm-shrink group.",
                  flush=True)
            return 1
    elif a.task_rho:
        print(f"[triage] FATAL: --task-rho is only read by --estimator taskrho; under "
              f"--estimator {a.estimator} it would not change the allocation.", flush=True)
        return 1
    # TRIAGE-B guards. Same discipline: a flag that silently does nothing, or a composition whose
    # two halves quietly disagree, is refused at argv rather than discovered in a log.
    if a.rule == "budget":
        if a.k_budget_max < 1:
            print("[triage] FATAL: --k-budget-max must be >= 1 (a funded task gets k_min = k "
                  "rollouts).", flush=True)
            return 1
        if a.k * a.k_budget_max > 30:
            print(f"[triage] FATAL: k*--k-budget-max = {a.k * a.k_budget_max} > 30. The value "
                  f"curve is 1 - E[q_k^m] with q_k a degree-k polynomial; past ~30 the alternating "
                  f"coefficients cancel and the closed form stops agreeing with Monte Carlo.",
                  flush=True)
            return 1
        if a.budget_draw == "greedy" and abs(a.temp - 1.0) > 1e-12:
            print(f"[triage] FATAL: --budget-draw greedy is a deterministic argmax over marginals; "
                  f"--temp {a.temp} would not change the allocation. Use --budget-draw sample or "
                  f"drop --temp.", flush=True)
            return 1
        if os.environ.get("TRIAGE_DECOR"):
            # awm_agent_loop.py recovers verl's rollout index as a counter over CONSECUTIVE
            # rollouts of one (scenario, task_idx). Row multiplicity puts k*m_i consecutive
            # rollouts of one task in a batch against a permutation family sized k.
            print("[triage] FATAL: --rule budget is not composable with TRIAGE_DECOR. Row "
                  "multiplicity breaks the rollout-index counter the decorrelation permutation "
                  "is keyed on.", flush=True)
            return 1
    elif (a.k_budget_max != 3 or a.budget_objective != "any" or a.budget_draw != "greedy"):
        print(f"[triage] FATAL: --k-budget-max/--budget-objective/--budget-draw only mean "
              f"anything to --rule budget; --rule {a.rule} would ignore them.", flush=True)
        return 1

    base = json.load(open(a.base_pool))
    tasks = [(p["scenario"], p["task_idx"]) for p in base]

    # CERTIFIED EXCLUSION, APPLIED HERE AND NOWHERE ELSE. Filtering the candidate list at the top
    # is what makes the exclusion HARD: every downstream object -- the weights, the bands, the
    # without-replacement draw AND the eps-uniform draw, which samples from `rest` and would
    # otherwise be the one path that puts a certified-dead task back in the batch -- is built from
    # this list. `base` is left whole so the parquet rows are still looked up from the original
    # pool records.
    n_pool_full, excl_keys, excl_reasons = len(tasks), set(), {}
    if a.exclude_tasks:
        excl_keys, excl_reasons = load_exclusions(a.exclude_tasks)
        tasks = [t for t in tasks if t not in excl_keys]
        n_ex = n_pool_full - len(tasks)
        if not tasks:
            print("[triage] FATAL: the exclusion list removed every task in the pool", flush=True)
            return 1
        print(f"[triage] excluded {n_ex}/{n_pool_full} pool tasks from the training allocation "
              f"({len(excl_keys)} rows in {a.exclude_tasks}, reasons {excl_reasons}); "
              f"{len(tasks)} tasks remain. Eval/val pools are untouched.", flush=True)
    key = {t: f"{t[0]}::{t[1]}" for t in tasks}

    st = {"counts": {}, "consumed_lines": 0, "cycles_built": []}
    if os.path.exists(a.state):
        try:
            st = json.load(open(a.state))
        except Exception:
            print(f"[triage] state at {a.state} unreadable; starting from no evidence", flush=True)
    st.setdefault("counts", {})
    st.setdefault("consumed_lines", 0)
    st.setdefault("cycles_built", [])

    # A cycle already allocated is never allocated twice. The driver replays cycle 1..n on every
    # relaunch (each replayed cycle is a no-op because the checkpoint is already past its target),
    # so without this the decay would fire once per replay and the posterior would be reset -- an
    # arm that crashed five times would be running a different method from one that did not.
    if a.cycle in st["cycles_built"] and os.path.exists(os.path.join(a.out_data, "train.parquet")):
        print(f"[triage] cycle {a.cycle} already allocated; parquet left as it is", flush=True)
        return 0

    counts = {k: [float(v[0]), float(v[1])] for k, v in st["counts"].items()}

    # DECAY FIRST, then this cycle's evidence. Order is the method: the new episodes are the ones
    # generated under the current policy and must not be discounted by the same step that ages the
    # ones generated under the old one.
    for v in counts.values():
        v[0] *= a.decay
        v[1] *= a.decay

    # WARM START, ONCE, AFTER THE DECAY OF THIS CYCLE AND BEFORE ITS LIVE EVIDENCE.
    #   after the decay  -- so the caps mean what they say on the cycle they are applied (a_i =
    #                       1 + min(s_bank, 20) exactly, not 0.9x that);
    #   once             -- st["warm"]["applied_cycle"] is the guard. From here on the warm counts
    #                       are ordinary counts: gamma ages them at the same rate as the arm's own
    #                       observations, which is the whole reason foreign evidence is safe to
    #                       use. Re-reading the growing bank each cycle would defeat that;
    #   into counts      -- not into a separate prior term, because those are the same thing:
    #                       Beta(1 + warm + live) is what both spellings produce, and one dict
    #                       means the decay, the bands and the state file cannot disagree.
    warm = st.get("warm") or {}
    lam = None
    if a.warm_bank and not warm.get("applied_cycle"):
        pattern = DEFAULT_BANK if a.warm_bank == "default" else a.warm_bank
        bc, files, n_used, n_skipped, val = bank_counts(pattern, a.episodes, set(key.values()))
        # CAPABILITY SHRINK, if asked for: scale each task's warm prior MEAN by lambda while
        # holding its evidence count, so the caps still mean what they say and only the location
        # moves. Applied to the capped counts, before they become ordinary counts -- from the next
        # cycle on they are indistinguishable from the arm's own observations and gamma ages them
        # identically, which is the property that makes foreign evidence safe to use at all.
        if a.warm_shrink == "capability":
            lam = a.shrink_lambda
            if lam <= 0:
                bs, bf = sum(v[0] for v in bc.values()), sum(v[1] for v in bc.values())
                base_rate = val[0] / max(val[0] + val[1], 1.0)
                bank_rate = bs / max(bs + bf, 1.0)
                lam = base_rate / bank_rate if bank_rate > 0 else 1.0
                print(f"[triage] capability shrink: lambda = base {base_rate:.4f} / bank "
                      f"{bank_rate:.4f} = {lam:.4f}, derived from this same bank pass "
                      f"({int(val[0] + val[1])} val, {int(bs + bf)} train episodes)", flush=True)
            else:
                print(f"[triage] capability shrink: lambda = {lam:.4f} (given)", flush=True)
        n_warm = 0
        for kk, (s, f) in bc.items():
            s, f = min(s, a.warm_cap_s), min(f, a.warm_cap_f)
            if s + f <= 0:
                continue
            if lam is not None:
                n = s + f
                s = min(max(lam * (s / n), 0.0), 1.0) * n
                f = n - s
            c = counts.setdefault(kk, [0.0, 0.0])
            c[0] += s
            c[1] += f
            n_warm += 1
        warm = {"applied_cycle": a.cycle, "n_tasks": n_warm, "pattern": pattern,
                "n_files": len(files), "n_episodes": n_used,
                "cap_s": a.warm_cap_s, "cap_f": a.warm_cap_f}
        if lam is not None:
            warm["shrink"] = "capability"
            warm["lambda"] = round(float(lam), 6)
        print(f"[triage] warm start: {n_warm}/{len(tasks)} pool tasks primed from {len(files)} "
              f"foreign episode logs ({n_used} train episodes, {n_skipped} val skipped), "
              f"capped at ({a.warm_cap_s:g}s, {a.warm_cap_f:g}f)", flush=True)
    elif a.warm_bank:
        print(f"[triage] warm start already applied at cycle {warm['applied_cycle']}; "
              f"{warm.get('n_tasks', 0)} primed tasks now decayed by gamma^"
              f"{a.cycle - int(warm['applied_cycle'])}", flush=True)

    # THE TWO WINDOWS THE LEARNING-PROGRESS RULE NEEDS, taken here because here is the one moment
    # they are separable at no cost: `counts` at this line is everything the arm knew BEFORE this
    # cycle, already gamma-aged (and warm-primed on cycle 1), and the loop below is this cycle's
    # own rollouts. Snapshotting rather than storing a second history keeps the state file and the
    # decay identical for every other rule -- nothing below this line is reachable unless
    # --rule progress asked for it.
    prev_counts = ({k: (v[0], v[1]) for k, v in counts.items()} if a.rule == "progress" else {})
    cur_counts: dict = {}

    rows, new_off = read_new_episodes(a.episodes, int(st["consumed_lines"]))
    n_val, n_add = 0, 0
    for r in rows:
        # VALIDATION EPISODES ARE NOT EVIDENCE. They are generated on a different (fixed) surface,
        # at temperature 0, on a held-out pool -- counting them would price training tasks by a
        # policy's greedy behaviour in an environment it is not training in.
        if r.get("split") == "val":
            n_val += 1
            continue
        try:
            kk = f"{r['scenario']}::{r['task_idx']}"
            good = float(r["reward"]) > 0
        except Exception:
            continue
        c = counts.setdefault(kk, [0.0, 0.0])
        c[0 if good else 1] += 1.0
        n_add += 1
        if a.rule == "progress":
            cc = cur_counts.setdefault(kk, [0.0, 0.0])
            cc[0 if good else 1] += 1.0

    # w. Under --warm-shrink group the SAME posterior is scored against a group model that knows
    # the k rollouts are correlated instead of one that assumes they are not; `nu is None` below
    # is the v1/v2 path and is left calling grad_mass so it stays bit-identical rather than
    # merely equal-in-the-limit.
    nu = a.shrink_nu if a.warm_shrink == "group" else None
    # TRIAGE-rho: the per-task concentration, read ONCE from the pinned file. Guarded above to be
    # reachable only under --estimator taskrho + --warm-shrink group, so `nu` is never None here.
    task_rho, rho_meta, rho_used = {}, {}, []
    if a.estimator == "taskrho":
        task_rho, rho_meta = load_task_rho(a.task_rho)
        n_hit = sum(1 for t in tasks if key[t] in task_rho)
        print(f"[triage] estimator=taskrho: {n_hit}/{len(tasks)} pool tasks carry a fitted rho "
              f"from {a.task_rho} ({rho_meta['n_groups_total']} banked groups, rho in "
              f"[{rho_meta['rho_min']:.4f},{rho_meta['rho_max']:.4f}] median "
              f"{rho_meta['rho_med']:.4f} IQR {rho_meta['rho_iqr']:.4f}); the other "
              f"{len(tasks) - n_hit} fall back to the global rho "
              f"{1.0/(1.0+nu):.4f}", flush=True)
    w = {}
    for t in tasks:
        s, f = counts.get(key[t], (0.0, 0.0))
        if a.estimator == "point":
            # TRACE's plug-in on the SAME posterior and the SAME group model; only E[g(p)] ->
            # g(E[p]) changes, which is the one thing this ablation is allowed to change.
            w[t] = grad_mass_point(s, f, a.k, nu)
        elif a.estimator == "taskrho":
            # THE ONE LINE THAT IS TRIAGE-rho. Same p_hat, same closed form, same group model --
            # only the CONCENTRATION becomes the task's own. A task with no fitted rho takes the
            # global one, which makes this weight identical to --estimator point for that task and
            # makes the whole rule identical to v3g's plug-in when the file is empty.
            r = task_rho.get(key[t], 1.0 / (1.0 + nu))
            rho_used.append(r)
            w[t] = grad_mass_point(s, f, a.k, (1.0 - r) / r)
        else:
            w[t] = grad_mass(s, f, a.k) if nu is None else grad_mass_group(s, f, a.k, nu)

    wv = np.array([w[t] for t in tasks], float)
    # THE SAMPLING WEIGHTS, which are w^(1/tau) and not w. At tau=1 this is x**1.0, exactly x in
    # IEEE754, so every number below and the allocation itself are bit-identical to v1 -- that is
    # what lets the three arms already in flight keep running against this file. Entropy and ESS
    # are reported on THESE weights because they describe the distribution actually drawn from;
    # w_min/w_max/w_mean stay on the raw posterior gradient mass, which is what those names mean.
    sv = wv ** (1.0 / max(a.temp, 1e-6))
    n_kept = None
    rule_extra: dict = {}
    if a.rule == "vip":
        # VIP. sv -- what is drawn from -- becomes the posterior form of the only task-dependent
        # factor in its Var(G~_q). w is deliberately NOT replaced (same argument as bandfilter):
        # it stays the posterior gradient mass so pred_degenerate_frac, w_mean_chosen and the band
        # mix mean on this arm exactly what they mean on a8T3g. An arm whose stats line is not
        # comparable to the arm it is a baseline for is not a baseline.
        vv = np.array([vip_variance(*counts.get(key[t], (0.0, 0.0))) for t in tasks], float)
        sv = vv ** (1.0 / max(a.temp, 1e-6))
        rule_extra = {"vip_score_min": round(float(vv.min()), 6),
                      "vip_score_max": round(float(vv.max()), 6),
                      "vip_score_mean": round(float(vv.mean()), 6)}
        print(f"[triage] rule=vip: E[p(1-p)] over {len(tasks)} tasks in "
              f"[{vv.min():.5f},{vv.max():.5f}] mean {vv.mean():.5f}; no task is zero-weighted "
              f"(VIP's L>=3 cannot skip a prompt)", flush=True)
    elif a.rule == "progress":
        # LEARNING PROGRESS. |delta p^| between the gamma-aged older window and this cycle's own
        # rollouts. Tasks without both windows take the MEAN of the measured values: explore-
        # neutral by construction, so the rule neither inherits TRIAGE's optimism nor punishes a
        # task for not having been sampled yet.
        lp = [learning_progress(prev_counts.get(key[t], (0.0, 0.0)),
                                cur_counts.get(key[t], (0.0, 0.0)), a.lp_min_obs) for t in tasks]
        meas = [x for x in lp if x is not None]
        mean_lp = float(np.mean(meas)) if meas else 0.0
        cold = not meas or mean_lp <= 0.0
        if cold:
            # NO SIGNAL ANYWHERE -- cycle 1 always, and any cycle where nothing moved. Uniform is
            # the honest answer and it is what "everyone gets the mean" degenerates to; spelled out
            # so the draw cannot silently become 0/0 in the normaliser.
            sv = np.ones(len(tasks), float)
            print(f"[triage] rule=progress: {len(meas)}/{len(tasks)} tasks have two evidence "
                  f"windows and mean |dp| is {mean_lp:.5f} -- no progress signal yet, so the draw "
                  f"is UNIFORM over the pool", flush=True)
        else:
            sv = np.array([mean_lp if x is None else x for x in lp], float) ** (
                1.0 / max(a.temp, 1e-6))
            print(f"[triage] rule=progress: {len(meas)}/{len(tasks)} tasks have two evidence "
                  f"windows; |dp| mean {mean_lp:.5f} max {max(meas):.5f}, "
                  f"{sum(1 for x in meas if x <= 0.0)} measured at zero progress; the other "
                  f"{len(tasks) - len(meas)} take the mean", flush=True)
        rule_extra = {"lp_n_two_windows": len(meas), "lp_mean": round(mean_lp, 6),
                      "lp_max": round(float(max(meas)), 6) if meas else 0.0,
                      "lp_uniform_fallback": bool(cold), "lp_min_obs": a.lp_min_obs}
    elif a.rule == "bandfilter":
        # THE NAIVE BASELINE. sv -- what is drawn from -- becomes a 0/1 band indicator on the
        # point estimate, so the draw is uniform inside the kept set and tau is inert on it
        # (0**x and 1**x are 0 and 1). w is deliberately NOT replaced: it stays the posterior
        # gradient mass so that pred_degenerate, pred_degenerate_frac, w_mean_chosen and the band
        # mix mean the same thing on this arm as on every other one. An arm whose stats line is
        # not comparable to the arm it is a baseline for is not a baseline.
        keep = []
        for t in tasks:
            s, f = counts.get(key[t], (0.0, 0.0))
            n = s + f
            # no evidence at all -> IN. The naive rule has to survive its own cold start; giving
            # it TRIAGE's optimism instead would be handing it the thing under test.
            keep.append(1.0 if n <= 0 else (1.0 if a.band_lo <= s / n <= a.band_hi else 0.0))
        sv = np.array(keep, float)
        n_kept = int(sv.sum())
        print(f"[triage] rule=bandfilter: p_hat in [{a.band_lo:g},{a.band_hi:g}] keeps {n_kept}"
              f"/{len(tasks)} tasks (no-evidence tasks kept); uniform inside the kept set",
              flush=True)
        if n_kept == 0:
            print("[triage] FATAL: the band filter kept no tasks", flush=True)
            return 1
        if n_kept < int(round((1.0 - a.eps) * min(a.budget, len(tasks)))):
            # The Efraimidis-Spirakis draw clips weights to 1e-12 rather than 0, so a kept set
            # smaller than the weighted half of the budget would quietly start returning band
            # REJECTS instead of failing. Say so in the log rather than let the arm drift.
            print(f"[triage] WARNING: kept set {n_kept} is smaller than the weighted budget; "
                  f"the draw will fall back onto band-rejected tasks", flush=True)
    n_evid = sum(1 for t in tasks if sum(counts.get(key[t], (0.0, 0.0))) > 0)
    p = sv / sv.sum()
    w_entropy = float(-(p * np.log(np.clip(p, 1e-300, None))).sum())
    w_ess = float(sv.sum() ** 2 / (sv ** 2).sum())

    band_counts = {name: 0 for name, _, _ in BANDS}
    for t in tasks:
        s, f = counts.get(key[t], (0.0, 0.0))
        if s + f >= a.min_obs:
            band_counts[band_of(s, f)] += 1

    # ---------------------------------------------------------------- the allocation itself
    rng = np.random.default_rng(a.seed * 1_000_003 + a.cycle)
    if a.rule == "budget":
        # TRIAGE-B. The decision variable is m_i, the number of ROWS (= groups of k) task i takes,
        # under sum_i m_i = a.budget exactly -- so the rollout budget k*a.budget is the same number
        # every other arm generates. `chosen` carries task i m_i times; the parquet row multiplicity
        # IS the allocation, and verl's per-row uuid makes each copy its own advantage group.
        curves = [budget_curve(*counts.get(key[t], (0.0, 0.0)), a.k, a.k_budget_max, nu,
                               a.budget_objective) for t in tasks]
        m_alloc, n_eps_rows = allocate_budget(tasks, curves, a.budget, a.k_budget_max, a.eps,
                                              a.budget_draw, a.temp, rng)
        assert sum(m_alloc) == a.budget, (sum(m_alloc), a.budget)
        chosen = [t for i, t in enumerate(tasks) for _ in range(m_alloc[i])]
        eps_chosen = [None] * n_eps_rows          # counted, not identified: the floor is rows
        m_hist = {mm: int(sum(1 for x in m_alloc if x == mm))
                  for mm in range(a.k_budget_max + 1)}
        n_funded = int(sum(1 for x in m_alloc if x > 0))
        obj_val = float(sum(curves[i][m_alloc[i]] for i in range(len(tasks))))
        rule_extra = {"budget_objective": a.budget_objective, "budget_draw": a.budget_draw,
                      "k_budget_max": a.k_budget_max,
                      "alloc_hist_m": m_hist,
                      "alloc_hist_k": {str(a.k * mm): m_hist[mm] for mm in m_hist},
                      "n_tasks_funded": n_funded, "n_rollouts": a.k * a.budget,
                      "alloc_objective": round(obj_val, 5),
                      "alloc_max_m": int(max(m_alloc))}
        print(f"[triage] rule=budget ({a.budget_objective}/{a.budget_draw}): B = k*N = {a.k}*"
              f"{a.budget} = {a.k * a.budget} rollouts over {n_funded} funded tasks; "
              f"allocation histogram k_i -> #tasks "
              + " ".join(f"{a.k*mm}:{m_hist[mm]}" for mm in sorted(m_hist))
              + f"; objective {obj_val:.4f}", flush=True)
        if max(m_alloc) <= 1:
            print(f"[triage] NOTE: no task took more than one group -- at this budget the "
                  f"allocation IS fixed-k selection of the top {n_funded} tasks. Expected on this "
                  f"pool (see the TRIAGE-B block): supply of first-look-worthy tasks exceeds the "
                  f"row budget.", flush=True)
    else:
        budget = min(a.budget, len(tasks))
        n_eps = int(round(a.eps * budget))
        n_w = budget - n_eps
        chosen = sample_without_replacement(tasks, list(sv), n_w, rng)
        rest = [t for t in tasks if t not in set(chosen)]
        idx = rng.permutation(len(rest))[:n_eps]
        eps_chosen = [rest[i] for i in idx]
        chosen = chosen + eps_chosen
    if a.rule != "budget" and a.budget > len(tasks):
        # Budget larger than the pool: every task is in, and the surplus is drawn WITH replacement
        # proportional to w. Cannot happen at 256 rows against 1127 tasks; here so that a future
        # horizon change fails by repeating tasks rather than by silently shrinking the batch.
        extra = a.budget - len(tasks)
        pr = sv / sv.sum()
        chosen = chosen + [tasks[i] for i in rng.choice(len(tasks), size=extra, p=pr)]

    chosen_mix = {name: 0 for name, _, _ in BANDS}
    chosen_mix["low_obs"] = 0
    for t in chosen:
        s, f = counts.get(key[t], (0.0, 0.0))
        if s + f >= a.min_obs:
            chosen_mix[band_of(s, f)] += 1
        else:
            chosen_mix["low_obs"] += 1

    by = {(q["scenario"], q["task_idx"]): q for q in base}
    vark_hist, vark_sum, vark_rows_per_block = None, None, None
    if getattr(a, "vark", False):
        # TRIAGE v4. Reorders `chosen` into per-step blocks and attaches k_i to each pool entry;
        # prep_awm.py carries the column into the parquet and the patched trainer repeats each row
        # k_i times. sum(k_i) per block is pinned to the fixed-k trajectory total so the batch
        # shape the trainer sees is unchanged.
        vark_rows_per_block = int(os.environ.get("BSZ", "32"))
        chosen, kvals, per_block, n_blocks = vark_allocate(
            chosen, counts, key, w, nu, a.k, vark_rows_per_block, rng)
        out = [dict(by[t], vark_k=int(kk)) for t, kk in zip(chosen, kvals)]
        vark_hist = {int(kk): int(kvals.count(kk)) for kk in VARK_CHOICES if kvals.count(kk)}
        vark_sum = int(sum(kvals))
        bad = [b for b in range(n_blocks)
               if sum(kvals[b * vark_rows_per_block:(b + 1) * vark_rows_per_block]) != per_block]
        if bad or vark_sum != n_blocks * per_block:
            raise ValueError(f"[triage] vark: block budget violated in blocks {bad}; "
                             f"sum k = {vark_sum} against {n_blocks * per_block}")
        print(f"[triage] vark: {len(chosen)} rows in {n_blocks} blocks of {vark_rows_per_block}; "
              f"k histogram {vark_hist}; sum k = {vark_sum} "
              f"(= {n_blocks} x {per_block}, identical to uniform k={a.k}); "
              f"mean k {vark_sum / len(chosen):.3f}", flush=True)
    else:
        out = [by[t] for t in chosen]
    os.makedirs(os.path.dirname(a.out_pool), exist_ok=True)
    json.dump(out, open(a.out_pool, "w"))

    wsel = [w[t] for t in chosen]
    # THE ONE-LINE SELF-CHECK. pred_degenerate is the plug-in degenerate probability
    # p^k + (1-p)^k at each chosen task's posterior MEAN, averaged over the batch. Printed every
    # cycle, it catches a broken allocator in one line: one that has silently reverted to uniform
    # prints ~0.55 on this pool, one that collapsed onto dead tasks prints ~1.0, and a working v2
    # allocation prints ~0.24. It is a RELATIVE instrument -- READ IT AGAINST 0.55, NOT AGAINST
    # THE MEASURED DEGENERATE FRACTION -- because every plug-in of this form is optimistic, by
    # three effects that were measured here rather than assumed (all on a8F, whose allocation IS
    # uniform over this pool and whose episode log measures 0.8741 degenerate at k=5):
    #     plug-in at capped posterior means (this field)          0.5523
    #     plug-in at cross-arm pooled empirical rates             0.7223   (+0.170: cap compression)
    #     plug-in at a8F's OWN empirical rates                    0.7473   (+0.025: cross-arm pooling)
    #     MEASURED                                                0.8741   (+0.127: within-group
    #         correlation -- k rollouts of one group share a policy checkpoint and a prompt, so
    #         they agree more often than k iid draws at the same p. This is the big one, it is a
    #         property of the environment and not of the allocator, and no plug-in can see it.)
    # So the field runs ~1.58x below what triage_report.py will measure on the same batch. A v2
    # cycle printing 0.24 forecasts a measured ~0.38-0.57 against the uniform baseline's 0.87.
    # pred_degenerate_frac below is the same object under the full posterior (1 - E[g]) rather
    # than at its mean; it is kept unchanged from v1 and carries the same three biases.
    phat = []
    for t in chosen:
        s, f = counts.get(key[t], (0.0, 0.0))
        phat.append((1.0 + s) / (2.0 + s + f))
    pred_deg = float(np.mean([q ** a.k + (1.0 - q) ** a.k for q in phat]))
    dead_frac = float(np.mean([q < 0.05 for q in phat]))
    rec = {
        "cycle": a.cycle, "k": a.k, "eps": a.eps, "gamma": a.decay, "budget": a.budget,
        "tau": a.temp, "warm_tasks": int(warm.get("n_tasks", 0)),
        "pred_degenerate": round(pred_deg, 5),
        # fraction of the budget spent on tasks whose posterior mean solve rate is under 5%: the
        # suppression the temperature exists to buy, measured directly on the chosen batch
        "chosen_dead_frac": round(dead_frac, 5),
        "n_tasks_pool": len(tasks), "n_tasks_with_evidence": n_evid,
        "n_episodes_new": n_add, "n_episodes_val_skipped": n_val,
        "band_counts": band_counts, "n_banded": int(sum(band_counts.values())),
        "w_entropy": round(w_entropy, 5), "w_entropy_max": round(float(np.log(len(tasks))), 5),
        "w_ess": round(w_ess, 2), "w_min": round(float(wv.min()), 5),
        "w_max": round(float(wv.max()), 5), "w_mean": round(float(wv.mean()), 5),
        "chosen_band_mix": chosen_mix, "eps_draws": len(eps_chosen), "n_chosen": len(chosen),
           # TRIAGE v4 telemetry. coadapt.py cross-checks vark_hist against the histogram
           # the PATCHED TRAINER prints per step; a mismatch (or a missing trainer line)
           # means the monkeypatch did not take and the arm must abort rather than train
           # at uniform k while claiming variable k.
           **({"vark_hist": vark_hist, "vark_sum_k": vark_sum,
               "vark_rows_per_block": vark_rows_per_block} if vark_hist else {}),
        "w_mean_chosen": round(float(np.mean(wsel)), 5),
        # 1-w is the posterior probability that a group on this task comes back degenerate, so
        # this is the mechanism's own prediction for the fraction triage_report.py will measure.
        # Under --warm-shrink group it is the CALIBRATED one -- read it against the measurement
        # directly; under v1/v2 it carries the three biases documented above and is not.
        "pred_degenerate_frac": round(float(np.mean([1.0 - x for x in wsel])), 5),
    }
    # v3 fields, present only when the v3 flags are, so the six arms already in flight keep
    # writing exactly the stats line their monitoring parses.
    if a.exclude_tasks:
        rec["n_tasks_excluded"] = n_pool_full - len(tasks)
        rec["n_tasks_pool_full"] = n_pool_full
        rec["exclude_reasons"] = excl_reasons
        rec["exclude_file"] = a.exclude_tasks
    if a.rule != "gradmass":
        rec["rule"] = a.rule
        if a.rule == "bandfilter":       # a8T2n's fields, unchanged and in the same order
            rec["n_kept"] = n_kept
            rec["band_lo"] = a.band_lo
            rec["band_hi"] = a.band_hi
        else:
            rec.update(rule_extra)
    if a.estimator != "posterior":
        # Only when it is not the default, so every arm in flight keeps writing exactly the stats
        # line its monitoring parses. On this arm pred_degenerate_frac = mean(1-w) is the PLUG-IN
        # prediction, which under --warm-shrink group makes it identical to pred_degenerate_cal by
        # construction -- that identity is the field-level check that the flag is in force.
        rec["estimator"] = a.estimator
    if a.estimator == "taskrho":
        # THE FIELD-LEVEL CHECK THAT THE FLAG IS IN FORCE. Under --estimator point the plug-in and
        # pred_degenerate_cal are the same number by construction (same p_hat, same global nu);
        # under taskrho they must DIFFER, because pred_degenerate_frac = mean(1-w) is taken at the
        # per-task nu_i while pred_degenerate_cal is kept at the global one for cross-arm
        # comparability. Equal values here mean the file did not reach the weights.
        ru = np.array(rho_used, float)
        rc = np.array([task_rho.get(key[t], 1.0 / (1.0 + nu)) for t in chosen], float)
        rec["task_rho_file"] = a.task_rho
        rec["n_tasks_task_rho"] = int(sum(1 for t in tasks if key[t] in task_rho))
        rec["task_rho_groups"] = rho_meta.get("n_groups_total", 0)
        rec["rho_i_pool_med"] = round(float(np.median(ru)), 5)
        rec["rho_i_pool_iqr"] = round(float(np.percentile(ru, 75) - np.percentile(ru, 25)), 5)
        rec["rho_i_chosen_mean"] = round(float(rc.mean()), 5)
        rec["rho_i_chosen_med"] = round(float(np.median(rc)), 5)
        rec["w_n_distinct"] = int(len(set(round(w[t], 10) for t in tasks)))
    if a.warm_shrink != "none":
        rec["warm_shrink"] = a.warm_shrink
        if nu is not None:
            rec["shrink_nu"] = nu
            rec["shrink_rho"] = round(1.0 / (1.0 + nu), 5)
            # the correlated analogue of pred_degenerate: same plug-in at the same posterior
            # means, under the group model rather than the iid one. pred_degenerate is left
            # alone, iid, on every arm forever -- it is the fleet's cross-arm instrument and a
            # field that means one thing on a8T2 and another on a8T3 is worse than no field.
            rec["pred_degenerate_cal"] = round(
                float(np.mean([degen_at_mean_group(q, a.k, nu) for q in phat])), 5)
        if lam is not None:
            rec["shrink_lambda"] = round(float(lam), 6)
    os.makedirs(os.path.dirname(a.stats), exist_ok=True)
    with open(a.stats, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[triage] cycle {a.cycle}: {n_add} new train episodes (+{n_val} val skipped); "
          f"{n_evid}/{len(tasks)} tasks with evidence; w in "
          f"[{wv.min():.4f},{wv.max():.4f}] mean {wv.mean():.4f}; ESS {w_ess:.1f}/{len(tasks)}",
          flush=True)
    print(f"[triage] bands(n>={a.min_obs:g} obs) {band_counts}", flush=True)
    print(f"[triage] chose {len(chosen)} tasks ({len(eps_chosen)} uniform) mix {chosen_mix}; "
          f"predicted degenerate fraction {rec['pred_degenerate_frac']:.3f}", flush=True)
    print(f"[triage] tau {a.temp:g}, warm tasks {rec['warm_tasks']}; pred_degenerate "
          f"{pred_deg:.3f} (plug-in at posterior means); {dead_frac:.3f} of the budget on "
          f"p_hat<0.05 tasks", flush=True)
    if a.warm_shrink != "none":
        print(f"[triage] warm-shrink {a.warm_shrink}"
              + (f" nu {nu:g} (rho {1.0/(1.0+nu):.3f}); calibrated pred_degenerate "
                 f"{rec['pred_degenerate_cal']:.3f} -- this one IS comparable to the measured "
                 f"fraction" if nu is not None else f" lambda {lam:.4f}"), flush=True)
    if a.exclude_tasks:
        print(f"[triage] excluded {rec['n_tasks_excluded']} of {n_pool_full} pool tasks; "
              f"0 of the budget and 0 of the eps floor can reach them", flush=True)

    if not a.no_parquet:
        cmd = [VERL_PY,
               f"{R}/surface/verl_rl/prep_awm.py", "--pool", a.out_pool, "--out", a.out_data]
        env = dict(os.environ)
        env["PYTHONPATH"] = ":".join([f"{R}/surface/gate_surface", f"{R}/surface/gate_caller",
                                      env.get("PYTHONPATH", "")])
        env.setdefault("AWM_PY", AWM_PY)
        env.setdefault("MCP_ROOT", R)
        env.setdefault("GATE_MAX_TURNS", "20")
        env.setdefault("GATE_MAX_NEW_TOKENS", "1024")
        env.setdefault("CALLER_FAMILY", "qwen")
        env.setdefault("HF_HOME", f"{R}/hf_cache")
        # OFFLINE, LIKE THE TRAINER ALREADY IS (slurm/verl_awm_train.sh:15 exports HF_HUB_OFFLINE=1).
        # prep_awm.py only loads a tokenizer that is already in hf_cache, so this step has no reason
        # to touch the Hub -- but transformers' _patch_mistral_regex calls is_base_mistral(), which
        # calls model_info() over the network for any NON-LOCAL model id, unconditionally. On
        # 2026-08-20 the Hub answered 429 Too Many Requests, that call raised, prep_awm died, the
        # parquet was never built, and this function therefore (correctly) never committed
        # cycles_built -- which tripped coadapt.py's allocation guard and killed q2bLp at cycle 1.
        # The guards were right; the network dependency was the defect. Offline resolves the same
        # cached files and makes _is_local true, so the Hub call is skipped entirely. Verified: the
        # 8B tokenizer loads under HF_HUB_OFFLINE=1 from hf_cache and tokenizes identically.
        env.setdefault("HF_HUB_OFFLINE", "1")
        print("[triage] rebuilding parquet", flush=True)
        subprocess.run(cmd, check=True, env=env)

    # State is committed ONLY after the parquet exists. A crash inside prep_awm would otherwise
    # consume this cycle's episodes and mark the cycle built while the trainer still holds the
    # previous cycle's task list -- evidence spent on an allocation that never happened.
    st["counts"] = {k: [round(v[0], 6), round(v[1], 6)] for k, v in counts.items() if sum(v) > 1e-6}
    st["consumed_lines"] = new_off
    st["cycles_built"] = sorted(set(st["cycles_built"]) | {a.cycle})
    if warm:
        st["warm"] = warm       # only written on a warm arm, so a v1 state file stays a v1 state file
    st["last"] = rec
    tmp = a.state + ".tmp"
    json.dump(st, open(tmp, "w"))
    os.replace(tmp, a.state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
