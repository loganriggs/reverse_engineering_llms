## H709 — Scalar intervention averages are dominated by particular context pairs

Audit all64H708coordinate edits using stored native logits, natural prefixes, edit norms and recorded KL results. No new model executions. Highlight121and208only because their previous aggregate effects were largest positive/negative; these are posthoc cases, not independently confirmed feature labels. Both directions of each context pair are kept together for influence analysis.

Direction121's four unordered-pair mean donorward gains are0.02208,0.40731,−0.01665and−0.00264. Its overall mean0.10253falls to0.000931when the0.40731pair is removed. All four pairs end in newline. The dominant pair contrasts an environmental-hydrodynamics prose paragraph with an interview-style passage ending with an answer. The interview receiver predicts 'What' with probability0.494; native entropy2.94nats. The prose donor has broader native predictions led by 'The' at0.130and entropy5.54nats.

Swapping the prose coordinate into the interview receiver gives donorward gain0.77495and receiver KL0.87304. Its edit norm is54.5%of the receiver's full residual norm. The reverse direction gains only0.03967. PCA/random norm-matched edits on the strongest direction also gain0.47290/0.15613, respectively. This combination suggests a substantial, asymmetric format/confidence perturbation; it does not establish transfer of a reusable question/prose variable. H708did not save edited logits, so this audit cannot yet determine whether edited entropy increased or which token probabilities changed.

Direction208's negative result is also concentrated. One pair, both ending in 'or', has mean gain−0.13359and accounts for most of its overall−0.03636. Removing that pair leaves−0.00394. The contexts differ between a conventional enumeration (services, programs, or...) and open-ended narrative coordination (...animals, or people, or...). Coordinate edits are13.1–15.8%of receiver residual norm. The remaining comma/period pairs have tiny effects, and an 'as'pair is modestly negative. No consistent semantic label follows from this small panel.

The most informative competing hypotheses are:

1. Direction121changes continuation format or discourse mode around newlines.
2. Its largest gain mostly comes from generic confidence/entropy changes rather than donor-specific content transfer.
3. Large oblique-coordinate edits cause nonlinear state distortion; local small edits would behave differently.
4. Contextual computation is distributed across coactive directions, so single-coordinate swaps omit compensating contributions.
5. The apparent121effect is a selected natural-context outlier and will not repeat on new examples.
6. Direction208tracks some context correlated with coordination, but swapping it alone does not isolate the continuation choice.

The immediate discriminating test is a dose sweep with edited logits retained, comparing donorward effects against a receiver-logit temperature control fitted to the edited distribution without optimizing toward the donor. That separates broad confidence changes from donor-specific probability shifts. Any format hypothesis then needs new, diverse, competence-checked interview/list/prose continuation prompts with matched controls. Coactivation-group analysis remains useful but should not be used to relabel a failed scalar test as a success.

Audits: all source hashes unchanged;64cases annotated, whole-model forwards0, test evaluations0, no new documents. Prefix excerpts and observed suffixes are descriptive source data, not instructions or task labels. No correctness conclusions are drawn from donor-distribution KL. Broad goal active.
