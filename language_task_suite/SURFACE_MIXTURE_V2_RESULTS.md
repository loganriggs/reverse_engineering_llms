# Six-frame joint-rule language: oracle, realism audit, and causal probes

The motivating target is the [joint-rule toy language proposal](https://www.lesswrong.com/posts/GncmyuKmp526RJpmj/ambitious-mech-interp-w-tensor-transformers-on-toy-languages): a known next-token data-generating process in which lexical statistics, skip rules, and induction interfere on the *same* token, followed by ground-truth checks of weight-based interpretations. This experiment extends [`surface_mixture_language.py`](surface_mixture_language.py); it does not replace the separate multi-genre or document-world probes.

`surface_mixture_v2.py` has six visible sentence frames, a 313-token vocabulary, optional tail clauses, and four independently specified verb proposals: subject-to-verb lexical 70/20/10, the previous verb for the same subject (induction), a current-frame cue, and a verb indexed by the prior sentence's place. A normalized weighted mixture of whichever proposals are available is the exact next-verb distribution. No latent template choice remains after conditioning on the visible prefix.

Run the reproducible audit with `python language_task_suite/audit_surface_mixture_v2.py`. It checked all 216,000 prefixes from 3,000 independently generated 72-token streams for oracle normalization, sampled-token support, replay, valid frame grammar, and independent weighted-mixture calculation. Of 12,131 verb decisions, 11,780 (97.1%) had at least two rules with different modal proposals. Induction, cue, and prior-place support vary naturally with history and sentence frame; the audit records counts in [`surface_mixture_v2_audit.json`](surface_mixture_v2_audit.json).

The matched-window form comparison is mixed. In 480 windows of 72 generator tokens versus the frozen FineWeb panel truncated to 72 GPT-2 tokens, synthetic mean/median/p90 sentence length is 15.9/14/29 words versus 16.3/14/29. But the generator has only 309 observed distinct words versus 6,305, 7.06 versus 5.25 commas per 100 words, and no colons. The longer tail clause largely explains the p90 match; prose still sounds templated and sometimes awkward (for example, “In the coast”). The tokenizer and window mismatch also limits the comparison. This is **not** evidence that the generated language matches FineWeb or that natural dependency frequencies are calibrated.

`train_surface_mixture.py --generator v2` trains the original attention-only bilinear model with ordinary next-token cross-entropy. It evaluates expected KL to the exact prefix oracle on a fixed, independently seeded test stream, including all verb slots and induction-active slots. The v1 default remains intact. `eval_surface_mixture_v2_edits.py` builds valid single-token lexical, prior-verb, previous-place, and cue edits, and an adjective edit with exactly zero oracle effect. It compares model probability *changes* to exact DGP changes, rather than counting a low perplexity as circuit recovery. Cue edits are supported only in the two frames whose cue can change without changing the grammatical frame.

An initial width-64, four-head, two-layer RMSNorm bilinear model trained for 3,000 steps with sampled next-token CE reaches verb KL **0.348** and induction-active verb KL **0.448** on 256 held-out streams. The fixed 3,004-pair causal panel gives:

| Edit | Pairs | Oracle effect TV | Model effect TV | L1 error of effect |
|---|---:|---:|---:|---:|
| Current cue | 324 | .262 | .302 | .288 |
| Current subject, lexical rule isolated | 744 | .579 | .602 | .336 |
| Previous sentence's place | 808 | .180 | .031 | .384 |
| Previous same-subject verb | 320 | .250 | .071 | .475 |
| Previous adjective, no-effect control | 808 | 0 | .010 | .019 |

The model responds to cue and lexical changes much more strongly than the two history-dependent rules. For prior place, the effect error exceeds the **0.360** L1 error of predicting no change. The matched three-layer run reaches verb/induction-active KL **0.332/0.438**, slightly lower than the two-layer run, yet has worse causal-effect L1 errors on all four active edits: cue **.304**, lexical **.375**, place **.456**, induction **.485**. This is a one-seed, in-distribution result; these checkpoints have *not* learned the full joint DGP, and a sparse decomposition cannot be credited with recovering a missing rule.

A diagnostic two-layer run uses `--verb-soft-weight 2`, adding expected cross-entropy under the exact oracle on each verb slot while retaining ordinary next-token CE. At 3,000 steps, verb/induction-active KL is **.063/.077**, compared with **.348/.448** for ordinary training. On the same causal panel, place/induction edit L1 errors fall from **.384/.475** to **.231/.278**. The soft-target model's place effect TV is **.217** versus the oracle's **.180** (ordinary model **.031**); its induction effect TV is **.177** versus the oracle's **.250** (ordinary model **.071**). This is a strong gain in causal behavior, though still incomplete.

The matched `--verb-sampled-weight 2` control adds the same amount of sampled-token verb CE. At 3,000 steps it reaches verb/induction-active KL **.172/.214**. Its causal-edit L1 errors are cue **.352**, lexical **.426**, place **.365**, induction **.331**, nuisance adjective **.028**; the full-oracle counterpart scores **.241/.290/.231/.278/.014**. The two added verb objectives have the same expected loss under the DGP, so this contrast points toward gradient sampling noise and its optimization consequences, rather than verb emphasis alone, as an important contributor. One seed does not establish a universal ordering, and the soft model still underreacts to induction (effect TV .177 versus .250 oracle).

Reproduction uses the local PyTorch environment and the original `BilinearGPT` source path set by the trainer. The main runs used `--generator v2 --depths 2,3 --steps 3000 --batch 64 --length 72 --neval 256 --rmsnorm` for ordinary CE, and `--depths 2 --verb-soft-weight 2` or `--depths 2 --verb-sampled-weight 2` for the diagnostics. All runs used default seed 0 and a distinct output directory. The edit panel uses `--n 256 --length 72 --seed 8701`. Checkpoints and JSON live in `surface_mixture_v2_models/`, `surface_mixture_v2_soft_models/`, and `surface_mixture_v2_sampled_models/`.

The first interpretation test compares an input-independent weight basis with a distribution-weighted activation basis at matched retained dimension, scoring function KL and each rule's causal-edit error. A visually sparse weight slice is suggestive, but only held-out rule-axis and composition tests would support the proposed mechanism-discovery loop.

## First bond comparison on the better-trained model

`eval_surface_bond_gauges.py --generator v2` now compares full-width attention-output bond gauges and orthogonal truncations. On the two-layer oracle-supervised model, the first attention output has width 64. At 16 retained directions, weight-only SVD of its output matrix gives held-out verb KL **.074** and model-to-truncated verb KL **.009**. Activation PCA fitted on 256 independent streams gives **.092/.029**. Raw physical axes give **4.016/3.969**. At 32 directions, weight SVD and activation PCA both nearly preserve the model (model-to-truncated verb KL **.001/.003**). At the second attention output, 16-direction weight SVD gives verb KL **.118** versus **.311** for activation PCA, again favoring the weight-derived subspace at this rank.

The 3,004 causal pairs give the same broad result. The table shows effect L1 errors after truncating one attention-output bond to 16 directions; the untruncated model's errors are cue **.241**, lexical **.290**, prior-place **.231**, induction **.278**.

| Bond | Basis | Cue | Lexical | Prior place | Induction |
|---|---|---:|---:|---:|---:|
| First | Weight SVD | .253 | .295 | .237 | .283 |
| First | Activation PCA | .294 | .342 | .281 | .306 |
| First | Random | .627 | 1.493 | .556 | .500 |
| Second | Weight SVD | .394 | .426 | .299 | .311 |
| Second | Activation PCA | .410 | .509 | .442 | .401 |
| Second | Random | .619 | .669 | .566 | .460 |

This is a **local compression** result. The weight basis is independent of sampled text, while the PCA basis uses a fit panel separate from the test and edit panels. Neither basis labels a planted rule, and the local cut does not establish that a 16-node circuit computes the same function globally. The next decisive test is supervised rule-axis discovery on training edits, evaluated on held-out frames/collisions and compared against this weight-SVD starting point.

## Rule-labeled response subspaces and causal mediation

`eval_surface_v2_rule_subspaces.py` fits each rule's orthogonal response basis from **single-token causal edits** on three sentence frames (`lead`, `subject`, `at`) and evaluates on disjoint frames (`when`, `in_place`, `during`) from another random stream. It measures changes at the final prefix position immediately before an attention output projection. The reverse split swaps the frame sets. The edit types are lexical, prior same-subject verb (induction), previous place, and a previous adjective with no oracle effect. There are 170/171 induction fit/test pairs and 378/431 place fit/test pairs in the forward split; 168/149 and 410/377 in reverse. The fit/test prefixes themselves use separate seeds.

The **basis fit** holds out entire sentence frames; the underlying language model was trained on all six frames. This tests transfer of a featurizer, not generalization of the model to a new task.

At the **second** attention output, eight directions fitted from induction edits capture mean **62.5%** of held-out induction activation-change energy (reverse split **68.0%**), versus **23.9%/24.8%** for eight weight-SVD directions. Eight directions fitted from place edits capture **87.2%/87.3%** of held-out place-change energy, versus **42.5%/42.1%** for weight SVD. The rule bases are not perfectly separate: in the forward split, the induction basis also captures **46.7%** of place-change energy. At the first bond there is similarly strong response geometry, but geometry alone does not establish causal relevance.

`eval_surface_v2_rule_mediation.py` tests output causality by running the original prefix while patching the final-position bond vector with either the **full** activation change induced by an edit or its projection into a learned eight-direction basis. On held-out frames, the complete model's mean effect TV is **.181** for induction and **.206** for prior place; the full local patch at the second bond gives **.156/.201**, but at the first bond only **.023/.012**. Thus a visually clean first-bond difference is mostly insufficient to produce the observed verb effect from that location alone.

At the second bond, the L1 discrepancy between an eight-direction patch and the **full local patch** is:

| Effect / held-out frame split | Matching edit basis | Weight SVD | Activation PCA | Other edit basis |
|---|---:|---:|---:|---:|
| Induction, forward | .124 | .230 | .295 | place .193 |
| Prior place, forward | .139 | .406 | .388 | induction .418 |
| Induction, reverse | .118 | .256 | .284 | place .217 |
| Prior place, reverse | .151 | .399 | .436 | induction .436 |

The matching rule basis is best in both frame directions. These are **supervised local response directions**, not a recovered end-to-end circuit or unsupervised task discovery. The full edited model itself differs from the exact oracle: on forward held-out induction/place edits, model-versus-oracle effect L1 is **.272/.209**; the full local patch gives **.319/.208**, and the eight-direction rule patch **.357/.270**. The local rule patch is more faithful to the model's local effect than a weight-only eight-direction patch, but does not improve oracle fidelity over the complete model. For lexical edits, the full second-bond patch is a poor substitute for the complete edit, consistent with direct residual and other-position paths. These distinctions are essential before assigning a semantic name to any direction.

The full-patch gate is a stopping criterion for single-bond circuit claims: when a full local patch fails to mediate an effect, analysis must move to multi-position or multi-layer paths for that rule. Whole-task holdouts and new-task discovery remain unverified.

As a first test of that prior, we constrained each eight-direction supervised rule basis to lie inside the top **16** or **32** weight-SVD directions at the second bond, then refit it on the same training edits. The held-out patch-to-full L1 error is:

| Rule / frame split | Unconstrained edit basis | Weight-16 prior + edit fit | Weight-32 prior + edit fit | Raw top-8 weight SVD |
|---|---:|---:|---:|---:|
| Induction, forward | .124 | .127 | .113 | .230 |
| Induction, reverse | .118 | .165 | .124 | .256 |
| Prior place, forward | .139 | .206 | .160 | .406 |
| Prior place, reverse | .151 | .206 | .160 | .399 |

Thus a **32-direction weight-only prior retains enough room** to fit compact, rule-specific causal responses on these two frame splits; restricting to 16 loses more of the place effect. The raw top-eight weight directions are much worse than the supervised rotation. This supports the proposed two-stage strategy—weight-derived reduction, then causal organization—at one local bond. It does not yet validate sparsity, identify atomic mechanisms, or prove transfer to an unseen task.

A sparse rotation of that weight-derived subspace is the next local test. A separate whole-task holdout remains required before claiming new-task discovery.

## Unlabeled sparse rotations within the weight-derived subspace

`eval_surface_v2_sparse_rotation.py` starts from the same top-32 weight-SVD subspace. It compares the raw weight axes, a random orthogonal rotation, a **usage quartimax** rotation fitted on ordinary activation vectors, and a **response quartimax** rotation fitted on pooled intervention differences *without edit-kind labels*. Quartimax chooses an orthogonal basis to concentrate each normalized fit vector in few coordinates. The fit stream and three fit frames are separate from the test stream and three held-out frames; we also reverse the frame assignment. Only after fitting the rotations do we use fit labels to name each axis by its largest mean squared response, then evaluate a largest-axis rule classifier on test edits.

| Axis system | Forward macro rule accuracy | Reverse macro rule accuracy | Forward test quartic concentration |
|---|---:|---:|---:|
| Raw weight-SVD axes | .623 | .564 | .114 |
| Random rotation | .398 | .466 | .082 |
| Usage quartimax | .672 | .633 | .219 |
| Unlabeled response quartimax | **.780** | **.721** | **.366** |

Induction classification specifically rises from **.345/.342** on raw weight axes to **.713/.644** after response quartimax (forward/reverse). This is evidence that an intervention-sparsity prior organizes the weight subspace more legibly than raw SVD or ordinary activation sparsity for these known edits. It is **not** unsupervised discovery of a new task: the fit examples are generated from known interventions, and their labels name axes during evaluation. Also, edit type may be partly decodable from the token class being changed.

The causal gate uses the same second-bond local patch as above. For each rule, we select the eight rotated axes with the largest **training** response energy, without changing the unlabeled rotation. Held-out patch-to-full-effect L1 is:

| Effect / frame split | Raw weight axes, selected 8 | Usage quartimax, selected 8 | Response quartimax, selected 8 | Supervised SVD fit within weight-32 |
|---|---:|---:|---:|---:|
| Induction, forward | .192 | .181 | **.146** | .113 |
| Induction, reverse | .211 | .201 | **.134** | .124 |
| Prior place, forward | .351 | .336 | **.178** | .160 |
| Prior place, reverse | .352 | .322 | **.158** | .160 |

Response sparsity improves **causal local-effect retention** in both splits, approaching an explicitly supervised rule subspace while using only labels to choose and name axes. It still does not improve the underlying model's imperfect agreement with the DGP oracle, and it does not establish a complete circuit. This is a meaningful first test of the proposed weight reduction followed by an intervention-based legibility objective; the next information-gaining step is to remove the known edit labels from axis selection and test whether inferred groups predict a truly held-out rule or task family.
