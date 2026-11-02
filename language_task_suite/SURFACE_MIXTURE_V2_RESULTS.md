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
