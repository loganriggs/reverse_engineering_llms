# Dependency proxies and the induction-density tradeoff

This audit asks whether the existing sampled-rule languages provide *opportunities* for natural-looking local dependencies. It is a development-panel diagnostic, not a claim that a word repeat in FineWeb is an induction computation. The synthetic generators still provide the only exact next-token rule labels.

`audit_dependency_proxies.py` decodes the frozen 192-window FineWeb `skip7000` development panel (source SHA-256 `10d66676c804569eaa501d0c3c425f357d1d4305eb2581f1e9a5403504f054c0`), takes 160 GPT-2 tokens per window, and compares up to the first 100 alphabetic words with 160-token synthetic windows. It exports aggregates to `dependency_proxy_audit_dev.json`, not source prose. A repeat opportunity is a repeated word passing a **heuristic** stopword/length filter within 64 words with an earlier and current successor. The raw copy rate asks whether those successors match; a deterministic unrelated prior word in the same window is a control. We also condition on the *current successor* passing the same filter, since matching fixed function-word continuations is a weak induction proxy. Bracket statistics count complete spans in the cropped text. Tokenization and genre mix remain unmatched.

| Stream | Repeats / 100 filtered words | Raw successor copy | Filtered-successor copy (opportunities) | Parenthesis spans | Square-bracket spans |
|---|---:|---:|---:|---:|---:|
| FineWeb development | 12.72 | 0.179 | 0.156 (588) | 101 | 11 |
| Original priority-rule toy, 16 tokens/category | 33.55 | 0.212 | 0.183 (2,460) | 497 | 461 |
| Priority toy, sparse bracket rates | 32.85 | 0.279 | 0.215 (2,199) | 119 | 18 |
| Six-frame joint language, subject repeat mass 0.4 | 19.07 | 0.606 | 0.247 (681) | 0 | 0 |
| Six-frame joint language, repeat mass 0.2 | 17.42 | 0.608 | 0.199 (559) | 0 | 0 |
| Six-frame joint language, repeat mass 0.1 | 16.93 | 0.619 | 0.188 (512) | 0 | 0 |
| Six-frame joint language, repeat mass 0 | 16.04 | 0.629 | 0.150 (427) | 0 | 0 |
| Six-frame joint language, repeat 0.1 + distant return 0.3 | 18.30 | 0.611 | 0.227 (613) | 0 | 0 |
| Low-rate source/scope stream | 20.91 | 0.574 | 0.253 (821) | 0 | 0 |
| Low-rate alias stream | 18.87 | 0.593 | 0.235 (703) | 67 | 0 |

The original priority-rule generator overproduces bracket spans and repeated content words. Lowering its parenthesis/quote opening probabilities to `.006/.0007` roughly matches the *count* of complete brackets on this panel, but the square brackets in the toy signify quotation whereas FineWeb brackets often mean citations or markup. Its next-token rules also use priority dispatch, so the counts of bigram, skip-trigram, induction, and closure labels do not themselves establish simultaneous rule competition. These sparse rates were selected with this development panel and require separate validation before any generalization claim.

The raw successor-copy statistic is misleading here. In the original six-frame stream, the top copied pairs are `near → the` (276), `asked → about` (200), and `during → the` (111): together 587/1,267 raw copies. They are fixed phrase continuations, not evidence of a learned induction opportunity. Once the current successor is filtered, the measured copy rate *does* fall as subject repetition falls, from 0.247 at mass 0.4 to 0.188 at 0.1, closer to FineWeb's 0.156. This is still a crude word-pattern proxy, not an estimate of natural induction prevalence; the filtered opportunities also change in number and composition.

Meanwhile, induction-active verb sites drop from 633/1638 (38.6%) at repeat mass 0.4 to 398/1638 (24.3%) at 0.2, 290/1638 (17.7%) at 0.1, and 135/1638 (8.2%) at zero. These are meaningful labels from the exact DGP, unlike the FineWeb word-repeat proxy. The 0.1 setting is therefore a *lower-density diagnostic*, not a realism fix. The matched 1,000-stream, 72-token oracle audit at that setting replays every prefix and finds 3,913/4,032 verb sites with conflicting modal proposals. Its 480-window surface check has 309 distinct words versus FineWeb's 6,305, despite close sentence-length medians (14 words in both). More lexical and structural variation is needed to address the surface gap.

## Matched model diagnostic

A two-layer, four-head, width-64 RMS-normalized attention-only bilinear model was trained at repeat mass 0.1 for 3,000 CPU updates, batch 64, length 72, seed 0, with the same sampled CE plus `2 ×` oracle-soft verb objective as the earlier 0.4 model. `surface_mixture_v2_repeat01_soft_models/results.json` records the run and checkpoint; `surface_mixture_v2_soft_models/results.json` is the original. Both use a 256-stream fixed evaluation panel drawn from their own DGP.

| Training repeat mass | Verb KL at step 3,000 | Induction-active verb KL | Non-induction verb KL |
|---|---:|---:|---:|
| 0.4 | 0.0634 | 0.0767 | 0.0573 |
| 0.1 | 0.0645 | 0.2574 | 0.0375 |

The aggregate verb metric conceals a large loss of induction competence. This is consistent with the reduced induction-site frequency, though a single training seed and different evaluation distributions do not isolate frequency as the only cause.

To compare causal effects directly, both checkpoints were evaluated on the *same* 1,024-stream, 72-token edit panels at each repeat mass (seed 20261116). Each edit changes one grammar-supported token and compares the full next-verb probability-vector change with the oracle's change. The error is the L1 difference between those effect vectors; smaller is better.

| Training mass | Eval mass | Induction edits | Induction model TV vs oracle TV | Induction effect L1 error | Lexical effect L1 error | Place effect L1 error |
|---|---|---:|---:|---:|---:|---:|
| 0.4 | 0.1 | 466 | 0.163 vs 0.250 | 0.298 | 0.298 | 0.243 |
| 0.1 | 0.1 | 466 | 0.064 vs 0.250 | 0.436 | 0.216 | 0.207 |
| 0.4 | 0.4 | 1,365 | 0.184 vs 0.250 | 0.269 | 0.286 | 0.233 |
| 0.1 | 0.4 | 1,365 | 0.071 vs 0.250 | 0.431 | 0.207 | 0.207 |

The low-repeat-trained model improves lexical and place edits while nearly ignoring the historical-verb edit in both evaluation regimes. This is stronger evidence than aggregate KL for a changed computation strategy. The two streams do not share identical edited prefixes, but within each evaluation row pair the checkpoints see exactly the same edit panel. It remains a one-seed diagnostic; repeat with multiple seeds if the next DGP design depends on the effect size.

## Separating local repetition from long-range return

`subject_return_mass` is a second, independent generator parameter. At each subject slot, it assigns probability to a subject from an earlier sentence *other than* the immediately preceding subject. If none exists yet, that mass returns to the broad uniform distribution. The original generator is recovered at return mass zero. This changes no grammar or exact-oracle rule formula; induction still copies the most recent prior verb of the chosen subject.

With local repeat mass 0.1 and distant return mass 0.3, the 192-window development audit has induction active at 577/1,638 verb sites (35.2%), close to the original 633/1,638 (38.6%), while the filtered-successor copy rate is 0.227 versus the original 0.247. In those 160-token streams, the median source-verb distance is 40 tokens and 70.7% of induction cases source a nonadjacent sentence; the original has median 19 tokens and 6.5% nonadjacent. A separate exact-oracle audit of 1,000 streams at 72 tokens replays all 72,000 sampled prefixes and finds conflicting modal proposals at 3,915/4,032 verb sites. This is a more informative long-range task distribution, **not** a solved naturalness problem: the unchanged scaffold and 309-word vocabulary remain far from FineWeb.

A matched two-layer bilinear model trained on this return setting for 3,000 updates has verb KL 0.0810 and induction-site KL 0.1771 on its fixed evaluation panel. These are worse than the original local-repeat model's 0.0634 and 0.0767 on its own DGP, yet better on induction sites than the low-repeat/no-return model's 0.2574. The decisive comparison is the *same* return-setting causal edit panels. Errors below are L1 errors in the induced next-verb probability-vector change, against the exact oracle:

| Training stream | Edit panel | Induction edits | ≤24-token source | 25–48-token source | >48-token source | Overall |
|---|---|---:|---:|---:|---:|---:|
| Original local repeat | seed 20261116 | 1,049 | 0.260 | 0.474 | 0.508 | 0.409 |
| Distant return | seed 20261116 | 1,049 | 0.442 | 0.381 | 0.418 | 0.406 |
| Original local repeat | seed 20261117 | 1,022 | 0.273 | 0.471 | 0.513 | 0.412 |
| Distant return | seed 20261117 | 1,022 | 0.443 | 0.387 | 0.420 | 0.409 |

The new training distribution improves the response to long-distance historical-verb edits and worsens short-distance edits, consistently across two independent edit panels. Its aggregate induction error is almost unchanged, hiding that redistribution. On the first panel, model effect TV for sources beyond 48 tokens rises from 0.018 to 0.074, still far below the oracle's 0.250. This is **partial** learning, not recovery of the long-range circuit. Training-seed replication and a longer-training or targeted-supervision control are needed before attributing the tradeoff to an architectural limit.

The next DGP step should alter the repeated frame scaffold itself while keeping controllable rule collisions. Candidate changes include optional clauses, variable sentence lengths, richer clause ordering, paragraph/genre transitions, and independently varied filler that does not leak the answer. Preserve a task-rich regime for mechanism recovery and a separately evaluated natural-rate regime; a single prevalence target need not serve both. Hold out complete rule families or combinations when evaluating discovered featurizer structure, and keep reconstruction, task KL, and causal edit fidelity separate. The frozen `skip11000` panel remains untouched by this development audit.
