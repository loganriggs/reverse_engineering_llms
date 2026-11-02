# Dependency proxies and the induction-density tradeoff

This audit asks whether the existing sampled-rule languages provide *opportunities* for natural-looking local dependencies. It is a development-panel diagnostic, not a claim that a word repeat in FineWeb is an induction computation. The synthetic generators still provide the only exact next-token rule labels.

`audit_dependency_proxies.py` decodes the frozen 192-window FineWeb `skip7000` development panel (source SHA-256 `10d66676c804569eaa501d0c3c425f357d1d4305eb2581f1e9a5403504f054c0`), takes 160 GPT-2 tokens per window, and compares up to the first 100 alphabetic words with 160-token synthetic windows. It exports aggregates to `dependency_proxy_audit_dev.json`, not source prose. A repeat opportunity is a repeated non-stop content word within 64 words with an earlier and current successor. The copy rate asks whether those successors match; a deterministic unrelated prior word in the same window is a control. Bracket statistics count complete spans in the cropped text. Tokenization and genre mix remain unmatched.

| Stream | Repeats / 100 content words | Successor copy given repeat | Complete parenthesis spans / 192 windows | Complete square-bracket spans / 192 windows |
|---|---:|---:|---:|---:|
| FineWeb development | 12.72 | 0.179 | 101 | 11 |
| Original priority-rule toy, 16 tokens/category | 33.55 | 0.212 | 497 | 461 |
| Priority toy, sparse bracket rates | 32.85 | 0.279 | 119 | 18 |
| Six-frame joint language, subject repeat mass 0.4 | 19.07 | 0.606 | 0 | 0 |
| Six-frame joint language, repeat mass 0.2 | 17.42 | 0.608 | 0 | 0 |
| Six-frame joint language, repeat mass 0.1 | 16.93 | 0.619 | 0 | 0 |
| Six-frame joint language, repeat mass 0 | 16.04 | 0.629 | 0 | 0 |
| Low-rate source/scope stream | 20.91 | 0.574 | 0 | 0 |
| Low-rate alias stream | 18.87 | 0.593 | 67 | 0 |

The original priority-rule generator overproduces bracket spans and repeated content words. Lowering its parenthesis/quote opening probabilities to `.006/.0007` roughly matches the *count* of complete brackets on this panel, but the square brackets in the toy signify quotation whereas FineWeb brackets often mean citations or markup. Its next-token rules also use priority dispatch, so the counts of bigram, skip-trigram, induction, and closure labels do not themselves establish simultaneous rule competition. These sparse rates were selected with this development panel and require separate validation before any generalization claim.

For the six-frame joint stream, lowering the probability mass explicitly assigned to repeating the immediately preceding subject barely changes the measured successor-copy excess. Repeated frame phrases and fixed tails appear to dominate this proxy. But induction-active verb sites drop from 633/1638 (38.6%) at repeat mass 0.4 to 398/1638 (24.3%) at 0.2, 290/1638 (17.7%) at 0.1, and 135/1638 (8.2%) at zero. These are meaningful labels from the exact DGP, unlike the FineWeb word-repeat proxy. The 0.1 setting is therefore a *lower-density diagnostic*, not a realism fix. The matched 1,000-stream, 72-token oracle audit at that setting replays every prefix and finds 3,913/4,032 verb sites with conflicting modal proposals. Its 480-window surface check has 309 distinct words versus FineWeb's 6,305, despite close sentence-length medians (14 words in both). More lexical and structural variation is needed to address the surface gap.

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

The next DGP step should alter the repeated frame scaffold itself while keeping controllable rule collisions. Candidate changes include optional clauses, variable sentence lengths, richer clause ordering, paragraph/genre transitions, and independently varied filler that does not leak the answer. Preserve a task-rich regime for mechanism recovery and a separately evaluated natural-rate regime; a single prevalence target need not serve both. Hold out complete rule families or combinations when evaluating discovered featurizer structure, and keep reconstruction, task KL, and causal edit fidelity separate. The frozen `skip11000` panel remains untouched by this development audit.
