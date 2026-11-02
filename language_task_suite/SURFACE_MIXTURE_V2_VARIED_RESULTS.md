# Structural variation around a competent joint-rule language

The six-frame v2 language gives exact, competing lexical, induction, frame-cue, and previous-place proposals at a verb site, but its fixed surface is a poor proxy for web text. `surface_mixture_v2_varied.py` is a **separate versioned branch** that retains those four verb rules and their frame positions while varying noun-phrase determiners (`the`/`this`/`that`) and allowing one- or two-clause parenthetical asides after the core sentence. For example, it generated: “Frank, after that lesson, moves the official watch beside the theater (near the court, during the opening).” It also restricts `In the …` openings to interior places, avoiding the earlier `In the coast` error. Parenthetical content is visible in the prefix and does not silently alter the narrative rule oracle.

The exact oracle and completed grammar were checked independently. On **1,000 × 160** task-rich streams, `audit_surface_mixture_v2_varied.py` replayed all 160,000 sampled prefixes, checked probability normalization and sampled-token support, and validated all complete parentheticals and core frame slots. There were 8,583 mixed-verb sites; induction was active at 3,389 (39.5%) and modal proposals conflicted at 8,409 (98.0%). The audit found 2,208 parenthesis-close decision sites, counting both a close-versus-continue choice and a final deterministic close. A lower-parenthetical-rate setting independently passed the same checks: 8,562 verb sites, 3,305 induction-active (38.6%), 8,396 modal conflicts, and 1,123 close decision sites. Grammar-supported lexical, induction, place, cue, and nuisance edits replay the exact before/after oracle on a separate 256-stream panel. Changing a supported current determiner produces **zero exact verb effect** in 886 such pairs; this is a new causal form-invariance control.

The parenthetical rate 0.1 was selected after inspecting the frozen FineWeb `skip7000` **development** panel, then checked without retuning on the separate `skip11000` panel previously used for an alias experiment. The comparison uses 192 windows per source, the first 160 GPT-2 tokens for FineWeb, 160 generator tokens, and at most 100 alphabetic words. Tokenization and genre mix differ. The counts are form diagnostics, not evidence that the toy has FineWeb's latent tasks.

| Source | Complete parentheses / 192 windows | Windows containing a complete pair | Median / p90 span words | Distinct words | Repeat opportunities / 100 filtered words |
|---|---:|---:|---:|---:|---:|
| FineWeb development | 101 | 63 | 2 / 7 | 4,909 | 12.72 |
| FineWeb separate form panel | 110 | 60 | 2 / 7 | 5,050 | 13.79 |
| Varied toy, task-rich parenthetical rate 0.2 | 213 | 134 | 3 / 6 | 311 | 20.40 |
| Varied toy, lower rate 0.1 | 106 | 85 | 3 / 6 | 311 | 19.47 |

The lower rate matches only the coarse number of complete spans; it spreads them across more windows and still has a different span-length distribution. Variable determiners reduce the raw repeated-word successor-copy rate from 0.606 in v2 to 0.398–0.399, mostly by disrupting fixed phrase continuations. After filtering for lexical successors, copying is still about 0.241 versus 0.156 on the development panel and 0.227 on the separate panel. The toy has only 311 observed word types versus roughly 5,000. This is a useful controlled structural stress test with a clear remaining realism gap.

The matched model test uses the same width-64, four-head, two-layer RMS-normalized bilinear architecture and 3,000 CPU updates as the competent v2 baseline, with sampled next-token CE plus an additional exact-oracle verb soft-target weight of 2. It trains at task-rich parenthetical rate 0.2 and evaluates both that rate and the lower 0.1 rate. The key gates are held-out verb KL, supported causal rule-edit effects, form-invariance under determiner edits, and form-site KL for the close-versus-continue choice. Good average KL alone is insufficient to claim retention of the four competing rules.

The training run completed with held-out verb KL **0.0560** and induction-active KL **0.0667** at step 3,000. The earlier fixed-surface v2 run had 0.0634 and 0.0767 under the same model/training budget; the generated distributions differ, so this is evidence of successful learning under added form variation, not a controlled improvement in general ability. On a fresh 1,024-stream panel, task-rich and lower-parenthetical-rate verb KL were **0.0629** and **0.0620**, respectively. Determiner-site KL was 0.00369/0.00350; parenthesis-close-site KL was 0.0245/0.0266. The latter sites include both a stochastic 0.6/0.4 close/continue choice and a forced final close, and do not test arbitrary nesting.

Supported one-token causal edits on separate 512-stream panels provide a more specific test. The following numbers are task-rich; lower-parenthetical-rate scores were similar. `oracle TV` and `model TV` are total-variation distances between before/after next-verb distributions. `effect L1 error` compares their *signed probability shifts*, so a small value is better.

| Edit | Pairs | Oracle TV | Model TV | Effect L1 error | Effect cosine |
|---|---:|---:|---:|---:|---:|
| Lexical subject | 1,435 | 0.581 | 0.601 | 0.278 | 0.962 |
| Frame cue | 637 | 0.262 | 0.299 | 0.216 | 0.958 |
| Previous place | 1,572 | 0.179 | 0.227 | 0.220 | 0.941 |
| Earlier verb (induction) | 649 | 0.250 | 0.220 | 0.247 | 0.950 |
| Current determiner (nuisance) | 1,754 | 0 | 0.00625 | 0.0125 | n/a |
| Previous adjective (nuisance) | 1,572 | 0 | 0.00808 | 0.0162 | n/a |

The signed-effect alignment shows that this model uses all four competing verb proposals on this panel, while nearly ignoring two grammar-valid nuisance edits. The magnitude is imperfect, especially for the previous-place rule. Induction degrades with source distance: at at most 24 tokens, 561 pairs gave model TV 0.227 and effect cosine 0.964; at 25–48 tokens, 83 pairs gave 0.186 and 0.896. Only five pairs exceeded 48 tokens, with model TV 0.053 versus oracle TV 0.250. That tail is too small for a precise estimate but is consistent with weak long-range induction. Repeating the panel with the lower parenthetical rate did not remove the pattern. The next gate is a distance-balanced evaluation and, if needed, a curriculum that preserves the same rule collisions at longer gaps.

This establishes a more varied **competent joint-rule model** for later representation experiments. It does not make the synthetic source FineWeb-like or identify a unique mechanistic component. Next compare weight-only, task-blind activation, and supervised/intervention-aware sparse bases at matched bond width, measuring held-out model fidelity, oracle KL, and signed causal effects separately. Whole-frame and held-out distance splits should test transfer, with the long-range weakness treated as a model limitation rather than something a featurizer can fix.
