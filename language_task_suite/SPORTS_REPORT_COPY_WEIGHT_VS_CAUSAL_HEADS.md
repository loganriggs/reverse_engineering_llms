# Score-copy positive control: weight-only OV versus causal heads

The [natural-report carrier pilot](SPORTS_REPORT_CARRIER_MARGIN_PILOT.md) produced a behaviorally competent two-block, 100-token winning-score copy model. This bounded probe asks whether a cheap weight-derived score identifies the heads needed for that rule. It does **not** analyze the failed margin/subtraction model.

[`probe_sports_carrier_copy_heads.py`](probe_sports_carrier_copy_heads.py) draws a fresh 64-pair game-disjoint held-out panel and tests the original and +1 winning-score edit. For each attention head, it zeros that head's input to the attention output projection during a full-document forward pass, then measures paired two-way accuracy, answer-site NLL, and edit-logit shift. It compares those causal effects with an isolated embedding → V → head slice → O → unembedding score for the two answer tokens. The first-layer path uses token embeddings; the second uses a **zero-first-attention, tokenwise MLP surrogate**. These scores use weights and the chosen numeric token pairs, but no real-document activations. They omit QK attention, context, and downstream MLPs, so their sign is not a full-path prediction.

| Panel seed | Baseline both variants | L1 head 0 accuracy drop | L2 head 1 drop | L2 head 3 drop | L2 head 3 answer-site NLL increase |
| --- | ---: | ---: | ---: | ---: | ---: |
| 76101 | 100% | 10.9 points | 26.6 points | 34.4 points | 4.98 nats |
| 76102 | 98.4% | 10.9 points | 26.6 points | 46.9 points | 5.29 nats |

The second-layer OV scores for heads 1 and 3 are **negative** on the first panel (−0.408 and −0.466 mean signed units), yet removing those heads hurts copying substantially. Their absolute OV scores are the two largest second-layer scores, so magnitude is a useful screen here; the direct-path sign is misleading. First-layer head 0 has the largest positive first-layer OV score and is also the most consequential first-layer head by paired accuracy. The pattern repeats on the independently sampled second held-out panel. Its games are disjoint from training but may overlap the first held-out panel; this is panel replication, not a wholly new corpus or model seed.

This strengthens the earlier delimiter finding that a narrow weight path can miss or mis-sign a real causal contribution. It does **not** establish a canonical decomposition or a reusable copy unit. The next method step, if this control is revisited, is to include the actual QK weighting and downstream MLP/readout in a controlled full-path score, then compare with the same ablations. The primary research should move back to variable multi-rule language generation and DGP-setting sweeps; this single-score arm has served its positive-control purpose.

Aggregate records: [panel 76101](results/sports_report_carrier_long_copy_weight_vs_causal_heads_n64.json), [panel 76102](results/sports_report_carrier_long_copy_weight_vs_causal_heads_n64_seed76102.json). The checkpoint remains local; the script and aggregates are sufficient to reproduce the probe when the RotoWire source and checkpoint are available.
