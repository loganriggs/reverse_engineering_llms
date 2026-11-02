# Source-position and QK/V circuits for joint language rules

The [head-level causal patching assay](VARIED_HEAD_PATH_PATCHING_RESULTS.md) found that the trained two-layer bilinear language model uses second-block attention heads 0 and 1 for induction and head 1 for the previous-place rule. Complete-head patches could not tell whether the same head uses the same source or the same kind of computation for both rules. These experiments decompose the **final query** of the second block into exact per-source QK1×QK2×V contributions, then intervene on the source position and factors of the edited token. The model has no MLP, but does have RMSNorm before attention and before its LM head; the interventions retain both norms.

[`eval_varied_second_layer_sources.py`](eval_varied_second_layer_sources.py) replays the native final-query attention output from the sum of its per-source contributions. Across the saved runs, attention replay differs by at most about `1.2e-5` and final logits by at most `6.7e-6` in absolute value. For each grammar-valid one-token edit, it replaces one original source contribution with its edited counterpart and asks how much the next-verb probability shift aligns with the **full model edit effect**. It also records whether the edited token is the most influential source in that head. The metric is a projection onto the full effect, not an additive fraction of probability mass or a score against the oracle.

Two independent panels use seeds 20261205 and 20261206, each with 128 length-72 streams and 1,871/1,960 supported edit pairs. The original model shows a stable source map:

| Edit, second-block head | Edited token's mean aligned effect, panel 1 / 2 | Edited token is top aligned source, panel 1 / 2 | Median source lag |
|---|---:|---:|---:|
| Previous place, head 1 | **.857 / .851** | **100% / 100%** | 14 |
| Earlier verb (induction), head 0 | **.527 / .484** | **95.9% / 96.0%** | 17 |
| Earlier verb (induction), head 1 | .244 / .296 | 88.3% / 89.2% | 17 |
| Current lexical subject, head 3 | .387 / .371 | 83.9% / 81.4% | 0 |

Thus head 1's place and induction roles involve **different source tokens**, even though the complete head mediates both. The observed-pair score-penalty checkpoint preserves the result: previous-place head 1 is top at the edited place in 100% of cases on both panels, with .847/.847 aligned effect; induction head 0 is top at the edited earlier verb in 97.7%/96.6%, with .503/.467 aligned effect. Score concentration did not create this circuit or make it more selective.

[`eval_varied_source_factor_patching.py`](eval_varied_source_factor_patching.py) then holds two factors of the edited source contribution fixed while replacing the third: QK1-only, QK2-only, combined routing QK1×QK2-only, V-only, or all factors. The native QK1×QK2×V factor replay is exact. These are **one-factor counterfactuals using factors computed in the edited full context**. They are intentionally not an additive decomposition; probabilities are nonlinear and the interaction among factors matters.

| Edit, second-block head | Full source patch, panel 1 / 2 | Routing-only | V-only | QK1-only | QK2-only |
|---|---:|---:|---:|---:|---:|
| Lexical, head 3 | .387 / .371 | -.001 / .000 | **.385 / .368** | -.011 / -.009 | .010 / .010 |
| Induction, head 0 | .527 / .484 | .399 / .308 | .213 / .285 | .135 / .124 | **.385 / .324** |
| Previous place, head 1 | **.857 / .851** | **-.280 / -.234** | .711 / .708 | .045 / .057 | -.204 / -.196 |

The lexical subject's contribution at the current token is principally **value content** in head 3. Induction uses both changed routing and changed value; QK2 is the stronger single score branch for head 0 in this counterfactual. The previous-place value alone carries much of the shift, while routing alone opposes it; combining edited routing and value produces the largest correctly aligned effect. This is a concrete multiplicative interaction rather than evidence that any one matrix is a self-contained rule. The observed-pair score-penalty model has nearly identical factor results, including previous-place V-only .706/.713 and routing-only -.272/-.208.

The ordinary panels are dominated by short source distances: only 16/19 induction edits have source lag above 24, and none/one exceed 48. We therefore ran a **return-heavy distance stress test** with `subject_return_mass=.5` (the training generator used zero), keeping length 72 and the exact oracle. Each of two new 512-stream seeds supplied hundreds of short (lag ≤24) and long (lag ≥25) induction edits:

| Original model, seed | Short / long edits | Mean model effect L1, short / long | Mean model-to-oracle effect L1 error, short / long | Oracle-effect cosine, short / long | Head 0's top source is edited verb, short / long |
|---|---:|---:|---:|---:|---:|
| 20261207 | 550 / 478 | .443 / **.122** | .255 / **.458** | .958 / **.430** | .940 / .801 |
| 20261208 | 572 / 440 | .450 / **.121** | .249 / **.457** | .964 / **.402** | .955 / .805 |

The oracle induction shift has L1 magnitude .5. The model still often identifies the earlier verb as head 0's most aligned source at median lag 35, but its next-verb probability shift is far too weak and poorly aligned with the oracle. The observed-pair score-penalty model shows the same failure, with long-distance effect L1 .109/.115 and cosine .361/.331. **Source localization and competent computation are distinct claims.** This test changes the subject-return distribution as well as distance, so it cannot isolate positional decay from an unfamiliar return pattern. The earlier [return-trained v2 control](DEPENDENCY_PROXY_REALISM_AND_REPEAT_TRADEOFF.md) already improved long-range effects while worsening short-range effects, but remained far from oracle fidelity; that supports testing a mixed-distance training regime rather than assuming an architectural impossibility.

A per-source patch uses the edited run's contextual QK and V values inside the original final residual; the intervention is exact as a computation on that hybrid state but does not identify which earlier paths produced those values. We have located meaningful **source and factor roles** within the second attention block, not yet a canonical sparse circuit across both blocks or a natural-language result. Next test held-out frame variants and train/evaluate on distance-balanced returns, then use these causal targets to evaluate any path-level sparsity objective.

Raw original-model outputs: [source panel 1](surface_mixture_v2_varied_soft_models/second_layer_source_paths_n128.json), [source panel 2](surface_mixture_v2_varied_soft_models/second_layer_source_paths_n128_seed2.json), [factor panel 1](surface_mixture_v2_varied_soft_models/source_factor_patching_n128.json), [factor panel 2](surface_mixture_v2_varied_soft_models/source_factor_patching_n128_seed2.json). The corresponding observed-pair-penalty outputs are in [`surface_mixture_v2_varied_functional_observed_p10/`](surface_mixture_v2_varied_functional_observed_p10/).

The return-heavy short/long records are `second_layer_induction_{short,long}_returnp5_n512{,_seed2}.json` in both model directories. Each records model-to-oracle causal fidelity alongside source alignment.
