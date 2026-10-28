# From ablation results to a taxonomy of computation

2026-10-28. User direction: explain computational strategies and their extremes using toy models
whose mechanisms are known, then test whether those mechanisms explain the real layers.
These are operational working labels, not a claim that each MLP has one canonical category.

## Three distinct questions

1. **What function does the MLP compute on the data?** Token clustering/re-clustering, a linear
   transformation, context-dependent feature conjunction, and conditional routing are possibilities.
2. **How do its channels implement that function?** Localized contributions, an overcomplete
   distributed code, overlapping supports, and large cancelling contributions are possibilities.
3. **Which parts matter to subsequent computation?** Some directions are read strongly, others
   weakly; a route may compensate for an intervention. Similar local errors can have different
   output consequences. This is not determined by clustering or channel count.

Clustering and cancellation therefore are not rival layer labels. A clustered function can use
cancelling channels. A distributed code need not contain duplicate functions. A readable feature
need not be necessary, and a robust binary decision need not imply a preserved output distribution.
Use a profile across these axes rather than “MLP1 is redundancy; MLP2 is cancellation.”

## Concrete working terms and discriminators

| Working term | Operational claim | Useful measurement |
|---|---|---|
| Token geometry transformation | Token-conditioned output changes similarities among tokens | Held-out token-mean geometry, within-class distances; separate context residual |
| Overcomplete distributed implementation | More channel contributions than independent output directions used on data | Effective contribution dimension plus finite deletion; width alone is insufficient |
| Functional overlap / removal robustness | Multiple subsets preserve a specified behavior approximately | Fixed-subset deletion at explicit fidelity tolerance; joint interventions and per-input tails |
| Cancellation-dominated implementation | Large channel writes oppose each other to produce a smaller net write | Sum of contribution norms versus net norm, signed interactions, deletion curves |
| Context-dependent conjunction/routing | The operation depends on combinations of token and context variables | Factorial input changes and transfer to held-out combinations |
| Downstream insensitivity or compensation | Large local changes have small final consequences | Matched local perturbations, layerwise propagation, selective downstream interventions |

“Cancellation-dominated” does not by itself prove floating-point numerical ill-conditioning.
“Low-dimensional” does not prove semantic simplicity. “Overlap” does not prove exact duplication.

## First executable toy: same clustered function, different implementation

[toy_computation_strategies.py](toy_computation_strategies.py) constructs64 one-hot tokens plus a
constant coordinate, four token classes, eight output dimensions, and32 bilinear product channels.
R(x)=1, so this is an explicit linear special case of the bilinear architecture. Each class is mapped
to one orthonormal centroid. Eight channels implement each class response.

- Duplicate implementation: eight contributions each equal one eighth of the class vector.
- Cancelling implementation: multiply alternating contributions by5 and−3. Their sum is unchanged.

Both have identical width, input functions and output function. Outputs match to2.23e-16. Both
are perfectly representable by the same four-class lookup. Define cancellation ratio as
sum of channel-write norms divided by norm of their sum: it is1 versus4 on every toy token.
Use exactly the same random deletion masks in both models (128 fixed seeds per fraction):

| Removed fraction | Duplicate: mean relative squared write error | Cancelling: same metric |
|---|---:|---:|
|25%|.0823|.4011|
|50%|.2754|.7236|
|87.5%|.7756|.9817|

This is a known-ground-truth diagnostic check, not a match to LLM KL or a trained-toy discovery.
The cancelling toy sometimes improves under particular removals; its distribution of effects is
broad. No learned optimizer chose either construction. No toy parameter was fitted to the real data.

```bash
python toy_computation_strategies.py --output validation/toy_computation_strategies.json
```

[Result receipt](validation/toy_computation_strategies.json) includes exact-function/clustering
assertions, source hash, seed and curves. These assertions test the construction; they are not an
independent scientific validation of an LLM explanation.

## Why a model might learn one implementation

There is no evidence yet that the real model deliberately selects one strategy. Different
parameterizations can implement the same function. Training trajectory, initialization, width,
regularization, input statistics and downstream gradients can select among them. Those are
hypotheses requiring interventions on training, not explanations supplied by an ablation curve.

A focused hypothesis set to avoid prematurely naming one mechanism:

1. Token-identity information dominates the early-layer input distribution.
2. Early token geometry is re-clustered for different downstream consumers.
3. Contextual corrections are small in average loss but large on specific inputs.
4. Multiple channels implement approximately the same input-output contribution.
5. Overlapping contributions span a low-dimensional space without literal duplication.
6. Width spreads a necessary computation across many weak contributors.
7. Large opposing writes are a consequence of unconstrained factorization.
8. Cancellation is needed for selective conjunctions on the observed data.
9. Cancellation suppresses unwanted cross-talk between functions.
10. Native-channel pruning fails because useful axes are rotated relative to channels.
11. Removed contributions mainly change write magnitude.
12. Removed contributions rotate writes in directions important to readers.
13. Downstream normalization attenuates some amplitude changes.
14. Later layers compensate for certain directional errors.
15. Alternative residual routes preserve task decisions after local deletion.
16. Binary task metrics hide substantial distributional changes.
17. Short prompt panels under-sample context-dependent computation.
18. Apparent layer differences arise from different pruning fractions or selectors.
19. Weight/contribution regularization would select a less cancelling equivalent implementation.
20. Different seeds reach equally accurate functions with different deletion profiles.

Current prioritization:16 is directly supported by D1;12 over amplitude-only11 is supported by D2;
18 is a live confound in the MLP1/MLP2 comparison;2 is supported by historical layer-geometry
changes but not an MLP-only causal attribution.4/5/7/8/9/14 remain unresolved. The toy proves
logical compatibility, not which explanation holds in the checkpoint.

## Next experimental sequence and stopping rule

1. **Matched real-layer measurement.** On the same opened inputs and masks, compare MLP0/1/2
   at identical removal fractions. Include25%,50% and87.5%; this is a fixed diagnostic comparison,
   not selector optimization. Record local write changes, cancellation ratio, effective contribution
   count, norm/cosine changes, full-output KL and task scores. Keep Down bias separate in the
   contribution diagnostics. Aggregate norms before dividing, and report near-zero net writes
   separately to avoid unstable ratios. MLP2's historical512/4608 retention is88.9% removal,
   whereas D1 tested25%/50%; the existing results cannot establish a clean layer-strategy contrast.
2. **Require several matching signatures.** A candidate toy mechanism must predict local geometry,
   cancellation statistics and deletion behavior together. Matching one KL curve is insufficient.
   Failure should reject that toy explanation; do not add free parameters until it fits everything.
3. **Then study learning causes.** Train matched-width bilinear toy networks on controlled token-map
   versus context-conjunction targets, with matched output scale and fixed train/held-out combination
   splits. Compare three seeds and an explicit baseline versus contribution-size penalty. Freeze
   optimizer, budget and penalty before outcomes. Compare only well-fitted solutions, reporting all
   failures rather than hiding them. This tests whether function demand or optimization selects a
   profile; it does not establish why the language-model checkpoint acquired that profile.
4. **Return to causal featurizers.** Choose objectives based on the surviving mechanism: shared
   coordinates for overlap, jointly represented signed contributions for cancellation, and conditional
   gates for context-dependent computation. Test transfer and preservation, not just reconstruction.

Do not start broad training sweeps before the matched layer comparison. Natural-text provenance
and generalization checks remain necessary before corpus-level claims; the user approved those
checks, but the next priority is this mechanism comparison. P10 projection/residual work remains
available as a focused test of shared directions, not the sole research direction.

## Matched evidence update

[M1](MATCHED_LAYER_MECHANISMS.md) completes the first matched layer comparison. All three layers have net constructive pair interference on the opened task panel; MLP2 is less constructive, with smaller downstream deletion effects. The sum-of-norms ratio alone is not a cancellation diagnostic: orthogonal contributions also increase it. Pair it with channel/net squared energy. The next toy should include shared-plus-private components and independently controlled reader sensitivity; do not assign the extreme cancelling toy to MLP2.

## Learned-capacity update

[T2](TOY_CAPACITY_RESULTS.md) adds the first24 trained toy runs under capacity and input noise. Spare output rank permits contribution spreading and deletion robustness; saturated rank changes that tradeoff. The next fixed causal check varies width on the same fine target task. Other toy mechanisms and corpus generalization remain open.

## Same-task width and specialization update

[T3](TOY_WIDTH_RESULTS.md) shows that capacity removes the reconstruction obstruction but does not unify robustness metrics. Input-noise training yields class-specific channels and targeted single-channel vulnerabilities in these toys. A fixed small dropout/width follow-up will test whether both robustness types can be trained; no universal tradeoff or LLM mechanism is established.

## Fixed grid closed

[T4](TOY_DROPOUT_RESULTS.md) completes the fixed capacity/dropout grid. Combined robustness is achievable by classification metrics in these toys, with vector-fidelity costs. Use [the synthesis](STRATEGY_SYNTHESIS.md) and return to empirical real-layer spectra/coordination; no further hyperparameter grid is planned.
