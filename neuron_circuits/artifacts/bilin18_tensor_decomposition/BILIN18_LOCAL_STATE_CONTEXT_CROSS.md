## H800 — Perturbation response follows surrounding context more than transplanted local state

We crossed the surrounding prompt (same versus different keys) with the second value source's entire local residual state, captured after L2 attention and before its MLP. L2 means the third block, zero-indexed. For each hybrid we applied the same projected L2H5 value direction at additive coefficients 0, +.25, and +.50. Every response is measured against that hybrid's own zero-perturbation baseline; the effect of the state transplant is therefore not counted as the perturbation response.

The panel uses matching values, where the model is competent: 16 vertical-bar items and 10 colon/newline items, with both query choices. The projected perturbation is exactly equal across the paired key contexts. Local-state substitutions are diagnostic hybrids and are not assumed to be on the native activation manifold.

### Main result

Mean change in intended-token log probability at coefficient +.50, in natural-log units:

| Format | Surrounding keys | Local state from different keys | Local state from same keys |
|---|---|---:|---:|
| Vertical bars | Different | -.0534 | -.0531 |
| Vertical bars | Same | -.4470 | -.4317 |
| Colon/newline | Different | +.1087 | +.1027 |
| Colon/newline | Same | -.1067 | -.1164 |

The response follows the surrounding context under either local state. In colon/newline prompts the mean response stays positive for different-key contexts and negative for repeated-key contexts, regardless of which local state was transplanted. Descriptive item-bootstrap intervals exclude zero for all four of those means.

Averaging over the two local states, different-minus-same context effects are +.3862 [.2641,.5037] for vertical bars and +.2172 [.1319,.3044] for colon/newline. Averaging over contexts, different-minus-same local-state effects are -.0078 [-.0560,.0416] and +.0078 [-.0101,.0257]. These are controlled contrasts, not unique causal shares. Their sum equals the native diagonal response gap algebraically, and a small average state contrast does not establish itemwise equivalence.

Swapping local states alone has modest but nonzero distributional effects: mean KL(native || hybrid baseline) is .0081–.0086 for vertical bars and .00136–.00141 for colon/newline. Hybrid-baseline intended log probabilities change by -.0040 to -.0281 on average. Full-vocabulary top-1 accuracy remains 96.9–100% in all tested cells; the effects concern confidence and distributions, not answer flips.

### Interpretation and next discriminating experiment

On this panel, the native pre-MLP source state is insufficient to explain the key-context dependence of the identical value perturbation. Holding that state fixed still leaves a large context effect. Other-token states and subsequent cross-token computation remain candidates. This does not identify a semantic key-equality gate or prove that all local effects are absent.

Next localize where the fixed perturbation's trajectories begin to diverge: compare the perturbed-minus-own-baseline residual changes across layers and token roles, then intervene on candidate reader paths with paired baselines. A response can change at the output because of readout/normalization or other state differences, so tracing alone should nominate tests rather than establish a mechanism. Freeze candidate selection before testing a fresh lexical panel and then a distinct competent task.

For the broader featurizer project, this is a concrete validation target: reusable content directions must be paired with representations that preserve their context-dependent causal responses. Neither this experiment nor the preceding edge experiments validate a manifold, a shared semantic dictionary, or a computational DAG. Reconstruction and average output similarity remain insufficient evidence of those claims.

### Implementation checks and limitations

728 model forwards completed. Projected-direction mismatch across key contexts is exactly zero. Relative local-residual clamp error is at most 1.40e-8. Native-state zero-delta logits replay within 2.05e-5; native-state responses match H799 within 4.93e-6. All response values are finite and all tracked hashes unchanged. Each intervention uses the full residual before normalization/MLP and preserves the attention value bus. Original reserved test untouched; natural-document pool remains 26,112.

The panel is reused, the two formats share lexical items, and same-key query variants are duplicate prompts. Queries are averaged within lexical item before bootstrap. Intervals are descriptive, with no multiplicity correction. The result is narrow evidence about these interventions and prompts, not a general statement about LLM representations.
