## H805 — L6 query MLP response transfers strongly alter the context gap, with format-dependent attention contributions

H804 localized the transition to L5–L6. H805 independently transfers attention-output or MLP-output perturbation responses at the final query in L5 or L6. Each transferred module output is recipient baseline + other-context response, paired by item, format, query, and transplanted L2 local state. Both zero and +.50 perturbation runs are included. A joint arm clamps both module outputs to their original captured recipient baselines plus donor responses; its MLP target is not recomputed after the attention intervention. Layer indices are zero-based.

### Results

Context gap = different-key minus same-key intended-token log-probability response, averaged over both local-state choices and query choices within lexical item.

| Layer | Transferred query module response | Vertical-bar gap | Colon/newline gap |
|---|---|---:|---:|
| — | None | +.3862 | +.2172 |
| L5 | Attention | +.1062 | +.0968 |
| L5 | MLP | +.2818 | +.1868 |
| L5 | Both | +.1442 | +.1103 |
| L6 | Attention | -.1877 | +.1139 |
| L6 | MLP | -.2206 | +.0193 |
| L6 | Both | -.4145 | -.0141 |

At L5, attention-response transfer has the larger average effect in both formats. At L6, MLP-response transfer has a strong effect in both formats: it reverses the vertical-bar gap (descriptive item-bootstrap interval [-.3315,-.1172]) and reduces the colon/newline gap close to zero ([-.0132,.0529]). The paired changes from no transfer are -.6068 [-.8198,-.4135] and -.1979 [-.2809,-.1258]. Near zero is not an equivalence claim.

L6 attention alone reverses the vertical-bar gap but leaves a positive colon/newline gap. Joint L6 transfer produces a clearly negative gap in vertical bars and a near-zero gap in colon/newline. Thus a single universal attention-only explanation is not supported across these two formats. Query residual transfer in H804 is broader than these module-output transfers because it also replaces the incoming residual response. This helps explain why joint module replacement need not match whole-residual replacement.

The joint effects are not the sum of single-module effects. In particular, L5 attention-only transfer has a somewhat stronger effect than the joint arm: freely recomputed downstream MLP behavior after the attention swap differs from explicitly clamping the MLP to its captured donor response. These are different interventions, not additive causal shares. No claim is made that an MLP-only or attention-only path is sufficient for the full model computation.

### Next information-gaining test

Separate input-response inheritance from local MLP transformation at L6. Capture the pre-MLP baseline and finite response at the query, then cross baseline context with the input response before computing the MLP. Audit the exact bilinear expansion, with RMS normalization treated explicitly rather than folded into a false linear approximation. Compare the resulting module responses and output behavior. This can distinguish a response already changed by attention from amplification or redirection by the local baseline. Freeze the resulting candidate mechanism for fresh lexical and task validation; do not continue selecting indefinitely on this panel.

For the featurizer program, the emerging target is a shared content perturbation whose reading and local transformation are context dependent. A useful shared dictionary, hierarchy, or computational DAG must retain this response family. These module tests do not yet establish a compact dictionary, a good manifold, or generalization beyond the diagnostic prompts.

### Verification and limits

3432 model forwards: 104 native captures and 3328 module/layer/state/delta runs. Tracked hashes unchanged; exact projected-direction matching; local clamp relative error 1.40e-8. Native-state zero-delta full-logit error is at most 2.05e-5. The no-transfer responses reproduce H800 across both local states within 4.53e-6. Transfer-arm baseline intended log probabilities agree with no-transfer baselines within 4.22e-6, and baseline KL values within 9.36e-8. H799 native-state response replay error is 4.93e-6. All response values finite. These controls check implementation and baseline consistency, not manifold validity.

Colon/newline full-vocabulary accuracy remains 100%; vertical-bar accuracy is 98.4% except joint L6 transfer at 97.7%. The same 16/10 aligned lexical items are reused. Intervals average state/query variants within item, are descriptive, and have no multiplicity correction. Same-key query variants duplicate prompts. Reserved test untouched; natural-document pool stays 26,112. The broad research goal remains unresolved.
