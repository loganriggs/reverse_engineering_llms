## H807 — Frozen vocabulary transfer supports input-response dominance in the eligible format

Executed the H806 frozen validation with bird values, the same L2H5 source perturbation and L6 query MLP, and no layer/head reselection. Each module response crosses the origin of raw pre-MLP baseline and finite input response, explicitly applies RMS normalization, and is added to the recipient's original MLP baseline. Test amplitudes are +.50 (primary) and +.25 (secondary), with zero-delta controls.

### Eligibility and competence

Of 16 planned bird words, seven fail the single-token value/answer requirement in both formats: robin, swan, finch, heron, stork, lark, wren. The bar-separated format retains nine items. Colon/newline additionally loses eagle, dove, and raven because paired key contexts do not align in length/positions, leaving six. All exclusions are recorded; no effects or correctness outcomes were used to select items.

Native full-vocabulary accuracy is 97.2% for bars and 95.8% for colon/newline. Candidate accuracy against the fixed train foil is 100% in both; this is a weak diagnostic, so the full-vocabulary measure is more informative. The foil was an implementation choice, not specified in the lock. Bars passes the frozen minimum-eight-items and competence gate. Colon/newline fails the item-count gate and is descriptive only, despite high competence. The panel was not replaced or topped up after seeing results.

### Frozen contrast

The context gap is different-key minus same-key intended-token log-probability response. All means average the two query choices and two L2 source-state choices within lexical item.

| Format | Amplitude | Own | Input response swapped | Baseline swapped | Both swapped |
|---|---:|---:|---:|---:|---:|
| Bars, eligible | .25 | +.1280 | -.1394 | +.1120 | -.1645 |
| Bars, eligible | .50 | +.4279 | -.2843 | +.3617 | -.3715 |
| Colon/newline, descriptive | .25 | +.0790 | -.0155 | +.0415 | -.0602 |
| Colon/newline, descriptive | .50 | +.1986 | -.1161 | +.1217 | -.2107 |

On the eligible bars panel, the primary paired input-swap minus baseline-swap gap contrast is -.6459 with descriptive item-bootstrap interval [-.9076,-.3980]. At half amplitude it is -.2514 [-.3540,-.1565]. Input-response transfer therefore has the larger effect at both tested amplitudes. The .50 gap shifts are -.7122 for input swap versus -.0663 for baseline swap; .25 shifts are -.2675 versus -.0160. Baseline effects remain measurable and should not be dismissed.

The under-sized colon/newline panel has the same tendency: contrasts -.2378 [-.4459,-.0645] at .50 and -.0570 [-.1032,-.0196] at .25. These do not override the frozen eligibility failure or establish a second successful replication. Several individual arm-gap intervals span zero.

### Probability effects and intervention limits

The fresh values are not behaviorally interchangeable with the discovery vocabulary. At .50 in bars, the own-response arm changes intended probability by -.0837 under different keys and -.2387 under same keys; input swap changes it by -.1509 and -.1301. Mean KL from each hybrid zero-delta baseline is .1575 in the own arm and .1162 under input swap. At .25 those KL values are .0251 and .0202. These are distributional effects, not accuracy differences or causal shares.

Bar-panel intervention accuracy ranges 91.7–94.4% across amplitudes/arms, versus 97.2% native accuracy; colon/newline intervention accuracy is 95.8%. The own arm includes the original source perturbation and possible local-state transplant, so its error increase is not attributable solely to the MLP response swap. Native-like accuracy and on-manifold validity cannot be assumed even at the smaller coefficient.

### Verification

1920 forwards in total: 60 fresh native edge captures plus 1860 local-state/input/response-intervention runs. The frozen configuration hash is tracked and unchanged. First-source mixed-value vectors match exactly across key contexts; RMS inputs reconstruct exactly. Bilinear expansion relative error is at most 1.77e-5 and direct versus captured module-response relative error 2.89e-5. Zero-delta transfers reproduce full-vocabulary logits exactly; own-response reconstruction matches logits within 1.39e-5. Local source clamp relative error is 1.34e-8; all responses finite. Native future-query source-coefficient invariance is exact. The original reserved test remains untouched and the natural-document pool remains 26,112.

One inherited metadata field initially listed only 0/.50, although the executed jobs and saved records included 0/.25/.50. The metadata was corrected after checking the complete recorded amplitude set; the executed runner and its hash were preserved. No numerical result was changed.

### Consequence for research direction

The incoming-response versus local-baseline distinction transfers to a new value vocabulary on one eligible format and survives halving the perturbation. This supports a reusable operation-level validation target, not a universal semantic feature or manifold. The limited vocabulary coverage, intervention-induced errors, and lack of a distinct-task replication remain substantive gaps.

Next test a genuinely distinct competent task with the mechanism fixed, beginning with native competence and token alignment before interventions. Use a role-binding or structured attribute-retrieval task with independently varied queried entity/attribute, repeated versus distinct contexts, and explicit expected answers. Report failure of the frozen L2H5→L6 hypothesis rather than reselecting the layer/head for that task. Only then assess whether compact shared directions or group gates preserve this verified response family.
