## H819 — Frozen shared/private representation partly generalizes to 32 new copying words

Execute the H818 locked validation on every eligible copying item, with the exact H815 dictionary hash, memberships, geometry gate, L2H5 source perturbation and L6 query MLP retained. No dictionary refit or gate selection occurs. New response captures use coefficients 0/.05/.10, both native/local-state context pairings, and own/input-swapped response arms. Each candidate response is evaluated against its own recipient baseline and the full uncompressed intervention reference.

All 32 words are disjoint from the previous response-fitting vocabulary. Keys, template, head/layer, and response family are reused. This is fresh lexical validation of one task, not general task transfer. The seven-item new IOI panel remains ineligible under H818 and is not included here.

### Full-distribution fidelity

| Components retained | Mean KL(full || candidate) | Ratio to zero-module KL |
|---|---:|---:|
| Neither | .00043414 | 1.000 |
| Shared + selected private | .00004113 | .0947 |
| Shared only | .00012446 | .2867 |
| Selected private only | .00029167 | .6718 |
| Shared + other private | .00005099 | .1174 |

The frozen representation substantially improves distributional fidelity over deleting the tested module response. A .0947 ratio is a ratio of mean divergences, not “90.5% of model computation explained.” Absolute mean total variation is .00273 with selected shared/private, compared with .00919 for zero-module response. Reverse KL gives the same aggregate ordering.

Ablations support the usefulness of both components within both selected groups. Group0 KL ratios are .0931 for both, .3305 for shared-only, and .7895 for private-only. Group1 ratios are .0954, .2690, and .6244. Thus shared-component reuse is not confined to the original three-item internal holdout. Nevertheless sharing is imposed by the architecture, and these results do not identify unique semantic atoms.

### Behavioral response pattern

| Coefficient | Full own-context gap | Full input-swapped gap | Frozen own-context gap | Frozen input-swapped gap |
|---|---:|---:|---:|---:|
| .05 | +.01477 | -.00835 | +.01336 | -.00597 |
| .10 | +.03438 | -.01902 | +.02790 | -.01153 |

The context gap is different-key minus same-key intended-token log-probability response. The frozen dictionary preserves its reversal under input swapping, while underestimating the swap-contrast magnitude: full -.02313/-.05340 becomes -.01933/-.03943. This is partial fidelity, not an exact data-based canonical form.

Native accuracy was 99.22%. Full-intervention accuracy is 97.66%/97.46% at .05/.10, and frozen-representation accuracy is 97.66% at both. These runs include local-state transplants and source perturbations; native-to-hybrid errors cannot be attributed solely to the dictionary. Small KL and correct qualitative response signs do not prove on-manifold validity.

### Gate anomaly remains unresolved

Group0 favors the selected private block in mean KL (.00002988 versus .00007116 for the other block). Group1 again favors the other block in mean (.00003816 versus selected .00004828), but the item-mean other-minus-selected interval [-4.10e-5,+5.07e-6] spans zero across 31 contributing lexical items. Group0's corresponding interval is positive [1.01e-5,6.71e-5] across 29 items. Overall, the other block is worse on average; its interval [-2.61e-7,+2.21e-5] narrowly spans zero across 32 items.

The other private block wins on 44.2% of group1 variants, not most or all examples. Group0/group1 contain 398/626 correlated response variants; item bootstrap averages within lexical item. Therefore the mean group1 anomaly is worth tracking but does not justify replacing the gate from these validation outcomes. Intervals are descriptive without multiplicity correction.

### Verification

9216 forwards total: 2432 fresh capture/reference forwards and 6784 frozen-representation evaluation forwards. Both runs complete their numerical checks. Fresh capture RMS inputs reconstruct exactly, float64 bilinear expansion error is at most 8.70e-14 relative, endpoint-scaled module reconstruction error 1.14e-6, zero-delta full-logit replay exact, and own-response logits replay within 1.46e-5.

In the final evaluation, full uncompressed log-probability effects replay the fresh capture references within 4.66e-6. Component sum error is 1.45e-15; dictionary orthogonality error 3.11e-15; paired intervention directions match exactly. The frozen dictionary matches the hash locked in H818, new value words are disjoint from the fitting panel, all source hashes unchanged, all KL values finite and nonnegative, and self-KL exactly zero. Full logits are saved. Reserved test untouched; natural-document pool 26,112.

### Scope and next step

This provides stronger evidence for operationally useful shared/private response compression within repeated-content copying. It does not establish that shared/private beats flat subspaces on fresh data, that the gates denote semantic manifolds, or that the representation transfers to a distinct task. Prior same-budget flat/group comparisons still apply and did not show a universal architecture advantage.

Next obtain a separately prespecified token-aligned fresh IOI panel while keeping the dictionary frozen. A fixed introductory prefix can make initial names use the same word-boundary tokenization as later names; treat this as a new surface condition, not a retroactive pass for H818's seven-item panel. Establish competence before interventions and report all exclusions. Then compare full-distribution fidelity and component usefulness without refitting, and retain flat same-budget controls before promoting the shared/private architecture.
