## H792 — Mixed-source content transfer is nonlinear without a clear average agreement bonus

Previous turn was progress: H791 confirmed donor-specific L1 source effects on fresh vocabulary while showing broad alternative promotion. H792 crosses donor identities to test how the two source representations combine.

For each H791 receiver, run all16 assignments of four native identities (receiver0 and donors1/2/3) to the first and second value-source MLP outputs at L1. This includes receiver/receiver, single-source, matching-donor and mixed-donor conditions. All other computation remains receiver-native and downstream computation is live. The complete factorial and candidate-resolved analysis were specified before outcomes. No model or featurizer was trained.

For candidate-logit changes, the additive prediction is the sum of the two single-source changes (minus the numerically nonzero identity change). The difference from this prediction is the interaction. Candidate vectors are centered before vector-error calculations to remove common logit shifts.

### Mixed donors still transfer content

| Format | Source assignment | Actual mean target specificity | Additive mean prediction | Mean interaction |
|---|---|---:|---:|---:|
| Vertical bars | Matching donors | .510 | .534 | -.024 |
| Vertical bars | Mixed donors | .273 | .315 | -.042 |
| Colon/newline | Matching donors | .417 | .449 | -.033 |
| Colon/newline | Mixed donors | .227 | .238 | -.011 |

For matching donors, target specificity is the donor's logit change relative to the mean of nonreceiver/nondonor candidates. For mixed donors it is the mean change of the two donor candidates relative to the remaining nonreceiver candidates. These use different target-set sizes, so comparing the first and second rows directly is not evidence of an agreement bonus. Mean interaction intervals span zero in all four cells, while both matching and mixed donor specificity are positive.

### Balanced agreement test

For each pair of donor identities A/B, compare half of (A/A+B/B) with half of (A/B+B/A), using the same two target words and background candidates. This cancels additive source main effects. Each donor occupies each source equally often, so average squared replacement norm is also balanced (numerical relative difference <6e-8).

The balanced matching-minus-mixed specificity difference is -.0156 for vertical bars (descriptive item-bootstrap interval [-.0605,.0289]) and -.0159 for colon/newline ([-.0726,.0329]). There is no clear positive average specificity bonus for source agreement under this test. This does not prove absence of an agreement mechanism: nonlinear geometry, remaining receiver-native paths and item-dependent effects remain possible. It does weaken the claim that H790's single-versus-joint pattern alone revealed a general agreement detector.

The balanced target-versus-receiver margin difference is -.181 in vertical bars (interval spans zero) and -.436 in colon/newline (interval [-.613,-.265]). A change in binary margin can therefore coexist with little change in target identity specificity, reinforcing the importance of candidate-resolved readouts.

### Means conceal substantial non-additivity

Adding single-source responses predicts average specificity reasonably closely, but individual predictions are poor:

| Format | Assignment | Additive RMS error in target specificity | Actual specificity RMS | Relative error |
|---|---|---:|---:|---:|
| Vertical bars | Matching | .344 | .647 | .531 |
| Vertical bars | Mixed | .143 | .392 | .363 |
| Colon/newline | Matching | .367 | .557 | .659 |
| Colon/newline | Mixed | .114 | .377 | .302 |

Across the full centered16-candidate logit vectors, additive relative RMS error is1.030/1.020 for matching donors and .486/.452 for mixed donors. Thus the near-zero mean interaction is cancellation across cases, not an accurate additive computational model.

A follow-up output-geometry diagnostic projects each centered interaction onto the receiver-versus-other-candidates axis. Only31–37% of interaction squared norm lies on that axis. After removing it, relative additive errors among nonreceiver candidates remain .926/.986 for matching donors and .482/.462 for mixed donors. Non-additivity is therefore not captured by a single receiver-confidence term. These squared-energy shares describe an orthogonal output decomposition, not unique causal mediation.

### Verification and scope

512 model forwards for the complete32-item-format by16-assignment factorial. Requested replacements exact. Identity logit replay <=1.77e-5; all candidate-logit changes for overlapping H791 single/matching conditions replay within7.63e-6. All metrics finite; model/input/reference/runner hashes unchanged. Original reserved test untouched; natural-document pool remains26,112.

Both formats retain strong H791 native competence. This is a reused panel for compositional discovery. Each format has16 lexical-item clusters; donor pairs and formats are correlated. Bootstrap intervals average donor pairs within item and are descriptive without multiplicity correction. Fixed output replacements leave other residual pathways native, so results do not directly characterize fully native mixed-value prompts or establish a minimal circuit.

### Research decision

L1 source representations carry partial donor content, including when sources disagree. Neither a simple additive model nor a universal agreement bonus adequately explains their downstream combination. The important unresolved structure is item-dependent nonlinear composition, not just the mean strength of a feature.

Next trace the factorial interaction through downstream attention outputs, MLP outputs and normalized inputs at source/query positions. Determine where interactions first appear and grow, while auditing that the traced runs reproduce the same final candidate responses. This can distinguish normalization effects, token-local bilinear mixing and later cross-position interactions before proposing a learned DAG. A useful featurizer must preserve per-example identity-specific finite responses; matching average reconstruction or average donor promotion is insufficient.
