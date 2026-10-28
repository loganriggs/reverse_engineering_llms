## H796 — Frozen L2H5 edge transfers across vocabulary and has more negative routing for matching identities

Previous turn was progress: H795 reconstructed the first-source→second-source L2 attention interaction and identified head5 as closely reproducing the all-head causal removal effect. H796 freezes that head and edge and evaluates the separate household-word panel used in H789/H790. No heads are reselected.

The household panel has16 items in each of two formats, with receiver/donor identities from furniture/kitchen vocabulary rather than the transport vocabulary used for head discovery. It was previously used for task and site analysis, so this is a frozen-head cross-lexicon test, not a wholly unseen research dataset. Both formats pass native copying competence. The two identities give a2x2 source factorial; there are no cases with two distinct nonreceiver donor words as in H795.

Actual L2 Q/K/Q2/K2/mixedV factors and the native module output are recaptured under the factorial. Arms are no removal, all edge-head interactions removed, full module interaction removed, and frozen H5 interaction removed. Here the behavioral readout is the two-answer margin, not H795's full16-candidate specificity vector; comparisons between experiments must respect that difference.

### Frozen causal fidelity

| Format | All-edge removal effect RMS on answer margin | H5-versus-all removal relative RMS discrepancy | Remaining interaction RMS, all edges removed | Remaining interaction RMS, H5 removed |
|---|---:|---:|---:|---:|
| Vertical bars | 2.117 | .0173 | .2980 | .2928 |
| Colon/newline | 1.714 | .0157 | .4255 | .4275 |

Remaining-interaction columns are relative to the unclamped baseline. The frozen head reproduces the all-edge effect closely at individual-example level on this lexical panel. This supports transfer of the specific edge computation; it is not a percentage of the whole task explained. Considerable interaction remains after removal.

Mean donor-directed margin change caused by all-edge removal is +1.842/+1.557; H5 removal gives +1.853/+1.551. These are changes relative to the source-swapped run. As before, removal strengthens counterfactual donor promotion; it does not show that removing the head improves native task performance.

### Signed routing structure

The model's edge coefficient is the product of two normalized query-key dot products, each divided by128. It can be negative and is not an attention probability. In the frozen H5 edge, matching-source assignments have much more negative coefficients than mismatching assignments:

| Format | Mean coefficient, matching | Mean coefficient, mismatching | Matching minus mismatching |
|---|---:|---:|---:|
| Vertical bars | -.4320 | -.0561 | -.3758 |
| Colon/newline | -.4851 | -.0695 | -.4156 |

Every one of the16 items in each format has a negative matching-minus-mismatching contrast. Descriptive item-bootstrap intervals are [-.4020,-.3496] and[-.4391,-.3923]. The first Q/K factor has a positive matching contrast (+.276/+.291, positive for all items), while the second Q2/K2 factor has a negative contrast (-.511/-.530, negative for all items). These are within-item contrasts with positions held fixed.

This motivates a similarity-sensitive subtractive read as a concrete hypothesis: matching source representations receive a more negative value weighting. The sign of an attention coefficient alone does not imply suppression of a particular output word, since value vectors and output projection matter. The causal removal results establish an effect on the tested downstream margin, while the coefficient analysis describes one part of the mechanism.

“Matching” here means matching donor identities used in partial L1 MLP-output substitutions. Other residual paths remain receiver-native. Consequently the result does not yet distinguish literal lexical equality, general activation similarity, context-dependent binding, or a hybrid-state artifact. It is not evidence by itself for a semantic hierarchy or manifold.

### Verification

640 model forwards:128 factor captures and512 removal-arm runs. Edge-to-module interaction reconstruction error3.96e-7 relative RMS. Q independence from first-source identity and K/V independence from second-source identity are exact. Original source-swap margins replay H790 within1.05e-5. Single-source cells are unaffected by removal exactly in the stored candidate comparison. All-edge versus full-module removal agrees within5.73e-6 candidate-logit change. Identity logits replay within2.77e-5 across batches. All effects finite and model/input/reference/runner hashes unchanged. Original reserved test untouched; natural-document pool remains26,112.

Bootstrap intervals use16 item clusters per format; formats share lexical items. This is descriptive routing analysis on a frozen-head validation panel, not multiplicity-adjusted evidence for every possible semantic interpretation. No compact model or featurizer has been trained to this result.

### Next discriminator

Test the frozen edge on fully native prompts crossing source-value identity and key identity: same/different words under same/different keys, preserving token roles and controlling positions. Use unambiguous distinct-key prompts for behavioral competence tests; do not assign an unsupported unique correct answer when a repeated key is given conflicting values. Measure signed routing and compare it with raw/normalized activation similarity. This can distinguish a lexical repetition-sensitive subtraction from a binding-specific computation or generic geometric similarity. Only after that distinction should a compact dictionary/operator be fit to preserve the verified causal response.
