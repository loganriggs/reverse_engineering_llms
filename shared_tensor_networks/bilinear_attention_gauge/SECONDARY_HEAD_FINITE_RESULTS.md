# The answer-free layer-11 bracket candidates have small, additive output paths

The [36-head unlabeled edit scan](ALL_HEAD_UNLABELED_EDIT_SCAN_RESULTS.md) ranked layer-11 heads 5 and 6 among the strongest bracket-responsive heads. This experiment freezes each head's leading 128-dimensional edit-PCA axis from the scan's 800-pair training panel, then tests finite native head-output patches on 20 independent validation edits each of brackets, noun number, and punctuation. The complete edited-minus-base head vector is a path upper control. Layer-13 head 8 is the previously established bracket-path positive control; layer-11 head 3 is the agreement/noun-number positive control. No answer or edit-type label chooses an axis, although the candidate heads and test types were selected after inspecting the earlier scan.

At the final prompt token, we add a full head-output edit vector or its one-axis projection to the unedited 546M model. We compare the resulting final-vocabulary logit change with the native input-edit logit change. The **signed projection** is the patched change dotted with the native change, divided by the native change's squared norm. It is a sufficiency-style path measure, not a fraction of causal mediation or a necessity estimate. Identity-basis round trips exactly matched the native logits on this panel.

| Edit type | Head | Complete head signed projection | Leading axis signed projection |
|---|---|---:|---:|
| Bracket | 11/5 | .0417 | .0375 |
| Bracket | 11/6 | .0198 | .0050 |
| Bracket | **11/5 + 11/6 jointly** | **.0618** | **.0424** |
| Bracket | 13/8 positive control | **.3115** | **.3130** |
| Noun number | 11/3 positive control | .1130 | .0835 |
| Noun number | 11/5 | .0028 | -.0001 |
| Noun number | 11/6 | .0197 | .0001 |
| Punctuation | 11/5 + 11/6 jointly | .0290 | .0009 |

The joint full-head bracket projection has a case-bootstrap 95% interval [.0556,.0678]. It is almost the sum of the two separate projections: joint minus sum is .00028, with paired bootstrap interval [-.00015,.00068]. The mean interaction-vector norm is .0061 relative to the native bracket edit norm. Thus the two layer-11 bracket responses do **not** hide a large synergistic or cancelling output path at this tested patch dose. Head 11/5 has a real but modest path, most of which its leading unlabeled axis retains. Head 11/6 has an input-side bracket cluster but its leading axis retains only about a quarter of its already small downstream projection. The known 13/8 head carries about five times the combined 11/5+11/6 directional path on these edits.

This does not establish that 11/5 or 11/6 is unimportant to all bracket behavior. The validation contexts use the same designed edit families and broadly similar templates as the fit panel. The test measures full-vocabulary directional output, not forced-choice bracket answer margins, and a patch need not reveal necessity or redundant backup computation. It does rule out treating their large edit-response energies as evidence of a comparably large bracket mechanism on this panel. For canonicalization, output-aware or finite-path objectives remain essential controls alongside response sparsity; the next head selection should freeze output predictions on new token identities and task families before assigning semantic names.

Reproduce with `taskset -c 2-5 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /home/loganriggs/Coding/neuron_circuits/.venv/bin/python flagship_secondary_head_finite.py` and `flagship_secondary_head_joint.py`. Per-case effects and summaries are in `full_width_results/secondary_head_finite.json` and `secondary_head_joint.json`.
