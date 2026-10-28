# M28: response supervision transfers to unused groups on familiar inputs, with weaker gains

2026-10-28. Freeze one previously unsupervised group per scheduled training endpoint/region, using the original M18 basis. Group exclusion uses exact token IDs rather than input names: every group supervised on that token sequence for that kind/seed is excluded across both regions and all aliases. Choose the smallest deterministic hash among remaining groups. No activity, native-effect or competence filtering. Keep repeated rows and inactive groups. No model training.

480 scheduled directions comprise400 distinct kind/seed/input/region/group interventions on158 distinct token sequences. All20 initial/joint/KL-only checkpoints are evaluated; both fixed displacement norms, both architectures, both seeds and all ten task families remain. This excludes discrete groups, not geometrically overlapping directions or semantic concepts.

## The joint objective helps beyond its supervised groups

R is native-probability-weighted squared log-probability response discrepancy. Ratios below1 favor joint causal supervision over the matched full-data KL-only control. Each ratio compares models on identical interventions. Raw R across seen and unseen panels is not directly comparable because their native effect magnitudes differ.

| Start | Full-step supervised-group joint/control R | Full-step unseen-group joint/control R | Tenth-step unseen-group joint/control R |
|---|---:|---:|---:|
|scalar71|0.5319|0.8746|0.9899|
|scalar72|0.6612|0.9893|0.9788|
|block71|0.4850|0.8534|0.9852|
|block72|0.4975|0.8505|0.9622|

The joint objective wins all4 model aggregates at both norms, and all8 model/region aggregates at both norms. Full-step unseen-group gains range1.1–14.9%, smaller than the33.9–51.5% gains on supervised directions. Tenth-step unseen gains are1.0–3.8%. This is evidence of partial transfer across discrete direction groups on these inputs, not merely matching the exact trained group IDs. It is not a clean comparison of difficulty across panels, which contain different effects.

The advantage is not universal within the aggregates: family-level raw response error improves25/40 full-step and24/40 tenth-step comparisons. All40 have nonzero native effect energy, so these counts do not hide zero-denominator families. Margin-response error improves only3/8 region aggregates at full size and4/8 at tenth size. Do not call every task's computation improved from the aggregate log-probability result.

Full-step joint normalized response errors are .8530,.7327,.9917,1.0337 for scalar71,scalar72,block71,block72 respectively. A zero-response predictor scores1; block72 still does worse than that baseline despite improving relative to its initial and KL-only representations. This remains incomplete causal fidelity, not validated semantic units or manifold structure.

## What this resolves, and the next distinction

The benefit is not confined to the supervised groups on the same inputs. This weakens a pure group-ID memorization explanation. It does not resolve transfer to new inputs, because the existing evaluation also changes the intervention basis from M18 training directions to M11 benchmark directions.

Next freeze a basis-only comparison on the existing evaluation inputs: keep the same22 pairs, endpoints, regions and group indices, but decode interventions from original M18 starting bases instead of M11. Evaluate the same20 checkpoints and compare joint versus KL-only within each basis, retaining both norms and all zero effects. Reuse already-audited ordinary output metrics because checkpoints and inputs do not change. Audit unchanged intact predictions and independent new native effects; do not require changed-basis effects or denominators to equal M11. Interpret within-basis joint/control contrasts and effect energies, not raw cross-basis error as a pure generalization score. This tests basis sensitivity while evaluation inputs stay fixed; differing direction distributions still prevent attributing all of the remaining gap to context. No new training, data mining, step-size or optimizer search.

## Validation and limitations

Probe33193 completed4526forwards and2400 candidate responses.59 scheduled directions were inactive: scalar71/72=0/3, block71/72=31/25. They remain in every aggregate. The400 initial reconstructed input branches replay their cached full outputs exactly. Candidate intact K values also reproduce prior supervised-direction results, confirming that only interventions changed.

Independent audit verifies all480 exact-input exclusions and deterministic choices, all20 checkpoint hashes,40 sampled NumPy M18 decodes/deltas/response scores,2400 unique target/arm records, noops, forward count and660 summaries. Maximum relative delta error1.43e-7, metric error2.78e-17, no near-tie support mismatches. Native transformer internals are not independently reimplemented. The experiment uses familiar development training inputs and correlated repeated observations, not pristine or whole-task holdouts. No task discovery occurred.

[Protocol](validation/unseen_direction_protocol.json), [results](validation/unseen_direction_results.json), [audit](validation/unseen_direction_audit.json), [paired analysis](validation/unseen_direction_analysis.json). Samples/progress tensors remain local under `artifacts/bilin18_tensor_decomposition/emlp_m28_*`, hash-linked from results. Sources: `prepare_unseen_directions.py`, `probe_unseen_directions.py`, `audit_unseen_directions.py`, `analyze_unseen_directions.py`.
