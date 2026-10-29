# M49: shared contributions have native effects that gate deletion can hide

M48 established that gate removal deletes exclusive atoms while shared atoms remain selected elsewhere. M49 applies these distinct represented directions to the native MLP1 write. Eight frozen candidates, all192inputs, and three components are retained. No fitting or group selection by outcome.

For each input, SHA256 fixes one group index, identical across candidates. At each position, the selected gate must be active; otherwise its contribution is zero. Encode the native product write once, freeze coordinates and other gates, and subtract exclusive, shared or combined decoded contributions from the native write. Bias and all downstream computation remain live. These are representation-derived external perturbations, not deletions of identified native neurons. Group indices match across candidates but have no established semantic alignment.

## Results

The table pools both seeds and source objectives within each grouping. Output KL averages all inputs, including no-ops. Removed energy is pooled squared delta norm divided by pooled native product-write energy. Different component magnitudes and meanings prevent interpreting larger output change as a better representation.

| Cohort | Grouping | Component | Mean output KL | Removed / native energy | No-ops / rows |
|---|---|---|---:|---:|---:|
| task train | random | exclusive | 0.000665 | 0.0051 | 219/528 |
| task train | random | shared | 0.000882 | 0.0054 | 245/528 |
| task train | random | union | 0.001657 | 0.0106 | 219/528 |
| task train | coactivation | exclusive | 0.000801 | 0.0049 | 238/528 |
| task train | coactivation | shared | 0.002021 | 0.0087 | 250/528 |
| task train | coactivation | union | 0.003950 | 0.0139 | 238/528 |
| task evaluation | random | exclusive | 0.000597 | 0.0055 | 52/112 |
| task evaluation | random | shared | 0.000941 | 0.0052 | 58/112 |
| task evaluation | random | union | 0.001723 | 0.0107 | 52/112 |
| task evaluation | coactivation | exclusive | 0.000378 | 0.0040 | 42/112 |
| task evaluation | coactivation | shared | 0.001667 | 0.0098 | 45/112 |
| task evaluation | coactivation | union | 0.002590 | 0.0141 | 42/112 |
| natural evaluation | random | exclusive | 0.000353 | 0.0038 | 2/128 |
| natural evaluation | random | shared | 0.000725 | 0.0060 | 2/128 |
| natural evaluation | random | union | 0.001153 | 0.0099 | 2/128 |
| natural evaluation | coactivation | exclusive | 0.000412 | 0.0040 | 1/128 |
| natural evaluation | coactivation | shared | 0.001019 | 0.0088 | 1/128 |
| natural evaluation | coactivation | union | 0.001662 | 0.0133 | 1/128 |

Shared removal produces larger mean output KL than exclusive removal in8/8candidates on natural text and task evaluation, and6/8on task training. For natural text, learned-group shared removal has KL.001019 versus exclusive.000412; random-group shared removal has.000725 versus exclusive.000353. Shared deltas also contain more energy, so this does not isolate semantic importance or superior causal organization.

Learned groups produce larger pooled shared/union effects than random groups, but the group vectors, energy and occupancy differ. This is not a matched-perturbation architecture contest. KL of a combined intervention is not an additive allocation of the two individual KLs; decoded contributions can also have a nonzero cross term despite disjoint atoms.

## Conceptual update

Group-gate robustness and atom-contribution robustness are different claims. A gate can be removable because another gate preserves the same encoded atom, while removing that atom has a measurable native effect. This illustrates how architectural protection can mimic a redundant unit under one intervention definition. It does not establish that native MLP channels compute duplicates or that these shared atoms are semantic subcomputations.

The next reuse test must preserve group identity and control token/context and intervention magnitude. A context-permuted removal may be more or less disruptive for reasons unrelated to faithful computational units; effect size alone is not a success criterion. Before running that control, specify whether the target is context dependence, selective causal role, or prediction of native intervention responses. Do not conflate those outcomes or optimize groups on this development panel.

The deliverable to the research loop is now a verified exclusive/shared native-effect inventory with no-ops and energies, complementing M46–M47 token/context pairing controls. No new manifold, task discovery, or architecture winner is claimed.

## Audit

3,091native forwards produce4,608rows, including1,709no-ops. Full assignments, candidate inventories, membership-count partitions, no-op identities, native source energy/task replays and summaries pass. Forty-eight samples independently reproduce set membership, decoded deltas and output metrics; maximum decode relative error6.71e-8 and metric discrepancy5.09e-16. Inactive selections remain in every summary. All cohorts are familiar development data; whole-task exclusions remain unused.

[Frozen protocol](validation/shared_group_native_protocol.json) · [Results](validation/shared_group_native.json) · [Audit](validation/shared_group_native_audit.json)
