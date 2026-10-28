# M6: error orientation matters at matched magnitude and angle

2026-10-28. Completes the fixed MLP1 directional control on the SAME64 reused natural-text training documents. No new data, model fitting, held-out task exposure or seed selection. Protocol and seeds17/29/43 were written before forwards.

## Construction

For native product write v and opposite-half token-table prediction t, let u=v/||v||, a=t dot u and b=||t-a*u||. Draw a Gaussian direction per position, remove its projection onto u and normalize to e. The control is q=a*u+b*e. Thus q and t have the same norm, projection onto v, angle from v and squared distance to v. At covered positions replace the product write, retain Down bias; preserve native writes at uncovered positions. Downstream computations recompute live.

Geometry is formed in float64 and interventions cast to float32. Production assertions check geometry for every random intervention, along with uncovered-write identity. Zero native norms and degenerate random tangents are rejected. Gaussian directions are independently sampled across positions; the reference does not preserve table-error correlations or natural activation geometry.

## Results

Mean across64documents; CE change is relative to native. Coverage and scoring match M5 (59.06% positions replaced, outputs scored over all positions).

| Intervention | Output KL | Change in CE (nats) | Native top1 agreement | Documents with more KL damage than table |
|---|---:|---:|---:|---:|
| Token table | 0.183539 | +0.192442 | 79.30% | — |
| Random tangent, seed17 | 0.045736 | +0.051371 | 88.56% | 0/64 |
| Random tangent, seed29 | 0.046916 | +0.052556 | 88.62% | 0/64 |
| Random tangent, seed43 | 0.045613 | +0.051735 | 88.90% | 0/64 |

Table replacement causes about four times the output KL of each matched random reference. Every document has greater KL under table replacement for all three random draws. Matching pointwise local geometry therefore does not match downstream damage: the orientation and/or joint structure of the errors matters.

This supports downstream-weighted evaluation of featurizers over Euclidean reconstruction alone. It does not identify semantic features, prove the table residual is a unique computational unit, or isolate context from limited-sample table estimation. Random directions occupy a high-dimensional ambient space, may leave the data manifold, and lack table errors' cross-position correlations. Those differences limit the interpretation; three draws are descriptive, not independent document-population significance evidence.

## Close controls and return to the featurizer comparison

Stop this fixed directional-control series. Do not add seed/angle/amplitude sweeps. The concrete deliverable is now an audited natural-text causal benchmark: native MLP1 product writes, document-separated token means, table and matched local-error controls, full-output KL/CE/top1 and retained document provenance. A prospective featurizer must be tested by native reconstruction interventions, not just labels or local error.

Before training, reconnect with existing implementations and failure evidence. H835 already tested scalar/block/shared models on block1/2 PRE-MLP token-residual states and found scalar best under its local-response metric; shared memberships collapsed. H984/H989 fitted MLP12 write featurizers; H1029's learned selector over the old multi-anchor dictionary failed to beat scalarTop32 and has a stopping rule. None is a directly compatible native MLP1 WRITE model. Do not silently transplant their weights or reopen the stopped gate search.

Next prepare a small matched scalar/block/fixed-shared MLP1-write comparison with document-separated fitting and causal evaluation, consulting those implementations first. Keep raw reconstruction, token/context errors, active-coordinate/parameter costs and native-output KL separate. Any claimed improvement from causal supervision needs a real supervised objective and matched control, not relabeling a reconstruction loss. Freeze the fit and evaluation budget before training; no semantic/new-task-discovery claim from this reused corpus.

## Audit and reproduction

320forwards completed:64native+64table replays+192random. All64 table outputs replay M5 metrics exactly. Independent NumPy audit covers documents0/32, eight full-output score calculations, six random geometry/reconstruction checks, and all256record aggregates. Maximum metric difference8.88e-16; relative geometry difference7.35e-9; random-write reconstruction difference2.55e-8. Production checks cover all random interventions. The audit does not independently implement the full native model.

Run `probe_matched_angle.py`, then `audit_matched_angle.py`. Requires pinned model, M4 cached writes and M4/M5 records. Full audit samples remain in the primary artifact directory as `emlp_m6_matched_angle_samples.pt`.

[Pre-run protocol](validation/matched_angle_protocol.json), [all results](validation/matched_angle.json), [audit](validation/matched_angle_audit.json), [M5](TOKEN_TABLE_CAUSAL.md).
