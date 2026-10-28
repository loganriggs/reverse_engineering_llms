## H983 — Joint featurizer pilot: answer labels alone do not establish useful transfer

Latest user steering restores the original objective: multiple supervised task anchors should help organize a shared representation, with unlabeled capacity supporting new task discovery and subsequent retraining. The intervening scalar sentiment work is a supporting case study, not fulfillment of this objective. The prior conversational turn was no experimental progress; this turn revalidated earlier implementations and completed the first direct joint comparison. The previously proposed agreement-only H983 experiment is superseded.

Earlier evidence inspected: H836 exact shared-union semantics passed, but learned membership coverage collapsed and reconstruction/response fidelity lagged signed scalar controls. H839 found weak shared-gate reproducibility. H946–948 showed supervision helps a different bilinear student’s aggregate responses, without stable individual products. These results do not demonstrate bootstrappable joint featurizers. Ten existing feature-structure tests passed again this turn.

Design: one common site, last-token MLP12 output, using cached H963 writes. Six training families: agreement, capital recall, entity binding, sentiment, negation, pronoun gender. Only discovery-format sources0–3:48 endpoint states. Copying and spatial inversion are excluded from all fitting, centering and scaling. All256 cached states are evaluated. Failed native-competence cohorts remain in raw results but are excluded from the competent-cohort summary below. A common site avoids mixing unrelated layer coordinates, but it is not every task’s best site.

Twelve fixed-budget fits: two seeds, two architectures, three objectives (reconstruction, reconstruction plus real answer labels, reconstruction plus independently shuffled within-family labels).128 shared dictionary atoms,32 groups, fixed membership matrices, same coordinate/gate encoders and six label heads. The block condition uses disjoint four-atom groups with eight active groups; shared uses two balanced random partitions, eight atoms/group and four active groups, with exact union decoding. Both register337,574 parameters, including inactive membership parameters. Active dimensions are32 for blocks and roughly29–30 for shared. No within-block orthogonality. No claim of equal inference cost or identical sparsity. All models run600 full-batch Adam steps; no validation selection.

Label heads can read only the first64 atom coordinates; remaining64 have no direct label-readout weights, but are still coupled through reconstruction and group gates. This reserves representational capacity without claiming independently discovered semantic groups. Fixed overlap avoids known learned-membership collapse; it does not test optimized group discovery.

Metric: squared error in reconstructed donor-minus-recipient write differences divided by squared norm of the actual write difference. Zero is perfect; one is the error from predicting no change. Table entries equally average per-family/per-split ratios and both seeds, restricted to non-discovery cohorts passing the existing native-competence gate. They are not pooled-energy ratios or output KL.

| Architecture/objective | Trained-task validation delta error | Whole-task holdout delta error |
|---|---:|---:|
| Block reconstruction |1.471|1.057|
| Block real labels |1.444|1.115|
| Block shuffled labels |1.511|1.055|
| Shared reconstruction |2.124|1.116|
| Shared real labels |1.613|1.061|
| Shared shuffled labels |1.558|1.079|

There is no convincing real-label advantage over shuffled labels, and held-out task differences are reconstructed poorly. Shared reconstruction improves with an auxiliary label loss, but shuffled supervision improves it at least as much in the aggregate trained-task metric. Do not attribute this to semantic structure. The tiny48-state training sample, fixed random topology, common output site and finite optimization budget prevent conclusions about the general value of supervised featurizers. Absolute activation reconstruction is also insufficient: validation reconstruction ratios around0.44–0.51 coexist with delta errors above1.

Audits: exact union masks for every fit, finite predictions, normalized dictionary columns, equal registered parameter counts, source hashes unchanged. Independent CPU replay of all384 delta-error rows agrees within9.54e-7. Train centering independently checked;48 unique training indices per fit and no whole-task holdout training access. Twelve saved models include predictions, codes, gates, heads and training histories. No new full-model forwards or reserved-test access. Natural-document pool remains26,112.

Next: retain this benchmark and test pair-change/causal targets against answer-label supervision, with the same splits and shuffled-pair controls. Improve data coverage separately rather than changing both objective and data at once. Then evaluate frozen reconstructed edits through the actual model, since activation-difference fidelity is not causal fidelity. Only after these gates should unlabeled-group discovery and fresh task validation feed into a retraining loop. H961 context-bridge predictor evaluation remains pending; agreement-only scalar replication is deprioritized. Broad research goal remains active.
