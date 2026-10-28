## H725 — Functional tests favor different representations for different heads

Previous goal turn classification: progress. H724trained18sparse representations across two verified heads, three architectures and three seeds. H725evaluates every frozen fit without selecting a seed or tuning on task results. Include adaptivePCA Top8andidentity controls.

For task/nuisance interventions compute F(h_donor)-F(h_receiver) from separately reconstructed activations and add that difference to the receiver's native head. This is necessary because sparse support selection is nonlinear; F(h_donor-h_receiver)would be a different experiment. Keep other heads, the value bus and downstream computation native/live as before. Separately replace the full native head by F(h) on the exact32H722validation documents and measure outputKL and observed next-token CE. The former tests a causal difference; the latter tests native replacement fidelity.

| Representation | Agreement fraction of full-head task gain | Agreement mean output KL | Capital fraction of full-head task gain | Capital mean output KL |
|---|---:|---:|---:|---:|
|Adaptive PCA Top8|97.1%–97.1%|0.000573–0.000573|44.6%–44.6%|0.003003–0.003003|
|Learned sparse rotation|76.1%–80.3%|0.000719–0.000881|49.5%–59.1%|0.002742–0.003564|
|Signed scalar Top8|90.9%–98.3%|0.000479–0.000599|65.2%–70.0%|0.001687–0.002131|
|Signed 4Dblocks, two active|98.4%–108.0%|0.000746–0.000751|50.1%–60.4%|0.002352–0.002853|

Ranges are across seeds, not confidence intervals; adaptivePCA is deterministic. All representations use8selected scalar slots per activation, with the previously disclosed parameter/objective differences. Task fractions refer to each head's partial mean donor-directed margin effect on transfer wording, not model accuracy or a complete algorithm.

For agreement, adaptivePCA retains97.1%of the full-head task effect with relative RMS error5.65%across individual directed effects. Learned rotations retain76.1–80.3%, with21.7–24.4%RMS effect error. Scalar models retain90.9–98.3%, with5.85–9.72%RMS error. Four-dimensional blocks retain98.4–108.0%, but one seed's overshoot yields11.9%RMS error; the other seeds are4.90–5.21%. Exceeding100%is distortion relative to the original intervention, not superior faithfulness. No agreement direction reverses sign relative to the full-head effect in these tests.

For capital recall, scalar models perform best among these sparse candidates:65.2–70.1%mean task-effect retention,28.9–37.9%individual RMS error and lower mean native-outputKLthan the tested rotations/blocks. AdaptivePCA retains44.6%, compared with14.3%for H723's fixed first-eight write-PCAspan. Learned rotations retain49.5–59.1%, blocks50.1–60.4%. One rotation seed reverses one of16task-effect signs; block seeds reverse4,0,and1respectively. No scalar seed reverses a capital task-effect sign. Aggregated mean preservation would have hidden these local failures.

Specificity remains separately measured. Agreement adaptivePCA scores4.726versus full-head4.761; scalar seeds4.352–4.661; rotation3.620–3.726. Capital scalar specificity2.034–2.186versus full-head3.015, adaptivePCA1.262androtations1.474–1.980. These are still partial approximations. Observed continuationCE changes are small, variable and can be negative despite nonzeroKL; all per-seed CEvalues remain in the result JSON. CEimprovement is not evidence of faithful replacement.

The reconstruction winner is a useful provisional capital representation, but not yet a verified manifold featurizer. Rotation's greater seed stability did not translate into best functional preservation. Conversely, agreement's weakly stable4Dblocks can approximate this one task well without establishing that their memberships have a unique or semantic meaning. Reconstruction, stability, output fidelity and causal selectivity are genuinely distinct tests here.

Next test the user's shared-dictionary/overlapping-group idea on the frozen scalar dictionaries, retaining all seeds. Learn or infer memberships from natural training activations only, use an exact union of active atoms so shared directions are counted once, and compare against disjoint/random grouping controls with disclosed membership/gate/active-dimension budgets. Freeze those memberships before functional tests. The purpose is to see whether groups provide reusable support structure or useful regime labels beyond ordinary scalar sparsity, without sacrificing the now-measured causal effects. If group constraints degrade fidelity without yielding reproducible/transferable structure, preserve that negative result rather than adding a hierarchy to unstable groups. Hierarchical and computational-DAG assumptions remain separate hypotheses.

Audits:1792whole-model prefix evaluations (1056task/nuisance,704native-reconstruction,32native references). Identity task swaps replay H721within3.82e-6margin error; identity native replacement maxKL3.65e-12. All input/model/implementation hashes unchanged, finite metrics. All18fits retained, no task-based seed selection. Original corpus reservedtest untouched; no new unique research documents, pool26112. Broad goal active.
