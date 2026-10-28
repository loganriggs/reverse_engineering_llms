## H493 — Shared principal core versus less stable complement

Follow H492's repeated train-half stability analysis. For each of its four training-only basis pairs A,B, compute A B^T=U S V^T and rotate the bases to their principal vectors. Freeze the core definition to principal cosine>=.95 before evaluating component outcomes. Retain every split and both sides. This gives13 shared directions for splits0/1/2 and12 for split3, out of64 retained directions. The remaining51–52 directions form each side's orthogonal complement within its original learned span. The two sides' cores are close principal subspaces, not mathematically identical intersections.

On the same512 reused H491 receivers, compare full64 cross projection, core-only projection and complement-only projection, always retaining the exact source4 quadratic term, scalar normalization and native baseline correction. The latter two are substitutions for the cross term; complement-only is not an isolated causal intervention with the other exact terms removed. Predictions are not refit. The receiving panel has been inspected before; this is exploratory, without new confirmation intervals or gates.

| Split/side | Core dimension | Core KL recovery −/+ | Full64 KL recovery −/+ | Core share of retained cross energy −/+ |
|---|---:|---|---|---|
|0_left|13|93.963% / 93.547%|96.036% / 95.872%|99.8789% / 99.8659%|
|0_right|13|93.963% / 93.526%|95.722% / 95.891%|99.8744% / 99.8577%|
|1_left|13|94.049% / 93.692%|95.710% / 95.769%|99.8820% / 99.8697%|
|1_right|13|94.000% / 93.693%|95.944% / 95.943%|99.8823% / 99.8672%|
|2_left|13|93.974% / 93.795%|95.791% / 95.910%|99.8754% / 99.8588%|
|2_right|13|93.952% / 93.836%|95.940% / 95.878%|99.8785% / 99.8578%|
|3_left|12|93.967% / 93.682%|95.992% / 95.896%|99.8720% / 99.8556%|
|3_right|12|93.628% / 93.667%|95.744% / 95.937%|99.8708% / 99.8523%|

The shared core captures99.85–99.88% of the retained cross-response energy, but core-only native KL recovery ranges93.63–94.05% negative and93.53–93.84% positive. Relative to each full64 map, removing the less stable complement costs1.66–2.12pp negative and2.04–2.37pp positive. These are descriptive paired point differences. The full64 maps retain~95.7–96.0% native KL recovery, so a reproducible small core carries most, but not all, of their behavior.

The complement-only cross projection plus exact structural terms performs poorly. It is not a useful predictor alone, and negative recovery can arise because the large cross core is needed to balance the exact normalization term. This should not be interpreted as a negative independent causal contribution from the complement. KL losses are not additive; even orthogonal raw cross components can interact through the nonlinear readout and normalization correction.

The raw core/complement projectors exactly partition each side's64-dimensional learned cross space. Both use the same source coefficients and receiver data. Hence the behavior loss from removing the complement is not caused by changing source projection or retraining. It demonstrates that low-energy directions that are unstable under this geometric criterion collectively matter to the model's probability responses. It does not establish stable meanings for those individual directions, or prove they cannot be replaced by another low-dimensional set.

Interpretation for computational units: the evidence supports a layered description—a small reproducible core plus a less geometrically reproducible correction space—rather than treating every SVD axis equally as a fundamental feature. However, even the core is defined relative to a source family, data distribution, rank and overlap threshold. Its individual principal vectors depend on the pair being compared; the analysis does not produce a unique universal canonical basis.

Audits: principal rotations orthonormal below1e−10; core and complement projectors sum to the original projector below1e−10; per-case projected cross responses sum correctly; full64 errors replay H492; finite arrays. Save core/complement bases, per-case native errors, cross energies and KL(full-model||core-model). No new data/calibration; total18880 tracked documents unchanged. No process remains running.

Next compare the less stable complements in function space. For each split, evaluate the baseline-readout tangents induced by each side's complementary cross response, and measure paired Fisher-weighted alignment and disagreement as well as their raw-vector alignment. Use the same native baseline metric for both sides, with pooled inner products and explicit effect-size weighting. This tests whether geometrically different corrections nonetheless induce similar probability changes, or whether they provide different approximations with similar aggregate fidelity. Also measure finite full-versus-core distributional changes, keeping local tangent and finite KL conclusions distinct. The result may support stable computational functions without stable individual vector axes. Freeze metrics before examining these new outcomes, retain all splits and signs, and label the reused-panel analysis exploratory. CE objectives, semantic interpretation, native-baseline removal and cross-head/layer transfer remain open.
