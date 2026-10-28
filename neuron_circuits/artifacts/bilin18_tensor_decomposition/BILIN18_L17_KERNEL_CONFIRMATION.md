

## H315 — Frozen interaction-kernel fresh-document confirmation

Freeze H314 choices: interaction ridge.01 for both families; additive ridge1 actual/10 angle. Same768 reference documents, descriptors, target responses and kernel normalization. Reconstruct selected readouts with float64 Cholesky, save all36 dual readouts plus reference descriptors/core/partner/mean. New128 query documents exclude3776 complete-document hashes across prior panels. Independent source-document/window/output hashes and slice audit passes;128 unique documents with zero listed overlap. No fresh tuning. Fresh angle seed20270211.

| Family | Predictor | Fresh fidelity | Paired document95% |
|---|---|---:|---:|
|actual|interaction|43.11%|[42.10,44.02]%|
|actual|additive|11.63%|[10.57,12.62]%|
|angle|interaction|25.20%|[23.74,26.47]%|
|angle|additive|2.09%|[1.87,2.34]%|

Interaction-minus-additive paired gains:

- actual: 31.48pp,95%[30.57,32.43].
- angle: 23.10pp,95%[21.61,24.40].

Locked gates: interaction_gain: PASS, interaction20: PASS. Gain requires>=5pp in both families with positive paired95% lower bounds; interaction coverage requires>=20% in both. Bootstrap2000 resamples128 whole documents, retaining positions/heads/prediction pairing. Intervals condition on fixed references, selected hyperparameters and one set of payload draws.

**Interpretation.** This tests transferable reconstruction of the missed CROSS response after the frozen core96/partner64 approximation, not fidelity to the full MLP or total model behavior. Explicit bilinear input products can predict responses whose output directions look broad under global PCA. A broad output covariance therefore need not imply an unstructured or unpredictable computation. The known model is bilinear, so the result validates a useful inductive bias and learning procedure, not discovery of bilinearity or semantic primitives.

The interaction representation is higher-dimensional than the additive control, and readouts are fitted separately for each head/payload group. Representation structure and capacity are not isolated. Each dual readout holds3,538,944 coefficients;18 interaction readouts total63,700,992, before reference descriptors and baseline components. Saving both comparison models doubles the coefficient count. This is a diagnostic predictor, not a storage-efficient replacement. Native supervision and fixed native-informed input geometry remain part of its provenance. No intact substitution, downstream KL or semantic identification is established.

**Audit.** Selection JSON and cached-reference-response hashes checked before fitting. All36 normalized ridge systems have relative residual<1e-8. Independent row-level aggregation replays all fresh fidelities within1e-12. Fresh panel hashes and slices independently verified. The Cholesky solve implements the same positive-ridge system as H314's eigensolve; this run logs linear-system residuals rather than claiming bitwise equality across solvers. All fitted coefficient/descriptor arrays saved under bilin18_l17_kernel_confirmation_frozen. No original model/compiler/base weights changed; all jobs terminal.

**Next discriminator.** Test cross-head application of the frozen interaction readouts before fitting a larger shared model: keep the feature map and normalization associated with each source readout, evaluate target head/payload responses, and compare same-head versus transferred prediction at controlled references. The exact cross polynomial is shared mathematically, but data support and estimation may make specialist readouts differ. Strong transfer supports a common fitted rule; weak transfer motivates coverage and sample-complexity tests rather than immediate head-specific ontology. Use the fresh panel now as a declared diagnostic panel, not a new confirmation set. Broad research goal remains active.
