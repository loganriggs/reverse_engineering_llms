## H1009 — Early MLP input-to-output prediction is weaker than activation reconstruction

Previous turn was progress: H1008 completed discovery utility tests and found no group-specific oracle advantage; the user asked for a quantitative measure of early-layer understanding. H1009 begins a separate computational-prediction benchmark rather than recycling activation reconstruction as understanding.257three-block prefix evaluations; no full18layer forwards in this experiment.

Reuse H998's256documents from the historical BSFtraining pool:192discovery and64confirmation documents, fourpositions31/63/95/119 each. Capture normalized local MLP inputs and output writes for layers0,1,2. Fit768discovery states, evaluate256confirmation states. No input to the surrogate contains the output it must predict. However the original attention/residual path supplies each native MLP input, so this is teacher-forced local prediction, not full-block or prefix replacement.

For each layer, train-only PCA compresses1152input dimensions to16or32coordinates. Standardize PCA coordinates; use linear features or all linear and upper-triangular quadratic terms. Center/standardize features on discovery only and fit all1152output dimensions by ridge regression with fixed0.01penalty under mean-squared objective. Feature dimensions are16/32linear and152/560quadratic. Target intercept is discovery mean. No validation choice of rank/penalty. Mean baseline and a32dimensional output-PCA oracle are included; the latter observes true output and is explicitly reconstruction-only, not a predictor or a bound on these differently parameterized students.

Score=1-sum||write-prediction||²/sum||write-training_write_mean||², pooled over confirmation states. Zero matches the constant mean baseline; negative is worse. This is squared write-prediction accuracy, not downstream probability fidelity or semantic coverage.

| Local MLP | Linear input16 | Quadratic input16 | Linear input32 | Quadratic input32 | Output-PCA32 oracle |
|---|---:|---:|---:|---:|---:|
| Layer0 |31.7%|37.4%|38.1%|4.9%|51.6%|
| Layer1 |22.9%|23.5%|26.5%|-10.3%|36.8%|
| Layer2 |15.9%|16.4%|19.4%|-25.2%|29.1%|

Quadratic32 discovery scores are roughly85–87%, while confirmation performance collapses. Richer polynomial expressivity is not sufficient with this small correlated sample and fixed regularization. The original MLP is bilinear/quadratic in its normalized input, but only32PCAcoordinates are retained and its output map must be estimated from768states. These numbers do not establish an intrinsic limit on reversibility or on polynomial models. Input compression, data coverage, conditioning and sample size remain confounded; do not tune on this confirmation table and report resulting scores as prospective.

The earlier~69–78%numbers concerned sparse reconstruction of residual states after blocks0–2, using the true state as encoder input and a different larger training panel. This experiment predicts only the MLP write from input on a smaller panel. Neither denominator nor task is comparable enough to call their difference a measured understanding gap. It does demonstrate why compressibility cannot substitute for an input-to-output computational test.

Next keep all four student families fixed and measure downstream fidelity after replacing MLP0,1,2 separately and jointly, recomputing later model states. Distinguish last-position-only from all-prefix-position replacement. Exact attention remains live, so even joint early-MLP replacement must not be labeled a replacement of complete first-three blocks. Measure output KL, native-top1 agreement and ground-truth next-token loss separately; compare constant-mean and identity controls. Do not choose the best student after confirmation without disclosing selection. This creates a quantitative baseline for the user's early-layer question while preserving the broader featurizer/causal-unit goal.

Audits:document splits disjoint; fixed positions; short causal-prefix capture replay relative1.21e-6. Ridge normal-equation residual1.31e-15. Independent NumPy feature construction and prediction replay differs by at most2.18e-11absolute; all scores independently replayed. Sources unchanged, predictions finite, no original reserved-test data. Natural-document pool26,112unchanged. H961 pending; broad goal active.
