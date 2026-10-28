## H1011 — Upstream causal-effect predictors transfer across templates but fail lexical coverage

Previous goal turn was progress: H1010 reproduced a nine-edge causal graph and exposed native intervention corners as its inference inputs. This turn replaces those suffix measurements with genuine predictions from upstream source states. This is a conditional computational-prediction pilot, not a trained BSF, complete block replacement, or validated manifold.

Capture five clean source-port vectors, their intervention deltas, and the clean residual at the subject position before layer11 (12672 input coordinates). The original native prefix supplies these inputs. Targets are nine scalar causal effects from the H1010 teacher plus a separately measured joint-intervention damage. Student inference receives no suffix margins, derivatives, logits or teacher outputs. Native generation still costs two prefixes and eleven suffix corners per example; that is label generation, not student inference.

Only32 original discovery examples train input PCA, feature scaling, and readouts. Historical context/lexical/joint panels (32each) and formerly fresh32examples remain evaluation only. Fixed families: mean;15features consisting of five clean norms, five delta norms and five clean/delta cosines; PCA4/8 linear; PCA4/8 full quadratic. Ridge penalty0.1, output intercept and centering trained on discovery. Quadratic8 has44features for32training rows; its capacity/data ratio is a limitation. No hyperparameters chosen from evaluation results. No new prospective OOD claim: all evaluation panels are historically opened.

| Family | Discovery | Context-only OOD | Lexical-only OOD | Joint OOD | Historical fresh |
|---|---:|---:|---:|---:|---:|
| Mean |56.0%|53.7%|75.7%|74.6%|69.4%|
| Norms/cosines15 |20.3%|21.1%|43.4%|43.7%|33.3%|
| PCA4 linear |39.9%|36.3%|71.7%|70.8%|66.1%|
| PCA4 quadratic |27.5%|26.3%|70.5%|69.4%|65.9%|
| PCA8 linear |38.5%|34.9%|71.9%|71.0%|65.7%|
| PCA8 quadratic |13.0%|15.2%|72.1%|71.6%|64.7%|

Scores are relative L2 error in predicted joint damage after summing predicted edges, not percentages of the model understood. The native-corner teacher has6.4% original/4.9% historical-fresh error but is not a comparable inference-budget student. Individual edges remain worse than favorable sums suggest: PCA8quadratic context error ranges7.1–51.6%, and its historical-fresh edge errors29.0–93.6%. Norms15 also has>100%error for the MLP10 main effect on historical-fresh examples despite a33.3%joint score. These are failures of comprehensive component recovery.

Post-result diagnostic, not an additional chosen model: training-PCA8 retains83.1%of centered discovery input energy,79.9%context energy, but only16.2%lexical,14.8%joint,13.8%historical-fresh. Source delta RMS approximately doubles under lexical shift (9.35discovery to19.18lexical); target RMS grows1.00to2.87. This supports a coverage/scale explanation for poor transfer but does not identify it uniquely: changed downstream sensitivities, insufficient source-position context, and only32training examples remain confounds. Norm-based transfer shows magnitude carries useful information; it does not prove a norm-only causal mechanism.

Audits: exact32discovery membership reconstructed from token-authority metadata despite different batch ordering;160unique token sequences;zero train/eval token-hash overlap; native original gates pass; independently generated teacher targets match H1010; independent PyTorch fitting reproduces NumPy predictions within7.32e-10absolute. Per-example inputs, nine targets, joint targets, model coefficients, predictions and train indices saved. No source modifications, no reserved BSFtest data, natural-document pool26112unchanged.320partial sequences,1760suffix sequences,160native sequences; no remaining process from this turn.

Next: expand discovery lexical diversity using a separately registered pool disjoint from all historical evaluation nouns and reserved tasks, rather than tuning on these results. Keep magnitude alongside direction and separate recipient context from source changes. Compare matched-size generic projections, source-specific representations and computationally derived weight directions using nine-edge accuracy, joint response, and unrelated-behavior controls. Freeze architectures before new evaluation contexts. Preserve multi-task scope: subject-number is the first causal anchor, not a substitute for the broader supervised-featurizer bootstrap. H961pending and broad goal active.
