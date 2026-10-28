## H754 — Task-blind natural baselines established for L5H3 and L5H5

Natural activations are collected from the existing2,304 documents:2,048 training and256 validation, at16 token positions7,15,...127. Each head has32,768 training vectors and4,096 validation vectors. These are pre-output-projection128-dimensional head outputs. No task examples enter the fits; head selection itself remains informed by preceding task experiments. Document hashes show no exact train/validation overlap. The original reserved test remains untouched.

Train-only baselines are centered coordinate PCA, residual-write-metric PCA, adaptive top-K in the write-PCA coordinates, and adaptive top-K in three fixed random orthogonal write-coordinate bases. Write PCA measures the vector actually written into the residual stream through the head output projection. The full128-coordinate transform is invertible and isometric in that write metric; it is not whitening by activation variance.

| Head and eight-coordinate representation | Natural validation write MSE | Narrative operation difference MSE: final query | Narrative operation difference MSE: source |
|---|---:|---:|---:|
| H3 coordinate PCA | 71.6% | 73.3% | 73.5% |
| H3 write PCA | 70.9% | 72.8% | 72.4% |
| H3 adaptive write PCA | 51.2% | 63.3% | 55.7% |
| H3 random adaptive, mean3seeds | 68.2% | 75.0% | 69.0% |
| H5 coordinate PCA | 62.9% | 74.4% | 71.0% |
| H5 write PCA | 62.1% | 70.8% | 70.5% |
| H5 adaptive write PCA | 43.2% | 51.7% | 52.8% |
| H5 random adaptive, mean3seeds | 68.2% | 65.7% | 81.1% |

These are relative squared reconstruction errors, not RMS errors or causal-effect retention. Natural denominators are validation variance around the training mean. Task denominators are full paired-operation write-difference energy. Task values use competent H747 narratives; diagnostic lists remain separately recorded. For adaptive representations, differences are F(donor)-F(receiver), not F(donor-receiver). Both members of each difference are reconstructed independently.

Adaptive top-eight coordinates can draw from all128 axes across examples; fixed rank-eight PCA uses the same eight directions everywhere. They share an active-coordinate budget but not a fixed global subspace. The adaptive advantage does not by itself establish a nonlinear semantic manifold. Random bases also improve when selecting coordinates adaptively, so this control remains important.

Interpretation: these upstream heads are not adequately reconstructed by a tiny fixed linear subspace on natural text. Data-derived adaptive coordinates help, especially at H5, but substantial operation variation is still omitted. Reconstruction geometry cannot establish whether those omissions matter for the verified downstream component. There are no new causal interventions in this experiment.

Artifacts include natural activation tensors, unchanged token documents and hashes, full fitted transforms, all rank/active-budget curves1,2,4,8,16,32,64,128, and paired task reconstruction summaries. They provide a common input and baseline for subsequent scalar and grouped/overlapping featurizers without fitting on the causal panel.

Audits:2,304 six-block prefix evaluations, no whole-model or reserved-test evaluations. Input/source hashes unchanged; all activations finite; maximum full-rank validation write MSE8.09e-30; inverse-transform error<=1.56e-15; write-isometry relative error<=2.42e-15. Natural document pool remains26,112 because the same documents are reused.

Next: train scalar and block-sparse baselines with identical width, active budget, seeds and optimization schedule on these write coordinates, then test preservation of the verified downstream dependence and diverse-task effects. Extend shared-group models only with explicit matched controls; do not declare a manifold from reconstruction alone.
