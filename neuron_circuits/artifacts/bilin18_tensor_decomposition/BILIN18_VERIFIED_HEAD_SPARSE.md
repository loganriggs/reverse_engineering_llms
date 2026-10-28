## H724 — Scalar sparsity wins reconstruction; stability and functional validation remain separate

Previous turn classification: progress. H723 supplied a strong task-blind write-metric PCA baseline. H724 fits sparse alternatives in the same full128Dwrite-coordinate space, using natural-text training/validation data only and retaining all three seeds7241–7243.

All candidates have128decoder directions and a nominal selected budget of8scalar coordinates. The rotation candidate whitens the full training covariance, learns an orthogonal rotation by smoothL1, then unwhitens for reconstruction; selectTop8by individual decoded-write contribution norm. Train3000steps,Adam0.003,batch512; checkpoint by validationL1every250steps. Compare the initial random rotations and adaptive PCA, which selects each example's eight largest write-PCAcoordinates instead of retaining a fixed first-eight span.

The signed BSF baselines use a128→128linear encoder with bias, a128×128decoder with unit-norm atoms, and output bias. Scalar TopKselects8atoms; four-dimensional blocks select2blocks. Train3000steps,Adam0.001,batch512on globally scaled write coordinates; checkpoint by validation reconstructionMSEevery250steps. This is a signed unit-column block-TopKbaseline, not a Grassmannian BSF: within-block orthogonality is not imposed. Both width and selected scalar budget match, but training objectives and parameter counts differ: rotation has16384raw trainable parameters with8128orthogonal degrees; each BSF has33024parameters. Equal steps are not equal compute or capacity.

| Candidate | Agreement validation write relative MSE | Capital validation write relative MSE |
|---|---:|---:|
|Adaptive PCA Top8|0.3691 [0.3691,0.3691]|0.5545 [0.5545,0.5545]|
|Random whitened rotation|0.6667 [0.6633,0.6695]|0.6852 [0.6835,0.6868]|
|Learned whitened rotation|0.4246 [0.4195,0.4282]|0.4987 [0.4968,0.5004]|
|Signed scalar Top8|0.3465 [0.3423,0.3492]|0.4420 [0.4383,0.4459]|
|Signed 4Dblocks, two active|0.4661 [0.4638,0.4680]|0.5497 [0.5476,0.5518]|

Brackets are ranges over seeds, not confidence intervals; adaptive PCAis deterministic. The scalar baseline wins reconstruction in both heads. Learned rotations improve strongly on random rotations, but lose to adaptive PCAforagreement and outperform it forcapitals. Four-dimensional blocks trail the scalar baseline and are approximately competitive with adaptive PCAonlyforcapitals. These results do not show that overlapping groups or computational DAGs fail: those structures have not been fit in this head pilot yet, and a shared support assumption is distinct from arbitrary four-coordinate blocks.

Many validation-selected checkpoints are atstep3000, including every scalar model, so convergence is unestablished. These are finite-budget results. Rotation optimizes whitened L1rather than the reported writeMSE, which can favor different solutions. No task or output-forward measurements were used to train/select these checkpoints.

### Stability is informative but not a semantic test

Match learned decoder directions across all three seed pairs by maximum absolute cosine using a one-to-one assignment. Mean matched cosine is0.678agreement/0.663capitalforrotations, versus0.538/0.638for scalar models. Mean matched validation-support Jaccard is0.322/0.272forrotations and0.163/0.242for scalar models. Thus the reconstruction winner need not have the most reproducible coordinates.

For4Dblocks compare subspaces using mean squared principal cosines, a different metric from scalar cosine; matched means are0.304agreement/0.288capital, with support Jaccard0.130/0.139. These are insufficient to regard the blocks as uniquely identified manifolds. Decoder blocks are not rank-collapsed: their minimum singular values across all six fits exceed0.863. Median active-code covariance effective rank is2.68–2.79agreement and2.92–3.03capital. Poorer reconstruction therefore is not simply caused by the decoder collapsing each4Dblock to one direction.

Next run the frozen task/nuisance and natural-output tests for every seed and candidate, including adaptive PCA. For nonlinear sparse encoders, use reconstructed_donor minus reconstructed_receiver, rather than encoding the donor difference as if the map were linear. Separately replace native head activations for output-KL/CEtests. Keep the receiver's native baseline in difference-transfer tests. This distinguishes reconstruction gains from preserved causal differences, and determines whether the more stable rotation is a useful intermediate for overlapping membership discovery. Do not assign semantic block labels or fit hierarchical relations merely because a grouping exists.

Validation:four independent small-case unit tests pass—signed large-negative coordinate retention, joint block-energy selection, finite reconstruction gradients, and full-span whiten/rotate/unwhiten identity. Independent CPUfloat64evaluation of every selected model reproduces validationMSEwithin2.90e-6absolute (hard support boundaries can differ under precision). All selected budgets are exactly8nominal scalar slots; decoder normalization and rotation orthogonality assertions pass. All source/data/fit hashes unchanged, metrics finite.0whole-model forwards and original reservedtest untouched. No new unique research documents; corpus pool26112. Broad goal active.
