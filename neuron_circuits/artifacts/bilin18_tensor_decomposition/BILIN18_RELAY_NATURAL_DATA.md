## H851 — Natural-text training data and PCA baselines at the actual causal relay site

Captured raw preMLP4 states on the original2048 training documents at16 sampled positions (32,768 states) and the same512 H832 development document/position selections. This is the exact after-attention/before-MLP site used in the contextual relay experiments. No task activations, labels, or supervised directions were loaded for this stage. Original reserved-test examples were not evaluated.

Fit training-only global shape mean, per-token shape means, and global/residual covariance PCA through rank128. Shape is rawstate divided by its RMS including float32 epsilon. Residual means token-centered shape; unknown development tokens fall back to the global mean. This centering is not proof that all lexical effects are removed. PCA is centered by the corresponding training baseline, not fit on tasks.

Residual-PCA development relative MSE: rank1 .9712, rank8 .8631, rank48 .6654, rank64 .6173, rank128 .4837. Global-PCA values use a different global-centered denominator, so they should not be directly compared as identical reconstruction objectives. Both bases will be tested with the same causal task-intervention protocol later.

Audits:2,560 five-block-prefix forwards; no full-model/task/test forwards; all captures finite; source/model hashes unchanged; PCA orthogonality error2.34e-15. Natural-document pool remains26,112. These are reused research partitions, not fresh confirmation. Next fit all ten H836 architecture/seed combinations from scratch at this site, selecting only by natural-text development reconstruction error.
