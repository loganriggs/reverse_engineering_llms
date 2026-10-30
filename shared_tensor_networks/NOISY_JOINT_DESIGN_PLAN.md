# Noisy joint-intervention design: bounded known-interface experiment

Protocol fixed before execution. This extends the exact blind-spot control, not the native-model work or the frozen h128 run.

Use eight duplicate feature pairs, with contrasts d_i=(h_2i-h_2i+1). Natural activations have all contrasts zero. An unknown quadratic dormant contribution is sum_{i<j} theta_ij d_i d_j, with 28 independent Gaussian coefficients. Any singleton physical-coordinate replacement leaves this contribution zero. Joint replacements can identify it. A common known ordinary output is subtracted from every measurement; learning that output is outside this experiment.

Twenty independent seeds generate teachers, candidate pools, observation noise and test interventions. Each pool contains eight random-angle unit-norm contrast interventions for each of 28 coordinate pairs. All joint strategies use this same pool without replacement, at budgets 14,28,56,112. Noise standard deviations are 0.05 and 0.2. Candidate perturbation norms are fixed; different products can still carry different information.

Compare uniform random selection, pair-balanced selection with random candidate per pair, and greedy log-determinant design (ridge information matrix 0.01 I). The latter sees candidate intervention coordinates, never target coefficients, outcomes or test interventions. Thus it is geometry-based design, not outcome-adaptive discovery. A singleton baseline uses the same number of noisy measurements but has identically zero design rows.

Fit coefficients using minimum-norm least squares with rcond 1e-12. Evaluate noise-free prediction MSE on 512 fresh unit-norm interventions with exactly three or four nonzero contrasts, which are support sizes absent from acquisition. Retain all seeds and failures; report paired seed-level comparisons and numerical rank. Also record coefficient error. This is known quadratic-family recovery, not recovery of unknown feature coordinates, graph structure, or arbitrary neural mechanisms. No test-driven tuning or additional rescue runs.

Validation: compare monomial features against an independent symmetric-matrix forward; compare fitted least squares against coordinate-wise scalar solutions (acquisition rows each involve one coefficient); verify singleton blindness, disjoint acquisition/test support sizes, fixed perturbation norm and no repeated candidates. Save script/protocol hashes with all records. CPU1, single-thread NumPy only. No GPU or changes to live training dependencies.
