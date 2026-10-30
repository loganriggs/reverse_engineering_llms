# Frozen-feature transfer decomposition at h64

Recorded before this analysis's results. Reuse all60audited h64models, their existing four whole-task holdouts, and their already-fitted8,192-example readouts. No representation retraining, new sampled labels, penalty tuning or seed selection. This extends the earlier smaller-width population-floor diagnosis to the current task/input grid.

For learned product features phi, target y and coefficient u, let G=E[phi phiᵀ], b=E[phi y], e=E[y²]. Then R(u)=uᵀGu−2uᵀb+e. For any reference u*, the exact identity is

R(u)=R(u*)+(u−u*)ᵀG(u−u*)+2(u−u*)ᵀ(Gu*−b).

If Gu*=b, the last term vanishes, separating representation error from excess readout error. The reference uses teacher-known Gaussian population moments and is privileged; it is not a practical finite-label estimator. Excess readout error includes finite-design misspecification and noise, not noise variance alone.

Normalize the Gram to unit feature second moments. Retain eigenvalues above1e-10and1e-12times the maximum, as in the prior smaller-width diagnosis. Report both cutoffs and ranks. A truncated solution is optimal only in its retained numerical subspace; keep the cross term explicit and report it rather than treating every refit as an exact global floor. Replay the1e-12solution with an independent least-squares solver. Compare existing finite-label error with analytic error using the saved coefficient arrays.

Compute all four tasks for every model and control. Aggregate starts/tasks within each teacher before medians; preserve all teacher differences and failures. Main question: do high-input near-recovery models retain a significant population approximation error, or is their remaining transfer error primarily readout estimation? The h128scientific protocol is unchanged. No optimizer rescue follows automatically.
