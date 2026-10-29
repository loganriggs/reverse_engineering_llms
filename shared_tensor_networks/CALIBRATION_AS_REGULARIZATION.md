# Feature calibration changes the ridge prior

Let F be fixed raw features on N adaptation inputs, D a positive diagonal matrix of their estimated RMS scales, and Y the labels. Normalized-feature ridge fits

min_B ||F D^-1 B-Y||_F²/N +lambda ||B||_F².

With raw coefficients C=D^-1B, this is exactly

min_C ||FC-Y||_F²/N +lambda tr(CᵀD²C).

Thus changing calibration while keeping F fixed changes a diagonal penalty, not the represented linear function class. An underestimated RMS penalizes its raw coefficient less; an overestimated RMS penalizes it more. These statements refer to raw coefficient units, not a universal feature importance ranking.

For lambda>0 and strictly positive D, H=FᵀF/N+lambda D² is positive definite even if feature count exceeds N. The fitted raw coefficient is C=H^-1 FᵀY/N. For two calibration matrices D0,D1 on the same data,

C1-C0 = -lambda H1^-1 (D1²-D0²) C0.

Proof: subtract H1 C1=FᵀY/N and H0 C0=FᵀY/N, obtaining H1(C1-C0)=-(H1-H0)C0. This identity is exact for finite calibration changes, not just a Taylor expansion. Its consequences depend on design conditioning and alignment; norm errors do not imply a monotonic change in held-out error.

At lambda=0 this inverse argument need not apply when underdetermined. A coordinate-dependent minimum-norm convention may still make normalization matter. Do not generalize “calibration is irrelevant without ridge” to that setting.

check_calibration_penalty.py verifies the equivalent objectives and finite-change normal equations on all48new readouts in the depth3measurement replication. These are elementary ridge identities, not a novelty claim. They turn normalization uncertainty into a concrete change in the estimator and help distinguish it from evaluation sampling noise.
