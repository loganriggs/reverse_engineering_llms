# Finite readout speed: feature recovery and transfer

All70fits completed their2000sweep budget; none failed its line search. These are finite-budget outcomes, not convergence claims. Five archived teachers, two starts each, seven predeclared methods; shared initial factors and optimal readout. This is a paired follow-up on earlier tasks, not a fresh-teacher replication. See READOUT_SPEED_RECOVERY_PLAN.md.

Each error is the median across five teacher means, with two starts averaged within teacher; transfer also averages four unseen source-task identities, each adapted using128labels. Recovery means every matched teacher/student quadratic feature has absolute Gaussian cosine>.99.

| Method | Readout speed | Source population MSE | Transfer MSE | All4features | Accepted hidden clock range |
|---|---:|---:|---:|---:|---:|
| joint | 1 | 0.145005 | 0.186676 | 1/10 | 20–20 |
| joint | 10 | 0.237292 | 0.347945 | 0/10 | 5.524–10.44 |
| joint | 100 | 0.74897 | 1.00502 | 0/10 | 0.5886–1.552 |
| reduced | exact optimum | 0.0553596 | 0.0940545 | 2/10 | 20–20 |
| split | 1 | 0.144391 | 0.186035 | 1/10 | 20–20 |
| split | 10 | 0.0442648 | 0.0619103 | 4/10 | 20–20 |
| split | 100 | 0.0555643 | 0.0944419 | 3/10 | 20–20 |

The split method takes a retracted hidden gradient step, then integrates the frozen-hidden readout flow exactly. It is not exact integration of both alternating blocks. Reduced steps solve the readout optimally at each candidate hidden state. Joint methods use simultaneous explicit gradient steps. All three differ from the earlier TRF/Gauss–Newton protocols.

Split speed10 has the lowest aggregate transfer median here; it beats joint speed1 on 5/5 teacher means, reduced on 3/5, and split speed100 on 3/5. The best median is descriptive across a predeclared panel, not a validated optimizer-selection rule. Failures and both starts remain included.

## Stability and the timescale confound

For fixed hidden features, write H=(Phi^T Phi+lambda I)/N, U*=H^{-1}Phi^T Y/N, and e=U-U*. An explicit readout update gives e_next=(I-tau H)e. In an eigenvector with eigenvalue h, its readout objective contribution is multiplied by (1-tau h)^2; nonincrease for every e requires0<=tau<=2/lambda_max(H). Exact readout flow gives e_next=exp(-tau H)e and multiplier exp(-2tau h), stable for every tau>=0. This is a fixed-hidden quadratic statement, not global hidden-network convergence.

In the joint speed100 arm, backtracking reduces accumulated hidden time to roughly.59–1.55 versus20 for joint speed1, all split arms and reduced. Hence equal sweep counts do not compare equal hidden optimization time. The bad fast-joint outcomes cannot be interpreted simply as evidence against fast readout adaptation. Exact readout flow removes this particular explicit-block stability restriction, but finite splitting and different trajectories remain.

## Verification and resources

Independent audit replays all7070stored states via literal forward objectives; final source and transfer use Wick moments independently of producer polynomial coefficients. Quadratic coefficient matching, SVD adaptation, equal initial states, scalar loss monotonicity and accepted clocks pass. All70final gradients independently agree with Torch autograd (max absolute error 5.78e-15); nine separate fixtures check exact readout flow against an augmented matrix exponential. Internal line-search decisions are not all independently replayed.

Each fit uses2000gradient evaluations. Reduced methods additionally solve readouts for candidate states, split methods use2000eigendecompositions/exact block flows, and joint methods pay extra forward evaluations for backtracking. Single-run CPU times and all call counts are saved in summary.json; subsecond timings are descriptive and unsuitable for a hardware speed ranking. No GPU was used.

## Decision

This supports testing moderate finite readout adaptation, but does not establish its superiority. Next use fresh teachers and match accepted hidden-time budgets for joint arms, with measured compute reported separately; include whole-task transfer and feature matching. Do not tune these same starts until every method recovers. Larger d64/h32 scaling remains open.

## Subsequent fresh-teacher confirmation

The moderate-speed advantage did not replicate on fresh teachers and initializations. See [READOUT_SPEED_CONFIRMATION_RESULTS.md](READOUT_SPEED_CONFIRMATION_RESULTS.md): split10 beats joint1 on2/5 and reduced on1/5 new teacher blocks, with worse median transfer. Do not use the earlier favorable panel alone to select an optimizer.
