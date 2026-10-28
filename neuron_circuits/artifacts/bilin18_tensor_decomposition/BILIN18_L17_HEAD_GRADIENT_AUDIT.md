

## H302 — All-head local descent exists; pooled weighting misses it

Audit rank64 proxy and refined bases, fixed core96/native coefficients, original768 training and256 validation documents, same payload seed20270121. Accumulate18 full gradients (each head × actual/angle), each normalized by its own target energy, then project into the horizontal tangent space. Compare three unit-Frobenius directions: negative sum of unit pooled-family gradients (H299–300 rule), negative mean of18 separately normalized gradients, and negative minimum-norm convex combination of18 unit gradients. The last uses a simplex Frank–Wolfe solve; report actual derivative signs rather than assuming solver convergence establishes descent. Fixed QR-retracted steps.03/.3 and central finite difference±.003. No new checkpoint promotion or fresh-document use.

Training head energy is uneven: angle head6 contributes48.36%, head1 another21.08%; angle heads3/5/7/8 together only6.22%. Those four are exactly the angle heads that regressed under the earlier refinement. Association alone is not proof that weighting caused all accumulated regressions.

Both simplex solves converged with reported dual gaps<1e-10 (7086/4760 iterations proxy/refined). Squared minimum-combination norms are.075917/.039110, well above zero. Orthogonal and exactly opposing synthetic-gradient checks pass. Central finite differences agree within the locked1e-4 absolute tolerance for all tested directions. Float32 gradients and finite differences are numerical evidence, not an exact symbolic proof.

| Basis | Direction | Negative loss derivatives /18 | Step.03 training improvements /18 | Step.03 validation improvements /18 | Worst validation gain |
|---|---|---:|---:|---:|---:|
|proxy|pooled_family|18|18|18|0.0027pp|
|proxy|equal_head|18|18|18|0.0139pp|
|proxy|common_min_norm|18|18|18|0.0201pp|
|refined|pooled_family|14|14|13|-0.0066pp|
|refined|equal_head|17|16|16|-0.0038pp|
|refined|common_min_norm|18|18|18|0.0045pp|

At the refined basis, common-direction derivatives are negative for all18 groups (range−.02185 to−.00767). Step.03 improves all18 training and validation fidelities, with worst gains.02284pp training/.00454pp validation. The larger step.3 improves only17/18 training groups, despite all negative initial derivatives: first-order compatibility does not guarantee a finite step remains beneficial. It improves18/18 validation groups, which cannot substitute for the intended training constraint. Equal-head averaging alone does not ensure common descent: only17/18 derivatives are negative at the refined basis, and16/18 improve at step.03.

Locked diagnostic gates: existence of a strictly common tested direction at BOTH bases PASSES; finite-difference absolute error<1e-4 PASSES. These are local numerical diagnostics, not compactness, meaningful magnitude, convergence, fresh transfer, or semantic-identification gates. Mean refined validation improvement at common step.03 is only.02393pp; at step.3 it is.21180pp. No80% success is claimed.

**Inference.** At these tested points, the head demands are not locally incompatible: one direction improves every objective. Pooled-family descent can sacrifice small-energy heads, and merely giving heads equal loss weights does not fully prevent this. Objective aggregation and finite-step selection are therefore sufficient local explanations for some missed opportunities. This does not prove the whole H300 trajectory could retain every head, nor rule out conflicts elsewhere. Rank/cost limitations remain substantial.

**Next discriminator.** At fixed rank64, run a short common-descent trajectory from the frozen refined basis, using18-group training-only Armijo constraints and all18 normalized gradients. Log rejected steps and convergence margins; choose validation checkpoints only after the trajectory. Compare against the frozen starting basis on every group. This tests whether the verified local opportunity accumulates into broadly useful improvements or rapidly stalls. Keep the budget fixed and do not interpret budget exhaustion as convergence; use a new confirmation panel only if a meaningful gain warrants it. A representation-family comparison remains necessary before declaring distributed irreducibility.

Startup failed before model loading because SciPy was absent; preserved startup log, replaced the small simplex solver with NumPy Frank–Wolfe, verified two known cases, then reran successfully. All jobs terminal. Existing validation only; H301 fresh panel unused here. Native model/compiler/base unchanged. Broad research goal active.
