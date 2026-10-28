## H1046 — Surrogate gradient implementation checks pass; hard-loss descent is unreliable

Previous goal turn was progress: H1045 showed that direct-change importance training worsened both full training loss and transfer. This turn performed a bounded training-only optimization audit, with no evaluation datasets loaded and no saved model updated.

Diagnostic batch fixed before results: first listed training pair in each of10families. Tested zero initialization and final160step weights for scalar, block, and shared selectors. Computed the exact implemented STE gradient of the mean bidirectional finite KL, without the tiny regularizer, and probed negative-gradient directions normalized to max-coordinate1at steps0.001,0.01,0.03,0.1. Also tested a0.03sign-gradient direction resembling Adam's first step. At final weights this is not a replay of the actual moment-preconditioned Adam update; it is a diagnostic direction.

At initialization, independently constructed the continuous local surrogate hard(w0)+soft(w)-soft(w0). Its forward anchor matches the hard baseline within5.97e-8. Central finite differences at0.001and0.01agree with autodiff's directional derivative:

| Selector | Maximum relative directional-gradient error |
|---|---:|
| Scalar |0.0872%|
| Block |0.0258%|
| Shared |0.00654%|

These checks support correctness of the implemented local surrogate derivative on the tested directions. They do not prove every gradient entry, every checkpoint, or every model path correct, nor do they validate the surrogate as a gradient of discrete hard selection.

All30probes predict descent from the surrogate. Actual hard-forward objective is unchanged in17, decreases in10, and increases in3. Constant loss under unchanged support is mathematically expected: coefficients/decoder are fixed and importance affects only discrete selection. It is not itself an implementation error. Once supports change, the surrogate can get direction or magnitude wrong. For example, block initialization sign-step predicts-0.03076but changes actual loss by+0.00585; shared final sign-step predicts-0.00659but actual change is+0.00510. Scalar initialization normalized0.1step predicts-0.06471versus actual-0.00102.

This identifies a concrete optimization mismatch capable of undermining gate learning. It does not prove that this mismatch alone caused the full160step failure; cyclic-family interference, step size, and data coverage remain possible contributors. The audit's ten-pair batch is not the full training distribution. H1045's full80pair initial/final loss comparison remains the evidence that the completed fitting recipe failed to optimize its target.

Verification:1040suffix calls, source/checkpoint hashes unchanged, finite losses, no fitting updates persisted, no evaluation rows loaded. Reused exact deployed decoder/forward definitions and separately assembled the continuous local surrogate for finite differences. No soft-support output was presented as a valid sparse candidate.

Decision: stop interpreting these importance-only STE fits as tests of whether causal supervision or shared representations can work. Retain the successful untrained direct-change encoding controls. Next fixed experiment should freeze the support rule and encoder coordinates, and learn decoder directions through a genuinely continuous finite causal objective. Compare scalar/block/shared support with identical parameter counts and unchanged evaluation access. A continuous decoder is a learned action dictionary, not an optimized manifold or learned taxonomy; shared atoms still decode once. Verify derivatives on a small training case and measure full-training initial/final hard-forward loss before interpreting transfer. If it cannot reduce actual training loss, investigate that failure rather than claiming a representation result. This isolates learnable geometry from the unreliable discrete gate surrogate without another threshold/duration sweep.
