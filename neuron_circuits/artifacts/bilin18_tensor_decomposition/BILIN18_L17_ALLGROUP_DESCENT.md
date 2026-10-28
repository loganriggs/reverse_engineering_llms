

## H303 — Eight-step all-group constrained descent

Start from frozen H300 refined rank64, fixed core96/native coefficients, original768 training/256 validation documents and identical18 head/payload objectives. Each iteration uses full separately energy-normalized gradients, unit-gradient minimum-norm convex combination, explicit negative-derivative verification, QR retraction and training-only Armijo constraints for ALL18 groups. Fixed descending steps.3/.15/.075/.0375/.01875/.009375, coefficient.0001, eight-iteration maximum. Afterward select the checkpoint maximizing minimum validation gain across18 groups, including step0. Validation does not choose step lengths.

Run stopped: **budget**. Selected **step1**. Accepted steps: 0.15, 0.3, 0.3, 0.15, 0.3, 0.15, 0.3, 0.075.

| Iteration | Step | Worst training gain | Mean training gain | Worst validation gain | Mean validation gain |
|---|---:|---:|---:|---:|---:|
|0|0|0.0000pp|0.0000pp|0.0000pp|0.0000pp|
|1|0.15|0.0712pp|0.2005pp|0.0152pp|0.1136pp|
|2|0.3|0.2491pp|0.5488pp|-0.0103pp|0.2848pp|
|3|0.3|0.3163pp|0.8690pp|-0.0589pp|0.4358pp|
|4|0.15|0.3436pp|1.0279pp|-0.0758pp|0.5139pp|
|5|0.3|0.4917pp|1.3136pp|-0.1552pp|0.6364pp|
|6|0.15|0.5294pp|1.4503pp|-0.1903pp|0.6949pp|
|7|0.3|0.5763pp|1.6975pp|-0.2894pp|0.7940pp|
|8|0.075|0.5997pp|1.7685pp|-0.3044pp|0.8234pp|

Locked gates: allgroup_validation_gain: FAIL, mean_validation_gain: FAIL, monotone_train: PASS. All-group validation requires>=.1pp in every group; mean validation requires>=1pp equal-group mean; monotone training audits the algorithm, not generalization.

Selected checkpoint improves 18/18 training and 18/18 validation groups. All logged accepted/rejected Armijo decisions independently replay; checkpoint rule and saved basis constraints verified.

At iteration8, every training group gains at least.600pp (equal-group mean1.769pp). Validation mean gains.823pp, but actual head6 declines.304pp; the other17 validation groups improve. Selection therefore retains step1, whose worst-group gain is.015pp and mean gain.114pp. This provides direct evidence that enforcing training compatibility does not ensure broad validation transfer.

**Interpretation and limits.** The trajectory tests accumulation of the H302 local opportunity within one fixed representation. Training constraints alone cannot guarantee validation improvement. A budget stop is not convergence, and low coverage does not establish irreducibility. This continuation consumes no fresh documents and changes no native weights or base program. Dense rank64 cross output cost remains7,077,888 coefficients. No intact behavior or semantic identification has been demonstrated.

**Next information-gain decision.** Compare the trajectory magnitude with the earlier rank gain and absolute missing cross response. Avoid indefinitely tuning the same optimizer if broad gains remain small. Move to a representation diagnostic: decompose the missed cross-response energy by output directions and test whether a compact output subspace captures it consistently across heads/payloads. This separates input-partner limitations from a shared output bottleneck without another fitting sweep. Use existing diagnostic documents first and reserve a fresh panel for a frozen materially improved candidate.

All jobs terminal. Broad research goal remains active.
