## H1013 — Equal-norm source/context responses are nearly separable despite multidimensional source geometry

Previous turn was progress: H1012 found that lexical expansion improves aggregate transfer but also shifts means and sacrifices old-domain fidelity. This discovery-only factorial separates edit magnitude, source direction, and recipient sensitivity before more predictor fitting.

Use24expanded-discovery noun pairs, both numbers, near-template with first attractor assignment:48recipient contexts. Sum five source deltas at pre-layer11subject position, normalize each to unit length, and transplant every same-number donor into every recipient:24x24matrix per number. All edits use the same pooled discovery median norm rho=1803.343. Apply signed scales+/-0.25,+/-1; four fixed random unit directions per recipient use identical norms. Native recipient residual, embedding recurrence and first-layer value bus remain fixed. No evaluation/test states used, no fitting to held-out outcomes.

Full rho is7.6–8.2%of recipient subject residual norm (median); quarter rho is1.9–2.1%. These are artificial source-position transplants and may be off manifold. Measuring both signs exposes nonlinear effects; this is not an infinitesimal-derivative claim.

| Quantity at full positive rho | Singular recipients | Plural recipients |
|---|---:|---:|
| Source edit damage RMS |3.849|2.308|
| Equal-norm random damage RMS |0.233|0.151|
| Source/random RMS ratio |16.5x|15.3x|
| Rank-one share of uncentered response energy |99.945%|99.964%|
| Recipient share of centered variation |24.3%|55.8%|
| Donor share of centered variation |74.2%|42.3%|
| Nonadditive donor-recipient interaction share |1.48%|1.88%|

Every donor has equal magnitude, so the donor effects cannot be attributed to differences in edit norm. Response is consistent with a source-specific potency times a recipient-specific gain. The grand mean dominates uncentered energy, so the99.9%figures must be read alongside the centered decomposition. Low centered interaction also holds at quarter scale (1.81/1.98%). Recipient changes include lexical residual state and prefix context; they do not isolate attention routing alone.

Source geometry is much less one-dimensional: uncentered rank-one direction energy76.4/83.1%, median pairwise cosine0.767/0.838, minimum0.361/0.488. Thus response similarity substantially exceeds vector similarity on this scalar task. This motivates a function-based equivalence class of directions, but does not prove a unique feature, a general-purpose manifold, or equivalence for other logits.

Post-result conditional cross-prediction diagnostic: use first four reference recipients and first four reference donors, chosen by existing order; rank-one pseudoinverse of their4x4intersection predicts the remaining20x20responses via the measured reference row/column responses. Relative error2.84%singular/2.54%plural. This consumes176of576native response cells per number and predicts400withheld cells. It is neither token-only nor activation-only prediction and the number4is a post-result diagnostic choice, not prospective validation. It demonstrates reusable response structure without claiming a deployed featurizer.

Nonlinearity matters: even/odd response norm ratio grows from6.8/10.0%at quarter scale to26.3/37.2%at full scale. Four-times-quarter odd response predicts full odd response to5.3/5.5%relative error; sixteen-times-quarter even predicts full even to12.6/2.8%. Therefore a scalar gain with linear magnitude alone is incomplete at realistic edit size. Same-donor diagonal mean and off-diagonal mean are nearly identical; within this discovery panel, effect does not require matching donor noun to recipient noun.

Audits: native full-model margin replay exactly0maxerror; saved-source recapture relative5.06e-7; equal edit-norm error2.42e-8; all finite; independent PyTorch SVD/ANOVA reproduces stats within3.34e-16and centered-energy fractions sum to1. Only expanded discovery rows selected.48native sequences,48prefix sequences,5424suffix sequences. Responses, directions, recipient indices and source hashes/config preserved; no process remains active. Natural-document pool26112unchanged; no reserved-test use; broad goal active.

Hypotheses narrowed: pure magnitude-only explanation is inadequate because equal-norm task directions beat random and donor identity explains substantial variation. Strong arbitrary donor-context conjunction is not needed for this one scalar task in these contexts. Leading interpretation is a shared functional read direction with recipient-dependent gain and finite-scale curvature. Competing explanations include correlated discovery prompts, a generic task-output bottleneck, and a shared component inherited from the rank-one embedding intervention itself. The latter is particularly important: we deliberately constructed all source changes from one number-decoder intervention, so discovered response rank is not evidence that arbitrary natural computation is rank one. Earlier100-output transport results and this single-output result measure different objects and are not contradictory.

Next fit the donor potency from source directions using discovery reference responses and predict recipient gain from clean context, with simple mean/norm controls. Freeze those factors and test new donor/recipient combinations without querying their suffix at inference. Then require selectivity against other output contrasts and multiple independently constructed task interventions before adopting shared atoms or groups. The concrete target for a featurizer is causal response coordinates, potentially simpler than activation geometry; no claimed manifold yet.
