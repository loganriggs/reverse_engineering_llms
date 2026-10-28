## H645 — Geometric gain correction removes little of the residual context-shift error

H644 was progress: exact interaction-order expansion showed that small shifts admit accurate low-order approximations, while larger shifts require more orders. Once enough orders are retained, the dominant limitation is the frozen mean-shift prediction itself. H645 asks whether that residual error is mainly a magnitude mismatch.

Keep the H642 fresh panel, four probes and frozen H641 mean shifts. Use the true donor-minus-receiver normalized MLP-context change only for oracle diagnostics. For each receiver and layer, project that true change onto the frozen mean shift to obtain the Euclidean least-squares scalar gain. Also fit one shared scalar for the entire eight-layer stack. Both gains are shared across input probes; neither uses output responses. Compare their downstream effects with the unchanged predictive model and exact donor-context intervention.

Separately fit the optimal scalar rescaling of each full predicted output effect toward the exact effect, in the same receiver-probability-weighted centered-logit metric. This output scalar is target- and probe-specific and is strictly an oracle diagnostic. It is not a learned predictor and does not correspond generally to multiplying all context shifts by one scalar, because their composition is nonlinear in that scalar.

**Simple gain correction provides only modest functional improvement.** Errors below are relative to the exact context-replacement effect, so no context change has error1. Every oracle uses information unavailable to a prospective predictor.

| Condition, donor input/readout | Frozen mean | One geometric chain gain | Geometric gain per layer | Optimal output-effect scalar |
|---|---:|---:|---:|---:|
|Copy matched|0.416|0.368|0.364|0.357|
|Copy broad|0.454|0.440|0.440|0.431|
|Fact matched|0.388|0.389|0.387|0.373|
|Fact broad|0.202|0.203|0.202|0.190|

Matched copying benefits most: per-layer gains lower mean error0.416→0.364. Broad copying improves only0.454→0.440, and factual cells are effectively unchanged. A direct optimal output scalar still leaves0.357/0.431 copying errors and0.373/0.190 factual errors. That latter residual is the component of the target output effect not captured by the ray through the predicted effect; one output gain cannot remove it.

Receiver-only input/readout gives the same qualitative result:

| Condition, receiver input/readout | Frozen mean | One geometric chain gain | Geometric gain per layer | Optimal output-effect scalar |
|---|---:|---:|---:|---:|
|Copy matched|0.408|0.360|0.357|0.352|
|Copy broad|0.423|0.411|0.410|0.408|
|Fact matched|0.393|0.395|0.395|0.382|
|Fact broad|0.211|0.212|0.211|0.203|

**The context-space discrepancy itself is predominantly not a scalar mismatch in these cells.** Mean relative Euclidean context-shift errors before/after per-layer projection are0.388→0.357 for matched copying,0.436→0.427 for broad copying,0.402→0.399 for matched facts and0.267→0.266 for broad facts. Mean layer gains are0.885,0.920,0.962,0.988 respectively. Projection minimizes the context-space error along each frozen direction, but substantial orthogonal residual remains.

The gain that minimizes geometric error need not minimize output error. In factual cells it can slightly worsen functional prediction despite improving context-space reconstruction. This reinforces the need to judge any data-based representation by its chosen functional metric, rather than Euclidean alignment alone.

**What this does and does not establish.** The tested gains cannot explain away the current predictive residual, and a single output-effect scalar is insufficient. This supports looking for item-dependent directional corrections. However, it does not rule out a better eight-parameter functional gain model: jointly choosing eight layer gains to optimize final output error could exploit different directions and interactions than geometric projection. Neither the geometric-gain tests nor the one-output-scalar optimum is an upper bound on that larger family. No new predictive gain model has been fitted or validated here.

Refreshed20-hypothesis map:

1. Shared instruction-conditioned context shifts represent a transferable component of computation. Strong support from frozen prospective H642 results.
2. Item-dependent deviations in shift direction account for much of residual error. Supported geometrically and consistent functionally, but not uniquely established.
3. Incorrect overall shift magnitude explains most residual error. Weakened by the chain-gain and output-scalar controls.
4. Independent geometric gains per layer suffice. Weakened by their small improvements.
5. Joint output-optimized layer gains can substantially improve prediction. Open; requires a separate functional oracle test before learning gains.
6. Receiver features predict the remaining gains or directional corrections. Open; no such predictor has yet been validated.
7. Four discovery items give a noisy estimate of a truly fixed shift. Plausible contributor; more discovery items could improve mean estimates without eliminating item dependence.
8. Residual shift changes lie in a small reusable subspace. Open; fit only on discovery residuals and evaluate fresh items/directions.
9. A single direction per template pair is intrinsically sufficient. Weakened for high-fidelity prediction, though useful as an approximation.
10. A template-independent universal shift exists. Weakened by wrong-family norm-matched controls; broader sharing remains untested.
11. Input-direction-specific fitting explains success. Weakened: shifts were fitted without directional responses and transfer to new orthogonal probes.
12. A single layer is the computational unit. Weakened by H643 fixed ablations.
13. Independent layer contributions add into the transferable effect. Incomplete, especially for broad shifts; H644 identifies higher-order composition.
14. Broad formats require a categorically different kind of interaction. Weakened by norm-exchange diagnostics; amplitude explains a substantial part of the difference.
15. Final readout alone explains observed context dependence. Insufficient, though its effects depend strongly on the internal response.
16. Normalization is irrelevant to the mechanism. Too strong: earlier factorials show conditional contributions even though it is not the leading individual cross-wording factor.
17. Attention and MLP context play permanently separate semantic roles. Weakened by item-transfer coordination and format-dependent effects.
18. Candidate-set or competence failures explain all findings. Insufficient for validated response prediction, but still prevent claims about successful task algorithms.
19. Infinitesimal derivative fidelity guarantees finite behavior/loss fidelity. Unproven; local operator results do not establish a finite counterfactual model or accuracy improvements.
20. This representation is canonical across models or architectures. Unproven: the evidence concerns one model and fixed task/template contrasts, retaining native receiver context and full weights.

Next address hypothesis5 directly. The difference between the full predictor and each leave-one-layer-out predictor gives the exact local derivative with respect to that layer's gain at gain1, because the fixed-state chain is affine in each individual gain. Use those eight functional directions to form an oracle weighted least-squares correction, pooled across probes per context. First measure the linearized residual, then propagate the fitted gains through the actual composed chain to test whether the correction survives its interactions. Keep all target-informed gains labeled oracle. Only if this offers substantial improvement should a receiver-based gain predictor be pursued; otherwise prioritize directional residual models.

Audits:512 cached attention-stage reconstructions;864 batched JVP calls; no new whole-model forwards; frozen-predictor replay exactly matches H642; all source/runner/model hashes unchanged; per-layer geometric projection never increases its own Euclidean objective; finite metrics. No new natural-text documents; FineWeb remains26,112. All previous competence and local-response limitations remain. Broad goal active and incomplete.
