## H646 — A functional gain oracle improves fitted directions but transfers poorly to a withheld direction

H645 was progress: geometric gain correction and one output-effect scalar left substantial residual prediction error. Those controls did not rule out a jointly output-optimized gain for each layer. H646 tests that larger family locally and verifies its actual composed effect.

At the frozen predictor's gain vector of all ones, the full response minus the response omitting layer l's shift is the exact partial derivative with respect to gain l. The fixed-state chain is affine in each individual gain even though it is not jointly affine in all gains. Use these eight functional directions from H643 to construct a local gain Jacobian.

For each of64 prompt contexts, fit an unconstrained eight-gain correction to the exact context-intervention target, pooling original/fresh0/fresh1 probes. Normalize each probe's squared-error contribution by its exact context-effect energy to avoid letting a larger response dominate. Keep fresh2 completely out of the gain fit. Fit receiver-input/readout and donor-input/readout conditions separately. Thus every gain vector uses target information for its own prompt and is an oracle, not a receiver-based predictive model.

Solve the weighted linearized least-squares problem with a fixed pseudoinverse tolerance1e-10. Then propagate both the full correction and a predeclared half correction through the actual composed chain. No clipping, line search or adaptive choice between full and half steps occurs. This is one local Gauss–Newton-style step, not global optimization of the nonlinear gain family.

**Fitted-direction improvement is modest; withheld-direction improvement is inconsistent.** Errors are relative to the exact context-replacement effect, with no context shift giving error1. Donor-input/readout results:

| Condition | Fitted probes: frozen mean | Fitted probes: actual full correction | Withheld fresh2: frozen mean | Withheld: linearized full correction | Withheld: actual full correction | Withheld: actual half correction |
|---|---:|---:|---:|---:|---:|---:|
|Copy matched|0.434|0.359|0.361|0.346|0.345|0.334|
|Copy broad|0.437|0.414|0.505|0.499|0.498|0.496|
|Fact matched|0.374|0.345|0.429|0.441|0.441|0.429|
|Fact broad|0.210|0.165|0.179|0.188|0.188|0.176|

Receiver-input/readout results:

| Condition | Fitted probes: frozen mean | Fitted probes: actual full correction | Withheld fresh2: frozen mean | Withheld: linearized full correction | Withheld: actual full correction | Withheld: actual half correction |
|---|---:|---:|---:|---:|---:|---:|
|Copy matched|0.428|0.356|0.348|0.329|0.328|0.320|
|Copy broad|0.420|0.384|0.433|0.435|0.436|0.428|
|Fact matched|0.387|0.361|0.413|0.425|0.425|0.413|
|Fact broad|0.205|0.177|0.228|0.243|0.243|0.231|

The actual full correction improves fresh2 in only28/64 donor-endpoint contexts and30/64 receiver-endpoint contexts. On average it slightly helps copying in some cells and worsens the two factual cells. The half step sometimes moderates the degradation, but was not selected as a preferred model. A single withheld direction is a limited generalization test; repeated contexts are not independent directional replications.

**The poor transfer is not mainly a failed linear approximation.** Fitted-probe cell means for linearized and actual full corrections agree closely; the largest individual difference in their error metrics is0.0149 for donor endpoints and0.0155 for receiver endpoints. Actual composed rollout therefore does not destroy a large linearized promise—the linearized held-out gain correction was already weak or harmful. This comparison concerns their error norms, not an assertion that the full response vectors are identical.

The numerical solve is also well behaved: maximum Gram-matrix condition number44.59, maximum relative normal-equation residual2.21e-15, and fitted gains range0.266–1.758. No extreme or negative gains are needed. Pooled fitted-probe linearized squared loss does not increase for any prompt/endpoint, as required by the least-squares solve. The optimization objective is pooled relative squared error, not the mean of relative error norms reported in the table, so individual probes need not improve.

The result weakens the hypothesis that the remaining context-shift error can be repaired by simple layerwise magnitude calibration shared across directions. It strengthens the priority of testing additional context-change directions. It does **not** prove that all gain-based explanations fail: more fit probes, repeated optimization, nonlinear constraints or a different gain parameterization might perform differently. Nor does it prove which directional representation is best. The tested local gain family simply offers limited benefit even with target access, and weak cross-direction transfer, so building a prospective gain predictor is not currently the most informative next step.

No predictive improvement is claimed. The frozen H641 model remains the validated predictor; these gains consume exact target responses for the same contexts. Fitting on three probes while withholding a fourth is a directional diagnostic, not a held-out-context evaluation. Existing task-competence, fixed-template and local-derivative limitations remain unchanged.

Next fit a small basis of **directional context-shift residuals** using only the original discovery items. Keep the frozen mean as a baseline, concatenate the eight layer residuals so coefficients describe a coordinated change across layers, and predeclare ranks1–3. On the fresh panel, use oracle coefficients only to test whether that discovery span can represent the missing shift and its functional effect across new probes. If even this representational ceiling fails, avoid training a predictor into an inadequate basis. If it succeeds, the next task is predicting its coefficients from receiver information without donor access. This separates basis capacity from coefficient prediction and retains strict discovery/evaluation provenance.

Audits:512 cached attention-stage reconstructions;768 batched JVP calls; no new whole-model forwards; all source/runner/frozen-model hashes unchanged; finite metrics; least-squares objective and normal equations independently checked; full and half corrections actually propagated. No new natural-text documents; FineWeb remains26,112. Broad goal active and incomplete.
