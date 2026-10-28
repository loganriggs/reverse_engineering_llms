

## H316 — Frozen interaction readouts across heads and payloads

Apply all18 H315 frozen interaction readouts to all18 target groups on the H315128-document panel, now explicitly reused diagnostic data. Each source retains its reference descriptors, fitted dual coefficients and kernel normalization. Common core96/partner64 feature geometry and native targets unchanged. No fitting or best-source selection. All source-artifact hashes locked and verified.

| Target family | Same-group fidelity | Mean other-head fidelity | Retention | Same-head other-payload fidelity |
|---|---:|---:|---:|---:|
|actual|43.11%|-1.62%|-3.8%|6.84%|
|angle|25.20%|2.48%|9.8%|9.00%|

Locked gates: other_head80retention: FAIL, cross_payload80retention: FAIL. Both require retaining>=80% of same-group fidelity in BOTH target families. Other-head mean averages squared errors of eight separately applied sources within the same payload family; it is NOT an ensemble prediction. Same-head other-payload swaps actual↔angle supervision. Pooled fidelity weights target groups by their energy; full18×18 matrix saved to expose heterogeneity.

Corrected same-group fidelity reproduces43.11% actual/25.20% angle, but other-head application averages−1.62%/2.48%. Opposite-payload same-head transfer reaches only6.84%/9.00%. Both80%-retention gates fail. Thus document transfer in H315 does not imply intervention-distribution transfer; narrow training support is a leading explanation under the mathematically shared cross polynomial.

**Interpretation limits.** The exact native cross-response polynomial is shared across heads in this fixed geometry. Different fitted readouts can nevertheless differ because their reference interventions cover different feature directions and ridge regularization resolves an underdetermined problem differently. Weak transfer therefore does not establish distinct fundamental units or mathematically incompatible head-specific rules. Strong transfer would support reusable estimation, but would not alone establish semantic labels or economical storage. This test is about learned coefficient coverage under a shared representation.

An initial run omitted the layer setting, causing the collector to default to layer8. The diagonal replay assertion rejected the run before result arrays were saved. The failed log is preserved; layer17 is now explicit and asserted. Only the corrected run is reported.

**Audit.** All324 matrix entries independently replay from saved per-document errors. The18 diagonal prediction error arrays reproduce H315 within relative1e-6, confirming unchanged query construction, random payloads, reference normalization and saved readout execution. Source hashes checked before evaluation. No parameter update or new native-model training; all jobs terminal. H315 fresh status is not claimed again here.

**Next discriminator.** Compare mixed-head supervision with specialist supervision at equal total reference count and fixed feature/kernel construction. Sample reference documents and heads by a locked balanced scheme; test whether shared coverage recovers specialist performance without storing18 readouts. Include specialist-to-own-head performance, transfer across heads and per-payload results. This separates lack of feature coverage from the hypothesis that one compact fitted rule is intrinsically inadequate. Any shared model must report storage and prediction cost, and eventually pass fresh-document and downstream tests. Broad research goal active.
