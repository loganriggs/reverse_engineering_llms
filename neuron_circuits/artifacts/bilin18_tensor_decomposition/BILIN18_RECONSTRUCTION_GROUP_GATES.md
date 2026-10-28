## H728 — Exhaustive group selection closes only about one tenth of the fidelity gap

Question: did H726/H727 overlapping groups perform poorly mainly because the gate selected groups by coefficient energy rather than reconstruction error?

We froze all six scalar dictionaries and all 54 membership sets. For each activation, we enumerated every distinct pair of groups: 8,128 candidates for each overlapping grouping, 496 for each disjoint partition. Each candidate uses the exact union of its atoms, the original encoder coefficients, the nonorthogonal decoder, and its output bias. The chosen pair minimizes input reconstruction error across this complete candidate set. This is an inference algorithm using the input activation, not parameter fitting on validation targets, but it is substantially more expensive than ordinary gating.

Evaluation uses the same 32 hash-selected validation documents as earlier functional tests, at all 16 sampled positions: 512 vectors per head. It is a limited, reused panel. Every method in this comparison uses the same vectors; values differ from H726's larger validation sample.

| Head and memberships | Scalar Top-8 MSE | Energy-gated MSE | Best-pair MSE | Gap to scalar closed |
|---|---:|---:|---:|---:|
| Agreement, raw groups | 0.3386 | 0.4529 | 0.4377 | 13.3% |
| Agreement, conditioned groups | 0.3386 | 0.4684 | 0.4547 | 10.6% |
| Capitals, raw groups | 0.4329 | 0.5291 | 0.5191 | 10.4% |
| Capitals, conditioned groups | 0.4329 | 0.5347 | 0.5236 | 11.0% |

All numbers average three dictionary seeds. Exact selection also helps degree-preserving rewires, but inferred groups retain their advantage. Best-pair MSE for raw/conditioned rewires is 0.4612/0.4737 for agreement and 0.5482/0.5502 for capitals. Disjoint partitions remain worse: 0.5154 and 0.5918.

Exhaustive selection changes the union support on roughly 65–71% of examples for inferred groups, yet closes only 10–13% of their reconstruction gap to scalar Top-8. The exact selected unions contain 7.97–7.98 unique atoms on average. Matching the scalar comparator's active count separately on each example barely changes its MSE: 0.3388–0.3393 agreement and 0.4333–0.4340 capitals. Therefore fewer unique active atoms do not account for most of the observed gap.

Interpretation: this specific energy gate is not the main explanation for the fidelity deficit. With these frozen coefficients and memberships, even the best possible two-group selection remains substantially worse than free scalar Top-8. This narrows the explanation to membership constraints and/or coefficients trained for freely selected scalar supports. It does not rule out jointly learned overlapping dictionaries, larger group inventories, different group sizes, or continuous manifold structure.

Correctness: two tests pass, including an independent atom-by-atom brute-force comparison with a nonorthogonal decoder, nonzero bias, and overlapping supports. A strict initial float32 absolute audit failed at 0.000244 SSE. Rather than discarding this discrepancy, we retained its logs and audited selected reconstructions in double precision, normalized by input energy. Maximum relative scoring error was 1.07e-7; no selected best-pair reconstruction was worse than energy selection in the double-precision comparison. All hashes remained unchanged and all results were finite. Zero model forwards, new documents, and reserved-test evaluations were used.

Next: test the coefficient-mismatch explanation by least-squares refitting on frozen supports, compared against the same refitting on scalar supports. Treat this as a diagnostic, not evidence of discovered manifolds. If that does not explain the deficit, prioritize better membership learning or scalar-coordinate geometry rather than polishing this gate. Functional and selective shared-component tests remain required before accepting a featurizer, followed by fresh diverse-task validation.
