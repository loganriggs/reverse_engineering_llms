# Exact tangent rank, parameter fibers, and local learning coefficient

The earlier numerical rank15finding is now exact at all six paired initializations of the d3/h3/q3depth2bilinear toy models. Full proof and scope are in sectionG of SHARING_COMPLEXITY_AND_FLOW.md; reproducible code is certify_bilinear_tangent_rank.py; matrices, pivot columns, nonzero modular determinants, and input hashes are saved in results/bilinear_tangent_certificate.

| Seed | Tied determinant mod2147483647 | Untied determinant mod2147483647 |
|---|---:|---:|
| 211001 | 502589261 | 831536948 |
| 211002 | 1210036133 | 1047832105 |
| 211003 | 845506621 | 396383612 |

Each determinant is a15×15minor of an output-evaluation Jacobian at15integer inputs. The15-dimensional homogeneous-quartic coefficient space bounds rank from above. Dyadic-rational parameter reduction has no denominator problem modulo the odd prime. Exact central differences at two finite-field offsets independently verify every derivative entry, exploiting degree at most2in each individual parameter. No numerical rank threshold is used in this certificate.

Consequences near these regular points:

- Both models can vary the output in every quartic direction locally. Untying does not add an output tangent direction here, though it changes the induced metric and nonlinear optimization paths.
- Fixed-function parameter fibers have local dimensions24(tied,39parameters) and42(untied,57parameters). This includes familiar symmetries and does not count distinct physical computations.
- With realizable Gaussian regression at the shared certified output and a smooth positive prior in a compact regular chart, both local learning coefficients are15/2. This follows from a15dimensional nondegenerate quadratic KL in coefficient coordinates, not from raw parameter count.

Limits: no global surjectivity, global RLCT, circuit identification, or generalization equality is proved. Singular representations elsewhere may matter globally. The theorem concerns the bias-free unnormalized toy models; the image models use RMSNorm and have a different function family.
