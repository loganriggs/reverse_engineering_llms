# Context-conditioned full-output response geometry

Bases depend on unperturbed state and native weights, not on the evaluated finite response. Oracle coverage and tangent prediction are distinct.

| Basis | Small-edit coverage | Small-edit prediction | Full-edit coverage | Full-edit prediction |
|---|---:|---:|---:|---:|
| full_tangent | — | 1.000 | — | 0.953 |
| matched4 | 0.398 | 0.398 | 0.338 | 0.321 |
| wrong_context4 | 0.052 | 0.052 | 0.032 | 0.024 |
| wrong_source4 | 0.131 | 0.131 | 0.110 | 0.103 |
| global4 | 0.124 | 0.124 | 0.077 | 0.061 |
| matched16 | 0.542 | 0.542 | 0.491 | 0.471 |
| wrong_context16 | 0.086 | 0.086 | 0.061 | 0.051 |
| wrong_source16 | 0.187 | 0.187 | 0.161 | 0.152 |
| global16 | 0.185 | 0.185 | 0.127 | 0.106 |
| matched64 | 0.811 | 0.811 | 0.779 | 0.756 |
| wrong_context64 | 0.161 | 0.161 | 0.135 | 0.121 |
| wrong_source64 | 0.302 | 0.302 | 0.277 | 0.264 |
| global64 | 0.287 | 0.287 | 0.231 | 0.207 |
| matched128 | 1.000 | 1.000 | 0.980 | 0.953 |
| wrong_context128 | 0.233 | 0.233 | 0.207 | 0.191 |
| wrong_source128 | 0.398 | 0.398 | 0.375 | 0.358 |
| global128 | 0.379 | 0.379 | 0.328 | 0.301 |

Locked gates: {'locally_compact': False, 'context_specific': True, 'source_specific': True, 'small_change_prediction': True}. Analytic derivative relative error versus independent autograd JVP: 7.45e-07.

Per state/source construct1152x128 derivative matrix from native weights and perform SVD. A rank-r chart stores1152*r numbers per state/source. This is expensive model-derived geometry, not a recovered cheap router, shared global feature list, or semantic unit.

The native full tangent supplies coefficients for every projected predictor; wrong-chart controls vary only the output basis. No projected predictor uses the changed output. Rank128 corresponds to the full source-local tangent span, not a128-dimensional global code.


## H235 — Source-response output spaces rotate with context, but remain moderately broad

**Question and frozen design.** H233 found broad fixed-output response geometry.
At128 fresh sites (rows19720–19751, four positions), all nine source heads were
tested at5% and100% removal. For each unperturbed state/source, the native bilinear
MLP derivative composed with the exact RMS derivative was restricted to an
orthonormal basis of that head's raw output span. Left singular vectors define
output charts without consulting finite-response targets. Controls use a chart
from the next text row at the same position, the next head at the correct state,
or the frozen global all-head response basis. Leading SVD columns were QR
orthonormalized, preserving every leading span. This same-panel numerical audit
rerun is not an independent replication.

Each chart has two scores: projection of the exact finite response (oracle
geometric coverage), and projection of the native full tangent prediction
(no changed-output inputs). Wrong-chart predictors still use the correct native
tangent coefficients; the controls isolate output-basis geometry.

| Chart | Rank | 5% coverage | Full coverage | Full prediction fidelity |
|---|---:|---:|---:|---:|
| State/source matched | 4 | .398 | .338 | .321 |
| State/source matched | 16 | .542 | .491 | .471 |
| Wrong context | 16 | .086 | .061 | .051 |
| Wrong source | 16 | .187 | .161 | .152 |
| Frozen global | 16 | .185 | .127 | .106 |
| State/source matched | 64 | .811 | .779 | .756 |
| Wrong context | 64 | .161 | .135 | .121 |
| Frozen global | 64 | .287 | .231 | .207 |
| State/source matched | 128 | 1.000 | .980 | .953 |
| Wrong context | 128 | .233 | .207 | .191 |
| Frozen global | 128 | .379 | .328 | .301 |

**Discriminating result.** Both context- and source-specificity gates pass.
At full scale, matched16's coverage advantage over wrong-context16 is .430 with
conditional row-bootstrap95% [.387,.486]; over wrong-source16 .329 [.299,.354];
over global16 .364 [.325,.425]. But the strong locally-compact gate fails:
rank16 does not reach80% coverage. Rank64 is needed for about78% full-edit coverage.
Within-source isotropic tangent energy is broader still: rank16 captures roughly
24–28% across heads. Actual source payloads align more with high-gain directions
than an isotropic capability metric would imply. Head3 matched16 full coverage
is .651; other heads range .243–.538. This is not a universal tiny feature set.

**Nonlinearity versus moving geometry.** The full tangent predicts5% edits at
.999918 fidelity, with interval [.999892,.999945], and full edits at .953434
[.932423,.972937]. The exact native derivative matches independent double-precision
autograd JVPs to relative RMS7.45e-7. Thus large pooled output dimension is not
mainly a failure of first-order prediction on these edits: the source-to-output
map varies with context. Finite curvature still accounts for the residual
full-edit error. Rank128 matching the small-edit response is expected from the
128-dimensional source span; it is not evidence for128 globally shared features.

**Cost and ontology boundary.** Each state/source chart builds a1152×128 matrix
and performs SVD. Its rank64 output basis alone has73728 values per state/source.
The dominant dense derivative multiplication is about0.68 billion multiply-adds
per state/source, before SVD, using the full native weights. These are expensive
model-derived charts, not a compressed executable model or recovered router.
They establish state/source-conditioned response geometry and support T09/T10
operator descriptions over a handful of fixed output axes, while leaving
semantic meaning, cheap representation, uniqueness and external validity open.

**Controls/artifacts.** Head reconstruction3.82e-7; raw source-span replay4.33e-7;
after ordered QR, max chart orthogonality error2.03e-6. Sixteen targeted tests pass,
including an independent derivative/RMS-radial-term check. The report, full JSON,
per-head scores, singular values, per-row errors/coverage and PNG/SVG are saved.
All runs completed. Config and bases are target-blind with respect to the finite
response; row-bootstrap intervals condition on this model, corpus and fixed
context/source controls.

**Next information experiment: compress context dependence, not fixed outputs.**
The expensive charts show what can be represented, not how cheaply. Test whether
a small shared context code can reproduce the full bilinear response program
without a separate SVD per state. Fit context encoders only on the existing2048
natural input sites; compare PCA and matched random codes over ranks0/16/64/128/
256/512/full. Use the exact quadratic identity for normalized endpoints:
F(x')−F(x)=2B((x+x')/2,x'−x). Compress the midpoint/context role while retaining the
observed change role, and evaluate full-output finite-response fidelity on fresh
rows19752 onward. Clearly count normalization and change-role information.
The resulting one-role approximation need not be an integrable state function:
compare its symmetrized, globally integrable counterpart and test triangle cycles.
Also report natural-value fidelity for any claimed state decoder. If context-role
compression works across heads, it supports reusable conditional tensor structure;
if many context reads are required, the observed moving charts do not imply a
small shared program. No output-target fitting or held-out response selection.
A weight/operator-aware context metric is a possible controlled comparator, but
must be derived, validated and locked before evaluation rather than introduced
after seeing a failed PCA result. Keep construction/storage costs explicit.
