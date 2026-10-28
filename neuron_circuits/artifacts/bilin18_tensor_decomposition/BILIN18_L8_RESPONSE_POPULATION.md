# Full L8 response population and nonlinear suffix coverage

Output projections use exact native responses: these are communication/sufficiency measurements, not executable input decoders.

| Output basis | Natural write coverage | All-head response coverage | Head1 | Head3 |
|---|---:|---:|---:|---:|
| selected4 | 0.079 | 0.051 | 0.043 | 0.141 |
| response_even4 | 0.011 | 0.035 | 0.005 | 0.014 |
| response_even16 | 0.064 | 0.098 | 0.050 | 0.136 |
| response_even64 | 0.180 | 0.203 | 0.152 | 0.252 |
| response_even128 | 0.290 | 0.301 | 0.259 | 0.336 |
| response_all4 | 0.031 | 0.077 | 0.074 | 0.178 |
| response_all16 | 0.076 | 0.130 | 0.112 | 0.214 |
| response_all64 | 0.200 | 0.237 | 0.246 | 0.295 |
| response_all128 | 0.310 | 0.338 | 0.351 | 0.378 |
| natural4 | 0.077 | 0.046 | 0.032 | 0.091 |
| natural16 | 0.156 | 0.101 | 0.088 | 0.167 |
| natural64 | 0.286 | 0.210 | 0.210 | 0.269 |
| natural128 | 0.390 | 0.304 | 0.314 | 0.345 |
| weight4 | 0.050 | 0.061 | 0.046 | 0.180 |
| weight16 | 0.113 | 0.108 | 0.108 | 0.229 |
| weight64 | 0.239 | 0.219 | 0.233 | 0.313 |
| weight128 | 0.354 | 0.329 | 0.346 | 0.404 |
| residual4 | 0.079 | 0.051 | 0.043 | 0.141 |
| residual16 | 0.160 | 0.103 | 0.101 | 0.182 |
| residual64 | 0.305 | 0.224 | 0.231 | 0.286 |
| random64 | 0.056 | 0.056 | 0.055 | 0.061 |

## Nonlinear suffix response fidelity

The denominator is native-response minus no-response with the SAME destination head removal. The intact model is a separate reference.

| Basis | Head1 logits | Head1 CE | Head3 logits | Head3 CE |
|---|---:|---:|---:|---:|
| no_response | 0.000 | 0.000 | 0.000 | 0.000 |
| replay_full | 1.000 | 1.000 | 1.000 | 1.000 |
| selected4 | -0.015 | 0.070 | 0.078 | -0.060 |
| natural64 | 0.151 | 0.331 | 0.264 | 0.016 |
| natural128 | 0.298 | 0.504 | 0.261 | 0.354 |
| response_even64 | 0.035 | 0.248 | 0.211 | 0.166 |
| response_all4 | 0.019 | 0.113 | 0.135 | -0.529 |
| response_all16 | 0.069 | 0.092 | 0.180 | -0.552 |
| response_all64 | 0.112 | 0.269 | 0.249 | -0.153 |
| response_all128 | 0.304 | 0.372 | 0.354 | 0.250 |
| weight64 | 0.130 | 0.472 | 0.257 | -0.036 |
| random64 | 0.024 | 0.109 | 0.049 | 0.181 |

Projection orthogonality error 4.77e-07; native head replay 3.91e-07; full-response suffix replay max logit error 2.24e-05.

Finite intervals in JSON resample32 complete rows, each containing2 positions. Ratios of mediation effect energy to total head effect energy are scale diagnostics, not additive causal shares; cancellation can make them exceed1. No semantic feature or unique basis is identified.


## H233 — The four-axis result has narrow population coverage and source-selection bias

**Question.** Do the carefully studied four L8 coordinates represent the full MLP
response population? Frozen output bases were tested on512 fresh sites across
all nine individual head removals. Natural-write bases use original2048 discovery
writes; response bases use cached512-row discovery responses, fit either even
heads or all heads. Weight and residual bases and a random64 control are included.
These projections receive the exact response and measure output communication
coverage, not an executable input-to-output decoder.

| Basis | Natural centered-write coverage | Pooled all-head response coverage | Head1 response | Head3 response |
|---|---:|---:|---:|---:|
| Four studied axes | .079 | .051 | .043 | .141 |
| Natural64 | .286 | .210 | .210 | .269 |
| Even-response64 | .180 | .203 | .152 | .252 |
| All-response64 | .200 | .237 | .246 | .295 |
| Weight64 | .239 | .219 | .233 | .313 |
| Random64 | .056 | .056 | .055 | .061 |
| Natural128 | .390 | .304 | .314 | .345 |
| All-response128 | .310 | .338 | .351 | .378 |
| Weight128 | .354 | .329 | .346 | .404 |

The four studied axes account for only5.1% of full head-induced MLP response
energy. Head3 contributes44.2% of energy inside that projection but16.1% across
the full1152-dimensional output. Full-output head1 contributes30.8%, and head8
17.6%, both above head3. Head0 contributes only0.097%. The head3 concentration
that guided earlier fitting work is substantially affected by axis selection.
Earlier pairing results remain valid for their coordinates, not model-wide.

At rank64, all-response PCA gains only.027 over natural-write PCA in pooled local
coverage, failing the locked >=.10 improvement prediction. Even response bases
lose coverage on unseen odd sources. At rank128, every tested fixed basis still
omits most response energy. This supports broad fixed-output geometry; it does
not prove a rank lower bound for nonlinear or context-dependent programs.
T12 selection bias gains direct support. T09/T10 context-dependent operators and
role-based representations become higher-information alternatives than further
four-axis tuning. The covariance/response bases are question-dependent summaries,
not identified fundamental features.

## H234 — Local response coverage does not establish nonlinear behavioral coverage

**Finite mediation design.** On32 separate fresh rows at two positions each,
heads1 and3 were removed at the destination. Every approximation keeps the same
changed pre-MLP residual and value bus. Gold uses the native MLP response; the
no-response control keeps the original MLP write; projected conditions add back
only the selected component of the exact MLP change. The full nonlinear suffix
recomputes. Fidelity is relative to gold-minus-no-response, not total head removal
or ordinary MLP deletion. These are sufficiency/communication tests.

| Basis | H1 same-site local | H1 logits | H1 CE | H3 same-site local | H3 logits | H3 CE |
|---|---:|---:|---:|---:|---:|---:|
| Four studied axes | .048 | -.015 | .070 | .115 | .078 | -.060 |
| Natural64 | .221 | .151 | .331 | .251 | .264 | .016 |
| All-response64 | .240 | .112 | .269 | .270 | .249 | -.153 |
| Weight64 | .245 | .130 | .472 | .298 | .257 | -.036 |
| Natural128 | .324 | .298 | .504 | .335 | .261 | .354 |
| All-response128 | .344 | .304 | .372 | .360 | .354 | .250 |

Response64 captures slightly more local energy on these exact sites but does not
show a supported downstream improvement over natural64. Paired row-bootstrap95%
intervals for its logit-fidelity advantage are [-.105,.035] for H1 and
[-.054,.095] for H3; CE intervals are [-.422,.124] and [-.762,.312]. These wide
intervals do not establish that response64 is worse either. CE point estimates
are noisy; do not promote negative point estimates to a general ranking.
The studied four axes preserve little of this MLP-mediated effect even when their
response coefficients are supplied exactly. This is a substantive coverage limit.

MLP-response logit energy divided by total head-removal logit energy is .270 for
H1 and .156 for H3; corresponding CE-energy ratios are .191/.032. These are scale
diagnostics, not additive causal shares: direct and MLP paths can cancel.

**Controls.** Full-response reinsertion versus an independently constructed native
changed-MLP state has max logit error2.24e-5; orthogonality error4.77e-7 and head
replay3.91e-7. A toy end-to-end test independently verifies that zero/full response
conditions hold upstream head removal fixed. Fifteen targeted tests pass.
The computation was rerun on the same locked panels to add the independent replay
and same-site local coverage; this is an audit rerun, not a new replication.
Population test19560–19687; finite test19688–19719. All runs completed.

**Next discriminator.** Broad pooled output space can mean either broad response
at each context or a compact response space that rotates with context. At fresh
rows19720–19751, four positions per row, test all nine source heads and small/full
removal scales. Construct per-state source-output tangent matrices analytically
from the native bilinear MLP and RMS derivative, using an orthonormal basis of
each head's raw output span. Choose output singular vectors without looking at
the finite response being evaluated. Compare ranks4/16/64/128 with frozen global
response bases, wrong-context charts and wrong-source charts. Evaluate both
oracle projection of finite changes (geometric sufficiency) and projected tangent
predictions (actual coefficient prediction). If only state-matched charts give
compact coverage, continuous state-conditioned operators are a better description
than a few fixed output features. If even those charts remain broad, input/source
capacity is genuinely distributed locally. Small versus full edits separates
first-order geometry from finite quadratic/normalization effects. Count the cost
of constructing these charts and avoid claiming a recovered cheap router or
semantic meaning. Freeze this next config before evaluating responses.

**Records.** BILIN18_L8_RESPONSE_POPULATION.md, PNG/SVG, frozen bases, full test
writes, per-row local/downstream arrays, full JSON and paired contrasts JSON.
