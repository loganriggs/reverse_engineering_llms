## H740 — Operation-linked gating mostly survives matched format, token and position controls

We evaluated L13H0 on the established 48-pair operation-by-format panel previously used for other heads. This is a new head test on reused prompts, not a globally fresh dataset. Four primary cells cross narrative/list presentation with entity binding/sequence copying using the same eight name pairs. Additional reversed binding lists and temporal copying narratives are retained.

Before intervention outcomes, we verified exact matching across each primary binding/copying comparison: source token, last changed source position, final token and final position. All 32 comparisons pass (two formats x eight name pairs x two receiver variants, corresponding to 64 primary prompt contexts). The intervention is restricted to that last changed source position; copying may have other changed positions that remain native.

All six cells pass the predeclared native-competence gate: both-endpoint contrast accuracy is 100%; full-vocabulary intended-answer top-1 is 87.5% for narrative binding and 100% for the other five cells. No pairs were excluded.

L13H0's coefficient alpha=lambda13*A(last changed source):

| Cell | Coefficient range | Positive coefficients | Mean donor-directed single-source task effect |
|---|---:|---:|---:|
| Narrative binding | +0.342 to +0.766 | 16/16 | +0.697 |
| List binding | +0.190 to +0.484 | 16/16 | +0.461 |
| Reversed-list binding | +0.320 to +0.508 | 16/16 | +0.503 |
| Narrative copying | -0.291 to -0.050 | 0/16 | +0.133 |
| List copying | -0.284 to +0.024 | 1/16 | +0.122 |
| Temporal-narrative copying | -0.120 to +0.049 | 3/16 | +0.025 |

The primary matched binding/copying coefficient signs differ in 31 of 32 comparisons. A rule based only on narrative versus list format, the source/final token identities, or their absolute positions cannot explain those matched sign differences. This is not proof of an abstract, universal binding/copying classifier: other contextual cues differ, one primary copy context has the unexpected sign, and the temporal-copy extension is weaker and has three positive coefficients. Native model competence does not imply that this particular source edge is a decisive contributor on every format.

We then injected the same absent name, object and country pairs from H739 (Kevin/Brian, stone/glass, Canada/Australia). At each source edge we replaced the bus contribution with alpha*g*B(absent), for g=+1,0,-1, leaving all other contributions and downstream computation live. The delta relative to the native model is alpha*(g*B(absent)-B(native)). Thus g=0 removes the original source contribution; the two absent-token conditions must then coincide. It is not an identity/no-edit condition.

Results:

- With the native coefficient, absent identities are suppressed in binding and generally promoted in copying, following the measured coefficient including its exceptions.
- Reversing the source contribution reverses the observed contrast sign in every tested context/token-pair comparison.
- The zero-coefficient condition yields exactly zero absent-token contrast in every case.
- Fixed-map alignment times the coefficient predicts all 576 nonzero-gate signs (96 contexts x three absent pairs x two nonzero gates). These are correlated repetitions using three fixed token pairs, not 576 independent generalization trials.

For example, native/reversed mean name-pair contrasts are -0.473/+0.473 for narrative binding, -0.339/+0.340 for list binding, +0.127/-0.126 for narrative copying and +0.116/-0.116 for list copying. The mechanism's sign is causally controllable at this source contribution. This does not establish that its reversal changes overall task accuracy or that the edge is necessary for the model's full solution.

Audits: native head factorization maximum relative error was 5.01e-7, static bus replay error 2.84e-7, identity margin error at most 5.72e-6, and zero-gate contrast error exactly zero. All hashes remained unchanged and results were finite. There were 2,016 full-model example forwards, no new natural documents and no original reserved-test evaluations.

Interpretation: the evidence now supports a reusable signed read mechanism at L13H0 beyond a simple narrative/list or fixed-position explanation. It is a context-dependent operation over a fixed token/value map, not evidence that the arbitrary coordinates of a high-capacity sparse decoder are fundamental units. The remaining question is where the operation-dependent sign is computed.

Next: cross the destination queries and source keys between the matched binding/copying contexts, separately for both bilinear QK factors. Preserve source values and positions, verify native coefficient reconstruction, and measure whether query-only, key-only or joint exchanges transfer the sign. Surgical single-edge interventions should retain the native/reversed/zero controls. This can localize the contextual decision before testing a compact, reproducible operation subspace. The sentiment diagnostic and manifold-validation objectives remain outstanding.
