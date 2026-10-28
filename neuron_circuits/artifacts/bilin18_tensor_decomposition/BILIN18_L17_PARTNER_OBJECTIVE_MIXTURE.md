

## H274 — A shared mixture basis approaches both specialists, but absolute coverage remains low

H273 found that actual-response optimization generalized to new documents but harmed matched-angle coverage. H274 fixes natural core32 and partner32 and changes only the training objective: actual-only, angle-only, or equal normalized mixture. Twelve runs total: each objective crosses weighted/random initialization and learning rates.003/.01;400 updates and checkpoints0/100/200/300/400. Single-family updates use256 examples; mixture uses128 from each family, maintaining total examples/update (not identical kernel cost).

Each family squared error is divided by its own TRAINING mean cross-target squared norm: actual114981.34, angle118743.28. The same constants are used for validation. Mixture validation minimizes the equally weighted normalized errors, not the better of the two family scores. First768 discovery documents train, next256 validate. Test is the reused H273128-document panel with new seeded angle draws. This is not another fresh-document replication. Train and test angle draws use independent seeds. All native/compiled coefficients and core directions remain fixed; only the partner projection changes.

Selected weighted/.003 checkpoints: actual300, angle300, mixture100. Step0 baselines remain eligible. The actual-only validation trajectory reproduces H272, checking that the training refactor preserves the control. No test-dependent model selection.

| Training objective | Actual small / full | Angle small / full |
|---|---:|---:|
|weighted|44.42% / 43.26%|28.52% / 28.84%|
|actual|51.91% / 50.79%|26.27% / 26.57%|
|angle|41.81% / 40.81%|31.02% / 31.50%|
|mixture|48.99% / 47.87%|31.23% / 31.62%|

Locked gates: angle_rescue: FAIL, actual_retention: FAIL, near_specialists: PASS, both_half: FAIL.


Only near-specialists passes: mixture minus each respective specialist has paired95% lower bound above−5 points at both scales. Actual-retention fails the stricter−3-point noninferiority criterion: full actual difference−2.92 points, CI[−3.44,−2.36]. The point estimate alone is not sufficient to pass noninferiority. Angle rescue requires>=10 points over actual-only, while the observed gain is about5 points. Both-half fails because mixture coverage is below50% in both full-edit families.

Full-edit mixture versus respective specialist: actual−2.92[−3.44,−2.36] points; angle+0.12[−1.10,1.80]. The angle result does not establish superiority to the angle specialist. These are paired document-bootstrap95% intervals conditional on the seeded perturbations, not uncertainty over every possible perturbation distribution.

| Selected model | Train actual / angle | Validation actual / angle |
|---|---:|---:|
|actual|54.43% / 27.33%|54.45% / 28.42%|
|angle|42.75% / 30.35%|44.09% / 33.95%|
|mixture|50.29% / 30.80%|51.32% / 33.64%|

Full total-residual response fidelity (distinct from cross-term fidelity):

- weighted: actual 35.65%, angle 28.44%.
- actual: actual 39.09%, angle 27.67%.
- angle: actual 34.55%, angle 28.89%.
- mixture: actual 37.54%, angle 29.12%.

**Interpretation.** Objective specialization is real, but the tradeoff is partly avoidable within one rank32 subspace: mixture retains most actual-specialist fidelity and matches the angle specialist. This weakens a strong claim that the two distributions require incompatible representations at this rank. It does not demonstrate a broadly sufficient shared representation, because even the specialist angle basis captures only31.5% of angle cross response. There is no large training–validation gap explaining this low selected angle coverage. Optimization limits, the projection family and fixed core remain alternative explanations; no intrinsic-dimension conclusion follows.

A useful candidate unit here is a pair of interacting subspaces with an explicit bilinear rule, assessed across response distributions. Its utility depends on objective and allowed interactions, and the coordinate axes within either span are not individually identified. This is still not a semantic feature ontology. Local geometric fidelity, economical computation, downstream relevance and semantic interpretation remain separate requirements.

**Next discriminator: compile and test the actual and mixture rules.** Further small selection variants should not substitute for checking what these candidate computations do. Materialize the exact low-rank cross-core polynomial using the frozen core/partner reads and residual-tensor contractions. Verify it against native-minus-compiled cross responses on arbitrary inputs, count all read/coefficient/storage costs, and test actual-only versus mixture corrections in the same conditional downstream-mediation setup. This will establish whether the local energy tradeoff changes next-token behavior. Keep the common native baseline explicit: conditional mediation is not intact model replacement. Use new documents for confirmation after freezing the implementation. If behavior remains weak, that is evidence against treating this response-energy unit as the behaviorally relevant unit, even if its algebra and geometry are reproducible.

**Audits.** Three targeted tests pass: normalized-family loss scale invariance and gradient weights, finite-difference gradient through partner QR/cross computation, and complement/coordinate identities. All12 fixed runs and both test families complete. Curves, selections, normalizers, raw document errors and frozen bases retained. Figures inspected. Full project suite not rerun. Native weights and original compiler unchanged. Broad goal remains active.
