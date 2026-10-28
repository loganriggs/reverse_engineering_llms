## H759 — Diverse tasks reveal representation-dependent tradeoffs and weak-effect limits

Twenty-four frozen representations are evaluated for L5H3/H5 on H732's five task families, two formats and nuisance variants. All scalar/block/shared-group seeds, degree-preserving group nulls, fixed/adaptive write PCA and random adaptive controls are retained. The panel contains77 token-valid pairs and231 native contexts. It is reused research data but was not used to fit these natural-text featurizers.

Edits add F(donor)-F(receiver) at the final position only, in two reciprocal task directions, one nuisance direction and an identity control. This differs from the earlier operation tests' two-position edits and their cross-operation donor pairs; numerical effect magnitudes should not be directly compared across those designs. No downstream query/key clamps are applied here.

Native competence reproduces the earlier panel exactly: agreement, binding and copying pass both formats; capital recall passes format B but fails A's full-vocabulary gate; sentiment fails both formats. Relative errors are reported only when competence passes and full-head margin-change RMS>=0.25logits. Otherwise all absolute errors remain recorded. Three cells meet this combined threshold.

| Eligible cell | Full effect RMS | Scalar error | Block error | Shared-group error | Group-null error | Adaptive PCA error |
|---|---:|---:|---:|---:|---:|---:|
| H3 agreement A, task | 0.293 | 42.3% | 58.6% | 42.7% | 65.3% | 42.9% |
| H5 copying A, task | 0.325 | 71.2% | 103.2% | 61.1% | 88.5% | 85.8% |
| H5 copying A, nuisance | 0.279 | 52.5% | 91.4% | 105.4% | 92.4% | 127.7% |

Errors are mean seed-level relative RMS against the full-head intervention. Fixed write-PCA errors on these rows are59.4%,63.1%,101.7%; random-adaptive means71.2%,93.5%,111.1%. Means over shared examples and rewires are descriptive, not independent statistical replication.

The apparent copying advantage of shared groups is not consistent across dictionary seeds. Scalar copying errors are64.1%,65.2%,84.4%; group errors77.8%,65.2%,40.4%. Thus the mean gain is driven largely by the third dictionary. Its grouping performed poorly on the previous operation panel, so choosing it now would be task-based selection, not proof of a generally superior featurizer. Nuisance errors worsen for all three group models relative to their scalar counterparts.

Nuisance fidelity and semantic invariance are different targets. Group nuisance effects here have smaller RMS(0.107–0.184) than the full-head effect0.279, but their alignment with it varies from-0.306 to0.240. This is not simply a demonstrated useful invariance; one dictionary reverses the aggregate effect direction. Task-specific abstraction could prioritize a different objective, but that would need a separate accuracy/invariance test rather than a faithfulness claim.

Most remaining cells have weak final-position head effects. H3 agreement B has full RMS0.230 with scalar/group absolute errors0.089/0.101logits. H5 copying B has full RMS0.164 with scalar/group absolute errors0.163/0.150. Competent capital-recall B has full RMS only0.005(H3)/0.004(H5). Binding task effects are0.057–0.075 at H3 and0.005–0.010 at H5. These values limit what this intervention can establish about representation quality on those tasks; they are not discarded or converted into misleading percentages.

Interpretation: head function and representation quality depend on context and intervention position. H3's clearest final-position effect in this suite is agreement; H5's is copying. Shared groups can help one task and damage preservation of a nearby nuisance effect, while scalar models remain stronger on the prior operation-dependent target. No universal featurizer winner or verified manifold emerges. Broadening the suite was informative precisely because it exposed these tradeoffs and weak-effect cells.

Audits:15,015 forwards =231 native captures +14,784 interventions including identity controls. Native logits replay exactly; identity logit error<=2.10e-5; full donor-head vector error<=1.34e-5; shared-atom union exact; all hashes unchanged and outputs finite. No original reserved-test evaluations; natural document pool26,112 unchanged. Native failures remain diagnostic.

Next: audit the failed-task answer coverage and establish a prospective, competence-valid copying/nuisance panel before interpreting the seed-specific shared-group advantage. Keep all seeds frozen. Separate final-query and source-position effects where possible, and retain total-effect faithfulness versus task-specific abstraction as distinct objectives. Do not choose groups or hierarchy based on the best observed task cell.
