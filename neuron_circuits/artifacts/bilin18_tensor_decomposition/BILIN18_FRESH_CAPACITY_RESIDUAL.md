## H734 — Agreement is compact in fixed directions; recall-head cross-task information is much more dispersed

We tested a frozen write-PCA capacity curve at ranks 1, 2, 4, 8, 16, 32, 64, 96 and 128, using the natural-training PCA from H723. No directions were fitted to task labels. We also reran scalar Top-8 for all three seeds and intervened on its omitted component, R(h)=h-F(h). The retained and omitted donor-minus-receiver edits sum to the exact head edit, but their downstream effects need not add.

All H732 task/format strata were retained; the table below includes only competent strata with non-negligible exact-head effects. Entries are mean donor-directed margin effect divided by the exact-head effect, not accuracy or per-example fidelity.

| Head and task / format | PCA rank 8 | Rank 32 | Rank 64 | Rank 96 | Rank 128 |
|---|---:|---:|---:|---:|---:|
| L11H3 agreement A | 96.3% | 99.0% | 99.5% | 100.0% | 100% |
| L11H3 agreement B | 94.7% | 99.1% | 99.8% | 99.9% | 100% |
| L13H0 capital recall B | 17.9% | 58.6% | 80.0% | 95.0% | 100% |
| L13H0 entity binding A | 6.4% | 38.0% | 67.2% | 88.0% | 100% |
| L13H0 entity binding B | 6.5% | 36.9% | 66.2% | 85.7% | 100% |
| L13H0 sequence copying A | 9.6% | 48.7% | 76.8% | 92.0% | 100% |
| L13H0 sequence copying B | 12.3% | 49.2% | 73.1% | 91.1% | 100% |

Agreement is concentrated in a small fixed span: even four write-PCA directions retain 90.8% / 89.0%. Recall-head effects accumulate across a much wider part of this particular variance-ordered basis. This is not a lower bound on intrinsic computational dimension: supervised, nonlinear or task-conditioned directions might be substantially more compact. In particular, adaptive scalar Top-8 retained more recall-head effect than fixed rank-eight PCA (H733), so the choice and adaptivity of directions clearly matter.

Omitted-component intervention results (means across the three scalar dictionaries):

| Task / format | Scalar-retained fraction from H733 | Omitted component alone | Relative RMS non-additivity |
|---|---:|---:|---:|
| Agreement A | 88.5% | 6.7% | 7.82% |
| Agreement B | 94.3% | 4.4% | 5.31% |
| Capital recall B | 65.1% | 33.4% | 2.75% |
| Entity binding A | 37.5% | 61.6% | 1.48% |
| Entity binding B | 35.1% | 63.5% | 2.23% |
| Sequence copying A | 41.0% | 59.4% | 1.00% |
| Sequence copying B | 43.6% | 56.7% | 1.40% |

Non-additivity is the RMS difference between the exact-head margin effect and the sum of retained-only and omitted-only effects, divided by the RMS exact-head effect, calculated per dictionary seed and then averaged. It is not the difference of the table's averaged fractions. For the recall head, most missing effect is independently present in the omitted component, rather than emerging only from large interactions between retained and omitted components. This is local to these intervention magnitudes and downstream readouts; it does not imply that the network is globally linear.

Native-head PCA reconstruction KL and intended-answer-set NLL changes are also archived at every rank. These objectives remain separate from task-margin preservation. We do not assess the omitted component as a standalone native reconstruction: it is a diagnostic intervention, not a proposed complete featurizer.

Audits: 13,398 full-model example forwards completed. Input/source hashes remained unchanged and all task/KL values were finite. Scalar task effects exactly reproduced H733; rank-128 PCA task effects matched exact head swaps within 4.77e-6. Identity native KL was zero, and rank-128 native KL passed the 1e-7 bound. Recombining retained and omitted captured vectors recovered native head coordinates with maximum relative error 1.15e-7. No new natural documents or original reserved-test evaluations were used.

Interpretation and next experiment: the eight-coordinate sparse budget is demonstrably incomplete at L13H0, and restricting supports into four-dimensional groups is not solving that problem. Test properly trained scalar capacity levels (16, 32 and 64 active coordinates) using the same dictionary width, training split, seeds and parameter count, followed by the frozen cross-task assay. Do not merely change Top-K at inference in the eight-coordinate-trained encoder. This measures whether a more faithful adaptive baseline is available before decomposing its effects into reusable units. In parallel conceptual terms, retain the competing hypothesis that this head carries a distributed token/value code while context supplies a reusable retrieval operation; the present results do not decide it. Selective feature interventions and task-conditioned compactness tests remain necessary. Sentiment's format/content diagnostic is still outstanding.
