## H751 — Earlier-output effects mostly persist when the selected L13 nodes are clamped

This systematic causal sweep exchanges each L0–12 attention or MLP output between matched binding/copy contexts at both the final-query and source-key positions. Each swap is evaluated with L13H0's selected second query and second key free, and again with their full128-dimensional raw states restored to native. All other paths run live. There are26 output sites and240 contexts, producing6,240 paired interventions; native-competent narratives and diagnostic lists remain separate.

The free-minus-clamped answer-margin change measures dependence on these selected nodes under this intervention. It is not a unique mediated path fraction: downstream interactions remain, earlier outputs have many consumers, and clamping the full nodes goes beyond the rank-four coordinates. Both positions are swapped together. Donor output replacements are not constrained to the operation subspace.

| Site, competent narratives | Free margin-change RMS | Clamped RMS | Free-minus-clamped RMS | Alignment of dependence with total change |
|---|---:|---:|---:|---:|
| L5 attention | 1.775 | 1.675 | 0.126 | 5.7% |
| L4 MLP | 1.127 | 1.020 | 0.125 | 9.7% |
| L6 MLP | 1.486 | 1.400 | 0.119 | 6.0% |
| L5 MLP | 1.209 | 1.125 | 0.114 | 7.1% |
| L3 MLP | 1.049 | 0.985 | 0.111 | 6.5% |
| L11 MLP | 0.637 | 0.569 | 0.088 | 11.1% |
| L11 attention | 0.633 | 0.618 | 0.035 | 2.5% |

RMS values are in intended-answer-margin logits. Dependence alignment is dot(free-minus-clamped,free)/squared_norm(free), aggregated over120 contexts; it is not an accuracy fraction. Across all26 sites, clamped RMS is89.1–99.9% of free RMS, and dependence alignment ranges0.15–11.24%. Thus most measured upstream-swap behavior survives this two-node clamp. This does not mean the head is irrelevant or that all remaining effects follow an independent additive bypass; only the specified query/key states and positions are fixed.

The algebraic and causal rankings differ. L11 attention had the largest direct query-coordinate alignment share in H750, but its clamp-dependent behavioral effect is weaker than several middle-layer outputs. L5 attention causes33.7% alignment with the full donor query-coordinate difference; L6 MLP causes26.6% alignment with the key-coordinate difference. L4/L5/L3 MLP key alignments are21.2%/24.0%/22.1%. These coordinate-movement scores also are not downstream causal fractions.

Interpretation: the verified compact operation component is a reproducible route through computation, not the dominant mediator of these broad upstream output swaps. Earlier output vectors contain effects on other heads, other edges and other features. Distributed construction plus multiple downstream routes remains plausible. We should not select an upstream featurizer site solely by the largest direct attribution or total behavioral effect.

Limitations: full-state clamps do not prove that dependence specifically uses the rank-four component. H747 contexts are reused, and lists remain native-incompetent diagnostics. No claim of unsupervised recovery, manifold identity or general task switching follows from this sweep.

Audits:13,252 model forwards, including240 native captures,12,480 swap/clamp runs,480 no-swap controls and52 self-swap controls. Native logits replay exactly. Maximum baseline/self-swap logit error1.34e-5. Raw selected-node clamps are exact; all outputs finite; input/source hashes unchanged. Zero original reserved-test evaluations; natural document count26,112 unchanged.

Next: compare restoring only the verified compact coordinates against full-node clamps, with equal-rank nuisance controls and explicit normalization audits. This tests whether the identified upstream dependence actually uses the component we intend to recover with unsupervised featurizers. After that, prioritize sites using both component specificity and transfer across tasks, rather than total output effect alone.
