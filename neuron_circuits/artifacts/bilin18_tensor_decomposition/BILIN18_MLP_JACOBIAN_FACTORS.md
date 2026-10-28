## H881 — Bilinear partner activations drive most of the transferable local transformation

H880 showed that sharing local MLP Jacobians makes matched copying and capital-recall responses more alike. H881 separates the Jacobian into inverse RMS scale, normalization's radial correction, and bilinear partner activations. Sharing partner activations alone reduces competent-format C mismatch from 27–36% to approximately 9–14%. Sharing both normalization factors without partners leaves 27–42%. Normalization interacts with the result, but does not explain away the context-dependent bilinear computation.

### Factorial intervention

For raw pre-MLP input x, r=sqrt(mean(x²)+epsilon) and u=x/r. A perturbation v passes through the normalization derivative as (v−u*mean(u*v))/r. The bilinear derivative then maps this vector through the sum of Left-direction changes multiplied by Right(u), and Right-direction changes multiplied by Left(u), followed by Down.

Independently take three factors from recipient or matched donor: inverse RMS r (bit 1), radial-correction direction u (bit 2), and bilinear partner activities Left(u)/Right(u) (bit 4). Evaluate all eight masks, using copying as donor for both contexts and separately capital recall as donor for both. Replace L8–17 final-token MLP responses; preserve each recipient's native output offset and subtract its own native input. Attention remains live. Original-amplitude common L8H3 writes, both signs and identity, on all 192 H873 endpoints. No factor combination was selected after seeing results.

Mixed-factor transformations need not be realizable as the Jacobian at a single activation. They are counterfactual factor interventions, not new naturally occurring features. The epsilon-aware radial correction is not treated as an exactly orthogonal projector.

### Competent format C, shared final scale and uncapped readout

|Shared factors|Country output, copy donor|Country output, capital donor|Capital output, copy donor|Capital output, capital donor|
|---|---:|---:|---:|---:|
|None|0.358|0.358|0.272|0.272|
|Scale|0.336|0.365|0.258|0.268|
|Radial correction|0.311|0.403|0.286|0.277|
|Scale + radial|0.293|0.416|0.271|0.274|
|Partners|0.137|0.086|0.137|0.104|
|Scale + partners|0.121|0.078|0.124|0.110|
|Radial + partners|0.067|0.085|0.076|0.093|
|All three|0.058|0.081|0.066|0.086|

Partner-only sharing reduces absolute mismatch as well: country RMSE 0.651→0.341/0.156 and capital 0.453→0.249/0.173 (copy/capital donors). Response norm ratios relative to native linearization remain approximately 0.86–1.37 in these conditions. Thus improvement is not merely response collapse or an increased relative-error denominator. All-factor sharing gives 5.8–8.6% mismatch, reproducing H880.

Normalization factors interact with partner sharing and are not uniformly helpful. For example, sharing the radial correction alongside copy-context partners lowers country mismatch 13.7%→6.7%; sharing normalization factors alone barely helps or worsens mismatch. Do not interpret differences between masks as unique independent percentages of causal mediation.

The pattern also has a native-readout counterpart: partner-only sharing gives country mismatch 18.3–20.2% versus 28.0% native-linearized, and capital mismatch 9.1–16.7% versus 36.2%. These values must stay separate from the controlled-readout results. Familiar A/B and failed-capital D are separately retained in the analysis; D remains a failed semantic-competence control.

### Research interpretation

The derivative of a bilinear MLP necessarily depends on partner activations algebraically. The new empirical evidence is that interchanging those partner factors across these matched tasks transfers most of the measured context-dependent response, whereas interchanging normalization factors alone does not. This does not identify which individual partners are meaningful, whether they form stable groups, or whether the effective transformation has low dimension. Full-vector partner interventions across ten layers are far richer than a verified computational unit.

### Hypothesis refresh: twenty possibilities and current priorities

1. **Leading:** context changes bilinear partner activities that select different local transformations; directly supported here.
2. **Leading:** this transformation difference is distributed across MLP layers; supported by H877–878.
3. **Supported contributor:** final normalization rescales otherwise reusable responses; H875–876 establish a contribution.
4. **Supported interaction:** MLP radial normalization interacts with partner activities; supported by factorial contrasts.
5. **Weak as sole explanation:** MLP inverse RMS alone produces most mismatch; contradicted on this panel.
6. **Weak as sole explanation:** radial correction alone produces most mismatch; contradicted here.
7. **Secondary:** finite nonlinear MLP effects contribute to individual interventions; H879 finds nonzero one-sided errors.
8. **Still open:** attention differences generate the residual mismatch after sharing MLP transformations.
9. **Still open:** a small set of partner coordinates explains most transformation variation.
10. **Still open:** many coordinates encode a low-rank transformation despite diffuse coordinate support.
11. **Still open:** a shared dictionary with overlapping groups captures those transformations more stably than activation reconstruction.
12. **Still open:** useful groups are conditional on task context rather than globally stable manifolds.
13. **Still open:** reusable computation is better represented by input/output direction pairs than activation features.
14. **Still open:** shared transformations require a DAG because downstream partners depend on earlier computations.
15. **Alternative:** the effect primarily reflects lexical identity rather than a task computation.
16. **Alternative:** prompt syntax or position drives partner differences attributed to task context.
17. **Alternative:** the country/capital relationship has an unusually reusable structure that fails on other domains.
18. **Alternative:** selected output contrasts hide substantial changes elsewhere in the vocabulary.
19. **Alternative:** pair averaging hides heterogeneous or opposite mechanisms across examples.
20. **Alternative:** oracle donor matching makes transfer look simpler than any task-blind featurizer can achieve.

The first four explain the current local evidence best. Hypotheses 9–14 concern candidate computational units and are not yet established. Hypotheses 15–20 are the most important generalization threats before declaring a better ontology or fitting another flexible model to this panel.

### Next information-gaining step

Broaden the competence screen using predeclared task families and matched source-token contrasts, including non-capital factual associations and syntactic or lexical transformations where feasible. Retain failed tasks explicitly and separate tokenization eligibility from correctness. Use frozen full-vocabulary and binary competence gates before interpreting intervention transfer. Then replay the minimal discriminating conditions (recipient-native, normalization-only, partner-only, full-factor sharing) on competent held-out combinations, examining individual responses and vocabulary-wide effects as well as selected margins. This is preferable to adding more increasingly specific variants on the same country/capital panel. Fresh behavioral evidence should precede claims that a conditional featurizer discovers useful manifolds.

### Verification and scope

Completed 9,408 full-model forwards. All identities and all-native/full-donor final-state endpoints reproduce their references exactly. Factorized Jacobian-vector products agree with the previously audited formula within relative 7.49e-16; all ten layers retain independent autodiff and finite-difference checks. Selected readout error is at most 9.38e-06. H880 endpoint response error replays within 0; 3,072 donor assignments independently checked. Source hashes unchanged; all outputs finite. The inherited audit field live_finalstate_replay_max_abs is an unused zero placeholder in this factorial runner, not a new live-model replay claim; endpoint checks are the applicable replay evidence. Pair-bootstrap intervals saved (4,096 samples, uncorrected, 12 reused lexical pairs). No reserved test evaluated, natural-document pool unchanged at 26,112. Broad research goal active.
