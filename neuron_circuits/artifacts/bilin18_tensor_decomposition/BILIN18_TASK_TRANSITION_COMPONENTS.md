## H720 — Attention carries the main new transfer effect at several selected transitions

Previous goal turn classification: progress. H719 supplied robust whole-state transfer controls at middle/late sites. H720 separates actual block contributions before deciding which activations should be featurized.

The inspected checkpoint block computes b=lambda0*x_in+lambda1*x0; a=Attention(RMSNorm(b),value_bus); m=MLP(RMSNorm(b+a)); output=b+a+m. The mixed input is not simply the previous residual. At all four selected blocks lambda1is8; lambda0is0.90234at9,1.23438at11,0.95703at13and1.05469at14. Capture the three actual terms and verify their sum against the native block output.

Use the same five competence-passing task families, all discovery/transfer pairs and nuisance variants, at H719's discovery-selected sites. For fixed-component interventions add a donor-minus-receiver difference in b,a,m,a+m,or allthree at the block output, keeping other native writes fixed. Separately replace attention's submodule output with the donor write and let the receiver's MLP recompute on that hybrid state ('live attention'). Other positions and the persistent value bus remain unchanged. All edits use natural differences without amplification or dose matching.

| Task and block | Mixed input recovery | Fixed attention recovery | Fixed MLP recovery | Live attention recovery | Whole-state recovery |
|---|---:|---:|---:|---:|---:|
|agreement, L11|30.7%|54.8%|6.7%|60.9%|94.0%|
|capital_recall, L13|14.9%|47.9%|9.5%|54.0%|74.6%|
|entity_binding, L14|30.6%|64.7%|-0.2%|65.1%|95.1%|
|sentiment, L9|31.5%|9.9%|12.3%|15.6%|54.7%|
|sequence_copy, L9|55.3%|2.4%|4.5%|2.8%|63.5%|

Recovery is fraction of the native donor-minus-receiver answer-margin difference on transfer wording, not accuracy or an additive attribution. Interventions differ in norm and downstream processing. Additive state decomposition does not imply additive behavioral effects.

Attention is the largest new-write effect at the agreement, capital and binding sites. Fixed attention specificity (task gain minus absolute nuisance change) is4.436foragreement,3.056forcapitals and4.975forbinding; pair-bootstrap95%intervals are[3.911,4.909],[1.494,4.433]and[4.825,5.122]. These are exploratory8pair transfer intervals, not multiplicity-adjusted mechanism guarantees. Binding attention alone produces the donor's full-vocabulary answer in15/16transfer swaps, versus16/16for whole-state replacement. Its MLP-only margin recovery is approximately zero, with a specificity interval spanning zero.

Letting the MLP respond to donor attention raises agreement recovery from54.8%to60.9%and capitals47.9%to54.0%; binding changes little,64.7%to65.1%. Sentiment's attention effect rises from9.9%to15.6%. These comparisons indicate that the intervention interface matters. They are not formal mediation estimates or proof that the MLP implements a named operation.

Copying atblock9is predominantly carried in the mixed incoming state:55.3%recovery versus only2.4%from the new attention write and4.5%from the MLP write. The block9whole-state sufficiency threshold therefore should not be read as 'copying is computed in block9.' Its earlier rise aroundblock8in H719is a better place to inspect the newly written signal. Sentiment is spread across incoming state and both new writes, so there is no single dominant write at this selected block.

The input skip also matters to interpretation: a donor mixed-input difference combines scaled previous residual and scaled original embedding. Task counterfactuals often share their final token whereas nuisance variants may not. Future carry analyses must separate those two sources rather than labeling the mixed term an abstract task representation.

### Implication for the featurizer/manifold investigation

A monolithic residual featurizer mixes transported information, newly read values and locally transformed outputs. The current results give concrete component-level targets. Before training a more elaborate shared-dictionary or DAG model, factor the successful attention writes by their native heads and test whether a small number of heads explains task transfer, with the same discovery/transfer and nuisance controls. Examine agreementL11,capitalsL13,bindingL14and the preceding copy transitionL8. Review earlier head/value-bus results to avoid duplicating already-established mechanisms. This can reveal whether the useful manifold is a value subspace, context-dependent routing coordinates, or their interaction. Then fit task-blind candidate representations to the causally relevant component activations and compare against the architectural factorization as a baseline. Keep sentiment's more distributed/MLP contribution as a contrasting case.

This still does not verify a superior featurizer or a fundamental unit. It replaces an unsuitable early final-residual target with demonstrated causal writes and explicitly distinguishes carrier state from new updates.

Validation: native b+a+m reconstruction error exactly0in the executed float32arithmetic. All-component swaps reproduce H719full-state answer effects within9.06e-6; native logits replay within2.20e-5. All source/model/data hashes unchanged and metrics finite.1680completed-run whole-model evaluations:240native and1440interventions. An initial attempt aborted with std::bad_alloc before results; its process was confirmed exited, failure log retained, and a synchronous-CUDA retry completed. No failed-run values enter the analysis. Original corpus reserved test untouched, no new unique research documents, pool26112. Broad goal active.
