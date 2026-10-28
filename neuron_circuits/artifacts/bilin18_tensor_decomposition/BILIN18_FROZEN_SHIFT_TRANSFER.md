## H642 — Frozen context-shift predictions transfer prospectively to new items and directions

H641 was progress: mean normalized-MLP context shifts learned from four items predicted a component intervention on four retrospective holdouts. H642 leaves those learned vectors unchanged and evaluates a new panel and new input probes.

New copying items are river, mountain, window, garden, bottle, pencil, paper, music. New fact subjects/targets are Canada/Ottawa, Ireland/Dublin, Egypt/Cairo, Belgium/Brussels, Peru/Lima, Thailand/Bangkok, Denmark/Copenhagen, Hungary/Budapest. The64 prompts retain H640's four fixed templates per family, including the equal-length instruction-verb contrast and broader format contrast. Items are new relative to the H633/H640 task panels. This tests new identities with known instructions, not unseen task families, template contrasts or model-training novelty.

A planned Kenya/Nairobi pair was replaced by Belgium/Brussels before model inference because Nairobi tokenizes into multiple tokens and the declared scoring protocol uses single-token targets. This is tokenizer-based design adjustment, not selection on model output. All final targets are single tokens. No prompt or item was removed after evaluation.

Three Gaussian probe directions, seeded642, were projected orthogonal to all nine earlier H632 probes and orthogonalized against each other, then scaled to the original vector's norm. Keep the original direction as an anchor. The three fresh probes are independent of the shift fitting and previous response selection. Their maximum overlap with the old unit probes is4.43e-17; mutual orthogonality error is4.45e-16. Direction generation and the prompt panel were fixed before evaluating model responses.

The learned H641 shift file is read unchanged; its SHA256 exactly matches the H641 record. No fitting occurs in H642. Predicted donor MLP contexts use receiver contexts plus frozen shifts. Compare exact donor-context replacement, no shift, and a norm-matched other-family shift. Native attention and MLP normalization are retained. Receiver and donor incoming/readout choices are explicitly crossed. Exact donor MLP contexts are accessed only to evaluate the target intervention, not to construct the prediction.

**The component prediction transfers prospectively.** All256 prompt–probe cases improve over no shift for both donor/donor and receiver/receiver endpoint conditions;192 of those cases use fresh directions, and64 use the original anchor. These repeated observations on a structured panel are not independent trials. Error is norm(predicted response − exact context-replacement response) divided by norm(exact context-replacement response − no-shift response), under receiver-probability-weighted centered logits. No shift has error1 by definition.

| Fresh condition | Predicted effect error, donor input/readout | Other-family error | Predicted effect error, receiver input/readout | Other-family error |
|---|---:|---:|---:|---:|
|copy / matched|0.416|1.670|0.408|1.649|
|copy / broad|0.454|1.172|0.423|1.174|
|fact / matched|0.388|1.263|0.393|1.185|
|fact / broad|0.202|1.021|0.211|1.258|

Donor-conditioned mean errors are0.202–0.454, versus0.185–0.359 on H641's retrospective held-out cells. The predictive advantage persists but is less accurate overall on this fresh panel; the two tests also use different probe sets, so that difference is not solely an item-generalization estimate. Maximum individual errors are0.820 with donor endpoints and0.814 with receiver endpoints. Norm-matched other-family shifts are worse than no shift on average in every cell, showing that amplitude alone does not explain the useful prediction.

The original anchor has mean error0.366 under donor endpoints; fresh0/fresh1/fresh2 have0.369/0.357/0.368. Thus prospective directional generalization is not merely an average dominated by the previously selected vector. The model of context change was fitted without response-direction labels or output-derivative supervision.

**This remains a component model, not a prediction of the full donor computation.** The remaining actual donor–receiver response gap under donor input/readout is:

| Fresh condition | No context shift | Exact context shift | Predicted context shift |
|---|---:|---:|---:|
|copy / matched|0.673|0.394|0.461|
|copy / broad|0.611|0.299|0.407|
|fact / matched|0.646|0.316|0.403|
|fact / broad|0.707|0.314|0.326|

With receiver input and readout, predicted remaining gaps are0.880,0.985,0.948,0.922 for copy-matched, copy-broad, fact-matched, fact-broad. A reasonably accurate prediction of the context-only effect can therefore coexist with weak reconstruction of the complete donor response. Other operator factors and incoming-response differences remain unmatched. No output loss or task accuracy improvement follows from these derivative metrics.

**Native competence again requires care.** All declared targets rank first among the eight candidates, but exact first-token correctness is:

| Family / template | Exact top1 | Candidate accuracy |
|---|---:|---:|
|copy / 0|62.5%|100.0%|
|copy / 1|25.0%|100.0%|
|copy / 2|0.0%|100.0%|
|copy / 3|87.5%|100.0%|
|fact / 0|25.0%|100.0%|
|fact / 1|0.0%|100.0%|
|fact / 2|75.0%|100.0%|
|fact / 3|12.5%|100.0%|

Template indices0/1 are the matched instruction-verb pair and2/3 the broad-format pair. Candidate ranking is not free-generation semantic correctness; capitalization or a preliminary token may explain some immediate-token failures but was not re-scored as success here. Every case is retained.

The research advance is a frozen, item-independent parameterization of a recurring operator change that predicts functional effects for new inputs and new perturbation directions. This is more than donor-operator lookup. It is still conditioned on the receiver's native context, uses the full original network, and only covers four known template-pair transformations across two families. It is not a unique canonical form or an identified indivisible semantic feature. The local derivative regime and retained endpoint conditioning remain material limits.

Next determine whether the transferable effect is concentrated in a few later layers or requires the distributed composition of eight shifts. Test fixed single-layer and leave-one-layer-out ablations of the frozen predictor, comparing each with the full predictor and exact context intervention. Use discovery items for any subset selection and reserve fresh evaluations for verification. This can establish whether a smaller reusable operator description preserves the demonstrated effect rather than treating eight layer-specific vectors as a unit by assumption.

Audits:576 whole-model native/±0.02 forwards;512 cached attention-stage evaluations;1536 batched JVP calls; zero fitting contexts;64 prospective prompts; unchanged frozen H641 shift hash; native residual chain error below0.000338 and weighted logit chain error below0.000530; fresh-probe orthogonality verified; all source/runner hashes unchanged; finite metrics. Supplemental audit independently checks the readout, model hash and probe geometry. No new natural-text documents; FineWeb remains26,112. Broad goal active and incomplete.
