## H714 — Coactivation neighbors do not outperform matched substitute neighborhoods

Previous goal turn classification: progress. H713 completed896evaluations and established that a small donorward group effect was uncertain relative to coarse controls. H714 tightens the comparison and separates anchor from neighbors, rather than assigning labels from coactivation.

Reuse every H713pair and all8frozen groups. For each of7neighbors, match a nonmember using training-only logTop32activation frequency, logcoordinate standard deviation and absolute decoder cosine with the anchor. Standardize each matching variable by its across-direction SD. Hungarian assignment minimizes squared distance; repeat three times excluding already chosen substitutes to form three distinct control neighborhoods. Freeze these controls before any new forwards. They are not independent random samples. The matches are approximate: group/control standardized RMS discrepancies range0.112–0.944; mean absolute decoder-cosine discrepancies by group range0.0083–0.0617. No claim of fully matched geometry.

Use the H7135%full-group scaling factor unchanged for separate anchor and neighbor contributions. Their vectors sum to the original full-group edit to1.03e-7relative error. Substitute-neighbor edits are rescaled to the real neighbor edit norm, matched to1.59e-7relative. For combined substitute groups, add the identical anchor edit. Those combined vectors are not full-norm-matched because angles differ. Component cancellation means individual anchor/neighbor edits can exceed the original full-group cap: observed maximum6.064%of residual norm; substitute combined maximum8.250%. Preserve the persistent value bus throughout.

| Intervention | Mean donor-KL gain | Mean receiver next-token CE change | Mean residual edit norm |
|---|---:|---:|---:|
|anchor|0.001529|0.000062|2.78%|
|neighbors|0.001099|-0.001584|4.39%|
|sham_neighbors_0|-0.001044|0.000748|4.39%|
|sham_neighbors_1|0.001485|-0.000078|4.39%|
|sham_neighbors_2|0.002296|-0.000246|4.39%|
|group|0.002429|-0.001647|4.92%|
|sham_group_0|0.000582|0.000757|5.29%|
|sham_group_1|0.002910|-0.000060|5.25%|
|sham_group_2|0.003844|-0.000230|5.39%|

The primary real-neighbor minus mean-substitute-neighbor donor-KL contrast is0.000187nats, with32pair-cluster bootstrap95%interval[-0.002659,0.002865]. Full-group minus anchor-only is0.000900[-0.001042,0.002991]. Full-group minus mean-substitute-group is-0.000017[-0.002937,0.002853], with the combined-norm caveat above. Both directions stay together within each of10000bootstrap resamples. These exploratory intervals do not prove equivalence or rule out effects on different tasks.

Neighbor-only temperature-adjusted native logits gain0.003220toward the donor, versus actual neighbor gain0.001099. Again this is a descriptive confidence control, not an additive causal decomposition. Real neighbor effects do not establish specificity over matched alternatives. ContinuationCE differences remain small and are not task correctness.

An output-interaction check combines the separate anchor and neighbor logits as logits_anchor+logits_neighbors-logits_native. The actual joint edit is close to this additive prediction: meanKL(actual joint||additive prediction)=3.42e-7, versus meanKL(actual joint||native)=6.09e-5. Median centered-logit interaction norm is3.46%of the full response, with IQR1.74–8.34%and maximum41.6%. This is local behavior at small perturbations; differentiable networks are expected to be approximately linear locally, so it neither proves independent units nor rules out larger-scale/context-gated composition. It does not rescue the current groups as nonlinear computational units.

Interpretation: conditional coactivation replicates, but its grouping adds no established donor-transfer advantage beyond matched alternatives on this panel. Imperfect matching, only32pairs and an indirect donor-distribution objective limit inference. More examples alone would estimate this indirect objective more precisely; the next experiment should change what is measured.

### Hypothesis refresh: twenty live explanations and discriminating tests

Ranks below are qualitative priorities, not posterior probabilities. Several explanations can coexist.

| # | Hypothesis | Current assessment and discriminating experiment |
|---|---|---|
|1|Shared context drives groups without shared computation.|Plausible: replicated covariance, no causal grouping advantage. Compare downstream response similarity against matched neighborhoods.|
|2|Decoder geometry explains apparent grouping.|Still plausible; present matches are imperfect. Condition on full Gram geometry and compare effective response geometry.|
|3|Token identity plus broader lexical statistics explains activity.|Token/position conditioning is insufficient for context. Test matched windows and lexical/format controls.|
|4|Groups organize confidence rather than content.|Temperature reproduces donorward score but misses shape. Remove the confidence component from response signatures.|
|5|The anchor carries most useful local computation.|Neighbor addition uncertain. Compare anchor and neighbors on predefined outputs across fresh contexts.|
|6|Distributed neurons share downstream targets despite weak donor transfer.|Plausible alternative metric. Cluster Fisher-weighted logit response signatures, then validate on separate contexts.|
|7|A group matters only when several members are jointly active.|Current selection requires only anchor activity. Stratify by frozen multi-member support before measuring effects.|
|8|The relevant unit is a nonlinear manifold rather than a linear span.|Current neighborhoods are not manifold fits. Test local tangent stability and reconstruction of held-out trajectories.|
|9|The unit is a gated subspace with context-dependent coordinates.|Compare response alignment across contexts with similar inferred gates versus unrelated contexts.|
|10|Shared dictionary atoms are meaningful but these group memberships are wrong.|Synthetic success does not identify real groups. Compare coactivation grouping with downstream-effect grouping.|
|11|Group size8is arbitrary and obscures variable-dimensional units.|Evaluate training-selected group sizes with held-out description length and causal selectivity.|
|12|Top32competition manufactures part of the covariance.|Current shuffle null does not preserve that budget. Use fixed-row/column-margin support controls.|
|13|The retained256Dspan omits important computational directions.|Known variance omission. Compare response coverage with directions outside the retained span.|
|14|Whitening/sparsity captures distributional axes, not causal axes.|Plausible. Compare whitened axes, BSF and response-based grouping at matched resources.|
|15|Sparse coding is inadequate because relevant representations are dense.|Current undercomplete candidate remains worse than BSF reconstruction. Test dense low-rank functional models alongside sparse ones.|
|16|Computational reuse is visible only across held-out combinations.|Untested here. Competence-check diverse compositional tasks before component intervention.|
|17|The persistent value bus supplies essential omitted partners.|Residual-only edits preserve bus. Test controlled residual-by-value interactions on demonstrated tasks.|
|18|Single-site edits miss sequence-distributed computation.|All edits target final position. Compare matched multi-position edits while controlling total intervention norm.|
|19|Donor similarity is the wrong objective for a reusable component.|High priority: broad donor contexts differ in many irrelevant ways. Use predefined output contrasts and selective task effects.|
|20|The current sample is simply underpowered.|Possible; intervals are broad. Expand only after choosing a more discriminating response metric and preregistered contrasts.|

Next priority: test whether true group members have more similar downstream response signatures than matched substitutes, across fresh hash-selected contexts. Use small symmetric finite differences, probability-weighted centered logit effects, and a separate projection removing temperature-like changes. Compare both within-context and across-context consistency; freeze context selection and the contrast before evaluation. This directly probes shared downstream computation without assuming that making the entire output more donor-like is the correct target. If it succeeds, validate selective effects on diverse competence-checked tasks and held-out combinations. If it fails, revisit grouping/featurizer assumptions rather than endlessly scaling this coactivation assay.

Audits:512additional whole-model prefix evaluations; all source/data/model hashes unchanged; finite metrics; component-sum and norm invariants above. Independent CPUfloat64 anchor donor-gain calculation agrees to1.47e-14absolute. Interaction analysis reuses saved logits and requires no forwards. Original reserved test untouched, no new unique research documents; pool26112. Broad research goal remains active.
