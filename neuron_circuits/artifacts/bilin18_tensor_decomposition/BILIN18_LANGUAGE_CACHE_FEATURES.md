## H1020 — A stable cached head-slot pathway transfers better than small vocabulary-trained direction sets

Previous turn was progress: H1019identified cached first-layer values as the dominant missing language port. This turn tests featurizers at that port, with the source-country position isolated and residual/x0recipient-native. Cached values have9headslots of128coordinates; these slots are mixed into corresponding head slots across suffix layers11–17. A slot is not a single layer-specific attention head.

Discovery gradients:8historical examples from4country pairs, no evaluation gradients. Compare gradient4/8, source-delta PCA4, top4/8/32left singular directions of the native first-layer value projection matrix, seeded random4/8, and individual native128-coordinate head slots. Learned bases frozen before evaluation. Architecturally grouped128dimensions are not parameter-budget-matched to learned4/8dimensions; head results are a separate pathway census.

| Projection | Familiar/templateBerror | New languages,Aerror | New languages,Berror |
|---|---:|---:|---:|
| Gradient4 |10.2%|91.1%|91.6%|
| Gradient8 |8.7%|89.2%|89.7%|
| Source-delta PCA4 |~0%|97.0%|97.2%|
| WeightSVD4 |98.0%|99.2%|99.3%|
| WeightSVD8 |97.5%|96.8%|96.7%|
| WeightSVD32 |88.2%|84.1%|83.8%|
| Random4/8 |~99.6%|~99.3–99.6%|~99.3–99.6%|

Target is the native cache-only finite margin change, not full donor behavior. That port already recovered73–85%of the natural effect in H1019. Source-PCA4perfect transfer across templates is NOT a discovered abstract operation: the fourdiscovery pair-difference vectors have exact numerical rank4, and the same country cache delta is exactly identical in both templates. At layer0the cached value vector derives from token-local normalized input before attention, so this identity is expected. Discovery reconstruction is8.12e-16relative; new languages have only1.5%mean energy in this span. Gradient8retains~21%discovery edit energy but only2.4%new-language energy. These compact spaces are vocabulary-limited.

Architectural census: cached headslot0alone has aligned recovery79.8%discovery,80.1%familiar/templateB,66.0%newA,67.2%newBrelative to the full cache-only effect. Headslot4grows from4.6%discovery to15.4–16.8%on new languages; slots3/5/6carry smaller positive effects. Summing all9single-slot measured effects predicts full cache swap to4.9–9.1%relative error, so interactions remain. Aligned shares are signed effect projections, not disjoint probabilities or exact causal fractions. No best-slot selection or new combined subset was evaluated after observing tests.

Interpretation: the pathway/architectural grouping is more stable across vocabulary than the tested small direction sets. This matches the earlier repo lesson that reusable transport can coexist with context/content-dependent payload directions. It does not prove a128-dimensional cache slot is a meaningful manifold, or that low-dimensional functional coordinates are impossible with larger discovery data or output-conditioned readouts. Fourtraining country pairs and changing target language logits strongly confound vocabulary coverage with intrinsic rank.

This third task also explains why blindly sharing a residual-only dictionary was inappropriate: the relevant input is a separate cached-value port, and coordinate identity must retain port/headslot information. A group can select a computational interface whose internal payload is richer than a single semantic direction. This is a candidate for the BSF/shared-dictionary comparison, not evidence that learned groups beat scalar selection. Existing H1007/H1008negative group-selection baselines remain applicable.

Audits: cache-only full effect replays H1019within3.81e-6absolute; base1.91e-6; independentTorch metrics exact; native-gradient finite-difference error0.00623%; unchanged non-source cache0; frozen basis hash unchanged; all finite; no evaluation gradients.632suffix sequences,8backwards; cached prior boundary states reused so no newprefixcapture. No reservedBSFtest data or new natural documents; pool26112unchanged. No running process,H961pending,broad goal active.

Next localize which downstream consumers read the stable cached slots, separating cached payload from context-dependent attention routing. A per-layer cached-input transplant must restore the original persistent bus after the one-layer intervention to avoid silently changing all later readers. Use discovery to choose a small transport hypothesis, then freeze and test held-out consumers/contexts; charge all native ports. Preserve this complete-boundary task when testing shared dictionaries rather than optimizing only the easy final-residual channel. Broader output-conditioned training and genuinely fresh task authorities remain required before adopting units.
