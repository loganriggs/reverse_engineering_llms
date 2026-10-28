## H861 — A reproducible sentiment atom is a selective partial contributor; block contributions vary across seeds

The previous turn made progress by completing H860's cross-seed stability experiment. H861 now intervenes on components in all five architectures and both seeds. The scalar48 sentiment atom identified by discovery-axis overlap and geometric matching transfers a small but selective sentiment signal in both seeds. Larger effects occur in some eight-dimensional blocks, but geometric matches do not reliably preserve their causal role. This supports partial contributors and possibly distributed computational subspaces, not a complete one-feature-per-concept ontology.

### Experiment

Freeze each seed's group with maximal overlap with each H848 discovery-supervised task axis. Also retain the seed2 geometric match of each seed1 selection. Add two randomly sampled nonselected groups per fit, matched in dimension. This yields50 labeled candidates, some deliberately duplicated when own and matched selection coincide; duplicated candidates are not independent evidence. Selection is saved before downstream outcomes. Evaluate all78 H849 aligned lexical/format/gap pairs in both directions,156 recipient contexts. Both agreement and sentiment are evaluated for every candidate.

Two interventions separate available geometry from encoder usage:

- Projection transfers the part of the native raw donor–recipient difference lying in the component's decoder subspace. It bypasses the encoder and tests geometric capacity.
- Encoded transfer adds the difference between donor and recipient decoded component vectors, scaled by each native state's raw RMS. The rest of the recipient state remains intact. Shared-group vectors require the group's own gate and use its union-coded member coordinates. Because groups overlap, these edits are not additive attributions of independently removable groups.

No gain tuning or norm calibration is applied. The encoded component map is not H857's globally renormalized full reconstruction. Random groups are dimension matched, not edit-norm matched; edit magnitudes are recorded. The natural-featurizer training was task blind, but component selection here is explicitly supervised.

### Causal contribution

The table reports effect slope: sum(component_effect × full_relay_effect) / sum(full_relay_effect²). One means the same overall scale as the full native relay swap; it does not guarantee per-example fidelity. All formats, gaps and recipient orientations are retained. Full per-example errors, KL, and gap strata are saved.

|Architecture|Task|Seed1 projection|Seed1 encoded|Seed2 projection|Seed2 encoded|
|---|---|---:|---:|---:|---:|
|scalar48|agreement|0.033|0.019|0.159|0.112|
|scalar48|sentiment|0.063|0.032|0.153|0.096|
|scalar64|agreement|0.102|0.096|0.046|0.051|
|scalar64|sentiment|0.056|0.054|0.109|0.063|
|block8|agreement|0.063|0.013|0.404|0.325|
|block8|sentiment|0.169|0.092|0.331|0.167|
|shared_fixed|agreement|0.087|0.000|0.128|0.041|
|shared_fixed|sentiment|0.213|0.166|0.261|0.052|
|shared_learned|agreement|0.028|0.000|0.042|0.000|
|shared_learned|sentiment|0.069|0.056|0.019|0.000|

Each entry uses that seed's own highest-axis-overlap group, not the best downstream effect selected after testing.

### The reproducible sentiment candidate

Scalar48 seed1 atom15 and seed2 atom126 are geometric matches, and atom126 is also seed2's own highest sentiment-axis-overlap atom. Their encoded sentiment slopes are0.0318 and0.0963. Pointwise lexical-pair bootstrap intervals are[0.0190,0.0484] and[0.0566,0.1538]. Projection slopes are0.0632 and0.1530: decoder geometry carries more signal than the learned encoded transfer extracts.

Their encoded agreement slopes are-.00018 and-.00045, close to zero compared with their sentiment effects. Scalar48's four random-control atoms have encoded sentiment slopes between-.0003 and.0029. Thus this candidate is a task-selective partial contributor on this panel. It is not a complete sentiment unit, a proof of manifold structure, or broad semantic specificity beyond these two tasks. Five independent sentiment lexical pairs remain a small evidence base, despite repeated templates and gaps.

### Group identity is not yet a reliable computational identity

Block8 seed2's own agreement-selected block reaches projection/encoded slopes0.404/0.325, while seed1's own block reaches0.063/0.013. More importantly, seed2's geometric match of the seed1 block has projection slope0.0066 and encoded slope0. The matched sentiment block similarly has projection slope0.0069 and encoded slope0. So choosing a geometrically corresponding block does not preserve its task contribution here.

Learned shared-group agreement candidates have exactly zero encoded transfer despite nonzero projection effects. Some groups contain relevant directions but their native gates/coordinates do not transmit the signal on these examples. This is consistent with previous unstable routing and coverage collapse. It does not establish that overlapping groups cannot work with different training.

### Updated hypotheses and discriminating next experiment

Evidence now favors three live explanations: task signals are distributed across several components; some individual partial contributors are reproducible; and gating/coefficient inference loses usable signal present in decoder spans. Finite optimization and seed-dependent allocation remain alternatives. Geometric alignment by itself is insufficient to assign a stable computational identity.

Next freeze a nested ordering of groups by discovery-axis overlap and test unions at increasing total dimension, retaining both seeds. Compare native encoded sums/unions with projection onto the combined decoder span and the existing supervised rank1 axis. Shared atoms must be included once. Include random unions at matched dimension and norm-matched edits where feasible. If small unions recover the effect consistently while individual identities differ, that favors a distributed subspace description. If projection succeeds but native encoding fails, prioritize inference/routing rather than semantic labeling. After sufficient coverage, test necessity by removal and extend across binding/copy/capital at their verified active sites. The user's diverse-task requirement remains outstanding beyond agreement/sentiment at this relay site.

### Verification and limitations

15,912 whole-model forwards. Identity replacement reproduced logits exactly; full-donor replay margin error1.44e-5; group orthogonality error4.00e-15. All source hashes unchanged, all outputs finite, no reserved-test evaluations. Natural-document pool remains26,112. Bootstrap uses4096 resamples of lexical-pair clusters, preserving formats, gaps and both orientations; only8 agreement and5 sentiment clusters. Intervals are uncorrected and exploratory. H849 is reused research data, so this is a mechanistic diagnostic rather than fresh prospective confirmation. Broad goal remains active.
