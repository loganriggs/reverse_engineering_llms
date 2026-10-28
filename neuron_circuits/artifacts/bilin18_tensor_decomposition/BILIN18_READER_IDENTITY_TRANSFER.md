## H539 — Shared readers oppose identity, with routing sign selecting suppression or copying

The preceding goal turn made progress by establishing a fixed-reader account and discovery-selected readers that transferred across the original task panel. Here retain L14 head 4 and L8 head 3 as leading readers, with L14 head 0 and L8 head 0 as fixed same-layer controls. No head is selected on the new examples.

Use the original eight discovery pairs per task to fit majority signs for two quantities separately: the head's token-identity transformation and the native routing coefficient. Then evaluate eight new lexical pairs per task in three formats. Binding formats include standard wording, reversed introduction order, and a shifted scene/verb. Copying uses new non-color values, a prefix variation and three repetitions. There are 48 fresh pairs (96 prompts), plus 16 reused discovery pairs (32 prompts). All examples are retained; no tokenization exclusions occurred. Each format passes the native competence gate: >=75% of pairs choose both intended answers and >=50% of endpoints have an intended full-vocabulary top-1 token.

### What is separated

For a head, compute its output projection applied to the difference between two initial token-value vectors. Measure this map's alignment with the unembedding difference for the same two tokens. Negative alignment means the map opposes token identity in this specific contrast. Separately compute the sum of native attention coefficients at the changed source positions, including the learned bus mixing coefficient. Multiplying map and coefficient gives the direction of the head's direct identity contribution. No semantic labels are fitted to hidden axes.

For causal tests, replace only the persistent-bus contribution from the changed source positions to one head's final attention destination, leaving every other contribution native and allowing all later computation to respond. Also replace those source values with either of two third tokens that are not task answers: Kevin/Brian for binding, stone/glass for copying. Compare the third0-minus-third1 output margin under third0 injection versus third1 injection. Positive differences mean copying the injected identity; negative differences mean suppressing it. This distinguishes a general identity effect from merely favoring the original recipient or answer token. These third-token interventions deliberately create hybrid states and are not claims about natural task accuracy.

### Fresh results

| Task | Reader | Negative identity-map alignment | Positive routing coefficient | Mean task-directed causal effect | Mean third-token identity effect | Coarse discovery sign correct |
|---|---|---:|---:|---:|---:|---:|
|entity_binding|L14H4|48/48|48/48|2.8164|-3.0839|48/48|
|entity_binding|L8H3|48/48|48/48|0.3393|-0.2447|48/48|
|entity_binding|L14H0|0/48|0/48|0.1205|-0.0830|48/48|
|entity_binding|L8H0|30/48|3/48|0.0003|-0.0003|4/48|
|sequence_copy|L14H4|48/48|8/48|0.7355|0.8520|40/48|
|sequence_copy|L8H3|48/48|0/48|1.4175|2.5645|48/48|
|sequence_copy|L14H0|0/48|45/48|0.2384|0.5619|45/48|
|sequence_copy|L8H0|18/48|22/48|-0.0018|0.0014|22/48|

Both leading heads oppose identity for all 48 fresh directions in each task. In binding, both use positive routing coefficients at the giver position, yielding identity suppression. L8H3 uses negative coefficients in all copying directions, reversing its identity-opposing map into copying. L14H4 does so in 40/48 copying directions; the other eight have the opposite routing sign and suppress instead. Thus “this head copies during copying tasks” is too coarse. The computation depends on the actual coefficient, not simply a task label.

The coarse discovery sign predicts the third-token effect in 184/192 leading-head comparisons. A secondary, parameter-free analysis using the discovery identity-map sign multiplied by each fresh native coefficient sign predicts 192/192. This finer rule is not treated as the original coarse preregistered test; the next experiment prospectively locks it and adds gate reversals and more third-token identities.

The L14H0 control also implements a coherent identity transformation but with positive identity-map alignment and smaller effects. Its presence argues for multiple identity-reading pathways, not uniqueness of the two leading heads. L8H0 has weak effects and unstable identity alignment; its coarse sign predictions fail frequently. The result supports a structured source/map/gate operation rather than arbitrary head directions, but does not establish a complete or unique circuit.

Directions within a lexical pair and its three formats are correlated. Counts such as 48/48 are observed test outcomes, not 48 independent lexical samples or population probabilities. The new lexical and position variants strengthen transfer evidence; broader tasks and model-training holdout claims are not established.

Audits: 512 source/head geometry records, 1,408 causal interventions and 384 third-token comparisons. Token-value replay error <=1.34e-5; scalar-coefficient/map factorization error <=5.6e-6 relative L2; self margins agree within 5.8e-6; all values finite and source/manifest hashes unchanged. An initial final-serialization failure from a NumPy boolean was corrected, and the completed run was repeated successfully; no scientific condition changed. No new FineWeb documents; tracked total 26,112.

Next: fix the map signs learned from old discovery, predict each third-token effect using the actual native coefficient sign, test another lexical panel with eight different third-token pairs per task, and deliberately reverse or zero the selected source-to-head bus coefficient. Later downstream computation stays active. This tests a conditional operation, rather than a blanket family label.
