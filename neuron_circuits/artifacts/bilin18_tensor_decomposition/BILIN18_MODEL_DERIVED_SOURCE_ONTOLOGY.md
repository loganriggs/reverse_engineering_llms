# Model-derived source ontologies

Five equal-budget source partitions are compared in the six-layer closed-loop typed-edge program: human token categories, spherical clusters of token embeddings, clusters of Layer-0--5 value signatures, clusters of Layer-0--5 key/key2/value signatures, and random token categories. Every ontology has 21 categories and receives 128 selected types per layer.

Maximum native-branch error: `4.58e-05`.

## Locked hypotheses

- **O1_qkv_is_competitive_with_human: not supported.**
- **O2_qkv_beats_embedding_only: not supported.**
- **O3_qkv_beats_random: not supported.**
- **O4_a_learned_ontology_improves_human: not supported.**
- **O5_a_learned_ontology_is_prediction_faithful: not supported.**
- **native_branch_matches_checkpoint: supported.**

| Ontology | KL recovery | CE delta | Top-1 agreement | Weighted NMI with human |
|---|---:|---:|---:|---:|
| human | 0.9579 | +0.3205 | 0.7505 | 1.000 |
| embedding | 0.8275 | +1.2638 | 0.4668 | 0.364 |
| value | 0.7045 | +2.1537 | 0.3242 | 0.328 |
| qkv | 0.7495 | +1.8415 | 0.3823 | 0.399 |
| random | 0.6162 | +2.7800 | 0.2388 | 0.188 |

## Representative learned clusters

### embedding

- Cluster 0 (3.3%; human plurality `suffix`): ' use', ' need', ' help', ' each', ' used', ' different', ' P', ' M'
- Cluster 1 (1.8%; human plurality `lowercase`): ' even', ' while', ' same', ' why', ' away', ' property', ' minutes', 'G'
- Cluster 2 (2.3%; human plurality `lowercase`): ' real', ' range', ' possible', ' true', ' series', ' complete', ' unique', ' natural'
- Cluster 3 (2.3%; human plurality `suffix`): ' F', ' case', 'F', 'L', ' children', ' car', 'ings', ' games'
- Cluster 4 (1.9%; human plurality `capitalized`): ' �', ' A', ' L', ' U', ' E', ' V', ' O', ' re'
- Cluster 5 (2.9%; human plurality `lowercase`): ' people', ' world', ' D', ' family', 'D', ' money', ' community', ' women'
- Cluster 6 (19.6%; human plurality `function_word`): ' and', ' to', ' of', ' in', ' is', ' for', ' with', ' on'
- Cluster 7 (1.5%; human plurality `capitalized`): ' service', ' W', ' N', "'ll", 'w', ' Dr', ' NHL', 'ens'

### value

- Cluster 0 (5.9%; human plurality `function_word`): ' that', ' as', ' or', ' but', ' which', ' so', ' when', ' what'
- Cluster 1 (3.4%; human plurality `lowercase`): ' I', 'I', ' first', ' C', ' home', ' water', ' book', ' p'
- Cluster 2 (6.7%; human plurality `period`): '.', ').', ' $', 'ed', ' used', ' down', ' available', 'd'
- Cluster 3 (3.2%; human plurality `suffix`): "'s", ' people', ' may', ' years', ' both', ' lot', ' things', ' days'
- Cluster 4 (4.2%; human plurality `lowercase`): ' it', ' time', ' up', ' It', ' there', ' work', ' use', ' business'
- Cluster 5 (2.4%; human plurality `lowercase`): ' them', ' me', ' us', ' look', ' thing', ' feel', ' God', ' others'
- Cluster 6 (2.5%; human plurality `suffix`): 's', ' they', ' need', 'ers', ' They', '%', ' needs', 'ings'
- Cluster 7 (4.5%; human plurality `lowercase`): ' �', ' game', ' take', ' experience', ' number', ' support', ' service', ' form'

### qkv

- Cluster 0 (2.1%; human plurality `numeric`): ' 1', ' 2', '1', '2', ' 3', ' 5', ' 4', '3'
- Cluster 1 (1.9%; human plurality `lowercase`): ' research', ' city', ' history', ' temperature', ' space', ' mind', ' itself', ' development'
- Cluster 2 (14.3%; human plurality `function_word`): ' to', ' of', ' in', ' is', ' for', ' with', ' on', ' are'
- Cluster 3 (8.7%; human plurality `function_word`): ' the', ' a', ' this', ' The', ' an', ' your', 'The', ' their'
- Cluster 4 (2.8%; human plurality `non_ascii`): '�', ' �', ' "', " '", ' de', ' l', '_', ' f'
- Cluster 5 (2.3%; human plurality `lowercase`): ' game', ' money', ' God', ' food', ' night', ' games', ' human', ' music'
- Cluster 6 (10.5%; human plurality `period`): '.', ',', ':', ' (', ')', '�', '?', '/'
- Cluster 7 (4.1%; human plurality `lowercase`): ' way', ' experience', ' data', ' number', ' book', ' form', ' system', ' part'

## Guardrails

The learned categories are static token-identity partitions, not contextual source-state clusters. They use only model weights plus token frequencies from a disjoint ontology-fit corpus. Random projections make clustering tractable and preserve geometry only approximately. Spherical k-means optimizes within-cluster cosine distortion, not causal program fidelity; a negative result would not rule out learned causal categories. Human categories were engineered with linguistic knowledge, whereas learned clusters are unlabeled and may require a separate interpretability step.
