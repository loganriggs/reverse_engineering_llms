# Variable-clause document pilot: form, learned rules, and sparse weights

This pilot tests the intended order: sample one joint document distribution, check its visible rules and natural-text form, train ordinary next-token bilinear models, gate each weight claim on held-out behavior, then compare a sparse-weight intervention. Capability is a rule-specific eligibility check, not the main research objective.

## What is sampled

Each document draws a genre, topic, three objects with owners/colors/places, aliases, an event ledger with named report sources, and interleaved questions or recap statements. Every introduction, event, and recap gets an independently sampled clause form. Optional headers, incidental sentences, and paragraph breaks vary the surface. The complete document, including all recaps, is the next-token training stream; site labels are used only for evaluation. The renderer uses the [V4 event ledger](document_world_v4_interleaved.py) and a visible-text parser to recover answers without consulting hidden sampled facts. An exact marginal distribution for *every* next token is not claimed.

The new `source_delay_rate` knob moves some source questions past intervening reports. A question such as “Who reported the latest place change for the badge?” can then refer to Theo even after a later color report from Nina. This is distinct from immediately repeating the name in the preceding sentence. The delayed-plan auditor checks visible-prefix answers, supported upstream edits, and surface-only invariances on two 300-document panels ([aggregate audit](results/document_world_v5_clause_delayed_audit_n300_seed33041.json)); 243/307 source questions in an independent 500-document diagnostic were delayed.

## Form is better, but still far from natural text

Two disjoint FineWeb `sample-10BT` offsets each yield 35 jointly eligible first-100-word windows. Only hashes and aggregate statistics of the natural documents are saved.

| Window | V4 types / top-ten share / first-40 repeat | V5 types / top-ten share / repeat | FineWeb types / top-ten share / repeat |
|---|---:|---:|---:|
| Offset 28000 | 82 / .389 / .464 | 195 / .349 / .277 | 1478 / .229 / .182 |
| Offset 29000 | 82 / .373 / .457 | 184 / .349 / .283 | 1493 / .194 / .192 |

V5 adds question, quotation, numeral, and newline variation, but the lexicon remains tiny and the prose visibly repeats frames. In one sample, a report about a badge's location is followed by a report about its color, and a later question asks for the location source. This is a useful controlled contrast, but it is not a realistic-language replacement. See the [form panels](results/document_world_v5_vs_v4_fineweb_skip28000_n64.json) and [replication](results/document_world_v5_vs_v4_fineweb_skip29000_n64.json).

## What the models actually learned

The architecture is the older `tensor_language` `DeepModel`: bilinear attention and bilinear MLP in each block, 64 residual dimensions and four heads. A depth-2 model trained for 3000 AdamW steps on 8000 V5 documents without source delays; 1000 documents were held out with zero exact document overlap. On the held-out set its source-answer full-vocabulary accuracy was .938 and token NLL .836. Across two fresh 1000-document causal panels, changing the relevant immediately preceding report raised the newly correct source's predicted probability by .902 and .925 (128 paired cases each); changing an unrelated report had mean responses around -.012. Surface-entropy shift retained a .898 response. This establishes an immediate-source retrieval behavior on this DGP, not object/attribute binding across intervening reports. Alias, color, and most world-state sites fail comparable causal gates and are ineligible for weight-level recovery claims ([training](results/document_world_v5_clause_depth2_4_3000.json), [causal panel](results/document_world_v5_clause_depth2_causal_n1000.json)).

The shortcut became clear only after a new DGP setting. With `source_delay_rate=1`, a matched depth-2, 3000-step model has held-out source-answer accuracy .336 and source-answer NLL 1.769. On 128 fresh delayed cases, changing the correct earlier report gives a mean target-probability response of .155; changing an unrelated report gives .004. Immediate cases from that same run also have weak response (.149). Owner/place edits have modest responses (.212/.136), while alias and color remain near zero. Thus this training run does **not** behaviorally qualify as a recovered delayed source-binding mechanism ([training](results/document_world_v5_clause_delayed_depth2_3000.json), [causal panel](results/document_world_v5_clause_delayed_depth2_causal_n1000.json)). The source question frequency, context length, and model architecture could all matter; this is one model seed, not an impossibility result.

## Weight sparsity on the eligible immediate-source behavior

Starting from the dense depth-2 model, the [sparsifier](sparsify_document_world_v5_clause.py) applied 80 L1 steps, global 40% attention-weight pruning, another 80 L1 steps, global 60% attention-weight pruning, then 200 masked updates with **no L1**. It pruned only `q1/k1/q2/k2/v/o` weights in two attention layers; embeddings, unembedding, MLPs, and norms stayed dense. The 60% attention mask leaves 19,661 of 49,152 attention weights and 156,749 of 186,240 total model parameters live.

| Stage | Held-out source accuracy | Source NLL | Full-token NLL | Fresh source-edit response |
|---|---:|---:|---:|---:|
| Dense | .938 | .199 | .836 | .902 |
| 40% pruned | .878 | .376 | .855 | — |
| 60% pruned, before recovery | .586 | 1.372 | 1.094 | .512 |
| 60% pruned, no-L1 recovery | .901 | .319 | .834 | .769 / .795 on two panels |

The final model retains much of the immediate-source behavior; the causal response is lower than the dense model. Head-output ablations identify layer 2 heads 1 and 2 as important in *both* dense and sparse models: source-response drops are .897/.845 dense and .764/.725 sparse. Sparsity has therefore not yet demonstrated improved head modularity or a readable QK/OV token circuit. The [sparse results](results/document_world_v5_clause_sparse_40_60/results.json), [final causal panel](results/document_world_v5_clause_sparse_40_60/final_no_l1_causal_n1000.json), and [head ablations](results/document_world_v5_clause_sparse_40_60/head_ablation_n1000.json) support only functional compression and head-level necessity. They do not establish circuit recovery.

## Decision and next discriminator

Do not interpret the immediate-source model as having learned delayed attribution. For the next weight-method comparison, either use immediate-source retrieval explicitly as the target or first obtain a model that passes delayed and cross-form behavioral gates. Compare direct QK/OV token-path inspection, the original sparse weights, and weight-only versus data/task-weighted bases at matched held-out fidelity and causal specificity. Keep whole-task/combination holdouts and separate label-free recovery from task-conditioned fitting. Stop a source-specific detour if another fixed-budget delayed model still fails; broaden the DGP task distribution and return to the joint method comparison. The main generator work remains improving lexical breadth, discourse transitions, and interactions among rules rather than maximizing any one capability.

A first [local bond-basis comparison](DOCUMENT_WORLD_V5_BOND_BASIS_COMPARISON.md) is now complete on the learned immediate rule. It includes a whole-genre basis-fit holdout, two independent fit/test samples, dense and sparse checkpoints, weight-only and unlabeled/task-conditioned bases, and separate next-token and causal measurements. It does not establish a canonical full attention circuit.
