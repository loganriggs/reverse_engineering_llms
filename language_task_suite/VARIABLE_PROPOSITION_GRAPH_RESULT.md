# Variable event graphs increase rule opportunities, not language variety

The [variable-graph sampler](document_world_v14_variable_graph.py) changes the *sampled proposition graph* rather than just sentence frames. It draws 2–6 entities, 4–12 state-changing events, source-attributed updates, alias references, state queries, and independent true discourse acts. It controls how often later events revisit an earlier entity. All four arms use the same V10 typed lexical renderer and visible-text answer checker. This isolates a data-law axis: the size and connectivity of the described world.

The [audit](audit_v14_variable_graph.py) sampled two independent 128-document panels for each arm, verified every scored answer from the visible prefix, and rerendered a supported upstream edit per document. All sites and 1,024 edited documents passed. The variable arm averaged 7.77/8.16 events and 5.68/6.18 scored sites per document, versus 5.50/5.34 events and 4.37/4.22 sites in the three-entity fixed control. The observed fraction of extra events revisiting a touched entity was .631/.658 under the .1 revisit control and .934/.941 under the .9 control, so the graph knob did alter the distribution. The controls are not paired identical worlds; each arm has an independent seed stream.

| First 100 words, 128 documents | Fixed 3 entities | Variable, revisit .55 | Variable, revisit .1 | Variable, revisit .9 |
|---|---:|---:|---:|---:|
| Word types, seed 54101 / 64101 | 548 / 519 | 538 / 540 | 536 / 540 | 541 / 538 |
| Top-ten word share | .309 / .309 | .316 / .314 | .318 / .321 | .319 / .317 |
| Cross-document repeated-trigram share | .486 / .482 | .501 / .512 | .517 / .513 | .519 / .511 |

At 41-word source-attribution windows, the repeated-trigram share is .408/.412 in the fixed control and .409/.400 in the variable arm. The small movement does not close the local form gap; the earlier FineWeb window reference is near 1%, though it is not source-task matched. The full-document comparison is likewise negative: varying graph size while reusing the same small predicate and realization inventory slightly *worsens* exact phrase reuse and common-word concentration. The higher site density is a useful controllable task-coverage axis, but this arm is not accepted as the main language DGP and no model was trained on it.

The result narrows the next step. Entity/event count, recurrence, clause packing, and frame variation have all failed as isolated realism fixes. A credible sampled language laboratory likely needs a much broader *typed proposition and discourse operation inventory*, with varied syntactic realizations of each and interactions among rules. Corpus-anchored text with controlled, independently checked rule sites is another route, provided task-bearing windows themselves stay natural and the sampling process integrates the rules rather than appending separate exercises. The next experiment should compare these routes on whole-document and rule-window form plus rule-site coverage; only behaviorally learned rules should enter weight-canonicalization and sparsity benchmarks.

The [aggregate JSON](results/v14_variable_graph_n128.json) contains generated samples and counts, no natural-source passages.
