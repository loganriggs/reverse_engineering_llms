## H746 — Operation edits outperform identity and wording nuisance subspaces

This retrospective diagnostic tests a stronger control than random directions: equal-rank subspaces derived from actual changes in identity or wording. All query and key subspaces have rank four and use the same receiver-norm-preserving projection. Training uses only the first four old name pairs. Evaluation uses the other four old pairs and all valid new pairs; an explicit name-set intersection audit confirms no overlap. These evaluation outputs were inspected in previous experiments, so this is not prospective confirmation.

Identity controls use endpoint changes within each operation and format under old wording (16 differences). Wording controls use old-to-new wording changes within each operation, format, and endpoint (32 differences). Operation subspaces use binding-to-copy differences: old wording (the frozen H742 baseline), new wording (16 differences), or both wordings pooled (32 differences). A rank-four union takes two old-wording and two new-wording directions and orthonormalizes them. Separate bases are fitted for query and key. Matched rank does not imply equal numbers of training rows; all row counts are recorded. The nuisance contrasts preserve operation but differ in other contextual details; they do not exhaust every possible surface-cue control.

| Evaluation names / wording / format | Frozen old operation | Identity nuisance | Wording nuisance | New-wording operation | Pooled operation | Two-plus-two union |
|---|---:|---:|---:|---:|---:|---:|
| Held old / old / narrative | 4.1% | 89.7% | 98.3% | 23.3% | 8.2% | 14.2% |
| Held old / old / list | 8.6% | 66.1% | 83.3% | 10.5% | 9.7% | 11.2% |
| Held old / new / narrative | 12.1% | 101.1% | 110.3% | 7.0% | 12.6% | 25.1% |
| Held old / new / list | 30.2% | 70.2% | 103.2% | 12.5% | 15.8% | 15.8% |
| New / old / narrative | 12.4% | 93.9% | 98.6% | 21.9% | 18.4% | 30.7% |
| New / old / list | 8.8% | 68.2% | 82.9% | 7.4% | 9.0% | 9.8% |

Values are RMS coefficient error relative to the full factor-two exchange, divided by its RMS change from native. These six cells pass both native task-competence gates. New-name/new-wording cells remain diagnostic: frozen errors16.2% narrative/33.5% list; pooled errors17.4%/10.2%; new-wording operation errors11.7%/9.3%. Their improvement does not establish successful task generalization because native binding fails the competence gates.

Findings: the operation subspace is substantially better than ordinary identity or wording variation at approximating this particular operation exchange. Training on new wording recovers much of its missing structure, especially for lists. However, pooling at fixed rank creates tradeoffs: old narrative error rises, while new list error falls. The simple two-plus-two union is especially weak for narratives. There is no basis here for declaring one universal rank-four operation manifold or for choosing a universally best featurizer. The weaker nuisance controls also do not rule out operation-correlated surface cues.

Scope: this evaluates the signed coefficient of one L13H0 attention edge with the first attention factor held native. No new downstream interventions were run for the nuisance or newly fitted bases. We have not shown their effects on task accuracy, whole-query/key replacement, or naturally occurring unsupervised task identification.

Verification: frozen H742 predictions replay H744 exactly; basis orthogonality error2.00e-15; finite predictions; frozen inputs unchanged; no training/evaluation name overlap. Zero model forwards and reserved-test evaluations; natural document count26,112 unchanged.

Next: freeze original, pooled, new-wording, identity, and wording bases before constructing a third independent wording family. Evaluate coefficient fidelity and actual single-edge downstream interventions without selecting a basis on that panel. Add lexical controls that vary content within an operation, and later seek equivalent operation changes with smaller surface differences. This separates a shared causal component from template-specific directions more strongly than compression alone.
