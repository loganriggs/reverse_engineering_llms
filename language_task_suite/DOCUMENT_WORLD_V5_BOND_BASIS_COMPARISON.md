# Local bond bases on the behaviorally learned source rule

This compares local pre-`W_O` coordinate choices in the second attention layer of the V5 variable-clause depth-2 bilinear model. It is a **bond compression experiment**, not a canonical decomposition of the full attention function or recovery of a unique circuit. The eligible rule is immediate-source retrieval. The same checkpoint's response to delayed, intervening-report source edits falls to **0.053** on 128 fresh cases, while its immediate-case response on that OOD panel is **0.812** ([delayed OOD result](results/document_world_v5_clause_depth2_causal_delayed_ood_n1000.json)). Claims about delayed object/attribute binding are excluded.

The [basis evaluator](eval_document_world_v5_clause_bond_bases.py) fits four classes of orthogonal coordinates: right singular vectors of `W_O` (weights only), uncentered PCA of pre-`W_O` activations (unlabeled), a trace-normalized Gram mixture (hybrid), and PCA of paired source-edit activation differences (explicitly task-supervised). Physical axes and one random orthogonal basis are controls. Fits use **story and log documents only**; every evaluation uses new **summary documents**. The model saw all three genres during pretraining, so this is a whole-genre **basis-fit** holdout, not a model-training holdout. The source edit pairs for supervised fitting are disjoint from evaluation pairs. We project the bond at all positions in that layer and measure ordinary next-token NLL, source-answer accuracy, relevant source-edit response, and unrelated-report nuisance response. Every 64-dimensional projection replays the checkpoint to numerical tolerance.

The following values use the first 200 held-out summary documents and 128 paired source edits per type. An independent second fit/test seed gives the ranges in parentheses.

| Model / basis / rank | Token NLL | Source answer accuracy | Source-edit response | Nuisance response |
|---|---:|---:|---:|---:|
| Dense native, 64 | .830 (.825) | — | .911 (.929) | — |
| Dense weight-only, 16 | .965 (.964) | .736 (.740) | .725 (.733) | -.020 (-.021) |
| Dense unlabeled PCA, 16 | 1.104 (1.101) | .112 (.102) | .051 (.058) | .003 (-.001) |
| Dense hybrid α=1, 16 | .959 (.954) | .552 (.488) | .282 (.330) | .002 (.005) |
| Dense weight-only, 32 | .841 (.837) | .928 (.961) | .884 (.906) | -.014 (-.023) |
| Dense unlabeled PCA, 32 | .901 (.889) | .928 (.969) | .821 (.892) | -.019 (-.026) |
| Dense supervised edit, 32 | 1.178 (1.384) | .952 (.969) | .911 (.929) | -.030 (-.029) |
| 60%-sparse native, 64 | .829 (.822) | — | .782 (.781) | — |
| 60%-sparse weight-only, 16 | 1.002 (1.000) | .648 (.732) | .499 (.518) | -.000 (.004) |
| 60%-sparse unlabeled PCA, 16 | 1.091 (1.092) | .096 (.157) | .041 (.053) | -.006 (-.017) |
| 60%-sparse weight-only, 32 | .838 (.832) | .904 (.937) | .729 (.735) | .011 (.015) |
| 60%-sparse unlabeled PCA, 32 | .911 (.897) | .904 (.945) | .768 (.771) | .010 (.004) |

At rank 16, the weight-only subspace preserves much more immediate-source behavior than the leading activation-energy directions. This is a stable **local compression** result across two fit/test samples and both dense/sparse checkpoints. It is not evidence that singular directions are semantically named components. The source signal may be lower energy than common surface features, which would explain why uncentered PCA under-ranks it; this explanation needs a direct test. At rank 32, unlabeled PCA recovers most source behavior but has worse general next-token NLL than weight SVD in the dense model. The supervised edit basis at rank 32 preserves the edit almost exactly while making overall token NLL much worse, demonstrating that a causal-task objective alone does not preserve the rest of the language function. The hybrid sweep does not dominate both baselines at a common rank on these panels.

These comparisons are rank-matched, not full-function-fidelity-matched. They omit direct token-level QK/OV path inspection, full attention tensor canonicalization, seed variation in **model training**, and recovery of harder delayed binding. The next useful discriminator is to target the learned rule's effective token interaction and test whether a direction or sparse path predicts the *correct source under edits* with fewer edges than these local bases, while matching whole-model fidelity. In parallel, broaden the joint DGP and gate any new rule before applying this analysis. The raw [dense panel](results/document_world_v5_clause_bond_bases_dense_summary_holdout.json), [dense replication](results/document_world_v5_clause_bond_bases_dense_summary_holdout_seed346101.json), [sparse panel](results/document_world_v5_clause_bond_bases_sparse_summary_holdout.json), and [sparse replication](results/document_world_v5_clause_bond_bases_sparse_summary_holdout_seed346101.json) include every rank and basis.
