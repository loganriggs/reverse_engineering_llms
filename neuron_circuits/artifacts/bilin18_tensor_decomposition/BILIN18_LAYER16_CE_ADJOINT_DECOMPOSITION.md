# Model-side versus target-token breadth in the Layer-16 CE adjoint

The exact CE write adjoint is decomposed as `g_CE = g_logsumexp - g_target`. The first term is determined by the model's probability distribution; the second selects the dataset-supplied true token. A top-1-versus-runner-up margin supplies a second model-chosen endpoint. All terms are contracted through the same exact `J B(x)^T g` tensor path.

## Locked hypotheses

- **D0_adjoint_decomposition_is_exact: not supported.**
- **D1_target_readout_dominates_ce_energy: supported.**
- **D2_prediction_term_has_compact_forward_interface: not supported.**
- **D3_target_term_is_broad_at_rank64: supported.**
- **D4_model_margin_has_compact_forward_interface: not supported.**
- **D5_target_adjoint_is_broader_than_prediction_adjoint: supported.**
- **D6_target_identity_predicts_repeated_target_reads: not supported.**
- **D7_target_identity_clusters_effective_adjoints: supported.**
- **D8_ce_breadth_is_primarily_target_readout_breadth: not supported.**
- **D9_target_prototypes_transfer_across_contexts: not supported.**

Autograd error: `6.27e-07`; gradient decomposition error: `3.33e-04`; contraction error: `8.49e-07`.

| Component | Output-covector R90 | Write-adjoint R90 | Effective-adjoint R90 |
|---|---:|---:|---:|
| ce | 240 | 230 | 260 |
| prediction | 117 | 93 | 83 |
| target | 249 | 226 | 188 |
| margin | 228 | 220 | 283 |

| Corpus | Component | Forward basis R32/R64/R128/R256 | Own-adjoint basis R32/R64/R128/R256 |
|---|---|---:|---:|
| fineweb | ce | 0.717/0.804/0.825/0.874 | 0.373/0.407/0.482/0.601 |
| fineweb | prediction | 0.538/0.794/0.827/0.870 | 0.728/0.779/0.820/0.859 |
| fineweb | target | 0.555/0.774/0.810/0.853 | 0.648/0.692/0.747/0.804 |
| fineweb | margin | 0.607/0.678/0.719/0.777 | 0.180/0.249/0.364/0.486 |
| wikitext | ce | 0.628/0.704/0.750/0.802 | 0.269/0.343/0.453/0.547 |
| wikitext | prediction | 0.568/0.850/0.868/0.897 | 0.812/0.841/0.864/0.893 |
| wikitext | target | 0.592/0.827/0.848/0.882 | 0.728/0.756/0.791/0.819 |
| wikitext | margin | 0.569/0.651/0.698/0.753 | 0.176/0.230/0.339/0.466 |
| tinystories | ce | 0.792/0.847/0.867/0.897 | 0.369/0.447/0.552/0.657 |
| tinystories | prediction | 0.693/0.892/0.910/0.936 | 0.855/0.896/0.919/0.943 |
| tinystories | target | 0.699/0.882/0.902/0.932 | 0.794/0.840/0.873/0.908 |
| tinystories | margin | 0.764/0.829/0.849/0.879 | 0.331/0.427/0.528/0.648 |

| Corpus | Prediction/CE energy | Target/CE energy | Pred-target cosine | Cancellation ratio | Prototype coverage/fidelity |
|---|---:|---:|---:|---:|---:|
| fineweb | 3.575 | 4.088 | 0.871 | 3.913 | 0.453/0.328 |
| wikitext | 3.460 | 4.951 | 0.895 | 4.085 | 0.430/-0.329 |
| tinystories | 5.605 | 6.357 | 0.918 | 4.889 | 0.586/0.526 |

Discovery repeated-target audit: 0.505 of sites are covered; leave-one-out prototype fidelity is 0.202; same-minus-different effective-adjoint cosine is 0.270.

This experiment assigns breadth to a behavioral question, not to an abstract model once and for all. Target-logit breadth describes the family of token readouts; prediction/margin breadth more directly describes model-chosen inference interfaces.
