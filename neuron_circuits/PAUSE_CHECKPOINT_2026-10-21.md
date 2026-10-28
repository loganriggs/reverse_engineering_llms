# Research pause checkpoint — 2026-10-21

## State at pause

The breadth-first analysis is paused cleanly after hypothesis H222. No training or
analysis process is using the GPU. The persistent `bilin18` supervisor is only the
pre-existing environment runner; its children are idle polling shells.

The latest sequence completed a task-blind MLP atlas, full-space and finite-causal
red teams, a data-versus-weight scaling comparison, a known-ontology residual
atlas, a recursive ontology test, and a prospective finite test of the layer-17
semantic code.

## Most important current conclusions

- A single native-neuron or discrete-cluster ontology does not describe the model.
  Continuous subspaces are the better compact object, and allocation, residual
  write, and CE-attribution views disagree enough that “feature” must name its
  intended role.
- Early and middle MLPs are not causally reconstructed by the current 50 semantic
  labels. Their semantic-only full-logit fidelities are `-0.515`, `-0.025`, and
  `-0.049` at L1/L8/L12. Adding a rank-64 residual raises them to `0.602`, `0.248`,
  and `0.197`.
- Layer 17 is qualitatively different. Semantic-only fidelity is `0.861`, and a
  16-dimensional residual raises it to `0.983`.
- The L17 semantic result is not merely a shared mean write: correct context-label
  binding gives full-logit/CE fidelity `0.886/0.837`, versus `0.770/0.589` for the
  mean and permutation-null p95 `0.714/0.606`.
- The 50-feature vocabulary explains only `0.176/0.040/0.054/0.450` of natural
  write variance at L1/L8/L12/L17. Stable unexplained residual geometry remains,
  especially in early and middle layers.
- Recursive human labeling finds real examples—such as byte-fragment continuation
  and function-to-continuation regimes—but adds at most `0.028` population write
  R2. Readable examples are not yet an efficient ontology.

## Interpretation boundary

The evidence supports depth-dependent compilation: early/middle layers use dense,
context-conditioned continuous computations that our present semantic vocabulary
does not causally isolate; the final MLP exposes a compact answer-facing quotient.
This does not establish a unique “true” basis, nor does it show that early layers
lack semantics. It shows that the present labels are descriptive but insufficient
there.

## Best next experiment

Start from the reproducible residual write directions at L1/L8/L12 and contract
them backward through the exact bilinear tensor and incoming attention paths. Test
whether each output direction has a low-rank input discriminator that predicts its
signed coefficient on fresh FineWeb text. Freeze those candidates before semantic
inspection, then validate them with finite path interventions and matched random
subspaces. This is the shortest route from stable unexplained geometry to explicit
input-conditioned computations.

## Key records

- `CURRENT_COMPUTATION_REGIME_UPDATE.md` through item 76
- `COMPUTATION_HYPOTHESIS_LEDGER.md` through H222
- `artifacts/bilin18_tensor_decomposition/BILIN18_TASKBLIND_MLP_MULTIVIEW_ATLAS.md`
- `artifacts/bilin18_tensor_decomposition/BILIN18_TASKBLIND_MLP_FULLSPACE_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/BILIN18_TASKBLIND_MLP_SUBSPACE_FINITE.md`
- `artifacts/bilin18_tensor_decomposition/BILIN18_MLP_WEIGHT_VS_DATA_SCALING_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/BILIN18_KNOWN_ONTOLOGY_RESIDUAL_ATLAS.md`
- `artifacts/bilin18_tensor_decomposition/BILIN18_RECURSIVE_FEATURE_ONTOLOGY_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/BILIN18_KNOWN_ONTOLOGY_RESIDUAL_FINITE.md`
- `artifacts/bilin18_tensor_decomposition/BILIN18_L17_SEMANTIC_MEAN_BINDING_REDTEAM.md`

