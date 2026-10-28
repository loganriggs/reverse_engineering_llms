# Importance-weighted coverage of the task-agnostic route ontology

All route modes and semantic descriptions were frozen before this disjoint FineWeb panel. Route fidelity measures the nine routing coefficients; edge fidelity measures the contextual OV vector for each source edge; write fidelity measures their summed local attention write. Assigned energy is the fraction of native per-edge squared write norm whose route lies within the frozen mode gate.

## Locked hypotheses

- **C0_panel_is_prospectively_held_out: supported.** Observed: `{'start_row': 2048, 'latest_prior_stop': 1536}`
- **C1_continuous_code_beats_centroid_code: supported.** Observed: `{'layers': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]}`
- **C2_geometric_atlas_covers_material_natural_computation: supported.** Observed: `{'candidate_layers': [1, 2, 3, 4, 6, 8, 11, 13, 14, 16, 17], 'coverage': 0.34025154117006673, 'edge_fidelity': 0.2705810321396378}`
- **C3_validated_semantic_modes_cover_material_natural_computation: supported.** Observed: `{'validated_layers': [1, 4, 6, 8, 11, 13, 14, 17], 'coverage': 0.24040032830487665, 'edge_fidelity': 0.19852403421621123}`
- **C4_semantic_binding_beats_head_permutation: supported.** Observed: `{'observed_sample': 0.15599472647517265, 'sample_edges': 32768, 'null_mean': 0.014059037892423378, 'null_p95': 0.053352647798062446}`
- **C5_route_reconstruction_predicts_contextual_write_reconstruction: supported.** Observed: `{'correlation': 0.7600692413632038}`
- **C6_semantic_coverage_is_not_a_single_threshold_artifact: supported.** Observed: `{'thresholds': {'0.6': {'conditional_layers': {'route_fidelity': 0.09624217423895864, 'route_cosine': 0.31919807566129355, 'route_norm_ratio': 0.39433287976729137, 'edge_fidelity': 0.23704154077277073, 'edge_cosine': 0.49053821514694906, 'edge_norm_ratio': 0.5504231019666047, 'write_fidelity': 0.24907957858688523, 'write_cosine': 0.500410500488513, 'write_norm_ratio': 0.4639264283937047, 'assigned_edge_energy_fraction': 0.3131469610539072, 'assigned_edge_fraction': 0.11661962431426333, 'edges': 653312}, 'model_wide_assigned_edge_energy_fraction': 0.14921908484313715, 'model_wide_edge_fidelity': 0.11295374435337731}, '0.7': {'conditional_layers': {'route_fidelity': 0.08579708623024185, 'route_cosine': 0.2971308300948844, 'route_norm_ratio': 0.34702719290218315, 'edge_fidelity': 0.19852403421621123, 'edge_cosine': 0.4475701885417713, 'edge_norm_ratio': 0.4899380611278741, 'write_fidelity': 0.2113599015412847, 'write_cosine': 0.46522732587268917, 'write_norm_ratio': 0.3939773519705487, 'assigned_edge_energy_fraction': 0.24040032830487665, 'assigned_edge_fraction': 0.08290066614420062, 'edges': 653312}, 'model_wide_assigned_edge_energy_fraction': 0.11455425550008168, 'model_wide_edge_fidelity': 0.09459959185109602}, '0.8': {'conditional_layers': {'route_fidelity': 0.06810119755061406, 'route_cosine': 0.2625779641292875, 'route_norm_ratio': 0.29166392227420906, 'edge_fidelity': 0.15169058576800976, 'edge_cosine': 0.3903865593814046, 'edge_norm_ratio': 0.4170526540984285, 'write_fidelity': 0.16079400703853453, 'write_cosine': 0.4118764310779747, 'write_norm_ratio': 0.3178116411921698, 'assigned_edge_energy_fraction': 0.1705483843485648, 'assigned_edge_fraction': 0.05282927605799373, 'edges': 653312}, 'model_wide_assigned_edge_energy_fraction': 0.08126878749938589, 'model_wide_edge_fidelity': 0.0722827719976668}}}`

| L | Atlas/semantic modes | PCA route/edge/write | Centroid route/edge/write | Atlas energy/edge | Semantic energy/edge |
|---:|---:|---:|---:|---:|---:|
| 0 | 0/0 | 0.639/0.912/0.867 | 0.161/-0.051/-0.983 | 0.000/0.000 | 0.000/0.000 |
| 1 | 1/1 | 0.595/0.887/0.875 | 0.291/0.579/0.184 | 0.360/0.343 | 0.360/0.343 |
| 2 | 3/0 | 0.666/0.810/0.765 | 0.449/0.576/0.463 | 0.658/0.518 | 0.000/0.000 |
| 3 | 2/0 | 0.686/0.799/0.789 | 0.420/0.548/0.446 | 0.354/0.266 | 0.000/-0.000 |
| 4 | 1/1 | 0.643/0.864/0.784 | 0.290/0.563/0.418 | 0.257/0.214 | 0.257/0.214 |
| 5 | 0/0 | 0.672/0.983/0.984 | 0.191/0.945/0.597 | 0.000/-0.000 | 0.000/-0.000 |
| 6 | 1/1 | 0.734/0.922/0.966 | 0.393/0.709/0.619 | 0.285/0.264 | 0.285/0.264 |
| 7 | 0/0 | 0.750/0.888/0.944 | 0.448/0.668/0.539 | 0.000/-0.000 | 0.000/-0.000 |
| 8 | 2/2 | 0.792/0.817/0.835 | 0.429/0.523/0.535 | 0.544/0.432 | 0.544/0.432 |
| 9 | 0/0 | 0.709/0.878/0.938 | 0.339/0.608/0.361 | 0.000/0.000 | 0.000/0.000 |
| 10 | 0/0 | 0.712/0.779/0.907 | 0.415/0.480/0.418 | 0.000/-0.000 | 0.000/-0.000 |
| 11 | 1/1 | 0.696/0.827/0.869 | 0.332/0.464/0.580 | 0.303/0.263 | 0.303/0.263 |
| 12 | 0/0 | 0.630/0.715/0.756 | 0.154/0.285/0.453 | 0.000/0.000 | 0.000/0.000 |
| 13 | 1/1 | 0.611/0.818/0.827 | 0.252/0.421/0.451 | 0.087/0.072 | 0.087/0.072 |
| 14 | 2/1 | 0.744/0.829/0.820 | 0.460/0.578/0.574 | 0.159/0.106 | 0.117/0.075 |
| 15 | 0/0 | 0.642/0.818/0.750 | 0.344/0.353/0.369 | 0.000/-0.000 | 0.000/-0.000 |
| 16 | 1/0 | 0.782/0.831/0.817 | 0.511/0.615/0.592 | 0.137/0.117 | 0.000/-0.000 |
| 17 | 1/1 | 0.799/0.882/0.920 | 0.634/0.668/0.722 | 0.111/0.086 | 0.111/0.086 |

## Model-wide coverage and threshold sensitivity

At the primary `0.70` gate, geometric atlas modes contain **0.287** of attention-edge energy model-wide and reconstruct **0.228**. Prospectively validated semantic modes contain **0.115** model-wide and reconstruct **0.095**.

| Gate | Atlas energy/fidelity model-wide | Semantic energy/fidelity model-wide |
|---:|---:|---:|
| 0.50 | 0.404/0.277 | 0.184/0.126 |
| 0.60 | 0.348/0.258 | 0.149/0.113 |
| 0.70 | 0.287/0.228 | 0.115/0.095 |
| 0.80 | 0.212/0.181 | 0.081/0.072 |
| 0.90 | 0.110/0.102 | 0.042/0.040 |

## Interpretation guardrails

Coverage is conditional on the audited attention-route representation. It does not cover MLP-only computation, QK mechanisms not expressed by the realized route, or causal interactions in later layers. A mode can contain substantial native edge energy while reconstructing it poorly, because route direction and contextual head values interact. Conversely, good local reconstruction is not yet end-to-end next-token sufficiency; that requires finite replacement through the suffix.
