# Preregistration: does second-order path composition reconstruct a triple?

The learned scalar pair-synergy router failed. This does not distinguish loss of
information in the scalar summary from genuine third-order propagation. Here we
retain the complete full-vocabulary, 512-mode softmax-Fisher response tensors.

For each site and every three-layer mask `(i,j,k)`, construct

`R_hat = R_empty + Delta_i + Delta_j + Delta_k + S_ij + S_ik + S_jk`,

where `Delta_i = R_i - R_empty` and
`S_ij = R_ij - R_i - R_j + R_empty`.

Compare `R_hat` with the exact response `R_ijk`, then select a 16-mode program
from each response and compare the two programs using exact finite 5%-norm
interventions through the complete model. Evaluate all 84 triples on one frozen
site for each of 16 tensor queries in FineWeb and WikiText.

Second-order response reconstruction requires median/10th-percentile response
cosine >=0.95/0.80. Causal program equivalence requires finite-effect fidelity
>=0.95/0.80. True third-order necessity is supported if median fidelity is below
0.80 or its 10th percentile is below 0.50 anywhere.

Each finite batch includes the exact and reconstructed program for the same
triples, sharing the native baseline. No full nine-layer response or atlas-best
label is used. This tests composition, not address selection.
