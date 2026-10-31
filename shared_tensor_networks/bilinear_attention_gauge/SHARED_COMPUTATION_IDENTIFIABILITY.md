# A function-only limit on recovering shared attention computations

`shared_computation_identifiability.py` constructs two **two-head bilinear-attention graphs** over a continuous query \(q\) and key \(k\). Each head has two QK factors and an OV value factor:

\[
a_h(q,k)=(q^\top A_h k)(q^\top B_h k)(c_h^\top k),\qquad
y_h=a_h(q,k)g_h(k).
\]

In the **shared DAG**, one detector \(g(k)=r^\top k\) is computed once and its result feeds both heads. In the **duplicated graph**, two separate detector nodes each compute \(r^\top k\), with independent parameter occurrences set to the same numerical values. The graphs have different computation and parameter-reuse structure but the same external polynomial map:

\[
(y_0,y_1)=(a_0r^\top k,\;a_1r^\top k)
\]

for **every** \((q,k)\). Thus no amount of external input–output data, even under arbitrary input edits, can distinguish these two particular graphs. A uniform metric, a corpus-weighted metric, a weight-only function tensor, and a canonicalization of that function all tie. A minimum-parameter description would select the shared graph (\(d\) distinct detector parameters versus \(2d\)), but that selection is a **prior**, not proof of the physical graph. If the actual implementation duplicates the computation, minimum description length makes the wrong internal causal prediction.

The predictions split under an internal intervention. Set the output of the first detector node to zero. In the shared DAG it is the sole detector, so both heads become zero; in the duplicated graph only head 0 becomes zero and head 1 is unchanged. Because both heads are generally active, one measured intervention distinguishes the candidate graphs.

Five seeds with \(d=6\), 8,192 samples per distribution, and three input laws were checked. External outputs and the response to deleting an **input** key coordinate agree exactly in float64 in every run. The internal-node intervention differs in every run. The untouched second head carries these fractions of total output energy:

| Input law | Minimum across seeds | Maximum across seeds |
|---|---:|---:|
| Gaussian | .347 | .624 |
| Shifted covariance | .002 | .618 |
| Sparse keys | .447 | .801 |

These fractions quantify the wrong prediction made by the alternative graph after the internal edit. The JSON files contain absolute MSE for each seed and law. This is an **exact counterexample**, not a statistical claim about how commonly trained networks land on equal detector weights. It also does not say model internals are unknowable when the architecture, weight tying, or activation traces are accessible; those are additional evidence beyond the external function.

This matters for the canonicalization project. A sparse or short decomposition can provide a useful explanatory model, but external fidelity cannot certify that a reused node is physically reused. A graph-recovery assay should keep the graph identity hidden during fitting, separate observational and internal-intervention data, and require prediction of **held-out intervention targets and input distributions**. A function-only method should be allowed to abstain about graph identity; an intervention-aware method should earn its graph claim by making the correct two-head effect prediction. The same logic applies to branch swaps and redistributions across heads: a canonical gauge of one representation is not automatically a causal decomposition of the implemented model.

## Noisy recovery and held-out intervention transfer

`intervention_graph_selection.py` gives both candidate graphs the true QK/OV factors but **estimates their detector weights** by least squares on 1,024 noisy external examples. The teacher graph is shared or duplicated, with equal detector vectors so external outputs remain exactly identical before observation noise. A shortest-parameter-count rule always chooses shared: it is correct for a shared teacher and wrong for a duplicated teacher. No external validation set can fix that structural ambiguity.

The recovery test instead selects the graph by squared error on \(n\) noisy **zero-patch** interventions to the first detector node, then evaluates a **held-out half-strength patch** on covariance-shifted inputs. Each condition uses 1,000 randomized teacher/measurement trials. Noise is Gaussian with standard deviation given as a fraction of the clean observational output standard deviation. Graph-selection accuracy is:

| Noise / true graph | 1 patch | 8 patches | 16 patches | 32 patches |
|---|---:|---:|---:|---:|
| 0.1 / shared | .647 | .947 | .994 | 1.000 |
| 0.1 / duplicate | .681 | .955 | .988 | 1.000 |
| 0.5 / shared | .563 | .783 | .889 | .967 |
| 0.5 / duplicate | .538 | .776 | .877 | .966 |
| 1.0 / shared | .538 | .676 | .793 | .885 |
| 1.0 / duplicate | .501 | .647 | .764 | .864 |
| 2.0 / shared | .548 | .631 | .713 | .800 |
| 2.0 / duplicate | .526 | .614 | .685 | .771 |

The held-out half-patch prediction error also falls as selection improves; the JSON records its MSE and the alternative graph's MSE for every cell. This is a controlled **information-gain** example: targeted internal data can resolve a distinction that arbitrarily large external data cannot. It is not a trained attention-graph discovery result. QK/OV factors and the two candidate graph families are supplied, the intervention address is known, and the equal-detector condition is exact by construction. A realistic recovery test must infer candidate components and show that a small number of interventions generalizes across additional targets and tasks.

Reproduce on CPU with `shared_computation_identifiability.py --seed 0` through `--seed 4`, followed by `intervention_graph_selection.py --trials 1000`. No GPU is used.

The [active-intervention follow-up](ACTIVE_GRAPH_INTERVENTION_RESULTS.md)
chooses zero-patch input pairs by the two candidate graphs' predicted
Gaussian measurement KL before seeing internal labels. It compares the
same candidate pool against random, high-input-norm, and high-unpatched-
activity choices, then tests a held-out half-strength patch under
covariance shift. A single active patch identifies the graph more
reliably than the controls across both teachers, three noise levels,
and 64/512-pair pools. The advantage shrinks as both strategies receive
more patches. This demonstrates *which internal observation to request*
in a known candidate-graph problem; it does not infer graph candidates
or physical nodes from a trained LLM.
