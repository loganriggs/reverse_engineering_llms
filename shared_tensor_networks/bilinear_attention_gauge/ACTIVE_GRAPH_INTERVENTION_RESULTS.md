# Targeted internal patches resolve a function-level graph ambiguity faster

The [shared-computation counterexample](SHARED_COMPUTATION_IDENTIFIABILITY.md)
has two bilinear-attention graphs with **exactly the same external
function** on every input. One computes a key detector once and sends
it to two heads; the other computes two equal-valued but distinct
detector nodes. Weight-only or data-aware canonicalization of the
external function cannot tell them apart. The graphs disagree when
the first detector is internally patched: the shared graph changes
both heads, while the duplicated graph leaves head 1 untouched.

This CPU experiment asks whether **choosing the patch inputs for
information gain** reduces the number of noisy internal measurements
needed to select the right graph.

## Selection rule and controls

In each trial, 1,024 noisy external examples estimate detector
weights; the two candidate graph families and their QK/OV factors are
supplied. The true graph, shared or duplicated, is hidden from the
selector. From a pool of 64 or 512 fresh Gaussian query/key pairs, we
predict each graph's output under a *zero patch* to the first detector.
For isotropic Gaussian measurement noise with variance $\sigma^2$,
the KL between their two predicted measurement laws at candidate
input $x$ is

$$
\operatorname{KL}_x
=\frac{\|\mu_{\mathrm{shared}}(x)-\mu_{\mathrm{duplicate}}(x)\|_2^2}
       {2\sigma^2}.
$$

We rank inputs by this disagreement **before observing any internal
patch labels**. Because $\sigma$ is constant across the pool, its
exact value does not affect the rank. The detector estimates are
noisy, so this is a plug-in information heuristic, not a proof of
optimal sequential design.

Three controls draw from the *same* pool: random order; highest
$\|q\|\|k\|$; and highest predicted **unpatched output activity**
averaged across the candidate graphs. The last control tests whether
graph disagreement adds anything beyond simply finding a large
response. After one, two, four, eight, 16, or 32 measured zero
patches, choose the graph with smaller squared prediction error.
Then predict a **held-out half-strength patch** on shifted-covariance
inputs. The selection data never include that patch strength or
input distribution.

## One-patch result

Each cell below uses 1,000 trials, varying graph factors, detector,
noise, and candidate inputs. The noise is a fraction of the clean
external-output standard deviation. Columns show graph-selection
accuracy after **one measured internal patch**.

| Candidate pool | True graph / noise | Disagreement | High activity | High input norm | Random |
|---:|---|---:|---:|---:|---:|
| 512 | Shared / 1.0 | **.993** | .858 | .735 | .553 |
| 512 | Duplicate / 1.0 | **.988** | .845 | .722 | .544 |
| 512 | Shared / 2.0 | **.970** | .828 | .648 | .516 |
| 512 | Duplicate / 2.0 | **.943** | .833 | .652 | .507 |
| 64 | Shared / 1.0 | **.887** | .775 | .663 | .524 |
| 64 | Duplicate / 1.0 | **.892** | .811 | .666 | .563 |
| 64 | Shared / 2.0 | **.825** | .719 | .600 | .539 |
| 64 | Duplicate / 2.0 | **.778** | .728 | .592 | .502 |

At noise 0.5, the active one-patch accuracy ranges .970–1.000
across the two pool sizes and both teacher graphs; high activity
ranges .845–.917. Paired-trial bootstrap intervals for
active-minus-high-activity accuracy are positive in **all twelve**
teacher × noise × pool cells. At noise 2.0 with a 512-input pool, for
example, the gap is +.142 [.118, .166] for a shared teacher and
+.110 [.085, .136] for a duplicated teacher. With a 64-input pool,
the respective gaps are +.106 [.078, .134] and +.050 [.022, .078].
The saved analyses contain intervals for one, four, and 16 patches
against all three controls.

The gain is largest at a tight measurement budget. At four patches,
high-activity selection often nearly catches up; by 16 patches
both can be near saturation. This is what an information-gain rule
should buy: fewer *expensive internal measurements*, using a larger
pool of cheap predicted candidates. The selected graph has lower
mean shifted-input half-patch prediction MSE than the random control
in all twelve one-patch cells and lower MSE than high-activity
selection in eleven. The exception is the shared-teacher, noise-2.0,
64-pool cell: 2,651 versus 2,644 MSE, despite higher graph-selection
accuracy. Heavy-tailed shifted outputs make absolute MSE sensitive
to a few trials.

## Scope

The external function offers **zero** graph-identifying evidence
in this construction; the internal patch does. The active rule
improves over generic high-signal inputs, so its advantage is not
merely selecting extreme norms. But the two graph candidates, QK/OV
factors, detector intervention address, and Gaussian input pool are
all supplied. The teacher has exactly equal detector weights in its
two copies. This experiment does **not** recover candidate graph
nodes from trained attention weights, identify a real LLM circuit,
or show that these gains persist under natural-language constraints.

For the canonicalization project, the operational lesson is to keep
two targets separate: a canonical external function, and a hypothesis
about physically reused nodes. When several physical graphs share
the same function, seek a feasible internal intervention where their
predicted outcomes diverge, then validate on a different intervention
strength and input distribution. A featurizer that claims graph
identity should face this held-out causal prediction gate.

Run active_intervention_graph_selection.py with seed 1201 and pool
512, or seed 1202 and pool 64, at 1,000 trials. Run
analyze_active_intervention_graph_selection.py for each matching
result stem. Plain JSON files retain summaries, compressed JSON
files retain all trial records, and separate analysis JSON files
retain paired bootstrap intervals.
