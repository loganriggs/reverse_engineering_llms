## H752 — Compact operation coordinates mediate a specific part of upstream effects

Following the H751 sweep, this test selects L3–6 MLPs and L5 attention for strong full-node dependence, plus L11 attention as a direct-attribution comparator. It restores native rank-four coordinates in both L13H0 query/key states while retaining the direction of the altered orthogonal complement. The complement is rescaled to its native norm, and the native raw-state scale is restored. This spherical coordinate clamp keeps both native selected coordinates and native total normalized norm. It is not a purely linear subtraction; complement rescaling is part of the intervention.

Original/pooled operation bases are compared with identity, wording and random equal-rank bases, full-node clamps and free controls. All240 H747 contexts remain; the table uses120 native-competent narratives. Sites were selected from previous results and data are reused, so this is an exploratory mediation test, not fresh prospective validation.

| Upstream output swapped | Original operation error | Pooled operation error | Identity error | Wording error | Random error |
|---|---:|---:|---:|---:|---:|
| L3 MLP | 32.2% | 38.1% | 50.5% | 86.9% | 99.9% |
| L4 MLP | 40.0% | 35.0% | 73.7% | 90.2% | 99.3% |
| L5 MLP | 33.1% | 49.2% | 63.2% | 88.4% | 99.5% |
| L6 MLP | 36.7% | 49.3% | 66.1% | 92.9% | 98.2% |
| L5 attention | 25.4% | 31.3% | 101.8% | 92.1% | 100.7% |
| L11 attention | 94.3% | 101.0% | 86.9% | 102.4% | 105.4% |

Error compares free-minus-compact-clamped intended-margin change with free-minus-full-clamped change, divided by the RMS of the latter. It measures approximation of the selected nodes' dependence contrast, not of the entire upstream-swap effect. Full-node dependence RMS is0.111,0.125,0.114,0.119,0.126,0.0345 logits for the listed sites, respectively. Most of the broader upstream-swap effect still survives even a full-node clamp, as H751 established.

Original operation dependence alignment is78.6%,65.8%,78.7%,69.4%,88.5%,80.7% across these rows. L11 attention illustrates why alignment alone is insufficient: high alignment coexists with94.3% RMS approximation error, implying large amplitude and/or context discrepancies. Pooled L5-attention alignment exceeds100% despite31.3% error, again precluding a literal fraction-of-causation interpretation.

Interpretation: the verified rank-four component captures a substantial, specific part of how L5 attention and middle-layer MLP output changes affect the selected downstream query/key nodes. L5 attention has the strongest separation from nuisance controls in this panel. MLP identity controls also recover some dependence, so operation and identity-related structure may overlap there. L11 attention is not validated as a clean compact mediator despite its large direct query attribution.

The complement correction ranges0.876–1.062 across all interventions and controls. Coordinate restoration and norm preservation are verified, but this correction means the result concerns the stated spherical intervention; it does not isolate four coordinates without any change outside them. Full-state clamping also includes coordinates outside the proposed feature. Downstream nonlinearities prevent unique additive mediation claims.

Audits:11,802 forwards =10,080 site/scope interventions +1,680 no-swap controls +42 self-swap controls. Exact H751 free/full margin replay; exact raw full-node clamps; compact coordinate error<=6.32e-7; normalized norm error<=1.65e-6; no-swap/self-swap logit error<=1.34e-5; all hashes unchanged and outputs finite. Native-incompetent lists are retained separately in the data. No original reserved-test evaluations; natural document count26,112 unchanged.

Next: resolve L5 attention into individual heads and test whether its component-specific influence is concentrated or distributed. Use that evidence, alongside the contrasting MLP route, to choose activations for unsupervised featurizer recovery. Continue checking diverse tasks and nuisance sensitivity; neither causal specificity in one operation panel nor low reconstruction error establishes a meaningful manifold by itself.
