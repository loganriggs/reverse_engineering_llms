

## H266 — Residual sensitivity modestly improves input-basis choice; source variance does not

H265 made progress: more labels improved transfer more than simply widening natural
PCA products. H266 fixes quadratic width32,528 products,624 linear natural PCs and
1,328,256 fitted readout scalars, comparing three discovery-selected input bases:

- Natural: first32 H259 natural-state PCs.
- Source: top32 eigenvectors of uncentered normalized-input source-displacement
  second moments, all9 actual full edits on training documents.
- Residual Jacobian: top32 eigenvectors of sampled mean J_E(x)'J_E(x), where
  E(x)=native bilinear write−compiled polynomial write, x is normalized input.
  Eight Gaussian output probes per training site estimate the covariance; exact
  pullbacks include BOTH native and compiled terms. RMS is not differentiated here
  because the feature function takes normalized x. Source test changes still
  recompute RMS exactly. This is not full exact tensor decomposition or Fisher.

All basis construction uses first768 discovery documents; last256 validate ridge.
Six fixed strengths0/1e-6/1e-4/.01/.1/1, no refit, all18 outcomes retained. All heads
supervised; same128 test documents reused. Linear feature map fixed; quadratic
feature centering/scales train-only. No basis chosen as a winner using test results.
Earlier tensor/context-partner machinery was inspected; this comparison targets
the residual AFTER that existing compiled program, not the original whole MLP.

| Quadratic basis | Selected ridge | Unregularized train fidelity | Selected train | Validation | Test actual, full | Test angle control |
|---|---:|---:|---:|---:|---:|---:|
|Natural-state PCA|.1|52.80%|50.23%|38.86%|37.87%|3.76%|
|Source-displacement PCA|.01|49.90%|49.46%|34.58%|33.56%|−13.04%|
|Residual Jacobian|.1|55.20%|53.02%|43.08%|41.65%|8.31%|

Source-basis gain gate FAILS with measured harm: actual full-edit fidelity drops
4.31 points, paired document-bootstrap95%[−5.00,−3.61]; small-edit drop4.21 points
[−4.95,−3.44]. Residual-Jacobian gain is positive but FAILS the locked>=5-point gate:
full gain3.78[3.02,4.57], small4.11[3.25,4.92]. The50%-fidelity gate also FAILS.
Jacobian full test interval[38.77,44.57]%. Failed threshold is not a null result.

Angle controls sharpen the distinction: source-basis change−16.81 points
[−18.87,−14.29] relative to natural, versus Jacobian+4.54[2.68,6.88]. A basis for
where source inputs vary is therefore not interchangeable with one for where the
remaining computation is sensitive. These results concern Euclidean response
energy; no downstream CE/KL or semantic-unit claim follows automatically.

**Costs and basis audits.** Core fitted readout counts are identical, but total
input reads are not guaranteed equal. Natural quadratic directions lie inside the
624 linear-PC span; source and Jacobian basis energy outside that span is19.19%
and25.62%, respectively. Each may require up to32 additional independent reads.
Basis construction differs: source covariance uses intervention displacements;
Jacobian construction uses model weights and8*3072 exact output-vector pullbacks.
Neither basis is rebuilt during inference. Top32 capture only13.99% of the sampled
Jacobian covariance trace, not a low-dimensional completeness claim. Mean-gradient
structure and stochastic probe error may affect the basis; no replicate-probe
stability claim is made. Orthogonality max error<=2.99e-7. Maximum fit normal-equation
relative residual1.23e-14. Five relevant tests pass, including exact residual
pullbacks against autograd through the compiled polynomial, response least-squares,
feature nesting and conditional-metric algebra. Full project suite not rerun.
Figure inspected. All H266 runs complete; original model unchanged.

**Scientific update.** Choice of input directions matters at fixed readout size,
but these32-direction within-subspace quadratic models remain incomplete. Source
variance alone can be actively misleading; sensitivity of the remaining function
is more informative locally. The Jacobian's modest improvement suggests that
changing the selected directions does not by itself remove omitted interactions
with coordinates outside the chosen subspace. That is a hypothesis, not an inferred
lower bound on every compact computation or a semantic interpretation of axes.

**Next discriminator: interaction partners rather than only within-subspace products.**
Use a fixed small core from the residual-Jacobian basis. Compare its within-core
quadratics plus mixed products with partner directions selected from the residual
quadratic tensor, against equally sized random-complement partners. Retain common
linear features, discovery validation and explicit coefficient/read costs. An example
budget is core16 full quadratic136 plus core16×partner24 mixed384 =520 nonlinear
features, close to528 but not exactly equal; state the difference. Partner selection
must subtract the existing compiled polynomial, not rediscover its already captured
interactions. If tensor-selected partners outperform random ones and the within32
baseline, that would connect an economical representation to actual bilinear
couplings. If not, preserve failure and examine output/rank/estimation limitations.
No new fit is yet run for this proposal. Broad research goal remains active.
