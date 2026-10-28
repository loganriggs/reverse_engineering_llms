

## H265 — More response labels transfer better than simply widening PCA products

H264 made progress: exact all-head response fitting showed both empirical class
error and a train/validation gap. H265 crosses quadratic input width32/64 with
256/768 documents carrying response supervision.624 linear features remain fixed;
all upper-triangle products of the first32 or64 natural-state PCs are added. Width32
has1152 total features, width64 has2704. This is a capacity expansion, NOT equal cost.

Common PCA/input geometry is frozen from H259's first768 discovery documents.
Width32 reuses old feature normalization; width64 normalization uses the same768
natural states WITHOUT response labels. Therefore the data axis measures labeled-
response supervision conditional on shared geometry, NOT total data exposure.
First256 is nested in first768; validation is the same last256 discovery documents.
All9 actual heads/full edits train; ridge0/1e-6/1e-4/.01/.1/1 selected by validation
response error in each cell. No refit or test selection. Save all24 fits and choices
before test; evaluate selected and unregularized candidates at both scales with
actual/angle payloads. Same128 test documents reused; no new-population claim.

| Quadratic width | Labeled documents | Unregularized train | Selected ridge | Selected train | Validation | Test actual full edit |
|---|---:|---:|---:|---:|---:|---:|
|32|256|64.64%|1|46.52%|31.04%|29.91%|
|32|768|52.80%|.1|50.23%|38.86%|37.87%|
|64|256|84.75%|1|63.44%|35.22%|33.80%|
|64|768|67.85%|.1|64.91%|43.24%|42.10%|

These are missing-response vector-energy fidelities. Decreasing empirical training
fidelity as more contexts enter the sample is not regression of the algorithm;
training populations differ while validation/test stay fixed. Wider features reduce
training residual strongly, but much less of that gain transfers.

**Locked gates.** Data-transfer PASSES: increasing256→768 labels improves full-edit
test fidelity7.96 points[7.08,8.94] at width32 and8.31[7.45,9.30] at width64. Small
edits give8.25 and8.54 points, also meeting>=5 points with paired interval>0.
Width-transfer FAILS its>=5-point threshold despite positive gains: full-edit32→64
gains3.88[3.40,4.33] points at256 docs,4.23[3.72,4.75] at768. Small-edit gains3.95
and4.24. Half-response gate FAILS: largest cell full-edit fidelity42.10% interval
[39.16,45.10], below50%. Intervals are paired whole-document bootstrap2000,
conditional on fitted models and fixed evaluation controls. Failure is not zero gain.

**Overfitting is visible.** Width64/256 unregularized training84.75% coexists with
validation−9.15%, test−15.79%. Regularization raises its test fidelity to33.80%.
At width64/768 unregularized validation38.45%, test36.76%, while selected ridge gives
43.24/42.10%. Thus high empirical capacity does not establish a learned transferable
computation. More labeled contexts help this family; they do not close the remaining
gap or establish what happens at much larger scales.

**Cost and direction coverage.** Fitted readout scalars including intercept rise
1,328,256→3,116,160 (2.35x); float32 readout bytes5,313,024→12,464,640. Both use624
linear PC reads; quadratic products increase528→2080. PCA projection/feature
normalization storage and unchanged compiled-program cost are additional, so these
are not total deployed parameter counts or runtime speedups. Largest-cell angle-
control fidelity is only7.72%; all four full-edit angle cells6.08/2.80/8.13/7.72%
in table order. Angle draws use a new fixed seed; only ACTUAL width32/768 is an exact
replay of H264. All-head supervision is not held-out-head transfer. No downstream
logit/CE or full replacement claim is made from this response-only audit.

**Numerics.** All1152/2704 feature directions retained. Maximum normal-equation
relative residual1.14e-14; worst Gram condition4.48e4. Width32/768 reproduces prior
training/validation/test actual results. Four relevant tests pass: width32 replay,
nested products at64, response solve against SVD, joint fit and causal patch (four
test functions total). Full project suite not rerun. Figure inspected. All H265
runs complete; original model unchanged; every ridge outcome retained.

**Scientific update.** T07 finite-data/generalization concerns gain support; expanding
this natural-PCA quadratic family mostly enlarges empirical fitting ability and
requires stronger regularization at low data. Useful interactions are still missing,
but simply multiplying more high-variance state coordinates is an inefficient route
to transferable response representation on this panel. This does not rank all data
versus architecture investments or rule out a better input basis at fixed width.

**Next discriminator.** Hold quadratic width/readout count fixed and compare natural-
state PCA directions with directions selected from discovery source perturbations
and residual-response sensitivity (or the known residual tensor). Use only training
states/targets to choose bases, keep common validation and fit protocol, and report
actual/angle transfer plus basis-construction costs. A better response-aware basis
at fixed width would distinguish omitted informative directions from merely needing
more products. First inspect earlier tensor/context-partner machinery so this
comparison targets the residual that remains after the existing compiled program,
not a repeat of already-established natural-state versus response mismatch. Goal active.
