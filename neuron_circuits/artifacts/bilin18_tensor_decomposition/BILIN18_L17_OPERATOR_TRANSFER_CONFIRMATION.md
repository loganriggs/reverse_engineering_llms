## H438 — Frozen shared directions transfer to independent contexts

H437 found promising small operators but left shared transfer untested. H438 freezes one shared source-coordinate basis from all32 H437 calibration trace-normalized Fisher Grams before collecting new outcomes. The basis hash is checked at evaluation. Fresh512 full documents exclude all11968 previous documents by independently rehashed identity and window comparison. Baseline-only selection chooses16 lowest-output-RMS and16 nearest-median sites at position63. Twelve new natural donor samples and12 new Gaussian draws provide shared test signals; donor pool and second moments remain frozen H413/H421. Total completed unique documents12480.

Compare exact native union128 response plus local rank8/16 omitted prediction, frozen shared rank8/16 prediction, dense analytic linear correction, and fixed random rank16. The shared basis is never recomputed on evaluation contexts. Context-dependent output writes still use the dense omitted operator. No finite targets are used to choose basis directions or prediction coefficients.

| Frozen contrast | Estimate | Joint95% interval | Gate passes |
|---|---:|---|---|
|shared16_low|0.901163|[0.863384, 0.938942]|False|
|shared16_median|0.974608|[0.936829, 1.012386]|True|
|local16_minus_shared16_low|0.012200|[-0.025579, 0.049978]|False|
|local16_minus_shared16_median|0.010237|[-0.027542, 0.048015]|False|

Shared rank16 omitted-Fisher coverage passes the >90% lower-bound gate in median contexts, but not low-RMS contexts. Neither local-minus-shared >5-point gate passes. Both predeclared joint difference intervals lie below a5-point local benefit (the low-stratum upper bound is0.049978, just below0.05); do not convert this into a newly predeclared equivalence success. All failed gates remain failed. Bootstrap resamples paired source draws and selected contexts within frozen strata; it conditions on the donor pool and does not resample baseline selection.

| Method | Low-RMS full finite-KL recovery | Median-RMS full finite-KL recovery |
|---|---:|---:|
|local8|98.283%|99.546%|
|local16|98.406%|99.653%|
|shared8|98.060%|99.252%|
|shared16|98.208%|99.411%|
|dense|98.575%|99.820%|
|random16|85.028%|80.262%|
|union128_only|82.219%|76.656%|

Frozen shared rank16 reaches98.21%/99.41% full finite-KL recovery, versus98.41%/99.65% for local rank16 and98.57%/99.82% for the dense linear correction. Shared rank8 also transfers well at98.06%/99.25%. These values concern the tested model/layer/head and intervention setup, not general next-token accuracy or model-wide explanation.

Union-only coverage on this new panel is82.22% low and76.66% median, unlike H434’s61.10%/84.67%. Low output RMS is therefore not a deterministic ordering of group coverage. The shared correction’s transfer is more robust in these observed aggregates than a blanket claim that low-RMS contexts always have worse recovery. Dense linear omitted-error coverage also changes across panels, so rank truncation is not the only source of residual error.

Dense source-restricted operator prediction agrees with an independently evaluated analytic omitted derivative within5.82e-07 relative error, below1e-5. Full-document audit passes, shared-basis hashes match, all arrays are finite, and the source second moment has positive eigenvalues without a fitted cutoff.

**What this establishes:** a small, fixed set of source directions supports accurate response prediction across independently selected receiving contexts and fresh draws, within this scope. Per-context basis adaptation is not established as necessary. It does not establish that the output writes themselves are fixed semantic features, nor does it eliminate dense operator-construction cost. The baseline model, retained-factor computation, source transforms and donor calibration are still required.

Next construction-cost test: for a fixed source direction H, the bilinear omitted derivative before input-normalization corrections is linear in the normalized baseline input x. Precompute its weight-only map T_H = D_omit[diag(L_omit H)R_omit + diag(R_omit H)L_omit]. Including RMS differentiation, the write is -(T_H x - 2 M_omit(x)*mean(x*H))/s for removal direction, where s is input RMS and M_omit(x) is the baseline omitted MLP output. The latter is available from the already-computed baseline full MLP minus retained baseline factors. Validate this exact compilation identity, then test low-rank/shared-output approximations of the fixed T_H maps without fitting finite target errors. Count stored maps and baseline dependencies: a dense compiled map for every direction can exceed the original weight cost. Any saving would concern additional intervention prediction conditional on a baseline forward pass, not replacement of the original model’s forward computation.
