# M72: dense least squares improves retention and causal changes; sparse least squares does not

The independent per-input encoder comparison is complete for all18 frozen dictionaries. Each was evaluated with its original encoder, least squares on its original selected support, and least squares on all128 dictionary atoms. No fitting, target-dependent encoding or candidate selection occurred.

**Dense least squares improves natural-text KL in all18 dictionaries by24.7–48.7%, and improves encoded-direction causal family scores in all18 at both sites.** But it uses128 coordinates instead of roughly28–32. Its gains are not evidence of sparsity-matched superiority or meaningful semantic units.

**Own-support least squares improves local natural reconstruction in every dictionary while worsening natural-text KL in every dictionary by9.8–17.9%.** This is a direct counterexample to treating locally optimal activation reconstruction as a sufficient computation-preservation objective.

## What changed

Each activation is encoded independently. For own-support LS, the original frozen encoder chooses the support, and a pseudoinverse supplies the coefficients on that support. For dense LS, all128 atoms are available. Both retain the frozen mean, scale, decoder bias and dictionary. Solves use float64 SVD with a fixed relative cutoff1e-10, then cast the per-input reconstruction to float32. Neither variant uses a donor, pair label or teacher response when encoding an input.

Original supports are exactly preserved by own-support LS. Numerical coordinate counts and ranks are saved. On natural text, original/own-support selected-coordinate means range28.289–32; dense encoding uses128. There are no newly trained parameters. Dense encoding can precompute a single pseudoinverse derived from its dictionary; own-support encoding requires support selection and a support-conditioned solve. This implementation uses cached pseudoinverses and retains an original-encoder call for uniform control handling, so it is not an optimized inference-speed benchmark.

## Paired improvements relative to original encoding

Counts are among18 dictionaries, keeping the original objective and seed paired. Response scores below use equal-family means, not energy pooling.

| Metric | Own-support LS better | Dense LS better |
|---|---:|---:|
| Source: native background, encoded direction | 15/18 | 18/18 |
| Final: native background, encoded direction | 9/18 | 18/18 |
| Source: reconstructed background, native direction | 9/18 | 18/18 |
| Final: reconstructed background, native direction | 7/18 | 14/18 |
| Source: both reconstructed | 12/18 | 18/18 |
| Final: both reconstructed | 12/18 | 18/18 |
| Natural-text KL | 0/18 | 18/18 |
| Natural local reconstruction error | 18/18 | 18/18 |
| Intact task KL | 3/18 | 6/18 |

The intact-task KL row limits a broad retention claim: dense LS helps natural-text retention consistently, but often worsens intact outputs on these task prompts. Improvements are context- and metric-dependent. Pooled response and answer-margin scores are retained separately in the result JSON.

## Absolute fidelity still falls short

All causal ranges below are equal-family relative squared errors. Zero response is1, perfect response is0. These are not percentages of the model understood.

| Metric | Original | Own-support LS | Dense LS |
|---|---|---|---|
| Source encoded direction in native background | 0.688–1.005 | 0.700–0.963 | 0.548–0.650 |
| Final encoded direction in native background | 1.185–3.093 | 1.154–3.216 | 0.720–0.926 |
| Source with both reconstructed | 0.765–2.124 | 0.736–1.782 | 0.502–0.659 |
| Final with both reconstructed | 1.754–3.960 | 1.345–4.496 | 0.906–1.397 |
| Natural-text KL | 0.127–0.227 | 0.143–0.257 | 0.096–0.116 |

Dense LS beats zero in all18 source cases and all18 final encoded-direction/native-background cases. When both state and direction are reconstructed, only5/18 dense encoders beat zero at final positions. This preserves M63's distinction between representing a change and supporting that change in a reconstructed state. Own-support LS crosses below zero in none of the final-position cases, despite the conditional union-projection gains in M71; pair-conditioned coefficient oracles are not equivalent to independent per-input sparse encoding.

## Conceptual consequence and next test

These results support a concrete baseline, not a new claim that a sparse block is a good manifold. The dense advantage could reflect greater linear dimension, avoiding hard support transitions, better coefficients, or a combination. The sparse result shows that reducing local error can damage the downstream computation that the original coefficients were trained to preserve.

The next bounded comparison is a train-only global PCA projection at fixed ranks32 and128, on the same natural training positions used by the original dictionary pilot. It asks whether a simple linear subspace at the sparse coordinate budget is competitive, and whether the dense gain is mostly a rank effect. Rank32 and rank128 are fixed in advance; no rank search, labels, causal targets or evaluation statistics enter fitting.

This is not an equal-parameter comparison: a globally rank32 linear map has different capacity from an input-dependent32-of128 representation. That difference is the hypothesis being tested. No result will by itself establish semantic axes, native channel-removal redundancy or genuine new-task discovery.

## Verification

The preflight independently checks1008 LS reconstructions,504 original encoder replays, same-support local optimality, separate QR solves and dense affine differences. Maximum QR relative error6.03e-15; maximum dense-difference relative squared discrepancy including casts4.29e-14.

Task evaluation completed10612 native forwards. All4536 original control response/margin metrics replay M70 exactly. The independent audit covers13608 final response vectors,4536 local geometry rows and1512 saved decoder reconstructions. Maximum metric absolute discrepancy7.45e-9, including large local energies. Independent gate/LS checks are in preflight; the response audit checks decoding from saved codes.

Natural evaluation completed1760 forwards across32 documents and54 encoders. All576 original-control document records replay exactly. All1728 inventory rows and54 aggregates are checked;108 fixed full-output samples (documents32 and63 for all54 encoders) are independently scored, not every natural output. Maximum metric discrepancy2.67e-15. The comparison script derives intact task metrics from all saved endpoint vectors and retains every variant.

All jobs are terminal. A comparison-script NumPy-to-JSON scalar serialization error was corrected before the final comparison artifact was written; no native runs or numerical metrics were changed. Data, code and protocols remain hash-bound. Whole-task training exclusions are unchanged.
