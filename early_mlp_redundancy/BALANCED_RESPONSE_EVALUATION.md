# M62: response balancing improves family coverage, with metric tradeoffs

**The matched experiment supports a real but limited coverage benefit.** Against raw causal supervision, the family/region-scaled arm lowers source-position holdout equal-family response error in all6architecture/seed comparisons, by8.4–28.9%. It lowers final-position equal-family error in5/6. These improvements use identical evaluation targets and denominators across arms; they are not merely a posthoc relabeling of scores.

The result is not uniformly better fidelity: final-position pooled error improves in only1/6 comparisons against raw supervision. Natural-text KL is worse than the KL-only control in all6starts, by1.2–2.6%, although still better than the common starting states in all6. This is foregone prediction improvement rather than regression below the initial state on that aggregate.

## What was controlled

All18final models from the fixed M60/M61 experiment, no checkpoint selection or tuning. Six common M36KL starts, three arms, identical examples/schedules/native deltas/update budget. The balanced arm changes response weighting with training-only family/region scales and a fixed floor; global initial calibration matches loss magnitudes. Task retention is final-position KL in all new arms, so historical M36 comparisons have an additional task-KL difference.

Evaluation includes all140training recipients,28pre-existing source-group holdout recipients and32natural evaluation documents. All task vectors, zeros, native errors and ineligible alignments remain. Holdouts cover six familiar development families; agreement, sentiment, pronoun_gender, number_successor and object_color have no source-group holdout coverage. This is neither pristine whole-task transfer nor genuine new-task discovery.

## Holdout causal fidelity and natural retention

Zero-response predictor=1 for normalized causal errors. Pooled error weights by native response energy; equal-family error averages each family's normalized error. Small denominators can amplify ratios, so raw errors/energies are preserved. Lower is better in every numeric column.

| Representation | Seed | Arm | Source pooled | Source family mean | Final pooled | Final family mean | Natural KL |
|---|---:|---|---:|---:|---:|---:|---:|
| scalar | 71 | intact_KL | 0.672 | 1.434 | 0.912 | 0.852 | 0.14094 |
| scalar | 71 | KL_raw_response | 0.549 | 1.041 | 0.781 | 0.782 | 0.14329 |
| scalar | 71 | KL_balanced_response | 0.479 | 0.740 | 0.807 | 0.805 | 0.14258 |
| block | 71 | intact_KL | 0.661 | 1.400 | 0.556 | 1.051 | 0.16574 |
| block | 71 | KL_raw_response | 0.563 | 1.219 | 0.542 | 0.944 | 0.16812 |
| block | 71 | KL_balanced_response | 0.487 | 0.934 | 0.654 | 0.844 | 0.16885 |
| shared_fixed | 71 | intact_KL | 0.794 | 1.520 | 1.024 | 1.142 | 0.22178 |
| shared_fixed | 71 | KL_raw_response | 0.574 | 1.082 | 0.964 | 1.201 | 0.22606 |
| shared_fixed | 71 | KL_balanced_response | 0.555 | 0.888 | 0.988 | 1.058 | 0.22661 |
| scalar | 72 | intact_KL | 0.515 | 1.135 | 0.805 | 0.991 | 0.12713 |
| scalar | 72 | KL_raw_response | 0.489 | 0.972 | 0.871 | 1.146 | 0.13067 |
| scalar | 72 | KL_balanced_response | 0.483 | 0.891 | 1.002 | 1.057 | 0.13045 |
| block | 72 | intact_KL | 0.955 | 2.005 | 0.892 | 1.263 | 0.17960 |
| block | 72 | KL_raw_response | 0.811 | 1.682 | 0.889 | 1.256 | 0.18089 |
| block | 72 | KL_balanced_response | 0.705 | 1.236 | 0.816 | 1.126 | 0.18187 |
| shared_fixed | 72 | intact_KL | 0.886 | 1.700 | 0.788 | 1.320 | 0.21541 |
| shared_fixed | 72 | KL_raw_response | 0.623 | 1.193 | 0.784 | 1.574 | 0.21730 |
| shared_fixed | 72 | KL_balanced_response | 0.678 | 0.938 | 0.836 | 1.238 | 0.21944 |

Five of six balanced models beat zero-response prediction on the source family average, versus one of six raw-response models. On final-position family averages, only two of six balanced models beat zero, so important limitations remain. Shared models do not become uniformly superior to scalar/block models; unequal active dimensions also limit architecture-wide ranking claims.

Common starting references, before this experiment:

| Representation | Seed | Source family mean | Final family mean | Natural KL |
|---|---:|---:|---:|---:|
| scalar | 71 | 1.316 | 0.844 | 0.14898 |
| block | 71 | 1.862 | 1.014 | 0.17677 |
| shared_fixed | 71 | 1.679 | 1.711 | 0.23966 |
| scalar | 72 | 1.212 | 0.938 | 0.13198 |
| block | 72 | 2.379 | 1.233 | 0.18755 |
| shared_fixed | 72 | 1.635 | 1.370 | 0.22384 |

## Every comparison retained

Balanced versus raw supervision:

- Training source and final family means improve6/6 each.
- Holdout source family mean improves6/6; final family mean5/6.
- Holdout source pooled error improves5/6; final pooled error1/6.
- Individual holdout family/model cells improve28/36 at source positions and23/36 at final positions.
- Natural KL improves2/6; holdout intact KL improves4/6.

Both causal arms lose natural-KL fidelity relative to the matched KL-only controls in all6comparisons. Raw and balanced causal supervision both improve holdout source pooled and equal-family error over KL-only in all6. The complete machine-readable comparison includes intact task top1, margin response, raw energies, initial states, and support costs; no single favorable aggregate selects a winner.

## What this establishes—and what it does not

The evidence supports the hypothesis that causal-objective scaling affects which task/region effects a representation preserves, and that training-only scaling can improve relative fidelity on previously excluded source groups. The particular floor, data and two seeds are not a universal recipe. Equal-family improvement can trade against large-effect accuracy and intact prediction.

Crucially, these probes supply the **native task-changing vector externally** after reconstructing the intact MLP1 write. They establish downstream response fidelity around the reconstructed context. They do not yet establish that the encoder/decoder can represent that change, much less discover a semantic manifold or reusable computational unit. Native neuron redundancy, cancellation and distributed implementation remain separate mechanistic questions.

This closes the objective-balance experiment without a floor/LR/seed sweep. The next discriminating test separates reconstruction of the background state from reconstruction of the task-changing direction, using all final candidates rather than choosing the best score.

## Verification

10146evaluation forwards:7942training-target,1596holdout-target,608natural. Independent NumPy audits rescore all9036task response vectors and3024task baselines, verify common delta hashes, complete inventories and all summaries/macros. Natural evaluation verifies all576model/document records and independently rescores36fixed full-output samples (docs32/63 for every model). Maximum task metric discrepancies are below1e-14; natural sample discrepancy1.78e-15. Native baselines/targets replay exactly. Transformer forwards are not independently reimplemented.

## Next: state-versus-direction factorial, no new fit

On the fixed14source-group holdout pairs, compare native versus reconstructed intact background and native versus reconstructed donor-minus-recipient direction. The native/native corner is the teacher; reconstructed-background/native-direction is already measured here. Add the two missing corners for all18models, both source/final regions and identities, without a norm oracle. Compute reconstructed direction by independently encoding/decoding recipient and donor native writes, then differencing the aligned positions. Do not freeze recipient supports onto donors.

This tests whether better response-context fidelity coexists with distorted task directions, or extends to the representation's own changes. Report local direction error and native response error together. Nonlinear interactions mean the four corners are not an additive attribution decomposition. This is a bounded validation step toward meaningful units, not another objective-tuning loop.
