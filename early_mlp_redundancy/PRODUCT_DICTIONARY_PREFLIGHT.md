# Product-dictionary initialization and derivative checks

Follow-up: [calibration and four fixed fits are now complete](PRODUCT_DICTIONARY_FITS.md). The account below records the earlier preflight stage.

M95 returns to the featurizer comparison after closing the M89–M94 deletion and error-decomposition series. The implementation and local derivative checks are complete. **No calibration or fitting has run.** Optimizer/schedule auditing and training-only calibration remain before fitting.

## Computational question

A low-dimensional output does not imply a small number of distinct input computations. Here all 4,608 native bilinear products are retained, while their outputs are expressed in a fixed 128-direction dictionary. Only the 128 × 4,608 coefficient matrix trains. This decouples output geometry from the number of input products; it does not claim compression of those products or discovery of semantic units.

The two output dictionaries are the previously frozen dense KL-only starts, seeds 71 and 72. They were chosen to avoid gating, not selected by evaluation rank. The controlled comparison will be KL-only versus joint three-corner causal supervision, with identical starting states, schedules, optimizer, data, and update budget. This has a stronger native prior and more coefficients than the previous input-featurizer architecture; cross-architecture differences cannot isolate architectural causation.

## Initialization and independent audit

For native products h(z) = (Lz) ⊙ (Rz), output dictionary B, native down matrix W, and center c, define

    prediction = offset + B C (h / RMS)
    C₀ = B⁺ W diag(RMS)
    offset = (I − BB⁺)c

This reproduces the full projected native function in real arithmetic. RMS uses only the 4,092 unique natural training positions; no product has zero RMS. Native Down_bias is excluded here and added exactly once by the caller.

NumPy/SVD initialization was independently checked with Torch/QR on all training positions. Relative errors were 2.05e-15 and 2.14e-15 in float64, and 3.61e-7 and 3.63e-7 in float32. Only the 589,824 coefficient parameters are trainable. Native products, output dictionary, normalization, and offset are frozen.

## Native execution, loss, and gradients

Eight fixed training cases cover both seeds, both endpoints, and the first equal-length and unequal-length pairs. Independent NumPy loss arithmetic agrees with every causal branch (maximum response-error discrepancy 3.33e-16). Teacher recovery loss is at most 2.03e-12. Four actual MLP forward overrides match callback replacement exactly while hooks forbid calling native MLP1 Left, Right, or Down. Identity responses are exactly zero. This stage used 70 native forwards.

The original finite-difference protocol used eight fixed directions and 24 steps (294 forwards). All four endpoint directions and both natural-loss coefficient directions passed. Both task coefficient directions passed h=.001 but failed h=.0003 and .0001; these failures remain in the original receipt.

A single bounded follow-up froze steps .1, .03, .01 before execution, retained the original small steps, and kept the same examples, directions, and absolute-plus-relative tolerance. It checked both direct coefficient perturbations and the exact linear coefficient-to-write tangent. Both seeds passed all three larger steps in both modes. An independent float64 analytic pullback agreed with autograd to 7.65e-7 and 8.75e-7 relative error.

At h=.0001, the computed write changes differ from the intended linear perturbation by roughly 25–32%; at h=.1, this falls below .041%. Even exact-write-tangent loss differences fail at the smallest steps, implicating downstream float32 resolution as well as coefficient-to-write arithmetic. These checks support a numerical-resolution explanation; they do not prove training stability or exhaustively test derivatives. The follow-up used 350 forwards. Total native preflight forwards: 714. No optimizer update occurred.

## What this can establish

If joint supervision improves simultaneous background-and-change replacement on held-out tasks while retaining natural behavior, it would support learning compatible coordinates over the retained computations. It would not establish that the products are redundant, that 128 computations suffice, or that the resulting coordinates are meaningful manifolds. If this strong baseline fails, inspect the fixed run before proposing another architecture; do not launch a rank or optimizer sweep.

Next: audit the exact optimizer/state/schedules, calibrate the one global causal weight from the two initial states on the fixed training schedules, independently audit that calibration, then freeze the numeric training receipt. Preserve all seven whole-task exclusions and fourteen source-group holdouts. Only after these gates pass run the four prescribed fits.

Receipts: [design](validation/product_dictionary_featurizer_design.json), [initial states](validation/product_dictionary_initial_states.json), [initialization audit](validation/product_dictionary_initialization_audit.json), [native checks](validation/product_dictionary_native_checks.json), [original derivative checks](validation/product_dictionary_gradient_checks.json), [bounded diagnostic](validation/product_dictionary_gradient_diagnostic.json).
