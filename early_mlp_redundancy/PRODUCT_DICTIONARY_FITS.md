# Four product-dictionary supervision fits completed

All four fixed fits completed and passed structural, schedule, and fixed-case loss audits. Held-out evaluation has **not** run. This establishes an executable comparison, not a finding that the learned coordinates are reusable computational units.

## The controlled question

The model retains 4,608 native bilinear products and uses a frozen 128-direction output dictionary. Only the coefficients connecting them train. Each of two initial states receives either ordinary output-distribution supervision (KL-only) or that same supervision plus causal response matching. The causal objective tests three situations: predicted change on native background, native change on predicted background, and both predicted together.

This targets the earlier finding that individually improved approximations can combine worse because their errors cease to cancel. It asks whether directly supervising their compatibility improves the *joint* computation. It does not assume that distribution across many channels means redundancy, or that a small output space implies few computations.

## Calibration and training

The setup audit verifies that only the 589,824 coefficient parameters are trainable, all other buffers remain fixed, and Adam starts fresh with the same defaults as M87: learning rate 1e-4, betas (.9, .999), epsilon 1e-8, no weight decay. There is no warmup, clipping, dictionary renormalization, optimizer sweep, or evaluation-based selection.

One global causal weight was calibrated from both initial states on the fixed training schedules:

| Quantity | Value |
|---|---:|
| Mean intact KL | 0.1011081831 |
| Mean balanced causal error | 0.5449660001 |
| Fixed causal weight (ratio) | 0.1855311764 |
| Scheduled endpoint cases | 264 |
| Independently checked response vectors | 2,370 |
| Exact-zero identity vectors | 792 |
| Maximum independent metric discrepancy | 9.99e-15 |

Calibration used 1,842 native forwards and zero optimizer updates. The weight and numeric training protocol were frozen before fitting.

Each fit used the same prescribed 132 updates: 66 natural-text updates interleaved with 66 task updates, averaging both task endpoints. The eligible task pool contains 70 pairs across 11 families. The sampled schedules visit **35 distinct pairs for seed 71 and 41 for seed 72, 49 in their union**; both cover all 11 families. Thus “70 available pairs” must not be reported as 70 actually visited per fit. All seven whole-task families and fourteen source-group pairs remain excluded from training.

All four final states were retained. Training used 528 total optimizer updates and 2,370 native forwards. The audit checks every schedule assignment, loss-part arithmetic, optimizer step count and defaults, exact frozen-buffer equality, finite final states, and matching final/progress checkpoints. It does not independently replay the entire optimizer trajectory.

A separate final-state arithmetic check used the first training pair and natural training document 0 for every model (36 forwards). Independent NumPy task and natural loss calculations agree within 4.90e-16 maximum discrepancy. The KL-only code path exactly matches the KL term in the joint code path. These fixed training cases are implementation checks, not transfer measurements.

## Next decisive evidence

The evaluation inventory is frozen to all four final models plus their two initial full-projection references. Evaluate all 32 natural evaluation documents, all seven whole-task families (also report the fixed five competent families), and all fourteen source-group pairs. Report reconstruction, KL, CE, top-token agreement, task accuracy, and all three causal corners separately; retain both pooled and equal-family aggregates and every weak/null/ineligible case.

The primary contrast is joint supervision versus KL-only within each seed, with both compared to their own initial state. Improvement must survive simultaneous background-and-change replacement. Better isolated responses or reconstruction alone are insufficient. These historical holdouts have been evaluated before; they are not pristine new tasks. Even successful transfer would justify subsequent grouping/manifold tests, not establish semantic manifolds by itself.

A [frozen interpretation plan](validation/product_dictionary_interpretation_plan.json) distinguishes compatible distributed coordinates, isolated gains that fail to compose, task-specific tradeoffs, and remaining output-space or optimization limitations. No new training should precede the six-candidate evaluation and its independent audits.

Evidence: [setup audit](validation/product_dictionary_setup_audit.json), [calibration audit](validation/product_dictionary_calibration_audit.json), [training protocol](validation/product_dictionary_training_protocol.json), [training audit](validation/product_dictionary_training_audit.json), [final loss audit](validation/product_dictionary_final_loss_audit.json), [six-model inventory](validation/product_dictionary_evaluation_models.json). Earlier initialization and derivative evidence is in [the preflight report](PRODUCT_DICTIONARY_PREFLIGHT.md).
