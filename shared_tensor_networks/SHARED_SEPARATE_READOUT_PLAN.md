# Shared versus independently trained readouts

Use the completed d8/h4high-budget ten-head data: five teachers225001–5, three initializations226001–3,5120source scalar labels (512shared inputs),noise.05. No new task selection or adaptive tuning.

Compare completed sharedh4models (164parameters) with ten independenth4heads (740parameters) and independenth1heads (170parameters, closest integer uniform width to164). Independenth4copies shared initial hidden features and each initial head, giving identical initial source predictions. Theh1control takes the first feature/readout coefficient; its initial function and capacity differ.

Each independent head gets the same200TRFfunction-evaluation cap and fixed tolerances. The joint objective is separable, but fitting heads independently changes the trust-region schedule and total work. This is not a compute-matched optimizer comparison or an isolation of tying alone. Report all per-head outcomes, total counts and time.300individual head fits,30independent model groups,15archived shared models.

Primary transfer features are the ten predicted source outputs, individually normalized to unit GaussianRMS, for every model. This fixes adaptation feature dimension at10instead of giving independent models their100hidden products. Four whole held-out targets receive the same32/128noisy labels as before. Retain the oracle true source-output span, exact population projection floor, and rawsource-error metrics. Fixed normalizations are part of the synthetic population evaluation, not estimated from held-out labels.

Aggregate held-out128-label error over four tasks and three initializations within each teacher; report all five teacher blocks. Retain failures and stop after all300headfits and independent audits. Do not choose heads, retries or transfer regularization based on results. These are mechanistic controls on already-used scaling teachers, not another independent teacher confirmation.
