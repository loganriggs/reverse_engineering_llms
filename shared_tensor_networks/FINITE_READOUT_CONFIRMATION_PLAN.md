# Fresh-teacher fixed-budget confirmation

Written before running the confirmation. Keep the recipe unchanged after observing results.

Five fresh d6/h3teachers222001–222005, three random initializations223001–223003, and one exact-teacher initialization diagnostic. Head counts1/3/6, source scalar-label budgets384/1536, normalized iid label-noise standard deviation.05. Total120fits. Same model, target normalization, optimizer,200evaluation cap, fixed tolerances and no restarts as the finite-data development panel.

Primary endpoint: mean exact Gaussian transfer MSE across four whole held-out heads with128new labels/head, then arithmetic mean across the three random initializations within each teacher. Compare three versus one head and six versus one at each source budget. Report all five paired teacher results, median and range, and win counts; do not count initializations or held-out heads as independent teachers. No pooled development teachers. Teacher-start comparisons are separate privileged diagnostics.

Retain source fit, all component cosines,32-label transfer, oracle transfer, polynomial baselines, optimizer status and every failed/budget-limited fit. Oracle transfer has the same adaptation labels/noise. Input/noise pools and head prefixes remain paired within teacher. More heads at fixed scalar-label budget mean fewer distinct input examples.

Stop after120fits and independent audits. Do not add selected seeds, modify head count, extend failed runs or change solver settings to obtain a preferred outcome. A mixed or negative replication is a result. This tests the fixed small architecture; it does not alone establish larger-network scaling or universally optimal annotation allocation.
