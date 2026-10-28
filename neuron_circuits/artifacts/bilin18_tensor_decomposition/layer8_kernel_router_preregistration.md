# Preregistration: nonlinear nearest-state route atlas

The linear state router plateaus below the locked lower-tail threshold. This
test asks whether the complete 84-mask response surface varies smoothly over
the normalized Layer-8 state manifold.

For a target state, cosine-match its nearest FineWeb training states, average
their full-fidelity surfaces with softmax weights, and choose the predicted-best
mask. Sweep neighbor counts 1, 2, 4, 8, and 16 and temperatures 0.01, 0.05, 0.2,
and 1.0. Select by query-grouped FineWeb 10th-percentile fidelity, then median,
then fewer neighbors and lower temperature. Refit on all 32 FineWeb sites and
evaluate WikiText once.

Validation requires median >=0.95 and 10th percentile >=0.80 on both grouped
FineWeb and WikiText. A nonlinear advantage requires a +0.05 lower-tail gain
over the frozen linear-router result on both. Failure rejects smoothness in raw
state cosine neighborhoods at this data scale, not all nonlinear routers.
