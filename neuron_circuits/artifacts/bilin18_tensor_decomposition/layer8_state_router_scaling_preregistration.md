# Preregistration: data/rank red team of the state router

The first router trained on 16 FineWeb sites reached high median fidelity but
10th percentiles of 0.648 on held-out FineWeb and 0.735 on WikiText. Before
rejecting a low-rank state address, use both FineWeb sites for training and
select state rank and ridge strength with query-grouped four-fold validation.

Ranks are frozen to 4, 8, 12, and 15; ridge strengths to 0.001, 0.1, 10, and
1000. Every fold holds out both sites for four tensor queries. PCA is fitted
inside each training fold. Choose the configuration by validation 10th-
percentile finite fidelity, then median fidelity, then lower rank and stronger
regularization. Refit on all 32 FineWeb sites and evaluate WikiText once.

Data limitation is supported only if WikiText 10th-percentile fidelity improves
by at least 0.10 and crosses 0.80. Validation requires both grouped FineWeb CV
and WikiText to reach median 0.95 and 10th percentile 0.80. Otherwise the tested
linear state address is classified as plateaued, not all routers falsified.
