# Preregistration: downstream-sensitivity fingerprint router

Raw Layer-8 state routers repeatedly choose the common early-layer triple and
miss contexts needing a different suffix route. This experiment uses features
from the tensor-network derivative itself.

At each site, compute the softmax-Fisher response over all 512 Layer-8 modes for
the direct/no-attention route and for each of the nine singleton attention-JVP
routes. Let `D_l = R_{singleton l} - R_direct`. The site fingerprint contains:

- `log ||D_l||_F` for nine layers;
- all 36 signed cosine similarities between `D_l` and `D_m`.

For each candidate triple, concatenate its static inclusion/pair bits with
those bits multiplied by the corresponding site fingerprint. A ridge router is
trained on all FineWeb sites, with strength selected by query-grouped four-fold
validation. It is then frozen for WikiText.

The feature extractor does **not** compute the full nine-layer response, inspect
the candidate triple's masked response, or use any finite intervention. Those
quantities occur only in the frozen atlas labels used for training/evaluation.

Validation requires median >=0.95 and 10th percentile >=0.80 on both grouped
FineWeb and WikiText. A seeded permutation of WikiText fingerprints across
sites must reduce the 10th percentile by 0.10 to establish that the router uses
the new features rather than only global mask biases. Failure rejects this
singleton/pairwise downstream fingerprint, not richer path features.
