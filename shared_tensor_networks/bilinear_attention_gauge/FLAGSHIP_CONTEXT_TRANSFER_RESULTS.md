# Context length, token frequency, and sequence transfer for 546M attention cuts

The [full-width first-layer pilot](FULL_WIDTH_FLAGSHIP_RESULTS.md) reduced the physical width-128 QK1 bond of head 0 using uniform-token or FineWeb-weighted four-position pairs. This follow-up tests whether those cuts generalize when attention has many more causal key positions, and refits the same bond on length-128 windows. No eight-token domain quotient is used.

`flagship_transfer.py` evaluates saved cuts on 32,768 fresh pairs per condition and seed, with seeds 23 and 29. FineWeb fit documents are 0–14,999; evaluation documents are 15,000–19,999. Long-window queries and keys are drawn from the same 128-token window at a uniformly chosen valid causal position pair. Additional uniform draws restrict both token IDs to pools defined by their **frequency in the fit documents**: 1–10 occurrences (rare), at least 1,000 (common), or zero (unseen). These are diagnostics, not new fitting data. `flagship_ood_corpora.py` uses separate tokenized WikiText-103 and TinyStories samples. Pair errors weight each QK1 error by the squared QK2 score and OV output norm. A second error replaces QK2 by a constant score, probing an internal branch intervention.

## The hidden position prior in a four-token measure

Sampling uniformly from valid causal position pairs gives self-pairs \(i=j\) probability \(2/(L+1)\). But self-pairs have much larger head-response energy than typical cross-pairs. Across two independent sampling seeds, the estimated self-pair **fraction of original pair-response energy** was:

| Measure | Length 4 | Length 128 |
|---|---:|---:|
| Uniform tokens | **89.0%** | 25.1% |
| FineWeb windows | 46.0% | 11.5% |

For uniform tokens at length four, self-pairs are only 40% of sampled positions yet dominate the squared-error objective. A short-window fit can therefore look excellent while preserving cross-token attention poorly. Uniform **token** weighting is not a complete weight-only metric until the position and sequence-length measure is also declared.

The rank-16 uniform cut directly fitted to length-four pairs has relative pair error **.163** on fresh uniform length-four pairs, but **.720** on length-four cross-pairs alone and **.748** on length-128 uniform pairs. The FineWeb-fitted length-four direct cut scores **.096** on fresh FineWeb length-four pairs and **.706** on FineWeb length-128 pairs; it exceeds 1 on uniform long-context, rare, and unseen-token pairs. The error ratio can exceed 1 because the reduced response can be worse than deleting the original response under a shifted measure. Neither short-window fit is a stable general-purpose reduction.

## Refit on length-128 windows

WTG truncation and direct rank-16 optimization were repeated with 65,536 fit pairs and 65,536 independent validation pairs for each length-128 measure. Direct fitting again lowers its sampled training error sharply but has substantial validation gaps. On uniform length-128 pairs, direct fit has about **.325 train / .645 validation** error, versus **.498 / .518** for WTG. On FineWeb length-128 pairs, direct fit has about **.083 / .420–.426**, versus **.343 / .449** for WTG. Both direct fits hit their 300-iteration cap, so their train values are not optima. Their validation and intervention performance must decide whether the extra fitting helps.

For WTG, a larger rank changes the picture. Relative squared **pair-function error** on fresh length-128 samples, averaged over the two evaluation seeds:

| Fit measure / evaluation | Rank 16 | Rank 32 | Rank 64 | Rank 96 |
|---|---:|---:|---:|---:|
| Uniform / uniform tokens | .504 | .317 | .160 | .059 |
| FineWeb / FineWeb windows | .404 | .242 | **.097** | .030 |
| Uniform / rare tokens | .509 | .325 | .168 | .062 |
| FineWeb / rare tokens | .588 | .394 | .215 | .078 |
| Uniform / unseen tokens | .566 | .372 | .199 | .078 |
| FineWeb / unseen tokens | .626 | .441 | .242 | .095 |

The FineWeb measure helps common and ordinary FineWeb pairs but gives up rare/unseen-token fidelity at matched rank. At rank 64, FineWeb-fitted WTG scores **.103** on common tokens, **.215** on rare tokens, and **.242** on unseen tokens; uniform-fitted WTG scores **.157**, **.168**, and **.199**. This is the actual tradeoff the measure encodes. The QK2-constant intervention also remains worse than the fitted natural measure: FineWeb rank-64 WTG scores **.097** on natural FineWeb long pairs but **.192** after QK2 is replaced by a constant. Low ordinary error is not broad causal fidelity.

## Aggregate the head over actual token sequences

`flagship_sequence_transfer.py` sums each causal pair through the original OV projection, producing the full **layer-0 head contribution** at every query position for each sampled sequence. Its identity-bond output matches an independent selected-token calculation to maximum absolute error `8.9e-16` and a direct checkpoint PyTorch calculation at length 128 to `1.5e-14`. The assay uses 256 documents per seed, two seeds, for each length and each of held-out FineWeb, WikiText-103, and TinyStories: **3,072 sequences** in all. It reports both all-position and final-position head-output error. It does **not** propagate the edited head through layers 1–17 or measure full-model KL/CE.

Relative squared **sequence-aggregated head-output error** for length-128 windows:

| WTG fit / rank | Held-out FineWeb | WikiText-103 | TinyStories |
|---|---:|---:|---:|
| Uniform / 16 | .433 | .365 | .475 |
| Uniform / 32 | .211 | .192 | .237 |
| Uniform / 64 | **.075** | **.070** | **.082** |
| Uniform / 96 | .027 | .027 | .028 |
| FineWeb / 16 | .404 | .392 | .450 |
| FineWeb / 32 | .219 | .218 | .246 |
| FineWeb / 64 | .086 | .094 | .095 |
| FineWeb / 96 | .025 | .026 | .023 |

Rank 16 is too lossy for a faithful long-context head explanation. Rank 64, a 50% reduction of this **one internal bond**, retains about 7–10% head-output error across these corpora; rank 96 retains about 2–3%. These are not full-model error guarantees. The value of a canonical gauge here is a reproducible, representation-specific ordering of bond directions; the useful rank depends on the downstream tolerance and the target input/intervention measure. A semantic label cannot be justified from this compression curve.

## Decision and next test

The short-window metric was the major failure: it overweights self-pairs and hides long-context cross-token error. Long-context WTG is more stable than unconstrained direct fitting at this sample budget, but a pure FineWeb measure gives up rare-token and QK2-intervention fidelity. The predeclared [mixed-metric follow-up](MIXED_METRIC_RESULTS.md) has now tested uniform–corpus and uniform–corpus–intervention fits at ranks 32 and 64. The half uniform–corpus cut improves natural sequence head-output fidelity on three held-out corpora, but an extra pair-level QK2-constant term does not improve sequence-level intervention fidelity. Full-model KL/CE and a component-level semantic or causal assay remain required before calling any direction an interpretable mechanism.

The JSON/NPZ files under `full_width_results/` contain the fit curves, pair-domain results, sequence aggregates, and saved bond matrices. CPU numeric runs were pinned to core 2 on this host. The external model and corpora are referenced by path rather than copied into this repository.
