# Which metric preserves later attention-head behavior?

## Setup

The [layer/head scan](LATER_LAYER_ATTENTION_LOCUS_RESULTS.md) found four
later-layer heads whose final-position outputs materially change competent
answers. We made the first **matched-rank reduction** comparison on those
128-dimensional head vectors. This is stage 1 of the canonicalization plan:
choose what a reduction must preserve. It is not a sparsifying rotation or a
canonical decomposition.

Each candidate is an orthogonal, zero-intercept rank-`r` projection of one
head's pre-output-projection vector at the final prompt position, with
`r=4,8,16`. The bases are:

* **Weight only:** top right singular vectors of that head's native output
  projection. Requires no examples.
* **Corpus:** top uncentered principal components of head vectors on 256
  FineWeb training contexts of length 32.
* **Task delta:** top principal components of edited-minus-base head vectors
  from 23 discovery prompt pairs, pooled across four task families.
* **Mixed:** equal-trace sum of corpus and task-delta second moments.
* **Random:** seeded orthogonal control at the same rank.

We evaluated on 21 new, unfiltered, single-token task pairs (new lexical
items and arithmetic expressions), plus 24 held-out length-32 contexts each
from disjoint FineWeb documents, WikiText-103, and TinyStories. All methods
use the same rank and projection family. The natural-corpus fitting length
differs from the short task prompts, so that comparison includes a real
distribution shift. Only one head and final position are reduced at a time;
the rest of the model remains native.

## The decisive metric: finite head-path fidelity

For each held-out pair, we first measured the native effect of transplanting
the edited-minus-base *complete head vector* into the base run. Under each
rank cut, we projected that same vector difference, transplanted it, and
compared the finite margin effect with native. The identity-rank reproduction
gate passed with maximum absolute error `1.05e-5` margin units. Numbers below
are mean absolute **path-effect error divided by mean absolute native
path-effect magnitude**. Lower is better.

| Head / task | Rank | Weight only | Corpus | Task delta | Mixed | Random |
|---|---:|---:|---:|---:|---:|---:|
| L13 H8 / bracket | 4 | **.013** | .833 | .027 | .014 | .941 |
| L13 H8 / bracket | 16 | .013 | .785 | **.007** | .015 | .833 |
| L11 H3 / agreement | 4 | .063 | .733 | **.037** | .052 | .965 |
| L11 H3 / agreement | 16 | .052 | .612 | **.037** | .040 | .882 |
| L13 H0 / capital | 4 | .937 | .958 | **.884** | .884 | .954 |
| L13 H0 / capital | 16 | **.602** | .844 | .785 | .777 | .862 |
| L8 H7 / antonym | 4 | .836 | .928 | .727 | **.717** | .925 |
| L8 H7 / antonym | 16 | **.333** | .787 | .632 | .624 | .852 |

Bracket and agreement have very low-rank head-response paths under weight
geometry or edit supervision. For bracket at rank 4, the weight basis captures
80% of the raw edit-vector energy and has .060 relative output-projection
MSE; corpus PCA captures only 7% and has .850 output-projection MSE. For
agreement the analogous raw captures are 68% versus 20%. This explains
much of the finite-path gap without assuming that a principal component is
human-readable. Capital and antonym path effects are substantially more
distributed; none of the rank-16 cuts is faithful by the same standard.

## Behavior fidelity can hide path loss

The model can retain the correct forced-choice answer while the reduced
head no longer reproduces its original contribution. At rank 4, the
weight-only capital cut keeps **5/5** held-out pairs correct but has **94%**
relative error on the native single-head path effect. The corpus cut
also keeps 5/5 correct with 96% relative path error. For bracket, all five methods keep
6/6 decisions correct, although corpus and random cuts miss most of that
head's finite path. Thus task accuracy and even total input-edit fidelity
cannot serve as a proxy for recovering a specific attention computation.

Natural-text full-vocabulary KL is small for this one-head intervention,
typically around `5e-4` to `6e-3` on the 24-context corpus panels. These
small panels do not resolve which low-rank basis is best for broad language
modeling. The task-pair results also cover only 4–6 cases per family.
Neither result warrants a general claim that weight-only reduction beats
data-aware reduction. They do show why both must be tested, and why the
causal path metric belongs beside KL and accuracy.

## Implication for the next legibility stage

The bracket and agreement heads provide unusually clean, low-rank native
paths for comparing **legibility priors at a fixed reduced function**. An
exact change of coordinates *within* a selected rank-4 subspace can test
weight sparsity, response sparsity across token edits, intervention-effect
sparsity, and reusable groups without changing the rank-4 projection.
Before naming a feature, use fresh prompt formats and whole-task holdouts,
intervene on individual coordinates in the native model, and check whether
the effect predicts unseen edits beyond matched random rotations. Capital
and antonym need a higher-rank or different reduction objective before
similar claims; their low-rank projections fail the path-fidelity gate.

Reproduce the basis/rank sweep with `flagship_later_head_rank_pilot.py`,
and the finite-path test with `flagship_later_head_path_fidelity.py`.
Numerical records and basis matrices are in
`full_width_results/later_head_rank_pilot.json`,
`later_head_rank_pilot_bases.npz`, and
`later_head_path_fidelity.json`.
