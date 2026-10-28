# Locked pilot: semantic organization versus factual constraint networks

Date locked: 2026-10-17, before computing any margin-gradient kernels or
finite parameter-edit outcomes.

## Question

Do the trained factual-transformer checkpoints admit a causal
constraint-network explanation in addition to the already-computed neuron
activation explanations?  In particular, does the local signed overlap of two
facts' margin gradients predict the collateral effect of an actual parameter
edit, and does arbitrary factual load produce nontrivial coupling that is not
just shared answer/query/input-token structure?

This pilot does not claim that a gradient kernel is a complete explanation of
memorization, and it does not call all distributed coupling superposition.

## Frozen substrate

Use the existing `factual_capacity_calibration_r2` manifest and the width 128
and 256 checkpoints at rho 0, 0.5, and 1.  No retraining or checkpoint
selection is allowed.  A fact enters the panel only if both widths predict it
correctly on templates 0 and 6.  Selected facts whose random target happens to
equal the structured target are excluded.  Use a deterministic, approximately
balanced panel of 32 facts per rho; at rho 0.5, split the panel equally between
changed/randomized and unselected/structured facts.

For each fact, freeze its baseline strongest incorrect competitor and define
the pairwise margin as correct logit minus that competitor logit.  Competitor
switching after finite edits is reported separately.

## Parameter groups and kernel

Audit input embeddings plus positional embeddings, each of the four complete
transformer blocks, and final normalization plus answer head.  For group l,

    K_l(i,j) = <grad_l m_i, grad_l m_j>

and its cosine-normalized version are computed exactly in float32.  Positive
and negative overlaps are retained separately.

## Primary causal validation

For eight deterministic source facts per condition and every parameter group,
apply a normalized gradient edit whose linearized self-margin increase is
0.25, 1.0, or 4.0 logits.  Predict every panel fact's fixed-competitor margin
change using K, then recompute the full network and measure the actual change.
Self-effects are excluded from prediction scores.

The constraint explanation is **validated locally** if, pooled over the four
transformer blocks in both high-fidelity rho=1 models, the 0.25-logit edits
have Pearson r >= 0.8 and origin-anchored R-squared >= 0.5, and both exceed a
deterministic source-column permutation control by at least 0.3.  It is
**finite-edit validated** if the same thresholds hold at 1.0 logits.  It is
falsified at the tested scale if Pearson r < 0.3 or it fails to beat the
permutation control by 0.1.  Intermediate outcomes are inconclusive and
require changing the edit family or sampling more facts.

## Web and coexistence tests

Network breadth is descriptive, not sufficient evidence.  Report off-diagonal
absolute cosine, positive/negative edge fractions, effective support, and
finite collateral.  Fit a locked linear control to cosine overlap using same
attribute, answer, competitor, randomized status, number of matching traits,
and number of matching ID digits.  Residual coupling is what remains after
these obvious relationships.

The stronger load-induced-web hypothesis is supported only if residual
absolute coupling or validated finite collateral rises from rho=0 to rho=1 in
both widths.  A mixed rho=0.5 model counts as coexistence evidence only if its
structured subset retains the previously locked semantic explanation signal
and its changed subset participates in a causally validated constraint
network.  Otherwise report the semantic and constraint findings separately.

All thresholds, failed cells, signs, parameter groups, and edit scales are
reported.  No post-hoc choice of layers or sources may replace this primary
analysis.
