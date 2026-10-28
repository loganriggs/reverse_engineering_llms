# EMLP-P2: amplification transfers to location nouns and future predictions

Exploratory native-model diagnostic, 2026-10-28. Extends EMLP-P1's single final token ` the`.

## Fixed design

Cross four authored context pairs with four token identities: ` house`, ` park`, ` school`,
` station`. Each sentence ends with the shared continuation `. It was quiet outside.`.
At the noun position, add the donor-minus-recipient MLP1 write while retaining recipient residual,
embeddings, first-layer value bus and all other writes. Test both directions per pair: 32 context
edits. Each has four fixed norm-matched signed permutations: 128 random-control edits.
No fits, task labels, amplitude search, or selection based on these results.

The MLP2 response has two intervention switches: at the edited position and at later positions.
“Frozen” means the unedited recipient's MLP2 output write, including bias. All unedited arms
therefore share exactly the intact baseline. Attention2 and all subsequent layers remain live.

- **Frozen all:** neither local nor future MLP2 write can respond.
- **Local only:** only MLP2's write at the edited position responds.
- **Future only:** freeze that position's MLP2 write; allow later MLP2 writes to respond.
- **Live:** allow all MLP2 writes to respond.

This is a factorial of physical writes, not a claim that the two response sources are semantic units.
Measure KL(intact recipient || edited output) at the edited token, averaged over future positions,
and at the final position. These are distribution changes, not errors against a behavioral target.

## Results

Mean output KL for context-derived changes:

| MLP2 response | Edited token | Future-position mean | Final position |
|---|---:|---:|---:|
| Frozen all | .00005538 | .00000820 | .00000559 |
| Local only | .00035025 | .00004225 | .00002701 |
| Future only | .00005538 | .00001066 | .00000699 |
| Live | .00035025 | .00004554 | .00002838 |

Live/frozen-all ratios are **6.325 at the edit, 5.556 over future positions, and 5.073 at the end**.
All 32 context edits increase edited-token and future-mean KL when MLP2 responds; 31/32 increase
final-position KL. Local response alone gives most of the observed increase. This comparison is
not an additive allocation of KL shares: KL is nonlinear.

Random signed-permutation controls show the same qualitative result: live/frozen-all ratios
are **4.790 / 4.797 / 4.623** respectively. Future-mean KL increases for all128 controls.
The higher aggregate context ratio is not a validated semantic-selectivity effect.

By anchor, context-edit live/frozen ratios are:

| Anchor | Edited token | Future mean | Final position |
|---|---:|---:|---:|
| house | 6.471 | 4.798 | 4.987 |
| park | 6.385 | 5.194 | 5.208 |
| school | 3.820 | 4.459 | 3.282 |
| station | 9.886 | 6.898 | 5.971 |

At the edited token, live and local-only are exactly identical; frozen-all and future-only are
exactly identical. Future writes cannot influence earlier outputs. These equalities pass on every row.
All pre-edit logits are also unchanged exactly.

In centered-logit space, the interaction
`live - local_only - future_only + frozen_all` has pooled future RMS only **0.346%** of the
live-minus-frozen response RMS for context edits (**0.300%** for controls). This is local near-additivity
on these edit amplitudes and prompts, not global linearity or a general composition theorem.

## Interpretation and next discriminating test

P1's amplification is not confined to the token ` the` or to the output at the edit site.
It reaches later predictions, predominantly through the changed local MLP2 write and its later
propagation. The same pattern for random directions argues against describing this as a special
semantic repair or as proof of a discovered feature.

This sharpens the early-redundancy question: repeated information can be transformed or amplified,
not merely duplicated. But this experiment edits MLP1 and measures MLP2; it does not yet establish
what MLP1 adds beyond MLP0. The next bounded assay should apply this same intervention/control
scheme to **MLP0 → MLP1**, on the same opened panel, rather than adding token examples indefinitely.
Its deliverable is a matched early-layer response comparison, potentially useful as causal supervision
for a featurizer. It is a diagnostic on reused prompts, not a fresh confirmation set.

## Limits and audit

All four anchors are location nouns. Context, token identity and edit norm are not independent random
samples; absolute position was not independently manipulated. Only one shared continuation is tested.
The panel is hand-authored and its native task competence is not measured because no task-correctness
claim is made. Context edits and random controls can be off-manifold. Frozen writes observe native
baseline outputs and are oracle instruments, not learned replacements. Absolute KL changes are small.

704 model forwards; zero identity-replay and pre-edit-logit error. All summary means, per-position
means, finiteness and causal equalities checked. Independent NumPy recomputation of 56 positionwise
KL values from two saved sample arms differs by at most 7.43e-16. No new forwards in that audit.

[Results](validation/mlp2_token_transfer.json) · [Audit](validation/mlp2_token_transfer_audit.json)
· [Run script](probe_mlp2_token_transfer.py) · [Metric audit](audit_token_transfer.py)

Run the probe with `--output validation/mlp2_token_transfer.json`; it also writes the audit logits
to `/tmp/emlp_p2_audit_logits.pt`. Then run `python audit_token_transfer.py`. Those sampled logits
are retained in the main research workspace but omitted from the compact share packet; rerunning
regenerates them. Source and result hashes are recorded. The original corpus experiments were not rerun.
