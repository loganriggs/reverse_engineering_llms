# Dynamic read-site selector

This follow-up asks whether the early joint-field gap comes from context-specific addressing. At each boundary, candidate-independent suffix-Fisher demand selects 4, 8, 16, or 32 token sites. Each selected invocation receives a local residual basis so every chart still uses exactly 128 runtime coordinates. The candidate replacement tangents are unseen during site and basis construction.

## Locked hypotheses

- **S1_dynamic_sites_beat_fixed_reads: not supported.**
- **S2_dynamic_sites_approach_joint: supported.**
- **S3_dynamic_reads_are_nonlocal: supported.**
- **S4_selected_sites_are_sampling_stable: supported.**
- **S5_joint_still_has_nonseparable_advantage: supported.**
- **attention_jvp_forward_invariant: supported.**

## Causal results

| Target | Boundary | Joint | Fixed best | Dynamic top4 | top8 | top16 | top32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| L0H7 | 6 | 0.883 | 0.691 | 0.618 | 0.698 | 0.753 | 0.711 |
| L0H7 | 10 | 0.810 | 0.749 | 0.757 | 0.709 | 0.675 | 0.611 |
| L0H7 | 14 | 0.724 | 0.725 | 0.732 | 0.609 | 0.548 | 0.488 |
| L0H7 | 17 | 0.690 | 0.692 | 0.699 | 0.584 | 0.518 | 0.475 |
| L4H2 | 6 | 0.817 | 0.534 | 0.501 | 0.587 | 0.586 | 0.559 |
| L4H2 | 10 | 0.749 | 0.648 | 0.654 | 0.611 | 0.547 | 0.491 |
| L4H2 | 14 | 0.618 | 0.602 | 0.604 | 0.484 | 0.426 | 0.355 |
| L4H2 | 17 | 0.575 | 0.572 | 0.569 | 0.451 | 0.373 | 0.318 |
| L5H1 | 6 | 0.822 | 0.541 | 0.446 | 0.542 | 0.585 | 0.589 |
| L5H1 | 10 | 0.749 | 0.667 | 0.659 | 0.613 | 0.559 | 0.494 |
| L5H1 | 14 | 0.676 | 0.674 | 0.671 | 0.557 | 0.491 | 0.405 |
| L5H1 | 17 | 0.643 | 0.647 | 0.643 | 0.529 | 0.457 | 0.373 |

## Selector diagnostics

| Boundary | Median top-8 split Jaccard |
|---:|---:|
| 6 | 1.000 |
| 10 | 1.000 |
| 14 | 0.778 |
| 17 | 1.000 |

Across top-8 selections, 25.8% of sites are more than seven tokens before the next scored position.

## Concrete selected reads

- Context row 0:
  - boundary 6: p63 lag0 ` Bel`, p95 lag0 ` other`, p127 lag0 ` notably`, p31 lag0 `uced`, p94 lag1 ` and`, p62 lag1 `;`, p30 lag1 ` ded`, p88 lag7 ` of`
  - boundary 10: p63 lag0 ` Bel`, p95 lag0 ` other`, p127 lag0 ` notably`, p31 lag0 `uced`, p90 lag5 ` hill`, p37 lag26 ` Ill`, p94 lag1 ` and`, p58 lag5 `ovic`
  - boundary 14: p63 lag0 ` Bel`, p95 lag0 ` other`, p127 lag0 ` notably`, p31 lag0 `uced`, p66 lag29 ` Bel`, p90 lag5 ` hill`, p53 lag10 ` cp`, p61 lag2 ` b`
  - boundary 17: p63 lag0 ` Bel`, p95 lag0 ` other`, p127 lag0 ` notably`, p31 lag0 `uced`, p3 lag28 ` in`, p2 lag29 ` preserved`, p0 lag31 `
`, p1 lag30 `As`
- Context row 1:
  - boundary 6: p95 lag0 ` traditionally`, p31 lag0 ` the`, p127 lag0 ` around`, p63 lag0 ` fact`, p94 lag1 ` is`, p10 lag21 ` Czech`, p30 lag1 ` in`, p29 lag2 ` people`
  - boundary 10: p95 lag0 ` traditionally`, p31 lag0 ` the`, p127 lag0 ` around`, p63 lag0 ` fact`, p10 lag21 ` Czech`, p30 lag1 ` in`, p94 lag1 ` is`, p104 lag23 ` 60`
  - boundary 14: p31 lag0 ` the`, p95 lag0 ` traditionally`, p127 lag0 ` around`, p63 lag0 ` fact`, p10 lag21 ` Czech`, p13 lag18 ` br`, p30 lag1 ` in`, p91 lag4 ` electricity`
  - boundary 17: p95 lag0 ` traditionally`, p31 lag0 ` the`, p127 lag0 ` around`, p63 lag0 ` fact`, p3 lag28 ` is`, p2 lag29 ` weather`, p0 lag31 ` of`, p1 lag30 ` cold`
- Context row 2:
  - boundary 6: p127 lag0 ` a`, p31 lag0 ` face`, p95 lag0 ` such`, p63 lag0 `,`, p126 lag1 `From`, p26 lag5 ` such`, p124 lag3 `
`, p25 lag6 ` with`
  - boundary 10: p127 lag0 ` a`, p31 lag0 ` face`, p95 lag0 ` such`, p63 lag0 `,`, p11 lag20 ` Catherine`, p126 lag1 `From`, p124 lag3 `
`, p125 lag2 `[`
  - boundary 14: p127 lag0 ` a`, p31 lag0 ` face`, p63 lag0 `,`, p95 lag0 ` such`, p11 lag20 ` Catherine`, p28 lag3 ` beautiful`, p125 lag2 `[`, p16 lag15 ` se`
  - boundary 17: p127 lag0 ` a`, p63 lag0 `,`, p31 lag0 ` face`, p95 lag0 ` such`, p3 lag28 ` at`, p2 lag29 ` present`, p0 lag31 ` to`, p1 lag30 ` be`
- Context row 3:
  - boundary 6: p63 lag0 `,`, p95 lag0 ` fee`, p31 lag0 ` breaking`, p127 lag0 ` auctions`, p121 lag6 ` how`, p123 lag4 ` participate`, p29 lag2 ` for`, p94 lag1 ` admission`
  - boundary 10: p63 lag0 `,`, p127 lag0 ` auctions`, p95 lag0 ` fee`, p31 lag0 ` breaking`, p54 lag9 ` Christie`, p29 lag2 ` for`, p30 lag1 ` record`, p41 lag22 ` sal`
  - boundary 14: p63 lag0 `,`, p127 lag0 ` auctions`, p95 lag0 ` fee`, p31 lag0 ` breaking`, p54 lag9 ` Christie`, p40 lag23 ` our`, p41 lag22 ` sal`, p21 lag10 ` we`
  - boundary 17: p63 lag0 `,`, p127 lag0 ` auctions`, p95 lag0 ` fee`, p31 lag0 ` breaking`, p3 lag28 ` areas`, p2 lag29 ` all`, p0 lag31 `,`, p1 lag30 ` including`

## Guardrails

Fisher demand is an output-sensitive dynamic address generator, not yet a human-readable function. Selected token examples are descriptive and were not used to choose the locked thresholds. A win would localize the missing object to context-dependent addressing; it would not explain why those tokens were selected. A loss would support genuinely joint position-by-residual structure or a selector that cannot be reduced to hard sites.
