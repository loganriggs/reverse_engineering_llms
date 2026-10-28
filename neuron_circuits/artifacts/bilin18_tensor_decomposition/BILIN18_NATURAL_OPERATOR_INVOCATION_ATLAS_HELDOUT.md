# Natural FineWeb invocation atlas for typed read operators

Frozen route templates from four controlled behaviors are scanned over unlabeled FineWeb edges. Ranking uses route-template cosine times a weight-aware edge-magnitude proxy. No task labels or answer contrasts enter the search. Every nonidentity permutation of route coefficients across the same three heads is a matched control.

## Locked hypotheses

- **N0_natural_matches_exist: not supported.** Observed: `{'median_cosines': {'agreement_attractor:0': 0.9643684923648834, 'ioi:0': 0.9492648839950562, 'capital_recall:0': 0.8719581961631775, 'repeated_sequence:0': 0.9532268047332764, 'repeated_sequence:1': 0.8722147643566132}}`
- **N1_native_head_typing_beats_permuted_templates: supported.** Observed: `{'families': ['agreement_attractor', 'capital_recall', 'repeated_sequence'], 'ratios': {'agreement_attractor:0': 1.1300758844257808, 'ioi:0': 1.0215106369494251, 'capital_recall:0': 1.3580398126041358, 'repeated_sequence:0': 1.0516918510320883, 'repeated_sequence:1': 1.1062625914896855}}`
- **N2_natural_invocations_are_not_one_token_artifacts: supported.** Observed: `{'distinct': {'agreement_attractor:0': 52, 'ioi:0': 63, 'capital_recall:0': 61, 'repeated_sequence:0': 63, 'repeated_sequence:1': 63}}`
- **N3_repetition_template_enriches_copy_relations: supported.** Observed: `{'repetition_copy_rate': 0.4921875, 'other_copy_rate': 0.020833333333333332}`
- **N4_examples_are_saved_for_semantic_redteam: supported.** Observed: `{}`

| Template | Site/program | Role | Median cosine | Native/permuted score | Distinct sources | Copy-next rate |
|---|---|---:|---:|---:|---:|---:|
| agreement_attractor | L11 H0,H2,H3 | 0 | 0.964 | 1.130 | 52 | 0.000 |
| ioi | L14 H4,H6,H7 | 0 | 0.949 | 1.022 | 63 | 0.016 |
| capital_recall | L13 H0,H5,H6 | 0 | 0.872 | 1.358 | 61 | 0.047 |
| repeated_sequence | L8 H3,H4,H6 | 0 | 0.953 | 1.052 | 63 | 0.500 |
| repeated_sequence | L8 H3,H4,H6 | 1 | 0.872 | 1.106 | 63 | 0.484 |

## Representative unlabeled matches

### agreement_attractor

- role 0, d=3, cos=0.976: ` treatment for years.\nThe feedback we have received since`; source ` feedback` → next ` since`; direct outputs ` was`, ` indicates`, ` is`, ` suggests`.
- role 0, d=4, cos=0.900: `othermy.\nAdvantages of homeothermy\nEn`; source `ages` → next `\n`; direct outputs ` include`, ` were`, ` weren`, ` are`.
- role 0, d=5, cos=0.998: ` employers. Do not let the thought of a post college career plan stress`; source ` thought` → next ` plan`; direct outputs ` appear`, ` make`, ` come`, ` occur`.
- role 0, d=2, cos=0.991: `.);\n- extends the period after sale during`; source ` period` → next ` during`; direct outputs ` to`, ` into`, ` extending`, ` ending`.
- role 0, d=3, cos=0.969: ` to stay solo, as many ladies who time overseas mankind has`; source ` ladies` → next ` mankind`; direct outputs ` have`, ` are`, `have`, ` want`.

### ioi

- role 0, d=1, cos=0.979: ` the pandemic hit, countless museums, national parks`; source ` museums` → next ` national`; direct outputs ` Danielle`, ` Grid`, ` Elise`, ` propagation`.
- role 0, d=1, cos=0.861: ` Windows, Mac OS X, Linux or even`; source ` Linux` → next ` even`; direct outputs ` Valencia`, ` Stuart`, ` Rik`, ` HPV`.
- role 0, d=1, cos=0.976: ` us to improve our products, services, functions and`; source ` services` → next ` functions`; direct outputs ` Haz`, ` reactor`, ` Hamb`, ` Bernie`.
- role 0, d=1, cos=0.965: ` established in 1852 to protect settlers and travelers`; source ` settlers` → next ` travelers`; direct outputs ` Thailand`, ` LSU`, ` torque`, ` Tottenham`.
- role 0, d=1, cos=0.980: ` authorized retailer. The packages, promotions and terms\n`; source ` promotions` → next ` terms`; direct outputs ` Uncharted`, ` Pel`, ` Gil`, ` vinegar`.

### capital_recall

- role 0, d=6, cos=0.888: ` had not fallen apart. And Russia was not Russia; it was USSR and`; source ` Russia` → next ` USSR`; direct outputs ` Russian`, ` Russia`, ` Moscow`, ` Putin`.
- role 0, d=20, cos=0.825: ` when it comes to innovations in espresso machine technology. The newest line from La Marzocco continues their progression in the specific areas of temperature stability`; source ` espresso` → next ` temperature`; direct outputs ` espresso`, ` coffee`, ` Coffee`, ` caffeine`.
- role 0, d=10, cos=0.799: `I’ve been using Kindle from the beginning and would have appreciated access to my subscriptions during`; source ` Kindle` → next ` subscriptions`; direct outputs ` Kindle`, ` Amazon`, `Amazon`, ` ebook`.
- role 0, d=15, cos=0.841: ` year I got absolutely addicted to sewing jeans. I felt like there was something missing anytime I didn’t have any`; source ` sewing` → next ` have`; direct outputs ` sewing`, ` sew`, ` Sew`, ` embro`.
- role 0, d=10, cos=0.906: ` So many people go to a Shiva Temple because He is the sweetest of all. People praise`; source ` Shiva` → next ` People`; direct outputs ` Shiva`, ` Hindu`, ` Indian`, ` Shiv`.

### repeated_sequence

- role 0, d=117, cos=0.966: ` Analysis\nStripovionline.com is created by Dom … There are 8 DNS records about Stripovionline`; source `line` → next `line`; direct outputs `line`, ` line`, `lines`, ` Line`.
- role 0, d=73, cos=0.958: ` for example, if an RWJBH end user needs to get … On the other hand, with Epic, RWJB`; source `JB` → next `JB`; direct outputs `JB`, ` J`, ` b`, ` j`.
- role 0, d=2, cos=0.952: ` part says, ’Namo, namo`; source `amo` → next `amo`; direct outputs `amo`, ` Antonio`, `o`, `am`.
- role 0, d=2, cos=0.957: `. Eat or be eaten. Kill or be killed.`; source ` Kill` → next ` killed`; direct outputs ` Kill`, `Kill`, ` KILL`, ` kill`.
- role 0, d=26, cos=0.980: ` Why are you mute?\njabberwacky: Because NOBODY expects the spanish inquisition.\nuser: Have you seen Monty Python?\njabber`; source `jab` → next `jab`; direct outputs `jab`, ` j`, `j`, ` J`.
- role 1, d=117, cos=0.879: ` Analysis\nStripovionline.com is created by Dom … There are 8 DNS records about Stripovionline`; source `line` → next `line`; direct outputs `line`, ` line`, `lines`, ` Line`.
- role 1, d=73, cos=0.853: ` for example, if an RWJBH end user needs to get … On the other hand, with Epic, RWJB`; source `JB` → next `JB`; direct outputs `JB`, ` J`, ` b`, ` j`.
- role 1, d=2, cos=0.874: ` part says, ’Namo, namo`; source `amo` → next `amo`; direct outputs `amo`, ` Antonio`, `o`, `am`.
- role 1, d=2, cos=0.978: `. Eat or be eaten. Kill or be killed.`; source ` Kill` → next ` killed`; direct outputs ` Kill`, `Kill`, ` KILL`, ` kill`.
- role 1, d=4, cos=0.956: ` & Staff: Calendar\nPublic Interest Law Foundation (PILF`; source ` Interest` → next `IL`; direct outputs ` Interest`, ` interest`, ` interests`, `Interest`.

## Guardrail

A high route-template match establishes recurrent tensor geometry, not the original task semantics. The saved examples are the discovery set for later semantic grouping; any labels inferred from them must be frozen and tested on disjoint corpus rows.
