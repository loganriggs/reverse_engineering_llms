# Paired event-frame bilinear language-model pilot

This is a controlled follow-up to [EVENT_FRAME_ORACLE_PILOT.md](EVENT_FRAME_ORACLE_PILOT.md), **not** a claim that the event-frame generator resembles natural language. The generator samples eight semantic events and a latent genre per document. A fact sets an object's attribute; a later action word depends on the latest visible attribute for that object. Two arms use exactly the same sampled latent events and genres: fixed wording has one fact and one action frame, while variable wording samples three fact and three action frames. The visible-prefix next-token oracle marginalizes the latent genre and competing surface continuations.

## Training and evaluation

The same two-layer, width-64, four-head bilinear Transformer architecture (155,136 parameters, RMS normalization) was trained separately on each arm. Each arm has 20,000 training documents and an independent 1,000-document test stream; the arms' latent train and test events are paired. AdamW at 0.0005, batch size 32, 6,000 steps, two initialization seeds. Loss is ordinary cross-entropy on **every available next-token prediction** after the first token of each document, with no action-site reweighting. Both runs use identical initialization seeds and training hyperparameters. The variable documents are longer, so the matched document/update budget is not a matched token budget.

Each test document contributes all of its action sites (3,134 total). The principal measure is mean KL divergence from the exact next-token oracle at those sites; this avoids labeling genuinely ambiguous prefixes as deterministic. The fixed arm has zero ambiguous action sites and the variable arm has 952. A separate supported edit set changes one earlier fact attribute and scores the probability change on the resulting action; it includes 48 fixed-frame and 432 variable-frame pairs. These short two-sentence edits are narrower than the full eight-sentence test stream.

| Seed | Arm | Full-token CE | Action-oracle KL (nats) | Edit directions correct | Mean edit response / oracle | Edit-effect MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 42811 | Fixed | 0.728 | 0.271 | 48/48 | 0.996 / 1.000 | 0.0037 |
| 42811 | Variable | 0.753 | 0.471 | 432/432 | 0.947 / 0.967 | 0.0289 |
| 42812 | Fixed | 0.719 | 0.122 | 48/48 | 0.999 / 1.000 | 0.0011 |
| 42812 | Variable | 0.758 | 0.537 | 432/432 | 0.960 / 0.967 | 0.0162 |

The 2,000-step first seed had KL 0.662 fixed / 0.701 variable; extending from saved checkpoints to 6,000 substantially improved both. The variable arm remains worse in both seeds, despite seeing more tokens per document. This is consistent with harder inference from variable frames, but two seeds do not isolate frame ambiguity from sequence length, token frequency, or different training exposure. All edit directions are correct, yet the nonzero full-stream KL means the entire conditional distribution is not recovered. A diagnostic by sentence position on seed 42811 found late-site KL larger than early-site KL in both arms (fixed: 0.001 at the second sentence, 0.439 at the eighth; variable: 0.017 and 0.817); these are descriptive, uncorrected post-hoc bins.

The paired generator and exact oracle pass the audit in the preceding report. The model run is reproducible with [`train_event_frame_bilinear.py`](train_event_frame_bilinear.py); the aggregate histories are [`seed 42811`](results/event_frame_paired_depth2_20k_6k_seed42811.json) and [`seed 42812`](results/event_frame_paired_depth2_20k_6k_seed42812.json). The cached samples and resumable model/optimizer states are kept locally in `/home/loganriggs/Coding/neuron_circuits/artifacts/event_frame_pilot/`, outside Git. The script also saves full-token CE, ambiguous-site KL, and the complete evaluation trajectory. No tensor canonicalization or sparse-weight interpretability comparison has been run on these checkpoints.

## Decision

The variable-frame oracle and its behavioral gate are useful controls for future weight-method comparisons. They do not repair the main DGP: on matched first-40-word windows the prior audit found 55 word types and 55.8% top-ten word share in the variable fixture versus 1,583 types and 19.4% in fresh FineWeb. The next generator experiment should embed the same marginal-oracle/event-state machinery in a broader document planner with more varied discourse acts and a longer typed lexical tail, then repeat independent form, task-density, behavioral, and held-out-combination checks before interpreting weights. The current two models are conditional pilot controls, not realistic-language interpretability targets.
