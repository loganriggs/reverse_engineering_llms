# Longer real-text prediction sites and their shortcuts

The [random-word pilot](FINEWEB_RANDOM_PREDICTION_SITE_AUDIT.md) mostly
selected local probability shifts in short prefixes. To inspect what
longer document context can supply, [this fetcher](fetch_fineweb_long_panel.py)
streamed a **disjoint 48-document** FineWeb sample-10BT development panel
at offsets 25000–25047 and saved only a [hash/length manifest](results/fineweb_long_panel_25000_n48_manifest.json)
in the repository. All 48 URLs and hosts are distinct. Sixteen documents
have at least 2,500 characters; seed 25017 selected ten of them *before*
target inspection. One reader then chose one interesting target span per
document. The [target ledger](results/fineweb_long_targets_25000_seed25017_n10.json)
records exact source hashes and offsets, paraphrased earlier evidence,
alternatives, and the most salient shortcut, without source passages.
Target choice **within** each document is purposive, so these ten are not
a prevalence sample.

The passages contain several different conditional structures. A legal
profile page introduces several attorneys in separate bios; a later first
name can use the current bio scope. A sports article reintroduces one
prospect after discussing others; the local phrase alone does not name
him. A long numerical table interrupts a story before a player is quoted.
A numbered interview moves from question 2 to question 3. A technical
lecture page starts a biography of the person named in its header. These
are *candidate* entity, source, and document-schema tasks. They also have
ordinary shortcuts: repeated names, article topic prominence, and local
section headings. One apparent later donor name introduces a new donor
fact, so it should not be treated as an exact binding target.

The [scorer](score_fineweb_random_sites.py) measured observed-target log
probability with the full prefix versus only its last 40 whitespace words,
using [GPT-2](results/fineweb_long_targets_gpt2_seed25017_n10.json) and
the [18-layer bilinear model](results/fineweb_long_targets_bilin18_seed25017_n10.json).
Unlike the random short-word panel, **8/10 GPT-2** and **6/10 bilinear**
targets gain over 1 nat from the full prefix. Median gains are 5.59 and
3.40 nats. This is a target-selected set, so the difference between panels
shows selection and context-length effects, not that 80% of web tokens
need long context.

| Target context | GPT-2 full minus last 40, nats | Bilinear, nats | What the score does and does not show |
|---|---:|---:|---|
| Later prospect name (`Bryant`) after several other players | +6.57 | −16.05 | GPT-2's equal-token-count margin over `Rogers` changes from −2.12 local to +4.21 full. Bilinear favors the wrong candidate with the full prefix (margin −4.05). A model-sensitive identity test, not proof of an entity circuit. |
| Interview heading `3.` after questions 1 and 2 | +11.66 | +5.56 | Both models gain from numbering. Equal-token-count margin over `4.` rises from +0.32 to +4.29 for GPT-2 and +0.24 to +6.67 for bilinear. This is a strong visible page-schema target. |
| Lecture speaker's name at start of biography | +21.58 | +19.96 | Header-to-section repetition is available; this may be long-range copying, not biographical reasoning. The alternative has a different token count. |
| Organization name later on a services page | +15.53 | +11.10 | Repeated organization identity is available, including nearby mentions; exact role binding is not isolated. |
| Former leader name in a lore page | +12.07 | +7.57 | The full prefix helps the observed name, but for GPT-2 the equal-token-count margin over the current leader is *smaller* than with a short prefix. More target probability alone need not mean better role discrimination. |

Two models, selected passages, and prefix truncation do not identify the
writer's rule. Some alternatives have different tokenizer lengths, so
their raw sequence log-probability margins are not directly comparable;
the table gives candidate margins only when token counts match. The
bilinear model's severe negative long-prefix result at the prospect-name
site is a genuine model failure on this diagnostic, not evidence that the
natural article lacks a long-range identity relation. No counterfactual
edit of real text has yet been validated for grammaticality. A second
reader and a longer untouched panel are still needed.

For the next **joint sampled DGP**, this evidence argues for a document
plan with stable topic/entity identities, several field/section forms,
and interruptions between introduction and reuse. A name's *first*
appearance should remain an exogenous sampled fact; subsequent mentions
can test identity. The page-schema continuation should have a separate
oracle from the content of the next question. Include both repeated-name
shortcuts and contrastive candidate entities, and test whether models
answer after distractors or when a nearby copy is absent. These
structures must coexist with the original local bigram, skip, induction,
and bracket rules in sampled documents, not be appended as a separate
answer-only benchmark. Keep a reference-rate sample and a task-rich
sample, then use whole-combination and document holdouts. The existing
[V6 result](DOCUMENT_MIXTURE_V6_TRANSFER.md) establishes that aggregate
token loss is not a substitute for those task and causal tests.
