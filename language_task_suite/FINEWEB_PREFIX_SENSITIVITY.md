# Natural-text prediction-site sensitivity pilot

This is a targeted check of whether earlier text affects a model's score at
the [eight hand-inspected FineWeb sites](results/fineweb_prediction_sites_skip23000_n8.json).
The sites were chosen for diversity, **not** sampled to estimate corpus
frequencies. The [scorer](score_fineweb_prediction_sites.py) evaluates the
observed continuation under three contexts: the full prefix within the
first 1,200 characters, its last 100 whitespace-delimited words, and its
last 40 words. Where possible, a specific earlier phrase is edited without
changing the target. JSON results are available for
[GPT-2](results/fineweb_prediction_sites_gpt2_scores_n8.json) and the
[18-layer bilinear model](results/fineweb_prediction_sites_bilin18_scores_n8.json).
Only source hashes, offsets, targets, and scores are committed; the local
source preview is not.

Numbers below are **log-probability differences in nats** for the observed
target. Positive `full − local` means the full prefix raises its score;
negative `edit − full` means the evidence edit lowers it. The edits are
specific to each example and vary in how natural they remain.

| Target site | GPT-2 full − 40 words | Bilinear full − 40 words | GPT-2 edit − full | Bilinear edit − full | Interpretation |
|---|---:|---:|---:|---:|---|
| Witness pronoun, 23006 | −0.53 | −0.61 | −4.56 | −4.08 | Changing the nearby female description to male strongly changes the score; long context is unnecessary here. |
| Repeated floor, 23014 | +0.14 | +0.47 | −0.36 | −0.29 | A stated goal and prior floor mentions leak the answer; this does not isolate procedure tracking. |
| Second record's category, 23019 | +6.10 | +5.74 | −3.53 | −1.65 | The first record's `11B` appears to prime the later `11`; this is **not** evidence of scope-sensitive retrieval. |
| Later sports score, 23033 | +1.40 | +0.70 | −1.35 | −0.57 | Altering the headline score hurts the later observed score despite the body arithmetic; a clear copying confound. |
| Repeated surname, 23037 | +11.14 | +9.42 | −10.91 | −12.37 | Both models are strongly sensitive to earlier name mentions. This is the clearest long-context identity site in the pilot. |
| Daylight-saving direction, 23038 | −1.44 | −0.63 | −0.26 | −0.07 | Both models favor the alternative `later` to the observed `earlier`; no evidence here of successful contrastive-schema reasoning. |
| First case dimension, 23043 | −0.18 | −0.19 | +0.02 | +0.02 | Product format may predict a unit, but the exact dimension behaves like a new fact. |
| First animal name, 23062 | −0.05 | −0.91 | — | — | The first proper name is new information; test later references instead. |

The surname result suggests an interpretable task target: introduce a person
by full name, insert competing speakers and unrelated facts, then require a
later attribution. The key control is to swap that person's identity while
keeping the local quote frame fixed. For records, train and evaluate both
cross-record **priming** and genuine **scoped lookup**: a later query should
specify one of two records that share a key but have conflicting values.
For arithmetic and procedures, include paired leaked/unleaked documents so
a model must choose between copying an explicit earlier answer and
computing from events. Exogenous first mentions should contribute to
ordinary language-model loss but should not be counted as a recoverable
semantic task.

This pilot cannot establish the true causal process of the source document.
Truncating context changes its beginning; phrase edits may create
distribution shift; the pretrained models could memorize source text; and
one selected target per family is far too small to rank task prevalence or
competence. The next natural audit should sample sites independently within
held-out documents and annotate minimal evidence, leakage, ambiguity, and
possible edits before using these patterns as calibration targets.
