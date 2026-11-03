# Purpose-driven report DGP V1: rules work, surface rejected

The [corpus construction audit](DISCOURSE_CONSTRUCTION_FORM_AUDIT.md)
showed that changing a few phrases in the old proposition generator did
not remove its language-form gap. This candidate tests a different change:
each document is one repair/reopening event chain rather than an independent
list of arbitrary object facts. It is an experimental DGP, not the accepted
training distribution.

The [generator](project_report_dgp.py) samples a typed project and issue,
two distinct reporters, two completed-work counts, a work target, and one
of four layout labels. Its three scored next-token sites are the sum of the
two counts, the final open/closed status, and the name of the second
reporter. A `coupling_rate` knob makes status either depend on the sum
crossing the visible target or follow a separately stated safety decision.
`surface_entropy` selects fixed or varied clause realizations. These knobs
affect the sampled data law rather than changing model architecture.

A text-only parser, which cannot read the sampled plan, computes all three
answers from the visible prefix. The [audit](results/project_report_v1_audit.json)
ran a four-cell `surface_entropy ∈ {0,1}` × `coupling_rate ∈ {0,1}`
factorial with 1,000 documents per cell. All 12,000 scored prefixes passed.
Across the four cells, 4,000 edits to the second reporter changed only the
source answer; 4,000 first-count edits changed the total; 2,000 threshold
edits flipped coupled status; and 2,000 decision edits flipped independent
status. Status was balanced near 50% open in every cell. This verifies the
local rules and causal controls; it does not establish that a bilinear model
learns them.

An example from the high-variation, coupled arm:

> Repair work at the Midtown school has prompted an update. A public timetable was requested when the site first closed for inspection. The site needed 12 desks repaired before it could reopen. An inspection found broken desks, and the site was closed for repairs. Classes continued in other rooms while the furniture was replaced. On Monday, Alice said crews had completed 11 desks. Another 2 desks were finished on Tuesday, Bob said. The department said these figures refer only to work already finished. Together, the two reports accounted for 13 desks. The final notice therefore listed the site as open. The final memo credited Bob with the second count.

The example is coherent at the event level, but its sentence structure is
still narrow. On 64 first-100-word documents per cell, the share of trigram
occurrences appearing in multiple documents is:

| Surface entropy | Coupling | Word types | Repeated-trigram share |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 211 | 95.0% |
| 0 | 1 | 225 | 94.4% |
| 1 | 0 | 271 | 89.7% |
| 1 | 1 | 278 | 89.7% |
| Fresh FineWeb reference | — | 2,397 | 0.73% |

The FineWeb reference is 64 documents at offsets 93,000 onward, cropped
to the same 100 words. It is broad web text, not a genre-matched set of
public-works reports; that weakens any claim about the *natural rate* of
repair content, but not the observation that a small hand-written template
bank repeats conspicuously. The report generator is even more repetitive
than the older V3 proposition surface. The source of the failure is
visible in the samples: a fixed sequence of target, inspection, count,
total, decision, and recap clauses, each drawn from only two to four
templates. The “purpose” label currently changes the heading more than
the discourse structure.

**Decision:** preserve V1 as a verified semantic/causal fixture and a
negative realism control. Do not train the main interpretability comparison
on it or call its rules learned. The next surface experiment needs a much
larger, corpus-backed realization space and genuinely variable discourse
structure. It must audit *task-bearing spans* as well as whole documents:
natural carrier text cannot be allowed to hide a repeated synthetic
sidebar. The rule-site parser and causal edits are reusable acceptance
tests for that successor.

## Corpus-backed continuation screen

A [10,000-document FineWeb yield screen](results/repair_corpus_carrier_yield_skip100000_n10000.json)
tested a simple alternative: select natural documents with a matching
project word and repair-related language, then use their prose as a carrier
for the controlled report. Of 10,000 sources, 6,334 had at least 200 words;
166 also had a project word and two repair terms; only 79 additionally
repeated a project word and had at least two line breaks. These are *upper
bounds* on usable carriers: the regex does not verify topic, entity
identity, or noncontradiction. A high-precision carrier pool at this
sampling rate would require far more source documents or repeated reuse
of the same prose, which could encourage memorization. A natural carrier
also would not make task-bearing clauses less templated, so whole-document
form scores alone would be misleading.

I also inspected [DART's official data-to-text release](https://github.com/Yale-LILY/dart)
as a possible source of richer semantic realizations. The
[paper](https://aclanthology.org/2021.naacl-main.37/) describes over 82,000
structured-record/text instances. The downloaded V1.1.1 training JSON has
30,526 triple-set records with multiple annotations. Its source partitions
mix human and automatically derived text; the official README explicitly
distinguishes these partitions. Direct inspection found some very short or
awkward annotations even outside the automatic partition. DART is a useful
candidate realization bank, but neither its triples nor its text directly
instantiate the multi-step event chain here, and it does not provide an
exact document-level rule oracle. The local training JSON has SHA-256
`92c8594979c05f508f5739047079ec2ffe5a244e58bfa2b50a9cb8b9c65f5a2b`
and stays outside Git. Do not count it as a solved renderer.

## Discourse-order control: verified but insufficient

The [order audit](audit_project_report_discourse.py) adds an opt-in
`discourse_entropy` control to the [same generator](project_report_dgp.py).
It permutes background facts among five plausible layouts while preserving
the work-report chronology and the visible evidence before all three answer
sites. The default remains the previous order. In a paired 64-document,
100-word screen, all 64 full texts changed. Four 1,000-plan semantic cells
(fixed/variable order × coupled/independent status) passed every total,
source, and status visible-prefix check and the corresponding selective
causal edits.

| Order | Whole-document word types | Repeated trigram share | Total-site window | Status-site window | Source-site window |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed | 289 | 88.8% | 90.3% | 92.9% | 92.9% |
| Variable | 292 | 86.9% | 91.5% | 93.0% | 93.1% |

The [aggregate results](results/project_report_discourse_order_audit.json)
show why order randomization is not enough. It modestly reduces full-text
repetition but leaves the *rule-bearing spans* as repetitive as before, or
slightly more so. Manual samples remain variations of one public-works
report. This is a negative control for discourse-order changes, not an
accepted successor distribution. The next meaningful change must diversify
the task-bearing realization and document process, not merely rearrange
the same sentences. No model was trained on this arm.
