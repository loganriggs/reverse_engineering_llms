# Sampled natural-claim document: source binding in one context

This pilot builds a **semi-parametric document distribution**, not the final
semantic/syntactic grammar. It samples two standalone sentences from the same
FineWeb source, assigns two sampled speakers, and later repeats a noun phrase
from one speaker's claim. The language inside the claims is human-written;
the attribution and recap frames are generated. Every token is a standard
next-token training target. The selected phrase is one of several eligible
noun chunks, so it is a sampled answer, not a unique exact-prefix oracle.

The [generator](sampled_quote_document_dgp.py) requires two distinct phrases
and checks the rendered document from visible text alone. Swapping the two
speaker assignments while keeping both claims and the recap fixed must make
the old recap invalid. The paired variant generator then changes the recap
phrase to one from the newly associated claim. This creates a direct causal
contrast: a model that follows the speaker binding should reverse its
preference between two phrase candidates. A fixed lexical prior cannot pass
the paired reversal test merely by favoring one phrase globally.

The [form and semantic audit](audit_sampled_quote_document.py) used two
source-disjoint 2,000-document local FineWeb preview panels. The same-source
natural crop is matched to each generated document's word count. No source
passages are committed.

| Panel | Eligible generated documents | Fit in 256 GPT-2 tokens | Visible checker and source-swap checks | Generated / natural repeated-trigram share | Generated / natural sentence-start types |
| --- | ---: | ---: | ---: | ---: | ---: |
| 130k–132k | 496 | 496 | 496 / 496 | .136 / .028 | 67 / 921 |
| 140k–142k | 519 | 519 | 519 / 519 | .141 / .027 | 71 / 966 |

The generated top-ten word share is .286/.294 versus .218/.220 for the paired
natural crops. This substantially reduces the old V4 proposition generator's
roughly .69 cross-document repeated-trigram share, but the fixed speaker and
recap wrappers still create a large form gap. Manual examples exposed source
fragments, nested attribution, and sometimes odd phrase recaps. Screening
removed obvious subordinate fragments, multiple periods, and byline-like
em-dash endings; it did not certify that every quote reads like a naturally
authored report. This arm should be treated as a useful controlled bridge,
not a realistically complete language distribution.

The [paired cache builder](build_sampled_quote_cache.py) keeps raw prose in a
local-only tensor cache, deduplicates sources across train/validation/test,
requires both arms to align at the same first phrase-token position, and
rejects pairs whose first BPE targets coincide. The source ranges are
100k–110k for training and 120k–122k for held-out evaluation. The
[shared bilinear trainer](train_sampled_quote_bilinear.py) uses full-token
cross-entropy with a separately reported first-phrase-token weight and paired
candidate, top-1, and causal-direction metrics. Those are behavioral gates;
weight interpretation is warranted only for models that pass them on held-out
sources.

The first full cache built 4,876 train pairs from 2,471 selected sources,
238 validation pairs from 238 sources, and 224 test pairs from 224 sources.
The two arms of every retained pair have the same target position and
different first-BPE targets. The data tensor is local-only at
`/home/loganriggs/Coding/neuron_circuits/artifacts/sampled_quote_paired_cache_10000_2000.pt`;
the [aggregate manifest](results/sampled_quote_paired_manifest_10000_2000.json)
records its source ranges and counts. A depth-2, width-64, four-head,
6.58-million-parameter smoke run at 100 steps reduced held-out token NLL
from 10.837 to 9.374 but had zero paired candidate reversals. More training
was needed before drawing a behavioral conclusion. At 12,000 steps, the
depth-2 model's held-out paired candidate-rank success was only .022 on 224
test pairs and .034 on 238 validation pairs. Its full-token test NLL fell
substantially, but source-swap logit direction stayed near chance. On a fixed
224-pair training slice, paired rank was .121 and first-site top-1 .449,
versus test .022 and .237. This suggests source-specific memorization or
partial training fit rather than transferable binding. The run was resumed
from a saved 6,000-step checkpoint after a single process exit code 139;
the final 12,000-step state and metrics completed normally. See
[depth-2 metrics](results/sampled_quote_depth2_d64_seed73161.json) and the
[train/held-out audit](results/sampled_quote_depth2_d64_binding_audit.json).

The depth-4, width-64 run with the same token budget and optimizer schedule
completed 12,000 steps (one seed per depth, so this is not a controlled
estimate of a depth effect). It fitted the
training slice better—paired rank .250, first-site top-1 .522, and source-swap
direction .915—but still gave only .029 validation and .022 test paired rank.
Test source-swap direction was .527, near chance. Restricting to test pairs
where both target BPE tokens appeared in training leaves paired rank .022
(178 pairs); unseen first-token types therefore do not explain the failure.
See [depth-4 metrics](results/sampled_quote_depth4_d64_seed73162.json) and
the [train/held-out audit](results/sampled_quote_depth4_d64_binding_audit.json).

The first marked BPE token is often an article: the most common target token
accounts for about 25% of test-site occurrences. To avoid treating that
single token as the whole copy task, the
[phrase-score audit](analyze_sampled_quote_phrase_scores.py) also forces both
complete candidate phrases after each arm's recap prompt (the recap wording
is the same; earlier speaker assignments differ) and
compares mean per-token autoregressive log probability. Its own-phrase score
agrees with direct checkpoint scoring to within 1e-6 on a checked example.
On fixed 224-pair slices, depth 2 gives paired phrase rank .071 train, .009
validation, .022 test; depth 4 gives .161, .009, .013. Symmetric source-swap
direction is near chance on held-out data (depth 2: .504 test; depth 4: .540)
despite .723/.871 on training slices. Full-phrase scoring therefore does not
rescue a hidden binding result. See [depth-2 phrase scores](results/sampled_quote_depth2_d64_phrase_scores.json)
and [depth-4 phrase scores](results/sampled_quote_depth4_d64_phrase_scores.json).

**Behavioral decision:** Neither model qualifies for source-binding weight
analysis. The deeper run improved in-sample fit without improving held-out
binding in this pair of runs, so we stop this arm as specified. The next generator iteration must
make useful semantic dependencies learnable across sources and preserve
natural variation without a rigid metalinguistic recap. This quote DGP remains
a controlled form/overfitting benchmark; it is not the main language DGP and
does not validate a canonical basis or sparse circuit.

This predeclared stopping rule was reached: source-disjoint two- and
four-layer pilots failed paired source-swap behavior after 12,000 steps.
Retain the generator/form control and do not interpret these checkpoints as
binding algorithms. The broader joint semantic/syntactic DGP remains the
main generator work.

Aggregate form outputs: [130k panel](results/sampled_quote_document_130000_n2000.json)
and [140k panel](results/sampled_quote_document_140000_n2000.json).
