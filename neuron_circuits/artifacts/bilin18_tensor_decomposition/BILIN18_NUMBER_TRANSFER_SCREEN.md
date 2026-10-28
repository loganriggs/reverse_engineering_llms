## H887 — Frozen number axes generalize to irregular plurals and new contexts

H886's regular-plural morphology confound is substantially weakened: the frozen discovery directions and thresholds classify all seven eligible irregular singular/plural pairs correctly at L11–16 in both competent templates B and C, using either original trained axis. No directions, thresholds or layer sets were refitted. This supports a number-related representation beyond an s-suffix detector. It remains a geometric transfer result; the causal coordinate intervention has not yet been repeated on these new nouns.

### New panel and frozen evaluation

Eight irregular pairs were declared: man/men, woman/women, child/children, person/people, foot/feet, tooth/teeth, mouse/mice and goose/geese. Goose/geese failed the single-token source eligibility rule and is retained as a tokenization exclusion. Eight new regular pairs are teacher(s), worker(s), student(s), farmer(s), artist(s), driver(s), doctor(s) and player(s). No pair was excluded for correctness.

Six agreement templates: A “The noun near the river”; B “The noun in the garden”; C “The noun beside the road”; D “The noun near the dogs”; E “The noun near the dog”; F “The noun that the guards noticed”. A/B are prior templates; C–F test new contexts including plural and singular distractors. Matched repeated-word copying controls accompany every cell. Total 15 eligible pairs and 360 endpoints across tasks, templates and orders. Copy prompts repeat across some template labels and are not independent new examples.

Apply both H886 original A/B trained directions and number-centroid thresholds to every new context at every layer L8–17. Directions are trained only on dog(s), cat(s), book(s), tree(s). No best-axis choice is made. The L11–16 set was specified by H886 before these new outcomes. Layer-wise repetitions are correlated predictions, not additional independent lexical examples.

### Behavioral gate and geometry kept separate

The frozen gate remains at least four eligible pairs, both endpoints binary-correct on at least75% of pairs, and pooled full-vocabulary exact is/are top-1 at least50%, with a passing same-template copy control for later paired interventions.

|Cohort|Template|Exact is/are top-1|Paired gate passes|L11–16 accuracy range, frozen A axis|L11–16 accuracy range, frozen B axis|
|---|---|---:|---|---|---|
|irregular|A|0.429|No|1.000–1.000|1.000–1.000|
|irregular|B|0.929|Yes|1.000–1.000|1.000–1.000|
|irregular|C|0.571|Yes|1.000–1.000|1.000–1.000|
|irregular|D|0.214|No|0.929–1.000|1.000–1.000|
|irregular|E|0.071|No|1.000–1.000|1.000–1.000|
|irregular|F|0.000|No|1.000–1.000|1.000–1.000|
|new_regular|A|0.500|Yes|1.000–1.000|1.000–1.000|
|new_regular|B|1.000|Yes|1.000–1.000|1.000–1.000|
|new_regular|C|0.375|No|1.000–1.000|1.000–1.000|
|new_regular|D|0.250|No|0.875–1.000|1.000–1.000|
|new_regular|E|0.125|No|1.000–1.000|0.938–1.000|
|new_regular|F|0.000|No|1.000–1.000|1.000–1.000|

All task cells have100% binary is-versus-are discrimination. The original gate selects irregular B/C and new-regular A/B. In particular C provides new-context irregular transfer. In distractor templates, number decoding remains nearly perfect, with a few errors at L16 in D/E. Most of those contexts do not pass the exact next-token gate, so they are reported as representation evidence separately from the selected behavioral cells.

### The strict gate does not imply grammatical incompetence

Inspection of native predictions shows that some failures are tense or continuation choices: F predicts “was” or “were” on every example. A descriptive, explicitly posthoc check accepting singular is/was and plural are/were gives100% number-correct top-1 in F for both cohorts; irregular A/B/C also reach100%. This check does not replace the original gate or retroactively add intervention cells. D/E also produce punctuation or noun continuations, so expanding tense alone does not rescue all failures. The distinction prevents calling a correct past-tense verb a model agreement failure.

The near-perfect frozen number decoding in F is therefore consistent with its number-correct past-tense behavior, even though the originally declared present-tense answer set was too narrow for that context. A later tense-inclusive task test should be explicitly declared before its new outcomes.

### Interpretation and next causal step

A simple regular-plural suffix explanation no longer accounts for these frozen-coordinate results. New lexical identities, irregular morphology, both original axes and the new neutral C template preserve the number distinction. This does not yet distinguish every possible lexical or semantic correlate of number, establish uniqueness of the coordinate, or prove native behavior requires it.

Next repeat coordinate interventions on the originally eligible irregular B/C and new-regular A/B cells, using frozen means/directions/centroids. Compare the prespecified L11–16 set with its complement. Keep both trained axes for C rather than selecting whichever works better. Preserve individual-response fidelity and full-distribution metrics, and separate manipulation of a local response Jacobian from direct interventions on the native MLP computation. The latter is needed before claiming the feature controls unperturbed agreement behavior. Copying coverage/learning-curve work remains outstanding.

### Verification

372 full-model forwards including replay, 360 panel endpoints and 15 eligible pairs. Native replay exact. Frozen fit and centroid hashes unchanged, as are model and runner hashes. Independently verified aligned source-only endpoint changes for all 360 contexts and complete evaluation of both axes at all ten layers. All outputs finite; all exclusions and failed gate cells retained. No reserved test evaluated; natural-document pool stays26,112. Broad research goal active.
