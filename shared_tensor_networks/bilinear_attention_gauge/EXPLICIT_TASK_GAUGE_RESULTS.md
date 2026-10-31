# Explicit task supervision versus causal use in the 546M model

## Why this test

The prior balanced next-token-form gauge made gradients sparse but did not
yield transferable named axes. We therefore screened explicit, forced-choice
behaviors before fitting a new exact gauge of the same fixed rank-256
first-layer QK1-key projector. This panel probes whether competent task
labels yield reusable coordinates and whether those coordinates actually
mediate the model's behavior.

## Native competence and reduction fidelity

Two independent screen seeds tested seven prompt families. Antonyms,
bracket closure, country capitals, number continuation, and subject–verb
agreement scored 100% forced-choice accuracy on both seeds. Addition scored
80% then 60%; randomized association retrieval scored 55% then 65%. We
excluded those two from gauge fitting. The retained families use distinct
next-token answers and single-token answer/foil choices.

Fresh train/validation/test prompt seeds then supplied 172/92/172 examples.
The five families stayed at 100% native forced-choice accuracy on the test
prompts, and the fixed rank-256 QK1-key cut preserved every decision. Mean
absolute margin changes from native to cut were .004–.040 across families.
The variable-length hook reproduced the original head output exactly;
central finite differences and autograd agreed within 1.4e-5 absolute error.

## Exact rotations and whole-family holdouts

One rotation used gradients of the correct-minus-foil logit margin on 120
training examples from **agreement, number continuation, and bracket
closure**. It did not see antonyms or capitals. A separate all-five-family
rotation is an oracle for evaluating the value of adding task coverage; it
sees those families on training prompts. All rotations preserve the fixed
rank-256 reduced function.

The metric below is effective axes of the full key-position gradient,
`(sum_j sqrt(sum_position g_position,j^2))^2 / sum_position,j g_position,j^2`.
Lower means fewer axes explain a task margin's local sensitivity.

| Gauge on fresh test prompts | Agreement | Number sequence | Bracket | Antonym | Capital |
|---|---:|---:|---:|---:|---:|
| Native coordinates | 149.8 | 121.7 | 130.0 | 124.3 | 172.6 |
| Weight rotation | 136.1 | 94.0 | 108.2 | 102.3 | 167.4 |
| Corpus full-context rotation | 73.4 | 25.1 | 50.7 | 37.5 | 119.3 |
| Fit three task families | **18.8** | **12.7** | **17.2** | 50.5 | 145.4 |
| Add antonyms | 21.3 | 13.5 | 18.1 | **15.3** | 138.7 |
| Add capitals | 21.9 | 13.6 | 18.4 | 46.9 | **17.6** |
| All five task families | 23.8 | 14.2 | 19.2 | 16.8 | 18.8 |

The three-family gauge transfers within its trained families (validation
and test), but not to the two whole-family holdouts. Adding antonyms mainly
helps antonyms; adding capitals mainly helps capitals. This is concrete
evidence that task coverage matters for a supervised sparsity prior, with
little loss of sparsity on previously covered families. Prompt variants
reuse a small set of templates and lexical items, so this is not broad
natural-language generalization.

Training-selected axes for all five families show strong task-label
gradient-energy enrichment on new prompts. For example, the capital axis
has 92.1× enrichment on train, 139.3× on validation, and 129.7× on test;
the agreement axis has 84.2×, 78.5×, and 67.8×. These are derivative
associations, not yet explanations of the answer. Importantly, the
highest-energy position for the capital and antonym axes is the **first
token “The”**, not the country or adjective; the number-sequence axis
peaks on a comma. Template and positional structure can generate stable
task labels without representing the task variable itself.

## Finite interventions and path mediation

We froze one axis and relative key position per family from training
gradients, then applied ±.5 pre-RoPE QK1-key edits in the rank-256 model on
172 test prompts. The finite slopes agree with analytic gradients within
1.5e-5 absolute error. Each axis has a larger mean absolute finite slope
on its named family than on other families (4.9×–32.7×). Across all five,
the mean own-axis fraction of the five selected axes' squared responses is
.462 versus .200 under a family-label permutation (exact permutation
`p=.0083`). This aggregate is driven mainly by number continuation and
capitals; agreement, bracket, and antonym own-axis fractions are below .2.

The absolute behavioral effects are small. Mean own-axis margin slopes per
unit pre-key edit are .0007 (agreement), .010 (sequence), .0006 (bracket),
.0009 (antonym), and .044 (capital), while correct-answer margins are
roughly 2.8–8.9. Full-vocabulary KL changes at this dose are at most about
6e-5 on average for any family. Axis selectivity therefore does not imply
material control of a decision.

We next made **23 minimal token-swap pairs**: change the subject's number,
opening bracket, adjective, or country while retaining the answer pair.
All base and edited prompts have the correct forced-choice sign. We patched
the edited token's entire first-layer QK1-key state into the base run,
holding other routes fixed; we also patched only the selected axis **at that
changed token** (regardless of its training-selected position) and,
separately, the entire layer-0 attention output. Mean absolute margin
changes were:

| Family | Native token edit | Full QK1-key path | Selected axis | Full layer-0 attention output |
|---|---:|---:|---:|---:|
| Antonym | 5.044 | .0275 | .00016 | .2343 |
| Bracket | 14.698 | .0128 | .00206 | .0760 |
| Capital | 17.914 | .0226 | .00084 | .0236 |
| Agreement | 8.057 | .0219 | .00036 | .0615 |

These path effects are not additive mediation fractions, and their signs
often differ from the native token-edit effect. They show that the chosen
first-layer QK1-key coordinates—and mostly even the entire layer-0 attention
output—are a poor causal locus for these competent behaviors. The answers
likely depend more on the direct residual input and/or later computations;
that attribution remains to be tested. This failed causal gate overrides
the tempting interpretation of sparse, task-labeled gradients.

## Course correction and reproduction

Do not optimize these first-layer axes further for these prompt families.
Seek competent **attention-dependent** tasks whose native behavior is
materially mediated by an attention path, or move the canonicalization
target to a later layer. Require a path-specific causal gate before naming
an axis; keep reduction fidelity, task accuracy, gradient sparsity,
finite-dose selectivity, and natural-edit mediation separate.

Run `flagship_explicit_task_screen.py` for the two competence screens,
`flagship_explicit_task_gauge.py` to collect and fit exact rotations,
`analyze_explicit_task_gauge.py` for frozen axis/position selection,
`flagship_explicit_task_finite.py` for finite edits, and
`flagship_explicit_task_path_patch.py` for token-swap mediation. Artifacts
under `full_width_results/` use the stems
`all_head_explicit_task_screen_s9611_n20`,
`all_head_explicit_task_screen_s9711_n20`, and
`all_head_explicit_task_gauge_train9811_val9911_test10011`.
Everything ran on CPU while the separate GPU task-coverage job continued.
