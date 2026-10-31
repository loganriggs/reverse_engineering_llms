# A whole-composition holdout exposes the coverage cost of reduction

We trained five independent query-conditioned, eight-head bilinear-attention
models on 16 tasks over seven key tokens. Four tasks ask for the majority of
one token bit; six ask for XOR of two majorities; six ask for AND. The query
token selects the task. The native models reach 100% accuracy to four decimal
places on 30,000 fresh uniform sequences per seed and essentially the same
accuracy under a predeclared shifted token-frequency distribution.

This is a **featurizer holdout**, not a model-training holdout: the native
models learned XOR. Reduction fitting and gauge fitting see only the four
primitive and six AND query IDs. All six XOR query IDs are withheld until
evaluation. The all-query weight cut below is an explicitly leaky oracle
included to show what coverage of that family could buy.

## Stage one: which head subspace survives?

Each model's eight-dimensional head output is an exact sum of
query-dependent, per-key contributions. We verified that the extracted
`16 query × 16 key × 8 head` table reproduces direct head outputs on fresh
sequences, then fitted orthogonal projector cuts. The **weight-only**
projector uses only the 10 allowed queries' table rows. The
**sequence-aware** projector uses head outputs from 20,000 fresh sequences
with those same queries. The oracle uses all 16 query table rows. Neither
fitted method uses task labels.

Mean accuracy across five seeds on independent uniform sequences:

| Cut | Rank | Primitive | AND | Held-out XOR | XOR relative logit MSE |
|---|---:|---:|---:|---:|---:|
| Weight-only, allowed queries | 4 | 99.56% | 96.42% | 65.03% | 1.131 |
| Sequence-aware, allowed queries | 4 | 99.39% | 96.81% | 65.20% | 1.068 |
| All-query weight oracle | 4 | 97.89% | 95.78% | 90.61% | 0.332 |
| Weight-only, allowed queries | 6 | 99.95% | 99.03% | 82.30% | 0.557 |
| Sequence-aware, allowed queries | 6 | 99.95% | 98.75% | 81.96% | 0.578 |
| All-query weight oracle | 6 | 99.48% | 99.02% | 97.39% | 0.123 |
| Weight-only, allowed queries | 7 | 99.96% | 99.71% | 93.81% | 0.239 |
| Sequence-aware, allowed queries | 7 | 99.96% | 99.61% | 92.03% | 0.266 |
| All-query weight oracle | 7 | 99.49% | 99.30% | 99.35% | 0.038 |
| Any full-rank cut | 8 | 100% | 100% | 100% | 0 |

On shifted token frequencies, the allowed-query rank-six weight/sequence
cuts retain 79.92%/80.63% XOR accuracy; the all-query oracle retains
96.99%. Thus the deficit is not repaired by more empirical samples of the
same ten queries. It is primarily a missing *task-family* direction in the
chosen reduction measure. The oracle's improvement is not a valid blind
result, because it sees the withheld query weights.

The five rank-six allowed-query weight cuts range from 64.1% to 94.8%
XOR accuracy. Reporting the average alone would conceal that variation.
At a gate requiring at least 99% XOR decisions plus small logit error,
none of the tested sub-full-rank allowed-query cuts qualifies. This is a
coverage failure before any legibility optimization.

## Stage two: exact gauges and sample-dependent gates

At full rank, orthogonal rotations preserve every head output exactly.
We compared physical heads, PCA, weight/sequence varimax, edit-response
varimax, and eight random rotations. Every fitted rotation used only
primitive and AND queries. For each test example, a gate kept its two,
four, or six largest absolute coordinates. This is a simple
sample-dependent sparse gate, not a learned shared-dictionary/group model.

For four retained coordinates on uniform tests:

| Exact basis | Primitive accuracy | AND accuracy | Held-out XOR accuracy | XOR head relative MSE |
|---|---:|---:|---:|---:|
| Physical heads | 99.48% | 98.60% | **97.59%** | **0.050** |
| Sequence PCA | 99.20% | 98.41% | 94.10% | 0.098 |
| Weight PCA | 99.25% | 98.42% | 93.22% | 0.099 |
| Weight varimax | 99.82% | 98.96% | 94.93% | 0.087 |
| Sequence varimax | 99.49% | 98.83% | 95.05% | 0.093 |
| Edit varimax | 99.67% | 98.92% | 95.40% | 0.089 |

The physical-head advantage holds in all five seeds for top-four XOR
accuracy relative to edit varimax and persists under the token-frequency
shift (96.38% versus 93.79%). At top six, physical heads retain 99.91%
XOR accuracy and edit varimax 99.71%. The physical basis also has lower
head reconstruction error, so this does **not** establish a special
causal or semantic advantage after matching reconstruction. The model's
architecture trains independent heads, making a head-aligned basis a
particularly strong baseline in this toy. Random rotations and the
other top-k counts are recorded in the JSON.

## What this changes

This toy separates two questions that are easy to conflate. Withholding
a whole composition family during reduction exposes a large failure that
more samples from the seen families do not fix. Within the exact full
head space, generic sparsity priors do not beat the trained physical heads
under a simple activation gate. A useful learned shared-dictionary model
must now beat that physical-head baseline at **matched head
reconstruction**, then show held-out XOR prediction, finite intervention
fidelity, and shifted-token transfer. Otherwise its extra graph structure
has no demonstrated value here.

The model was trained on all tasks, the task list is deliberately simple,
and the cut is a linear head-state projector. These results do not show
automatic new-task discovery or generalize directly to the 546M model.
They do give a controlled, competent multi-rule benchmark with a genuine
whole-family featurizer holdout and a strong baseline for the next
sample-dependent group experiment.

Reproduce with `train_contextual_composition_attention.py` for seeds 0–4,
then `contextual_composition_reduction.py` and
`contextual_composition_gauge.py`. The JSON files of matching names contain
per-seed training histories, exact head tables, projectors, gauges, and
per-family uniform/shifted evaluations.
