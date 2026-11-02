# Numeric binding as a new toy-language capability gate

The existing toy languages model counting and numbered lists, but not a person–number binding followed by a comparison. A second, uniformly selected 32-window FineWeb review found overt entity-linked numbers in **12** excerpts, alongside attribution (10), section/list scope (10), condition-controlled advice (8), and structured field values (5). Labels overlap and come from one annotator reading only the first 160 GPT-2 tokens; they identify plausible local dependencies, **not** unique next-token answers or corpus-wide prevalence. The [selection and labels](fineweb_dependency_second_panel.py) save row IDs and short notes without source prose. This panel was previously used for *form-only* validation and is now development evidence for task design.

The new [`numeric_report_language.py`](numeric_report_language.py) is a controlled capability probe inspired by score and price bindings in those excerpts and by [numeracy probes using synthetic maximum tasks](https://arxiv.org/abs/1909.07940). Comparison and contextual retrieval are also distinguished in [NumericBench](https://arxiv.org/abs/2502.11075). The generator is not a copy of either benchmark. One example is:

> Scores:
>
> Henry scored 1 point.
>
> Leo earned 9 points.
>
> Who scored fewer points, Leo or Henry? Henry did.

Each short block independently chooses two distinct people and two distinct scores from 1–9, varies `scored`/`earned`, asks `more`/`fewer`, and randomizes person order in the question. The answer requires score–person binding and a magnitude comparison. Its visible-prefix oracle is exact. The audit replayed every prefix in **2,000 × 64** streams, independently recomputed **4,000** comparison answers from completed records, checked singular/plural agreement, and tested **4,000** instances of each supported edit: swap score assignments, swap person assignments, reverse the question order, or change an irrelevant record verb. The first two flip the oracle answer; the latter two preserve it. Answer position is balanced: first/second query name wins 1,956/2,044 times.

This addition does **not** make the toy language realistic. On 480 length-64 windows, it has **23** distinct alphabetic words versus **5,882** in a frozen FineWeb panel, with mean sentence length 7 versus 16.1 words. Every toy window contains a score, question, newline, and header colon, while those forms occur at different rates in natural windows. The tokenizers differ. The natural panel supports studying numeric bindings; it does not establish that explicit comparison questions are common in web prose. See [`numeric_report_audit.json`](numeric_report_audit.json) for exact surface counts and examples.

## Model competence

We trained the existing width-64, four-head, RMSNorm bilinear transformer for 3,000 steps at batch 64 and length 63, with depths 1–4 and ordinary next-token CE. An independent two-panel evaluation used 1,024 new streams per panel. A Bayes control that knows the two people and the two scores but cannot see **which score belongs to which person** assigns 1/2 to each answer, giving **ln 2 = 0.6931 nats** of answer KL. This is the relevant binding-blind gate. The fresh stream split does not hold out person–score combinations from language-model training.

| Model | Parameters | All-token oracle KL | Comparison-answer oracle KL | Score-swap model TV (oracle 1) | Person-swap model TV (oracle 1) |
|---|---:|---:|---:|---:|---:|
| 1-layer attention only | 48,384 | 0.0296 | 0.7256 | 0.0016 | 0.0220 |
| 2-layer attention only | 73,024 | **0.0271** | **0.7219** | 0.0033 | 0.0280 |
| 3-layer attention only | 97,664 | 0.0291 | 0.7373 | 0.0053 | 0.0595 |
| 4-layer attention only | 122,304 | 0.0278 | 0.7326 | 0.0028 | 0.0485 |
| 2-layer, answer CE ×10 | 73,024 | 0.0360 | 0.7461 | 0.0022 | 0.0180 |
| 2-layer, bilinear MLP width 128, answer CE ×10 | 122,304 | 0.0349 | 0.7456 | 0.0003 | 0.0090 |

Each causal-edit value uses **1,024** independently generated answer prefixes. The model distributions are almost invariant to score assignment, even though the oracle answer flips. Reversing question order, an exact no-effect edit, moves the ordinary models by about 0.015–0.027 TV; changing `scored` to `earned` moves them by at most about 0.0023. Thus low all-token KL is dominated by easy syntax and record generation. The model's average probability on the correct answer remains near 0.5. A score-answer-weighted objective and the native bilinear MLP do not rescue the comparison in these runs. The MLP control changes parameter count and architecture, so it is diagnostic rather than a matched capacity claim.

The negative result is a **model-competence gate**, not evidence that bilinear transformers cannot represent comparison. A larger or differently trained model might pass. Stop this local depth/loss sweep; these checkpoints should not be used to claim recovery of a numeric-comparison circuit through weight sparsity, tensor canonicalization, or featurizers. The score-report generator remains an auditable new task family for future mixtures. Before folding it into a richer joint language, require answer KL below the binding-blind baseline and selective, near-oracle score/person edits on held-out person–score combinations. Competent existing verb, FAQ, and minimal alias behaviors remain the current targets for decomposition experiments.

Reproduction: [oracle/realism audit](audit_numeric_report.py), [trainer](train_multi_genre.py) with `--language scores`, [fresh KL evaluator](eval_numeric_report_fresh.py), and [causal edit evaluator](eval_numeric_report_edits.py). The [ordinary model directory](numeric_report_models_seed0/), [answer-weighted control](numeric_report_weight10_depth2_seed0/), and [bilinear-MLP control](numeric_report_mlp_weight10_depth2_seed0/) include configurations, checkpoints, and separate fresh/effect JSONs. The first draft grammar accidentally produced `1 points`; it was corrected before the reported models were trained. Incomplete earlier checkpoints were excluded from the repository and analysis.
