# Two-site joint-rule toy language

`surface_mixture_v3.py` preserves the six-frame, four-way verb mixture from v2 and adds a second contested prediction at the **object noun in every sentence**. Four visible-prefix mechanisms compete there: a 70/20/10 adjective→object bigram, the previous object for the same subject, a frame cue, and the previous sentence's event. Candidate rules have fixed positive weights and are renormalized among active sources. The model is trained by ordinary next-token prediction, with an optional diagnostic that adds exact-oracle soft targets at both verb and object slots. This is a test of possible algorithm reuse across two token classes and two positions in one stream; it is not a claim that real language uses these artificial numerical mappings.

`audit_surface_mixture_v3.py` replayed all 216,000 prefixes of 3,000 independently generated 72-token streams, checked sampled-token support and probability normalization, independently recomputed both mixtures, and checked the six frame grammars on complete sentences. There were **12,088** verb decisions and **11,572** object decisions. Distinct rule modes conflicted at **11,725** verb slots (97.0%) and **11,533** object slots (99.7%). Both sites had 2, 3, and 4 active proposals in substantial numbers. The exact counts and representative generated text are in `surface_mixture_v3_audit.json`.

This adds mechanism coverage but does **not** fix the realism gap. In 480 matched 72-token windows, the synthetic stream has 309 observed words versus 6,305 in the frozen FineWeb panel; mean/median/p90 sentence length is 15.8/14/29 versus 16.3/14/29 words. Tokenization differs, and the generated prose remains templated. The sentence-length match should not be mistaken for linguistic diversity or realistic dependency prevalence.

`audit_surface_repetition.py` adds a more targeted form check. In the 457 windows where both streams have at least 40 alphabetic words, comparing exactly the first 40 words gives repeated-word fractions **27.5% synthetic versus 16.6% FineWeb**. After removing a fixed stopword list, repeated-content-word fractions are **11.0% versus 8.4%**. The content gap is smaller than the raw token-window comparison, which was confounded by different words per window. Exact word repetition is not coreference or induction, but this suggests that repetitive function-word templates are a larger realism defect than an extreme excess of repeated content words. Further realism work should vary sentence and document structures and manually annotate actual dependencies, rather than infer natural induction rates from repetition alone.

Other surface differences are more decisive. Among the same 480 token windows, **69.0%** of FineWeb windows contain a newline, **41.5%** a digit, **22.1%** a quotation mark, **20.6%** a parenthesis, **20.4%** a colon and **12.7%** a question mark. The v3 stream has none of these forms. These are presence counts, not task labels; they argue for adding document layout, lists/numbers, quotes and questions to the next *joint* generator rather than further padding the current sentence templates. A spot check of natural windows also showed repeated topic terms and discourse connections that are not equivalent to last-object copying; no prevalence estimate follows from that small inspection.

`train_surface_mixture.py --generator v3` reports exact-oracle KL at both decision sites, including the induction-active subset. `eval_surface_mixture_v3_edits.py` constructs supported one-token edits for both sites. At objects, the edits isolate the adjective bigram, frame cue, previous same-subject object, or previous event; changing the previous place is an exact no-effect control. The existing v2 verb edits are also scored on the v3 stream. Next-token KL, rule-specific causal fidelity, and local representation fidelity remain separate metrics.

The immediate comparison is a two-layer width-64, four-head RMSNorm bilinear LM trained for 3,000 steps at batch 64 under (a) sampled next-token CE and (b) the same CE plus exact-oracle verb and object soft-target terms of weight 2 each. Both use the same initialization seed and generated stream seed, with separate output directories. These are diagnostic single-seed runs, not a sample-efficiency conclusion. We should only test cross-site reuse after verifying competence and causal effects at both sites.

## Two-layer competence and causal fidelity

On a fixed 256-stream test set, the ordinary model reaches verb/induction-active verb/object KL **.364/.471/.693**. The soft-target model reaches **.098/.138/.225**. These are expected KLs to the exact prefix oracle, not sampled-token CE. The same 256-stream causal panel contains **3,004 verb edits** and **3,186 object edits**; the latter include 302 previous-object induction edits and 771 previous-event edits.

| Object edit | Oracle effect TV | Ordinary model effect TV / L1 error | Soft-target model effect TV / L1 error |
|---|---:|---:|---:|
| Current adjective | .527 | .573 / .534 | .598 / .424 |
| Current frame cue | .262 | .401 / .528 | .335 / .342 |
| Previous event | .180 | .085 / .466 | .290 / .380 |
| Previous same-subject object | .250 | .104 / .579 | .180 / .525 |
| Previous place, oracle no-effect | 0 | .015 / .029 | .019 / .037 |

The soft objective improves all four active object edit errors but still misses the precise historical effects. In particular, predicting **no change** for the previous-object edit has L1 error **.500**, slightly better than the soft model's **.525**. Its effect direction cosine rises from **.381** to **.680**, so it detects part of the dependency without reproducing the correct distributional change. Lower object KL is insufficient evidence of a recovered induction circuit. A three-layer soft-supervised capacity check is needed before testing cross-site mechanism reuse.

The matched three-layer, 3,000-step soft-target run improves verb/induction-active verb/object KL to **.060/.069/.196** and lowers verb-induction effect error to **.244**. But its previous-object induction error is still **.486**, only marginally better than no change; previous-event error is **.338**. This model is more competent, yet still does not justify an object-induction circuit interpretation. Its previous-place no-effect control at the object slot has L1 **.048**.

On a larger 1,024-stream panel with **1,234** previous-object edits, the three-layer model's induction effect error is **.494** for lags ≤24 tokens (1,008 cases) and **.506** for lags 25–48 (216 cases). Only 10 cases have lag >48. Thus the failure is present at ordinary short lags, not confined to very long retrieval. The two-layer soft model has corresponding errors **.530/.557**.

## First cross-site representation check and its gate

`eval_surface_v3_cross_site.py` fits eight-direction local response bases on verb induction and previous-place edits from three sentence frames, then tests on object induction and previous-event edits in the other three frames. It also fits object-rule bases as same-site upper bounds and repeats the frame split in reverse. This is a **featurizer task holdout**: the model was trained on both sites. At the two-layer model's final attention bond, full local patches nearly reproduce the model's effects for all four rules. At the first bond, full final-position patches have very small effects, so a first-bond response subspace would be geometrically interesting but causally inconclusive.

The cross-site result at the final bond is mixed. On held-out object-induction edits, an eight-direction verb-induction basis captures only **9.1%** of local activation-change energy (reverse **8.9%**), versus **34.3%/35.6%** for an object-induction basis fitted on different object examples. Even the object-specific eight-direction patch barely recovers the full output effect (patch-to-full L1 **.341/.344**, close to the roughly **.35** zero-patch baseline). At 32 directions, the object-specific basis captures **91.3%** of the activation difference and reduces patch error to **.104**, while the verb-induction basis captures **41.7%** and has patch error **.317**. This is evidence against a *compact shared final-bond response subspace* for these trained tasks, not against a shared algorithm elsewhere.

For the previous-event object rule, a verb previous-place basis captures **40.0%/41.6%** of its activation-change energy, versus **73.2%/72.9%** for an object-event basis (forward/reverse). Its forward patch-to-full error is **.519**, compared with **.243** for the object-specific basis and about **.592** for no patch. There is partial local transfer but substantial task-specific structure. Because the object rule itself is not yet accurately learned against the oracle, these cross-site results describe the model's current behavior, not ground-truth mechanism reuse.

In the three-layer model, the **middle attention-output bond** separates the two induction responses even more sharply. An eight-direction object-induction basis captures **60.9%/61.1%** of held-out object-induction response energy across the forward/reverse frame splits; the verb-induction basis captures only **0.4%/0.5%**. The full local patch at this bond has object-induction effect TV **.119/.110** versus the complete model's **.181/.187**, so this is a partially mediating location, not a complete circuit. At the final bond, the full patch nearly mediates object induction (**.184/.170**), but even the object-specific eight-direction basis captures only about one-third of its activation-change energy. The negative transfer therefore says specifically that *these local response spaces* are distinct. It does not test whether the attention score computation or a higher-level retrieval algorithm is shared.

The object-only training ablation keeps the same generated sequences and object soft targets while masking ordinary verb CE and removing verb soft targets. At 3,000 steps it improves object KL to **.193** from the joint two-layer model's **.225**. Its adjective/cue/object-induction causal errors improve to **.378/.304/.492** from **.424/.342/.525**, and the no-effect previous-place error improves to **.016** from **.037**. Yet object induction remains essentially at the **.500** no-change baseline. Removing the competing verb task helps general object fit but does not solve the critical historical computation. The corresponding verb KL is about **12**, as expected when verb prediction is deliberately excluded from training; it is not a viable full-language model.

This is the stopping criterion for interpreting an object-induction circuit in these checkpoints. The next model-side control should change **object mapping complexity or width** while keeping the exact oracle and paired edit panel fixed. Only after object-induction causal fidelity is materially above the no-change baseline should a shared-computation claim be revisited. The result also shows why a task-family holdout for a featurizer must be distinguished from model competence on that family.

## Object mapping and capacity controls

The v3 generator now accepts `object_vocab_size`, which restricts the **object prediction site** to a chosen prefix of the 63 available object nouns. The global model vocabulary stays at 313 and the verb mixture and six sentence frames are unchanged. All four object rules still compete, with the same weights; only their output mappings are reduced modulo the chosen object set. The eight-object audit replayed all 216,000 prefixes and found **11,061 of 11,572** object slots (95.6%) still have conflicting modal rule proposals. This is an easier conditional distribution, so raw KL across eight and 63 objects must be interpreted alongside oracle entropy and causal effects.

Two matched 3,000-step, depth-two, four-head, RMSNorm diagnostic runs compare (1) width 64 with eight object-site nouns against the existing width-64, 63-object model, and (2) width 128 with the original 63-object generator against width 64. Both retain exact-oracle soft terms of weight 2 at verb and object slots. The width comparison uses the same generated-stream seed and held-out data; the cardinality comparison changes the DGP while retaining architecture and global vocabulary. The decision criterion is whether the previous-object edit error falls clearly below the **.500** no-change baseline without inflating no-effect edits.

The first two controls show both effects matter, with cardinality stronger in this one seed:

| Object-site nouns | Model width | Object KL | Induction-active object KL | Previous-object edit L1 | Model/oracle effect TV |
|---:|---:|---:|---:|---:|---:|
| 8 | 64 | .098 | .133 | **.365** | .140/.250 |
| 63 | 64 | .225 | .359 | .525 | .180/.250 |
| 63 | 128 | .168 | .258 | .459 | .170/.250 |

The eight-object model also improves previous-event edit L1 to **.211** from **.380**, with object-induction direction cosine **.891** versus **.680** for the original width-64 model. Its oracle object entropy is lower (**1.207** versus **1.396** nats), and object-rule modal conflicts remain common (**11,061/11,572** object slots). The width-128 model improves the 63-object task without changing its oracle entropy, but still only modestly beats the .500 no-change induction baseline. Thus the large output mapping and model capacity both contribute; neither control fully recovers the rule.

The intermediate 16- and 32-object runs completed the width-64 sweep. All four cardinalities passed 216,000-prefix oracle audits, with modal rule conflicts at **95.6%, 97.9%, 99.1%, and 99.7%** of object slots for 8, 16, 32, and 63 objects. At 3,000 steps with the same objective and first training seed:

| Object nouns | Oracle entropy | Object KL | Induction-active object KL | Previous-object edit L1 | Induction effect cosine | Previous-place no-effect L1 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1.207 | .098 | .133 | **.365** | .891 | .047 |
| 16 | 1.317 | .144 | .187 | .421 | .858 | .046 |
| 32 | 1.368 | .164 | .227 | .440 | .827 | .072 |
| 63 | 1.396 | .225 | .359 | .525 | .680 | .037 |

The causal-error and effect-direction trends are monotonic across this sweep, while the model effect magnitude is not. Restricting to eight outputs makes the rule *more learnable* in this architecture/training regime, but also lowers the oracle entropy and changes collision identities; it is not a controlled change in parameter count alone. The fixed model vocabulary and presence of all four rule sources rule out a trivial change of model output dimension or removal of interference.

A second seed at the eight- and 63-object endpoints reproduced the direction of the result. Both models use the same 89,408-parameter architecture and the same training objective, but seed 1 changes the initialization and sampled training streams. The paired causal evaluation uses the same 256 base streams and edits for both. Results across the two seeds are:

| Object nouns | Seed | Object KL | Previous-object edit L1 | Induction effect cosine | Previous-place no-effect L1 |
|---:|---:|---:|---:|---:|---:|
| 8 | 0 | .098 | .365 | .891 | .047 |
| 63 | 0 | .225 | .525 | .680 | .037 |
| 8 | 1 | .100 | .378 | .881 | .035 |
| 63 | 1 | .207 | .491 | .760 | .062 |

The eight-object models are consistently better on this edit, but their errors remain substantial. In seed 1, the model's mean induction effect TV is **.129** versus the oracle's **.250**. The full-vocabulary seed-1 edit error of **.491** is nearly the **.500** no-change baseline. This is evidence about learnability under this generator and training setup, not evidence that a specific circuit has been recovered or that vocabulary size alone causes the gap. The cardinality diagnostic stops here. The next generator should broaden realistic document forms and distinct rule types, then retrain and test interventions before interpreting weight-only decompositions.
