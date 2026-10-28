## H685 — Stable early blocks have lexical support and contextual internal coordinates

Inspect eight exploratory width16 blocks each from value0,residual0,residual1,seed681. Rank candidates by the minimum product of H684subspace overlap and activation-support Jaccard across matches to seeds682/683; require32validation selections. No task labels enter selection. This is validation-selected exploration, not a fresh confirmation set.

Fit simple activation-support predictors on training examples only: token identity, log2 token-frequency bin, absolute sampled position, and a token-plus-position log-odds combination. Token estimates use eight pseudo-observations at the training marginal activation probability. Evaluate Brier skill on validation:1−predictor squared error/marginal-predictor squared error. This is predictive error reduction, not a variance-explained percentage. Limited predictors do not remove every lexical/context confound.

For within-block coordinates, restrict to active training occurrences of token identities seen at least eight times in that block. Compute within-token squared variation divided by total variation on that subset. A posthoc magnitude control repeats this after unit-normalizing each coordinate vector, to distinguish directional context dependence from amplitude alone. This control was added after inspecting the first raw-coordinate audit. These are encoder-coordinate metrics and are not invariant to arbitrary nonorthogonal reparameterizations of a block; decoded-residual metrics remain an important follow-up.

| Site | Block | Validation active | Token Brier skill | Position skill | Within-token coordinate fraction | Unit-coordinate fraction |
|---|---:|---:|---:|---:|---:|---:|
|value0|121|192|0.952|-0.001|0.000|0.000|
|value0|132|153|0.946|-0.001|0.000|0.000|
|value0|35|78|0.877|0.000|0.000|0.000|
|value0|67|103|0.873|0.000|0.000|0.000|
|value0|71|57|0.394|-0.001|0.000|0.000|
|value0|116|42|0.779|-0.000|0.000|0.000|
|value0|47|63|0.822|-0.000|0.000|0.000|
|value0|23|86|0.599|0.001|0.000|0.000|
|residual0|27|173|0.809|-0.001|0.837|0.885|
|residual0|125|88|0.809|-0.002|0.926|0.948|
|residual0|56|52|0.954|-0.000|0.723|0.715|
|residual0|104|264|0.217|-0.001|0.315|0.363|
|residual0|143|184|0.767|-0.001|0.836|0.844|
|residual0|96|109|0.549|-0.001|0.635|0.955|
|residual0|134|48|0.633|-0.002|0.710|0.751|
|residual0|111|148|0.269|0.000|0.208|0.186|
|residual1|36|229|0.803|-0.000|0.665|0.724|
|residual1|8|185|0.775|-0.001|0.763|0.819|
|residual1|88|96|0.666|-0.001|0.883|0.899|
|residual1|44|142|0.770|-0.001|0.895|0.894|
|residual1|91|40|0.737|-0.000|0.937|0.916|
|residual1|16|124|0.798|0.000|0.735|0.756|
|residual1|73|162|0.506|0.000|0.761|0.924|
|residual1|30|44|0.588|-0.001|0.763|0.728|

Value0 provides a useful architecture-based control: all selected blocks show effectively zero within-token coordinate variation and no mixed activation support for repeated validation token identities. Token prediction is imperfect mainly because of finite counts/smoothing and rare or unseen identities; it must not be interpreted as contextual information in value0.

Residual support remains strongly lexical. Many stable blocks are dominated by punctuation, common function words, and byte fragments. Mean token-predictor skill is0.626 for selected residual0 blocks and0.705 for residual1. Position alone provides essentially no predictive improvement under this coarse16-position test. Frequency sometimes helps but generally less than token identity.

Yet active residual coordinates vary within token identities: average within-token fraction0.649 for residual0 and0.800 for residual1; after unit normalization,0.706 and0.833. These averages describe the selected blocks and eligible repeated-token occurrences, not the whole representation. They show that contextual variation exists within lexically associated blocks; they do not identify its semantic meaning or demonstrate a low-dimensional smooth manifold.

The promising unit may be a token-conditioned contextual state rather than a task label indicated solely by block activation. For example residual0 block27 often activates at periods, while residual1 block36 often activates at commas/parenthetical boundaries. These are descriptive lexical associations, not semantic names. Differences in internal coordinates could reflect syntax, discourse, nearby content, activation geometry or other nuisance factors. A stable punctuation block is not automatically a sentence-boundary computation.

### Exploratory atlas examples

Examples below are selected by coordinate norm, with at most one per validation document and two per token identity. They are illustrative, not a random sample or independently validated labels. Full metadata and six examples per block are saved in JSON.

**value0, block121** — frequent active tokens: '.' (156), ' up' (8), ' so' (6), '?' (6).

> /preview builds of latest OS which requires unlocked bootloaders, as they did with Xperia 2011-2012 line, so ⟦next text:  who knows may be Sony may give early⟧

>  to tell their stories, which Jackson-Beckham has been archiving. / “I am so happy that something so ⟦next text:  ugly turned out to be something so beautiful⟧

**value0, block132** — frequent active tokens: ' the' (122), ' The' (10), 'The' (6), ' just' (4).

> €36,616| / |2005||Form for the balance of requests for investments||€7,560| / | ⟦next text: 2004||Form for the balance of requests⟧

>  the night before, Cornell jumped out to an insurmountable 3-1 lead in the first period. / “Corn ⟦next text: ell is a tough team to play,⟧

**value0, block35** — frequent active tokens: ' in' (55), ' -' (3), ' within' (3), ' Sc' (3).

> + Statue series brings you brand new non-articulated comic book characters based on vintage 1980’s Super Powers action ⟦next text:  figure sculpts enlarged to 1/10⟧

> , Chile and Colombia. In Mexico, Walmart’s Bodega Aurrera brand ranked 22nd overall and fifth within ⟦next text:  the list of Mexican brands. Walmart�⟧

**residual0, block27** — frequent active tokens: '.' (148), '?' (3), ' of' (2), '...' (2).

>  have an extra hour or two in the week, consider volunteering as a way to help the community and to make new friends. ⟦next text:  Places that typically need volunteers include theaters,⟧

>  presents Prizes at Finals of All Britain Cookery Contest. / GV Kitchens. CU 'Festival Fare' banner. ⟦next text:  SV Three Kitchens CU Hands sifting⟧

**residual0, block125** — frequent active tokens: ' of' (72), ' than' (2), ' the' (2), ' up' (2).

>  the other. THE EXORCIST is presented at approximately 1.85:1 and the well balanced transfer places all the ⟦next text:  essentials up on the screen. Additionally,⟧

> ” / Callaghan Innovation legally came into being today. It is named after the late Sir Paul Callaghan, one of ⟦next text:  New Zealand’s top scientists who⟧

**residual0, block56** — frequent active tokens: '�' (41), ' �' (9), ' but' (1), '?' (1).

>  when she was senior in high school. / “I knew the clerk, he had been a teacher of mine,� ⟦next text: � said Johnson on how she got started⟧

>  a Parisian on a good day and even those snooty, chic folks will even say “Bon jour,� ⟦next text: � to a stranger. / But not⟧

**residual1, block36** — frequent active tokens: ',' (162), ' (' (13), ' and' (12), ':' (6).

> ique Ropion, Daniela ( ⟦next text: Roche) Andrier, Antoine⟧

>  civil records. The surname Moyney did not occur around Rathdowney, the surname Miney did. John Moyney ( ⟦next text: aka Jack Miney) is buried as⟧

**residual1, block8** — frequent active tokens: '.' (147), '\n' (14), ',' (7), '!' (4).

> ite for a fancy night out or a larger bag for everyday activities, our selection of crossbody bags has something for everyone! ⟦next text:  Choose from different sizes, colors, textures⟧

>  to meditate effectively yet! / The first step in getting started with meditation is finding the right location for your meditation session. ⟦next text:  You want an environment that will allow you⟧

**residual1, block88** — frequent active tokens: '\n' (81), '�' (5), '�' (3), ".'" (2).

> “I absolutely think if my father was alive he would have totally loved to be a part of this record,” ⟦next text:  she said. “I mean,⟧

> “Magnetars are fascinating objects, and this baby one appears to be especially intriguing given its extreme characteristics,” ⟦next text:  says Nanda Rea of the Institute⟧

Next verify directional variation in decoded residual space, then test whether within-token coordinates predict downstream computation beyond token identity and position. Use natural same-token donors, preserving lexical identity, to intervene on selected blocks and compare matched-size controls. Reserve a confirmation split for the frozen hypotheses. PCA, conventional ReLU SAE and causal fidelity comparisons remain outstanding; no unsupervised task ontology has yet been established.

Audits:24blocks, three unchanged checkpoints, all activation/stability/trainer/runner hashes unchanged; finite predictor scores. Zero original-model forwards, no test evaluations, no new documents. Magnitude-control rerun recomputed the same fixed block set and does not change featurizer training. FineWeb26,112 unchanged. Prior turn completed extended fitting/stability. Broad goal active and incomplete.
