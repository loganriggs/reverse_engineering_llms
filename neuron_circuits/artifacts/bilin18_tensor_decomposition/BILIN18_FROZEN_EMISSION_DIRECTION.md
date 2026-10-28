## H621 — A frozen absolute six-position direction transfers capital-emission effects, with limited broader generality

The previous turn was progress: H620 established cross-country transfer of nearly identical complement writes. H621 freezes their mean using only France, Germany, Italy and Japan in capital template0. The mean is an absolute6×1152 Attention5 body-write intervention outside the previous joint64 copying span. It has no recipient-dependent coefficient or norm matching. Apply it unchanged to the remaining four countries, another capital template, and language, chemical-symbol, plural and opposite-word prompts. Every recipient has six body tokens; all retain a common initial "the" and final "is".

Four native prefix/case conditions per item give192 prompts. On48 prefixed-lowercase recipients, evaluate native replay, common-direction addition/subtraction at0.25 and1, and random/reversed-position controls in both signs at1:432 evaluations, including48 native replays. A positive dose means subtraction; a negative dose means addition. Random vectors lie outside joint64 and match each position's intervention norm. The reversed control preserves total norm but reverses the six rows and changes per-position magnitude. No fit or scale changes follow evaluation.

**The frozen direction generalizes across country identity and to the second capital template.** On countries excluded from the mean, subtraction reduces correct-token mass from0.355 to0.069 and top1 from100% to25%. Addition increases mass to0.452, retaining correct top1. In capital template1, subtraction reduces mass0.265→0.023 and top1 to0%; addition increases mass to0.436. The correct capital remains ranked first among all eight capitals for every sign, dose and control. Random and reversed controls preserve capital top1 throughout. “Excluded” means excluded from this mean fit; these countries and templates appeared in prior experiments, so this is adaptive evaluation, not independent confirmation.

| Panel | Intervention | Correct top1 | Correct among declared candidates | Correct-token probability | Conditional candidate probability | Correct vs generic margin | Native-reference KL |
|---|---|---:|---:|---:|---:|---:|---:|
|capital0/fit|native|1.0000|1.0000|0.3657|0.9707|1.4463|0.0000|
|capital0/fit|add common|1.0000|1.0000|0.4546|0.9569|2.4444|0.0979|
|capital0/fit|subtract common|0.0000|1.0000|0.0528|0.9369|-1.5644|0.9939|
|capital0/fit|subtract random|1.0000|1.0000|0.3472|0.9639|1.2359|0.0140|
|capital0/fit|subtract reversed|1.0000|1.0000|0.3631|0.9723|1.4086|0.0009|
|capital0/heldout_identity|native|1.0000|1.0000|0.3552|0.9760|1.3573|0.0000|
|capital0/heldout_identity|add common|1.0000|1.0000|0.4524|0.9705|2.2003|0.0818|
|capital0/heldout_identity|subtract common|0.2500|1.0000|0.0695|0.9507|-1.2839|0.7821|
|capital0/heldout_identity|subtract random|1.0000|1.0000|0.3358|0.9691|1.1786|0.0155|
|capital0/heldout_identity|subtract reversed|1.0000|1.0000|0.3498|0.9771|1.3117|0.0010|
|capital1/transfer|native|1.0000|1.0000|0.2645|0.9715|0.5827|0.0000|
|capital1/transfer|add common|1.0000|1.0000|0.4356|0.9699|1.9534|0.2299|
|capital1/transfer|subtract common|0.0000|1.0000|0.0228|0.9163|-2.5298|0.9099|
|capital1/transfer|subtract random|1.0000|1.0000|0.2680|0.9664|0.6064|0.0111|
|capital1/transfer|subtract reversed|1.0000|1.0000|0.2506|0.9720|0.4800|0.0018|
|language/transfer|native|0.8750|0.8750|0.4877|0.6924|2.0580|0.0000|
|language/transfer|add common|0.8750|1.0000|0.5182|0.7172|2.3373|0.0169|
|language/transfer|subtract common|0.8750|0.8750|0.3470|0.6851|1.0449|0.1286|
|language/transfer|subtract random|0.8750|0.8750|0.4513|0.6750|1.6668|0.0102|
|language/transfer|subtract reversed|0.8750|0.8750|0.4809|0.6943|1.9924|0.0012|
|chemical/transfer|native|0.1250|0.7500|0.0407|0.4008|-1.1522|0.0000|
|chemical/transfer|add common|0.0000|0.6250|0.0281|0.3813|-1.6659|0.0715|
|chemical/transfer|subtract common|0.1250|0.7500|0.0401|0.3797|-1.4025|0.1237|
|chemical/transfer|subtract random|0.0000|0.6250|0.0360|0.3735|-1.6077|0.0298|
|chemical/transfer|subtract reversed|0.1250|0.7500|0.0408|0.4074|-1.1385|0.0013|
|plural/transfer|native|0.0000|1.0000|0.0034|0.9255|-4.4674|0.0000|
|plural/transfer|add common|0.0000|1.0000|0.0039|0.8978|-4.2567|0.0348|
|plural/transfer|subtract common|0.0000|1.0000|0.0013|0.9286|-5.5856|0.2498|
|plural/transfer|subtract random|0.0000|1.0000|0.0037|0.9185|-4.7789|0.0260|
|plural/transfer|subtract reversed|0.0000|1.0000|0.0029|0.9236|-4.6834|0.0038|
|opposite/transfer|native|0.1250|0.1250|0.0189|0.2319|-2.7285|0.0000|
|opposite/transfer|add common|0.1250|0.1250|0.0197|0.2328|-2.5537|0.0272|
|opposite/transfer|subtract common|0.0000|0.0000|0.0115|0.1690|-3.6083|0.0883|
|opposite/transfer|subtract random|0.0000|0.0000|0.0131|0.1761|-3.1904|0.0232|
|opposite/transfer|subtract reversed|0.1250|0.1250|0.0206|0.2360|-2.6566|0.0018|

**Language has weaker transfer in the same broad direction.** Subtraction lowers intended-language mass0.488→0.347, and addition raises it to0.518. Full-vocabulary accuracy stays87.5%; the model's native Japan answer is "English" rather than the predeclared "Japanese". Addition raises within-candidate accuracy from87.5% to100%, but does not yield100% unrestricted top1. Thus there is some identity-sensitive change as well; describing this as a perfectly pure emission variable would be inaccurate. The language relation remains close in scaffold and entities to the capital task.

**Broader-task evidence is limited by response format and competence.** Chemical symbols have12.5% native accepted-token top1 and75% within-candidate accuracy. Plurals have0% top1 despite100% within-candidate accuracy, often beginning with a quote instead of the answer token. Opposite prompts have12.5% for both metrics. These exact-token failures are retained; they are not proof of absent underlying knowledge. Multi-token continuations were not scored. Chemical-symbol addition actually decreases answer mass and candidate accuracy, contrary to a universal monotonic “emit the answer” direction. Plurals show the expected mass trend but start at only0.0034 native probability, so strong relative suppression does not imply substantial lost task performance.

Native competence across all four prefix/case conditions, reported as unrestricted top1 / within-candidate choice:

| Family | No prefix uppercase | No prefix lowercase | Prefix uppercase | Prefix lowercase |
|---|---|---|---|---|
|capital0|1.000/1.000|0.000/1.000|1.000/1.000|1.000/1.000|
|capital1|0.250/1.000|0.000/1.000|1.000/1.000|1.000/1.000|
|language|1.000/1.000|0.875/0.875|0.875/0.875|0.875/0.875|
|chemical|0.250/0.750|0.250/0.750|0.125/0.750|0.125/0.750|
|plural|0.000/1.000|0.000/1.000|0.000/1.000|0.000/1.000|
|opposite|0.125/0.125|0.000/0.000|0.125/0.125|0.125/0.125|

Generic competition remains the fixed set "the", "a", "not"; it is not an exhaustive nonanswer set. Candidate probabilities are conditional on the eight predeclared answers within a family, not semantic correctness over every possible completion. Exact language/chemistry labels can miss alternative spellings, synonyms or explanatory starts. These tasks test this frozen panel, not general knowledge competence. No new corpus documents were consumed.

**This is not yet one residual feature vector.** It is one fixed point in the space of six-position interventions: six different1152-dimensional vectors, scaled together. A positional reversal mostly removes the capital effect, showing sensitivity to placement and/or local direction-magnitude pairing. The last row contains about51.0% of squared norm and the penultimate24.0%; earlier rows collectively25.0%. These energy shares are not causal effect shares. Rank1 approximation of the6×1152 matrix captures78.6% of squared norm, rank2 captures89.6%. Neither approximation has been causally tested here, and Euclidean energy cannot certify behavioral fidelity.

The result strengthens a reusable local intervention hypothesis: a fixed scaffold-dependent direction changes direct factual-answer emission without usually changing the preferred capital. It weakens the claim that all coefficients must be inferred separately for each country. It does not establish a universal semantic feature, a single-vector unit, a unique canonical form, or a causal source for the direction. Common syntax, token positions, and downstream recipient sensitivity remain plausible explanations.

Next dissect the six-position intervention before calling it a unit. Compare individual-row removals, their combinations, and rank1/rank2 approximations at fixed overall strength on the frozen capital/language panel; include addition and small-dose checks. Measure actual downstream logit nonadditivity separately from KL, and determine whether final-position concentration is sufficient or non-final coordination matters. In parallel design a response-format evaluation that scores short continuations so quotes and explanatory starts are not automatically treated as wrong. Do not fit new directions to rescue failed transfer cells.

Audits: locked source/manifest/basis/vectors/runner hashes unchanged; first four unique countries verified as fit sources; all bodies exactly six tokens with matched prefix boundaries; random row norms match reference and reverse preserves total norm;192 native prompts,432 evaluations; exact native replay and finite metrics. Mean vector and manifest saved before model evaluation. FineWeb remains26,112 documents. Broad research goal active and incomplete.
