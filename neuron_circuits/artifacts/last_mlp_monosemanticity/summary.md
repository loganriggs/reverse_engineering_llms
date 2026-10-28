# Last-MLP activation/output agreement

## Question

Does a final-layer MLP neuron's natural activation condition agree with the vocabulary
role implied by its decoder direction? This is an operational test for a coherent functional
role, not a proof that a neuron has exactly one human-interpretable meaning.

## Experiment

- Model: `Qwen/Qwen2.5-0.5B`; final MLP layer L23, 4,864 neurons
- Natural corpus: FineWeb `sample-10BT`
- Screening: 1,000,064 tokens for activation statistics over every L23 neuron
- Detailed search: 2,000,000 tokens for 12 stratified neurons
- Selection: six highest positive max-z neurons plus six activation-selectivity quantiles
- Evidence split: alternating natural-text windows form disjoint A and B partitions
- Exact causal effects: remove `h_k d_k`, then recompute final RMSNorm and the target logit

The primary cross-prediction metric is

`enrichment@100 = P(actual next token is in neuron's top-100 decoder tokens | top activation) / P(same token set in ordinary corpus)`.

This avoids choosing a token contrast after seeing the same activation examples. The report
also retains decoder percentile, exact causal help rate, activation/outcome concentration,
split-half stability, negative examples, and a conservative composite score.

## Main results

| Neuron | Decoder direction | Held-out enrichment@100 | Exact help rate | Interpretation |
|---|---|---:|---:|---|
| L23/N1368 | corrupted/replacement-character tokens | **3,596×** | 59% | Highly coherent technical encoding feature, but dominated by malformed Unicode/tokenization |
| L23/N703 | `may`, `might`, and multilingual possibility forms | **16.9×** | 42% | Best clean human-readable agreement, but strongly sign/context conditional |
| L23/N4718 | variants/translations of `since` | **5.18×** | 34% | Partial agreement with many exceptions |
| L23/N2264 | mixed fragments | **2.15×** | 47% | Weak evidence |
| L23/N3846 | almost exclusively `figure` variants | **0×** at 100; 6.49× at 1,000 | 57% | Extremely coherent output vector but mismatched positive activation conditions |
| L23/N3381 | mixed multilingual/code fragments | **0×** | 49% | No held-out activation/output agreement |

Eight of the twelve sampled neurons had zero held-out hits in their top 100 decoder tokens.
This sample was chosen for activation selectivity, so the table is not an unbiased estimate
of the percentage of all final-layer neurons that are monosemantic.

## A clean-looking but conditional neuron: L23/N703

Its strongest decoder tokens are:

` may`, `may`, ` MAY`, `May`, ` May`, Chinese possibility forms, and ` might`.

Strong positive natural activations include:

- `there [may] be some citations waiting...` — activation 38.25
- `you [may] want to look for a different version...` — 27.75
- `copyrighted material [may] not be re-published...` — 26.88
- `THIS COPY [MAY] NOT BE IN ITS FINAL FORM...` — 17.75

On held-out text, its actual next token landed in the decoder top 100 on 5.47% of top
activations versus a 0.323% ordinary-corpus baseline: 16.9-fold enrichment.

But its strongest negative activations also occur before `may`:

- `the template file [may] be...` — activation -20.63
- `it [may] remain constant...` — -18.00
- `we [may] arrive...` — -17.00
- `your doctor [may] also order...` — -15.88

Because a negative activation reverses the fixed decoder direction, the neuron suppresses
`may` in these contexts. It is therefore not a simple “predict may” neuron. A plausible
hypothesis is that it distinguishes subtypes or registers of modal constructions, but that
requires targeted counterfactual testing.

## Clean output does not guarantee clean activation: L23/N3846

Its top fifteen decoder tokens are almost entirely variants of `figure`, `fig`, and
`figures`. Nevertheless, its strongest positive activations occur on capitalized name
fragments such as `F` in *Figueiredo*, *Fogleman*, *Fidrych*, and *Ficus*, and `T` in
*Tanna* and *Tweed*. None of the 128 held-out top activations had an actual next token in
the top 100 decoder set.

Some of its strongest negative activations occur directly before `Figure`, for example
`[ Figure 1]`, which makes its fixed `figure`-promoting direction act with the opposite
sign. This is a direct counterexample to the claim that a coherent decoder direction by
itself establishes a coherent input-side concept.

## Conclusion

The final MLP contains locally interpretable output directions, but activation/output
agreement varies dramatically. The cleanest measured cross-prediction is an encoding
artifact; the best ordinary-language example is strongly context- and sign-dependent.
These results favor graded, predictive measures over binary neuron labels and do not
support treating a clean decoder vocabulary list as sufficient evidence of monosemanticity.

See `results.json` for all 12 neurons, 30 promoted and suppressed tokens per neuron,
positive and negative natural contexts from both splits, and exact RMSNorm-aware effects.
