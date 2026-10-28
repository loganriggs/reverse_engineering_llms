## H547 — Distractor-path deletion rescues the failure without adding answer names

Follow up H546's early-distractor failure on the same 144-prompt panel. The L14H4 distractor source was selected for this diagnostic after observing the candidate-mean rescue; this is not independent fresh confirmation. Freeze seven conditions before running them: self, candidate-name mean, unrelated-name mean, retaining 75%/50%/25% of the native contribution, and zero. Only the persistent-bus contribution from that source to L14H4's final destination is altered. Other paths, attention values and downstream computation remain active.

The unrelated mean uses eight specified names, all token-disjoint from the candidates in this panel. Deletion and attenuation inject no recipient-name representation. The head, source role and conditions are fixed; no per-example tuning is performed.

| Early-distractor intervention | Candidate accuracy | Full-vocabulary top-1 | Mean correct-answer margin change |
|---|---:|---:|---:|
|self|33.3%|33.3%|+0.0000|
|candidate_mean|97.9%|97.9%|+1.2129|
|unrelated_mean|97.9%|97.9%|+1.2243|
|keep_75|52.1%|52.1%|+0.3093|
|keep_50|79.2%|79.2%|+0.6185|
|keep_25|89.6%|89.6%|+0.9253|
|zero|97.9%|97.9%|+1.2267|

Deletion reaches 47/48 correct, the same observed accuracy as candidate-name and unrelated-name means, versus 16/48 native. Accuracy improves progressively as the original contribution is attenuated: 25/48 at 75% retained, 38/48 at 50%, 43/48 at 25%, and 47/48 at zero. Correct-answer margins also improve monotonically in the aggregate. Thus candidate-answer injection is not necessary for this rescue. The deletion is local to one source/head bus edge, not an ablation of the entire head or distractor token.

The control conditions also improve candidate evidence when the distractor is later, but gains in candidate accuracy can differ from full-vocabulary top-1. With a middle distractor, deletion gives 48/48 candidate-correct and 46/48 full-vocabulary-correct, versus 45/48 and 44/48 natively. Parenthetical candidate accuracy remains 48/48, while full-vocabulary accuracy remains 43/48 under deletion. This distinction is retained rather than reporting only the easier candidate metric.

### Mechanistic implication and limits

On this panel, a value contribution from the irrelevant person's name through L14H4 causally favors an incorrect answer. Removing that contribution largely removes the early-distractor failure without supplying the correct name. This is a concrete, experimentally localized harmful computation. It does not prove a complete recipient-selection algorithm or that the same deletion improves general language modeling. The experiment uses the known distractor role to identify the source: it is not an automatic deployment-ready repair.

The conditional identity-reader account now explains both useful and harmful behavior: value copying can supply evidence for a recipient or a distractor depending on routing. The task label alone does not tell us whether that reader's contribution helps. Earlier successes with two names therefore should not be interpreted as proof of a general binding unit.

Leading alternatives for why the wrong source receives weight are: primacy or positional bias; insufficient context in the source key; an inadequate destination query; or narrative priors tied to clause order. A relevant architectural constraint is that the name's causal key cannot encode words that occur after it. In “C stayed home,” the disqualifying cue follows C, so any use of that later information must reach the reader through the query or another path. This constraint motivates a hypothesis, not a conclusion about the cause of the observed failure.

Next H548: move the irrelevance cue before versus after the distractor name while matching name and final positions, clause lengths and identity roles as closely as possible. Use fresh lexical combinations and keep L14H4/source interventions frozen. If earlier cue availability changes routing, cross source-key and destination-query states to locate the effect. Compare this with the primacy prediction; do not assume “irrelevance” has already been identified as a feature. Preserve fresh native accuracy and all counterexamples alongside deletion results.

Audits: 1,008 control interventions; self margins and candidate-mean replay agree within numerical error; unrelated donor tokens are disjoint from every candidate set; all outputs finite and source hash unchanged. The existing H546 panel is reused. No new prompts or FineWeb documents in this diagnostic. This goal turn added 576 new prompts across H545/H546 and completed 10,800 interventions across H545–H547. Tracked FineWeb total remains 26,112; broad goal active and incomplete.
