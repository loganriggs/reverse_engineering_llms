# Reconnect the language benchmark to the existing mixed-rule generator

The user's [project proposal](https://www.lesswrong.com/posts/GncmyuKmp526RJpmj/ambitious-mech-interp-w-tensor-transformers-on-toy-languages) states the organizing loop: build a toy LLM from known computational primitives, use it to study phenomena and discover better interpretation tools, then feed the new primitives back into the language. The explicit appeal of the toy is knowing the *joint* DGP, including how bigram statistics, skip rules, and induction interfere. This is a stronger and more relevant target than merely adding individually labeled prompt families.

The earlier toy-language work was already implemented in a **different local repository**, [`/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits`](../../toy_models_of_tensor_networks/attn_circuits). The first survey concentrated on `tensor_language` and its natural-text experiments and missed this directory. That was a research-direction error: the new 12-family prompts and document worlds are useful for semantic state tracking, but they do not replace a generative language model whose rule distributions interfere at every token.

## Existing assets worth reusing

- `generator.py` implements autoregressive mixed and isolated streams with token bigrams, token trigrams, skip-bigrams, induction, brackets, configurable class weights, and per-token rule labels. Shared output tokens let different rules favor the same next token. Its weights are normalized **among the candidates currently active**; setup tokens and whole bracket spans follow additional sampling paths. It is therefore not literally a fixed unconditional weighted sum of rule distributions at every position.
- `generator_v3.py` has a more readable category-based stream: probabilistic noun bigrams (70/20/10), category transitions, token/category trigrams, skip-bigrams, noun induction, and bracket/quote spans. Here rule dispatch is **priority-based**, not the earlier sampled mixture. It retains stochastic distributions within rules. The label `skip_trigram` denotes a bracket closer, not a general skip-trigram grammar.
- `train.py`, `eval_rules.py`, `model.py`, saved configurations/checkpoints, and circuit analyses already support and evaluate these generators. For example, `models/bilinear_gpt_v3_all_2L_4H_d64_5000steps.pt` is a two-attention-layer, 64-dimensional, no-MLP checkpoint. Its saved per-rule held-out CE includes noun bigram 0.890, induction 0.124, skip-bigram 0.073, token trigram 0.086, and bracket close 0.605 nats. These are **not** directly comparable across rule types without their true conditional entropies; the saved entropy estimates for some stochastic labels are approximate.

`audit_existing_mixture.py` sampled 1,000 × 128 tokens from each generator. In the original mixed generator, 54,639 positions were labeled as one of the four ordinary rules. Among these, 21,103 had one active candidate, 23,374 had two, and 10,162 had three. **37.9% of rule-labeled positions had multiple active candidates proposing different outputs.** This is the concrete collision/interference regime the newer document-world work omitted. See `existing_mixture_audit.json`.

A direct conditional-probability check used context `a B0 a`: the bigram proposes `X0`, while induction proposes `B0`. Their configured weights imply P(`X0`) = 0.25/(0.25+0.20) = 0.556. Across 20,000 draws, the generator emitted `X0` with frequency 0.564. This checks one collision branch, not the full generator oracle.

## Corrected experimental role

The mixed-rule generator should be the **primary sandbox for competition and shared computation**: vary the bigram distribution and induction/skip/bracket prevalence, measure per-rule held-out CE or exact KL, then ask whether sparse/canonical attention coordinates recover the planted rules and their conflicts. The document-world generator is a complementary stress test for longer context, state updates, and cross-sentence binding. It should be added as another task distribution only after the mixed-rule model's own conditional probabilities and evaluation are audited.

Immediate tests, in order:

1. Construct an exact next-token oracle for the original mixed generator, including active-candidate normalization, setup, and bracket decisions. Verify Monte Carlo frequencies against the oracle on controlled contexts. Existing labels alone do not give the full distribution.
2. Sweep rule collisions by varying configured weights and shared-output assignments while holding marginal token counts and model size as stable as possible. Train ordinary full next-token models; report per-rule CE/KL, not only answer accuracy. Include isolated-rule and no-conflict controls.
3. Use existing trained two-layer checkpoints as baselines before new training. Test exact gauge invariance and sparse/intervention-aware decompositions on held-out conflict patterns. A component should predict the *change* when a rule's weight or trigger is altered, including cases where another rule overrides it.
4. Extend to 1–4 bilinear blocks with MLPs, then test transfer to the document-world tasks and natural-text cue classes. Keep whole-rule-family and conflict-pattern holdouts distinct from ordinary token holdouts.

The v3 surface stream is only loosely English-like, and the document world is templated. Neither has yet passed a meaningful natural-text similarity check. FineWeb should calibrate frequencies and contexts, while the exact synthetic rules remain separate ground truth.
