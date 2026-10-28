# Breadth-first ontology protocol

## Why the four familiar tasks are insufficient

Spelling, sentiment, gender, and tense are useful calibration cases because clean
counterfactuals can be constructed for them. They are not a representative sample
of the computations performed by a language model. A method that explains those
four tasks could still miss syntax, copying, retrieval, entity binding, factual
recall, formatting, discourse state, code, numbers, web boilerplate, and the many
unnamed distinctions that occur naturally in FineWeb.

The existing evidence already demonstrates this selection problem:

- a manually broadened screen tested 17 task families, but only seven passed the
  full-vocabulary competence gate;
- label-blind FineWeb discovery found document-boundary formatting, punctuation
  closure, determiner/preposition relations, lexical echo, long-range copy, and
  name-valued association without starting from the four familiar labels;
- the nine prospectively validated named route modes explain 11.5% of model-wide
  attention-edge energy at the primary membership threshold, while all sixteen
  geometric modes explain 28.7%.

The last result is the most important guardrail. Interpretable examples are real,
but they do not yet cover most of the measured computation.

## Evidence hierarchy

Use three complementary sources of evidence, in this order.

1. **Natural discovery population.** Sample random FineWeb positions, weighting
   analyses by causal effect on the model's own predictive distribution. Discover
   coordinates and clusters without semantic labels.
2. **Controlled behavior bank.** Use tasks with exact counterfactuals to determine
   whether a discovered mechanism performs a proposed operation. Include syntax,
   binding, retrieval, copying, factual recall, negation/composition, formatting,
   numerical or algorithmic behavior, and local lexical prediction. First verify
   that the model can actually perform each task under a full-vocabulary metric.
3. **Synthetic falsification tasks.** Construct cases that separate competing
   explanations—for example, semantic category versus token identity, copy versus
   factual association, current-token response versus contextual relation, and
   generic transport versus task-specific content.

No claim about the model-wide ontology should be based only on the controlled bank.

## Primary sampling frame

The denominator is a preregistered set of random FineWeb prediction sites, not a
set of successful tasks. Stratify sites by quantities known before interpretation:

- layer and source-to-destination distance;
- full-model CE or KL sensitivity;
- attention-led, MLP-led, and carried-state contribution;
- local versus long-range source position;
- common versus rare token frequency;
- document region and coarse tokenizer form;
- confidence, entropy, and loss quantile.

Within every stratum, retain both high-effect and ordinary sites. High-effect-only
sampling finds important circuits but exaggerates sparsity and cleanliness.

## Discovery and interpretation loop

1. Represent each natural event using symmetry-invariant, behaviorally grounded
   descriptors: QK routing, OV payload, bilinear MLP response, source/destination
   geometry, and its finite effect on logits and CE.
2. Remove already validated modes and cluster or factorize the remaining
   importance-weighted residual. Do not use semantic labels during this step.
3. Freeze the discovered axes, clusters, thresholds, and example-selection rule.
4. Draw at least 50 discovery examples and 50 disjoint validation examples for
   every sufficiently common cluster, including both poles and low-activation
   controls.
5. Label recursively: prefer the broadest description that predicts held-out
   membership and causal effects. A broad family such as “close a local syntactic
   constituent” is better than a list of unrelated punctuation tokens if it makes
   new predictions.
6. Test the frozen explanation with finite ablations, activation or route patches,
   and counterfactual source changes. Compare against matched random subspaces,
   frequency-matched clusters, and generic carrier programs.
7. Measure incremental CE, KL, and full-logit coverage on the original random
   population. Repeat discovery on the residual until gains saturate.

## Controlled behavior bank

The bank should span mechanisms rather than merely topics:

| Mechanism family | Example behaviors |
|---|---|
| Local lexical/morphological | token continuation, affixes, capitalization, spelling |
| Local syntax | agreement, determiner/noun compatibility, constituent closure |
| Binding/reference | IOI, pronoun/entity tracking, repeated-name disambiguation |
| Transport/retrieval | previous-token, induction/copy, repeated sequence, key-value recall |
| Stored association | capitals, common entity properties, lexical relations |
| Composition/control | negation, conditionals, quotation/bracket scope, instruction boundaries |
| Quantitative/algorithmic | order, counting, arithmetic where the model is competent |
| Document/discourse | newlines, headings, topic or register transitions, boilerplate |
| Specialized distributions | code, tables, URLs, citations, markup, multilingual fragments |

Failures of competence remain informative negative controls, but are not eligible
as positive circuit targets.

## Locked success criteria for an ontology item

A proposed feature or operation is retained only if it:

- recurs on a disjoint FineWeb panel;
- predicts more than token frequency, current-token identity, position, answer,
  and source-distance controls;
- has a finite causal effect in the predicted direction;
- beats an equal-budget random or frequency-matched subspace;
- transfers to new contexts within its claimed scope;
- improves importance-weighted residual coverage rather than only producing vivid
  top examples.

Report three separate quantities: descriptive purity, causal fidelity, and global
coverage. None substitutes for the others.

## Red teams

- **Curated-task bias:** compare every task-derived mechanism with the label-blind
  natural atlas.
- **Frequency masquerading as semantics:** match token and context frequencies and
  evaluate rare strata separately.
- **Current-token shortcut:** condition on the repeated input embedding injected at
  every layer and test same-token/different-context pairs.
- **Generic carrier mistaken for content:** swap payloads and task families while
  holding the route fixed.
- **Cluster instability:** rerun discovery on disjoint corpora and compare
  projectors, not arbitrary axis rotations.
- **Human hindsight:** freeze names and operational predictions before viewing the
  validation examples.
- **Energy without relevance:** evaluate CE/KL and finite interventions, not only
  activation or route energy.
- **Sparse-tail storytelling:** report coverage over the whole preregistered
  population, including unexplained events.

## How the original tasks should now be used

The four familiar tasks remain valuable unit tests. They establish that the tools
can recover known counterfactual distinctions and reveal concrete variants of
transport, carried state, and compact OV payloads. They should not determine the
feature vocabulary, the sampling distribution, or the model-wide conclusion.

