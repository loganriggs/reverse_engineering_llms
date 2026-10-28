# Natural-corpus scan of maximum-circuit neurons

## Scope

- Model: `Qwen/Qwen2.5-0.5B`
- Corpus: `HuggingFaceFW/fineweb`, `sample-10BT`, training split
- Tokens forwarded: 10,000,000
- Tokens eligible for extrema after excluding window boundaries and document separators: 6,241,017
- Context retained: 24 tokens before and after the activating token
- Units: the four neurons highlighted by the `8 -> 2` maximum counterfactual

FineWeb is a natural web-text proxy, not Qwen's unreleased pretraining dataset. The raw
maximum/minimum is the literal extreme among eligible tokens. The JSON also contains ten
high and ten low readable contexts for each neuron and per-token activations throughout
each retained context.

## Literal extrema

| Unit | Mean ± SD | Raw maximum | Raw minimum |
|---|---:|---|---|
| L0/N2905 | -0.003 ± 0.233 | **3.297** on `8` in `Code 8000E-48P ... Weight 11339` | **-1.594** on ` Consum` in `Healthy Consuming...` |
| L0/N2206 | 0.030 ± 0.418 | **4.688** on `2` in `4 GB DDR2 800 Mhz...` | **-1.648** on ` Nigel` in `...supervised by Nigel Lilley...` |
| L21/N2550 | -0.201 ± 0.536 | **61.500** on `9` in a patent/date table (`...1983...1985...`) | **-10.750** on ` Camb` in `Meneghini La Cambusa...` |
| L22/N30 | -0.381 ± 0.567 | **52.250** on `1` in an IP-address/configuration listing | **-12.375** on ` Ham` in `Mark Hamill...` |

## What the larger search changes

The early layer-0 interpretation becomes stronger. Every one of the ten highest readable
contexts for L0/N2905 targets token `8`; every one of the ten highest contexts for
L0/N2206 targets token `2`. These neurons appear to be local token-identity features that
the maximum circuit can reuse, not neurons dedicated to the maximum algorithm.

The late-neuron interpretation becomes less literal. L21/N2550 and L22/N30 were selective
for answers 8 and 6 inside the controlled maximum dataset, but their largest natural-text
responses occur on digits in dates, identifiers, tables, IP addresses, and other structured
numeric passages. They are therefore better described as broad, context-sensitive numeric
features that causally influence the answer in the tested prompt—not globally pure output-
digit neurons.

The activation extrema establish natural selectivity. The counterfactual patching results
in the hierarchical-circuit experiment separately establish causal influence on the model's
8-versus-6 decision.

## Full contexts

See `examples.json` for:

- the exact raw extrema;
- ten high and ten low readable contexts per neuron;
- the activating token and its offset;
- all 49 token strings and activation values in every context;
- source window indices and aggregate activation statistics.
