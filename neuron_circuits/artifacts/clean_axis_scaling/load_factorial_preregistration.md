# Class-load × width follow-up preregistration

Sealed on 2026-10-17 after the one-factor scaling sweep and before running the four missing factorial corners.

## Motivation

The one-factor sweep found that class count increased both absolute raw-axis advantage and the normalized Clean Axis Index, while increasing width had metric-dependent effects. A 3×3 class-count by width factorial distinguishes a class-complexity effect from a simple classes-per-neuron load effect.

## Frozen design

- Classes: 10, 30, 100.
- Neurons per layer: 64, 128, 256.
- Operations: 4; depth: 2; all other settings identical to the initial scaling study.
- Eight seeds per cell. Five existing cells are reused exactly; the four missing corners are run without modifying their experiment configuration.
- Experiment hash: `5b64704e7578a0da4a0f055e645735d73da4bb423b12d5863acb6202829db285`.
- Follow-up runner hash: `75098a12f296cb65ced841bc1d013a53c71b42d065445a098dbf3d53139dc9b7`.

## Analysis

Report the preregistered Clean Axis Index, raw-axis advantage, and threshold efficiency for all nine cells. Fit an exploratory two-factor model using `log(classes)`, `log(width)`, and their interaction. Accuracy and convergence remain mandatory controls.

The key robustness question is whether class count has a positive effect at every width. A trend that appears only at one width will not be described as a general scaling law.

