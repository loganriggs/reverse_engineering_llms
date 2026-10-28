## H558 — Context PCA gives limited held-out response fidelity

Fit centered PCA on native normalized L8 contexts from lexical items 0–3, excluding items 4–7 across every family. There are 40 fitting and 40 evaluation prompts. Family bases have eight fitting endpoints each; global basis has 40. Fix family ranks 0/1/2/4/7 and global ranks 4/8/16/32 before evaluation. This is a retrospective split on previously inspected prompts, not fresh prospective evidence. Lexical blocks are disjoint in fitting/evaluation.

Project each test context onto the fixed basis and use it only as the base context in the local bilinear response operator. Retain recipient normalized displacement, quadratic term, baseline MLP output and all later computation. This compresses context modulation, not the full MLP or circuit. PCA optimizes context reconstruction; response-aware and output-aware bases are not tested here.

Run 880 interventions: 40 evaluation prompts × two amplitudes (+1/16,+1) × eleven modes (nine approximations, exact, zero-response). Compare each output distribution with the exact perturbed model using KL(exact||approximate), computed with float64 log probabilities. The zero-response baseline retains the attention perturbation but clamps L8's MLP to its native output. It is not the unperturbed model. Also retain exact-perturbed versus native KL so tiny-intervention fidelity is not mistaken for full-model equivalence.

| Family, +1 amplitude | Family mean: response error / KL ratio | Family rank 7: error / KL ratio | Global rank 32: error / KL ratio |
|---|---|---|---|
|binding|21.2% / 0.036|19.9% / 0.037|19.1% / 0.036|
|color_prose|55.7% / 0.319|52.2% / 0.258|51.0% / 0.242|
|color_record|62.2% / 0.338|59.0% / 0.319|58.1% / 0.302|
|name_prose|45.0% / 0.392|42.5% / 0.351|39.9% / 0.281|
|name_record|63.6% / 0.637|60.9% / 0.587|57.2% / 0.520|

KL ratios divide mean approximation KL by mean zero-response KL within each family/amplitude; they are not fractions of the whole model's computation explained. Binding's mean context already reduces that baseline KL by about 96%, while leaving 21% internal response error. Adding directions gives little improvement there. Copying remains less faithful: even 32 global directions leave 40–58% mean response error, and 24–52% of the zero-response KL. More directions help some cases but do not yield a uniformly accurate local operator.

At full amplitude, global-rank-32 mean KL is about 9.4e−7 for binding, 2.75e−4/2.61e−4 for color prose/record, and 7.50e−5/4.01e−4 for name prose/record. Absolute KL alone is not evidence of a fundamental unit or of faithfulness outside these selected interventions. Full-rank family PCA is limited to a seven-dimensional training span, so failure could reflect insufficient context sampling or a misaligned objective, not intrinsic high-dimensionality.

Audits: exact-response replay max margin error 1.91e−5 and max KL 2.77e−11; finite nonnegative KL; basis/source hashes unchanged; fitting/test IDs disjoint. Bases saved before interventions. No new prompts or FineWeb documents; corpus total 26,112. Next H559 freezes these bases and tests newly constructed lexical examples with no refitting or rank selection.
