# Typed lexical sweep: useful control, insufficient realism

The research target is a family of sampled language distributions with known, variable rules. Models should be behaviorally tested for each rule before their weights are interpreted. Capability is an eligibility check for a mechanistic claim, not the main target. The intended comparison is whether weight-based sparsification, factorization, or canonicalization recovers rules across changes in lexical frequencies, genre, syntax, and task interactions.

V17 isolates one data-generating-process lever. It keeps V2's six visible sentence frames, subject-dependent verb proposal, previous-verb induction, lead proposal, and previous-place proposal. It expands the typed lexical roles from the V2 baseline and introduces a nonnegative `frequency_exponent` over smoothed FineWeb part-of-speech counts. Zero gives a uniform draw within each role; 0.5 and 1 progressively favor more frequent role words. The lexical prior was fitted on 512 FineWeb sample-10BT documents at offset 32000, first 5000 characters per document. Candidate roles were manually curated, and the POS counts are noisy frequency estimates, not semantic labels. The runtime generator reads a frozen JSON table; no FineWeb text or NLTK package is needed to sample.

The frozen inventory has 97 people, 146 objects, 88 places, 60 events, and 81 adjectives, for a 524-token total generator vocabulary. Its exact visible-prefix next-token distribution remains available. In a 200-stream × 100-token check for each of three exponents, all 20,000 sampled tokens had positive oracle support, every prefix replayed the recorded distribution exactly, and normalization error stayed below 1e-9. About 1,092 mixed-verb slots occurred per setting; 1,066–1,070 had distinct named proposals. Induction was available at 403–410 slots and previous-place at 892. These are rule opportunities, not evidence a trained model learned them.

The frequency knob does change content-word sampling: the top-ten object words carry .068/.133/.241 mass at exponents 0/.5/1, and the top-ten places carry .114/.225/.399. It leaves the fixed grammatical frame probabilities alone.

Two untouched FineWeb offsets, 34000 and 35000, each contribute 64 first-100-word windows. Generated streams use distinct seeds by offset. The code saves aggregate statistics only.

| Offset | Panel | Word types | Top-ten word share | First-40 repeated-word fraction |
| --- | --- | ---: | ---: | ---: |
| 34000 | V2 | 309 | .398 | .270 |
| 34000 | V17 exponent 0 | 512 | .398 | .267 |
| 34000 | V17 exponent .5 | 505 | .398 | .269 |
| 34000 | V17 exponent 1 | 485 | .398 | .274 |
| 34000 | FineWeb | 2412 | .224 | .173 |
| 35000 | V2 | 309 | .396 | .269 |
| 35000 | V17 exponent 0 | 512 | .396 | .261 |
| 35000 | V17 exponent .5 | 508 | .396 | .261 |
| 35000 | V17 exponent 1 | 485 | .396 | .267 |
| 35000 | FineWeb | 2489 | .216 | .154 |

The immediate finding is that more content words do not change the dominant fixed-frame word share. The frequency exponent can alter content-word frequency and hapax counts, but it cannot address the overuse of `the`, frame cues, and repeated short clauses. The generator remains far from a realistic text distribution. This V17 variant is a **lexical stress-test axis**, not the replacement training corpus. Training another model solely on V17 would spend compute on a known form defect.

The next generator experiment should vary discourse acts and sentence realization within the same documents, while keeping the rules auditable. It should be compared against the existing V7 joint-task mixture on independent FineWeb form panels and against its own latent-rule checks. Once the joint DGP has demonstrably broader form, train a small architecture grid and evaluate each rule and combination behaviorally before testing weight methods. For those methods, keep reconstruction, task prediction, causal response, and genuine held-out-rule discovery separate. Hold out whole rule families or combinations from featurizer fitting, and compare methods at matched model fidelity and circuit budget.

Reproduce the lexical table with `build_typed_lexicon_v17.py` (optional `datasets` and NLTK at build time), the oracle checks with `check_surface_mixture_v17.py`, and the form screens with `audit_surface_mixture_v17_fineweb.py`. The JSON tables and form panels are versioned alongside these scripts. The curated semantic grouping was checked with Princeton WordNet; see its [license and attribution](https://wordnet.princeton.edu/license-and-commercial-use). No WordNet synset data is distributed here.
