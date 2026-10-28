# Held-out semantics of the dominant destination-Fisher mode

The leading candidate-space sensitivity mode is frozen from the earlier L3H8 Fisher panel, then applied to 512 positions from later FineWeb rows. Energy is next-token Fisher energy of the combined logit tangent; no semantic label is used to define the mode.

## Locked hypotheses

- **M1_dominant_mode_transfers_to_fresh_text: supported.**
- **M2_mode_energy_has_a_known_semantic_condition: not supported.**
- **M3_signed_output_poles_are_lexically_coherent: supported.**
- **M4_mode_is_not_only_one_outlier_context: supported.**

Fresh-panel Fisher energy captured: `0.694`. Largest single-site share: `3.898%`.

Top known next-token enrichments:

- `suffix_s`: 20 sites, energy ratio `1.29`
- `lowercase`: 348 sites, energy ratio `1.29`
- `alphabetic`: 398 sites, energy ratio `1.23`
- `leading_space`: 373 sites, energy ratio `1.12`
- `punctuation`: 69 sites, energy ratio `1.09`
- `word_continuation`: 35 sites, energy ratio `1.09`
- `auxiliary`: 23 sites, energy ratio `1.09`
- `determiner`: 44 sites, energy ratio `1.07`
- `period`: 28 sites, energy ratio `0.90`
- `pronoun`: 51 sites, energy ratio `0.83`
- `preposition`: 43 sites, energy ratio `0.82`
- `capitalized`: 50 sites, energy ratio `0.79`

Positive pole (lowercase, purity 0.31): `['.")', '").', '.)', '—', 'qu', '"—', '):', ')—', ' Dickens', ' invariably', 'ib', ' literary', 'in', 'de', ' ("', 'ch']`
Negative pole (suffix, purity 0.75): `[' again', ' allowing', ' continue', ' continues', ' causing', ' willing', 'izons', ' continued', ' continuing', ' Lets', ' completing', ' wanting', ' Again', ' knowing', ' allow', ' freeing']`
Largest-RMS tokens: `[' murm', ' answered', ' muttered', ' spat', ' whispers', ' observed', 'answered', 'att', 'aff', ' Cas', ' mut', 'own', 'so', ' whistle', ' countered', ' swallowed']`

Highest-energy contexts:

- ` "Diana," she said after some obvious thought. "If we're fighting...I don't believe I'm saying this` → `,`; energy `3.279e+00`
- ` sicced hellhounds on us and I froze up then, too. One of them had me and I froze up` → `.`; energy `1.862e+00`
- ` the floor and looked around, trying in vain to find anything to brace the door with. "There's a window," Cast` → `iel`; energy `1.846e+00`
- `."\n"Those aren't the only options." She took two steps to the right, keeping eye contact with Xaphan` → ` and`; energy `1.740e+00`
- ` "Lucifer's demon bitch lackey sicced hellhounds on us and I froze up then, too. One` → ` of`; energy `1.675e+00`
- `...." Dean felt silly even thinking about saying it out loud. "She says you and her went AWOL for the same` → ` reasons`; energy `1.557e+00`
- `, Sam, that's what matters." Sam didn't need to worry about this; by the time he saw Sam again Cas` → ` would`; energy `1.534e+00`
- `'re honest about it. If you're...hunting in my jurisdiction," he said, raising one eyebrow, "at least` → ` give`; energy `1.272e+00`
- ` handing it to Castiel. "Here," he said, "I'm guessing you want this back---"\nThe sword` → ` cl`; energy `1.241e+00`
- ` cold fingers wrap around his, squeezing so hard he felt his fingers start to go numb. "Stay with me, Cas,"` → ` Dean`; energy `1.190e+00`
- ` felt it like it had scored down his own side. Dean wanted to move, wanted to help but it was like it was` → ` all`; energy `1.023e+00`
- ` air. "Can you see them?"\n"A gun's not going to do anything."\n"Can you see` → ` them`; energy `9.986e-01`
- ` was looking at her now. It was the way Sam looked at him when hunts went bad, the way Sam had looked to` → ` him`; energy `9.097e-01`
- ` is our fucking lives lately," he said, his hands balling into fists. "Lucifer's demon bitch lackey s` → `ic`; energy `9.038e-01`
- `." He found another faint scratch on Cas' neck and stopped himself from visualizing how bad that could have been had the claw` → ` gone`; energy `8.569e-01`
- `." There was the same far-off look in her eyes Dean knew was in his own when he thought about Carthage and` → ` Cold`; energy `7.126e-01`

## Guardrails

This is a semantic audit of a destination-sensitivity mode, not a native neuron. Feature ratios are exploratory after the mode was frozen and are not multiple-comparison corrected. Mean logit poles can be affected by vocabulary frequency and unembedding anisotropy. A readable label requires replication or an intervention selected from these examples; unexplained but stable remains an acceptable outcome.
