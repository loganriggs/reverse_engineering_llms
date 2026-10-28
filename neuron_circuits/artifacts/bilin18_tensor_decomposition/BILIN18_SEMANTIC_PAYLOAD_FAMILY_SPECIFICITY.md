# Semantic-family specificity of weight-derived payload modes

Each selected head program and its previously frozen k90 OV basis are held fixed while the cue edit family is varied among spelling, sentiment, gender, and tense. State fidelity tests whether the basis generically compresses that head's attention change; output fidelity tests the declared edit family's semantic contrast; branch projection measures how much the fixed program contributes to the complete native semantic edit.

## Locked decisions

- **G0_semantic_edits_prefer_their_own_payload_modes: supported.** Observed: `{'spelling': 0.3575030274306533, 'sentiment': 0.12014352823148755, 'gender': 0.4569188912779001, 'tense': 0.45091199236786283}`
- **G1_semantic_output_reconstruction_is_family_specific: supported.** Observed: `{'spelling': 0.5260263742724154, 'sentiment': 0.45973794186215994, 'gender': 0.8602175909764478, 'tense': 0.49994621402983064}`
- **G2_payload_modes_are_generic_operator_compressors: not supported.** Observed: `{'spelling': 0.48678270868851226, 'sentiment': 0.5477918982706147, 'gender': 0.32276023568776036, 'tense': 0.4686894716968071}`
- **G3_head_invocation_is_family_selective: supported.** Observed: `{'spelling': 0.18524088819773085, 'sentiment': 0.040504494838300736, 'gender': 0.4290055471437457, 'tense': 0.3136860715951864}`
- **G4_state_and_output_specificity_agree: supported.** Observed: `0.8643526430118559`

## State fidelity

| Program \ edit | spelling | sentiment | gender | tense |
|---|---:|---:|---:|---:|
| spelling | 0.960 | 0.602 | 0.408 | 0.487 |
| sentiment | 0.516 | 0.691 | 0.571 | 0.548 |
| gender | 0.219 | 0.526 | 0.983 | 0.323 |
| tense | 0.449 | 0.469 | 0.503 | 0.954 |

## Semantic-output fidelity

| Program \ edit | spelling | sentiment | gender | tense |
|---|---:|---:|---:|---:|
| spelling | 0.999 | 0.294 | 0.072 | 0.473 |
| sentiment | 0.469 | 0.949 | 0.316 | 0.490 |
| gender | 0.116 | -0.009 | 1.000 | 0.139 |
| tense | 0.498 | 0.279 | 0.309 | 0.998 |

## Branch projection onto complete native semantic effect

| Program \ edit | spelling | sentiment | gender | tense |
|---|---:|---:|---:|---:|
| spelling | 0.193 | 0.007 | 0.004 | 0.003 |
| sentiment | 0.000 | 0.112 | 0.002 | 0.071 |
| gender | 0.012 | 0.009 | 0.441 | 0.002 |
| tense | 0.000 | 0.000 | 0.000 | 0.314 |

## Interpretation boundary

If a head's top OV modes compress every edit equally, they are generic operator coordinates even if one supervised invocation is semantic. Diagonal state specificity instead means that the target edit preferentially occupies those weight modes. Branch projection is a separate gate: an accurately compressed head can remain behaviorally irrelevant for a given edit.
