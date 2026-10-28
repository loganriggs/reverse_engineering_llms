# Variable/function atlas for early attention

The human token class is treated as a source variable. `(layer, head)` is treated as a native value/write function, while distance and category determine its route invocations. Each function is represented weight-only by its exact current-value OV map `W_O,h W_V,h`; route selection uses the frozen operator-effect score on FineWeb.

## Locked hypotheses

- **F1_variables_are_reused_across_functions: not supported.**
- **F2_functions_are_reused_across_variables: supported.**
- **F3_same_variable_uses_distinct_transforms: supported.**
- **F4_routing_similarity_predicts_transform_similarity: supported.**
- **F5_native_head_modes_are_lexically_readable: not supported.**

## Main measurements

- Median effective functions per variable: `3.88`.
- Median variables per used function: `6.0`.
- Median within-variable absolute OV cosine: `0.0080`.
- Route-profile/OV-similarity Spearman correlation: `0.507`.
- Post-hoc layer-control red team: within-layer `0.497`, cross-layer `0.514`, median layer-pair `0.546`.
- Leading-mode pole category purity: median `0.500`, random p95 `0.625`.

## Representative high-mass native functions

- **L0H3** reads `newline, non_ascii, numeric, period, comma, colon_semicolon, question_exclamation, opening_delimiter, closing_delimiter, quote, dash, slash, code_operator, programming_keyword, function_word, suffix, capitalized, lowercase, other_punctuation, other`; input + `[' technicians', ' processors', ' athletes', ' vessels', ' soldiers']`, input − `['Effective', 'irable', 'rehensible', 'Critical', ' Critical']`; output + `[' their', ' perfection', ' land', ' geographic', ' BMI']`, output − `['upid', ' Curve', ' loop', ' paradigm', 'umption']`.
- **L0H8** reads `newline, numeric, opening_delimiter, closing_delimiter, programming_keyword, function_word, suffix, capitalized, lowercase, other_punctuation, other`; input + `[' gives', ' give', ' giving', ' gave', ' Giving']`, input − `[' made', ' makes', ' make', ' making', 'made']`; output + `[' administer', ' lose', ' interfere', ' meddling', ' shouting']`, output − `[' searches', 'umer', ' eve', ' watches', ' refriger']`.
- **L2H6** reads `newline, non_ascii, numeric, period, comma, colon_semicolon, closing_delimiter, dash, slash, code_operator, programming_keyword, function_word, suffix, capitalized, lowercase, other_punctuation, other`; input + `[' environments', 'riages', ' Walls', 'cers', ' tiers']`, input − `[' involving', ' of', ' with', ' robotic', ' into']`; output + `['ilaterally', 'cerning', 'annot', 'matically', 'mitted']`, output − `[' syndrome', 'pathic', ' Syndrome', 'bin', ' bod']`.
- **L0H7** reads `newline, non_ascii, numeric, period, question_exclamation, dash, function_word, suffix, capitalized, lowercase, other_punctuation`; input + `[' additionally', ' likewise', ' residing', ' utilized', ' utilizing']`, input − `[' Thatcher', ' Quinn', ' Reagan', ' 2012', 'Charlie']`; output + `[' his', ' Storm', ' He', ' yet', ' re']`, output − `[' obtain', ' ascert', ' acquire', ' Apply', ' comply']`.
- **L3H6** reads `newline, non_ascii, numeric, period, comma, colon_semicolon, opening_delimiter, dash, slash, code_operator, programming_keyword, function_word, suffix, capitalized, lowercase, other_punctuation, other`; input + `[' composure', ' Himself', ' gently', ' whites', ' who']`, input − `[' involving', ' regarding', ' with', ' using', ' ruining']`; output + `[' wave', ' waves', ' bullet', ' talk', ' local']`, output − `['tained', 'only', ' vertex', 'ented', 'vised']`.
- **L2H2** reads `non_ascii, numeric, period, comma, colon_semicolon, opening_delimiter, dash, programming_keyword, function_word, suffix, capitalized, lowercase`; input + `[' have', ' delivers', ' withheld', ' deliver', ' withholding']`, input − `[' from', ' visited', ' told', 'At', ' appease']`; output + `[' "', '"', 'ionic', ' es', ' sh']`, output − `[' region', ' comprehend', ' Cyn', ' McGu', ' Alliance']`.
- **L1H1** reads `newline, non_ascii, numeric, period, comma, colon_semicolon, question_exclamation, opening_delimiter, closing_delimiter, quote, dash, slash, code_operator, programming_keyword, function_word, suffix, capitalized, lowercase, other_punctuation, other`; input + `['�', 'ocry', 'cffff', 'ocl', 'о�']`, input − `[',', ' –', ' ,', ' —', '),']`; output + `[' just', ' that', ' in', ' as', ' to']`, output − `['ギ', ' defe', 'tnc', ' hemor', ' juven']`.
- **L5H7** reads `newline, non_ascii, numeric, period, comma, opening_delimiter, dash, code_operator, programming_keyword, function_word, suffix, capitalized, lowercase, other_punctuation, other`; input + `['�', ' cour', 'exc', ' Games', ' explorers']`, input − `[' Poor', ')=(', './', '&', ' "(']`; output + `[' myster', ' Berserker', ' defe', ' nomine', ' srf']`, output − `[' in', ' and', ' on', ' all', ' (']`.
- **L0H6** reads `newline, non_ascii, numeric, period, opening_delimiter, quote, dash, slash, code_operator, function_word, capitalized, lowercase, other_punctuation, other`; input + `['Mrs', ' Mrs', ' vs', '…………', 'Sept']`, input − `[' eleven', 'TG', 'AB', ' ATI', ' AA']`; output + `['Jun', 'ains', ' blockers', ' Avg', 'epad']`, output − `[' You', ' We', ' When', ' In', ' It']`.
- **L3H4** reads `numeric, period, comma, programming_keyword, function_word, suffix, capitalized, lowercase`; input + `[' guy', ' he', ' anyone', ' husband', ' minister']`, input − `[' collisions', 'appings', ' massacres', ' explosions', ' LSD']`; output + `[' He', ' his', 'He', ' himself', 'His']`, output − `[' lim', ' notably', 'ATTLE', 'nexpected', ' necess']`.
- **L4H1** reads `comma, programming_keyword, function_word, suffix, capitalized, lowercase`; input + `[' had', ' confronting', ' encountering', ' juggling', ' have']`, input − `[' is', ' be', ' Be', ' was', ' are']`; output + `[' aware', 'ologically', ' unable', 'atible', 'bed']`, output − `['protection', ' slots', 'ousands', 'appiness', 'links']`.
- **L4H5** reads `newline, numeric, function_word, suffix, capitalized, lowercase`; input + `['Children', ' children', ' baby', ' pet', ' Children']`, input − `[' economists', ' experts', ' regulators', ' psychiatrists', ' psychologists']`; output + `[' natural', ' gains', ' loved', ' undergo', ' undergoing']`, output − `[' oversee', ' overseeing', ' oversees', ' scathing', ' appoint']`.

## Guardrails

A native head is a convenient exact function boundary, not necessarily an MDL-optimal function. The OV map omits both squared-QK routing factors, RMS normalization, the persistent Layer-0 value bus, and downstream nonlinear computation. Token decoding of singular vectors is descriptive and can miss a distributed semantic direction. Low OV cosine establishes distinct linear transforms, not distinct human meanings. The layer-controlled correlations are post-hoc red teams; if they collapse, the preregistered pooled correlation was a layer-identity confound. Causal output-vector interventions are the next validation step.
