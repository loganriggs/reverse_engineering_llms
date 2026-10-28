# Semantic atlas of the compact router graph

The atlas audits 96,000 held-out FineWeb token sites. Thirty-two weight-canonicalized coordinates and the pooled 32-edge graph are compared with random rotations/edges. Candidate labels are selected on the first half and evaluated on the second; tail contexts are descriptive held-out examples.

## Locked hypotheses

- **I0_large_panel_completed: supported.**
- **I1_canonical_axes_have_simple_token_semantics: supported.**
- **I2_canonical_axes_beat_random_rotations: not supported.**
- **I3_selected_edges_have_simple_token_semantics: supported.**
- **I4_selected_edges_beat_random_edges_semantically: supported.**
- **I5_tail_examples_are_not_single_token_detectors: supported.**

## Most label-readable coordinates

| Axis | Degree | Held-out label / AUROC | Top-100 token support | Positive example | Negative example |
|---:|---:|---|---:|---|---|
| 6 | 7 | current:newline / 0.996 | 1.0 | `ax Springs MO Homes For Sale[<|endoftext|>]Best Dating Sites & Apps For` | ` | Clinical | Educational | Industrial[ |] Professional items | World psychology |` |
| 27 | 7 | current:newline / 0.987 | 15.3 | ` presented in a summary fashion,[ and] is intended to be used for` | ` on Zappos," said[ Peter] Lee, executive director of the` |
| 1 | 10 | current:newline / 0.987 | 2.5 | ` switching|\n|US62[16]174||Sep 29, 1998` | ` Allan Fung of Cranston[ and] Scott Avedisian of` |
| 12 | 7 | current:newline / 0.958 | 64.1 | ` with internet connection. Become happy[,] you could reach tens of thousands` | ` & DISTRIBUTION MANAG[EMENT]\nJonathan engages widely with industry` |
| 16 | 0 | current:newline / 0.951 | 10.4 | ` helped lead Illinois Central to a[ 32]-3 record and a third` | `restrial life, in locations such[ as] the bodies of water thought to` |
| 11 | 0 | current:newline / 0.905 | 67.6 | ` ago, the School Board had[ directed] him not to handle routine legal` | ` hike at this impressive 1,[600]-hectare nature preserve` |
| 2 | 7 | next:newline / 0.889 | 40.3 | `. Blackjack app or tablet[.] Bovada s country b` | ` leading to serious health consequences that[ could] have been easily avoided. Hence` |
| 7 | 0 | current:newline / 0.874 | 15.6 | ` learn not Still been. If[ they] influence in your globe, they` | ` a military clinic or hospital,[ received] a letter earlier this year explaining` |
| 25 | 3 | next:comma_period / 0.870 | 8.8 | ` potential when paired with bubbles and[ booze]:\nAntique Bakery` | ` front of him; the driver[ was] fined $20.\nResp` |
| 18 | 1 | current:opening_delimiter / 0.862 | 9.0 | ` upon during future days when life[ becomes] difficult.\nIn Heaven this` | `. In all, sales totaled[ $]680,000.\n"` |
| 14 | 1 | next:suffix_ed / 0.847 | 3.3 | `A horse from Montana that has[ never] had EEE/WEE` | ` had skippered his country[ to] World Cup glory two years earlier` |
| 26 | 0 | current:newline / 0.826 | 45.5 | ` billed the district $102,[896] for legal work on charter schools` | `irteenth-century, Northern[ Italian] fragment of the Arthurian legend` |

## Most label-readable selected interactions

| Edge | Held-out label / AUROC | Support | Raw promoted tokens | Positive example |
|---|---|---:|---|---|
| 0×6 | current:newline / 0.996 | 1.0 | `GGGGGGGG`, ` 4090`, ` \xa0 \xa0`, ` UNCLASSIFIED`, `20439` | `ax Springs MO Homes For Sale[<|endoftext|>]Best Dating Sites & Apps For` |
| 6×15 | current:newline / 0.995 | 1.4 | ` horr`, ` blurred`, ` doomed`, ` presided`, ` crippled` | ` (4)|\n|External[ Links]: USPTO, US` |
| 4×6 | current:newline / 0.993 | 3.3 | ` share`, ` feed`, ` lean`, ` guaranteed`, ` clear` | `, Woodside acupunct[ur]ists, Kittybrewster ac` |
| 1×6 | current:newline / 0.992 | 1.0 | `skip`, `prev`, `Pok\xe9mon`, `moderate`, `Haunted` | `ax Springs MO Homes For Sale[<|endoftext|>]Best Dating Sites & Apps For` |
| 0×1 | current:newline / 0.990 | 1.0 | `echo`, `prototype`, `failed`, `instance`, `missing` | ` https://youtu.be/[DJ]QnJ2J27` |
| 12×27 | current:newline / 0.989 | 50.0 | ` subscript`, ` colle`, ` Okin`, `i`, ` departed` | `state. wittier Juan[ita] fornicating her incriminating` |
| 4×27 | current:newline / 0.988 | 41.3 | `wolves`, ` subscript`, ` unbeaten`, `men`, ` coli` | `, Managing Partner, Fich[te] & Co.; Vuay Kam` |
| 0×27 | current:newline / 0.987 | 20.8 | ` non`, ` exact`, ` question`, ` mostly`, ` big` | ` fix Apple Push Notification issue.)[\n]0). I got a simple` |
| 15×27 | current:newline / 0.983 | 13.3 | ` Telescope`, ` Barrier`, ` Warfare`, ` Disorder`, ` Traff` | ` President of Industry and Regulatory Affairs[ Sean] Elliott, and I participated.` |
| 1×15 | current:newline / 0.981 | 8.2 | `prototype`, `multipl`, `errors`, `successfully`, `echo` | ` at the time," said Purdue[ spokeswoman] Liz Evans.\nInstead,` |
| 1×25 | current:newline / 0.974 | 6.5 | `dylib`, `uyomi`, `iterator`, `GGGGGGGG`, `awaru` | ` 993|<|endoftext|>Fears[ that] the fiscal cliff could send the` |
| 1×4 | current:newline / 0.973 | 10.7 | ` UNCLASSIFIED`, ` newsp`, `dylib`, ` Hait`, ` Pelicans` | ` not want to become formally involved[ with] this film, given its understanding` |
| 6×12 | current:newline / 0.959 | 6.0 | `dylib`, `byss`, `ailability`, `ahime`, `shapeshifter` | ` if the league as a whole[ was] losing money , but it\ufffd` |
| 4×12 | current:newline / 0.957 | 67.6 | ` subscript`, ` unbeaten`, `matically`, ` coli`, `?:` | ` is 197 persons per square kilomet[er]. The adult literacy rate is` |
| 0×12 | current:newline / 0.957 | 79.4 | ` Ex`, ` Water`, ` Hol`, ` non`, ` big` | ` with internet connection. Become happy[,] you could reach tens of thousands` |
| 1×18 | current:opening_delimiter / 0.941 | 8.2 | `GGGGGGGG`, ` looph`, `bryce`, `awaru`, `STDOUT` | `\nPromises of Jobs and[ "]Transparency" Helped ALEC` |

A high category AUROC gives a simple descriptive handle, not a complete meaning. A coordinate can participate in several edges with different output directions; an edge can be a clean relational verb even if neither endpoint is independently monosemantic. Raw output tokens use the unembedding without the context-dependent final RMSNorm and are therefore role hints, not exact causal effects.
