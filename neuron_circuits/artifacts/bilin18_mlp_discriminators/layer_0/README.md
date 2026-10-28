# Bilin18 layer-0 native MLP discriminator atlas

This is a first-pass input-selectivity audit of all 4,608 native bilinear product channels. Each channel computes `Left_j(x) * Right_j(x)` after attention 0 and RMS normalization. A token-class label is selected on disjoint discovery rows and its absolute-activation lift is evaluated on held-out rows.

## Architecture correction: the embedding is reinjected

Every block first forms `r_l = alpha_l * x_l + beta_l * x0`, where `x0` is the
RMS-normalized token embedding. The checkpoint's learned `(alpha, beta)` values
are `(6.094, 6.094)` in block 0; in blocks 1--17, `beta` is 5.063--8.000 while
`alpha` is 0.013--1.977. These raw coefficients are not variance shares because
the mixture is subsequently RMS-normalized and the propagated stream can have a
different norm. They nevertheless establish a direct lexical pathway into every
block.

This does not invalidate the layer-0 audit. In block 0, `x_l = x0`, so the two
terms are collinear and RMS normalization cancels their common positive scale.
For later layers, the reinjected embedding must be separated from propagated
context rather than treating the MLP input as an undifferentiated residual.

For native bilinear channel `j`, ignoring only the common positive RMS scale for
one token position and writing the MLP input as `alpha*c + beta*e`, the weights
give an exact decomposition:

`(L_j x)(R_j x) = alpha^2 (L_j c)(R_j c)
+ alpha*beta[(L_j c)(R_j e) + (L_j e)(R_j c)]
+ beta^2 (L_j e)(R_j e)`.

These are respectively context-only, embedding/context interaction, and
embedding-only terms. This is the next weight-based audit. It can reveal what a
channel is capable of separating directly from token identity, but empirical
activations and finite interventions are still needed to establish which terms
the model actually uses in context and how they affect loss.

## Aggregate

- Discovery tokens: 20,480; held-out tokens: 10,240.
- Left/right factors choose the same coarse class for 41.5% of all channels and 50.5% of the more usable half.
- Among the more usable half, median held-out class lift is 1.83; 72.8% exceed 1.5× and 40.1% exceed 2×.

- Their median held-out signed class-to-class gap is 0.88 product standard deviations; 40.4% exceed one standard deviation.

`Lift` is mean absolute product activation on the frozen preferred class divided by its mean over all held-out tokens. `Use score = product standard deviation × Down-column norm`; it is only a first-order availability score, not a causal effect on loss.

## Representative native channels

| unit | Left | Right | signed product separator | held-out gap (SD) | lift | strongest + / − contexts |
|---:|---|---|---|---:|---:|---|
| 1737 | determiner | determiner | determiner → + / capitalized → − | 2.41 | 3.50 | + ` Black Grand Champion Male form[ the] Hemingway & Bu`; ` approximate sound made on board[ a] US submarine as it rises`<br>− ` on your far-sighted[ leading] asticle of May I`; ` 6 south [Mon.-[Fri].] to 56th Street` |
| 2264 | conjunction | conjunction | capitalized → + / conjunction → − | 1.35 | 1.88 | + ` Infants' School then[ Brisbane] Girls' Grammar School`; `-led perinatal[ hosp]ice and infant care system`<br>− `[ when], in his words,`; `[ and] an airplane manufacturing facility.` |
| 3326 | capitalized | capitalized | punctuation → + / capitalized → − | 2.14 | 5.08 | + ` Rush University Medical Center Alzheimer[�]�s Disease Center made`; ` airplane manufacturing facility. That[�]�s functional. But`<br>− ` Elimination of Stress will[ H]alt the progression\nof`; ` On Behalf Of Tom[ H]ightower\nSent:` |
| 1492 | determiner | determiner | capitalized → + / determiner → − | 2.17 | 3.26 | + ` you can read Katie Cro[pper]'s tidy summary of it`; `- published: 19[ Mar] 2013\n- views:`<br>− `adiators who fight to[ the] death. The Games are`; ` end up siding with[ the] latter as part of its` |
| 3926 | capitalized | capitalized | capitalized → + / content_word → − | 2.03 | 4.08 | + ` hotel bookings.\n[T]UNE IN: The mobile`; `MUP), 2005\n[E]leanor Constance (E`<br>− ` left, Triche sw[ished] a pair of free throws`; ` supply of cigarettes, pepp[ered] sometimes with wine, they` |
| 3722 | capitalized | pronoun | capitalized → + / punctuation → − | 0.75 | 2.92 | + `[How] AP reported in all formats`; `[Five] Reasons I Love Boston\n`<br>− ` for Super Tuesday, booked[ several] flights to the cities closest`; `century Europe and then clamp[ down] on Europe for resisting the` |
| 3574 | punctuation | punctuation | punctuation → + / preposition → − | 1.27 | 4.38 | + ` boost voter participation among troops[.]\nThe blame for the`; ` for preventive action among others[.]\n"If there was`<br>− ` subsequent travels.\n3[.] Clearing Israel Custom\n`; ` I Love Boston\n1[.] The water.\nThe` |
| 3069 | content_word | content_word | content_word → + / subword → − | 1.83 | 1.76 | + `Below, I essentially[ replicate] his results with the purpose`; ` himself away from the six[ household] dogs), you can pick`<br>− ` Rush University Medical Center Alzheimer[�]�s Disease Center made`; `. ET, bringing users[�]� streams to a scree` |
| 3389 | content_word | content_word | content_word → + / preposition → − | 1.61 | 2.61 | + ` The surf rolls in with[ t]umbling white crests over`; `�s not uncommon for[ bl]enders to base their recipes`<br>− ` was "Regarding James We[ldon] Johnson."\n- Created`; ` Arctic Biology at the University[ of] Alaska Fairbanks, R` |
| 3120 | punctuation | punctuation | punctuation → + / preposition → − | 1.86 | 4.33 | + `MAGIC NUMBERS[ (]sport)* Steve McCl`; `24[/]7 culture, autogenic`<br>− ` newest book "Scra[ps] & Shirtails 2"`; `[The] first day of spring was` |
| 906 | determiner | determiner | content_word → + / determiner → − | 2.59 | 5.41 | + ` a portable alternative to a[ land]line. It can have`; ` their jobs – and a[ much] worse education for students and`<br>− ` how we sleep, and[ our] feelings about ourselves as a`; ` make her more receptive to[ his] advances in the future.` |
| 3128 | auxiliary | auxiliary | determiner → + / auxiliary → − | 0.65 | 3.24 | + ` one of many of Bout['s] front companies -- companies that`; ` the public interest of Hawaii['s] residents because (agencies`<br>− `IRVS)?\nIt['s] worth a quick review.`; ` exhalation. It['s] typical of Crumb's` |
| 874 | punctuation | number | punctuation → + / auxiliary → − | 0.06 | 1.94 | + `ican America - Introduction\n["]Mexican America" is`; `ler Viktor Bout.\n["]The Gambia is the`<br>− ` as agents of "illegal["] government authority. Further,`; ` conditions classified as "natural["] pine forest. While` |
| 597 | determiner | determiner | determiner → + / number → − | 2.37 | 3.80 | + ` Hotel Corp. is negotiating[ the] purchase of a Near North`; ` the House begins to debate[ a] major overhaul of the campaign`<br>− `. Razib concludes:[\n]The general qualitative result:`; ` immigration to the cities […][\n]I wasn’t` |
| 3844 | pronoun | pronoun | punctuation → + / pronoun → − | 0.73 | 3.01 | + ` demonstrates. Featuring Steve on[ his] many 9a’`; `, the FIGER on[ his] underwear waistband peeping`<br>− ` to familiarize her with[ his] scent and perhaps make her`; ` Stiles was understanding,[ he] was too nice for his` |
| 4099 | punctuation | punctuation | preposition → + / punctuation → − | 0.75 | 3.34 | + ` only attempts of the game[ —] and Syracuse held on for`; ` region surrounding modern Mexico City[—]was known`<br>− ` adopt a more multicultural mindset[ --] and that includes increased multicultural`; ` a pair of free throws[ —] his only attempts of the` |
| 1642 | number | number | number → + / pronoun → − | 3.27 | 10.39 | + `-foot-3 and[ 215]-pound Rivals100 member`; ` 2013\n- views:[ 42]\n- author: T`<br>− ` Broncos released the nine-[year] veteran Monday on the eve`; ` rowdy little five-[year]-old, always bouncing` |
| 3088 | number | number | determiner → + / number → − | 2.17 | 9.42 | + ` confirmations back to the[ message] bus. This way subscribers`; `mites de Méx[ico] han ido c`<br>− ` 353-5279x[516]; 516 Hill Hall`; `istserv: lynch[560] @ andromeda.r` |
| 3305 | number | number | preposition → + / number → − | 2.87 | 9.04 | + `. He was born Sept[.] 8, 1929, in`; ` 12-ounce cotton sand[stone] duck fabric, this Car`<br>− ` victory and the score tied[ 61]-all with 88 seconds`; `\nForest Glen Morse,[ 81], of Harbor Springs,` |
| 1107 | number | number | number → + / punctuation → − | 3.23 | 8.35 | + `World leaders from[ 47] nations convene this week`; ` Production: France currently counts[ 58] commercial nuclear reactors in operation`<br>− ` is AP's official Spanish[-]language word for what many`; ` week a maid would grim[ace] over how they had worn` |

## Red-team limits

- The ten token classes are deliberately coarse and incomplete. Low lift means only that this vocabulary failed, not that a channel is incoherent.
- Selection and top contexts come from discovery rows; only the class lift is held out.
- Layer 0 already includes attention-0 context, so a channel can separate contextual states rather than current-token identity.
- The signed separator names are frozen on discovery data and evaluated as a standardized class-mean gap on held-out data. They remain coarse descriptions, not exhaustive meanings.
- Activation selectivity is descriptive. Finite channel ablations and cross-context transfer are required before assigning a causal verb.
- Native product channels have bilinear scaling and swap gauges. Interpret Left/Right labels as a pair, not as independently canonical neurons.
