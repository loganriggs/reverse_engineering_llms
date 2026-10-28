# Bilin18 layer-17 native MLP discriminator atlas

This is a first-pass input-selectivity audit of all 4,608 native bilinear product channels. Each channel computes `Left_j(x) * Right_j(x)` after attention 17 and RMS normalization. A token-class label is selected on disjoint discovery rows and its absolute-activation lift is evaluated on held-out rows.

## Aggregate

- Discovery tokens: 2,048,000; held-out tokens: 512,000.
- Left/right factors choose the same coarse class for 29.6% of all channels and 36.4% of the more usable half.
- Among the more usable half, median held-out class lift is 1.37; 36.5% exceed 1.5× and 14.3% exceed 2×.

- Their median held-out signed class-to-class gap is 0.55 product standard deviations; 19.4% exceed one standard deviation.

`Lift` is mean absolute product activation on the frozen preferred class divided by its mean over all held-out tokens. `Use score = product standard deviation × Down-column norm`; it is only a first-order availability score, not a causal effect on loss.

## Representative native channels

| unit | Left | Right | signed product separator | held-out gap (SD) | lift | strongest + / − contexts |
|---:|---|---|---|---:|---:|---|
| 1239 | punctuation | preposition | pronoun → + / preposition → − | 1.47 | 1.46 | + ` to use the host.[ We] present only this love with`; ` of participant or MS metaphor[ you] are contributing to arrive does`<br>− ` the Terran Dominion, opposed[ by] rebel groups, specifically Ray`; ` Sing Se comes under attack[ by] the Fire Nation and political` |
| 2563 | auxiliary | pronoun | capitalized → + / auxiliary → − | 1.52 | 1.36 | + `yana xxxl[\n]indian auntisex`; `://youtu.be/[6]CKddw--`<br>− ` (true pdf) nearest[!] perissodactyl`; `hrall baggily[?] determinate Waylin dec` |
| 137 | number | pronoun | conjunction → + / number → − | 1.48 | 1.74 | + ` just divorced dating site imposes[?] Smarter Merrick lesson`; ` bangalore dating and See[able] Tore stickybeaks`<br>− ` crude oil earns Vietnam US[$]54.4 on average`; ` 2017, bringing home US[$]3.17 billion.` |
| 3533 | pronoun | auxiliary | auxiliary → + / capitalized → − | 1.84 | 1.58 | + ` of Practices, who could[ elsewhere] instruct program's evidence.`; `linguistic figures we[ cannot] but be disciplines of an`<br>− `1896a, 143[).]\nChem. 5X`; `erall be filled early[.]Became thicker and thicker` |
| 644 | punctuation | pronoun | pronoun → + / preposition → − | 0.81 | 2.00 | + ` only available for Greek citizens[<|endoftext|>]- Open Door\n-`; `no trivia found]||[<|endoftext|>]Make it "Back to`<br>− ` site or used profanity[.] The princeton review offers`; ` help, and book publishing[.] What adjustments should i make` |
| 1811 | auxiliary | pronoun | auxiliary → + / punctuation → − | 1.49 | 1.41 | + ` of Practices, who could[ elsewhere] instruct program's evidence.`; ` just divorced dating site imposes[?] Smarter Merrick lesson`<br>− `://youtu.be/[6]CKddw--`; `://bit.ly/[2]WfnxhC` |
| 919 | determiner | auxiliary | punctuation → + / auxiliary → − | 1.66 | 1.47 | + `://youtu.be/[6]CKddw--`; ` http://youtu.be[/]6CKddw`<br>− `ions in practitioners which may[ finally] do seen mental.\n`; ` of Practices, who could[ elsewhere] instruct program's evidence.` |
| 3343 | preposition | pronoun | capitalized → + / auxiliary → − | 1.40 | 1.41 | + `://youtu.be/[6]CKddw--`; `://bit.ly/[2]gOKLL7\n`<br>− ` just divorced dating site imposes[?] Smarter Merrick lesson`; `hrall baggily[?] determinate Waylin dec` |
| 3237 | punctuation | pronoun | pronoun → + / punctuation → − | 1.39 | 1.56 | + `rail; pharmaceutical contented[ minutes] explain that 1990s lead`; ` a ardour for a[ many] testosterone to compensate the plan`<br>− ` site or used profanity[.] The princeton review offers`; ` safedisc backup capability[.] Eggplants are incredibly` |
| 3504 | punctuation | punctuation | punctuation → + / auxiliary → − | 1.29 | 4.52 | + ` and made it count...[<|endoftext|>]Read the many ways Russell`; `ha – SA Campsites[<|endoftext|>]About this paper\nSmart`<br>− ` the most important, up[-]to-date information right`; `.\nWe are n['t] learn your shop Escrit` |
| 1043 | punctuation | punctuation | punctuation → + / determiner → − | 1.57 | 3.76 | + ` AIR applications on Wednesday.[<|endoftext|>]Regular exercise, along with`; ` and made it count...[<|endoftext|>]Read the many ways Russell`<br>− `ें\n[�]�ाक�`; ` most magnificent and ...\n[�]�स्` |
| 2479 | pronoun | punctuation | auxiliary → + / capitalized → − | 1.62 | 1.49 | + ` lead for useful content of[ the] cialis offer h`; `, there will be first[ other] effects fantastic to symptoms abusing`<br>− ` site or used profanity[.] The princeton review offers`; ` help, and book publishing[.] What adjustments should i make` |
| 2902 | pronoun | preposition | punctuation → + / pronoun → − | 1.40 | 1.51 | + ` that sent a noticeable shudder[ through] his victim. Draike`; ` of Lily.\nA[ concerned] Chloe stopped by to see`<br>− ` you have protecting for Is[ also] Still. Since we only`; ` exact advice and how it[ Also] were Transcribed to Tibet` |
| 59 | pronoun | number | capitalized → + / number → − | 1.03 | 1.25 | + `s, Cheshire from[ B]ournes, Caer`; `alore dating and Seeable[ T]ore stickybeaks its`<br>− `ions in practitioners which may[ finally] do seen mental.\n`; `acy. This millennium does[ Furthermore] relevant and makes known to` |
| 1111 | auxiliary | pronoun | pronoun → + / capitalized → − | 1.31 | 1.68 | + ` lead for useful content of[ the] cialis offer h`; `avis staff writer wednesday[,] april 24, 2013`<br>− ` to be easy as wh[omever] the candidate is will only`; ` absolutely unique!! We cannot[ rate] Flawless highly enough` |
| 4094 | auxiliary | number | auxiliary → + / number → − | 2.26 | 1.99 | + `. wittier Juan[ita] fornicating her incrim`; ` imposes? Smarter Mer[rick] lesson, its very scient`<br>− ` a weekly performance of 4[.]08%. On July 8`; `1), 1996, pp[.] 4-6.\n` |
| 2701 | determiner | determiner | capitalized → + / determiner → − | 0.93 | 6.38 | + `ly spectre, but[ an] apparition full of mild`; ` instance, this one had[ an] awesome system of nets and`<br>− ` of an excellent imagination and[ a] steady and meticulous drawing hand`; ` and better things, including[ a] personal friendship with Louis XIV` |
| 4344 | punctuation | punctuation | punctuation → + / preposition → − | 1.26 | 3.77 | + ` determine assembly contestants.\n[||]Devote time in art`; `that features these books\n[||]Survey local card and`<br>− ` dates https://fors[omes]ex.com/bel`; `mierci: 7[ s]ierpnia 1992 Pop` |
| 1831 | punctuation | punctuation | subword → + / punctuation → − | 1.10 | 4.59 | + `-attorney-in[-]utah/<|endoftext|>One`; `-trial-attorney[-]in-utah/`<br>− ` starts to rise.\n["]Michigan Girls" (Calif`; `’s hospitals.[<|endoftext|>]This` |
| 1937 | punctuation | punctuation | conjunction → + / punctuation → − | 1.57 | 3.25 | + ` of the words that come[ to] mind.\n"Even`; ` the first words that come[ to] mind when you think of`<br>− ` and made it count...[<|endoftext|>]Read the many ways Russell`; `no trivia found]||[<|endoftext|>]Make it "Back to` |

## Red-team limits

- The ten token classes are deliberately coarse and incomplete. Low lift means only that this vocabulary failed, not that a channel is incoherent.
- Selection and top contexts come from discovery rows; only the class lift is held out.
- Layer 17 already includes attention-17 and all earlier blocks, so a channel can separate contextual states rather than current-token identity.
- The signed separator names are frozen on discovery data and evaluated as a standardized class-mean gap on held-out data. They remain coarse descriptions, not exhaustive meanings.
- Activation selectivity is descriptive. Finite channel ablations and cross-context transfer are required before assigning a causal verb.
- Native product channels have bilinear scaling and swap gauges. Interpret Left/Right labels as a pair, not as independently canonical neurons.
