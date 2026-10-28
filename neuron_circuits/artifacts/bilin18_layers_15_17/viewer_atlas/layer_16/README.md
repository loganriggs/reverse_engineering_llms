# Bilin18 layer-16 native MLP discriminator atlas

This is a first-pass input-selectivity audit of all 4,608 native bilinear product channels. Each channel computes `Left_j(x) * Right_j(x)` after attention 16 and RMS normalization. A token-class label is selected on disjoint discovery rows and its absolute-activation lift is evaluated on held-out rows.

## Aggregate

- Discovery tokens: 2,048,000; held-out tokens: 512,000.
- Left/right factors choose the same coarse class for 30.7% of all channels and 35.6% of the more usable half.
- Among the more usable half, median held-out class lift is 1.30; 26.0% exceed 1.5× and 7.5% exceed 2×.

- Their median held-out signed class-to-class gap is 0.40 product standard deviations; 9.7% exceed one standard deviation.

`Lift` is mean absolute product activation on the frozen preferred class divided by its mean over all held-out tokens. `Use score = product standard deviation × Down-column norm`; it is only a first-order availability score, not a causal effect on loss.

## Representative native channels

| unit | Left | Right | signed product separator | held-out gap (SD) | lift | strongest + / − contexts |
|---:|---|---|---|---:|---:|---|
| 2098 | number | pronoun | pronoun → + / preposition → − | 0.89 | 1.27 | + ` be used for educational and[ entertainment] purposes only. It is`; `. Please contact your trusted[ healthcare] practitioner for direct product recommendations`<br>− `-solving.\n[-] Connect communities, people,`; `, as Detroit Board Set[ To] Vote on Plan 'To` |
| 3031 | punctuation | pronoun | preposition → + / punctuation → − | 1.35 | 1.81 | + `<|endoftext|>Report: News mixed[ on] Charlotte for first half of`; `, and the SKY[ that] under the extra blog of`<br>− ` would lead when he ...[<|endoftext|>]I have sent more than`; ` pace and more pace.[<|endoftext|>]Have you tried zood` |
| 4364 | pronoun | punctuation | pronoun → + / punctuation → − | 1.85 | 1.84 | + `attacked, his violence[ is]`; `, his dating rules he[ is] seeing other people published poly`<br>− ` ing on the receptor surface[.] HotsenpillerG`; ` join on a trial period[.]<|endoftext|>For decades, small` |
| 2926 | auxiliary | auxiliary | capitalized → + / auxiliary → − | 1.72 | 1.58 | + ` my Strategies out there.[\n]In our analysis, we`; `US7881244[||]May 17, 2009||`<br>− ` page.<|endoftext|>Originally Posted[ by] DougZ73\nSince`; ` nuns. The most irrit[able] devil of Irvin his` |
| 799 | preposition | pronoun | subword → + / preposition → − | 0.89 | 1.47 | + ` her deputy Edwina[ (]Kirsten Dunst`; ` their production manager Shelli[ (]Wayward Pines�`<br>− `pired with co-def[endants] Gabriel Doyle Laskey,`; `A pained expression came[ into] her eyes for a moment` |
| 2536 | punctuation | punctuation | punctuation → + / auxiliary → − | 1.75 | 3.21 | + ` - Chris Gray[<|endoftext|>]Thursday, April 9,`; ` the redesign that never ended[<|endoftext|>]I don’t`<br>− ` Medical marijuana will certainly remain[ to] expand sought after and more`; `.\nare you are[ to] select how to be Microsoft` |
| 3043 | punctuation | punctuation | punctuation → + / pronoun → − | 1.67 | 3.28 | + `Massive Load Pills[\n]Please feel free to taste`; ` requested other members of team[\n]Promote positive attitude and`<br>− `ritos is our crops[,] the Post feature will seek`; `, up to this point[,] has chosen not to receive` |
| 769 | pronoun | auxiliary | punctuation → + / auxiliary → − | 1.83 | 1.63 | + ` feeding experience for a[.] Saver high chair to`; `15, (202)[ 6]90-7850\n`<br>− `avis staff writer wednesday[,] april 24, 2013`; `yles socrates fresco[ from] ancient ephesus the` |
| 1388 | conjunction | auxiliary | auxiliary → + / punctuation → − | 1.77 | 1.35 | + ` the tour I would have[ been] much`; ` monsoon, chocolate lovers[ are] always faced with the tricky`<br>− `Spector, D.[\n]Journal cara memb`; `ax Bagus Mana?[\n]Egil nodded Then His` |
| 3467 | preposition | pronoun | pronoun → + / preposition → − | 0.90 | 1.25 | + ` questions regarding orders or our[ policies] and programs, please contact`; ` information and credit card is[ encrypted].\nWe have put`<br>− ` Arts grant helped established perennial[ recipients] like the Krampus`; `iding over the city walls[.] Be sure to include your` |
| 2687 | auxiliary | auxiliary | auxiliary → + / number → − | 1.92 | 1.71 | + `linguistic figures we[ cannot] but be disciplines of an`; ` you feel not may n['t] influence contemporary of your Converted`<br>− `://bit.ly/[2]WfnxhC`; `://bit.ly/[2]gOKLL7\n` |
| 4569 | pronoun | pronoun | punctuation → + / pronoun → − | 2.07 | 1.80 | + `cession found in special relativity[.]\nBut we can say`; ` join on a trial period[.]<|endoftext|>For decades, small`<br>− `agra upset article, there[ will] be first other effects fantastic`; ` age. Forzest[ can] help you get sugar when` |
| 1355 | punctuation | punctuation | punctuation → + / auxiliary → − | 1.85 | 3.37 | + ` nodded, I found it[.]\nThe girl surnamed`; ` was still immersed in ecstasy[.]\nAfter speaking, seeing`<br>− ` is an logic over perspective[ can] Learn shaped as many school`; ` a " that this text[ could] once Mind.\nI` |
| 762 | conjunction | preposition | punctuation → + / pronoun → − | 0.97 | 1.66 | + ` exactly likeable, except[ maybe] Dmitri (Carter�`; ` Yushin.\n[-] Dana Lee as Captain Tr`<br>− `amol, unless your[ doctor] has told you not to`; ` raw saw significant improvements in[ their] dog's energy, coat` |
| 4034 | preposition | punctuation | punctuation → + / pronoun → − | 1.07 | 1.90 | + ` the second-half,[ when] Nacer Chadli was`; ` exactly likeable, except[ maybe] Dmitri (Carter�`<br>− ` provide you with the services[ you] request, tell you about`; `\nGame and Fish biologists[ will] conduct aerial surveys through January` |
| 4250 | punctuation | punctuation | conjunction → + / punctuation → − | 1.16 | 4.19 | + ` plugging the Strait of[ H]ormuz, a vital`; `lukorede A[iy]egbusi (North`<br>− ` 2015||Yazaki Corporation[||]Signal determination apparatus and`; `ative Concepts, Inc.[||]Diskette storage container for` |
| 3270 | punctuation | punctuation | preposition → + / punctuation → − | 1.13 | 4.57 | + ` percent from the floor including[ a] 1-of-8`; ` a partial recovery is put[ at] 40%.read more...`<br>− ` and made it count...[<|endoftext|>]Read the many ways Russell`; ` in the decking.[<|endoftext|>]The most exclusive and costly` |
| 4236 | punctuation | punctuation | auxiliary → + / punctuation → − | 1.71 | 4.10 | + ` FML story\n- -[ Please] note that spam and nonsensical`; ` when posted. All information[ is] subject to modification from time`<br>− ` be affordable and simple.[\n]Dumpster rentals are`; ` issued by the authorities.[\n]Any dumpster is a` |
| 74 | determiner | determiner | determiner → + / auxiliary → − | 1.20 | 4.59 | + ` Parliament, Lord Holmes was[ a] Paralympic swimmer`; ` sisters Martha and Mary about[ a] future resurrection of the faithful`<br>− ` Department said GDP grew at[ an] annualised rate of 4`; `raluent group experienced[ an] average 64 percent greater reduction` |
| 3483 | punctuation | punctuation | determiner → + / punctuation → − | 0.98 | 3.55 | + `\nTuesday: Chris Sale[ (]0-0, 0`; `, like lilacs and[ hy]drangeas, don`<br>− `iki A, Wynshaw[-]Boris A, et`; ` applying to medical school.[\n]Ryan Lane: Travel before` |

## Red-team limits

- The ten token classes are deliberately coarse and incomplete. Low lift means only that this vocabulary failed, not that a channel is incoherent.
- Selection and top contexts come from discovery rows; only the class lift is held out.
- Layer 16 already includes attention-16 and all earlier blocks, so a channel can separate contextual states rather than current-token identity.
- The signed separator names are frozen on discovery data and evaluated as a standardized class-mean gap on held-out data. They remain coarse descriptions, not exhaustive meanings.
- Activation selectivity is descriptive. Finite channel ablations and cross-context transfer are required before assigning a causal verb.
- Native product channels have bilinear scaling and swap gauges. Interpret Left/Right labels as a pair, not as independently canonical neurons.
