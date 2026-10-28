# Bilin18 layer-15 native MLP discriminator atlas

This is a first-pass input-selectivity audit of all 4,608 native bilinear product channels. Each channel computes `Left_j(x) * Right_j(x)` after attention 15 and RMS normalization. A token-class label is selected on disjoint discovery rows and its absolute-activation lift is evaluated on held-out rows.

## Aggregate

- Discovery tokens: 2,048,000; held-out tokens: 512,000.
- Left/right factors choose the same coarse class for 36.5% of all channels and 39.7% of the more usable half.
- Among the more usable half, median held-out class lift is 1.29; 19.5% exceed 1.5× and 5.2% exceed 2×.

- Their median held-out signed class-to-class gap is 0.31 product standard deviations; 4.3% exceed one standard deviation.

`Lift` is mean absolute product activation on the frozen preferred class divided by its mean over all held-out tokens. `Use score = product standard deviation × Down-column norm`; it is only a first-order availability score, not a causal effect on loss.

## Representative native channels

| unit | Left | Right | signed product separator | held-out gap (SD) | lift | strongest + / − contexts |
|---:|---|---|---|---:|---:|---|
| 2070 | punctuation | pronoun | pronoun → + / punctuation → − | 1.17 | 1.59 | + `ivari Sustains Career[-]Ending`; ` Looking For Adventures Girl|[\n]|Seeking: ||`<br>− ` have especially taken pleasure in[.]\n30. 4 Easy`; ` fact looks like my face[.]\nA union takes place` |
| 2416 | punctuation | punctuation | pronoun → + / punctuation → − | 1.27 | 2.38 | + `EST.;ASSIGNORS[:]GELARDI,`; `EST;ASSIGNORS[:]JANZEN,`<br>− ` LET US KNOW AND WE[']LL UPDATE THE INFORMATION!`; `RIBE TO OUR WEEK[LY] NEWSLETTERSub` |
| 978 | preposition | punctuation | determiner → + / punctuation → − | 1.43 | 1.99 | + ` “What will happen[ if] the Greeks do default?`; `ually every article ever written[ about] ResEdit warns you to`<br>− `||Jan 11, 2001[||]Sep 12, 2006||`; `EST;ASSIGNOR[:]CRAY RESEARCH` |
| 2350 | punctuation | punctuation | conjunction → + / punctuation → − | 1.16 | 3.63 | + `So…Okay. I[�]�ve been looking forward`; ` into your truck's wiring[ harness], but it's much`<br>− `. The student, Jeremiah[ (]Nat Wolff), could`; ` article\n- Open Access[\n]- Open Peer Review\n` |
| 1577 | punctuation | punctuation | punctuation → + / pronoun → − | 1.43 | 1.99 | + ` crash: Transcript reveals passengers[ ']screamed for over`; ` Hanlon said he is[ "]cautiously optimistic"`<br>− ` annually, he’[d] like to see it frequent`; ` He crowed that he['d] had a sixth sense about` |
| 438 | punctuation | auxiliary | auxiliary → + / punctuation → − | 1.49 | 1.68 | + `Deposited By:[||]Concordia University`; ` Constructions in practitioners which[ may] finally do seen mental.`<br>− `US7881244[||]May 17, 2009||`; `View Single Post\n07[-]11-2006, 11` |
| 1018 | number | number | conjunction → + / content_word → − | 0.28 | 1.24 | + ` members may contribute stories and[ comments] as well as view additional`; ` couch. You may choose[ to] have it in Faux`<br>− `routine servicing. Please[ get] in touch with details of`; ` often need to be booked[ well] in advance, and most` |
| 3599 | punctuation | punctuation | auxiliary → + / punctuation → − | 1.35 | 3.77 | + `ant all hot, and[ was] slumberously enjoying a`; ` up bareback. disgu[ising] Arturo shag`<br>− ` Leon and Lorna[ (]Greer) Tyrrell`; ` Jeremy Sex Pills?[\n]Xueliang's sword` |
| 311 | pronoun | auxiliary | pronoun → + / capitalized → − | 1.87 | 2.12 | + ` Sun, IN is used[ to] dispose excessive amounts of garbage`; ` sort of waste is always[ to]`<br>− `|\n|US45[75]806 *||Mar 26`; `\n|US200600[98]509 *||May 31` |
| 3506 | punctuation | auxiliary | auxiliary → + / capitalized → − | 1.72 | 1.11 | + ` gradually and little by[ little] for one more second in`; ` it was his arthritis specifically[ that] created this difficulty, he`<br>− `7 percent (from 2[,]760,505 to 2`; ` home near school||10[,]681|\n|N` |
| 1693 | punctuation | punctuation | punctuation → + / number → − | 0.45 | 1.49 | + ` 28277, Telephone Number[ 8]66-501-23`; ` QSR's flagship e[-]newsletter, A.`<br>− ` time of 1:00[.]52. Alana Dil`; `), Referenced by[ (]91), Class` |
| 1575 | auxiliary | punctuation | auxiliary → + / capitalized → − | 1.73 | 1.11 | + ` it was his arthritis specifically[ that] created this difficulty, he`; ` head straight while the seam[stress] did my hem and I`<br>− ` spend countless hours attempting to[ work] out which wires goes to`; `, the home wiring diagram[ is] simply` |
| 3355 | preposition | auxiliary | auxiliary → + / capitalized → − | 1.70 | 1.11 | + ` starkly how far he[ is] willing to go to prevent`; ` with one or two exceptions[,] settled there in defiance of`<br>− ` have a website that isn[�]�t smooth to navigate`; ` measuring 2691 square feet[ (]250 square meters) include` |
| 901 | capitalized | auxiliary | auxiliary → + / capitalized → − | 1.84 | 1.17 | + ` and Childbirth\nStudies[ have] linked cigarette smoking to infertility`; ` eating right. Other studies[ have] found that kids who play`<br>− ` crude oil earns Vietnam US[$]54.4 on average`; `Spector, D.[\n]Journal cara memb` |
| 2036 | pronoun | auxiliary | auxiliary → + / capitalized → − | 1.76 | 1.12 | + ` That can be difficult,[ requires] a lot of L1`; ` is not easy and does[ require] constant attention. There are`<br>− `. Overall the wiring is[ really] straight forward. There's`; `ixões),[ A] Coruña, Bil` |
| 2271 | pronoun | auxiliary | pronoun → + / capitalized → − | 1.53 | 2.06 | + ` at the law firm Davis[,] Miner, Barnhill and`; ` at all,”[ she] said, and she nodded`<br>− `stam; Editing by[ Matt] Driskill)<|endoftext|>`; ` Eliot Gardiner, Bach['s] St Matthew` |
| 3417 | pronoun | pronoun | pronoun → + / capitalized → − | 1.88 | 4.62 | + `hisattva and how[ they] are it( be extremely`; `aa reveal only Third because[ they] learn their factors of the`<br>− ` offensive in the direction of[ C]zernovitz (`; `BILITY AND WARRANTY SHALL[ AP]PLY TO THE MAX` |
| 3173 | pronoun | pronoun | preposition → + / pronoun → − | 1.56 | 3.76 | + `, if for some reason[ you] are not able to download`; ` to third-party websites[ that] are not affiliated with us`<br>− ` their court.\nThis[ article] appeared in National Post`; `t be the awesome place[ it] is!\n- What` |
| 4408 | punctuation | punctuation | punctuation → + / auxiliary → − | 1.34 | 3.31 | + ` statements of what the Church[ has] defined, but also an`; `, simply give us a[ call] and we will get your`<br>− ` dumpsters can be purchased[ that] can be wheeled about`; ` said these things while they[ looked] on, He was raised` |
| 1106 | number | number | conjunction → + / number → − | 1.53 | 4.16 | + ` your truck's wiring harness[,] but it's much`; ` from beyond the box.[\n]The control box may have`<br>− `7 percent (from 2[,]760,505 to 2`; `é 272 PS||14[,]541||46,` |

## Red-team limits

- The ten token classes are deliberately coarse and incomplete. Low lift means only that this vocabulary failed, not that a channel is incoherent.
- Selection and top contexts come from discovery rows; only the class lift is held out.
- Layer 15 already includes attention-15 and all earlier blocks, so a channel can separate contextual states rather than current-token identity.
- The signed separator names are frozen on discovery data and evaluated as a standardized class-mean gap on held-out data. They remain coarse descriptions, not exhaustive meanings.
- Activation selectivity is descriptive. Finite channel ablations and cross-context transfer are required before assigning a causal verb.
- Native product channels have bilinear scaling and swap gauges. Interpret Left/Right labels as a pair, not as independently canonical neurons.
