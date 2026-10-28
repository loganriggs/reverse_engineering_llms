# Bilin18 layer-2 native MLP discriminator atlas

This is a first-pass input-selectivity audit of all 4,608 native bilinear product channels. Each channel computes `Left_j(x) * Right_j(x)` after attention 2 and RMS normalization. A token-class label is selected on disjoint discovery rows and its absolute-activation lift is evaluated on held-out rows.

## Aggregate

- Discovery tokens: 8,192,000; held-out tokens: 2,048,000.
- Left/right factors choose the same coarse class for 40.1% of all channels and 44.9% of the more usable half.
- Among the more usable half, median held-out class lift is 1.52; 51.9% exceed 1.5× and 16.3% exceed 2×.

- Their median held-out signed class-to-class gap is 0.56 product standard deviations; 13.4% exceed one standard deviation.

`Lift` is mean absolute product activation on the frozen preferred class divided by its mean over all held-out tokens. `Use score = product standard deviation × Down-column norm`; it is only a first-order availability score, not a causal effect on loss.

## Representative native channels

| unit | Left | Right | signed product separator | held-out gap (SD) | lift | strongest + / − contexts |
|---:|---|---|---|---:|---:|---|
| 2013 | subword | punctuation | punctuation → + / capitalized → − | 0.30 | 2.53 | + `huset in Oslo.[<|endoftext|>]Update on Russia and Ukraine`; ` Festival.READ the article[<|endoftext|>]NZ Olympic class 49er`<br>− ` page about Irish pro-[Palestinian] activists - then reported it`; `Number of children born to[ Yemen]-born construction magnate` |
| 526 | pronoun | subword | pronoun → + / conjunction → − | 0.10 | 1.40 | + `[oy]’s epic works`; `[ker] Sports Radio Network, including`<br>− ` Free Stuff? Click here[ to] see all current Freebie`; ` (FB readers click here[ to] listen)<|endoftext|>Don't` |
| 3013 | preposition | content_word | conjunction → + / auxiliary → − | 0.39 | 1.49 | + ` who doesnâ€™[t] have health care in America`; ` not to say there weren['t] brave attempts in the mean`<br>− ` the track gearing is anything[ but] simple. We are getting`; ` but modern pirates are anything[ but] romantic. In 2005,` |
| 4412 | number | capitalized | number → + / punctuation → − | 1.37 | 2.39 | + `, long, long,[ long]) ways to go until`; ` (long, long,[ long], long) ways to`<br>− ` (R$ 2,[55])\n*** From Ana`; ` #3: 6,[95] (incl. 6` |
| 2273 | punctuation | punctuation | number → + / punctuation → − | 0.59 | 2.66 | + ` rubbish.�[�]\n‘Of course`; ` giving me books.�[�]” ? J.`<br>− ` is essential for today�[�]s Travel Retail market.`; ` cut it for today�[�]s Travel Retailers.` |
| 2907 | content_word | subword | subword → + / pronoun → − | 0.47 | 1.91 | + ` for WAY MORE TLD[s]!\nIn the past`; ` with nearly 130 active volcan[oes], more than any other`<br>− ` the joining of a bell[ows]\nunder the right arm`; ` the guttural bell[ows] of those who insist the` |
| 598 | number | subword | subword → + / capitalized → − | 0.48 | 1.29 | + `Codes: 1 -[ 6]||Yes|\n|`; `Codes: 1 -[ 6]||Yes|\n|`<br>− `8.4 s||[13].3 s||-`; `Sep 7, 2009||[Jan]` |
| 1737 | capitalized | capitalized | capitalized → + / punctuation → − | 2.45 | 4.68 | + `[Added] source code of Item Reader`; ` for a certificate background.[ Russian], Using them as a`<br>− ` Game development company in ind[ia]\n4%)] or`; ` i was back in ind[ia]. i felt like i` |
| 1142 | punctuation | punctuation | capitalized → + / punctuation → − | 0.61 | 2.28 | + ` Guides[\]Poetry Rules.wp`; `raulic cylinders['] system) and solutions to`<br>− ` substantial move and they couldn[�]�t get it done`; `. However, it shouldn[�]�t bother me too` |
| 899 | number | capitalized | number → + / pronoun → − | 0.45 | 1.65 | + ` after Game 2 against the[ Wizards] — will lead the team`; `Codes: 1 -[ 6]||Yes|\n|`<br>− ` (877) 320-[320]3 Toll Free\nJob`; ` *||Apr 30,[ 1984]||May 6, 1986` |
| 2852 | punctuation | pronoun | conjunction → + / capitalized → − | 0.43 | 1.57 | + `hiwa (France)[ —] at his offices.\n`; ` ends up okay too)[ but] this guy has had some`<br>− ` we just leave!�[�]\n“You can`; ` at the moment,�[�] Suze glared.\n` |
| 605 | subword | subword | conjunction → + / subword → − | 0.80 | 2.75 | + ` among their peers. Jud[ged] on the creativity expressed in`; ` of being behind the Ron[in] bridge hacking incident. As`<br>− ` Clamp, Alist[air] Gibson pen)\nAug`; `; 2 – Alist[air] Gibson; 2 – Adam` |
| 3080 | number | number | conjunction → + / capitalized → − | 0.30 | 1.50 | + `IFAS Wakulla County[ Extension] Director\nWhile there are`; ` a Member\nDecember 2016[ onwards]\nSubmit An Event\n`<br>− ` try try fail try try[ try] fail take a break from`; `, long, long,[ long]) ways to go until` |
| 3337 | punctuation | punctuation | pronoun → + / punctuation → − | 0.43 | 2.52 | + ` "[\n]- "If our tax`; `."[\n]But Centcom spokesman Owens`<br>− ` for everyone” ([�]�Umdat al`; `�d already gotten ([�]�With all the negative` |
| 66 | content_word | subword | number → + / content_word → − | 0.25 | 1.66 | + `locked in each[ others] arms, floated downstream,`; `lin and I were each[ others]' favorite sisters when we`<br>− ` compares to other top VPN[s] on the market. I`; ` the many benefits TNO[YS] offers our member organizations is` |
| 2572 | punctuation | punctuation | number → + / punctuation → − | 1.22 | 3.15 | + ` Wednesday, the Westchester[ County] Medical Examiner's Office said`; ` about looking back, though[.] If we agree on these`<br>− `LHWVx[ ).] "Cars were driving`; `/opportunitiesstudy[/.]\nThe next day,` |
| 223 | number | number | number → + / determiner → − | 2.44 | 7.68 | + ` higher price: AU$[69].95, NZ$`; `\n|4||$[39].00||  ADD`<br>− ` 4th year in a[ row] as the only representatives of`; ` about two days in a[ row]? You want to learn` |
| 4027 | auxiliary | auxiliary | conjunction → + / auxiliary → − | 2.86 | 6.59 | + `lda.\nIt looks[ and] sounds to me like Mc`; ` raised in the petition could[ and] should have been raised on`<br>− ` their business plan. How[ are] they doing that? �`; ` Hearts Meet Dating Agency You[re] happily matched and Eith` |
| 3610 | auxiliary | auxiliary | auxiliary → + / number → − | 2.94 | 6.01 | + ` looks like he’[s] going to be your next`; `Already, she’[s] excited about swimming in Rio`<br>− ` Bromer.<|endoftext|>Where[ did] the NEWS 2 Info Box`; ` Loniewski\nWhere[ did] you get the idea for` |
| 1167 | number | number | number → + / conjunction → − | 2.40 | 6.08 | + `:08|\n|[7]||Mirage||S`; `. Portulaca\n[4]. Dragon carrots\nOh`<br>− ` FL-Q-04[2]Z\nBe an instant`; ` Inc., C-370[4], 62 Fed. Reg` |

## Red-team limits

- The ten token classes are deliberately coarse and incomplete. Low lift means only that this vocabulary failed, not that a channel is incoherent.
- Selection and top contexts come from discovery rows; only the class lift is held out.
- Layer 2 already includes attention-2 and all earlier blocks, so a channel can separate contextual states rather than current-token identity.
- The signed separator names are frozen on discovery data and evaluated as a standardized class-mean gap on held-out data. They remain coarse descriptions, not exhaustive meanings.
- Activation selectivity is descriptive. Finite channel ablations and cross-context transfer are required before assigning a causal verb.
- Native product channels have bilinear scaling and swap gauges. Interpret Left/Right labels as a pair, not as independently canonical neurons.
