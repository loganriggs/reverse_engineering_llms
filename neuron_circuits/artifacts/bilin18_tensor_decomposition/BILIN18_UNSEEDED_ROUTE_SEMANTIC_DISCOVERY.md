# Task-agnostic semantic discovery for natural route modes

Candidates were frozen from two token-blind geometry panels and a discreteness red team. Only high-margin, well-separated modes enter this fresh semantic discovery panel. The examples below may motivate labels, but those labels require another disjoint validation panel.

## Locked gates

- **D0_selection_is_label_blind: supported.** Observed: `{'selection_fields': ['center_cosine', 'cluster_size', 'member_cosine', 'assignment_margin', 'selection_score']}`
- **D1_candidate_set_is_broad: supported.** Observed: `{'candidates': 16, 'layers': 11}`
- **D2_natural_examples_are_diverse: supported.** Observed: `{'distinct': {'L17C1': 47, 'L2C0': 61, 'L6C2': 31, 'L3C1': 60, 'L16C1': 58, 'L8C1': 61, 'L11C0': 40, 'L3C3': 62, 'L8C2': 63, 'L2C3': 61, 'L13C0': 31, 'L14C3': 61, 'L4C0': 43, 'L14C1': 63, 'L1C2': 56, 'L2C2': 55}}`
- **D3_examples_support_prospective_labels: supported.** Observed: `{}`

| Candidate | Effective heads | Margin D/R | Median route cosine | Median distance | Copy | Self | Distinct sources |
|---|---:|---:|---:|---:|---:|---:|---:|
| L17C1 | 4.12 | 0.667/0.677 | 0.929 | 1.0 | 0.000 | 0.219 | 47 |
| L2C0 | 3.32 | 0.485/0.474 | 0.920 | 0.0 | 0.000 | 0.984 | 61 |
| L6C2 | 2.01 | 0.495/0.437 | 0.982 | 31.0 | 0.016 | 0.000 | 31 |
| L3C1 | 4.36 | 0.527/0.515 | 0.883 | 0.0 | 0.000 | 0.859 | 60 |
| L16C1 | 2.09 | 0.293/0.336 | 0.976 | 0.0 | 0.000 | 0.547 | 58 |
| L8C1 | 2.86 | 0.479/0.453 | 0.861 | 1.5 | 0.016 | 0.109 | 61 |
| L11C0 | 3.32 | 0.504/0.493 | 0.913 | 1.0 | 0.000 | 0.031 | 40 |
| L3C3 | 3.22 | 0.410/0.426 | 0.921 | 1.0 | 0.016 | 0.000 | 62 |
| L8C2 | 4.99 | 0.485/0.531 | 0.902 | 13.5 | 0.281 | 0.016 | 63 |
| L2C3 | 3.89 | 0.361/0.360 | 0.923 | 1.0 | 0.000 | 0.000 | 61 |
| L13C0 | 2.71 | 0.569/0.629 | 0.792 | 11.5 | 0.047 | 0.031 | 31 |
| L14C3 | 4.06 | 0.646/0.631 | 0.859 | 1.0 | 0.000 | 0.344 | 61 |
| L4C0 | 3.78 | 0.304/0.278 | 0.934 | 1.0 | 0.000 | 0.000 | 43 |
| L14C1 | 5.89 | 0.551/0.774 | 0.840 | 1.0 | 0.000 | 0.406 | 63 |
| L1C2 | 3.17 | 0.212/0.218 | 0.982 | 0.0 | 0.000 | 1.000 | 56 |
| L2C2 | 3.38 | 0.328/0.296 | 0.902 | 1.0 | 0.000 | 0.000 | 55 |

## Representative discovery examples

### L17C1

- d=1, cos=0.929: ` shop Escritos Politicos (1917`; source `os` → next `19`; outputs ` Venezuelan`, `Mexico`, ` Jorge`, ` Mexican`.
- d=1, cos=0.931: `\n- Talk about stuff\nMatthewKlug`; source `Matthew` → next `l`; outputs ` --------------------------------`, ` Decoder`, ` Errors`, ` Modes`.
- d=1, cos=0.935: ` shop Escritos Politicos (1917`; source `os` → next `19`; outputs `Mexico`, `ó`, `Spanish`, ` Venezuelan`.
- d=0, cos=0.896: ` Wylie: gzhan stong`; source `han` → next ` st`; outputs `Chinese`, `Tai`, ` Zh`, ` Zhang`.
- d=1, cos=0.877: ` Elsevier Science. Turun yliop`; source ` y` → next `op`; outputs ` Turkish`, ` Dj`, ` Nur`, `Chinese`.
- d=1, cos=0.911: `( Wylie: gzhan stong`; source `z` → next ` st`; outputs `German`, `�`, ` Wilhelm`, ` Mold`.

### L2C0

- d=0, cos=0.963: `Related from other country.\nRobert Redford`; source `Robert` → next ` Red`; outputs `'s`, `wig`, ` skill`, ` will`.
- d=54, cos=0.881: ` Rhetorical and linguistic shop Escritos Politicos ( … .\n03) has possible for shop Escrit`; source ` Esc` → next `rit`; outputs `-`, `é`, `w`, ` button`.
- d=0, cos=0.984: ` Chan, the partial relation of encouragement that was`; source ` encouragement` → next ` that`; outputs ` and`, ` for`, ` to`, ` in`.
- d=25, cos=0.863: `1917 1933) of the Whip for Spurring skills Onward Through the Chan Barrier Checkpoints( Changuan cejin), which he is to Chan Whip. This`; source ` Whip` → next `.`; outputs `'s`, `-`, ` on`, `,`.
- d=18, cos=0.896: ` rumours start circling that Emma and Andrew are secretly dating. November : Emma hosts U. S TV show Saturday Night Live and Andrew joins her`; source ` Andrew` → next ` joins`; outputs ` class`, ` really`, ` was`, ` loved`.
- d=0, cos=0.897: ` constitute continuous on the four physical antennas of child`; source ` antennas` → next ` of`; outputs ` for`, ` along`, ` and`, ` on`.

### L6C2

- d=31, cos=0.985: `( CMs the exercise is a …  identify conclusively concerned by this CM. conceptually,`; source `(` → next `ually`; outputs ` (`, ` in`, ` all`, ` and`.
- d=27, cos=0.993: ` would just clutter the site.<|endoftext|>- Pick location for next weekly\n- Review of new nodes - progress, delays, etc.\n- Talk about stuff\nMatthewKlug`; source `<|endoftext|>` → next `l`; outputs ` […]`, ` (`, ` in`, `...`.
- d=1, cos=0.741: ` of ICIJ contributed reporting.<|endoftext|>A statement`; source `<|endoftext|>` → next ` statement`; outputs ` just`, ` (`, ` in`, ` post`.
- d=5, cos=0.972: ` product may leave a review.<|endoftext|>Painting: Acrylic, Oil`; source `<|endoftext|>` → next `,`; outputs ` (`, ` […]`, ` in`, ` just`.
- d=2, cos=0.842: `Post Comments (Atom)<|endoftext|>Remote Call Center\n`; source `<|endoftext|>` → next ` Center`; outputs ` (`, ` in`, ` just`, ` […]`.
- d=31, cos=0.973: ` in Europe, North America, and …  pixels with a founding of quick &. philosophical colony approaches`; source ` in` → next ` colony`; outputs ` in`, ` (`, ` just`, ` all`.

### L3C1

- d=0, cos=0.927: ` 01 October 2020\n- Medway Community`; source `way` → next ` Community`; outputs `'s`, ` receives`, ` SC`, ` bron`.
- d=11, cos=0.939: `, the organizer of the Faery Weekends.\nDressing Up for the Faery Ball\n`; source `ery` → next ` Ball`; outputs `''`, ` Parkway`, ` Combo`, ` exceeds`.
- d=10, cos=0.962: ` Spirit: traditions ', ' M jS, cognition: activities ': ' M jS`; source ` j` → next `S`; outputs ` Donovan`, ` bron`, ` figure`, ` routine`.
- d=15, cos=0.875: ` is to use a variety of passwords.\nThe survey revealed that 48% of adults have used the same password for more`; source ` passwords` → next ` for`; outputs ` in`, `they`, ` poses`, ` items`.
- d=0, cos=0.939: `oti said that the number of SARs was`; source ` SAR` → next `s`; outputs ` should`, ` must`, `master`, ` and`.
- d=0, cos=0.914: ` (Atom)<|endoftext|>Remote Call Center\n`; source ` Call` → next ` Center`; outputs ` while`, ` /`, ` wire`, ` without`.

### L16C1

- d=1, cos=0.983: ` and arrangements for the drafting deposit can be found`; source ` can` → next ` found`; outputs ` be`, ` easily`, ` afford`, `'t`.
- d=0, cos=0.953: ` summer we utilize a theme to help guide`; source ` help` → next ` guide`; outputs ` achieve`, ` develop`, ` reduce`, ` generate`.
- d=1, cos=0.990: `. It doesn't necessarily have all the fancy`; source ` all` → next ` fancy`; outputs ` except`, ` kinds`, `ocating`, ` types`.
- d=0, cos=0.987: `ants (NetIQ) and have reasonable`; source ` have` → next ` reasonable`; outputs ` been`, `been`, ` Been`, ` gotten`.
- d=0, cos=0.978: ` of sediment into the marine system will affect the`; source ` will` → next ` affect`; outputs ` be`, ` continue`, ` remain`, ` strengthen`.
- d=0, cos=0.998: ` or something too localized that would just clutter the`; source ` just` → next ` clutter`; outputs `ifiable`, `!"`, ` poking`, ` but`.

### L8C1

- d=1, cos=0.942: `% of adults have used the same password for more`; source ` same` → next ` for`; outputs ` identical`, ` same`, `same`, ` as`.
- d=1, cos=0.952: ` you have any problem you can shoot us a message`; source ` shoot` → next ` a`; outputs ` shoot`, ` shoots`, ` video`, ` chefs`.
- d=8, cos=0.575: ` emergence block in abusive convergence. 1995a Viewing in review and translator. Philadelphia:`; source ` 1995` → next ` Philadelphia`; outputs ` 1995`, ` 1996`, `1996`, ` 1997`.
- d=2, cos=0.978: ` ProLink\nIdeal for newer racers, PRO`; source ` newer` → next `,`; outputs ` than`, `than`, ` newer`, ` Than`.
- d=7, cos=0.955: ` South Korea. Health authorities have approved the 8-Minute Rapid Test in Argentina`; source ` approved` → next ` in`; outputs ` approved`, ` permitted`, ` approve`, `approved`.
- d=3, cos=0.875: ` Levi right now? On Google Trends Zachary Levi had`; source ` Trends` → next ` had`; outputs ` trends`, ` Trends`, ` trend`, ` popularity`.

### L11C0

- d=1, cos=0.969: ` signage can be used to build a professional and welcoming`; source ` a` → next ` and`; outputs ` relationship`, ` system`, ` connection`, ` network`.
- d=2, cos=0.916: `.\nDressing Up for the Faery Ball\n`; source ` the` → next ` Ball`; outputs ` event`, ` festival`, ` conference`, ` contest`.
- d=1, cos=0.977: ` find that at the arithmetic of their life simplicity`; source ` their` → next ` simplicity`; outputs ` journey`, ` desires`, ` careers`, ` lives`.
- d=3, cos=0.984: ` bit louder still.\nThe one Wi-Fi–`; source ` one` → next `–`; outputs ` item`, ` thing`, ` issue`, ` element`.
- d=3, cos=0.946: ` yards. That was it for his offensive playmaking.`; source ` his` → next `.`; outputs ` career`, ` abilities`, ` skills`, ` debut`.
- d=1, cos=0.824: ` to prepare individual components to avoid your food becoming water`; source ` your` → next ` becoming`; outputs ` being`, ` getting`, `being`, ` becoming`.

### L3C3

- d=1, cos=0.916: ` is a extrapolation with two 2-dimensional`; source ` 2` → next `dimensional`; outputs ` buck`, ` stal`, `Max`, `emp`.
- d=1, cos=0.924: ` is currently being processed.\nRESAAS`; source `RES` → next `AS`; outputs ` stands`, `ullah`, ` Punjab`, `urrection`.
- d=1, cos=0.986: `, searchable detail than FinCEN itself`; source `C` → next ` itself`; outputs ` Founder`, ` Slayer`, ` Rewards`, `Located`.
- d=1, cos=0.927: ` jS ', ' M d ': '`; source ` '` → next ` '`; outputs ` 3`, ` Holy`, ` est`, ` bring`.
- d=1, cos=0.655: ` the WHO has estimated that 1 in 10 people`; source ` in` → next ` people`; outputs `591`, `����`, `ividual`, `529`.
- d=1, cos=0.918: `)\n- An Apple Watch Series 3 for $`; source ` Series` → next ` for`; outputs ` chronological`, ` Horizon`, ` nominate`, ` epist`.

### L8C2

- d=19, cos=0.855: ` for a family.\n-$360 provides a single immigrant mother with a microloan to start her own business.\n- $450 provides`; source `360` → next `450`; outputs ` 360`, `360`, ` 350`, ` 180`.
- d=56, cos=0.932: ` is global shop Escritos nations and region Aspects not, … Get slides here\nA shop Escritos Politic`; source ` nations` → next ` Polit`; outputs ` nations`, ` nation`, ` Nations`, ` Nation`.
- d=90, cos=0.944: ` the slides In this shop Escritos I perceive deleted a …  Langacker 2004). It expounds a shop Escrit`; source ` Esc` → next ` Esc`; outputs ` Esc`, `Esc`, ` esc`, `esc`.
- d=38, cos=0.896: ` a true patriot,” Malloy said. The governor also …  volumes,” he said.\nMalloy`; source ` Mall` → next `all`; outputs ` Mall`, ` mall`, `M`, ` M`.
- d=104, cos=0.938: ` of countries overseas where RESAAS already has several agents and customers …  and is currently being processed.\nRESAAS`; source `AS` → next `AS`; outputs `AS`, ` AS`, ` ASD`, ` ASP`.
- d=30, cos=0.926: `orrortu 1999a Experiments in moving a number of conceptualizer others. Marcu, Daniel, Magdalena Romera and Estibaliz Amorrortu`; source `ort` → next `ort`; outputs `ort`, `ORT`, `orts`, ` Cort`.

### L2C3

- d=1, cos=0.804: ` bag was still sitting there. Nope, call this`; source ` Nope` → next ` call`; outputs ` nor`, ` censorship`, ` anymore`, ` refuses`.
- d=1, cos=0.932: ` recommend you watch now (or later, if your`; source ` later` → next ` if`; outputs ` results`, ` dialog`, ` December`, `Late`.
- d=1, cos=0.890: `. The dependable meaning of extrapolation between`; source ` extrap` → next ` between`; outputs ` forward`, ` onto`, ` flow`, ` form`.
- d=1, cos=0.945: `ourses of the Buddha, the few of the four`; source ` few` → next ` the`; outputs ` whatsoever`, `months`, ` Yet`, `but`.
- d=1, cos=0.956: ` & consume research content, from start-finish`; source ` start` → next `fin`; outputs ` Barron`, `byter`, ` fail`, ` fails`.
- d=1, cos=0.921: ` set to be rolled out in mid-October`; source ` mid` → next `October`; outputs ` console`, `December`, ` Janeiro`, `Fif`.

### L13C0

- d=5, cos=0.646: ` request, the result of presentations( and influenced complement more just) has`; source `(` → next `)`; outputs `)`, `),`, `).`, `.)`.
- d=4, cos=0.768: `\nI have some additional podcasts (several from before Brad�`; source ` (` → next ` Brad`; outputs `)`, `),`, `).`, `.)`.
- d=7, cos=0.744: ` Escritos Politicos (1917 motion faith-imagination. Wi`; source ` (` → next `.`; outputs `)`, `),`, `).`, `.)`.
- d=9, cos=0.832: ` the stupid games that women play( yes! Girls will do this to check you as I`; source `(` → next ` as`; outputs `)`, `),`, `).`, `.)`.
- d=18, cos=0.950: ` hearer Rappaport, ' Simulation of UHF several Note aspects for frame-like decision-making text-types, ' direct`; source ` '` → next ` '`; outputs `',`, `,'`, `'.`, `.'`.
- d=7, cos=0.895: ` electronic signatures, duplicate copies shouldn't be required.\nGovernment reporting systems have a`; source `'t` → next ` have`; outputs ` '`, ` "`, `'m`, `'t`.

### L14C3

- d=2, cos=0.970: ` transactions, they are supposed to file a report within 30`; source ` file` → next ` within`; outputs ` with`, `with`, ` paperwork`, ` against`.
- d=2, cos=0.983: ` we share cultural bonds. We challenged the students to consider`; source ` challenged` → next ` to`; outputs ` to`, ` concepts`, ` against`, ` assumptions`.
- d=1, cos=0.933: ` learned from others, and we bring those insights together`; source ` bring` → next ` insights`; outputs ` into`, ` back`, ` together`, ` closer`.
- d=2, cos=0.969: ` up to a water supply which saves the children a walk`; source ` saves` → next ` a`; outputs ` money`, ` $$`, ` cash`, ` $`.
- d=1, cos=0.864: `! Girls will do this to check you as I`; source ` check` → next ` as`; outputs ` out`, ` boxes`, ` box`, ` check`.
- d=1, cos=0.925: ` you have any problem you can shoot us a message`; source ` shoot` → next ` a`; outputs ` photos`, ` photo`, ` pictures`, ` photographs`.

### L4C0

- d=1, cos=0.836: ` voice assistant.\nIt also beats the cheaper Vers`; source ` beats` → next ` cheaper`; outputs ` competitors`, ` competitor`, ` stereotype`, ` conventional`.
- d=1, cos=0.957: `. There King Peter was regarded by the people of`; source ` by` → next ` people`; outputs ` crowd`, ` audience`, ` owners`, ` approving`.
- d=1, cos=0.958: ` nothing I can say that will explain this short film`; source ` explain` → next ` short`; outputs ` findings`, ` why`, ` differences`, ` plight`.
- d=2, cos=0.949: ` you find his links!\nFollow Razur Cuts`; source `Follow` → next ` C`; outputs ` guidelines`, ` blindly`, ` rules`, ` trend`.
- d=1, cos=0.977: `ationalization has generated early example for verses and`; source ` for` → next ` and`; outputs ` purposes`, ` sake`, ` purpose`, ` reason`.
- d=1, cos=0.963: ` make only hoped with circular links by explicit liberties`; source ` by` → next ` liberties`; outputs ` lab`, ` margin`, ` experts`, ` staff`.

### L14C1

- d=2, cos=0.864: ` monk). Journal of Psychology, Neuroscience, and Economics.`; source ` Neuroscience` → next ` Economics`; outputs ` Barber`, ` stall`, ` pag`, ` Maz`.
- d=6, cos=0.883: ` This activates the nerves, and dopamine is able to communicate with the nervous`; source ` dopamine` → next ` nervous`; outputs `arrett`, ` welding`, ` Barron`, `Elf`.
- d=1, cos=0.843: ` Emma Stone and Saturday Night Live writer and segment director`; source ` writer` → next ` segment`; outputs ` Clifford`, ` interoper`, ` robe`, ` Sutherland`.
- d=1, cos=0.900: ` as a witness in any legal proceeding or seeks to`; source ` proceeding` → next ` seeks`; outputs ` recruit`, ` meme`, ` restaurants`, ` swim`.
- d=4, cos=0.876: ` Cebus monk). Journal of Psychology, Neuroscience, and Economics.`; source ` Psychology` → next ` Economics`; outputs ` Hud`, ` complications`, ` bandits`, `Luc`.
- d=1, cos=0.865: ` by ensuring we provide outstanding customer service, and our`; source ` service` → next ` and`; outputs ` Caleb`, ` Industrial`, ` reactor`, ` lucid`.

### L1C2

- d=0, cos=0.983: ` public. They have through general B sensation 3`; source ` B` → next ` sensation`; outputs ` made`, ` really`, ` in`, ` take`.
- d=0, cos=0.980: ` English.\nRicardo Mairal`; source ` M` → next `air`; outputs ` really`, ` made`, ` in`, ` take`.
- d=0, cos=0.989: ` 2009), and most only, Tomasello`; source ` Tom` → next `ase`; outputs ` in`, `.`, ` just`, ` get`.
- d=0, cos=0.987: ` TV show Saturday Night Live and Andrew joins her`; source ` Andrew` → next ` joins`; outputs `.`, ` in`, ` just`, ` has`.
- d=0, cos=0.973: ` WAR AMES US MODEL 1832 FO`; source ` 18` → next `32`; outputs ` that`, ` in`, ` to`, ` really`.
- d=0, cos=0.978: ` 5, e-interview 8-`; source ` 8` → next `-`; outputs ` really`, ` that`, ` in`, ` to`.

### L2C2

- d=2, cos=0.806: ` thing about dating is that it saves a lot of money`; source ` saves` → next ` of`; outputs ` damage`, ` keep`, ` shields`, ` restores`.
- d=1, cos=0.907: ` Comments (Atom)<|endoftext|>Remote Call Center\n`; source `Remote` → next ` Center`; outputs ` mobile`, ` enables`, ` multit`, ` Survive`.
- d=1, cos=0.687: ` passed down. It is no surprise that the influence`; source ` surprise` → next ` the`; outputs ` parad`, `Strange`, ` discrepancies`, ` discomfort`.
- d=1, cos=0.711: ` times had been good, we imagined ourselves difficult`; source ` imagined` → next ` difficult`; outputs ` involvement`, ` possibilities`, `Maybe`, ` Maybe`.
- d=1, cos=0.972: `A Special Thank you to Derek and Razur`; source ` and` → next `ur`; outputs ` alike`, ` walk`, ` respectively`, ` thereof`.
- d=1, cos=0.838: ` ability to be wonderful. Samantha suggests you want to`; source ` suggests` → next ` want`; outputs ` to`, ` drop`, ` following`, ` "`.

## Guardrail

A candidate can be a coherent route mode without admitting a short human semantic label. Unexplained candidates remain valid geometric objects; broad labels will be accepted only if frozen here and replicated on fresh rows.
