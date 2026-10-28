# Task-agnostic semantic discovery for natural route modes

Candidates were frozen from two token-blind geometry panels and a discreteness red team. Only high-margin, well-separated modes enter this fresh semantic discovery panel. The examples below may motivate labels, but those labels require another disjoint validation panel.

## Locked gates

- **D0_selection_is_label_blind: supported.** Observed: `{'selection_fields': ['center_cosine', 'cluster_size', 'member_cosine', 'assignment_margin', 'selection_score']}`
- **D1_candidate_set_is_broad: supported.** Observed: `{'candidates': 16, 'layers': 11}`
- **D2_natural_examples_are_diverse: supported.** Observed: `{'distinct': {'L17C1': 57, 'L2C0': 64, 'L6C2': 17, 'L3C1': 64, 'L16C1': 52, 'L8C1': 62, 'L11C0': 33, 'L3C3': 61, 'L8C2': 62, 'L2C3': 62, 'L13C0': 34, 'L14C3': 63, 'L4C0': 50, 'L14C1': 64, 'L1C2': 62, 'L2C2': 60}}`
- **D3_examples_support_prospective_labels: supported.** Observed: `{}`

| Candidate | Effective heads | Margin D/R | Median route cosine | Median distance | Copy | Self | Distinct sources |
|---|---:|---:|---:|---:|---:|---:|---:|
| L17C1 | 4.12 | 0.667/0.677 | 0.929 | 1.0 | 0.000 | 0.281 | 57 |
| L2C0 | 3.32 | 0.485/0.474 | 0.901 | 0.0 | 0.000 | 0.969 | 64 |
| L6C2 | 2.01 | 0.495/0.437 | 0.981 | 22.0 | 0.016 | 0.047 | 17 |
| L3C1 | 4.36 | 0.527/0.515 | 0.915 | 0.5 | 0.000 | 0.953 | 64 |
| L16C1 | 2.09 | 0.293/0.336 | 0.972 | 0.0 | 0.000 | 0.641 | 52 |
| L8C1 | 2.86 | 0.479/0.453 | 0.873 | 1.0 | 0.000 | 0.078 | 62 |
| L11C0 | 3.32 | 0.504/0.493 | 0.945 | 1.0 | 0.000 | 0.016 | 33 |
| L3C3 | 3.22 | 0.410/0.426 | 0.926 | 1.0 | 0.016 | 0.000 | 61 |
| L8C2 | 4.99 | 0.485/0.531 | 0.930 | 23.5 | 0.438 | 0.000 | 62 |
| L2C3 | 3.89 | 0.361/0.360 | 0.903 | 1.0 | 0.000 | 0.000 | 62 |
| L13C0 | 2.71 | 0.569/0.629 | 0.761 | 13.5 | 0.047 | 0.031 | 34 |
| L14C3 | 4.06 | 0.646/0.631 | 0.888 | 1.0 | 0.000 | 0.453 | 63 |
| L4C0 | 3.78 | 0.304/0.278 | 0.938 | 1.0 | 0.000 | 0.000 | 50 |
| L14C1 | 5.89 | 0.551/0.774 | 0.857 | 1.0 | 0.000 | 0.344 | 64 |
| L1C2 | 3.17 | 0.212/0.218 | 0.979 | 0.0 | 0.000 | 0.969 | 62 |
| L2C2 | 3.38 | 0.328/0.296 | 0.920 | 1.0 | 0.000 | 0.000 | 60 |

## Representative discovery examples

### L17C1

- d=1, cos=0.944: ` in top money aura list\nRubaab�`; source `Rub` → next `ab`; outputs `Kh`, `Ku`, `opter`, `Ta`.
- d=1, cos=0.946: `Supermarket: supermeracado (SOO`; source `ado` → next `S`; outputs ` Colombian`, ` Mexican`, `Spanish`, ` Venezuelan`.
- d=1, cos=0.872: `\nCentcom spokesman Maj. Brad Bartelt`; source ` Brad` → next `elt`; outputs `KER`, `ull`, `429`, `246`.
- d=1, cos=0.953: `ín.\nPopulation pyramid of Crucero de`; source ` Cru` → next `o`; outputs `Spanish`, `Italian`, ` Chilean`, ` Catalan`.
- d=0, cos=0.950: `.\nPopulation pyramid of Crucero de`; source `cer` → next `o`; outputs `Spanish`, `Mex`, ` Chilean`, `Mexico`.
- d=1, cos=0.971: `ehl SOO-pehr-me`; source `hr` → next `me`; outputs ` Democr`, ` Loot`, `763`, `107`.

### L2C0

- d=0, cos=0.948: ` Pop out window function in a Discord Voice chat`; source ` Discord` → next ` Voice`; outputs `'s`, ` on`, ` and`, ` content`.
- d=15, cos=0.892: ` 7 pm<|endoftext|>Brian Clarke: Microcosm (Stained Glass and Paintings)\nBrian Clarke: Microcosm`; source ` Micro` → next `cos`; outputs ` B`, ` project`, `.`, ` mod`.
- d=17, cos=0.869: `, we finally settled on a salad.\nNow, in Spain, it should be noted that the idea of a salad seems to`; source ` salad` → next ` seems`; outputs ` of`, ` —`, ` in`, ` with`.
- d=0, cos=0.861: ` emergency, please dial 999<|endoftext|>Parent Teacher Association`; source `Parent` → next ` Teacher`; outputs ` with`, ` parenting`, ` every`, ` happiness`.
- d=79, cos=0.888: ` the ferry flight of space shuttle Endeavour atop the 747 Shuttle …  2011, NASA transferred title and ownership of Ende`; source ` End` → next `e`; outputs `-`, ` flow`, `master`, `ite`.
- d=82, cos=0.882: ` 11 am – 7 pm<|endoftext|>Brian Clarke: Microcosm ( …  Museum of Modern Art, Tokyo, and Brian Clarke:`; source `Brian` → next ` Clarke`; outputs `'s`, ` continued`, ` --`, `'`.

### L6C2

- d=1, cos=0.875: ` paisa bachao.<|endoftext|>Vina Lag`; source `<|endoftext|>` → next `ina`; outputs ` (`, ` just`, ` in`, ` […]`.
- d=12, cos=0.991: `15 from Mitchell & Son.<|endoftext|>Not all wireless systems are the same.\nTypical wireless systems`; source `<|endoftext|>` → next ` systems`; outputs ` […]`, ` (`, ` in`, ` just`.
- d=3, cos=0.968: ` to pay...ahahahah)<|endoftext|>In a country that sal`; source `<|endoftext|>` → next ` that`; outputs ` (`, ` […]`, ` in`, ` just`.
- d=1, cos=0.843: `ers get 15% off!<|endoftext|>Facts about`; source `<|endoftext|>` → next `acts`; outputs ` (`, ` just`, ` in`, ` post`.
- d=6, cos=0.977: ` also provided on our website.<|endoftext|>All NuHeat Electric Tile Floor Warming`; source `<|endoftext|>` → next ` W`; outputs ` […]`, ` (`, ` just`, ` in`.
- d=1, cos=0.875: ` medical emergency, please dial 999<|endoftext|>Parent Teacher Association`; source `<|endoftext|>` → next ` Teacher`; outputs ` (`, ` just`, ` in`, ` post`.

### L3C1

- d=9, cos=0.878: ` & Son.<|endoftext|>Not all wireless systems are the same.\nTypical wireless systems`; source ` wireless` → next ` systems`; outputs ` one`, ` must`, ` Brah`, ` should`.
- d=28, cos=0.922: `, British Defense Secretary Geoff Hoon seemed to agree with the need for international verification, telling reporters that "it is important that we have an objective source of verification." Hoon said he`; source `oon` → next ` said`; outputs ` rides`, ` opens`, ` Tup`, `allery`.
- d=15, cos=0.915: ` 7 pm<|endoftext|>Brian Clarke: Microcosm (Stained Glass and Paintings)\nBrian Clarke: Microcosm`; source ` Micro` → next `cos`; outputs ` flow`, ` envelop`, ` project`, ` whenever`.
- d=10, cos=0.953: ` View\nRooftop Restaurant & Bar\nLulu Rooftop Restaurant\n`; source ` Restaurant` → next `\n`; outputs `Financial`, ` accessibility`, ` mercury`, ` majesty`.
- d=79, cos=0.739: ` the ferry flight of space shuttle Endeavour atop the 747 Shuttle …  2011, NASA transferred title and ownership of Ende`; source ` End` → next `e`; outputs `iana`, ` skin`, `oddy`, ` transaction`.
- d=0, cos=0.891: ` have been onboard the International Space Station for two`; source ` Station` → next ` for`; outputs `Sent`, ` basically`, ` jointly`, ` essentially`.

### L16C1

- d=0, cos=0.990: ` Voicemail facility. You may even use`; source ` may` → next ` even`; outputs ` be`, ` include`, ` someday`, ` contribute`.
- d=0, cos=0.991: `\nMake sure your brand is easily reachable`; source ` easily` → next ` reach`; outputs ` than`, ` intoxicated`, ` locate`, ` Raise`.
- d=0, cos=0.985: `<|endoftext|>Valley Torah High School will be (`; source ` will` → next ` be`; outputs ` be`, ` continue`, ` remain`, ` Be`.
- d=1, cos=0.593: ` location\nIf you have any questions or concerns,`; source ` questions` → next ` concerns`; outputs ` combine`, ` combining`, ` Merge`, ` Citadel`.
- d=0, cos=0.976: ` wifi / Ethernet port embedded. Only the standard`; source ` Only` → next ` the`; outputs ` but`, ` survived`, ` BUT`, ` But`.
- d=0, cos=0.838: ` to make all of our dreams come true.`; source ` come` → next ` true`; outputs ` Mis`, ` with`, ` descriptions`, `36`.

### L8C1

- d=1, cos=0.650: ` been found there," about the Tribune-Review report`; source ` Tribune` → next `Review`; outputs ` Tribune`, ` newspaper`, ` t`, ` county`.
- d=1, cos=0.959: `, Costco and station.\nFreshly painted throughout`; source `Fresh` → next ` painted`; outputs `Fresh`, ` fresh`, ` Fresh`, `fresh`.
- d=3, cos=0.767: `Many sources recommend extra calcium, potassium and/or magnesium`; source ` potassium` → next ` magnesium`; outputs ` potassium`, ` Pot`, ` K`, `assium`.
- d=2, cos=0.929: ` your cards to make plays or burn your opponents faster`; source ` burn` → next ` faster`; outputs ` burn`, ` burns`, ` burned`, ` burning`.
- d=0, cos=0.980: ` mean). It’s cool how`; source ` cool` → next ` how`; outputs ` quirky`, ` uniqueness`, ` digitally`, ` animations`.
- d=5, cos=0.771: `.\nMany sources recommend extra calcium, potassium and/or magnesium`; source ` calcium` → next ` magnesium`; outputs ` calcium`, ` mineral`, ` magnesium`, ` deposit`.

### L11C0

- d=2, cos=0.966: ` or team building event, our skilled event planning team will`; source ` skilled` → next ` team`; outputs ` staff`, ` team`, ` crew`, ` technicians`.
- d=2, cos=0.978: ` Saadawi. Having undertaken multiple Gender Election Monitoring`; source ` multiple` → next ` Monitoring`; outputs ` campaigns`, ` projects`, ` initiatives`, ` tests`.
- d=1, cos=0.977: `�t must be in an official composition kind`; source ` official` → next ` kind`; outputs ` document`, ` broch`, ` publication`, ` paper`.
- d=2, cos=0.940: ` 1894 in the midst of a stuttering progression towards`; source ` a` → next ` progression`; outputs ` era`, ` campaign`, ` situation`, ` period`.
- d=2, cos=0.978: ` entrance was only the start of the STEAM event,`; source ` the` → next ` event`; outputs ` process`, ` journey`, ` saga`, ` debate`.
- d=3, cos=0.953: `chenko bid farewell to painting with a triptych entitled �`; source ` a` → next ` entitled`; outputs ` approach`, ` palette`, ` theme`, ` attitude`.

### L3C3

- d=1, cos=0.884: ` materials:\n- White 4×4”`; source `×` → next `�`; outputs ` dinosaur`, ` coverage`, ` wrap`, ` Blank`.
- d=1, cos=0.898: `\nTel: +39 059 349811`; source `59` → next `498`; outputs `poly`, `Absolutely`, ` UN`, ` tracing`.
- d=1, cos=0.934: `monitor.c\nindex 873108b..`; source `73` → next `b`; outputs ` Bermuda`, `Tang`, `Prince`, ` 88`.
- d=1, cos=0.938: `adawi. Having undertaken multiple Gender Election Monitoring`; source ` Gender` → next ` Monitoring`; outputs ` disorder`, ` secular`, ` passion`, ` heritage`.
- d=1, cos=0.874: `When you perform a search through MyMapsWizard`; source ` My` → next `W`; outputs ` bid`, ` sleep`, ` sometime`, ` wider`.
- d=1, cos=0.876: `.<|endoftext|>All NuHeat Electric Tile Floor Warming`; source ` Tile` → next ` W`; outputs ` ho`, `:(`, ` Erie`, ` hegemony`.

### L8C2

- d=19, cos=0.816: ` 31\nStartup Command File 33\nConfiguring the Interface with PI ICU 33\nManual Maintenance of the Startup Command File 34\n`; source ` 33` → next ` 34`; outputs ` 33`, ` 34`, ` 32`, ` 35`.
- d=21, cos=0.927: ` SIGINT so exit() is reached and flushes stdout */\n+ /* we handle SIGINT and SIGTERM so exit() is reached`; source ` reached` → next ` reached`; outputs ` reached`, ` reach`, ` reaching`, ` Reach`.
- d=23, cos=0.934: `.\nDo not take Thyrogen if you are pregnant. Please consult your doctor if you are pregnant or think you may be pregnant.\nThyrogen should`; source `rogen` → next `rogen`; outputs `rogen`, ` hydrogen`, ` auto`, ` Down`.
- d=14, cos=0.942: ` pm<|endoftext|>Brian Clarke: Microcosm (Stained Glass and Paintings)\nBrian Clarke: Microcosm`; source `cos` → next `cos`; outputs `cos`, ` cos`, `Cos`, ` Cos`.
- d=14, cos=0.940: ` and Yom Tov minyan this week. Erev Shavuos will be the first minyan and`; source `yan` → next `yan`; outputs `yan`, ` Cyan`, ` cyan`, ` blue`.
- d=54, cos=0.945: ` shed light on how altrenogest entered Alligator Blood� …  last month showed an irregularity to altrenog`; source `og` → next `og`; outputs `og`, `OG`, `ogging`, `ogged`.

### L2C3

- d=1, cos=0.806: `. Then don’t worry, we�`; source ` worry` → next ` we`; outputs `ione`, ` keep`, ` currents`, ` regarding`.
- d=1, cos=0.956: ` all first reports that we get turn out to`; source ` turn` → next ` to`; outputs `473`, ` Merlin`, ` ignition`, ` ushered`.
- d=1, cos=0.879: ` there isn’t much hope for a`; source ` hope` → next ` a`; outputs `hesis`, ` persuade`, ` easing`, `Italy`.
- d=1, cos=0.817: `, you’ll become immersed in the game`; source ` immersed` → next ` the`; outputs ` somewhere`, `Looking`, ` indoors`, ` immersed`.
- d=1, cos=0.947: ` your inbox<|endoftext|>Hey everyone, today is my client`; source ` today` → next ` my`; outputs `February`, `December`, `January`, `April`.
- d=1, cos=0.975: `o de Camichín by age and sex`; source ` age` → next ` sex`; outputs `chron`, ` Requires`, ` unemployed`, ` Veronica`.

### L13C0

- d=6, cos=0.925: ` English Literature.\nwww.*** ;s Financial Services Position: Teacher /`; source `***` → next ` Teacher`; outputs `***`, ` ***`, `****`, `**`.
- d=2, cos=0.756: `.\nIt truly is your (intercourse) living`; source ` (` → next `)`; outputs `)`, `),`, `).`, `.)`.
- d=16, cos=0.846: ` said that the Bush administration has "never ruled out the possibility of U.N. inspectors playing some type of role in the`; source ` "` → next ` in`; outputs `"`, `."`, `".`, `",`.
- d=17, cos=0.637: ` and SOCKS5 proxies (which change your IP address but don’t encrypt your data to provide faster speeds).\n`; source ` (` → next `).`; outputs `)`, `),`, `).`, `.)`.
- d=16, cos=0.814: ` soon as we find them. "There's strong evidence and no question about the fact there are weapons of mass destruction," Powell`; source ` "` → next `,"`; outputs `."`, `"`, `",`, `".`.
- d=24, cos=0.976: ` an expansive spectrum of red fruit — red raspberry, red plum, pomegranate and red currant, grounded by delicate spice and a subtle dustiness.�`; source ` —` → next `.`; outputs ` —`, `—`, ` ―`, ` –`.

### L14C3

- d=1, cos=0.959: ` lack of knowledge for your desires causes him to act`; source ` causes` → next ` to`; outputs ` problems`, ` abnormalities`, ` damage`, ` delays`.
- d=1, cos=0.954: ` is Christmas. Each week we invite you to join`; source ` invite` → next ` to`; outputs ` into`, `into`, ` onto`, ` INTO`.
- d=0, cos=0.956: ` pocket-friendly device lets him print out 2`; source ` print` → next ` out`; outputs ` out`, ` copies`, ` off`, ` copied`.
- d=0, cos=0.906: ` have a happy relationship if I refer you to`; source ` refer` → next ` you`; outputs ` to`, ` To`, ` refer`, ` towards`.
- d=0, cos=0.973: ` Angus Fraser and Co. was described as`; source ` described` → next ` as`; outputs ` as`, ` by`, ` vividly`, ` accurately`.
- d=2, cos=0.933: `iest pregnancy,\nIf you found this article helpful`; source ` found` → next ` helpful`; outputs ` tucked`, ` hidden`, ` somewhere`, ` similarities`.

### L4C0

- d=2, cos=0.952: ` free utilities\n- How to stop MyMapsWizard`; source ` stop` → next `W`; outputs ` mistakes`, ` unnecessary`, ` cancer`, ` pain`.
- d=1, cos=0.958: ` I knelt\nBeside my mother`; source `ide` → next ` mother`; outputs ` confines`, ` deadlines`, ` posted`, ` walls`.
- d=2, cos=0.981: `\nWhen you perform a search through MyMapsWizard`; source ` through` → next `W`; outputs ` various`, ` channels`, `Sphere`, ` boxes`.
- d=1, cos=0.955: ` a CP, but it really matters finding a CP`; source ` matters` → next ` a`; outputs ` staying`, ` because`, ` quality`, ` keeping`.
- d=2, cos=0.893: `\n• Model 4133: like 4132 but with`; source ` like` → next ` but`; outputs `!,`, ` competitors`, ` other`, ` namesake`.
- d=1, cos=0.967: ` from viewers at home. As for this morning's`; source ` for` → next ` morning`; outputs ` other`, ` criticism`, ` question`, ` specifics`.

### L14C1

- d=3, cos=0.910: `Many sources recommend extra calcium, potassium and/or magnesium`; source ` potassium` → next ` magnesium`; outputs `flash`, ` Fran`, ` SF`, ` linger`.
- d=5, cos=0.948: `.\nMany sources recommend extra calcium, potassium and/or magnesium`; source ` calcium` → next ` magnesium`; outputs ` Hunt`, ` Sight`, ` Danny`, ` BAT`.
- d=1, cos=0.902: `, surrounded by the mass of flowers and organic pattern`; source ` flowers` → next ` organic`; outputs ` ARM`, ` Klaus`, ` Winchester`, `isbury`.
- d=1, cos=0.889: ` to achieve equality, justice, peace and development.`; source ` peace` → next ` development`; outputs ` Roberto`, ` Shanahan`, `Jay`, ` Morgan`.
- d=1, cos=0.882: `o de Camichín by age and sex`; source ` age` → next ` sex`; outputs ` rubber`, ` Hendricks`, ` Brenda`, ` Sudan`.
- d=1, cos=0.829: ` access to Manhattan and to nearby cultural and recreational amenities`; source ` cultural` → next ` recreational`; outputs ` flare`, ` tractor`, ` Schmidt`, ` inflammation`.

### L1C2

- d=0, cos=0.980: ` goodness of Android Lollipop, L is swe`; source ` L` → next ` is`; outputs ` in`, ` take`, ` really`, ` made`.
- d=0, cos=0.983: ` MyMapsWizard search with Zemana`; source ` Z` → next `em`; outputs ` in`, ` just`, ` made`, ` will`.
- d=0, cos=0.982: ` zone 4 and RHS zone H5,`; source ` H` → next `5`; outputs ` in`, ` made`, ` take`, ` that`.
- d=0, cos=0.979: ` of Valencia, Mr. Juan Piqu`; source ` P` → next `iqu`; outputs ` in`, ` made`, ` take`, ` really`.
- d=0, cos=0.980: `and far more in our dry Saskatchewan climate)`; source ` Saskatchewan` → next ` climate`; outputs ` just`, ` in`, ` that`, ` really`.
- d=0, cos=0.981: `, or call (757) 563-`; source ` 5` → next `63`; outputs ` really`, ` that`, ` in`, ` to`.

### L2C2

- d=1, cos=0.859: `. But I’m glad I didn`; source ` glad` → next ` didn`; outputs ` fair`, ` But`, ` am`, ` bother`.
- d=1, cos=0.790: `unkett said it wouldn't matter to him`; source ` matter` → next ` him`; outputs ` isn`, `ilial`, ` whether`, ` Kang`.
- d=1, cos=0.833: ` is Christmas. Each week we invite you to join`; source ` invite` → next ` to`; outputs ` to`, ` gently`, ` Clar`, `operation`.
- d=1, cos=0.934: ` from viewers at home. As for this morning's`; source ` for` → next ` morning`; outputs ` specific`, ` sticking`, ` stone`, ` given`.
- d=2, cos=0.919: ` facing a demographic disaster.\nWithin five years there will`; source `Within` → next ` there`; outputs ` Alone`, ` scenarios`, `Veh`, `terms`.
- d=1, cos=0.927: ` I knelt\nBeside my mother`; source `ide` → next ` mother`; outputs ` fades`, `ettel`, ` unpleasant`, ` evils`.

## Guardrail

A candidate can be a coherent route mode without admitting a short human semantic label. Unexplained candidates remain valid geometric objects; broad labels will be accepted only if frozen here and replicated on fresh rows.
