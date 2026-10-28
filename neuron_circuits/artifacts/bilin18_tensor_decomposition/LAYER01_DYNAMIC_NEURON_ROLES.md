# Factor roles of dynamically selected Layer-0/1 neurons

For the top-256 native channels by exact local-write contribution, each finite product change is split into scale-invariant left-cross, right-cross, and quadratic terms. Dominance is invariant to reciprocal factor scaling; swapping a neuron's factors flips the side name but preserves dominance and cross-example agreement. The same lexical pairs are compared across FineWeb and TinyStories.

## Locked hypotheses

- **R0_top_channels_are_factor_asymmetric: supported.**
- **R1_balanced_channels_are_a_minority: supported.**
- **R2_recurring_channels_keep_their_factor_role: not supported.**
- **R3_channel_identity_is_contextual: supported.**
- **R4_factor_role_transfers_when_identity_recurs: not supported.**
- **R5_contribution_sign_transfers_when_identity_recurs: supported.**
- **R6_quadratic_term_is_negligible_per_channel: supported.**

| Layer | Corpus | Single-sided ≥.8 | Balanced ≤.6 | Quadratic share | Cancellation | Recurring channels / side agreement |
|---:|---|---:|---:|---:|---:|---:|
| 0 | fineweb | 0.533 | 0.153 | 0.0074 | 0.216 | 857 / 0.667 |
| 0 | tinystories | 0.524 | 0.158 | 0.0084 | 0.211 | 840 / 0.667 |
| 1 | fineweb | 0.550 | 0.144 | 0.0085 | 0.228 | 812 / 0.667 |
| 1 | tinystories | 0.550 | 0.141 | 0.0082 | 0.231 | 789 / 0.667 |

| Layer | Cross-corpus pair Jaccard | Intersection size | Side agreement | Sign agreement |
|---:|---:|---:|---:|---:|
| 0 | 0.087 | 41 | 0.732 | 1.000 |
| 1 | 0.041 | 20 | 0.522 | 1.000 |

## Post-run selection-artifact red-team

The locked role hypotheses selected channels by exact local contribution. The table compares that population with activation-selected, fixed-random, and all channels. This diagnostic was added after the first run.

| Layer | Corpus | Single-sided contribution / activation / random / all | Quadratic share contribution / all | Positive contribution fraction |
|---:|---|---:|---:|---:|
| 0 | fineweb | 0.543 / 0.516 / 0.480 / 0.477 | 0.0078 / 0.0084 | 0.974 |
| 0 | tinystories | 0.523 / 0.508 / 0.477 / 0.476 | 0.0079 / 0.0094 | 0.970 |
| 1 | fineweb | 0.543 / 0.539 / 0.473 / 0.479 | 0.0079 / 0.0094 | 0.976 |
| 1 | tinystories | 0.555 / 0.547 / 0.480 / 0.481 | 0.0087 / 0.0097 | 0.974 |

## Concrete FineWeb examples

### Layer 0

- ** great-> terrible** in ` businesses like this.
Great Concept!. This is such a great idea - I love it! Every time you get your car washed, you are`
  - channel 3819: projection +0.024, left dominance 0.581, cross terms +8.52/+6.14; embedding-nearest edit/gate tails ' Im', 'Were', ' Were' / 'Am', ' im', ' isn'
  - channel 710: projection +0.010, right dominance 0.943, cross terms -0.796/+13.2; embedding-nearest edit/gate tails ' do', 'have', ' Do' / ' Would', ' would', ' Wouldn'
  - channel 2983: projection +0.010, right dominance 0.702, cross terms -3.68/-8.69; embedding-nearest edit/gate tails 'aug', ' Stark', ' ferry' / ' Caval', ' fought', ' armies'

- ** king-> queen** in ` on a hot night was enough to send me running for my king-sized bed. John got to stay and have the delightful "bonding"`
  - channel 3790: projection +0.045, left dominance 0.551, cross terms +18.5/+15.1; embedding-nearest edit/gate tails 'urb', 'outhern', 'Car' / ' borne', ' extant', ' scarce'
  - channel 874: projection +0.035, right dominance 0.732, cross terms +7.59/+20.7; embedding-nearest edit/gate tails " '[", ' \\"', ' "[' / '"', "''", '"['
  - channel 1144: projection +0.019, left dominance 0.671, cross terms -13.9/-6.81; embedding-nearest edit/gate tails 'usual', ' also', '5' / '.–', '"—', '─'

- ** do-> did** in ` access in accordance with the UK data protection legislation.
We do not collect sensitive information about you except when you specifically knowingly provide it. In order`
  - channel 869: projection +0.010, right dominance 0.595, cross terms +6.19/+9.1; embedding-nearest edit/gate tails 'inning', 'hips', 'planes' / ' originally', ' 2006', ' 2600'
  - channel 4073: projection +0.010, right dominance 0.629, cross terms -4.63/-7.86; embedding-nearest edit/gate tails ' Stephenson', ' Cousins', ' Burgess' / '519', 'trop', ' bron'
  - channel 80: projection +0.009, left dominance 0.846, cross terms -11.3/-2.05; embedding-nearest edit/gate tails ' `', 'Â', ' «' / 'activ', ' Policy', ' Chick'

- ** cold-> hot** in ` be.
And the raw silk? Well, it was cold that day and a short sleeved shirt wasn't going to cut it. And`
  - channel 1152: projection +0.018, left dominance 0.546, cross terms +6.16/+5.12; embedding-nearest edit/gate tails ' creating', ' make', ' playing' / ' create', ' creating', 'Cut'
  - channel 3883: projection +0.011, left dominance 0.528, cross terms +3.26/+2.92; embedding-nearest edit/gate tails ' have', ' hath', ' Have' / ' Are', ' Can', ' must'
  - channel 1100: projection +0.009, right dominance 0.569, cross terms +3.8/+5.03; embedding-nearest edit/gate tails ' And', 'And', ' ans' / ' And', ' Or', 'And'
### Layer 1

- ** great-> terrible** in ` businesses like this.
Great Concept!. This is such a great idea - I love it! Every time you get your car washed, you are`
  - channel 2501: projection +0.018, left dominance 0.898, cross terms -2.09/-0.238; embedding-nearest edit/gate tails ' Benefits', '', 'ioned' / ' slapped', ' knew', ' shrugged'
  - channel 2766: projection +0.012, left dominance 0.986, cross terms +2.4/+0.0342; embedding-nearest edit/gate tails ' Ame', ' waiver', ' decentral' / ' blatantly', ' CHAR', 'Song'
  - channel 2975: projection +0.009, left dominance 0.604, cross terms +1.29/+0.847; embedding-nearest edit/gate tails '678', ' Indiana', '389' / ' Aviv', ' Connecticut', ' Ont'

- ** king-> queen** in ` on a hot night was enough to send me running for my king-sized bed. John got to stay and have the delightful "bonding"`
  - channel 2302: projection +0.019, left dominance 0.552, cross terms -1.06/-0.858; embedding-nearest edit/gate tails ' Such', 'Such', '))' / ' (', ' [', 'atton'
  - channel 4491: projection +0.019, left dominance 0.572, cross terms +1.02/+0.762; embedding-nearest edit/gate tails ' differs', ' govern', ' differ' / ' briefing', 'oda', ' circle'
  - channel 2159: projection +0.013, right dominance 0.649, cross terms -0.593/-1.1; embedding-nearest edit/gate tails '�', ' "', '�' / ' Ventura', ' Vendor', 'ATIVE'

- ** do-> did** in ` access in accordance with the UK data protection legislation.
We do not collect sensitive information about you except when you specifically knowingly provide it. In order`
  - channel 2501: projection +0.070, left dominance 0.909, cross terms -10.9/-1.09; embedding-nearest edit/gate tails ' Benefits', '', 'ioned' / ' slapped', ' knew', ' shrugged'
  - channel 244: projection +0.019, right dominance 0.550, cross terms +3.49/+4.26; embedding-nearest edit/gate tails 'erenn', ' newspaper', ' distur' / 'Sold', 'dr', 'vor'
  - channel 1754: projection +0.017, left dominance 0.703, cross terms +4.2/+1.78; embedding-nearest edit/gate tails 'xs', ' midfielder', 'buf' / 'HER', ' pts', 'uint'

- ** cold-> hot** in ` be.
And the raw silk? Well, it was cold that day and a short sleeved shirt wasn't going to cut it. And`
  - channel 1754: projection +0.024, right dominance 0.580, cross terms -1.06/-1.47; embedding-nearest edit/gate tails ' situational', ' Brookings', ' inviting' / ' prompt', ' reviewed', ' Failure'
  - channel 4387: projection +0.009, left dominance 0.710, cross terms +1.37/+0.56; embedding-nearest edit/gate tails 'reasonable', 'ent', ' reconc' / 'and', 'Based', 'oubt'
  - channel 3696: projection +0.007, left dominance 0.510, cross terms -0.869/-0.834; embedding-nearest edit/gate tails ' Summers', ' Mul', ' worries' / ' summar', ' turrets', ' pins'

Embedding-tail labels are weight-derived clues, not validated semantic names: the MLP reads contextual RMS-normalized states, not isolated token embeddings. The causal evidence is the exact cross-term value and downstream channel contribution.
