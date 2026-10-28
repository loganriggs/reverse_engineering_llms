## H673 — Carried content has opposing readouts across heads

H672 traced most head4 suppression to the carried value. H673 compares all nine matched carried-value/output-projection maps at layer14 on the same128 contexts (16 unique input strings), covering four domains and copy/task instructions. Each head uses its own128-dimensional slice of the persistent bus and its corresponding projection columns, with the native mixing scalar4.625. This changes both slice and projection, so it does not identify either factor alone.

We calculate the direct residual/final-readout derivative, including residual scaling and final normalization/saturation. Native attention coefficients are deliberately absent: the measurement describes the direction available to each head, not its actual weighted contribution. No new whole-model forwards are required. Slopes are also divided by projected residual norm to compare directions at equal size.

| Head | Negative input slope /128 | Mean input slope | Unit-norm input slope ×10⁴ | Unit-norm other-input−current margin ×10⁴ |
|---|---:|---:|---:|---:|
|0|0|1.0587|2.3830|-2.0783|
|1|0|0.5393|1.9141|-1.5780|
|2|60|0.0231|0.1395|-0.1267|
|3|128|-1.2094|-1.6703|0.6675|
|4|128|-3.1801|-3.9139|3.9910|
|5|104|-0.0778|-0.3991|0.1154|
|6|128|-1.0297|-3.2217|2.3844|
|7|0|1.2553|3.2868|-1.0423|
|8|24|0.1416|0.5076|-0.1042|

Heads0,1,7 boost the current input in every context; heads3,4,6 suppress it in every context. Head4 has the strongest mean input suppression and strongest mean other-input-minus-current margin after residual norm matching, but suppression is not unique to this head. Head3 is less selective against current inputs relative to other domain inputs despite consistently negative current logits.

For head4,32 fixed seed673 signed coordinate permutations scramble its carried-value coordinates before applying its unchanged output projection. Each permutation is shared across contexts. This preserves the head-space value norm; dividing the resulting slopes by projected residual norm also removes output-size differences. These are off-manifold alignment controls, not natural alternative heads or a population null distribution.

| Family | Native unit margin ×10⁴ | Smallest scrambled mean ×10⁴ | Largest scrambled mean ×10⁴ |
|---|---:|---:|---:|
|plural|4.9227|-0.8195|0.5664|
|antonym|3.1000|-0.2709|0.3168|
|capital|4.6180|-0.5027|0.6565|
|successor|3.3234|-0.5132|0.7795|

Native head4 alignment exceeds every scrambled family mean in all four domains. This supports structured alignment of carried content with its learned readout, rather than a consequence of vector size alone. Repeated contexts share inputs and all four families were previously inspected; do not treat128 contexts as independent discoveries or infer statistical significance from the control rank.

The revised leading hypothesis is that multiple heads provide opposing readouts of carried input content, with context-dependent signed gates determining the net effect. This is a candidate mechanism rather than an established circuit: the actual gates can reverse signs, other token positions contribute, local values also contribute, and later branches can change the response. Calling head4 a dedicated successor feature remains unsupported.

Next measure actual coefficient-weighted input-token writes across the nine heads, separate carried and local contributions, and compare their sums under matched copying/task instructions. This will test whether the available opposing directions participate in native cancellation or whether gating makes most irrelevant. Retain full-head/input-token distinctions and follow any direct-readout finding with finite interventions before claiming necessity or behavioral benefit.

Audits: head4 carried vector replays H672 with maximum relative error2.904e-8; selected direct slopes replay with maximum absolute error3.224e-8. All five tracked hashes unchanged, finite outputs. Zero new whole-model forwards. FineWeb26,112 unique documents unchanged. Prior turn was progress; broad research goal active and incomplete.
