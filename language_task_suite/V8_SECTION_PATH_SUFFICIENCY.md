# Are the section attention routes sufficient?

The [necessity screen](V8_SECTION_ATTENTION_PATHS.md) found reproducible,
teacher-dependent attention paths for the v8 section lookup. Necessity does
not imply that those paths alone compute the answer. This follow-up keeps
only selected paths at the affected query positions in the specified layers,
while leaving attention at every other query position and the rest of the
model unchanged. It measures supported section-answer accuracy, cross-entropy
against the exact generator, and fidelity of the key-edit effect.

The [runner](eval_v8_section_attention_paths.py) now accepts `--operation
keep` and multi-layer `--circuit` specifications. A one-edge keep operation
agrees with an independent masked-V computation at the selected query to
maximum absolute error `2.98e-8`. The empty-edge hook also replays the
unmodified model exactly. The keep operation is conditional on *structural*
positions: all section headers, all fact colors, and a 16-token recent
question window. It does not use the ground-truth matching section to
select one header or color.

The first candidate kept only the three dominant source routes in seed 1:
layer-0 query-key→all headers, layer-1 answer→all colors, and layer-2
answer→query-key. On a four-key-edit smoke panel, paired answer accuracy
fell from 1.00 to .50. A larger candidate also retained the recent 16
tokens at the affected query positions. For seed 4, its known earlier
color and key paths were placed in layers 0 and 1 respectively. These
circuits were specified from the necessity screen, then evaluated on the
same two panels used there. These panels are held out from model training,
but route selection was informed by their necessity results, so they are
**not an independent discovery holdout**. Neither compact circuit was
selected using its keep result.

| Teacher, panel | Native paired answer accuracy | Keep-circuit accuracy | Native answer CE | Keep-circuit CE | Native key-effect-to-oracle L1 | Keep-circuit L1 |
|---|---:|---:|---:|---:|---:|---:|
| Seed 1, A | .92 | .68 | .226 | 1.105 | .434 | 1.310 |
| Seed 1, B | .92 | .79 | .204 | .809 | .492 | 1.008 |
| Seed 4, A | .93 | .72 | .237 | 1.145 | .540 | 1.093 |
| Seed 4, B | .88 | .82 | .244 | .839 | .552 | .957 |

The candidate paths plus local question context are **not sufficient** at
the tested budget. They damage both prediction and supported causal edits
in both teachers. This does not contradict the necessity result. Other
attention edges at the answer or query-key position may provide relevant
features, stabilization, or cancellation; restricting several layers at
once may also create interactions absent from single-edge ablations. The
experiment does not isolate which missing edge is responsible, nor prove
that a compact sufficient route set is impossible.

This is a stopping point for treating the current hand-picked path set as
an extracted circuit. The useful deliverable for the main featurizer loop is
a set of **causal supervision targets and negative controls**: preserve the
query-key-to-header effect, the teacher-specific answer-to-color layer, and
later answer-to-key effects while monitoring irrelevant-color spillover.
The next comparison should train sparse/shared-dictionary representations
with whole section–color combination holdouts, then test whether their
components preserve these effects and other language rules at matched
teacher fidelity and active-connection budgets. Any claimed circuit needs
separate necessity *and* sufficiency evidence.

Records: [seed 1 A](results/v8_section_attention_keep_compact_seed1_panel74001.json),
[seed 1 B](results/v8_section_attention_keep_compact_seed1_panel74002.json),
[seed 4 A](results/v8_section_attention_keep_compact_seed4_panel74001.json), and
[seed 4 B](results/v8_section_attention_keep_compact_seed4_panel74002.json).
