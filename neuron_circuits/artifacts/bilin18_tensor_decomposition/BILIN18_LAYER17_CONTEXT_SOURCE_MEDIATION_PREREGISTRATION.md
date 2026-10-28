# Preregistration: attention-mediated source red team

The finite source decomposition found that Layer-17 carried-state variation,
not the direct `8×` embedding reinjection, selects almost all of the compact
context operator. That decomposition treated the already-computed final
attention output as a third source. Direct embedding can also influence the
attention query/key/value computation, so holding attention fixed may
understate its total local effect.

This follow-up replaces the selected-position embedding or carried term by a
position-conditioned discovery mean, or cyclically substitutes it from another
batch example. It evaluates each edit twice: once with native attention held
fixed and once after exactly recomputing Layer-17 attention with the edited
mixed state and native persistent value bus. Other positions and every earlier
block remain native.

This localizes the final Layer-17 injection boundary. It does not remove the
same embedding from earlier layers, and therefore cannot say that lexical
identity is unimportant upstream.
