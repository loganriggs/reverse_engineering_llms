# A bracket-response group needs two axes once brace edits are included

The [unseen token-pair test](UNSEEN_TOKEN_EDIT_TRANSFER_RESULTS.md)
found that the frozen one-axis parenthesis/square gauge misses most of
the brace→parenthesis head path, even though the fixed rank-four cut
retains it. We tested whether that reflects a second direction within
the same reduced head rather than a failed rank-four projector.

At layer 13 head 8, fit inside the **unchanged weight-only rank-four
subspace** using 80 original parenthesis/square edits and 12 brace
edits from “Try” or “Compute” contexts (six brace→parenthesis, six
brace→square). The additional edits provide input-side supervision but
no output answers or gradients. We compared raw edit covariance with
an equal-token-pair-group measure that normalizes and averages the
three swap-type covariances. We froze the resulting rank-one, rank-two,
and rank-three projectors, then tested six unseen “If” brace contexts
and twelve old-pair prompts in other formats. The references are finite
rank-four head patches into the native downstream model.

| Held-out panel | Old one-axis gauge | New raw rank one | New balanced rank one | New raw rank two | New balanced rank two |
|---|---:|---:|---:|---:|---:|
| Brace→parenthesis, 3 contexts | .875 | .908 | 1.127 | **.0009** | **.0008** |
| Brace→square, 3 contexts | .174 | .147 | .020 | **.0009** | **.0010** |
| Parenthesis→square, 12 new formats | .021 | .027 | .275 | **.0069** | **.0071** |

Each cell is mean absolute finite path error divided by mean absolute
rank-four path magnitude; lower is better. A rank-one choice can favor
one or two contrasts, but misses another. Both rank-two fits retain all
three swap-type paths on these held-out contexts. The balanced edit
covariance has eigenvalues approximately `.635, .365, .00050, .00004`;
the old parenthesis/square-only response has **99.2%** of its edit
energy in one axis. Thus changing which token identities are excited
changes the apparent dimensionality of the head's response manifold.

This is evidence for a **two-axis response group**, not proof of a
general abstract bracket feature. The model fails the expected
brace-closing answer comparison on all 18 predeclared brace prompts;
the finite brace paths are real but are not shown to support a
competent brace task. The holdout contexts reuse the brace token
identities added during fitting. More bracket identities, prompt
formats, and a competent downstream behavior would be needed to call
the group semantic. The one-axis parenthesis/square result remains
valid on its narrower domain.

For the canonicalization program, this is a concrete reason to allow
small groups or subspaces after reduction: optimizing a single sparse
coordinate can make a local contrast legible while excluding a second
direction needed to represent a broader input family. The group size
must be validated by finite effects and identity holdouts, rather than
chosen from an activation spectrum alone.

Reproduce with `flagship_bracket_edit_span.py`; fit spectra, frozen
test cases, and all finite effects are in
`full_width_results/bracket_edit_span.json`.
