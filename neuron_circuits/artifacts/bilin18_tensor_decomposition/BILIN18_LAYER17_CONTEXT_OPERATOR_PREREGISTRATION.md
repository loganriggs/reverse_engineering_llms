# Preregistration: context-conditioned tangent operator atlas

The rank-16 Tucker program defines a local tangent-to-output operator

```text
A(x)[output, tangent] = sum_context T[output, tangent, context] x_context.
```

This test treats the context coordinates as a control register rather than as
sixteen independent concepts. It asks:

- whether individual context axes have held-out semantic descriptions;
- whether those axes beat rotations of exactly the same context subspace;
- whether each local 16×16 operator is itself low rank;
- how many tensor-ordered context coordinates reproduce the operator family;
- and whether current-token or coarse-token lookup can route the operator
  without access to the full hidden state.

The full state projection is an exact observable router by algebraic
construction and is only an upper bound. Lexical lookup is the nontrivial
compression test. Discovery and external ranges match the output semantic
atlas, with the first and second halves of discovery used for fitting and
confirmation.

Failure of lexical routing does not make the operator arbitrary. It means the
required control state contains broader context than one token or one coarse
class. Failure of axis privilege with successful subspace behavior means the
register is interpretable only jointly.
