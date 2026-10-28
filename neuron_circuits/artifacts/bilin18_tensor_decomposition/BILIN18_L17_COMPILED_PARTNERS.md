# Compiled fixed quadratic forward programs

| Output directions | Normalized linear reads | Stored scalars | Native storage fraction | Natural-write R2 | Actual full-response fidelity | Angle full-response fidelity |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 84 | 131172 | 0.0082 | 0.848 | 0.921 | 0.809 |
| 16 | 288 | 1683360 | 0.1057 | 0.933 | 0.945 | 0.860 |

Locked gates: {'compiler_equivalence': True, 'compact_actual_forward': False, 'compact_universal_forward': False}.

Adaptive H247 follow-up on same64 discovery/test documents and frozen context/output bases. No new optimization. Exact full RMS normalization retained. Fixed quadratic program stores read basis, read offset, reduced forms, linear terms, output vectors and constant full write. No native MLP weights required for inference. Normalization still reads all1152 raw inputs. No semantics/minimality or downstream sufficiency claim without testing.
