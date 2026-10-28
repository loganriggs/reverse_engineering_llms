# Routing variables and read-function primitives

The nonnegative routing profile is factorized as `token memberships × read-function uses`. A variable can therefore be reused by several layer/head/distance functions, and a token can participate in several variables. Hard argmax assignments are evaluated only as a lossy bridge to the existing typed-edge executor.

## Locked hypotheses

- **F1_routing_profiles_are_low_dimensional: not supported.**
- **F2_read_function_primitives_are_stable: not supported.**
- **F3_soft_variable_memberships_are_stable: supported.**
- **F4_hard_variable_assignments_are_more_stable: not supported.**
- **F5_hard_nmf_variables_are_causally_useful: supported.**
- **F6_hard_nmf_variables_are_prediction_useful: supported.**
- **F7_primitive_uses_are_sparse: not supported.**
- **native_branch_matches_checkpoint: supported.**

Weighted rank-90: `171` / `168`; participation ratio: `10.9` / `10.1`.
Mean/median matched function cosine: `0.786` / `0.854`. Soft membership cosine: `0.723`. Hard assignment NMI: `0.053`.

| Hard view | KL recovery | CE delta | Top-1 |
|---|---:|---:|---:|
| nmf_a | 0.9622 | +0.2537 | 0.7852 |
| nmf_b | 0.9695 | +0.1990 | 0.8086 |

## Candidate variable/function primitives

- **V0** (split function cosine 0.74): tokens '.', ',', '\\n', ' the', ' and', '-', ' a', ' to'; reads L2 H0 d[0,1), L1 H0 d[0,1), L1 H2 d[0,1), L2 H4 d[0,1), L2 H0 d[8,16).
- **V1** (split function cosine 0.66): tokens '.', ',', '\\n', '-', ':', '�', ' A', ' ('; reads L2 H1 d[4,8), L4 H2 d[4,8), L5 H5 d[4,8), L4 H3 d[4,8), L2 H1 d[16,32).
- **V2** (split function cosine 0.92): tokens '�', ' the', '�', '\\n', ' of', ' a', ',', ' to'; reads L0 H3 d[32,128), L1 H3 d[32,128), L3 H7 d[32,128), L2 H8 d[32,128), L0 H5 d[32,128).
- **V3** (split function cosine 0.84): tokens ',', ' the', '�', ' and', ' I', '�', '-', ' is'; reads L0 H0 d[8,16), L2 H5 d[8,16), L1 H4 d[8,16), L0 H4 d[8,16), L3 H8 d[8,16).
- **V4** (split function cosine 0.10): tokens '.', '-', '\\n', ')', ' and', ' it', ' Mac', ','; reads L5 H5 d[8,16), L2 H1 d[8,16), L1 H3 d[8,16), L4 H2 d[8,16), L1 H6 d[8,16).
- **V5** (split function cosine 0.97): tokens '.', '-', '\\n', '�', ',', '�', ' and', ' the'; reads L4 H5 d[1,2), L3 H7 d[1,2), L2 H8 d[1,2), L4 H3 d[1,2), L4 H7 d[1,2).
- **V6** (split function cosine 0.94): tokens ' the', ' to', ',', ' a', ' of', ' and', ' in', ' that'; reads L1 H0 d[4,8), L1 H5 d[4,8), L5 H8 d[4,8), L2 H3 d[4,8), L1 H3 d[4,8).
- **V7** (split function cosine 0.36): tokens '.', '-', ' the', ' a', '�', ',', '�', '\\n'; reads L3 H7 d[2,4), L4 H3 d[2,4), L4 H5 d[2,4), L2 H8 d[2,4), L2 H1 d[2,4).
- **V8** (split function cosine 0.91): tokens ' that', '.', '\\n', ' you', ' it', ' I', ' cells', '<|endoftext|>'; reads L5 H4 d[32,128), L2 H7 d[32,128), L4 H6 d[32,128), L4 H8 d[32,128), L3 H5 d[32,128).
- **V9** (split function cosine 0.68): tokens ' to', ' in', '.', ' of', ' on', ' for', ' with', ','; reads L5 H6 d[4,8), L5 H4 d[4,8), L4 H6 d[4,8), L2 H7 d[4,8), L3 H1 d[4,8).
- **V10** (split function cosine 0.96): tokens '.', ',', ' the', ' and', '\\n', ' of', ' a', ' to'; reads L0 H4 d[2,4), L3 H5 d[2,4), L1 H4 d[2,4), L1 H8 d[2,4), L2 H7 d[2,4).
- **V11** (split function cosine 0.83): tokens ',', ' to', ' the', ' that', ' of', ' and', ' with', ' ('; reads L3 H4 d[8,16), L1 H0 d[8,16), L5 H8 d[8,16), L5 H3 d[8,16), L2 H3 d[8,16).
- **V12** (split function cosine 0.91): tokens ' the', '�', ' and', ' I', ' vision', ' tenants', ' in', ' is'; reads L1 H4 d[16,32), L2 H5 d[16,32), L0 H4 d[16,32), L3 H5 d[16,32), L2 H7 d[16,32).
- **V13** (split function cosine 0.95): tokens ' to', ',', ' of', ' the', ' and', ' in', ' for', ' a'; reads L2 H3 d[2,4), L1 H5 d[2,4), L4 H1 d[2,4), L5 H8 d[2,4), L1 H3 d[2,4).
- **V14** (split function cosine 1.00): tokens ' to', ' of', ' in', ' the', ' a', '�', ' for', ' on'; reads L2 H5 d[0,1), L1 H4 d[0,1), L3 H5 d[0,1), L2 H8 d[0,1), L0 H0 d[0,1).
- **V15** (split function cosine 0.72): tokens ' to', ' is', '�', ' will', ' are', ' out', ' said', ' that'; reads L1 H3 d[8,16), L0 H7 d[32,128), L5 H0 d[32,128), L1 H8 d[32,128), L2 H7 d[32,128).
- **V16** (split function cosine 0.85): tokens ',', '.', ' that', ' to', ':', ' with', ' is', '\\n'; reads L3 H4 d[16,32), L1 H0 d[16,32), L3 H0 d[16,32), L5 H3 d[16,32), L5 H8 d[16,32).
- **V17** (split function cosine 0.93): tokens '.', '\\n', ',', ' and', ' to', ' this', '<|endoftext|>', ' an'; reads L5 H7 d[32,128), L5 H7 d[16,32), L5 H2 d[32,128), L5 H2 d[16,32), L5 H7 d[8,16).
- **V18** (split function cosine 0.56): tokens ' a', ' the', '\\n', '.', ' of', ' your', ' an', ' this'; reads L0 H4 d[1,2), L1 H7 d[1,2), L2 H7 d[8,16), L3 H1 d[8,16), L1 H4 d[1,2).
- **V19** (split function cosine 0.74): tokens ' to', ' in', ',', ' of', ' for', ' and', ' the', ' is'; reads L2 H3 d[1,2), L4 H1 d[1,2), L1 H5 d[1,2), L5 H8 d[1,2), L5 H3 d[1,2).
- **V20** (split function cosine 0.91): tokens ' the', ' a', '.', ',', ' and', ' of', '�', '�'; reads L0 H0 d[4,8), L2 H5 d[4,8), L0 H4 d[4,8), L1 H4 d[4,8), L3 H5 d[4,8).

## MDL interpretation

This decomposition separates four potential code-length charges: declare a variable (one token-membership column), declare a read function (one layer/head/distance pattern), specify the variable–function links, and invoke the resulting route in an example. The current report measures factor rank and 90%-mass support counts but does not yet claim a calibrated bit-level MDL. A primitive is readable only if its token membership and function uses remain stable and admit a short description; causal fidelity alone is insufficient.

## Guardrails

NMF is non-convex and the greedy component matching is not an optimal assignment. Magnitude profiles omit signs and contextual token-state changes. The hard causal evaluation discards all but the largest membership per token, so it can underestimate a valid multiplexed variable representation. Conversely, a good hard score would not prove that NMF components are semantic. The next test should execute soft memberships directly and compare description length at matched behavior.
