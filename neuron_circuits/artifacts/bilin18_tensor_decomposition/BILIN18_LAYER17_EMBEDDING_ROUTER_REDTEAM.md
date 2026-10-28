# Unseen-identity red team for the Layer-17 embedding router

The earlier aggregate embedding result is split by whether the exact token identity occurred in discovery. Matched response permutations preserve coarse class and absolute position while destroying lexical pairing.

## Locked hypotheses

- **E0_full_embedding_router_replicates: supported.** The full embedding router agrees with the prior aggregate operator fidelity within 1e-5 on every corpus.
- **E1_embedding_geometry_generalizes_unseen_tokens: supported.** On token identities absent from discovery, the embedding router exceeds the coarse-class router by at least 0.02 on every corpus.
- **E2_embedding_geometry_beats_response_null: supported.** On unseen token identities, the real embedding router exceeds the p95 of 32 coarse-class-and-position-matched response permutations by at least 0.03 on every corpus.
- **E3_context_span_contains_a_usable_lexical_router: supported.** A ridge router restricted to the 16 tensor context directions reaches at least 0.75 aggregate operator fidelity on every corpus.
- **E4_embedding_router_is_output_low_rank: supported.** A rank-8 truncation of the 16-output embedding map remains within 0.02 operator fidelity of the full map on every corpus.
- **E5_embedding_and_context_input_subspaces_align: not supported.** The mean canonical cosine between the embedding-router input span and tensor context span is at least 0.50.
- **E6_unseen_identity_cost_is_bounded: supported.** Embedding-router fidelity on unseen identities is within 0.15 of fidelity on seen identities on every corpus.
- **E7_geometry_beats_lookup_on_unseen_identities: not supported.** On unseen identities, the embedding router exceeds the exact-token table, which falls back to its mean, by at least 0.05 on every corpus.

Prior-run replication error: `2.91e-08`. Mean canonical cosine between the fitted embedding-input span and tensor context span: `0.260`.

## Seen versus unseen token identities

| Corpus/subset | N | mean | coarse | token table | full embedding | context-span embedding | rank-4 / rank-8 | null p95 (unseen) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fineweb/all | 32000 | 0.711 | 0.790 | 0.846 | 0.881 | 0.832 | 0.875/0.877 | 0.700 |
| fineweb/seen | 21556 | 0.709 | 0.796 | 0.896 | 0.903 | 0.856 | 0.895/0.898 | 0.700 |
| fineweb/unseen | 10444 | 0.719 | 0.776 | 0.719 | 0.825 | 0.773 | 0.826/0.826 | 0.700 |
| wikitext/all | 12800 | 0.721 | 0.798 | 0.808 | 0.850 | 0.809 | 0.847/0.853 | 0.711 |
| wikitext/seen | 7303 | 0.724 | 0.833 | 0.868 | 0.870 | 0.835 | 0.862/0.871 | 0.711 |
| wikitext/unseen | 5497 | 0.716 | 0.743 | 0.716 | 0.819 | 0.768 | 0.824/0.826 | 0.711 |
| tinystories/all | 12800 | 0.736 | 0.809 | 0.893 | 0.895 | 0.850 | 0.892/0.895 | 0.735 |
| tinystories/seen | 9443 | 0.713 | 0.815 | 0.920 | 0.921 | 0.868 | 0.914/0.918 | 0.735 |
| tinystories/unseen | 3357 | 0.808 | 0.788 | 0.808 | 0.814 | 0.790 | 0.822/0.823 | 0.735 |

## Interpretation

A high unseen-token score that beats the matched response null identifies shared geometry, not merely recurring token IDs. The restricted context-span map asks a stronger architectural question: whether the same residual directions that parameterize the tensor context also read lexical identity. Rank truncation concerns the fitted router map, not the exact native operator or the whole model.
