# Preregistration: unseen-identity red team for the embedding router

The first router red team found that a linear map from the model's native token
embedding to the rank-16 context register outperformed exact token lookup on
FineWeb, WikiText, and TinyStories. Because the embedding width is 1,152, this
could still be a high-dimensional identity code fitted to recurring tokens.

This follow-up freezes the earlier data ranges, tensor core, ridge penalty, and
site positions. Before inspecting unseen-token results, it locks four controls:

1. score tokens absent from the discovery half separately from recurring tokens;
2. compare with coarse-token and exact-token-table routers on that subset;
3. refit 32 ridge maps after permuting context responses within coarse-token and
   absolute-position cells; and
4. restrict the embedding input to the tensor's 16 context directions, truncate
   the fitted map by singular rank, and measure canonical input-span alignment.

The matched-response null preserves embedding inputs, context-response
distribution, coarse class, absolute position, sample size, and regression
capacity. It destroys which lexical embedding receives which response.

Success on unseen identities would support a reusable lexical geometry for
operator routing. Success only on recurring identities would downgrade the map
to a compressed lookup mechanism. In either case, it predicts a prior over the
local operator, not the full site-specific operator.
