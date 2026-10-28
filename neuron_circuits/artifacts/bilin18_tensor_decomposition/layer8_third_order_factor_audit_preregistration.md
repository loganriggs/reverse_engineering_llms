# Preregistration: QK×OV factors of the singular third-order head

The exceptional third-order paths are localized to one final attention head in
the WikiText singular state. Decompose that head's exact JVP into five additive
weight-native factors:

1. first QK-form derivative routing current-layer values;
2. first QK-form derivative routing the persistent value bus;
3. second QK-form derivative routing current-layer values;
4. second QK-form derivative routing the persistent value bus;
5. the direct derivative of the current-position value.

For each frozen causal failure, use the dominant head from the prior audit and
form the third-order Mobius residual separately for every factor. Matched
controls remain the same. Factor residuals must recompose the dominant-head
residual within 1e-4 relative to the full exact response.

Localization passes at median top absolute share >=0.50 and effective support
<=3. Routing dominance requires at least 0.80 of absolute projection mass in
the four QK-routing terms; direct-value dominance requires current-value share
>=0.50. This isolates the final path factor only.
