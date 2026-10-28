# Exact source response terms

Descriptive exact algebra audit on H241 actual-addition pairs. No fresh replication. Decomposition convention uses final RMS denominator for cross/self terms and initial context for normalization term. Terms are nonorthogonal; energy ratios do not sum to one. Signed projections sum to one but are not unique causal credits.

F(c+h)-F(c)=2B(c,h)/s(c+h)+B(h,h)/s(c+h)+B(c,c)*(1/s(c+h)-1/s(c)); s(z)=mean(z^2)+eps.

| Function | Cross signed projection | Self signed projection | Normalization signed projection | Cross energy / target | Self energy / target | Normalization energy / target |
|---|---:|---:|---:|---:|---:|---:|
| native | 0.879 | 0.127 | -0.006 | 0.812 | 0.048 | 0.009 |
| joint | 0.879 | 0.127 | -0.006 | 0.812 | 0.048 | 0.009 |
| right0 | 0.958 | 0.045 | -0.003 | 0.946 | 0.023 | 0.008 |
| right1 | 0.960 | 0.043 | -0.002 | 0.950 | 0.022 | 0.009 |
| right2 | 0.957 | 0.045 | -0.002 | 0.945 | 0.023 | 0.008 |
| down0 | 0.933 | 0.070 | -0.003 | 0.905 | 0.031 | 0.009 |
| down1 | 0.931 | 0.072 | -0.003 | 0.901 | 0.032 | 0.009 |
| down2 | 0.922 | 0.080 | -0.003 | 0.886 | 0.034 | 0.008 |
| direction0 | 0.900 | 0.103 | -0.003 | 0.848 | 0.042 | 0.008 |
| direction1 | 0.898 | 0.105 | -0.003 | 0.845 | 0.042 | 0.008 |
| direction2 | 0.899 | 0.104 | -0.003 | 0.846 | 0.042 | 0.008 |

Float32 maximum relative replay error 0.000234; float64 native audit (four sites/head, float32 epsilon) 2.86e-13.
