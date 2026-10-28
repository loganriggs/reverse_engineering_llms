# Compact input coordinates versus broad interaction partners

| Layer | Strict16 actual | Both16 + full linear actual | Strict16 natural R2 | Strict64 actual | Strict128 actual |
|---|---:|---:|---:|---:|---:|
| L1 | 0.094 | 0.139 | 0.113 | 0.190 | 0.270 |
| L8 | 0.003 | -0.001 | 0.031 | 0.023 | 0.055 |
| L17 | 0.606 | 0.842 | 0.806 | 0.658 | 0.729 |

Locked gates: {'late_actual_code16': False, 'late_universal_code16': False, 'broad_partner_needed': True}.

Strict code: y=P(x-mu), state B(mu+y,mu+y), discovery mean calibrated; response2B(mu,Pdelta)+2B(Pmidpoint,Pdelta). Both-with-full-linear retains2B(mu,delta), so it is not a16-total-read code. Cross also retains broad partners. Same test data, no fresh replication.
