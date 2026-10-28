# Actual source execution versus intrinsic local concentration

Same states, source spans and per-site raw norms across payload families. No fitted decoder; all outputs are evaluated.

| Local function | Actual16 | Shuffled16 | Gaussian16 | Actual site-mean16 | Shuffled site-mean16 | Gaussian site-mean16 |
|---|---:|---:|---:|---:|---:|---:|
| native | 0.451 | 0.277 | 0.262 | 0.302 | 0.255 | 0.245 |
| joint | 0.451 | 0.277 | 0.262 | 0.302 | 0.255 | 0.245 |
| right0 | 0.277 | 0.228 | 0.222 | 0.235 | 0.226 | 0.230 |
| right1 | 0.292 | 0.232 | 0.227 | 0.238 | 0.226 | 0.229 |
| right2 | 0.278 | 0.223 | 0.220 | 0.235 | 0.225 | 0.229 |
| down0 | 0.330 | 0.224 | 0.230 | 0.245 | 0.222 | 0.229 |
| down1 | 0.345 | 0.228 | 0.223 | 0.250 | 0.222 | 0.228 |
| down2 | 0.330 | 0.227 | 0.230 | 0.245 | 0.220 | 0.231 |
| direction0 | 0.348 | 0.240 | 0.239 | 0.266 | 0.236 | 0.237 |
| direction1 | 0.371 | 0.241 | 0.234 | 0.271 | 0.239 | 0.236 |
| direction2 | 0.353 | 0.238 | 0.238 | 0.270 | 0.239 | 0.238 |

Locked gates: {'execution_alignment': True, 'context_pairing': True, 'marginal_anisotropy': False, 'intrinsic_operator': False}. Joint invariant metric change: 3.53e-06.

All9 heads and full1152-dimensional outputs; each local function targets its own responses. Report pooled-energy, uniform-head and uniform-site averages, paired row uncertainty, and per-draw controls. Norm matching controls state-dependent payload magnitude, not all functional response-energy weighting. No semantic or language-quality claim.
