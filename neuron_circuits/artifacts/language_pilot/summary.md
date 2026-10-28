# Language neuron-ablation pilot

Model: `Qwen/Qwen2.5-0.5B`. Extraction examples: 9 per task; held-out examples: 9 per task.

| Task | Intact aligned pool | Median 90%-margin circuit (zero) | Median 90%-margin circuit (corpus mean) | Random (zero) | Accuracy after ~0.01% targeted deletion |
|---|---:|---:|---:|---:|---:|
| IOI | 132 | 43.4% | 20.0% | 95.3% | 16.7% |
| Agreement | 99 | 43.4% | 38.7% | 92.2% | 50.0% |
| Bracket closing | 24 | 22.8% | 29.4% | 89.1% | 50.0% |

At 20% retained, corpus-mean ablation was especially helpful for IOI: 100% exact accuracy versus 44.4% under zero ablation. Shared 20% circuits reached 66.7%/88.9% (zero/mean) for IOI, 77.8%/77.8% for agreement, and 100%/88.9% for bracket closing.

Frozen circuits often preserved the exact answer on a changed input (55.6–100%), but retained 90% of the original logit margin much less often (11.1–55.6%). Exact output transfer therefore overstates mechanistic invariance.

Semantic failures were recorded as role reversal, another known name, wrong agreement, wrong closer, opener, or unrelated output; full-vocabulary KL and expected semantic cost are included in every evaluation row.
