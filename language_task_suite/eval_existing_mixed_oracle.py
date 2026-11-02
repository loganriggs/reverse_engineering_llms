"""Score an existing mixed-rule tensor-transformer against the observable DGP.

The generator used here is the original Python LanguageGenerator. A saved
checkpoint may have been trained with the separate FastLanguageGenerator;
the config must be inspected before treating this as an on-distribution score.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mixed_rule_oracle import observable_distributions_for_sequence

DEFAULT_SOURCE = '/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits'


def candidate_info(gen, context):
    candidates = []
    for cls, fn in [('bigram', gen._try_bigram),
                    ('trigram', gen._try_trigram),
                    ('skip_bigram', gen._try_skip_bigram),
                    ('induction', gen._try_induction)]:
        token, _ = fn(context[-gen.window_size:])
        if token is not None:
            candidates.append((cls, token))
    return len(candidates), len({token for _, token in candidates}) > 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('config')
    p.add_argument('--source', default=os.environ.get('ATTN_CIRCUITS_DIR', DEFAULT_SOURCE))
    p.add_argument('--n', type=int, default=256)
    p.add_argument('--length', type=int, default=64)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    sys.path.insert(0, str(Path(a.source).resolve()))
    from generator import LanguageGenerator, ALL_RULES, DEFAULT_MIXING
    from model import BilinearGPT, Config

    config_json = json.loads(Path(a.config).read_text())
    assert config_json['vocab_size'] == 26 and config_json['rules'] == 'all'
    assert a.length <= config_json['seq_len']
    keys = {field.name for field in fields(Config)}
    config = Config(**{k: v for k, v in config_json.items() if k in keys})
    device = torch.device(a.device)
    torch.set_num_threads(4)
    model = BilinearGPT(config).to(device)
    model.load_state_dict(torch.load(a.checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    gen = LanguageGenerator(ALL_RULES, mixing_weights=DEFAULT_MIXING,
                            mode='mixed', seed=1703)
    all_ids, all_labels = gen.sample_batch(a.n, a.length)
    x = torch.tensor([row[:-1] for row in all_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = torch.cat([model(batch)[0].float().cpu()
                            for batch in x.split(32)])
        logp = F.log_softmax(logits, dim=-1).numpy()
    grouped = defaultdict(list)
    n_supported = 0
    for b, ids in enumerate(all_ids):
        tokens = [gen.id2token[i] for i in ids]
        distributions = observable_distributions_for_sequence(gen, tokens)
        for t in range(1, a.length):
            q = np.zeros(config.vocab_size, dtype=np.float64)
            for token, mass in distributions[t].items():
                q[gen.token2id[token]] = mass
            assert q[ids[t]] > 0
            n_supported += 1
            entropy = -np.sum(q[q>0]*np.log(q[q>0]))
            cross_entropy = -np.dot(q, logp[b, t-1])
            kl = cross_entropy-entropy
            assert kl > -1e-5, (b, t, kl)
            observed_ce = -logp[b, t-1, ids[t]]
            n_active, conflicting = candidate_info(gen, tokens[:t])
            label = all_labels[b][t]
            row = (float(kl), float(cross_entropy), float(entropy),
                   float(observed_ce))
            grouped['all'].append(row)
            grouped[f'active_{n_active}'].append(row)
            grouped['conflict' if conflicting else 'no_conflict'].append(row)
            if label.startswith(('bigram_', 'trigram_', 'skip_', 'induction_')):
                grouped['ordinary_rule_label'].append(row)
            elif label.startswith('bracket_'):
                grouped['bracket_label'].append(row)
            else:
                grouped[label].append(row)
    summary = {name: dict(n=len(rows), kl=np.mean([r[0] for r in rows]),
                          expected_ce=np.mean([r[1] for r in rows]),
                          oracle_entropy=np.mean([r[2] for r in rows]),
                          sampled_ce=np.mean([r[3] for r in rows]))
               for name, rows in grouped.items()}
    output = dict(config=vars(a), model_config=dict(config_json),
                  exact_observed_token_support=n_supported,
                  results=summary,
                  caveat='Original Python generator evaluation. Checkpoint training '
                         'may have used FastLanguageGenerator or another source revision; '
                         'this is an empirical score on the specified Python DGP.')
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')
    print({k: {m: round(v[m], 4) for m in ('kl', 'oracle_entropy', 'sampled_ce')}
           for k, v in summary.items() if k in ('all', 'conflict', 'no_conflict')})


if __name__ == '__main__':
    main()
