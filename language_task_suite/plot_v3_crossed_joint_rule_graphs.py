"""Compare planted, learned, and weight-path maps for three joint v3 rules.

The noun direct path uses dense embedding/unembedding weights. The two
long-range paths use the saved sparse attention projections and fresh inputs.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, '/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
from model import BilinearGPT, Config
from original_v3_crossed_skip import LanguageV3CrossedSkip
from train_original_v3_shared_bracket import make_batch
from plot_original_v3_sparse_rule_graphs import events
from plot_v3_crossed_skip_sparse_paths import records_for_rules, paths
from original_v3_shared_bracket import bracket_close_distribution


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--skip-path-json', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=99301)
    parser.add_argument('--streams', type=int, default=512)
    parser.add_argument('--per-pair', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    gen = LanguageV3CrossedSkip(seed=args.seed)
    x, y, labels, _ = make_batch(gen, args.streams, 129, device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    assert saved['generator'] == 'LanguageV3CrossedSkip'
    model = BilinearGPT(Config(**saved['config'])).to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    parameters = dict(model.named_parameters())
    for name, mask in saved['mask'].items():
        assert torch.count_nonzero(parameters[name][~mask]) == 0

    # Full-vocabulary probabilities, aggregated only over fresh sampled sites.
    probs = []
    for batch in x.split(32):
        probs.append(model(batch)[0].float().softmax(-1).cpu().numpy())
    probs = np.concatenate(probs)

    nouns = list(gen.categories['NOUN'])
    verbs = list(gen.categories['VERB_T'])
    verb_ids = [gen.token2id[v] for v in verbs]
    noun_truth = np.zeros((len(nouns), len(verbs)))
    noun_full = np.full_like(noun_truth, np.nan)
    noun_counts = []
    for i, noun in enumerate(nouns):
        for verb, probability in gen.noun_bigram_dists[noun]:
            noun_truth[i, verbs.index(verb)] = probability
        sites = [(b, q) for b, row in enumerate(labels) for q, label in enumerate(row)
                 if label == 'noun_bigram' and int(x[b, q]) == gen.token2id[noun]]
        noun_counts.append(len(sites))
        assert sites
        noun_full[i] = np.mean([probs[b, q, verb_ids] for b, q in sites], axis=0)
    direct_logits = (model.embed.weight[[gen.token2id[n] for n in nouns]] @
                     model.lm_head.weight[verb_ids].T)
    noun_direct = direct_logits.softmax(-1).cpu().numpy()

    grouped = records_for_rules(gen, x, y, labels, args.per_pair)
    order = [(a, t) for a in gen.crossed_skip_anchors
             for t in gen.crossed_skip_triggers]
    skip_outputs = list(gen.crossed_skip_outputs)
    skip_ids = [gen.token2id[v] for v in skip_outputs]
    skip_truth = np.zeros((len(order), len(skip_outputs)))
    skip_full = np.zeros_like(skip_truth)
    for i, pair in enumerate(order):
        skip_truth[i, skip_outputs.index(gen.skip_bigrams[pair])] = 1
        skip_full[i] = np.mean([probs[b, q, skip_ids] for b, q, *_ in grouped[pair]], axis=0)
    path_json = json.loads(args.skip_path_json.read_text())
    assert path_json['config']['seed'] == args.seed
    assert path_json['config']['streams'] == args.streams
    assert path_json['config']['per_pair'] == args.per_pair
    assert path_json['config']['sparse'] == str(args.checkpoint)
    skip_path = np.asarray(path_json['matrices']['sparse']['output_logits'])
    assert skip_path.shape == skip_truth.shape

    # In this generator the legacy name "skip_trigram" denotes bracket close.
    bracket_records = events(x, y, labels, gen, 256)['skip_trigram']
    bracket_records = [r for r in bracket_records if r[5] in ('(', '[')]
    bracket_outputs = [')', ']']
    bracket_ids = [gen.token2id[v] for v in bracket_outputs]
    bracket_truth = np.zeros((2, 2))
    bracket_full = np.zeros((2, 2))
    bracket_counts = []
    for i, opener in enumerate(('(', '[')):
        group = [r for r in bracket_records if r[5] == opener]
        assert group
        bracket_counts.append(len(group))
        bracket_truth[i, i] = np.mean([next(iter(bracket_close_distribution(
            opener, q-k).values())) for _, q, k, *_ in group])
        bracket_full[i] = np.mean([probs[b, q, bracket_ids] for b, q, *_ in group], axis=0)
    edge_records = [(b, q, k, target) for b, q, k, target, *_ in bracket_records]
    bracket_edges, identity_error = paths(model, x, edge_records, bracket_ids)
    bracket_edges = bracket_edges - bracket_edges.mean(-1, keepdims=True)
    bracket_path = np.stack([bracket_edges[[j for j, r in enumerate(bracket_records)
                                            if r[5] == opener]].sum(1).mean(0)
                             for opener in ('(', '[')])

    rows = [
        ('Adjacent noun → verb', nouns, verbs, noun_truth, noun_full, noun_direct,
         'Direct E→U weights; conditional softmax over verbs'),
        ('Place … adjective → function word',
         [f'{a} … {t}' for a, t in order], skip_outputs,
         skip_truth, skip_full, skip_path,
         'Sparse attention: selected-anchor L1 QK₁×QK₂×OV→U'),
        ('Opener … content → bracket closer', ['( …', '[ …'], bracket_outputs,
         bracket_truth, bracket_full, bracket_path,
         'Sparse attention: selected-opener L1 QK₁×QK₂×OV→U'),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 13), layout='constrained')
    for row, (title, ylabels, outputs, truth, full, path, note) in enumerate(rows):
        for col, (matrix, heading) in enumerate(zip(
                (truth, full, path), ('Planted rule', 'Sparse model probability', 'Weight path'))):
            ax = axes[row, col]
            if col == 2 and row > 0:
                bound = max(float(np.abs(path).max()), 1e-6)
                image = ax.imshow(path, cmap='coolwarm', vmin=-bound, vmax=bound,
                                  aspect='auto')
                fig.colorbar(image, ax=ax, fraction=.045, pad=.02, label='Centered local logit')
            else:
                image = ax.imshow(matrix, cmap='viridis', vmin=0, vmax=1,
                                  aspect='auto')
                if col == 2 or row == 0:
                    fig.colorbar(image, ax=ax, fraction=.045, pad=.02,
                                 label='Probability')
            ax.set_title(heading)
            ax.set_xticks(range(len(outputs)), outputs, rotation=60 if row == 0 else 0,
                          fontsize=8)
            ax.set_yticks(range(len(ylabels)), ylabels, fontsize=7 if row == 1 else 8)
        axes[row, 0].set_ylabel(title)
        axes[row, 2].set_xlabel(note, fontsize=8)
    fig.suptitle('Jointly sampled bigram and long-range rules: planted vs learned vs paths\n'
                 'Attention projections 60% zero; embedding and output weights dense',
                 fontsize=13)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = args.output_dir / 'bigram_skip_bracket_comparison.png'
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    result = dict(seed=args.seed, streams=args.streams,
                  checkpoint=str(args.checkpoint), figure=str(figure),
                  noun=dict(rows=nouns, outputs=verbs, counts=noun_counts,
                            planted=noun_truth.tolist(), full=noun_full.tolist(),
                            direct_weight_path=noun_direct.tolist()),
                  crossed_skip=dict(rows=[f'{a} … {t}' for a, t in order],
                                    outputs=skip_outputs, planted=skip_truth.tolist(),
                                    full=skip_full.tolist(), sparse_path=skip_path.tolist()),
                  bracket=dict(rows=['(', '['], outputs=bracket_outputs,
                               counts=bracket_counts, oracle_closer_mass=bracket_truth.tolist(),
                               full=bracket_full.tolist(), sparse_path=bracket_path.tolist(),
                               local_path_identity_max_error=identity_error),
                  caveat='The noun direct E→U path uses dense matrices. The attention '
                         'paths use pruned weights but depend on fresh activations. '
                         'They are exact local logit contributions, not complete '
                         'end-to-end causal circuits. The code label skip_trigram '
                         'means bracket closing.')
    (args.output_dir / 'bigram_skip_bracket_comparison.json').write_text(
        json.dumps(result, indent=2) + '\n')
    print('saved', figure, 'bracket path identity error', identity_error, flush=True)


if __name__ == '__main__':
    main()
