"""Sparse final-query source paths in a competent bilinear toy-language LM.

Rank complete second-layer QK1*QK2*V source contributions, keep K per head,
then run the native output projection and final norm. This is a dynamic
source-edge intervention, not parameter sparsity or end-to-end circuit recovery.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_surface_mixture_edits import score
from eval_surface_mixture_v2_edits import build_pairs
from eval_varied_second_layer_sources import source_contributions
from surface_mixture_v2_varied import SurfaceMixtureV2Varied
from surface_mixture_v2 import OBJECTS, VERBS


METHODS = ('random_position', 'score_product', 'pre_output_norm',
           'output_norm', 'all_vocab_contrast', 'object_contrast',
           'random_vocab_contrast', 'verb_contrast')


@torch.no_grad()
def predict_all(model, prefixes, edit_positions, keeps, device):
    vocab = model.config.vocab_size
    output = {f'{method}:{keep}': np.zeros((len(prefixes), vocab), np.float32)
              for method in METHODS for keep in keeps}
    output['full'] = np.zeros((len(prefixes), vocab), np.float32)
    selection = {f'{method}:{keep}': np.zeros((len(prefixes), model.config.n_head),
                                            dtype=bool)
                 for method in METHODS for keep in keeps}
    reconstruction = {f'{method}:{keep}': dict(error=0., energy=0.,
                                               selected_fraction_sum=0.)
                      for method in METHODS for keep in keeps}
    groups = defaultdict(list)
    for i, prefix in enumerate(prefixes):
        groups[len(prefix)].append(i)
    attn = model.blocks[1].attn
    h, d = attn.n_head, attn.head_dim
    out_weight = attn.out.weight.reshape(model.config.n_embd, h, d)
    gen = SurfaceMixtureV2Varied()
    verb_ids = torch.tensor([gen.token2id[word] for word in VERBS], device=device)
    object_ids = torch.tensor([gen.token2id[word] for word in OBJECTS[:len(VERBS)]],
                              device=device)
    rng = np.random.default_rng(20261102)
    random_ids = torch.tensor(rng.choice(
        [i for i in range(vocab) if i not in set(verb_ids.tolist())],
        len(VERBS), replace=False), device=device)
    def output_contrast(ids):
        rows = model.lm_head.weight[ids]
        return torch.einsum('vc,chd->vhd',
                            rows-rows.mean(0, keepdim=True), out_weight)
    contrast_paths = dict(verb_contrast=output_contrast(verb_ids),
                          all_vocab_contrast=output_contrast(
                              torch.arange(vocab, device=device)),
                          object_contrast=output_contrast(object_ids),
                          random_vocab_contrast=output_contrast(random_ids))
    max_replay_error = 0.
    for indices in groups.values():
        for start in range(0, len(indices), 32):
            current = indices[start:start+32]
            ids = torch.tensor([prefixes[i] for i in current],
                               dtype=torch.long, device=device)
            contributions, residual, logits, factors = source_contributions(model, ids)
            native = model(ids)[0][:, -1]
            max_replay_error = max(max_replay_error, float((native-logits).abs().max()))
            output['full'][current] = F.softmax(logits.float(), -1).cpu().numpy()
            batch, _, positions, _ = contributions.shape
            full_sum = contributions.sum(2).reshape(batch, h*d)
            full_output = attn.out(full_sum)
            base = residual-full_output
            product = factors['score1']*factors['score2']
            composed = torch.einsum('bhtd,chd->bhtc', contributions, out_weight)
            random_generator = torch.Generator(device='cpu').manual_seed(
                20261102+positions)
            random_positions = torch.rand((1, h, positions),
                                          generator=random_generator).to(device)
            rankings = dict(
                random_position=random_positions.expand(batch, -1, -1),
                score_product=product.abs(),
                pre_output_norm=contributions.norm(dim=-1),
                output_norm=composed.norm(dim=-1),
                **{name: torch.einsum('bhtd,vhd->bhtv',
                                      contributions, path).norm(dim=-1)
                   for name, path in contrast_paths.items()})
            edited_at = torch.tensor([edit_positions[i] for i in current],
                                     device=device)[:, None, None]
            for method, importance in rankings.items():
                top = importance.argsort(dim=-1, descending=True)
                for keep in keeps:
                    chosen = top[:, :, :min(keep, positions)]
                    selected = contributions.gather(
                        2, chosen.unsqueeze(-1).expand(-1, -1, -1, d)).sum(2)
                    sparse_output = attn.out(selected.reshape(batch, h*d))
                    sparse_residual = base+sparse_output
                    sparse_logits = model.lm_head(model.final_norm(sparse_residual))
                    key = f'{method}:{keep}'
                    output[key][current] = F.softmax(sparse_logits.float(), -1).cpu().numpy()
                    selection[key][current] = (chosen == edited_at).any(-1).cpu().numpy()
                    reconstruction[key]['error'] += float(
                        (full_output-sparse_output).square().sum())
                    reconstruction[key]['energy'] += float(full_output.square().sum())
                    reconstruction[key]['selected_fraction_sum'] += (
                        batch*min(keep, positions)/positions)
    assert max_replay_error < 2e-4, max_replay_error
    reconstruction = {key: dict(relative_attention_mse=value['error']/max(
                            value['energy'], 1e-30),
                            mean_source_fraction=value['selected_fraction_sum']/len(prefixes))
                      for key, value in reconstruction.items()}
    return output, selection, reconstruction, max_replay_error


def summary(pairs, before, after, select_before, select_after, gen):
    native_b, native_a = before['full'], after['full']
    groups = defaultdict(list)
    for i, pair in enumerate(pairs):
        groups[pair['kind']].append(i)
    result = {}
    for key in before:
        causal = score(pairs, before[key], after[key], gen)
        group = {}
        for kind, indices in groups.items():
            i = np.array(indices)
            p0, p1 = before[key][i], after[key][i]
            q0, q1 = native_b[i], native_a[i]
            native_effect_error = np.abs((p1-p0)-(q1-q0)).sum(-1)
            model_kl = (q0*(np.log(np.maximum(q0, 1e-30))-
                            np.log(np.maximum(p0, 1e-30)))).sum(-1)
            model_kl += (q1*(np.log(np.maximum(q1, 1e-30))-
                             np.log(np.maximum(p1, 1e-30)))).sum(-1)
            oracle_kl = []
            for j in i:
                for distribution, p in ((pairs[j]['q_before'], before[key][j]),
                                        (pairs[j]['q_after'], after[key][j])):
                    oracle_kl.append(sum(mass*(np.log(mass)-np.log(max(
                        float(p[gen.token2id[token]]), 1e-30)))
                        for token, mass in distribution.items()))
            row = dict(**causal[kind],
                       native_effect_l1_error=float(native_effect_error.mean()),
                       model_to_sparse_kl=float(model_kl.mean()/2),
                       oracle_kl=float(np.mean(oracle_kl)))
            if key != 'full':
                row['edited_source_selected_any_head'] = float(
                    np.logical_or(select_before[key][i],
                                  select_after[key][i]).any(-1).mean())
                row['edited_source_selected_by_head'] = [float(
                    np.logical_or(select_before[key][i, head],
                                  select_after[key][i, head]).mean())
                    for head in range(select_before[key].shape[1])]
            group[kind] = row
        result[key] = group
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', type=Path,
                   default=Path('language_task_suite/surface_mixture_v2_varied_soft_models'))
    p.add_argument('--nstreams', type=int, default=128)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--seed', type=int, default=20261120)
    p.add_argument('--keeps', default='1,2,4,8,16')
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config
    torch.set_num_threads(4)
    metadata = json.loads((a.model_dir/'results.json').read_text())
    assert metadata['config']['generator'] == 'v2_varied'
    record = next(row for row in metadata['records'] if row['depth'] == 2)
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(a.model_dir/'model_depth2.pt',
                                     map_location='cpu', weights_only=True))
    model = model.to(a.device).eval()
    assert model.blocks[1].mlp is None
    gen = SurfaceMixtureV2Varied(seed=a.seed, **metadata['generator_kwargs'])
    pairs = build_pairs(gen, a.nstreams, a.length)
    assert pairs
    keeps = sorted(set(int(value) for value in a.keeps.split(',')))
    assert min(keeps) >= 1 and max(keeps) < a.length
    edit_positions = [next(i for i, (x, y) in enumerate(zip(
        pair['original'], pair['edited'])) if x != y) for pair in pairs]
    assert all(sum(x != y for x, y in zip(pair['original'], pair['edited'])) == 1
               for pair in pairs)
    with torch.no_grad():
        before, selected_before, recon_before, replay0 = predict_all(
            model, [pair['original'] for pair in pairs], edit_positions,
            keeps, a.device)
        after, selected_after, recon_after, replay1 = predict_all(
            model, [pair['edited'] for pair in pairs], edit_positions,
            keeps, a.device)
    rows = summary(pairs, before, after, selected_before, selected_after, gen)
    result = dict(config={**vars(a), 'model_dir': str(a.model_dir),
                          'output': str(a.output)},
                  n_pairs=len(pairs), pair_counts=dict(Counter(
                      pair['kind'] for pair in pairs)),
                  max_logit_replay_error=max(replay0, replay1),
                  reconstruction=dict(before=recon_before, after=recon_after),
                  rows=rows,
                  caveat='The gate acts only on final-query second-block '
                         'source contributions, independently on original '
                         'and edited prefixes. It is a dynamic hard mask, '
                         'not sparse weights or an end-to-end circuit. '
                         'Verb contrast uses known output tokens and ignores '
                         'the final RMSNorm Jacobian. Oracle KL and model '
                         'fidelity are reported separately; the generator '
                         'remains much less diverse than real language.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print('pairs', len(pairs), 'replay', result['max_logit_replay_error'])
    for key in ('full', 'score_product:4', 'pre_output_norm:4',
                'output_norm:4', 'verb_contrast:4'):
        print(key, {kind: round(row['effect_l1_error'], 3)
                    for kind, row in rows[key].items()}, flush=True)


if __name__ == '__main__':
    main()
