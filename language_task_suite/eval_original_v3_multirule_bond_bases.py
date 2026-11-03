"""Compare weight-only and unlabeled activation bond bases on joint v3 rules.

The basis fit can omit an entire rule's positions. This is a featurizer-fit
holdout, not a model-training holdout. A single attention-output bond is
projected at a time; full-rank projections must replay the checkpoint.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from sparsify_original_v3_for_rule_graphs import make_data, model_and_generator


RULES = ('noun_bigram', 'skip_bigram', 'skip_trigram')
FIT_KINDS = (('all',) + tuple('without_' + rule for rule in RULES) +
             tuple('random_without_' + rule for rule in RULES))


def batches(generator, count, batch_size, length, device):
    for start in range(0, count, batch_size):
        yield make_data(generator, min(batch_size, count-start), length, device)


@torch.no_grad()
def activation_grams(model, generator, count, batch_size, length, device, seed):
    """Uncentered Gram matrices of each pre-W_O bond, fit without task labels."""
    dimension = model.config.n_embd
    grams = {layer: {kind: torch.zeros((dimension, dimension),
                                   dtype=torch.float64, device=device)
                     for kind in FIT_KINDS}
             for layer in range(len(model.blocks))}
    counts = {kind: 0 for kind in FIT_KINDS}
    random_generator = torch.Generator(device=device).manual_seed(seed + 7741)
    for x, _, labels, _ in batches(generator, count, batch_size, length, device):
        captured = {}
        handles = []
        for layer, block in enumerate(model.blocks):
            def capture(module, inputs, layer=layer):
                captured[layer] = inputs[0].detach().flatten(0, 1).double()
            handles.append(block.attn.out.register_forward_pre_hook(capture))
        try:
            model(x)
        finally:
            for handle in handles:
                handle.remove()
        flat_labels = [label for row in labels for label in row]
        masks = {'all': torch.ones(len(flat_labels), dtype=torch.bool, device=device)}
        for rule in RULES:
            masks['without_' + rule] = torch.tensor(
                [label != rule for label in flat_labels], device=device)
            same_count = int((~masks['without_' + rule]).sum())
            random_mask = torch.ones(len(flat_labels), dtype=torch.bool,
                                     device=device)
            chosen = torch.randperm(len(flat_labels), generator=random_generator,
                                    device=device)[:same_count]
            random_mask[chosen] = False
            masks['random_without_' + rule] = random_mask
        for kind, mask in masks.items():
            counts[kind] += int(mask.sum())
            for layer, values in captured.items():
                selected = values[mask]
                grams[layer][kind] += selected.T @ selected
    return grams, counts


def bases_from_grams(model, grams):
    bases = {}
    for layer, block in enumerate(model.blocks):
        weight = block.attn.out.weight.float()
        _, _, vh = torch.linalg.svd(weight)
        bases[layer] = {'weight_only': vh.T.contiguous()}
        for kind, gram in grams[layer].items():
            _, vectors = torch.linalg.eigh(gram)
            bases[layer][kind] = vectors.flip(-1).float().contiguous()
        # A predeclared prior sweep; normalize traces so alpha has a stable
        # meaning across checkpoints and fit samples. This is still a local
        # Gram basis, not a unique tensor canonical form.
        data_gram = grams[layer]['all'].float()
        data_gram = data_gram / data_gram.trace()
        weight_gram = weight.T @ weight
        weight_gram = weight_gram / weight_gram.trace()
        for alpha in (0.1, 1.0, 10.0):
            _, vectors = torch.linalg.eigh(data_gram + alpha*weight_gram)
            bases[layer][f'hybrid_{alpha:g}'] = vectors.flip(-1).contiguous()
    return bases


@torch.no_grad()
def evaluate(model, x, y, labels, layer, basis, rank,
             native_logp, native_ce, native_bond):
    block = model.blocks[layer]
    U = basis[:, :rank]
    def project(module, inputs):
        z = inputs[0]
        return ((z @ U) @ U.T,)
    hook = block.attn.out.register_forward_pre_hook(project)
    try:
        logp = F.log_softmax(model(x)[0].float(), -1)
    finally:
        hook.remove()
    ce = -logp.gather(-1, y[..., None]).squeeze(-1)
    kl = (native_logp.exp() * (native_logp-logp)).sum(-1)
    reconstruction = (native_bond - (native_bond @ U) @ U.T).square().mean(-1)
    rows = defaultdict(list)
    for batch, row in enumerate(labels):
        for pos, name in enumerate(row):
            if name in RULES:
                rows[name].append((float(ce[batch,pos]), float(kl[batch,pos]),
                                   float(native_ce[batch,pos]),
                                   float(reconstruction[batch,pos])))
    return {name: dict(n=len(values), ce=sum(v[0] for v in values)/len(values),
                       native_ce=sum(v[2] for v in values)/len(values),
                       teacher_kl=sum(v[1] for v in values)/len(values),
                       bond_reconstruction_mse=sum(v[3] for v in values)/len(values))
            for name, values in rows.items() if values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path,
                        default=Path('/home/loganriggs/Coding/toy_models_of_tensor_networks'))
    parser.add_argument('--checkpoint', type=Path,
                        default=Path(__file__).parent / 'results' /
                        'original_v3_sparse60_recovery_seed94001' / 'final_no_l1.pt')
    parser.add_argument('--fit-streams', type=int, default=256)
    parser.add_argument('--eval-streams', type=int, default=128)
    parser.add_argument('--length', type=int, default=129)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--fit-seed', type=int, default=95101)
    parser.add_argument('--eval-seed', type=int, default=95102)
    parser.add_argument('--ranks', default='8,16,32,64')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    model, Generator, _ = model_and_generator(args.source, device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload['model'])
    model.eval()
    masks = payload['mask']
    total_weights = sum(m.numel() for m in masks.values())
    live_weights = sum(int(m.sum()) for m in masks.values())
    grams, counts = activation_grams(model, Generator(seed=args.fit_seed),
                                    args.fit_streams, args.batch_size,
                                    args.length, device, args.fit_seed)
    bases = bases_from_grams(model, grams)
    data = make_data(Generator(seed=args.eval_seed), args.eval_streams,
                     args.length, device)
    x, y, labels, _ = data
    native_bonds = {}
    handles = []
    for layer, block in enumerate(model.blocks):
        def capture(module, inputs, layer=layer):
            native_bonds[layer] = inputs[0].detach()
        handles.append(block.attn.out.register_forward_pre_hook(capture))
    with torch.no_grad():
        try:
            native_logp = F.log_softmax(model(x)[0].float(), -1)
        finally:
            for handle in handles:
                handle.remove()
        native_ce = -native_logp.gather(-1, y[..., None]).squeeze(-1)
    rows = []
    for layer in range(len(model.blocks)):
        for name, basis in bases[layer].items():
            for rank in (int(r) for r in args.ranks.split(',')):
                scores = evaluate(model, x, y, labels, layer, basis,
                                  rank, native_logp, native_ce,
                                  native_bonds[layer])
                rows.append(dict(layer=layer, basis=name, rank=rank,
                                 rule_scores=scores))
                print(layer, name, rank,
                      {k: round(v['teacher_kl'], 4) for k,v in scores.items()},
                      flush=True)
    for row in rows:
        if row['rank'] == model.config.n_embd:
            assert max(v['teacher_kl'] for v in row['rule_scores'].values()) < 1e-5
    result = dict(config={k:str(v) if isinstance(v,Path) else v
                          for k,v in vars(args).items()},
                  checkpoint_attention_zero_fraction=1-live_weights/total_weights,
                  fit_counts=counts,
                  native_rule_ce={name: sum(float(native_ce[b,t])
                       for b,row in enumerate(labels) for t,label in enumerate(row)
                       if label==name)/sum(label==name for row in labels for label in row)
                       for name in RULES}, rows=rows,
                  caveat='Uncentered pre-W_O activation PCA and right singular '
                         'vectors of W_O; one local attention bond projected '
                         'at a time. Rule-family exclusions apply to basis '
                         'fitting only, not model training. Sampled-target CE '
                         'and teacher KL are distinct from causal edit fidelity '
                         'or rule discovery.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
