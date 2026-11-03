"""Compare a data-free first-layer QK opener score with head necessity.

The QK score is a deliberately narrow weight-only baseline. It ignores OV,
later layers, and the query-token distribution, so agreement is not assumed.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from train_corpus_delimiter_bilinear import _model, evaluate


@torch.no_grad()
def opener_qk_signal(layer, embedding, tokenizer,
                     distances=(1, 8, 32, 64)):
    assert layer.norm.__class__.__name__ == 'Identity'
    q1 = F.linear(embedding, layer.q1.weight).reshape(
        -1, layer.n_head, layer.d_head)
    q2 = F.linear(embedding, layer.q2.weight).reshape(
        -1, layer.n_head, layer.d_head)
    result = {}
    for prefix in ('', ' '):
        token_ids = []
        for char in ('(', '['):
            ids = tokenizer.encode(prefix+char,
                                   add_special_tokens=False).ids
            assert len(ids) == 1, (prefix, char, ids)
            token_ids.append(ids[0])
        k1 = F.linear(embedding[token_ids], layer.k1.weight).reshape(
            2, layer.n_head, layer.d_head)
        k2 = F.linear(embedding[token_ids], layer.k2.weight).reshape(
            2, layer.n_head, layer.d_head)
        differences, total_scores = [], []
        for delta in distances:
            assert delta < layer.rotary.cos_cached.size(1)
            cos = layer.rotary.cos_cached[0, delta]
            sin = layer.rotary.sin_cached[0, delta]
            def rotate(q):
                left, right = q.chunk(2, dim=-1)
                return q*cos + torch.cat((-right, left), dim=-1)*sin
            p1 = torch.einsum('vhd,shd->vhs', rotate(q1), k1)
            p2 = torch.einsum('vhd,shd->vhs', rotate(q2), k2)
            scores = p1*p2/layer.d_head**2
            differences.append(scores[:, :, 0]-scores[:, :, 1])
            total_scores.append(scores)
        delta_scores = torch.cat(differences, dim=0)
        all_scores = torch.cat(total_scores, dim=0)
        rms = delta_scores.square().mean(dim=0).sqrt()
        reference = all_scores.square().mean(dim=(0, 2)).sqrt()
        result['space' if prefix else 'bare'] = dict(
            opener_token_ids=token_ids,
            difference_rms=rms.cpu().tolist(),
            relative_difference_rms=(rms/reference).cpu().tolist(),
            overall_score_rms=reference.cpu().tolist())
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache-path', type=Path, default=Path(
        '/home/loganriggs/Coding/neuron_circuits/artifacts/'
        'corpus_delimiter_pilot/dataset.pt'))
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    data = torch.load(args.cache_path, map_location='cpu', weights_only=True)
    checkpoint = torch.load(args.checkpoint, map_location='cpu',
                            weights_only=True)
    device = torch.device(args.device)
    model = _model(data['vocab_size'], data['n_ctx'],
                   checkpoint['depth'], checkpoint['d_model'],
                   checkpoint['heads'], device, checkpoint['norm'])
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    raw_table = model.embed.weight.detach()
    # First attention is causally negligible in this pilot. Exactly zeroing
    # its O input leaves 0.5*embedding under the lerp residual; MLP1 then
    # gives a weight-only effective token table for second-layer QK.
    with torch.no_grad():
        effective_table = model.layers[1](.5*raw_table)
    weight_scores = dict(
        first_attention_raw_embedding=opener_qk_signal(
            model.layers[0], raw_table, tokenizer),
        second_attention_after_first_mlp=opener_qk_signal(
            model.layers[2], effective_table, tokenizer))
    baseline = evaluate(model, data['test'], device, args.batch_size)
    head_ablations = []
    for layer_index, layer in enumerate(model.layers):
        if not hasattr(layer, 'n_head'):
            continue
        for head in range(layer.n_head):
            def zero_head(module, inputs, *, head=head, width=layer.d_head):
                value = inputs[0].clone()
                value[..., head*width:(head+1)*width] = 0
                return (value,)
            handle = layer.o.register_forward_pre_hook(zero_head)
            try:
                metrics = evaluate(model, data['test'], device,
                                   args.batch_size)
            finally:
                handle.remove()
            head_ablations.append(dict(
                layer_index=layer_index, head=head,
                both_two_way=metrics['both_two_way'],
                drop_both_two_way=(baseline['both_two_way']-
                                   metrics['both_two_way']),
                mean_correct_margin_shift=metrics['mean_correct_margin_shift'],
                drop_mean_margin_shift=(
                    baseline['mean_correct_margin_shift']-
                    metrics['mean_correct_margin_shift']),
                positive_shift_fraction=metrics['positive_shift_fraction']))
    mlp_ablations = []
    for layer_index, layer in enumerate(model.layers):
        if not hasattr(layer, 'D'):
            continue
        handle = layer.D.register_forward_hook(
            lambda module, inputs, output: torch.zeros_like(output))
        try:
            metrics = evaluate(model, data['test'], device,
                               args.batch_size)
        finally:
            handle.remove()
        mlp_ablations.append(dict(
            layer_index=layer_index,
            both_two_way=metrics['both_two_way'],
            drop_both_two_way=(baseline['both_two_way']-
                               metrics['both_two_way']),
            drop_mean_margin_shift=(baseline['mean_correct_margin_shift']-
                                    metrics['mean_correct_margin_shift'])))
    result = dict(checkpoint=str(args.checkpoint),
                  data_free_first_layer_qk=weight_scores,
                  baseline_test=baseline,
                  head_ablations=head_ablations,
                  mlp_ablations=mlp_ablations,
                  caveat='QK differential is a weight-only screening score, '
                         'not a complete path or proof of a delimiter circuit.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('heads', [(row['layer_index'], row['head'],
                     round(row['drop_both_two_way'], 4))
                    for row in head_ablations], flush=True)
    print('mlps', [(row['layer_index'],
                    round(row['drop_both_two_way'], 4))
                   for row in mlp_ablations], flush=True)


if __name__ == '__main__':
    main()
