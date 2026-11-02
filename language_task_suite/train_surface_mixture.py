"""Train 1–4-layer bilinear LMs on the readable mixed-rule stream.

Evaluate expected KL to the exact prefix oracle, especially at verb slots.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from surface_mixture_language import SurfaceMixtureLanguage
from surface_mixture_v2 import SurfaceMixtureV2
from surface_mixture_v3 import SurfaceMixtureV3
from surface_mixture_v4 import SurfaceMixtureV4
from surface_mixture_v5 import SurfaceMixtureV5


def make_batch(gen, n, length, device):
    ids, _, _ = gen.sample_batch(n, length+1)
    z = torch.tensor(ids, dtype=torch.long, device=device)
    return z[:, :-1], z[:, 1:]


def make_eval(gen, n, length, device):
    ids, labels, qs = gen.sample_batch(n, length+1)
    z = torch.tensor(ids, dtype=torch.long, device=device)
    q = np.zeros((n, length, len(gen.vocab)), dtype=np.float32)
    verb = np.zeros((n, length), dtype=bool)
    induction = np.zeros((n, length), dtype=bool)
    obj = np.zeros((n, length), dtype=bool)
    object_induction = np.zeros((n, length), dtype=bool)
    answer = np.zeros((n, length), dtype=bool)
    count_answer = np.zeros((n, length), dtype=bool)
    for b, (row, labs, distributions) in enumerate(zip(ids, labels, qs)):
        for t in range(1, len(row)):
            for token, p in distributions[t].items():
                q[b, t-1, gen.token2id[token]] = p
            if labs[t] == 'mixed_verb':
                verb[b, t-1] = True
                induction[b, t-1] = 'induction' in gen.candidates(row[:t])
            elif labs[t] == 'mixed_object':
                obj[b, t-1] = True
                object_induction[b, t-1] = ('object_induction' in
                                            gen.object_candidates(row[:t]))
            elif labs[t] == 'question_answer':
                answer[b, t-1] = True
            elif labs[t] == 'count_answer':
                count_answer[b, t-1] = True
    assert np.allclose(q.sum(-1), 1)
    return dict(x=z[:, :-1], y=z[:, 1:], q=torch.from_numpy(q).to(device),
                verb=torch.from_numpy(verb).to(device),
                induction=torch.from_numpy(induction).to(device),
                obj=torch.from_numpy(obj).to(device),
                object_induction=torch.from_numpy(object_induction).to(device),
                answer=torch.from_numpy(answer).to(device),
                count_answer=torch.from_numpy(count_answer).to(device))


def query_counterfactuals(gen, x, answer_mask):
    """One supported alternative query key with a different answer per slot.

    Question text after the object key is fixed, and questions do not alter
    narrative history. Each selected row therefore admits a matched answer
    target without changing the rest of its sampled narrative.
    """
    rows = torch.where(answer_mask.any(-1))[0].tolist()
    if not rows:
        return None
    edited = x[rows].clone()
    targets = torch.zeros_like(edited)
    mask = torch.zeros_like(edited, dtype=torch.bool)
    for local, batch_row in enumerate(rows):
        ids = x[batch_row].tolist()
        for j in torch.where(answer_mask[batch_row])[0].tolist():
            key_pos = j-4
            prefix = [gen.id2token[i] for i in ids[:j+1]]
            original_answer = gen.question_answer(prefix)
            key_q = gen.next_distribution(prefix[:key_pos])
            alternative = None
            for key in key_q:
                if key == prefix[key_pos]:
                    continue
                changed = prefix.copy()
                changed[key_pos] = key
                answer_q = gen.next_distribution(changed)
                if len(answer_q) == 1 and original_answer not in answer_q:
                    alternative = (key, next(iter(answer_q)))
                    break
            if alternative is None:
                continue
            edited[local, key_pos] = gen.token2id[alternative[0]]
            targets[local, j] = gen.token2id[alternative[1]]
            mask[local, j] = True
    return (edited, targets, mask) if mask.any() else None


@torch.no_grad()
def evaluate(model, data):
    logits = torch.cat([model(batch)[0].float()
                        for batch in data['x'].split(32)])
    logp = F.log_softmax(logits, -1)
    q = data['q']
    entropy = -(q*q.clamp_min(1e-30).log()).sum(-1)
    kl = -(q*logp).sum(-1)-entropy
    sample_ce = -logp.gather(-1, data['y'].unsqueeze(-1)).squeeze(-1)
    masks = {'all': torch.ones_like(data['verb']),
             'verb': data['verb'],
             'verb_with_induction': data['verb'] & data['induction'],
             'verb_without_induction': data['verb'] & ~data['induction']}
    if data['obj'].any():
        masks.update(object=data['obj'],
                     object_with_induction=data['obj'] & data['object_induction'],
                     object_without_induction=data['obj'] & ~data['object_induction'])
    if data['answer'].any():
        masks['question_answer'] = data['answer']
    if data['count_answer'].any():
        masks['count_answer'] = data['count_answer']
    return {name: dict(n=int(mask.sum()), kl=float(kl[mask].mean()),
                       oracle_entropy=float(entropy[mask].mean()),
                       sampled_ce=float(sample_ce[mask].mean()))
            for name, mask in masks.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--depths', default='1,2,3,4')
    p.add_argument('--generator', choices=['v1', 'v2', 'v3', 'v4', 'v5'], default='v1')
    p.add_argument('--object-vocab-size', type=int, default=None,
                   help='For v3/v4, restrict object-site outputs while keeping model vocabulary fixed')
    p.add_argument('--hard-query-rate', type=float, default=None,
                   help='For v4, probability of querying a nonlatest object when available')
    p.add_argument('--counterbalanced-queries', action='store_true',
                   help='For v4, ask only when at least two object keys have distinct answers')
    p.add_argument('--question-rate', type=float, default=.05,
                   help='For v4/v5, probability of a place question at an eligible boundary')
    p.add_argument('--count-rate', type=float, default=.20,
                   help='For v5, probability of a count question at an eligible boundary')
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--length', type=int, default=70)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--rmsnorm', action='store_true',
                   help='Use per-block and final RMSNorm for deep stability')
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--verb-soft-weight', type=float, default=0.0,
                   help='Additional exact-oracle expected CE on verb slots')
    p.add_argument('--verb-sampled-weight', type=float, default=0.0,
                   help='Additional sampled-token CE on verb slots')
    p.add_argument('--object-soft-weight', type=float, default=0.0,
                   help='Additional exact-oracle expected CE on object slots')
    p.add_argument('--answer-soft-weight', type=float, default=0.0,
                   help='Additional exact-oracle expected CE on question answers')
    p.add_argument('--query-counterfactual-weight', type=float, default=0.0,
                   help='For v4, CE on alternative supported query keys with different answers')
    p.add_argument('--count-soft-weight', type=float, default=0.0,
                   help='For v5, exact-oracle expected CE on person-count answers')
    p.add_argument('--mask-verb-ce', action='store_true',
                   help='Exclude verb slots from ordinary CE for object-only ablation')
    p.add_argument('--neval', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    if a.verb_soft_weight and a.verb_sampled_weight:
        p.error('Select only one extra verb objective')
    if a.mask_verb_ce and (a.verb_soft_weight or a.verb_sampled_weight):
        p.error('Cannot mask verb CE while adding a verb objective')
    if a.object_vocab_size is not None and a.generator not in ('v3', 'v4', 'v5'):
        p.error('Object vocabulary restriction requires v3/v4/v5 generator')
    if a.answer_soft_weight and a.generator not in ('v4', 'v5'):
        p.error('Question-answer supervision requires v4/v5 generator')
    if a.hard_query_rate is not None and a.generator not in ('v4', 'v5'):
        p.error('Hard-query rate requires v4/v5 generator')
    if a.counterbalanced_queries and a.generator not in ('v4', 'v5'):
        p.error('Counterbalanced queries require v4/v5 generator')
    if a.query_counterfactual_weight and a.generator not in ('v4', 'v5'):
        p.error('Query counterfactuals require v4/v5 generator')
    if a.count_soft_weight and a.generator != 'v5':
        p.error('Count supervision requires v5 generator')
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    device = torch.device(a.device)
    output_dir = Path(a.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator_class = {'v1': SurfaceMixtureLanguage,
                       'v2': SurfaceMixtureV2,
                       'v3': SurfaceMixtureV3,
                       'v4': SurfaceMixtureV4,
                       'v5': SurfaceMixtureV5}[a.generator]
    generator_kwargs = ({'object_vocab_size': a.object_vocab_size}
                        if a.generator in ('v3', 'v4', 'v5') else {})
    if a.generator in ('v4', 'v5'):
        generator_kwargs['hard_query_rate'] = a.hard_query_rate
        generator_kwargs['counterbalanced_queries'] = a.counterbalanced_queries
        generator_kwargs['question_rate'] = a.question_rate
    if a.generator == 'v5':
        generator_kwargs['count_rate'] = a.count_rate
    vocab = generator_class(**generator_kwargs).vocab
    test = make_eval(generator_class(seed=8101, **generator_kwargs), a.neval,
                     a.length, device)
    records = []
    for depth in [int(d) for d in a.depths.split(',')]:
        torch.manual_seed(8200+a.seed)
        config = Config(vocab_size=len(vocab), n_layer=depth,
                        n_head=a.heads, n_embd=a.width, seq_len=a.length,
                        block_has_mlp=(False,)*depth,
                        use_rmsnorm=a.rmsnorm)
        model = BilinearGPT(config).to(device)
        gen = generator_class(seed=8300+a.seed, **generator_kwargs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
        t0 = time.time()
        snapshots = []
        model.train()
        for step in range(a.steps):
            if (a.verb_soft_weight > 0 or a.verb_sampled_weight > 0
                    or a.object_soft_weight > 0 or a.answer_soft_weight > 0
                    or a.query_counterfactual_weight > 0 or a.count_soft_weight > 0
                    or a.mask_verb_ce):
                train = make_eval(gen, a.batch, a.length, device)
                x, y = train['x'], train['y']
            else:
                x, y = make_batch(gen, a.batch, a.length, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)[0]
            per_token_ce = F.cross_entropy(logits.reshape(-1, len(vocab)),
                                           y.reshape(-1), reduction='none')
            if a.mask_verb_ce:
                loss = per_token_ce.masked_fill(
                    train['verb'].reshape(-1), 0).mean()
            else:
                loss = per_token_ce.mean()
            if a.verb_soft_weight > 0:
                logp = F.log_softmax(logits.float(), dim=-1)
                expected_ce = -(train['q'] * logp).sum(-1)
                loss = loss + a.verb_soft_weight * expected_ce[train['verb']].mean()
            if a.verb_sampled_weight > 0:
                sampled_ce = F.cross_entropy(logits[train['verb']],
                                             y[train['verb']])
                loss = loss + a.verb_sampled_weight * sampled_ce
            if a.object_soft_weight > 0:
                if not train['obj'].any():
                    raise ValueError('Object supervision requires v3 generator')
                logp = F.log_softmax(logits.float(), dim=-1)
                expected_ce = -(train['q'] * logp).sum(-1)
                loss = loss + a.object_soft_weight * expected_ce[train['obj']].mean()
            if a.answer_soft_weight > 0:
                if train['answer'].any():
                    logp = F.log_softmax(logits.float(), dim=-1)
                    expected_ce = -(train['q'] * logp).sum(-1)
                    loss = (loss + a.answer_soft_weight *
                            expected_ce[train['answer']].mean())
            if a.query_counterfactual_weight > 0:
                counterfactual = query_counterfactuals(gen, x, train['answer'])
                if counterfactual is not None:
                    cf_x, cf_y, cf_mask = counterfactual
                    cf_logits = model(cf_x)[0]
                    cf_loss = F.cross_entropy(cf_logits[cf_mask].float(),
                                              cf_y[cf_mask])
                    loss = loss + a.query_counterfactual_weight*cf_loss
            if a.count_soft_weight > 0 and train['count_answer'].any():
                logp = F.log_softmax(logits.float(), dim=-1)
                expected_ce = -(train['q'] * logp).sum(-1)
                loss = (loss + a.count_soft_weight *
                        expected_ce[train['count_answer']].mean())
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at depth {depth}, step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            if (step+1) % 1000 == 0 or step+1 == a.steps:
                model.eval()
                scores = evaluate(model, test)
                snapshots.append(dict(step=step+1, train_ce=float(loss.detach()),
                                      scores=scores, elapsed=time.time()-t0))
                print('depth', depth, 'step', step+1,
                      'verb KL', round(scores['verb']['kl'], 4),
                      'induction KL', round(scores['verb_with_induction']['kl'], 4),
                      'object KL', round(scores['object']['kl'], 4)
                      if 'object' in scores else 'n/a',
                      'answer KL', round(scores['question_answer']['kl'], 4)
                      if 'question_answer' in scores else 'n/a',
                      'count KL', round(scores['count_answer']['kl'], 4)
                      if 'count_answer' in scores else 'n/a',
                      'elapsed', round(time.time()-t0, 1), flush=True)
                model.train()
        torch.save(model.state_dict(), output_dir/f'model_depth{depth}.pt')
        records.append(dict(depth=depth, model_config=asdict(config),
                            parameters=model.count_params(), snapshots=snapshots))
        (output_dir/'results.json').write_text(json.dumps(dict(
            config=vars(a), vocabulary=list(vocab), records=records), indent=2)+'\n')


if __name__ == '__main__':
    main()
