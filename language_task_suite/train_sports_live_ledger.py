"""Full-token bilinear LM probe on paired live-score ledger surfaces.

This is a controlled score-arithmetic experiment, not the accepted natural
language DGP. Held-out games are disjoint from training games.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
import time

import torch
import torch.nn.functional as F

OLD_REPO = Path(__file__).resolve().parents[2]/'reverse_engineer/tensor_language'
sys.path.insert(0, str(OLD_REPO))
from deep_model import DeepModel, SPECS  # noqa: E402
# The language-audit dependency tree also imports a different module named
# `model`. DeepModel has already bound the original Attention class above.
sys.modules.pop('model', None)
sys.path.remove(str(OLD_REPO))
from sports_live_ledger_dgp import sample_document  # noqa: E402


def draw(n, seed, group, descriptions):
    return [sample_document(random.Random(seed+i), 'rotowire', group,
                            descriptions) for i in range(n)]


def encode(docs, vocab, n_ctx):
    unknown = vocab['<UNK>']
    x = torch.full((len(docs), n_ctx), vocab['<PAD>'], dtype=torch.long)
    y = torch.full((len(docs), n_ctx), -100, dtype=torch.long)
    oov = 0
    total = 0
    for i, doc in enumerate(docs):
        ids = [vocab.get(token, unknown) for token in doc.tokens]
        assert len(ids)-1 <= n_ctx
        x[i, :len(ids)-1] = torch.tensor(ids[:-1])
        y[i, :len(ids)-1] = torch.tensor(ids[1:])
        oov += sum(token not in vocab for token in doc.tokens)
        total += len(doc.tokens)
    return (x, y), oov/total


@torch.no_grad()
def evaluate(model, docs, data, vocab, device, batch_size=32):
    model.eval()
    counts = defaultdict(lambda: Counter())
    nll = 0.
    tokens = 0
    number_ids = {
        'run': torch.tensor([vocab[str(i)] for i in range(41)], device=device),
        'copy': torch.tensor([vocab[str(i)] for i in range(201)], device=device),
    }
    for start in range(0, len(docs), batch_size):
        batch = docs[start:start+batch_size]
        x, y = (v[start:start+batch_size].to(device) for v in data)
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                               ignore_index=-100, reduction='sum')
        nll += loss.item()
        tokens += int((y != -100).sum())
        for j, doc in enumerate(batch):
            for site in doc.sites:
                assert doc.tokens[site.index] == site.target
                assert site.target in vocab
                position = site.index-1
                row = counts[site.family]
                row['n'] += 1
                row['full_vocab_correct'] += int(
                    logits[j, position].argmax().item() == \
                    vocab[site.target])
                choices = number_ids['copy' if site.family ==
                                     'live_score_copy' else 'run']
                chosen = int(logits[j, position, choices].argmax())
                row['number_candidate_correct'] += chosen == int(site.target)
    return {'token_nll': nll/tokens,
            'sites': {kind: {'n': row['n'],
                             'full_vocab_accuracy':
                             row['full_vocab_correct']/row['n'],
                             'number_candidate_accuracy':
                             row['number_candidate_correct']/row['n']}
                      for kind, row in counts.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--train-n', type=int, default=2000)
    parser.add_argument('--test-n', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--d-model', type=int, default=48)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--depths', default='1,2')
    parser.add_argument('--arms', default='bare,described')
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=10307)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save-models', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    out = {'config': vars(args) | {'output': str(args.output)},
           'objective': 'full next-token cross-entropy on every non-padding '
                        'token; run sites are evaluation labels only',
           'caveat': 'Controlled live-feed genre pilot. Original NBA play '
                     'descriptions are empirical text and have no exact '
                     'all-token oracle. Test games are entirely held out.',
           'results': {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for arm in args.arms.split(','):
        if arm not in ('bare', 'described'):
            raise ValueError(arm)
        descriptions = arm == 'described'
        train = draw(args.train_n, args.seed+100000, 'train', descriptions)
        test = draw(args.test_n, args.seed+200000, 'heldout', descriptions)
        train_games = {doc.latent['game_id'] for doc in train}
        test_games = {doc.latent['game_id'] for doc in test}
        assert not train_games & test_games
        vocab = {token: i for i, token in enumerate(sorted(
            {'<PAD>', '<UNK>'}|{token for doc in train
                               for token in doc.tokens}|{str(i)
                                                       for i in range(201)}))}
        n_ctx = max(len(doc.tokens)-1 for doc in train+test)
        train_data, train_oov = encode(train, vocab, n_ctx)
        test_data, test_oov = encode(test, vocab, n_ctx)
        baseline = {}
        for family in ('live_score_copy', 'live_run_first',
                       'live_run_second'):
            counts = Counter(site.target for doc in train for site in doc.sites
                             if site.family == family)
            popular = counts.most_common(1)[0][0]
            actual = [site.target for doc in test for site in doc.sites
                      if site.family == family]
            baseline[family] = {'most_common_train_answer': popular,
                                'heldout_accuracy':
                                sum(x == popular for x in actual)/len(actual)}
        arm_out = {'vocab': len(vocab), 'n_ctx': n_ctx,
                   'train_games': len(train_games),
                   'heldout_games': len(test_games),
                   'train_oov_token_share': train_oov,
                   'heldout_oov_token_share': test_oov,
                   'majority_baseline': baseline,
                   'depths': {}}
        out['results'][arm] = arm_out
        for depth in (int(x) for x in args.depths.split(',')):
            torch.manual_seed(args.seed+depth)
            model = DeepModel(len(vocab), args.d_model, args.heads,
                              SPECS[f'block{depth}'], n_ctx, norm='rms',
                              residual='lerp', attention='bilinear').to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            sampler = random.Random(args.seed+depth+500)
            losses = []
            start = time.time()
            model.train()
            for step in range(args.steps):
                indices = torch.tensor(sampler.sample(range(len(train)),
                                                      args.batch_size))
                x, y = (tensor[indices].to(device) for tensor in train_data)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                                       ignore_index=-100)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite {arm} depth{depth} '
                                       f'step{step}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                losses.append(loss.item())
                if (step+1) % 100 == 0:
                    print(arm, depth, step+1, round(loss.item(), 4),
                          flush=True)
            result = {'parameters': model.n_params,
                      'train_loss_last50':
                      sum(losses[-50:])/len(losses[-50:]),
                      'heldout': evaluate(model, test, test_data, vocab,
                                          device),
                      'seconds': round(time.time()-start, 1)}
            arm_out['depths'][str(depth)] = result
            args.output.write_text(json.dumps(out, indent=2)+'\n')
            if args.save_models:
                path = args.output.with_name(
                    args.output.stem+f'_{arm}_depth{depth}.pt')
                torch.save({'state_dict': {k: v.detach().cpu() for k, v in
                                           model.state_dict().items()},
                            'vocabulary': vocab, 'n_ctx': n_ctx,
                            'arm': arm, 'depth': depth,
                            'model_kwargs': {'d_model': args.d_model,
                                             'n_head': args.heads,
                                             'norm': 'rms',
                                             'residual': 'lerp',
                                             'attention': 'bilinear'},
                            'evaluation': result}, path)
        print('finished', arm, flush=True)
    print('saved', args.output, flush=True)


if __name__ == '__main__':
    main()
