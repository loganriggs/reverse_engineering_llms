"""Train bilinear LMs on audited multi-genre language and score exact oracle KL.

Evaluation keeps dependency labels separate from easy frame tokens. The genre
header is part of the prefix, so all oracle distributions are observable.
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

from multi_genre_language import GENRES, MultiGenreLanguage, VOCAB


TASK_LABELS = ('report_mixed_verb', 'policy_mixed_action',
               'faq_answer_copy', 'faq_place_recall', 'faq_new_place',
               'list_number')


def make_batch(gen, n, length, device):
    ids, _, _ = gen.sample_batch(n, length+1)
    z = torch.tensor(ids, dtype=torch.long, device=device)
    return z[:, :-1], z[:, 1:]


def make_eval(gen, n, length, device, task_labels=TASK_LABELS, genres=GENRES):
    ids, labels, qs = gen.sample_batch(n, length+1)
    z = torch.tensor(ids, dtype=torch.long, device=device)
    q = np.zeros((n, length, len(gen.vocab)), dtype=np.float32)
    masks = {name: np.zeros((n, length), dtype=bool)
             for name in ('all',) + tuple(genres) + tuple(task_labels)}
    masks['all'][:] = True
    for b, (row, labs, distributions) in enumerate(zip(ids, labels, qs)):
        genre = gen.id2token[row[0]]
        masks[genre][b] = True
        for t in range(1, len(row)):
            for token, p in distributions[t].items():
                q[b, t-1, gen.token2id[token]] = p
            if labs[t] in task_labels:
                masks[labs[t]][b, t-1] = True
            if (labs[t] in ('catalog_owner_answer', 'catalog_status_answer')
                    and 'catalog_first_owner_answer' in task_labels):
                from multi_genre_v2_language import catalog_state
                words = [gen.id2token[i] for i in row[:t]]
                if not catalog_state(words[2:])[1]:
                    first_label = labs[t].replace('catalog_', 'catalog_first_')
                    masks[first_label][b, t-1] = True
    assert np.allclose(q.sum(-1), 1)
    return dict(x=z[:, :-1], y=z[:, 1:], q=torch.from_numpy(q).to(device),
                masks={name: torch.from_numpy(mask).to(device)
                       for name, mask in masks.items()})


@torch.no_grad()
def evaluate(model, data):
    logits = torch.cat([model(batch)[0].float()
                        for batch in data['x'].split(32)])
    logp = F.log_softmax(logits, -1)
    q = data['q']
    entropy = -(q*q.clamp_min(1e-30).log()).sum(-1)
    kl = -(q*logp).sum(-1)-entropy
    sampled_ce = -logp.gather(-1, data['y'].unsqueeze(-1)).squeeze(-1)
    out = {}
    for name, mask in data['masks'].items():
        count = int(mask.sum())
        out[name] = dict(n=count,
                         kl=float(kl[mask].mean()) if count else None,
                         oracle_entropy=float(entropy[mask].mean()) if count else None,
                         sampled_ce=float(sampled_ce[mask].mean()) if count else None)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--language', choices=('v1', 'v2', 'scores'), default='v1')
    p.add_argument('--genre-weights', default='',
                   help='Optional comma-separated weights in genre order')
    p.add_argument('--catalog-record-rate', type=float, default=.60,
                   help='V2 probability of adding another record after two exist')
    p.add_argument('--attention', choices=('bilinear', 'softmax'),
                   default='bilinear')
    p.add_argument('--first-answer-weight', type=float, default=0.,
                   help='Extra CE on the first Catalog answer in each stream')
    p.add_argument('--score-answer-weight', type=float, default=0.,
                   help='Extra CE on every numeric-report comparison answer')
    p.add_argument('--depths', default='1,2,3,4')
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--length', type=int, default=71)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--with-mlp', action='store_true',
                   help='Enable the native bilinear MLP in every block')
    p.add_argument('--mlp-hidden', type=int, default=512)
    p.add_argument('--rmsnorm', action='store_true')
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--neval', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    if a.first_answer_weight < 0 or a.score_answer_weight < 0:
        raise ValueError('Answer weights must be nonnegative')
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config
    from softmax_attention_control import configure_attention

    torch.set_num_threads(4)
    device = torch.device(a.device)
    if a.language == 'scores':
        from numeric_report_language import (GENRES as SCORE_GENRES,
                                             TASK_LABELS as SCORE_TASK_LABELS,
                                             VOCAB as SCORE_VOCAB,
                                             NumericReportLanguage)
        generator_class = NumericReportLanguage
        genres = SCORE_GENRES
        vocab = SCORE_VOCAB
        task_labels = SCORE_TASK_LABELS
        if a.first_answer_weight or a.genre_weights:
            raise ValueError('Scores language does not use genre or Catalog-answer weights')
    elif a.language == 'v2':
        from multi_genre_v2_language import (GENRES_V2, VOCAB_V2,
                                             MultiGenreV2Language)
        generator_class = MultiGenreV2Language
        genres = GENRES_V2
        vocab = VOCAB_V2
        task_labels = TASK_LABELS + (
            'catalog_record_object', 'catalog_owner_value',
            'catalog_status_value', 'catalog_query_object',
            'catalog_owner_answer', 'catalog_status_answer',
            'catalog_first_owner_answer', 'catalog_first_status_answer')
    else:
        generator_class = MultiGenreLanguage
        genres = GENRES
        vocab = VOCAB
        task_labels = TASK_LABELS
    if a.score_answer_weight and a.language != 'scores':
        raise ValueError('Score-answer weighting requires the Scores language')
    if a.genre_weights:
        genre_weights = tuple(float(value) for value in a.genre_weights.split(','))
        if len(genre_weights) != len(genres):
            raise ValueError('Genre weight count must match language genres')
    else:
        genre_weights = None
    def generator(seed):
        kwargs = {'genre_weights': genre_weights} if genre_weights is not None else {}
        if a.language == 'v2':
            kwargs['record_rate'] = a.catalog_record_rate
        return generator_class(seed=seed, **kwargs)
    output_dir = Path(a.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test = make_eval(generator(9101), a.neval, a.length, device,
                     task_labels, genres)
    records = []
    for depth in [int(d) for d in a.depths.split(',')]:
        torch.manual_seed(9200+a.seed)
        config = Config(vocab_size=len(vocab), n_layer=depth,
                        n_head=a.heads, n_embd=a.width, seq_len=a.length,
                        mlp_hidden=a.mlp_hidden,
                        block_has_mlp=(a.with_mlp,)*depth,
                        use_rmsnorm=a.rmsnorm)
        model = configure_attention(BilinearGPT(config), a.attention).to(device)
        gen = generator(9300+a.seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
        t0 = time.time()
        snapshots = []
        model.train()
        for step in range(a.steps):
            if a.first_answer_weight or a.score_answer_weight:
                ids, labels, _ = gen.sample_batch(a.batch, a.length+1)
                z = torch.tensor(ids, dtype=torch.long, device=device)
                x, y = z[:, :-1], z[:, 1:]
                if a.first_answer_weight:
                    first_mask = torch.zeros_like(y, dtype=torch.bool)
                    for b, labs in enumerate(labels):
                        first = next((at for at, label in enumerate(labs)
                                      if label in ('catalog_owner_answer',
                                                   'catalog_status_answer')), None)
                        if first is not None and first > 0:
                            first_mask[b, first-1] = True
                    if not first_mask.any():
                        raise RuntimeError('No first Catalog answers in batch')
                if a.score_answer_weight:
                    score_mask = torch.tensor([
                        [label == 'score_comparison_answer' for label in labs[1:]]
                        for labs in labels], dtype=torch.bool, device=device)
                    if not score_mask.any():
                        raise RuntimeError('No numeric comparison answers in batch')
            else:
                x, y = make_batch(gen, a.batch, a.length, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)[0]
            loss = F.cross_entropy(logits.reshape(-1, len(vocab)), y.reshape(-1))
            if a.first_answer_weight:
                answer_ce = F.cross_entropy(logits[first_mask], y[first_mask])
                loss = loss + a.first_answer_weight*answer_ce
            if a.score_answer_weight:
                score_ce = F.cross_entropy(logits[score_mask], y[score_mask])
                loss = loss + a.score_answer_weight*score_ce
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at depth {depth}, step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            if (step+1) % 500 == 0 or step+1 == a.steps:
                model.eval()
                scores = evaluate(model, test)
                snapshots.append(dict(step=step+1, train_ce=float(loss.detach()),
                                      scores=scores, elapsed=time.time()-t0))
                print('depth', depth, 'step', step+1,
                      {name: round(scores[name]['kl'], 4) if scores[name]['kl'] is not None else None
                       for name in task_labels},
                      'elapsed', round(time.time()-t0, 1), flush=True)
                model.train()
                (output_dir/'results.json').write_text(json.dumps(dict(
                    config=vars(a), vocabulary=list(vocab), records=records,
                    current=dict(depth=depth, snapshots=snapshots)), indent=2)+'\n')
        torch.save(model.state_dict(), output_dir/f'model_depth{depth}.pt')
        records.append(dict(depth=depth, model_config=asdict(config),
                            parameters=model.count_params(), snapshots=snapshots))
        (output_dir/'results.json').write_text(json.dumps(dict(
            config=vars(a), vocabulary=list(vocab), records=records), indent=2)+'\n')


if __name__ == '__main__':
    main()
