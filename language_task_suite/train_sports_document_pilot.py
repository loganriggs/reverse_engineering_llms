"""Full-LM bilinear pilot; site labels are used only for behavioral evaluation.

This is a screening experiment for a candidate DGP, not an interpretability
result or a claim of natural-language realism.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

OLD_REPO = Path(__file__).resolve().parents[2]/'reverse_engineer/tensor_language'
from sports_document_dgp import edit, realize, sample_plan  # noqa: E402

# The source-audit import chain loads this package's ``model`` module; the
# historical DeepModel imports a different top-level module with that name.
sys.modules.pop('model', None)
sys.path.insert(0, str(OLD_REPO))
from deep_model import DeepModel, SPECS  # noqa: E402


TOKEN = re.compile(r'\n\n|[A-Za-z]+|\d+|[^\w\s]')


def tokens_with_offsets(text):
    return [(match.group(), match.start(), match.end())
            for match in TOKEN.finditer(text)]


def tokenize(text):
    return [word for word, _, _ in tokens_with_offsets(text)]


def draw(n, seed, sbnation_rate, form_entropy, pair_rate):
    rng = random.Random(seed)
    return [realize(sample_plan(rng, sbnation_rate=sbnation_rate),
                    form_entropy=form_entropy, pair_rate=pair_rate)
            for _ in range(n)]


def changed(doc, field, form_entropy, pair_rate):
    plan = doc.plan
    if field == 'swap_teams':
        plan = edit(plan, field, True)
    elif field == 'high_score':
        plan = edit(plan, field, plan.high_score+1)
    elif field == 'points':
        points = list(plan.points)
        j = next(j for j, p in enumerate(points) if p != max(points))
        points[j] = max(points)+1
        plan = edit(plan, field, points)
    else:
        raise ValueError(field)
    return realize(plan, form_entropy=form_entropy, pair_rate=pair_rate)


def encoded(doc, vocab, n_ctx):
    words = tokenize(doc.text)
    ids = [vocab['<BOS>']]+[vocab.get(t, vocab['<UNK>']) for t in words]+[
        vocab['<EOS>']]
    assert len(ids)-1 <= n_ctx
    x = torch.full((n_ctx,), vocab['<PAD>'], dtype=torch.long)
    y = torch.full((n_ctx,), -100, dtype=torch.long)
    x[:len(ids)-1] = torch.tensor(ids[:-1])
    y[:len(ids)-1] = torch.tensor(ids[1:])
    return x, y


def candidate_batch(doc, site, vocab, device):
    rows = tokens_with_offsets(doc.text)
    before = [word for word, _, end in rows if end <= site.start]
    assert any(start == site.start for _, start, _ in rows)
    assert all(word in vocab for word in tokenize(site.answer))
    prefix = [vocab['<BOS>']]+[vocab.get(t, vocab['<UNK>']) for t in before]
    candidates = [tuple(vocab[t] for t in tokenize(v))
                  for v in site.candidates]
    inputs = [prefix+list(candidate[:-1]) for candidate in candidates]
    width = max(map(len, inputs))
    batch = torch.full((len(inputs), width), vocab['<PAD>'],
                       dtype=torch.long, device=device)
    for i, ids in enumerate(inputs):
        batch[i, :len(ids)] = torch.tensor(ids, device=device)
    return batch, candidates, len(prefix)-1


@torch.no_grad()
def evaluate(model, docs, vocab, n_ctx, device, batch_size=16):
    model.eval()
    data = [encoded(doc, vocab, n_ctx) for doc in docs]
    total_nll, total_tokens = 0., 0
    by_site = defaultdict(lambda: [0, 0, 0.])
    per_doc = []
    for start in range(0, len(docs), batch_size):
        xs = torch.stack([xy[0] for xy in data[start:start+batch_size]]).to(device)
        ys = torch.stack([xy[1] for xy in data[start:start+batch_size]]).to(device)
        logits = model(xs)
        total_nll += F.cross_entropy(logits.flatten(0, 1), ys.flatten(),
                                    ignore_index=-100, reduction='sum').item()
        total_tokens += int((ys != -100).sum())
    for doc in docs:
        answers = {}
        for site in doc.sites:
            batch, candidates, pos = candidate_batch(doc, site, vocab, device)
            logp = model(batch).log_softmax(-1)
            scores = [sum(logp[i, pos+j, target].item()
                          for j, target in enumerate(candidate))
                      for i, candidate in enumerate(candidates)]
            chosen = site.candidates[max(range(len(scores)),
                                         key=lambda i: scores[i])]
            hit = int(chosen == site.answer)
            by_site[site.kind][0] += hit
            by_site[site.kind][1] += 1
            by_site[site.kind][2] += scores[site.candidates.index(site.answer)]
            answers[site.kind] = chosen
        per_doc.append(answers)
    return dict(token_nll=total_nll/total_tokens,
                site={kind: dict(correct=row[0], n=row[1],
                                 candidate_accuracy=row[0]/row[1],
                                 target_candidate_logp=row[2]/row[1])
                      for kind, row in by_site.items()},
                predictions=per_doc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--depths', default='1,2,4')
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--train-n', type=int, default=4000)
    p.add_argument('--test-n', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--sbnation-rate', type=float, default=.5)
    p.add_argument('--form-entropy', type=float, default=1.)
    p.add_argument('--pair-rate', type=float, default=1.)
    p.add_argument('--seed', type=int, default=99021)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--save-models', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(4)
    kwargs = dict(sbnation_rate=a.sbnation_rate,
                  form_entropy=a.form_entropy, pair_rate=a.pair_rate)
    train = draw(a.train_n, a.seed, **kwargs)
    test = draw(a.test_n, a.seed+1, **kwargs)
    assert not {d.text for d in train} & {d.text for d in test}
    edits = {field: [changed(d, field, a.form_entropy, a.pair_rate)
                     for d in test]
             for field in ('swap_teams', 'high_score', 'points')}
    # Vocabulary is fixed before optimization. This pilot exposes test word
    # types, but no held-out document or label, so no lexical OOD claim.
    vocab = {word: i for i, word in enumerate(sorted(
        {'<BOS>', '<EOS>', '<PAD>', '<UNK>'} |
        {t for doc in train+test+sum(edits.values(), [])
         for site in doc.sites
         for answer in site.candidates
         for t in tokenize(answer)} |
        {t for doc in train+test+sum(edits.values(), [])
         for t in tokenize(doc.text)}))}
    n_ctx = max(len(tokenize(d.text))+1
                for d in train+test+sum(edits.values(), []))
    train_data = [encoded(doc, vocab, n_ctx) for doc in train]
    xtrain = torch.stack([v[0] for v in train_data])
    ytrain = torch.stack([v[1] for v in train_data])
    output = dict(config={**vars(a), 'output': str(a.output)},
                  vocab_size=len(vocab), context=n_ctx,
                  objective='ordinary full next-token CE; no rule labels in training',
                  caveats=['Candidate DGP fails natural sports form audit',
                           'Test vocabulary exposed before training; no lexical OOD claim',
                           'Site accuracy ranks candidates, not full-vocabulary accuracy',
                           'Multi-token candidate sums can favor shorter names',
                           'Random world split; no whole-template or game holdout'],
                  results={})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for depth in [int(v) for v in a.depths.split(',')]:
        torch.manual_seed(a.seed+depth)
        model = DeepModel(len(vocab), a.d_model, a.heads,
                          SPECS[f'block{depth}'], n_ctx, norm='rms',
                          residual='lerp', attention='bilinear').to(a.device)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
        rng = random.Random(a.seed+1000+depth)
        clock = time.time()
        losses = []
        model.train()
        for step in range(a.steps):
            idx = torch.tensor(rng.sample(range(a.train_n), a.batch_size))
            x, y = xtrain[idx].to(a.device), ytrain[idx].to(a.device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                                   ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite loss at depth {depth}, step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(loss.item())
            if (step+1) % 50 == 0:
                print(depth, step+1, round(loss.item(), 4), flush=True)
        base = evaluate(model, test, vocab, n_ctx, a.device)
        edit_eval = {field: evaluate(model, docs, vocab, n_ctx, a.device)
                     for field, docs in edits.items()}
        switch = {}
        for field, ev in edit_eval.items():
            changed_kinds = {'swap_teams': 'winner',
                             'high_score': 'margin',
                             'points': 'top_player'}
            target = changed_kinds[field]
            hits = sum(
                base['predictions'][i][target] != ev['predictions'][i][target]
                and ev['predictions'][i][target] == docs[i].sites[
                    next(j for j, s in enumerate(docs[i].sites)
                         if s.kind == target)].answer
                for i in range(len(test)) for docs in [edits[field]])
            switch[field] = dict(correct_switches=hits, n=len(test))
        for row in [base]+list(edit_eval.values()):
            row.pop('predictions')
        output['results'][str(depth)] = dict(
            parameters=model.n_params,
            train_loss_last50=sum(losses[-50:])/len(losses[-50:]),
            heldout=base, edits=edit_eval, selective_switch=switch,
            seconds=round(time.time()-clock, 2))
        a.output.write_text(json.dumps(output, indent=2)+'\n')
        if a.save_models:
            torch.save(dict(state_dict={k: v.detach().cpu()
                                        for k, v in model.state_dict().items()},
                            vocab=vocab, context=n_ctx, depth=depth,
                            config=output['config']),
                       a.output.with_name(a.output.stem+f'_block{depth}.pt'))
    print('saved', a.output, flush=True)


if __name__ == '__main__':
    main()
