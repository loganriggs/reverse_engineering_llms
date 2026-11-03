"""Full-token bilinear LM pilot on a corpus-derived agreement DGP."""

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
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


OLD_REPO = Path(__file__).resolve().parents[2]/'reverse_engineer/tensor_language'
sys.modules.pop('model', None)
sys.path.insert(0, str(OLD_REPO))
from deep_model import DeepModel, SPECS  # noqa: E402
sys.modules.pop('model', None)
sys.path.remove(str(OLD_REPO))
from treebank_agreement_dgp import PAIRS, load_world  # noqa: E402


def sentence(clause):
    return ' '.join(clause.tokens)


def draw(world, n, seed, balanced=False, subject_split='all',
         attractor_rate=.2):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        if balanced:
            out.append(world.sample(rng, number=i % 2,
                                    attractor=(i//2) % 2 == 1,
                                    subject_split=subject_split))
        else:
            out.append(world.sample(rng, attractor_rate=attractor_rate,
                                    subject_split=subject_split))
    return out


def fit_tokenizer(train, vocab_size):
    tokenizer = Tokenizer(models.BPE(unk_token='<UNK>'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.train_from_iterator(
        (sentence(doc) for doc in train),
        trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                            special_tokens=['<PAD>', '<UNK>']))
    for forms in PAIRS.values():
        for word in forms:
            ids = tokenizer.encode(word).ids
            if len(ids) != 1:
                raise RuntimeError(f'Verb {word!r} has {len(ids)} BPE tokens')
    return tokenizer


def encode(tokenizer, docs, n_ctx):
    pad = tokenizer.token_to_id('<PAD>')
    x = torch.full((len(docs), n_ctx), pad, dtype=torch.long)
    y = torch.full((len(docs), n_ctx), -100, dtype=torch.long)
    sites = []
    unknown = 0
    total = 0
    for i, doc in enumerate(docs):
        ids = tokenizer.encode(sentence(doc)).ids
        prefix = tokenizer.encode(' '.join(
            doc.tokens[:doc.target_index])).ids
        assert ids[:len(prefix)] == prefix
        assert ids[len(prefix)] == tokenizer.encode(
            PAIRS[doc.lemma][doc.number]).ids[0]
        assert len(ids)-1 <= n_ctx
        x[i, :len(ids)-1] = torch.tensor(ids[:-1])
        y[i, :len(ids)-1] = torch.tensor(ids[1:])
        sites.append(len(prefix)-1)
        unknown += ids.count(tokenizer.token_to_id('<UNK>'))
        total += len(ids)
    return (x, y), sites, unknown/total


@torch.no_grad()
def evaluate(model, data, sites, docs, tokenizer, device, batch_size=64):
    model.eval()
    counts = defaultdict(Counter)
    loss_sum = 0.
    n_tokens = 0
    for start in range(0, len(docs), batch_size):
        x, y = (t[start:start+batch_size].to(device) for t in data)
        logits = model(x)
        loss_sum += F.cross_entropy(
            logits.flatten(0, 1), y.flatten(), ignore_index=-100,
            reduction='sum').item()
        n_tokens += int((y != -100).sum())
        for j, doc in enumerate(docs[start:start+batch_size]):
            singular, plural = PAIRS[doc.lemma]
            choice = [tokenizer.encode(word).ids[0]
                      for word in (singular, plural)]
            predicted = int(logits[j, sites[start+j], choice].argmax())
            key = 'attractor' if doc.attractor else 'plain'
            counts[key]['n'] += 1
            counts[key]['correct'] += predicted == doc.number
            counts['all']['n'] += 1
            counts['all']['correct'] += predicted == doc.number
            counts['lemma/'+doc.lemma]['n'] += 1
            counts['lemma/'+doc.lemma]['correct'] += \
                predicted == doc.number
    return {'token_nll': loss_sum/n_tokens,
            'site_pair_accuracy': {
                key: {'n': row['n'], 'accuracy': row['correct']/row['n']}
                for key, row in counts.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--treebank', type=Path, required=True)
    parser.add_argument('--fit-trees', type=int, default=20000)
    parser.add_argument('--train-n', type=int, default=12000)
    parser.add_argument('--train-attractor-rate', type=float, default=.2)
    parser.add_argument('--test-n', type=int, default=1000)
    parser.add_argument('--vocab-size', type=int, default=4000)
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--depths', default='1,2,4')
    parser.add_argument('--seed', type=int, default=41301)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-models', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    world = load_world(str(args.treebank), args.fit_trees)
    train = draw(world, args.train_n, args.seed+100000,
                 subject_split='train',
                 attractor_rate=args.train_attractor_rate)
    test = draw(world, args.test_n, args.seed+200000, balanced=True,
                subject_split='heldout')
    train_subjects = {tuple(word.lower() for word in doc.subject)
                      for doc in train}
    test_subjects = {tuple(word.lower() for word in doc.subject)
                     for doc in test}
    assert not train_subjects & test_subjects
    tokenizer = fit_tokenizer(train, args.vocab_size)
    lengths = [len(tokenizer.encode(sentence(doc)).ids)-1
               for doc in train+test]
    n_ctx = max(lengths)
    train_data, _, train_oov = encode(tokenizer, train, n_ctx)
    test_data, test_sites, test_oov = encode(tokenizer, test, n_ctx)
    out = {
        'config': {key: value for key, value in vars(args).items()
                   if key not in ('treebank', 'output')} |
                  {'treebank_file': args.treebank.name,
                   'output': str(args.output)},
        'objective': 'Full next-subword cross-entropy on every non-padding '
                     'token. At the verb site, compare the two inflections '
                     'of the same latent lemma; balanced test subject number '
                     'and opposite-number PP attractors.',
        'caveat': 'Exploratory single-sentence syntax arm. Source phrase '
                  'recombination can be semantically odd. The latent verb '
                  'lemma is not visible before its target, so exact token '
                  'accuracy is not a valid grammar score. Subject phrases '
                  'are disjoint across train and test, but head nouns and '
                  'lexical morphemes can recur.',
        'vocab': tokenizer.get_vocab_size(), 'n_ctx': n_ctx,
        'oov': {'train': train_oov, 'test': test_oov},
        'baseline_pair_accuracy': .5,
        'whole_subject_holdout': {
            'train_unique_subjects': len(train_subjects),
            'test_unique_subjects': len(test_subjects),
            'overlap': len(train_subjects & test_subjects)},
        'results': {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for depth in (int(x) for x in args.depths.split(',')):
        torch.manual_seed(args.seed+depth)
        model = DeepModel(tokenizer.get_vocab_size(), args.d_model,
                          args.heads, SPECS[f'block{depth}'], n_ctx,
                          norm='rms', residual='lerp',
                          attention='bilinear').to(device)
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
                raise RuntimeError(f'Nonfinite depth{depth} step{step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(loss.item())
            if (step+1) % 250 == 0:
                print(depth, step+1, round(loss.item(), 4), flush=True)
        result = {'parameters': model.n_params,
                  'train_loss_last50': sum(losses[-50:])/50,
                  'heldout': evaluate(model, test_data, test_sites, test,
                                      tokenizer, device),
                  'seconds': round(time.time()-start, 1)}
        out['results'][str(depth)] = result
        args.output.write_text(json.dumps(out, indent=2)+'\n')
        if args.save_models:
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in
                                       model.state_dict().items()},
                        'vocab': tokenizer.to_str(), 'n_ctx': n_ctx,
                        'depth': depth, 'config': out['config']},
                       args.output.with_name(
                           args.output.stem+f'_depth{depth}.pt'))
    print('saved', args.output, flush=True)


if __name__ == '__main__':
    main()
