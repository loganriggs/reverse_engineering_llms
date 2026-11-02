"""Same-vocabulary source-key lookup curriculum for the v6 joint language.

Two claims have different speakers and objects. The answer is the speaker of
the queried object. A checkerboard of speaker–object statement pairings is
excluded from pretraining, matching the optional v6 joint-training split.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from surface_mixture_v6 import (CLAIM_OBJECTS, SPEAKERS, SurfaceMixtureV6,
                                source_pair_heldout)


# One fixed, fully supported narrative clause makes source statements legal
# under v6's visible-prefix grammar, which does not permit "As" at position 0.
START = ('During', 'the', 'tour', 'Ben', 'picks', 'the', 'soft', 'umbrella',
         'near', 'the', 'kitchen', ',', 'while', 'Leo', 'watches', ',',
         'although', 'Olive', 'said', 'the', 'local', 'coat', 'was',
         'missing', '.')
LENGTH = len(START)+31


def cases(n, seed, split):
    assert split in ('train', 'pair_holdout')
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < n:
        s = rng.integers(len(SPEAKERS), size=(max(256, 2*n), 2))
        o = rng.integers(len(CLAIM_OBJECTS), size=(len(s), 2))
        q = rng.integers(2, size=len(s))
        for i in range(len(s)):
            s0, s1 = map(int, s[i])
            o0, o1 = map(int, o[i])
            qi = int(q[i])
            if s0 == s1 or o0 == o1:
                continue
            h0 = source_pair_heldout(SPEAKERS[s0], CLAIM_OBJECTS[o0])
            h1 = source_pair_heldout(SPEAKERS[s1], CLAIM_OBJECTS[o1])
            if split == 'train' and (h0 or h1):
                continue
            if split == 'pair_holdout' and not ((h0, h1)[qi]
                                                 and not (h0, h1)[1-qi]):
                continue
            rows.append((s0, o0, s1, o1, qi))
            if len(rows) == n:
                break
    return np.asarray(rows, dtype=np.int64)


def encode(gen, rows, edit='none'):
    assert edit in ('none', 'query', 'order', 'speaker_swap',
                    'irrelevant_speaker')
    token = gen.token2id
    out = np.empty((len(rows), LENGTH), dtype=np.int64)
    targets = np.empty(len(rows), dtype=np.int64)
    for i, (s0, o0, s1, o1, q) in enumerate(rows):
        s0, o0, s1, o1, q = map(int, (s0, o0, s1, o1, q))
        pairs = [(SPEAKERS[s0], CLAIM_OBJECTS[o0]),
                 (SPEAKERS[s1], CLAIM_OBJECTS[o1])]
        query = 1-q if edit == 'query' else q
        if edit == 'order':
            pairs.reverse()
        if edit == 'speaker_swap':
            pairs = [(pairs[1][0], pairs[0][1]),
                     (pairs[0][0], pairs[1][1])]
        if edit == 'irrelevant_speaker':
            other = 1-q
            replacement = next(s for s in SPEAKERS
                               if s not in (pairs[q][0], pairs[other][0]))
            pairs[other] = (replacement, pairs[other][1])
        obj = CLAIM_OBJECTS[(o0, o1)[query]]
        facts = {object_: speaker for speaker, object_ in pairs}
        targets[i] = token[facts[obj]]
        words = list(START)
        for speaker, object_ in pairs:
            words += ['As', speaker, 'said', ',', '"', 'the', object_,
                      'was', 'missing', '"', '.']
        words += ['Who', 'said', 'the', obj, 'was', 'missing', '?',
                  'Answer', ':']
        assert len(words) == LENGTH
        out[i] = [token[word] for word in words]
    return out, targets


def check_oracle(gen, rows):
    for edit in ('none', 'query', 'order', 'speaker_swap',
                 'irrelevant_speaker'):
        x, target = encode(gen, rows, edit)
        for prefix, answer in zip(x, target):
            words = [gen.id2token[int(t)] for t in prefix]
            for t, word in enumerate(words):
                assert gen.next_distribution(words[:t]).get(word, 0) > 0
            assert gen.next_distribution(words) == {gen.id2token[int(answer)]: 1.}


@torch.no_grad()
def predict(model, x, device, batch_size=512):
    model.eval()
    out = []
    for start in range(0, len(x), batch_size):
        xt = torch.tensor(x[start:start+batch_size], dtype=torch.long,
                          device=device)
        out.append(F.softmax(model(xt)[0][:, -1].float(), -1).cpu().numpy())
    return np.concatenate(out)


def evaluate(model, gen, rows, device):
    x, y = encode(gen, rows)
    p = predict(model, x, device)
    result = dict(n=len(rows), accuracy=float(np.mean(p.argmax(-1) == y)),
                  answer_ce=float(-np.log(p[np.arange(len(y)), y]
                                           .clip(1e-30)).mean()))
    q0 = np.zeros_like(p)
    q0[np.arange(len(y)), y] = 1.
    for edit in ('query', 'order', 'speaker_swap', 'irrelevant_speaker'):
        xe, ye = encode(gen, rows, edit)
        pe = predict(model, xe, device)
        q1 = np.zeros_like(p)
        q1[np.arange(len(ye)), ye] = 1.
        oracle = q1-q0
        model_effect = pe-p
        result[edit] = dict(oracle_tv=float(np.abs(oracle).sum(-1).mean()/2),
                            model_tv=float(np.abs(model_effect).sum(-1).mean()/2),
                            effect_l1=float(np.abs(model_effect-oracle)
                                            .sum(-1).mean()))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    device = torch.device(a.device)
    gen = SurfaceMixtureV6(quote_rate=.30, source_rate=.30,
                           question_rate=.05, counterbalanced_queries=True)
    allowed = cases(2048, 17001+a.seed, 'train')
    heldout = cases(2048, 18001+a.seed, 'pair_holdout')
    check_oracle(gen, allowed[:64])
    check_oracle(gen, heldout[:64])
    assert all(not source_pair_heldout(SPEAKERS[s], CLAIM_OBJECTS[o])
               for s0, o0, s1, o1, _ in allowed
               for s, o in ((s0, o0), (s1, o1)))
    torch.manual_seed(19000+a.seed)
    config = Config(vocab_size=len(gen.vocab), n_layer=2, n_head=4,
                    n_embd=64, seq_len=128,
                    block_has_mlp=(False, False), use_rmsnorm=True)
    model = BilinearGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    rng = np.random.default_rng(20000+a.seed)
    snapshots = []
    for step in range(1, a.steps+1):
        model.train()
        batch = cases(a.batch, int(rng.integers(2**31)), 'train')
        x, y = encode(gen, batch)
        xt = torch.tensor(x, dtype=torch.long, device=device)
        yt = torch.tensor(y, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xt)[0][:, -1].float(), yt)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite source pretraining loss at step {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        if step == 1 or step % 500 == 0 or step == a.steps:
            scores = dict(allowed=evaluate(model, gen, allowed, device),
                          heldout=evaluate(model, gen, heldout, device))
            snapshots.append(dict(step=step, train_ce=float(loss.detach()),
                                  scores=scores))
            print(step, {k: (round(v['accuracy'], 4),
                             round(v['query']['model_tv'], 4),
                             round(v['irrelevant_speaker']['model_tv'], 4))
                         for k, v in scores.items()}, flush=True)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), a.output_dir/'model_depth2.pt')
    (a.output_dir/'results.json').write_text(json.dumps(dict(
        config={**vars(a), 'output_dir': str(a.output_dir)},
        model_config=asdict(config), vocabulary=list(gen.vocab),
        split='(speaker_index+claim_object_index) % 4 == 0; both training statements allowed',
        snapshots=snapshots), indent=2)+'\n')


if __name__ == '__main__':
    main()
