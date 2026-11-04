"""Test GPT-2 likelihood as a bounded form prior for joint syntax DGP.

All candidate documents already pass the rule checker. The LM is used only
to rank linguistic form, not to label agreement or other target rules.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from audit_totto_table_description_dgp import form
from treebank_joint_syntax_document import load, sample


@torch.no_grad()
def nll(model, tokenizer, text, device):
    ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
    logits = model(ids).logits[:, :-1].float()
    targets = ids[:, 1:]
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                           reduction='mean').item()


def family_counts(docs):
    counts = Counter()
    for doc in docs:
        families = {site.family for site in doc.sites}
        counts['all_three'] += len(families) == 3
        counts['bracket'] += 'bracket_close' in families
        counts['reuse'] += 'subject_reuse' in families
        counts['sites'] += len(doc.sites)
    return dict(counts)


def audit(treebank, *, fit_trees=100000, n=512, chosen=100,
          seed=75101, device='cuda'):
    world, pp, source = load(treebank, fit_trees)
    docs = [sample(world, pp, seed+i, source_clauses=source,
                   source_rate=0., bracket_rate=.25, reuse_rate=.35)
            for i in range(n)]
    joint = [doc for doc in docs if len({site.family
                                        for site in doc.sites}) == 3]
    if len(joint) < chosen:
        raise ValueError('Not enough three-rule documents')
    model = AutoModelForCausalLM.from_pretrained(
        'gpt2', local_files_only=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('gpt2', local_files_only=True)
    scored = [(nll(model, tokenizer, doc.text(), device), doc)
              for doc in joint]
    scored.sort(key=lambda row: row[0])
    best = scored[:chosen]
    random_rows = random.Random(seed+1).sample(scored, chosen)
    output = {
        'config': {'fit_trees': fit_trees, 'n': n,
                   'chosen': chosen, 'seed': seed,
                   'source_rate': 0., 'bracket_rate': .25,
                   'reuse_rate': .35,
                   'selection': 'rank by GPT-2 mean subword NLL among '
                                'documents with all three rule families'},
        'eligible_joint': len(joint),
        'arms': {},
        'caveat': 'GPT-2 likelihood is a form proxy, not a semantic '
                  'validator or the DGP oracle. Selecting low NLL changes '
                  'the sampled law and can favor common short forms; '
                  'length and repetition are reported explicitly.'}
    for name, rows in (('best', best), ('random', random_rows)):
        texts = [doc.text() for _, doc in rows]
        output['arms'][name] = {
            'mean_gpt2_nll': sum(v for v, _ in rows)/len(rows),
            'form': form(texts),
            'family_counts': family_counts([doc for _, doc in rows]),
        }
    print('best examples')
    for score, doc in best[:3]:
        print(round(score, 3), doc.text()[:450])
    print('worst examples')
    for score, doc in scored[-3:]:
        print(round(score, 3), doc.text()[:450])
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--treebank', type=Path, required=True)
    parser.add_argument('--fit-trees', type=int, default=100000)
    parser.add_argument('--n', type=int, default=512)
    parser.add_argument('--chosen', type=int, default=100)
    parser.add_argument('--seed', type=int, default=75101)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = audit(args.treebank, fit_trees=args.fit_trees, n=args.n,
                   chosen=args.chosen, seed=args.seed, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
