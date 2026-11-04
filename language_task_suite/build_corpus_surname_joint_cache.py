"""Build a source-disjoint mixed FineWeb rule cache; token rows stay local."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from build_fineweb_surname_copy_cache import NearDuplicateFilter
from corpus_rule_reference_dgp import chunk_document, sample_document
from corpus_surname_joint_dgp import KINDS, paired_document
from fineweb_name_copy_dgp import NameCopyDGP


RULE_RATES = dict(list_rate=.5, bracket_rate=.5, nonce_rate=0.,
                  list_shift_rate=.5, square_rate=.5,
                  literal_copy_rate=1., natural_style_rate=1.)
UNEDITED_RATES = dict(list_rate=1., bracket_rate=1., nonce_rate=0.,
                      list_shift_rate=0., square_rate=0.,
                      literal_copy_rate=1., natural_style_rate=1.)


def token_row(chunk):
    positions = torch.full((len(KINDS),),-1,dtype=torch.long)
    targets = torch.full((len(KINDS),),-1,dtype=torch.long)
    for site in chunk['sites']:
        if site['kind'] in KINDS and len(site['positions']) == 1:
            j = KINDS.index(site['kind'])
            if positions[j] < 0:
                positions[j] = site['positions'][0]
                targets[j] = site['target_ids'][0]
                assert chunk['y'][positions[j]] == targets[j]
    return {'x':chunk['x'], 'y':chunk['y'], 'positions':positions,
            'targets':targets}


def stack(rows):
    if not rows:
        return None
    return {key:torch.stack([row[key] for row in rows])
            for key in ('x','y','positions','targets')}


def _name_chunks(pair, tokenizer, n_ctx):
    arms = []
    for doc in pair:
        chunks = chunk_document(doc, tokenizer, n_ctx, max_tokens=2048)
        candidates = [token_row(chunk) for chunk in chunks
                      if any(site['kind']=='surname_copy'
                             for site in chunk['sites'])]
        if not candidates:
            return None
        arms.append(candidates[0])
    j = KINDS.index('surname_copy')
    if arms[0]['targets'][j] == arms[1]['targets'][j]:
        return None
    return tuple(arms)


def build(train_rows, heldout_rows, n_ctx=256, seed=46401,
          unedited_background=False):
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    dedup = NearDuplicateFilter()
    cache = {split:{'background':[], 'sites':{kind:[] for kind in KINDS},
                    'name_pairs':[[],[]], 'source_hashes':[]}
             for split in ('train','val','test')}
    counts = {split:Counter() for split in cache}
    for source_rows,is_train in ((train_rows,True),(heldout_rows,False)):
        dgp = NameCopyDGP(source_rows)
        for index,source in enumerate(source_rows):
            sha = source['sha256']
            split = ('train' if is_train else
                     'val' if int(sha[:8],16)%2==0 else 'test')
            counts[split]['input_sources'] += 1
            reason = dedup.add(source['preview'],sha)
            if reason:
                counts[split][reason] += 1
                continue
            counts[split]['accepted_sources'] += 1
            cache[split]['source_hashes'].append(sha)
            offset = source['index']
            rng = random.Random(seed+offset)
            enriched = sample_document(source['preview'],
                                       random.Random(seed+offset+20_000_000),
                                       (), **RULE_RATES)
            counts[split]['enriched_documents_with_text_change'] += (
                enriched.text != enriched.source_text)
            background_doc = (sample_document(
                source['preview'], random.Random(seed+offset+30_000_000),
                (), **UNEDITED_RATES) if unedited_background else enriched)
            counts[split]['background_documents_with_text_change'] += (
                background_doc.text != background_doc.source_text)
            background_chunks = chunk_document(background_doc, tokenizer,
                                               n_ctx, max_tokens=2048)
            if background_chunks:
                background = token_row(rng.choice(background_chunks))
                cache[split]['background'].append(background)
                counts[split]['background_rows'] += 1
            chunks = chunk_document(enriched,tokenizer,n_ctx,max_tokens=2048)
            found = set()
            for chunk in chunks:
                row = token_row(chunk)
                for j,kind in enumerate(KINDS):
                    if row['positions'][j] >= 0 and kind not in found:
                        cache[split]['sites'][kind].append(row)
                        counts[split][f'{kind}_site_rows'] += 1
                        found.add(kind)
            pair = paired_document(dgp,index,rng,[],**RULE_RATES)
            if pair is not None:
                counts[split]['surname_pairable_sources'] += 1
                aligned = _name_chunks(pair,tokenizer,n_ctx)
                if aligned is not None:
                    counts[split]['surname_aligned_pairs'] += 1
                    extra = set()
                    for arm in aligned:
                        extra.update(KINDS[j] for j,value in
                                     enumerate(arm['positions'])
                                     if j > 0 and value >= 0)
                    counts[split]['surname_pair_with_other_site'] += bool(extra)
                    for name in extra:
                        counts[split][f'surname_pair_with_{name}'] += 1
                    for j,arm in enumerate(aligned):
                        cache[split]['name_pairs'][j].append(arm)
            if counts[split]['input_sources'] % 1000 == 0:
                print(split,counts[split]['input_sources'],flush=True)
    assert len(set(cache['train']['source_hashes']) &
               set(cache['val']['source_hashes'])) == 0
    assert len(set(cache['train']['source_hashes']) &
               set(cache['test']['source_hashes'])) == 0
    assert len(set(cache['val']['source_hashes']) &
               set(cache['test']['source_hashes'])) == 0
    tensors = {}
    for split,section in cache.items():
        tensors[split] = {'background':stack(section['background']),
                          'sites':{kind:stack(rows)
                                   for kind,rows in section['sites'].items()},
                          'name_pairs':tuple(stack(rows)
                                             for rows in section['name_pairs']),
                          'source_hashes':section['source_hashes']}
    manifest = {'source':'HuggingFaceFW/fineweb sample-10BT train',
                'train_offsets':[train_rows[0]['index'],
                                 train_rows[-1]['index']],
                'heldout_offsets':[heldout_rows[0]['index'],
                                   heldout_rows[-1]['index']],
                'config':{'n_ctx':n_ctx,'seed':seed,'rates':RULE_RATES,
                          'unedited_background':unedited_background,
                          'tokenizer':str(DEFAULT_TOKENIZER)},
                'counts':{split:dict(counts[split]) for split in cache},
                'source_digest':{split:hashlib.sha256(''.join(sorted(
                    cache[split]['source_hashes'])).encode()).hexdigest()
                                 for split in cache},
                'scope':'One random background chunk per accepted source; '
                        'when unedited_background=true it comes from '
                        'the unchanged source text, otherwise from the '
                        'enriched edit draw. '
                        'at most one natural site row per family per source; '
                        'paired surname edit when BPE-aligned. All rows '
                        'carry ordinary next-token targets. Site sampling '
                        'during training may reweight task prevalence. '
                        'Text/token cache stays local; no full-token oracle.'}
    return tensors,manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-preview',type=Path,required=True)
    p.add_argument('--heldout-preview',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-limit',type=int)
    p.add_argument('--heldout-limit',type=int)
    p.add_argument('--n-ctx',type=int,default=256)
    p.add_argument('--seed',type=int,default=46401)
    p.add_argument('--unedited-background',action='store_true')
    a = p.parse_args()
    train = json.loads(a.train_preview.read_text())[:a.train_limit]
    heldout = json.loads(a.heldout_preview.read_text())[:a.heldout_limit]
    result,manifest = build(train,heldout,a.n_ctx,a.seed,
                            a.unedited_background)
    a.cache.parent.mkdir(parents=True,exist_ok=True)
    temp = a.cache.with_suffix('.tmp')
    torch.save(result,temp)
    temp.replace(a.cache)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(manifest,indent=2)+'\n')
    print(manifest['counts'],flush=True)


if __name__ == '__main__':
    main()
