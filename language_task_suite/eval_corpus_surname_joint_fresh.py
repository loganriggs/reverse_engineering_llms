"""Fresh-source paired surname and bracket gates for frozen joint models."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from build_corpus_delimiter_pilot import simhash
from corpus_rule_reference_dgp import chunk_document
from corpus_surname_joint_dgp import paired_document
from eval_corpus_rule_reference_transfer import one_kind, pair_for_text
from fineweb_name_copy_dgp import NameCopyDGP
from train_corpus_delimiter_bilinear import _model
from train_fineweb_surname_copy_bilinear import evaluate as evaluate_name


RULE_RATES = dict(list_rate=.5,bracket_rate=.5,nonce_rate=0.,
                  list_shift_rate=.5,square_rate=.5,
                  literal_copy_rate=1.,natural_style_rate=1.)


def _fresh_indices(prior_rows,rows):
    # Avoid carrying source prose or token IDs into the committed aggregate.
    hashes={row['sha256'] for row in prior_rows}
    signatures=[simhash(row['preview']) for row in prior_rows]
    buckets=defaultdict(list)
    for i,signature in enumerate(signatures):
        for block in range(4):
            buckets[(block,(signature>>(16*block))&65535)].append(i)
    keep=[]
    reasons=Counter()
    for i,row in enumerate(rows):
        if row['sha256'] in hashes:
            reasons['exact_duplicate']+=1
            continue
        signature=simhash(row['preview'])
        candidates=set()
        for block in range(4):
            candidates.update(buckets[(block,(signature>>(16*block))&65535)])
        # Hamming distance <=3 across four 16-bit blocks implies at least
        # one exactly matching block, so this has the same result as a scan.
        if any((signature^signatures[j]).bit_count()<=3
               for j in candidates):
            reasons['near_duplicate']+=1
            continue
        hashes.add(row['sha256'])
        j=len(signatures)
        signatures.append(signature)
        for block in range(4):
            buckets[(block,(signature>>(16*block))&65535)].append(j)
        keep.append(i)
    return keep,reasons


def _paired_name_rows(dgp,index,tokenizer):
    offset=dgp.rows[index]['index']
    pair=paired_document(dgp,index,random.Random(46401+offset),[],
                         **RULE_RATES)
    if pair is None:
        return None
    rows=[]
    for document in pair:
        found=[]
        for chunk in chunk_document(document,tokenizer,256,max_tokens=2048):
            sites=[site for site in chunk['sites']
                   if site['kind']=='surname_copy']
            if sites:
                site=sites[0]
                found.append({'x':chunk['x'],'y':chunk['y'],
                              'site':site['positions'][0],
                              'target':site['target_ids'][0]})
        if not found:
            return None
        rows.append(found[0])
    if rows[0]['target']==rows[1]['target']:
        return None
    return tuple(rows)


def _stack_name(records):
    return tuple({key:torch.stack([pair[arm][key] for pair in records])
                  if key in ('x','y') else
                  torch.tensor([pair[arm][key] for pair in records])
                  for key in ('x','y','site','target')}
                 for arm in (0,1))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-preview',type=Path,required=True)
    p.add_argument('--old-heldout-preview',type=Path,required=True)
    p.add_argument('--additional-prior-preview',type=Path,nargs='*',
                   default=[])
    p.add_argument('--fresh-preview',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-bracket-pairs',type=int,default=200)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    a=p.parse_args()
    torch.set_num_threads(4)
    prior=(json.loads(a.train_preview.read_text())+
           json.loads(a.old_heldout_preview.read_text())+
           [row for path in a.additional_prior_preview
            for row in json.loads(path.read_text())])
    rows=json.loads(a.fresh_preview.read_text())
    keep,reasons=_fresh_indices(prior,rows)
    if len(keep)<len(rows)*.9:
        raise RuntimeError('Fresh panel has too many near duplicates')
    dgp=NameCopyDGP(rows)
    tokenizer=Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    names=[]
    brackets=[]
    bracket_reasons=Counter()
    for i in keep:
        name=_paired_name_rows(dgp,i,tokenizer)
        if name is not None:
            names.append(name)
        bracket,reason=pair_for_text(rows[i]['preview'],tokenizer,
                                     rows[i]['index'],'bracket',256,2048)
        bracket_reasons[reason]+=1
        if bracket is not None and len(brackets)<a.max_bracket_pairs:
            brackets.append(bracket)
    if not names or not brackets:
        raise RuntimeError('No fresh rule pairs')
    packed_names=_stack_name(names)
    results={}
    for checkpoint in a.checkpoint:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        _,depth,d_model,heads=state['identity'][:4]
        model=_model(50257,256,depth,d_model,heads,a.device,'rms')
        model.load_state_dict(state['model'])
        model.eval()
        results[checkpoint.stem]={
            'step':state['step'],
            'surname':evaluate_name(model,packed_names,a.device,8),
            'bracket':one_kind(model,brackets,a.device,8,tokenizer)}
        print(checkpoint.stem,'surname paired rank',
              round(results[checkpoint.stem]['surname'][
                  'candidate_rank_paired'],3),
              'bracket paired two-way',
              round(results[checkpoint.stem]['bracket'][
                  'paired_two_way_fraction'],3),flush=True)
    report={'source':'HuggingFaceFW/fineweb sample-10BT train',
            'fresh_offsets':[rows[0]['index'],rows[-1]['index']],
            'screened_sources':len(keep),
            'duplicate_reasons':dict(reasons),
            'screened_source_digest':hashlib.sha256(''.join(sorted(
                rows[i]['sha256'] for i in keep)).encode()).hexdigest(),
            'name_pairs':len(names),'bracket_pairs':len(brackets),
            'bracket_build_reasons':dict(bracket_reasons),
            'models':results,
            'scope':'Previously unseen source range; no training, fitting, '
                    'or checkpoint selection on this panel. Exact full-source '
                    'hash and preview SimHash Hamming<=3 screen against '
                    'all provided prior source ranges. Same paired '
                    'surname generator and bracket intervention as the '
                    'exploratory pilot. Name identity remains heuristic; '
                    'source prose/token rows stay local.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
