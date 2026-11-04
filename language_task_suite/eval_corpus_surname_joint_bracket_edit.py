"""Compare frozen mixed-rule models on heldout bracket counterfactuals."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from eval_corpus_rule_reference_transfer import one_kind, pair_for_text
from train_corpus_delimiter_bilinear import _model


def build_records(rows,tokenizer,max_pairs=200):
    records=[]
    reasons=Counter()
    for source in rows:
        if int(source['sha256'][:8],16)%2==0:
            continue  # validation partition; evaluate test sources only
        pair,reason=pair_for_text(source['preview'],tokenizer,
                                  source['index'],'bracket',256,2048)
        reasons[reason]+=1
        if pair is not None:
            records.append(pair)
        if len(records)>=max_pairs:
            break
    if not records:
        raise RuntimeError('No BPE-aligned bracket counterfactuals')
    return records,reasons


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--heldout-preview',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-pairs',type=int,default=200)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    a=p.parse_args()
    torch.set_num_threads(4)
    tokenizer=Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    rows=json.loads(a.heldout_preview.read_text())
    records,reasons=build_records(rows,tokenizer,a.max_pairs)
    results={}
    for checkpoint in a.checkpoint:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        identity=state['identity']
        _,depth,d_model,heads=identity[:4]
        model=_model(50257,256,depth,d_model,heads,a.device,'rms')
        model.load_state_dict(state['model'])
        model.eval()
        results[checkpoint.stem]=one_kind(model,records,a.device,8,tokenizer)
        print(checkpoint.stem,results[checkpoint.stem],flush=True)
    report={'source':'HuggingFaceFW/fineweb sample-10BT train',
            'source_offsets':[rows[0]['index'],rows[-1]['index']],
            'selection':'sha256 parity odd: mixed-cache test split',
            'max_pairs':a.max_pairs,'n_pairs':len(records),
            'build_reasons':dict(reasons),'models':results,
            'scope':'Natural round bracket versus same-source square-bracket '
                    'edit. Source-disjoint from 10k training range but not '
                    'a final untouched holdout: the test range was monitored '
                    'during exploratory training. Raw prose/token pairs '
                    'are not committed.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
