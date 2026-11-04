"""Frozen fresh-source bracket test for attention sparse causal-fit arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from eval_corpus_rule_reference_ov_rank import fidelity
from eval_corpus_rule_reference_transfer import one_kind
from eval_corpus_surname_joint_bracket_edit import build_records
from eval_corpus_surname_joint_fresh import _fresh_indices
from sparsify_corpus_rule_reference_attention import attention_parameters
from train_corpus_delimiter_bilinear import _model


def load_model(checkpoint,device):
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    _,depth,width,heads=state['identity'][:4]
    model=_model(50257,256,depth,width,heads,device,'rms')
    model.load_state_dict(state['model'])
    return model.eval(),state


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prior-preview',type=Path,nargs='+',required=True)
    p.add_argument('--fresh-preview',type=Path,required=True)
    p.add_argument('--teacher-checkpoint',type=Path,nargs='+',required=True)
    p.add_argument('--sparse-dir',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-pairs',type=int,default=200)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if len(args.teacher_checkpoint)!=len(args.sparse_dir):
        raise ValueError('Pair each teacher with a sparse directory')
    torch.set_num_threads(4)
    prior=[row for path in args.prior_preview
           for row in json.loads(path.read_text())]
    rows=json.loads(args.fresh_preview.read_text())
    keep,reasons=_fresh_indices(prior,rows)
    if len(keep)<.9*len(rows):
        raise RuntimeError('Excessive source duplicates')
    tokenizer=Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    panel=[rows[i] for i in keep]
    records,pair_reasons=build_records(panel,tokenizer,args.max_pairs)
    result={'source_offsets':[rows[0]['index'],rows[-1]['index']],
            'screened_sources':len(keep),'duplicate_reasons':dict(reasons),
            'bracket_pairs':len(records),'pair_reasons':dict(pair_reasons),
            'scope':'Frozen teacher and both recovered sparse arms. '
                    'Sources screened by exact full-source hash and '
                    'preview SimHash<=3 against every supplied prior panel. '
                    'No fitting or stage selection on this source panel. '
                    'Only attention edges are pruned; other weights dense.',
            'models':{}}
    for teacher_path,directory in zip(args.teacher_checkpoint,
                                      args.sparse_dir,strict=True):
        teacher,state=load_model(teacher_path,args.device)
        key=teacher_path.stem
        baseline=one_kind(teacher,records,args.device,8,tokenizer)
        result['models'][key]={'teacher_step':state['step'],
                               'teacher':baseline,'arms':{}}
        for arm in ('distribution','causal'):
            saved=torch.load(directory/f'{arm}_recovered.pt',
                             map_location='cpu',weights_only=True)
            if Path(saved['teacher_checkpoint'])!=teacher_path:
                raise ValueError('Sparse checkpoint from another teacher')
            student,_=load_model(teacher_path,args.device)
            student.load_state_dict(saved['model'])
            parameters=attention_parameters(student)
            if set(parameters)!=set(saved['masks']):
                raise ValueError('Sparse attention mask keys differ')
            if any(torch.count_nonzero(parameters[name].detach().cpu()[
                    ~mask]).item() for name,mask in saved['masks'].items()):
                raise AssertionError('Masked attention edge is nonzero')
            live=sum(int(mask.sum()) for mask in saved['masks'].values())
            total=sum(mask.numel() for mask in saved['masks'].values())
            behavior=one_kind(student,records,args.device,8,tokenizer)
            preserved=fidelity(teacher,student,records,args.device,8)
            result['models'][key]['arms'][arm]={
                'attention_edges_live':live,
                'attention_edges_total':total,
                'behavior':behavior,'teacher_fidelity':preserved}
            print(key,arm,'edges',live,'two-way',
                  round(behavior['paired_two_way_fraction'],3),
                  'effect MAE',
                  round(preserved['effect_mean_abs_error'],3),flush=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
