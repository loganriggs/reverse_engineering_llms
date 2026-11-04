"""Weight-only versus train-activation OV rank cuts on joint-rule model."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from eval_corpus_rule_reference_transfer import one_kind
from eval_corpus_rule_reference_ov_rank import fidelity
from eval_corpus_surname_joint_bracket_edit import build_records
from ov_linear_reduction import activation_grams, replace_ov
from train_corpus_delimiter_bilinear import _model


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--heldout-preview',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fit-rows',type=int,default=128)
    p.add_argument('--max-pairs',type=int,default=200)
    p.add_argument('--panel-status',choices=('monitored','fresh'),
                   default='monitored')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    a=p.parse_args()
    torch.set_num_threads(4)
    cache=torch.load(a.cache,map_location='cpu',weights_only=True)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    identity=state['identity']
    if Path(identity[0])!=a.cache.resolve():
        raise ValueError('Model and cache differ')
    _,depth,d_model,heads=identity[:4]
    teacher=_model(50257,256,depth,d_model,heads,a.device,'rms')
    teacher.load_state_dict(state['model'])
    teacher.eval()
    train_background=cache['train']['background']
    fit={key:value[:a.fit_rows]
         for key,value in train_background.items() if key in ('x','y')}
    grams,counts=activation_grams(teacher,fit,a.device)
    tokenizer=Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    sources=json.loads(a.heldout_preview.read_text())
    records,reasons=build_records(sources,tokenizer,a.max_pairs)
    original=one_kind(teacher,records,a.device,8,tokenizer)
    report={'checkpoint_step':state['step'],
            'source_offsets':[sources[0]['index'],sources[-1]['index']],
            'fit_background_rows':len(fit['x']),
            'fit_activation_counts':counts,
            'heldout_pairs':len(records),'pair_build_reasons':dict(reasons),
            'teacher':original,'reductions':{},
            'panel_status':a.panel_status,
            'scope':'Per-head OV product SVD; all QK, MLP, embedding, and '
                    'unembedding weights fixed. Weight-only versus '
                    'train-background-activation input metric. Both fit '
                    'without evaluation sources. Evaluate full-vocab '
                    'behavior, two-candidate bracket response, teacher KL, '
                    'and paired edit effect. Panel status is recorded '
                    'separately. No unique circuit claim.'}
    for rank in (4,8,16):
        for method in ('weight','data'):
            student=deepcopy(teacher).eval()
            errors=replace_ov(student,grams,rank,method)
            behavior=one_kind(student,records,a.device,8,tokenizer)
            preserved=fidelity(teacher,student,records,a.device,8)
            key=f'{method}_rank{rank}'
            report['reductions'][key]={'behavior':behavior,
                                       'teacher_fidelity':preserved,
                                       'reduction_errors':errors}
            print(key,'paired_two_way',
                  round(behavior['paired_two_way_fraction'],3),
                  'paired_top1',round(behavior['paired_top1_fraction'],3),
                  'effect_MAE',
                  round(preserved['effect_mean_abs_error'],3),flush=True)
            if rank==16 and (preserved['effect_mean_abs_error']>1e-3 or
                             preserved['site_teacher_kl']>1e-4):
                raise AssertionError('Full-rank OV refactor is not faithful')
            a.output.parent.mkdir(parents=True,exist_ok=True)
            a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('saved',a.output,flush=True)


if __name__=='__main__':
    main()
