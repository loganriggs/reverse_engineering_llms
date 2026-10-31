"""Same-gauge, different-label controls for full-model shared-key edits."""

import json

import numpy as np
import torch

from flagship_full_model_eval import GPT,GPTConfig,logits
from flagship_function_aware_shared_key import make_panels
from flagship_matrix_free import CORPUS,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_shared_key_counterfactual_causality import CaptureEdit,norm2
from flagship_shared_key_input_edit_fidelity import kl,records_from_reports
from flagship_shared_key_label_transfer import component_energy
from flagship_transfer import ART


SOURCES=(
 'all_head_shared_key_counterfactual_causality_n8_s91-97.json',
 'all_head_shared_key_counterfactual_causality_n8_s101-107_punct25-26_word286-257.json',
)
TARGETS=(('response_0',16,94),('direct_l1_native_0',3,106))


def controls(state,basis,rotations):
    panel=make_panels(state,128,4096,128)[0]['validation_fineweb']
    report=json.loads((ART/'all_head_shared_key_label_transfer_n32768_clip99.json').read_text())
    random_energy=component_energy(panel,basis@rotations['random_0']).sum(0)
    methods={};detail={}
    for gauge,candidate,random_control in TARGETS:
        u=basis@rotations[gauge]
        energy=component_energy(panel,u).sum(0)
        eligible=[r for r in report['records']
            if r['law']=='validation_fineweb' and r['gauge']==gauge
            and r['proposed_label']!='space_word -> punctuation']
        row=min(eligible,key=lambda r:abs(np.log(
                    energy[r['component']]/energy[candidate])))
        component=row['component']
        control_energy=float(energy[component])
        target_energy=float(random_energy[random_control])
        dose=float(np.sqrt(target_energy/control_energy))
        assert 0<dose<=1,(gauge,dose)
        weighted=u.copy();weighted[:,component]*=np.sqrt(1-dose)
        name=f'same_gauge_{gauge}_c{component}'
        methods[name]=weighted
        detail[name]=dict(gauge=gauge,component=component,
            validation_label=row['proposed_label'],
            validation_label_selection_rank=row['selection_rank'],
            full_pair_energy=control_energy,dose=dose,
            matched_pair_energy=dose*dose*control_energy,
            candidate_component=candidate,random_control_component=random_control,
            candidate_matched_pair_energy=target_energy)
    return methods,detail


def run(batch_size=2):
    torch.set_num_threads(4)
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    basis=np.load(ART/'all_head_function_aware_shared_key_s642_n32768_t160.npz')[
          'basis'].astype(np.float64)
    rotations=np.load(ART/'all_head_shared_key_gauge_priors_function_n2048_p2048_i120_d600.npz')
    methods,detail=controls(state,basis,rotations)
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval();model.load_state_dict(state);del state
    editor=CaptureEdit(model,128)
    corpora={'fineweb_test':CORPUS,**DATASETS}
    rows=records_from_reports(SOURCES)
    original={}
    for source in SOURCES:
        for r in json.loads((ART/source).read_text())['records']:
            if r['method']=='response_0_c16_matched':
                original[(source,r['corpus'],r['seed'],r['document_id'])]=r
    records=[];max_response_gate=0.
    with torch.no_grad():
        for offset in range(0,len(rows),batch_size):
            part=rows[offset:offset+batch_size]
            variants=[]
            for row in part:
                corpus=np.load(corpora[row['corpus']],mmap_mode='r')
                tokens=np.asarray(corpus[row['document_id'],
                          row['start']:row['start']+128],dtype=np.int64)
                assert len(tokens)==128 and tokens[row['key_position']]==row['original_key_id']
                tri=np.stack((tokens,tokens.copy(),tokens.copy()))
                tri[1,row['key_position']]=row['punctuation_id']
                tri[2,row['key_position']]=row['control_word_id']
                variants.append(tri)
            idx=torch.from_numpy(np.concatenate(variants).copy())
            editor.set_shared(basis)
            reference=logits(model,idx).numpy()
            reference_head=editor.last_attention.numpy()
            for local,row in enumerate(part):
                b=3*local;i=row['query_position']
                prior=original[(row['source'],row['corpus'],row['seed'],row['document_id'])]
                for v,key in ((1,'punctuation'),(2,'word')):
                    current=norm2(reference[b+v,i]-reference[b,i])
                    stored=prior[f'{key}_logit_response_energy']
                    gate=abs(current-stored)/max(stored,1e-10)
                    max_response_gate=max(max_response_gate,gate)
                    assert gate<1e-4,(row,key,gate)
            for name,cut in methods.items():
                editor.set_shared(cut)
                candidate=logits(model,idx).numpy()
                candidate_head=editor.last_attention.numpy()
                for local,row in enumerate(part):
                    b=3*local;i=row['query_position']
                    delta_p=reference[b+1,i]-reference[b,i]
                    delta_w=reference[b+2,i]-reference[b,i]
                    med_p=delta_p-(candidate[b+1,i]-candidate[b,i])
                    med_w=delta_w-(candidate[b+2,i]-candidate[b,i])
                    h_p=((reference_head[b+1,:,i]-reference_head[b,:,i])-
                         (candidate_head[b+1,:,i]-candidate_head[b,:,i]))
                    h_w=((reference_head[b+2,:,i]-reference_head[b,:,i])-
                         (candidate_head[b+2,:,i]-candidate_head[b,:,i]))
                    records.append(dict(**row,method=name,
                        base_query_kl=kl(reference[b,i],candidate[b,i]),
                        base_head_deletion_energy=norm2(reference_head[b,:,i]-candidate_head[b,:,i]),
                        punctuation_logit_response_energy=norm2(delta_p),
                        word_logit_response_energy=norm2(delta_w),
                        punctuation_mediated_logit_energy=norm2(med_p),
                        word_mediated_logit_energy=norm2(med_w),
                        punctuation_mediated_head_energy=norm2(h_p),
                        word_mediated_head_energy=norm2(h_w)))
            if (offset+len(part))%16==0:print('evaluated',offset+len(part),flush=True)
    output=dict(source_panels=list(SOURCES),rank=256,length=128,
        selection='same-gauge top-20 validation component with different proposed label and closest full pair-output energy; partial dose matches existing random-control energy',
        detail=detail,max_original_response_replay_relative_error=max_response_gate,
        records=records,
        scope='same prompts and query positions as candidate panel; full-model different-label control')
    (ART/'all_head_shared_key_same_gauge_controls_n96.json').write_text(
        json.dumps(output,indent=2)+'\n')
    return output


if __name__=='__main__':run()
