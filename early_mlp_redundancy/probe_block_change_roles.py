"""M105 resumable native-background group-change roles; all final vectors saved."""
import json,sys,hashlib
from pathlib import Path
import torch
from fit_pca_encoder import V,A,sha
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
from block_change_components import grouping,changes


def margin(z,g,v):
    z=z.double();return float(torch.logsumexp(z[g[v]],0)-torch.logsumexp(z[g[1-v]],0))


def main():
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
    cohort=sys.argv[1];assert cohort in ['train','source','whole']
    xp=V/'block_change_roles_execution.json';execution=json.loads(xp.read_text());assert execution['source_sha256']==sha(__file__)
    for name,h in execution['helpers'].items():assert sha(V.parent/name)==h
    for name,h in execution['inputs'].items():assert sha(V/name)==h
    protocol=json.loads((V/'block_change_roles_protocol.json').read_text());assert json.loads((V/'block_change_roles_preflight.json').read_text())['status']=='passed'
    stem={'train':'training_causal_coverage','source':'source_holdout_native','whole':'whole_task_native'}[cohort]
    native=json.loads((V/(stem+'.json')).read_text());pairs=json.loads((V/(stem+'_protocol.json')).read_text())['pairs'];wanted=protocol[{'train':'training_pair_ids','source':'source_group_pair_ids','whole':'whole_task_pair_ids'}[cohort]];byid={p['id']:p for p in pairs};pairs=[byid[pid] for pid in wanted]
    assert sha(native['samples_path'])==native['samples_sha256'];saved=torch.load(native['samples_path'],weights_only=True)
    model,receipt=load_model();assert receipt.weights_sha256==native['checkpoint_sha256'];mlp=model.transformer.h[1].mlp;models=[]
    for e in protocol['models']:
        assert sha(e['path'])==e['sha256'];state=torch.load(e['path'],weights_only=True);f=Featurizer('block',e['seed']).cuda();f.load_state_dict(state['state']);models.append((e,f,state['mean'].cuda(),state['scale'].cuda(),torch.nn.functional.normalize(f.dictionary,dim=0)))
    dest=A/f'emlp_m105_{cohort}_samples';dest.mkdir(exist_ok=True);progress=A/f'emlp_m105_{cohort}_progress.pt';op=V/f'block_change_roles_{cohort}.json';assert not op.exists()
    rows=[];controls=[];interactions=[];files=[];calls=0;completed=0;replays=0;ineligible=[]
    if progress.exists():
        old=torch.load(progress,weights_only=True);assert old['execution_sha256']==sha(xp)
        rows,controls,interactions,files,calls,completed,replays,ineligible=[old[k] for k in ['rows','controls','interactions','files','calls','completed','replays','ineligible']]
        assert [f['pair_id'] for f in files]==wanted[:completed]
        for f in files:assert sha(f['path'])==f['sha256']
    for pi,pair in enumerate(pairs):
        if pi<completed:continue
        pid=pair['id'];blob=dict(pair_id=pid,writes=[],baselines=[],reconstructions={},changed={},full={});cs=[]
        for ep in [0,1]:
            ids=torch.tensor([pair['token_ids'][ep]],device='cuda');out,cap=trace(model,ids);calls+=1;key=pid+f'|{ep}';base=out[0,-1];assert torch.equal(base.cpu(),saved['baselines'][key]['logits']);replays+=1;write=mlp.Down(cap[1]['products']);cs.append(dict(ids=ids,base=base,write=write));blob['writes'].append(write.cpu());blob['baselines'].append(base.cpu())
        targets=[t for t in native['records'] if t['pair_id']==pid and t['layer']==1];assert len(targets)==6
        for target in targets:
            if target['status']!='evaluated':continue
            ep=target['endpoint'];c=cs[ep];pos=target['recipient_positions'];dp=target['donor_positions'];nd=torch.zeros_like(c['write']);nd[:,pos]=cs[1-ep]['write'][:,dp]-c['write'][:,pos]
            assert torch.equal(nd[:,pos].cpu(),saved['deltas'][target['sample_key']]['delta'])
            if bool(nd.count_nonzero()):out,_=trace(model,c['ids'],replace=lambda layer,z,w:w+nd if layer==1 else w);calls+=1;teacher=out[0,-1]
            else:teacher=c['base']
            assert torch.equal(teacher.cpu(),saved['changed'][target['sample_key']]);replays+=1
        for e,f,mu,scale,D in models:
            modelkey=f"{e['seed']}:{e['arm']}";recs=[];codes=[];supports=[]
            for ep,c in enumerate(cs):
                r,support,code=f((c['write']-mu)/scale);recs.append(r*scale+mu);codes.append(code);supports.append(support);blob['reconstructions'][modelkey+f'|{ep}']=dict(write=recs[-1].cpu(),code=code.cpu(),support=support.cpu())
            for target in targets:
                ep=target['endpoint'];region=target['region'];ident=dict(pair_id=pid,family=pair['family'],endpoint=ep,region=region,seed=e['seed'],arm=e['arm'])
                if target['status']!='evaluated':ineligible.append(ident|dict(status=target['status'],reason=target.get('reason')));continue
                c=cs[ep];pos=target['recipient_positions'];dp=target['donor_positions'];g=pair['answer_groups'][ep];variant=pair['variants'][ep];base_lp=c['base'].double().log_softmax(-1);prob=base_lp.exp();base_margin=margin(c['base'],g,variant);teacher=saved['changed'][target['sample_key']].cuda();teacher_response=teacher.double().log_softmax(-1)-base_lp
                def patched(local_delta):
                    nonlocal calls
                    delta=torch.zeros_like(c['write']);delta[:,pos]=local_delta
                    if not bool(delta.count_nonzero()):return c['base'],True
                    out,_=trace(model,c['ids'],replace=lambda layer,z,w:w+delta if layer==1 else w);calls+=1;return out[0,-1],False
                def scored(out):
                    response=out.double().log_softmax(-1)-base_lp
                    return response,dict(response_energy=float((prob*response.square()).sum()),teacher_error=float((prob*(response-teacher_response).square()).sum()),margin_change=margin(out,g,variant)-base_margin)
                full=recs[1-ep][:,dp]-recs[ep][:,pos];joint,noop=patched(full);joint_response,met=scored(joint);key=modelkey+f'|{ep}|{region}';blob['full'][key]=joint.cpu();controls.append(ident|met|dict(noop=noop,delta_energy=float(full.double().square().sum()),teacher_response_energy=float((prob*teacher_response.square()).sum()),teacher_margin_change=target['margin_change']))
                for shuffled in [False,True]:
                    groupname='shuffled' if shuffled else 'trained';groups=grouping(e['seed'],shuffled);parts=changes(codes[ep],codes[1-ep],D,scale,pos,dp,groups);err=float(torch.linalg.vector_norm(parts.sum(0).double()-full.double())/max(float(torch.linalg.vector_norm(full.double())),1.));assert err<1e-5
                    summed=torch.zeros_like(joint_response);summed_margin=0.;energy_sum=0.
                    for group in range(32):
                        output,isnoop=patched(parts[group]);response,metrics=scored(output);summed+=response;summed_margin+=metrics['margin_change'];energy_sum+=metrics['response_energy'];atoms=torch.as_tensor(groups[group],device='cuda');active=(supports[ep][:,pos][:,:,atoms].bool()|supports[1-ep][:,dp][:,:,atoms].bool()).sum()
                        rows.append(ident|metrics|dict(grouping=groupname,group=group,noop=isnoop,delta_energy=float(parts[group].double().square().sum()),active_coordinate_positions=int(active),positions=len(pos),local_sum_relative_error=err));blob['changed'][key+f'|{groupname}|{group}']=output.cpu()
                    interactions.append(ident|dict(grouping=groupname,sum_response_energy=float((prob*summed.square()).sum()),sum_individual_response_energies=energy_sum,joint_response_energy=met['response_energy'],nonadditive_response_energy=float((prob*(summed-joint_response).square()).sum()),sum_margin_change=summed_margin,joint_margin_change=met['margin_change'],margin_nonadditivity=summed_margin-met['margin_change']))
        path=dest/(hashlib.sha256(pid.encode()).hexdigest()+'.pt');torch.save(blob,path);files.append(dict(pair_id=pid,path=str(path),sha256=sha(path)));tmp=progress.with_suffix('.tmp');torch.save(dict(execution_sha256=sha(xp),rows=rows,controls=controls,interactions=interactions,files=files,calls=calls,completed=pi+1,replays=replays,ineligible=ineligible),tmp);tmp.replace(progress);print('COHORT',cohort,'PAIRS',pi+1,len(pairs),'FORWARDS',calls,flush=True)
    result=dict(status='completed',cohort=cohort,source_sha256=sha(__file__),execution_sha256=sha(xp),forward_count=calls,native_replays=replays,rows=rows,controls=controls,interactions=interactions,files=files,ineligible=ineligible,scope='Every individual group and joint encoded change applied in native background; all final logits saved. Native baselines and native full-change teachers exactly replayed. Individual responses not assumed additive. Frozen trained/shuffled groups, allmodels; no fit or selection. Must independently audit before inference.')
    op.write_text(json.dumps(result,indent=2)+'\n');print('DONE',cohort,len(rows),'GROUPS',len(controls),'JOINT',calls,'FORWARDS',flush=True)
if __name__=='__main__':main()
