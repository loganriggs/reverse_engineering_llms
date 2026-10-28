from pathlib import Path
import json,hashlib
P=Path('/home/loganriggs/Coding/reverse_engineering_llms/early_mlp_redundancy');V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((V/'context_training_protocol.json').read_text());starts=json.loads((V/'response_supervision_protocol.json').read_text())['initial_models'];b=dict(old);b['initial_models']=starts;b['parent_protocol_sha256']=sha(V/'context_training_protocol.json');b['basis_change']='M18 expanded starting bases replace M11 bases; all causal input/group/region selections unchanged.'
for key in old:
 if key!='initial_models':assert b[key]==old[key]
f=V/'m18_basis_benchmark_protocol.json';assert not f.exists();f.write_text(json.dumps(b,indent=2)+'\n')
p=json.loads((V/'finite_candidate_protocol.json').read_text());p['arms']=[f'{name}:{m}:live' for name in ['aggregate_gradient','aggregate_KL'] for m in [1.,.1]];p['basis_protocol_sha256']=sha(f);p['scope']='M29 fixed evaluation inputs/candidates/group indices/regions, M18 versus M11 intervention basis. Native effects deliberately change; compare joint/control within basis. No new fits or ordinary output evaluation.';pp=V/'basis_transfer_protocol.json';assert not pp.exists();pp.write_text(json.dumps(p,indent=2)+'\n');models=json.loads((V/'unseen_direction_protocol.json').read_text())['models'];assert len(models)==20
for e in models:assert sha(Path(e['path']))==e['sha256']
rp=V/'basis_transfer_models.json';rp.write_text(json.dumps(dict(protocol_sha256=sha(pp),models=models),indent=2)+'\n');(V/'basis_transfer_models_audit.json').write_text(json.dumps(dict(status='passed',result_sha256=sha(rp),models_verified=20,scope='Identical already-evaluated M26/M27 checkpoints; hash verification only, no refitting.'),indent=2)+'\n')
s=(P/'probe_finite_candidate_native.py').read_text().replace('M26','M29').replace('finite_candidate','basis_transfer').replace('emlp_m26','emlp_m29').replace('torch.set_num_threads(6)','torch.set_num_threads(1)').replace("bp=P/'context_training_protocol.json'","bp=P/'m18_basis_benchmark_protocol.json'")
s=s.replace("assert old['protocol_sha256']==sha(bp)","assert old['protocol_sha256']==benchmark['parent_protocol_sha256']==sha(P/'context_training_protocol.json')")
line="       prior=oldrecords[pair['id'],ep,kind,seed,spec['region'],g];assert abs(prior['native_margin_effect']-neffect)<1e-10 and abs(prior['native_logprob_response_energy']-response_energy)<1e-10;assert prior['noop']==noop;assert abs(prior['delta_energy']-float(delta.double().square().sum()))<1e-8\n";assert line in s;s=s.replace(line,'')
# Per-pair resume keeps completed scientific observations after runtime failures.
s=s.replace('records=[];baselines=[];samples=[];calls=0;noop_directions=0',"records=[];baselines=[];samples=[];calls=0;noop_directions=0;completed=0;progress=A/'emlp_m29_basis_progress.pt'\n if progress.exists():\n  prior=torch.load(progress,weights_only=True);assert prior['protocol_sha256']==sha(pp);records=prior['records'];baselines=prior['baselines'];samples=prior['samples'];calls=prior['calls'];noop_directions=prior['noop_directions'];completed=prior['completed'];print('RESUME',completed,flush=True)")
s=s.replace("for pi,selected in enumerate(benchmark['causal_pairs']):\n  pair=", "for pi,selected in enumerate(benchmark['causal_pairs']):\n  if pi<completed:continue\n  pair=")
s=s.replace("  print('CAUSAL_PAIRS',pi+1,len(benchmark['causal_pairs']),'FORWARDS',calls,flush=True)","  print('CAUSAL_PAIRS',pi+1,len(benchmark['causal_pairs']),'FORWARDS',calls,flush=True)\n  tmp=progress.with_suffix('.tmp');torch.save(dict(protocol_sha256=sha(pp),records=records,baselines=baselines,samples=samples,calls=calls,noop_directions=noop_directions,completed=pi+1),tmp);tmp.replace(progress)")
(P/'probe_basis_transfer_native.py').write_text(s)
s=(P/'audit_finite_candidate_native.py').read_text().replace('M26','M29').replace('finite_candidate','basis_transfer').replace("bp=P/'context_training_protocol.json'","bp=P/'m18_basis_benchmark_protocol.json'")
a=s.index("oldrecords={(x['pair_id']");b=s.index('oldbaselines=',a);s=s[:a]+s[b:]
# Keep initial checkpoint replay, then require every joint/control intact output to replay.
a=s.index("out=dict(status='passed'")
extra="""reference={}
for filename in ['finite_candidate_native_fidelity.json','aggregate_kl_native_fidelity.json']:
 for x in json.loads((P/filename).read_text())['baselines']:reference[x['pair_id'],x['endpoint'],x['kind'],x['seed'],x['arm']]=x
for x in r['baselines']:
 y=reference[x['pair_id'],x['endpoint'],x['kind'],x['seed'],x['arm']]
 for key in ['task','native_task']:assert x[key]==y[key]
 close(x['final_KL'],y['final_KL'])
parent=json.loads((P/'context_training_protocol.json').read_text());assert benchmark['parent_protocol_sha256']==sha(P/'context_training_protocol.json')
for key in parent:
 if key!='initial_models':assert benchmark[key]==parent[key]
"""
s=s[:a]+extra+s[a:];s=s.replace('old_benchmark_native_effects_replayed=True','all_candidate_intact_baselines_replayed=True,only_basis_protocol_changed=True').replace('every intervention key/noop/shared-native invariant','every intervention key/noop/within-basis shared-native invariant');(P/'audit_basis_transfer_native.py').write_text(s)
print('Frozen basis-only comparison')
