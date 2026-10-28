import sys,torch,json
from pathlib import Path
sys.path.insert(0,'/home/loganriggs/Coding/neuron_circuits/scripts')
from run_bilin18_binding_role_order_panel import ART,load_full_model
import torch.nn.functional as F
torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
tasks=torch.load(ART/'bilin18_compact_fresh_contexts_captures.pt',weights_only=False);res=json.loads((ART/'bilin18_compact_fresh_contexts.json').read_text());axes=torch.load(ART/'bilin18_relay_contrast_subspaces_fits.pt',weights_only=False)['bases'];ex=tasks['examples'];caps=tasks['premlp_relay'];lookup={(e['id'],e['variant']):i for i,e in enumerate(ex)};jobs=[];xs=[];ds=[]
for r in res['records']:
 if r['kind']!='full':continue
 i=lookup[r['id'],r['recipient']];j=lookup[r['id'],r['donor']];x=caps[i].double();d=caps[j].double()-x;q=axes[r['family']][:,0].double()
 for arm,delta in [('full',d),('rank1',q*(d@q))]:
  jobs.append({k:r[k] for k in ['id','family','format','gap','source_pair','recipient','donor']}|dict(arm=arm,index=i));xs.append(x);ds.append(delta)
X=torch.stack(xs).float();D=torch.stack(ds).float();m=load_full_model('cuda');mlp=m.blocks[4].mlp.float().eval()
with torch.inference_mode():
 def truth(x):return torch.cat([mlp(F.rms_norm(b.cuda(),(1152,))).cpu() for b in x.split(64)])
 Y=truth(X);E=truth(X+D)-Y
assert torch.isfinite(E).all()
torch.save(dict(jobs=jobs,inputs=X,deltas=D,native_updates=Y,true_effects=E),ART/'bilin18_compact_fresh_transfer_captures.pt')
print('prepared',len(jobs),'jobs',2*len(jobs),'local teacher evaluations')
