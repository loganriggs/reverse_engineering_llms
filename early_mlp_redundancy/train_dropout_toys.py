"""T4 fixed nine new runs plus three reused T3 controls; no outcome tuning."""
import argparse,hashlib,json
from pathlib import Path
import torch
from train_capacity_toys import batch,score
P=Path(__file__).resolve().parent
class Toy(torch.nn.Module):
 def __init__(self,w):
  super().__init__();self.L=torch.nn.Parameter(torch.randn(w,25)*.2);self.R=torch.nn.Parameter(torch.randn(w,25)*.2);self.D=torch.nn.Parameter(torch.randn(16,w)*.2);self.b=torch.nn.Parameter(torch.zeros(16))
 def forward(self,x):
  h=(x@self.L.T)*(x@self.R.T);return h@self.D.T+self.b,h

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');device='cuda' if torch.cuda.is_available() else 'cpu'
 oldpath=P/'validation/width_toys.json';old=json.loads(oldpath.read_text());samples=torch.load('/tmp/emlp_t3_models.pt',weights_only=True);controls={r['seed']:(r,samples[i]) for i,r in enumerate(old['rows']) if r['train_noise']==.25 and r['penalty']==.01};assert len(controls)==3
 rows=[];saved=[];masksets={};replay=[]
 for width in (16,32):
  masks=[]
  for j in range(16):
   keep=torch.ones(width,device=device);keep[torch.randperm(width,generator=torch.Generator().manual_seed(600+j))[:width//4]]=0;masks.append(keep)
  masksets[str(width)]=[m.cpu().tolist() for m in masks]
  for dropout in (0.,.25):
   for seed in (0,1,2):
    torch.manual_seed(100+seed);model=Toy(width).to(device);history=[];reused=width==16 and dropout==0
    if reused:model.load_state_dict(controls[seed][1]['state']);history=controls[seed][0]['history']
    else:
     opt=torch.optim.Adam(model.parameters(),lr=.01);g=torch.Generator(device=device).manual_seed(200+seed);dg=torch.Generator(device=device).manual_seed(900+seed)
     for step in range(2000):
      x,y,labels,centers=batch(256,16,.25,g,device);pred,h=model(x)
      if dropout:
       keep=torch.ones(width,device=device);keep[torch.randperm(width,device=device,generator=dg)[:width//4]]=0;pred=(h*keep)@model.D.T+model.b
      mse=(pred-y).square().sum(-1).mean();energy=(h.square()*model.D.square().sum(0)).sum(-1).mean();loss=mse+.01*energy
      opt.zero_grad();loss.backward();opt.step()
      if (step+1)%500==0:history.append(dict(step=step+1,error=float(mse.detach()),contribution_energy=float(energy.detach()),objective=float(loss.detach())))
    with torch.no_grad():
     tests=[];evalsamples=[]
     for noise in (0.,.25,.5):
      eg=torch.Generator(device=device).manual_seed(800);x,y,labels,centers=batch(1024,16,noise,eg,device);pred,h=model(x);tests.append(dict(noise=noise,**score(pred,y,labels,centers)));evalsamples.append(dict(noise=noise,inputs=x.cpu(),targets=y.cpu(),labels=labels.cpu(),centers=centers.cpu(),predictions=pred.cpu()))
      if noise==0:clean=(x,y,labels,centers,pred,h)
     x,y,labels,centers,pred,h=clean;net=pred-model.b;e=h.double().square()*model.D.double().square().sum(0);energy=e.sum();netenergy=net.double().square().sum();mass=(h.double().abs()*model.D.double().norm(dim=0)).sum(-1);perclass=torch.stack([e[labels==i].mean(0) for i in range(16)])
     geometry=dict(channel_net_energy_ratio=float(energy/netenergy),participation=float(mass.square().sum()/energy),class_energy_specificity=float(perclass.max(0).values.sum()/perclass.sum()))
     deletions=[score((h*keep)@model.D.T+model.b,y,labels,centers) for keep in masks];singles=[]
     for j in range(width):
      hh=h.clone();hh[:,j]=0;singles.append(score(hh@model.D.T+model.b,y,labels,centers))
     byclass=torch.tensor([[r['confusion'][i][i]/sum(r['confusion'][i]) for i in range(16)] for r in singles]);targeted=dict(mean_single_deletion_accuracy=float(byclass.mean()),worst_channel_global_accuracy=float(byclass.mean(1).min()),mean_class_accuracy_under_its_worst_channel=float(byclass.min(0).values.mean()),classes_with_any_single_deletion_error=int((byclass.min(0).values<1).sum()))
     row=dict(width=width,dropout=dropout,seed=seed,reused_T3=reused,parameters=sum(p.numel() for p in model.parameters()),history=history,evaluations=tests,deletions=deletions,singles=singles,targeted=targeted,geometry=geometry)
     if reused:
      reference=controls[seed][0]
      for left,right in zip(tests+deletions,reference['evaluations']+reference['deletions']):
       for key in ('squared_error','accuracy'):replay.append(abs(left[key]-right[key]))
     rows.append(row);saved.append(dict(row=len(rows)-1,state={k:v.cpu() for k,v in model.state_dict().items()},evaluation_samples=evalsamples));print('completed',len(rows),12,width,dropout,seed,flush=True)
 assert max(replay)==0
 out=dict(status='completed',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((P/'TOY_DROPOUT_PROTOCOL.md').read_bytes()).hexdigest(),shared_helpers_sha256=hashlib.sha256((P/'train_capacity_toys.py').read_bytes()).hexdigest(),T3_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest(),new_runs=9,reused_runs=3,reused_control_max_error=max(replay),masks=masksets,rows=rows)
 args.output.write_text(json.dumps(out,indent=2)+'\n');torch.save(saved,'/tmp/emlp_t4_models.pt')
if __name__=='__main__':main()
