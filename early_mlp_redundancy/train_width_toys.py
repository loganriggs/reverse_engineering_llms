"""T3 fixed twelve width16 runs; compare with existing T2 width8 fine-task runs."""
import argparse,hashlib,json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
class Toy(torch.nn.Module):
 def __init__(self):
  super().__init__();self.L=torch.nn.Parameter(torch.randn(16,25)*.2);self.R=torch.nn.Parameter(torch.randn(16,25)*.2);self.D=torch.nn.Parameter(torch.randn(16,16)*.2);self.b=torch.nn.Parameter(torch.zeros(16))
 def forward(self,x):
  h=(x@self.L.T)*(x@self.R.T);return h@self.D.T+self.b,h

def batch(n,C,sigma,g,device):
 ids=torch.arange(n,device=device)%16;z=torch.eye(16,device=device)[ids]+sigma*torch.randn(n,16,device=device,generator=g)
 x=torch.cat([z,torch.randn(n,8,device=device,generator=g),torch.ones(n,1,device=device)],-1)
 centers=torch.eye(16,device=device)[:C];centers=(centers-centers.mean(0))/((1-1/C)**.5)
 labels=ids//(16//C);return x,centers[labels],labels,centers

def score(pred,y,labels,centers):
 guess=(pred@centers.T).argmax(-1);C=len(centers);conf=torch.bincount(labels*C+guess,minlength=C*C).reshape(C,C)
 return dict(squared_error=float((pred-y).square().sum(-1).mean()),accuracy=float((guess==labels).float().mean()),confusion=conf.cpu().tolist())

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');device='cuda' if torch.cuda.is_available() else 'cpu'
 rows=[];saved=[]
 masks=[]
 for j in range(16):
  keep=torch.ones(16,device=device);idx=torch.randperm(16,generator=torch.Generator().manual_seed(600+j))[:4];keep[idx]=0;masks.append(keep)
 for C in (16,):
  for sigma in (0.,.25):
   for penalty in (0.,.01):
    for seed in (0,1,2):
     torch.manual_seed(100+seed);model=Toy().to(device);opt=torch.optim.Adam(model.parameters(),lr=.01);g=torch.Generator(device=device).manual_seed(200+seed);history=[]
     for step in range(2000):
      x,y,labels,centers=batch(256,C,sigma,g,device);pred,h=model(x);mse=(pred-y).square().sum(-1).mean();energy=(h.square()*model.D.square().sum(0)).sum(-1).mean();loss=mse+penalty*energy
      opt.zero_grad();loss.backward();opt.step()
      if (step+1)%500==0:history.append(dict(step=step+1,error=float(mse.detach()),contribution_energy=float(energy.detach()),objective=float(loss.detach())))
     with torch.no_grad():
      tests=[];evaluation_samples=[]
      for noise in (0.,.25,.5):
       eg=torch.Generator(device=device).manual_seed(800);x,y,labels,centers=batch(1024,C,noise,eg,device);pred,h=model(x);tests.append(dict(noise=noise,**score(pred,y,labels,centers)))
       evaluation_samples.append(dict(noise=noise,inputs=x.cpu(),targets=y.cpu(),labels=labels.cpu(),centers=centers.cpu(),predictions=pred.cpu()))
       if noise==0:
        clean=(x,y,labels,centers,pred,h)
      x,y,labels,centers,pred,h=clean;net=pred-model.b;energy=(h.double().square()*model.D.double().square().sum(0)).sum();netenergy=net.double().square().sum();mass=(h.double().abs()*model.D.double().norm(dim=0)).sum(-1)
      geometry=dict(channel_net_energy_ratio=float(energy/netenergy),signed_interference=float((netenergy-energy)/energy),participation=float(mass.square().sum()/energy),output_singular_values=torch.linalg.svdvals((pred-pred.mean(0)).double()).cpu().tolist())
      deletions=[]
      for keep in masks:deletions.append(score((h*keep)@model.D.T+model.b,y,labels,centers))
      row=dict(width=16,classes=C,train_noise=sigma,penalty=penalty,seed=seed,parameters=sum(p.numel() for p in model.parameters()),history=history,evaluations=tests,geometry=geometry,deletions=deletions)
      rows.append(row);saved.append(dict(row=len(rows)-1,state={k:v.detach().cpu() for k,v in model.state_dict().items()},clean_inputs=x.cpu(),clean_targets=y.cpu(),labels=labels.cpu(),centers=centers.cpu(),predictions=pred.cpu(),evaluation_samples=evaluation_samples))
      print('completed',len(rows),12,C,sigma,penalty,seed,'clean',tests[0]['squared_error'],flush=True)
 out=dict(status='completed',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((P/'TOY_WIDTH_PROTOCOL.md').read_bytes()).hexdigest(),device=device,steps=2000,masks=[m.cpu().tolist() for m in masks],rows=rows)
 args.output.write_text(json.dumps(out,indent=2)+'\n');torch.save(saved,'/tmp/emlp_t3_models.pt')
if __name__=='__main__':main()
