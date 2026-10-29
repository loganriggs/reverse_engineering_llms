import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from readout_speed import calc,features,optimum,retract,readout_flow
from multitask import monomials,product_map
from experiment import gram

ROOT=Path('results/readout_speed_confirmation');SOURCE=ROOT/'data'
METHODS=[('joint',s) for s in [1,10,100]]+[('reduced',0)]+[('split',s) for s in [1,10,100]]

def cosines(rows,teacher):
    def matrices(r):return (r[:4,:,None]*r[4:,None,:]+r[4:,:,None]*r[:4,None,:])/2
    def inner(a,b):return 2*np.einsum('iab,jab->ij',a,b)+np.outer(np.trace(a,axis1=1,axis2=2),np.trace(b,axis1=1,axis2=2))
    a,b=matrices(rows),matrices(teacher);C=abs(inner(a,b))/np.sqrt(np.outer(np.diag(inner(a,a)),np.diag(inner(b,b))))
    i,j=linear_sum_assignment(-C);return C[i,j]

def generate_data(keys,G):
    exponents=np.array(keys)
    def design(x):return np.prod(x[:,None,:]**exponents[None,:,:],axis=2)
    for seed in range(249101,249106):
        rng=np.random.default_rng(seed);A=rng.normal(size=(4,8));B=rng.normal(size=(4,8))
        A/=np.linalg.norm(A,axis=1)[:,None];B/=np.linalg.norm(B,axis=1)[:,None]
        W=rng.normal(size=(14,10))/np.sqrt(10);C=product_map(A,B,keys)@W.T
        scale=np.sqrt(np.sum(C*(G@C),axis=0));W/=scale[:,None];C/=scale
        x=rng.normal(size=(512,8));xa=rng.normal(size=(128,8));noise=rng.normal(size=(512,10));adaptation_noise=rng.normal(size=(128,4))
        y=design(x)@C[:,:10]+.05*noise;ya=design(xa)@C[:,10:]+.05*adaptation_noise
        np.savez_compressed(SOURCE/f'{seed}_data.npz',x=x,y=y,xa=xa,ya=ya,teacher_A=A,teacher_B=B,teacher_W=W,truth=C,noise=noise,adaptation_noise=adaptation_noise)

def main():
    ROOT.mkdir(exist_ok=False);SOURCE.mkdir();records=[];keys=monomials(8);G=gram(keys)
    generate_data(keys,G)
    for seed in range(249101,249106):
      data=np.load(SOURCE/f'{seed}_data.npz');x,y=data['x'],data['y']
      for init in [250001,250002]:
       rng=np.random.default_rng(init);rows0=rng.normal(size=(8,8));rows0/=np.linalg.norm(rows0,axis=1)[:,None];u0=optimum(rows0,x,y)
       for method,speed in METHODS:
        rows=rows0.copy();U=u0.copy();states=[np.r_[rows.ravel(),U.ravel()]];indices=[0];logs=[]
        gradient_calls=0;forward_calls=0;readout_solves=1;flow_calls=0;clock=0.;status='step_cap';start=time.perf_counter()
        for step in range(100000):
            loss,gr,gu=calc(rows,U,x,y);gradient_calls+=1
            descent=np.sum(gr*gr)+(speed*np.sum(gu*gu) if method=='joint' else 0)
            accepted=False
            for trial in range(13):
                dt=min(.01,20-clock)*2**(-trial);rr=retract(rows,gr,dt)
                if method=='reduced':uu=optimum(rr,x,y);readout_solves+=1
                elif method=='joint':uu=U-dt*speed*gu
                else:uu=U
                candidate=calc(rr,uu,x,y,False);forward_calls+=1
                if candidate<=loss-1e-4*dt*descent:
                    accepted=True;break
            if not accepted:status='line_search_failed';break
            if method=='split':
                uu=readout_flow(rr,uu,x,y,speed*dt);flow_calls+=1
                final=calc(rr,uu,x,y,False);forward_calls+=1
                assert final<=candidate+1e-10
            else:final=candidate
            logs.append([step+1,loss,final,dt,trial,descent]);clock+=dt;rows,U=rr,uu
            if (step+1)%20==0:states.append(np.r_[rows.ravel(),U.ravel()]);indices.append(step+1)
            if clock>=20-1e-12:status='clock_reached';break
        steps=len(logs)
        if indices[-1]!=steps:states.append(np.r_[rows.ravel(),U.ravel()]);indices.append(steps)
        seconds=time.perf_counter()-start
        F=product_map(rows[:4],rows[4:],keys);C=data['truth'];error=F@U-C[:,:10]
        transfer=np.linalg.lstsq(features(rows,data['xa'])[0],data['ya'],rcond=None)[0];te=F@transfer-C[:,10:]
        name=f'{seed}_{init}_{method}_{speed}.npz'
        np.savez_compressed(ROOT/name,rows=rows,U=U,initial_rows=rows0,initial_U=u0,transfer=transfer,
                            states=np.array(states),state_steps=indices,logs=np.array(logs))
        record=dict(seed=seed,init=init,method=method,speed=speed,checkpoint=name,steps=steps,status=status,seconds=seconds,
                    hidden_clock=clock,gradient_calls=gradient_calls,forward_calls=forward_calls,readout_solves=readout_solves,flow_calls=flow_calls,
                    objective=float(calc(rows,U,x,y,False)),source_mse=float(np.sum(error*(G@error))/10),
                    transfer_mse=float(np.sum(te*(G@te))/4),cosines=cosines(rows,np.r_[data['teacher_A'],data['teacher_B']]).tolist())
        records.append(record);(ROOT/'results.json').write_text(json.dumps(records,indent=2)+'\n');print(record,flush=True)
    (ROOT/'completed.json').write_text(json.dumps(dict(fits=len(records)))+'\n')

if __name__=='__main__':main()
