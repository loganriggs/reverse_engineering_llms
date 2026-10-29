import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from readout_speed import calc,features,optimum,retract,readout_flow
from multitask import monomials,product_map
from experiment import gram

ROOT=Path('results/readout_speed');SOURCE=Path('results/normalized_readout/panel')
METHODS=[('joint',s) for s in [1,10,100]]+[('reduced',0)]+[('split',s) for s in [1,10,100]]

def cosines(rows,teacher):
    def matrices(r):return (r[:4,:,None]*r[4:,None,:]+r[4:,:,None]*r[:4,None,:])/2
    def inner(a,b):return 2*np.einsum('iab,jab->ij',a,b)+np.outer(np.trace(a,axis1=1,axis2=2),np.trace(b,axis1=1,axis2=2))
    a,b=matrices(rows),matrices(teacher);C=abs(inner(a,b))/np.sqrt(np.outer(np.diag(inner(a,a)),np.diag(inner(b,b))))
    i,j=linear_sum_assignment(-C);return C[i,j]

def main():
    ROOT.mkdir(exist_ok=False);records=[];keys=monomials(8);G=gram(keys)
    for seed in range(232101,232106):
      data=np.load(SOURCE/f'{seed}_data.npz');x,y=data['x'],data['y']
      for init in [233001,233002]:
       initial=np.load(SOURCE/f'{seed}_{init}_joint.npz')['initial'];rows0=initial[:64].reshape(8,8);u0=optimum(rows0,x,y)
       for method,speed in METHODS:
        rows=rows0.copy();U=u0.copy();states=[np.r_[rows.ravel(),U.ravel()]];indices=[0];logs=[]
        gradient_calls=0;forward_calls=0;readout_solves=1;flow_calls=0;clock=0.;status='cap';start=time.perf_counter()
        for step in range(2000):
            loss,gr,gu=calc(rows,U,x,y);gradient_calls+=1
            descent=np.sum(gr*gr)+(speed*np.sum(gu*gu) if method=='joint' else 0)
            accepted=False
            for trial in range(13):
                dt=.01*2**(-trial);rr=retract(rows,gr,dt)
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
