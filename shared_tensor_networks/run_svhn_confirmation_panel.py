"""Sequential, fail-visible five-seed GPU data/capacity confirmation and audits."""
import os
import subprocess
import sys
from pathlib import Path
from run_svhn_standard import write,digest

root=Path(__file__).parent
out=root/'results/svhn_confirmation'
out.mkdir(exist_ok=True)
completed=[]
for seed in [212001,212002,212003,212004,212005]:
    for regime in ['standard','extra']:
        for architecture in ['stack','tree','wide_stack']:
            name=f'seed{seed}_{regime}_{architecture}'
            args=['--depth','3','--seed',str(seed),'--regime',regime,'--architecture',architecture]
            for script,phase in [('run_svhn_confirmation.py','training'),('audit_svhn_confirmation.py','audit')]:
                print(name,phase,flush=True)
                with (out/f'{name}_{phase}.log').open('a') as log:
                    child=subprocess.Popen([sys.executable,str(root/script),*args],cwd=root,stdout=log,stderr=subprocess.STDOUT)
                    write(out/'panel_status.json',dict(status='running',pid=os.getpid(),child_pid=child.pid,
                        configuration=name,phase=phase,completed=completed,script_sha256=digest(root/script)))
                    code=child.wait()
                if code:
                    write(out/'panel_status.json',dict(status='failed',pid=os.getpid(),configuration=name,
                        phase=phase,exit_code=code,completed=completed,log=f'{name}_{phase}.log'))
                    raise SystemExit(code if code>0 else 1)
            completed.append(name)
            print('Verified',name,flush=True)
write(out/'panel_status.json',dict(status='complete',configurations=len(completed),completed=completed))
