#!/usr/bin/env python3
"""Fill the missing class-count by neuron-width scaling conditions."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def main():
    root=Path("artifacts/clean_axis_scaling")
    runs=[]
    for classes in (10,30,100):
        for neurons in (64,128,256):
            slug=f"c{classes}_o4_d2_w{neurons}"
            for seed in range(8):
                destination=root/slug/f"seed_{seed}"
                runs.append({"classes":classes,"operations":4,"depth":2,"neurons":neurons,"seed":seed,"slug":slug})
                if (destination/"results.json").exists():
                    print(f"SKIP {slug} seed {seed}",flush=True); continue
                command=[sys.executable,"scripts/run_multiclass_subspace_pilot.py","--input-mode","prototypes","--classes",str(classes),"--content-dim","128","--operations","4","--code-dim","16","--neurons",str(neurons),"--depth","2","--train-samples-per-cell","32","--confirmation-samples-per-cell","8","--max-steps","5000","--rotations","5","--seed",str(seed),"--device","cuda","--output",str(destination)]
                print(f"START {slug} seed {seed}",flush=True); subprocess.run(command,check=True,stdout=subprocess.DEVNULL); print(f"DONE  {slug} seed {seed}",flush=True)
    (root/"load_factorial_manifest.json").write_text(json.dumps({"classes":[10,30,100],"neurons":[64,128,256],"runs":runs},indent=2),encoding="utf-8")


if __name__=="__main__": main()
