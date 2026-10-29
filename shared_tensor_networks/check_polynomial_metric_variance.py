"""Exact sampling variance of diagonal feature-metric estimates for Gaussian monomials."""
from fractions import Fraction
from math import prod,sqrt
from pathlib import Path
import numpy as np
from run_svhn_standard import digest,write

def moment(n):return prod(range(1,n,2)) if n%2==0 else 0
cases={'bilinear_independent':[1,1],'quartic_distinct':[1,1,1,1],'squared_bilinear':[2,2],'quadratic_repeated':[2],'quartic_repeated':[4]}
z,w=np.polynomial.hermite_e.hermegauss(9);w=w/np.sqrt(2*np.pi);rows=[]
for name,powers in cases.items():
 second=prod(moment(2*a) for a in powers);fourth=prod(moment(4*a) for a in powers);relative=Fraction(fourth,second**2)-1
 numeric2=prod(float(w@(z**(2*a))) for a in powers);numeric4=prod(float(w@(z**(4*a))) for a in powers)
 np.testing.assert_allclose([numeric2,numeric4],[second,fourth],atol=1e-8,rtol=1e-12)
 rows.append(dict(name=name,powers=powers,second_moment=second,fourth_moment=fourth,relative_variance_times_sample_count=str(relative),relative_std_second_moment_512=sqrt(float(relative)/512),relative_std_second_moment_4096=sqrt(float(relative)/4096)))
root=Path(__file__).parent;write(root/'results/empirical_extraction/metric_variance_checks.json',dict(status='passed',rows=rows,source_sha256=digest(Path(__file__))))
for r in rows:print(r)
