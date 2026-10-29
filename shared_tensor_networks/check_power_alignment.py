"""Exact normalized Gaussian alignment of powers of linear forms."""
import json,math
from fractions import Fraction
from pathlib import Path
import numpy as np


def double_factorial_odd(n):
    return math.prod(range(1,n+1,2))


def coefficients(p):
    # Wick pairings: k cross pairs, all remaining pairs internal.
    return {k:Fraction(math.comb(p,k)**2*math.factorial(k)*double_factorial_odd(p-k-1)**2,double_factorial_odd(2*p-1)) for k in range(p%2,p+1,2)}


def main():
    rows=[];theta=np.arange(256)*2*np.pi/256
    for p in [4,8,16,32]:
        c=coefficients(p);assert sum(c.values())==1
        for rho in [0.,.25,.5,.9,1.]:
            predicted=sum(float(a)*rho**k for k,a in c.items())
            x=np.cos(theta);y=rho*x+np.sqrt(max(0.,1-rho*rho))*np.sin(theta)
            observed=np.mean(x**p*y**p)/np.sqrt(np.mean(x**(2*p))*np.mean(y**(2*p)))
            assert abs(predicted-observed)<1e-12
            rows.append(dict(degree=p,linear_direction_cosine=rho,function_cosine=predicted,quadrature_error=abs(predicted-observed)))
    out=Path(__file__).parent/'results/depth5_controls/power_alignment.json'
    out.write_text(json.dumps(dict(status='passed',scope='rank-one powers, not a theorem for arbitrary deep networks',rows=rows),indent=2)+'\n')
    for r in rows:print(r['degree'],r['linear_direction_cosine'],r['function_cosine'])


if __name__=='__main__':main()
