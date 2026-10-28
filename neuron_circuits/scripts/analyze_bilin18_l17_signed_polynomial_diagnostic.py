from pathlib import Path
import json,hashlib,numpy as np
root=Path('/home/loganriggs/Coding/neuron_circuits');art=root/'artifacts/bilin18_tensor_decomposition';out=Path('/home/loganriggs/Documents/Codex/2026-10-21/he/outputs')
src=art/'bilin18_l17_normalization_curvature_factorial_bilinear_polynomial_coefficients.npz'
c=np.load(src)['coefficients'].reshape(3,25,4,1024,4).mean(-1)
# Pool errors and effects before forming ratios. N/E is the combination score.
def pooled(docs,draws):
 return np.stack([c[:,1+draws][:,:,:,docs].mean((1,3)),c[:,13+draws][:,:,:,docs].mean((1,3))])
def nd(v):return v[:,:,1]+v[:,:,2]-v[:,:,3]-v[:,:,0],v[:,:,0]
def scores(v,a,keep=(1,1,1)):
 n,d=nd(v);p=np.array([1,a,a*a])*keep;return (n@p)/(d@p)
def contrast(v):
 # Gaussian-minus-raw, large-minus-small; and its signed odd/even parts.
 def gap(a):return np.diff(scores(v,a))[0]
 pos=gap(4)-gap(.25);neg=gap(-4)-gap(-.25)
 return np.array([pos,neg,(pos-neg)/2,(pos+neg)/2])
v=pooled(np.arange(1024),np.arange(12));point=contrast(v)
rng=np.random.default_rng(20270906);boot=np.stack([contrast(pooled(rng.integers(0,1024,1024),rng.integers(0,12,12))) for _ in range(2000)])
rad=np.quantile(abs(boot-point).max(1),.95);ci=np.stack([point-rad,point+rad],1)
amps=[-4,-2,-1,-.25,.25,1,2,4];pred=[scores(v,a).tolist() for a in amps]
# Symmetric two-path accounting separates numerator and denominator changes across amplitude.
n,d=nd(v);p0=np.array([1,.25,.25**2]);p1=np.array([1,4,16]);n0=n@p0;n1=n@p1;d0=d@p0;d1=d@p1
num=.5*(n1-n0)*(1/d0+1/d1);den=.5*(n0+n1)*(1/d1-1/d0)
closure=float(abs(num+den-(scores(v,4)-scores(v,.25))).max())
# Exact odd/even rational identity; not independent forward validation.
odd_err=0.
for a in [.25,1,2,4]:
 ne=n[:,0]+a*a*n[:,2];de=d[:,0]+a*a*d[:,2]
 odd=a*(n[:,1]*de-ne*d[:,1])/(de*de-a*a*d[:,1]**2)
 odd_err=max(odd_err,float(abs(odd-(scores(v,a)-scores(v,-a))/2).max()))
j={'experiment':'H427','scope':'Post-result diagnostic on reused H425/H426 coefficients. Predictions, not new forward evaluations or fresh confirmation. Fixed Fisher metric and linearized input RMS only.','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'families':['raw','gaussian'],'amplitudes':amps,'predicted_combination_scores':pred,'contrast_names':['positive_amplitude_interaction','negative_amplitude_interaction','odd_part','even_part'],'contrasts':point.tolist(),'exploratory_joint4_ci95':ci.tolist(),'bootstrap_samples':2000,'bootstrap_seed':20270906,'numerator_change_by_family':num.tolist(),'denominator_change_by_family':den.tolist(),'gaussian_minus_raw_numerator_change':float(np.diff(num)[0]),'gaussian_minus_raw_denominator_change':float(np.diff(den)[0]),'closure_error':closure,'odd_identity_error':odd_err,'new_documents':0,'completed_unique_documents':10944}
assert closure<1e-12 and odd_err<1e-12
(art/'bilin18_l17_signed_polynomial_diagnostic.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(art/'bilin18_l17_signed_polynomial_diagnostic_bootstrap.npz',contrasts=boot)
lines=['## H427 — Signed polynomial predictions and numerator/denominator accounting','',j['scope'],'','Combination score S = R_union - R_core - R_added. With pooled full energy D and error combination N, S(a)=(N2+a*N3+a^2*N4)/(D2+a*D3+a^2*D4). The common a^2 cancels. Cubic terms determine sign asymmetry, but both numerator and denominator matter.','', '| Edit amplitude | Predicted raw S | Predicted Gaussian S |','|---|---:|---:|']
for a,vals in zip(amps,pred):lines.append(f'|{a:g}|{vals[0]:.6f}|{vals[1]:.6f}|')
lines+=['','| Diagnostic contrast | Estimate | Exploratory joint 95% interval |','|---|---:|---|']
for name,val,b in zip(j['contrast_names'],point,ci):lines.append(f'|{name}|{val:.6f}|[{b[0]:.6f}, {b[1]:.6f}]|')
lines += ['',f'Symmetric numerator/denominator accounting of the Gaussian-minus-raw positive amplitude interaction: numerator {np.diff(num)[0]:+.6f}; denominator {np.diff(den)[0]:+.6f}. This averages the two orders of changing numerator and denominator. It is descriptive ratio accounting, not unique causal mediation.',f'Accounting closure error {closure:.2g}; odd rational identity error {odd_err:.2g}. Neither is an independent model evaluation. Bootstrap resamples recipient documents and paired source draws conditional on the fixed donor pool; bands are exploratory and do not replace H426 gates.','', 'Next: evaluate new signed/intermediate amplitudes directly in the controlled model cell, first checking polynomial implementation, then confirm any selected scientific contrast on independent recipients/draws. Exact input normalization and exact KL require separate tests. Total distinct documents remains 10944. Broad goal open.']
report='\n'.join(lines)+'\n';(out/'signed_polynomial_diagnostic.md').write_text(report);(art/'BILIN18_L17_SIGNED_POLYNOMIAL_DIAGNOSTIC.md').write_text(report)
print(json.dumps(j,indent=2))
