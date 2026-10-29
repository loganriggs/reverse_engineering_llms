"""M101 consistent quadratic candidate retaining selected/complement mixed interactions."""
from native_projected_quadratic import write,action,project

def prediction(L,R,D,mean,encoder,decoder,z):
 u=z-mean;p=project(u,encoder,decoder);h=u-p
 return write(L,R,D,mean)+action(L,R,D,mean,u)+write(L,R,D,p)+action(L,R,D,p,h)

def change(L,R,D,mean,encoder,decoder,recipient,donor):
 delta=donor-recipient;mid=(donor+recipient)/2-mean
 hmid=mid-project(mid,encoder,decoder);hdelta=delta-project(delta,encoder,decoder)
 return action(L,R,D,(donor+recipient)/2,delta)-action(L,R,D,hmid,hdelta)
