"""Integrable shared-linear plus projected-quadratic function; product write only."""
import numpy as np

def write(L,R,D,z):return ((z@L.T)*(z@R.T))@D.T

def action(L,R,D,z,v):return ((z@L.T)*(v@R.T)+(v@L.T)*(z@R.T))@D.T

def project(z,encoder,decoder):return (z@encoder)@decoder.T

def prediction(L,R,D,mean,encoder,decoder,z):
 centered=z-mean;pz=project(centered,encoder,decoder)
 return write(L,R,D,mean)+action(L,R,D,mean,centered)+write(L,R,D,pz)

def change(L,R,D,mean,encoder,decoder,recipient,donor):
 delta=donor-recipient;mid=(donor+recipient)/2-mean
 return action(L,R,D,mean,delta)+action(L,R,D,project(mid,encoder,decoder),project(delta,encoder,decoder))
