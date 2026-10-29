"""Walsh algebra on three binary input factors; coefficients use uniform cube measure."""
import numpy as np

BITS=np.array([[i>>j&1 for j in range(3)] for i in range(8)])
SIGNS=2*BITS-1
H=np.array([[np.prod(SIGNS[i,[j for j in range(3) if mask>>j&1]]) for mask in range(8)] for i in range(8)])


def coefficients(values):return np.tensordot(H.T,values,axes=(1,0))/8

def reconstruct(c):return np.tensordot(H,c,axes=(1,0))

def order_energies(c):
    energy=np.square(c).reshape(8,-1).sum(1)
    return [float(sum(energy[m] for m in range(8) if m.bit_count()==order)) for order in range(4)]

def bilinear_coefficients(L,R,D,zcoeff):
    left=zcoeff@L.T;right=zcoeff@R.T
    out=np.zeros((8,D.shape[0]))
    for mask in range(8):
        out[mask]=sum((left[a]*right[a^mask])@D.T for a in range(8))
    carry=((left[0]*right[7]+left[7]*right[0])@D.T)
    mixed=sum((left[a]*right[a^7])@D.T for a in range(1,7))
    return out,carry,mixed
