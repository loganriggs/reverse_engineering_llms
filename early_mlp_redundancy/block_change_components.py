"""Decode disjoint block/shuffled-group changes exactly once per atom."""
import numpy as np
import torch


def grouping(seed, shuffled=False):
    atoms=np.arange(128) if not shuffled else np.random.default_rng(10500+seed).permutation(128)
    return atoms.reshape(32,4)


def changes(code_recipient,code_donor,dictionary,scale,recipient_positions,donor_positions,groups):
    """Return [32,batch,positions,output_dim]; means and decoder bias cancel."""
    difference=code_donor[:,donor_positions]-code_recipient[:,recipient_positions]
    result=[]
    for atoms in groups:
        atoms=torch.as_tensor(atoms,device=difference.device)
        result.append((difference[:,:,atoms]@dictionary[:,atoms].T)*scale)
    return torch.stack(result)
