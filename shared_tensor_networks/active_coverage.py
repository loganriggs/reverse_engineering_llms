"""Evaluate only the active input and task prefix of independent models.

Keep the full readout parameter tensor so inactive weights remain explicitly
auditable. Splitting independent models into separate optimizers preserves Adam
updates when steps, clipping per model, and all hyperparameters are identical.
"""
import torch

def active_predictions(model,x,tasks):
    z=torch.einsum('nd,bhd->bnh',x,model.A)*torch.einsum('nd,bhd->bnh',x,model.B)
    return torch.einsum('bnp,btp->bnt',z[:,:,model.i]*z[:,:,model.j],model.W[:,:tasks])

def active_losses(model,x,y):
    return ((active_predictions(model,x,y.shape[1])-y[None])**2).mean(dim=(1,2))
