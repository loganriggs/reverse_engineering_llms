"""M77 native bilinear MLP input-to-write directional operators."""
import torch


def write(L,R,D,z):
    return ((z@L.T)*(z@R.T))@D.T


def jvp(L,R,D,anchor,direction):
    return ((anchor@L.T)*(direction@R.T)+(direction@L.T)*(anchor@R.T))@D.T


def predictions(L,R,D,mean,recipient,donor):
    delta=donor-recipient
    return dict(shared_J_at_training_mean=jvp(L,R,D,mean,delta),
                recipient_J=jvp(L,R,D,recipient,delta),
                midpoint_J_exact_control=jvp(L,R,D,(recipient+donor)/2,delta))
