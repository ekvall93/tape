import torch.nn.functional as f
import torch
import numpy as np


def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

def masked_spectral_distance(true, pred, epsilon = 1e-12):
    pred_masked = ((true + 1) * pred) / (true + 1 + epsilon)
    true_masked = ((true + 1) * true) / (true + 1 + epsilon)
    
    pred_norm = f.normalize(pred_masked,dim=-1,p=2)
    true_norm = f.normalize(true_masked,dim=-1,p=2)
    product = torch.sum(pred_norm * true_norm, axis=1)
    
    arccos = torch.acos(product)
    spectral_distance = 2 * arccos / np.pi

    return nanmean(spectral_distance)
