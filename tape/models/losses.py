import torch.nn.functional as f
import torch
import numpy as np


def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

def nanmedian(v):
    is_nan = torch.isnan(v)
    filtered_v = v[~is_nan]
    return torch.median(filtered_v)

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)


def masked_spectral_distance(true, pred, epsilon = 1e-7):
    #pred_masked = ((true + 1) * pred) / (true + 1 + epsilon)
    #true_masked = ((true + 1) * true) / (true + 1 + epsilon)

    pred_masked = ((true + 1) * pred) / (torch.clamp(true + 1, min=epsilon))
    true_masked = ((true + 1) * true) / (torch.clamp(true + 1, min=epsilon))
    
    cosSim = cos(pred_masked, true_masked)
    #arccos = torch.acos(spectral_distance)
    product_clipped = torch.clamp(cosSim, min=-0.99999, max=0.99999)
    arccos = torch.acos(product_clipped)
    spectral_distance = 2 * arccos / np.pi
    return torch.mean(spectral_distance)

def masked_spectral_distance_old(true, pred, epsilon = 1e-7):
    pred_masked = ((true + 1) * pred) / (true + 1 + epsilon)
    true_masked = ((true + 1) * true) / (true + 1 + epsilon)
    
    pred_norm = f.normalize(pred_masked,dim=-1,p=2)
    true_norm = f.normalize(true_masked,dim=-1,p=2)
    product = torch.sum(pred_norm * true_norm, axis=1)
    
    product_clipped = torch.clamp(product, min=-1, max=1)
    arccos = torch.acos(product_clipped)
    spectral_distance = 2 * arccos / np.pi

    return nanmedian(spectral_distance)

