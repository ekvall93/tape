import torch.nn.functional as f
import torch
import numpy as np

def masked_spectral_distance(true, pred, epsilon = 1e-12):
    pred_masked = ((true + 1) * pred) / (true + 1 + epsilon)
    true_masked = ((true + 1) * true) / (true + 1 + epsilon)
    
    pred_norm = f.normalize(pred_masked,dim=-1,p=2)
    true_norm = f.normalize(true_masked,dim=-1,p=2)
    product = torch.sum(pred_norm * true_norm, axis=1)
    
    arccos = torch.acos(product)
    spectral_distance = 2 * arccos / np.pi

    return spectral_distance.mean()