import torch.nn.functional as f
import torch
import numpy as np


cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

def masked_spectral_distance(true, pred, epsilon = torch.finfo(torch.float16).eps):
    pred_masked = ((true + 1) * pred) / (true + 1 + epsilon)
    true_masked = ((true + 1) * true) / (true + 1 + epsilon)
    
    cosSim = cos(pred_masked, true_masked)
    product_clipped = torch.clamp(cosSim, -(1-epsilon), (1 - epsilon))
    arccos = torch.acos(product_clipped)
    spectral_distance = 2 * arccos / np.pi
    return torch.mean(spectral_distance)
