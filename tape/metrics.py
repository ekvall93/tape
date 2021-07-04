from typing import Sequence, Union
import numpy as np
import scipy.stats
import functools
from .registry import registry
from sklearn.preprocessing import normalize


@registry.register_metric('mse')
def mean_squared_error(target: Sequence[float],
                       prediction: Sequence[float]) -> float:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return np.mean(np.square(target_array - prediction_array))

@registry.register_metric('spectral_angle')
def spectral_angle(target: Sequence[float],
                       prediction: Sequence[float],
                       sequence: Sequence[int],
                       charge : Sequence[int]) -> float:


    def normalize_base_peak(array):
        # flat
        maxima = array.max(axis=1)
        array = array / maxima[:, np.newaxis]
        return array

    def mask_outofrange(array, lengths, mask=-1.0):
        # dim
        for i in range(array.shape[0]):
            array[i, lengths[i] - 1 :, :, :, :] = mask
        return array
    
    def reshape_dims(array):
        MAX_SEQUENCE = 30
        ION_TYPES = ["y", "b"]
        MAX_FRAG_CHARGE = 3

        n, dims = array.shape
        assert dims == 174
        nlosses = 1
        return array.reshape(
            [array.shape[0], MAX_SEQUENCE - 1, len(ION_TYPES), nlosses, MAX_FRAG_CHARGE]
        )

    def mask_outofcharge(array, charges, mask=-1.0):
        # dim
        for i in range(array.shape[0]):
            if charges[i] < 3:
                array[i, :, :, :, charges[i] :] = mask
        return array
    
    def reshape_flat(array):
        s = array.shape
        flat_dim = [s[0], functools.reduce(lambda x, y: x * y, s[1:], 1)]
        return array.reshape(flat_dim)

    def masked_spectral_distance(true, pred, epsilon = 1e-7):
        pred_masked = ((true + 1) * pred) / (true + 1 + epsilon)
        true_masked = ((true + 1) * true) / (true + 1 + epsilon)
        
        pred_norm = normalize(pred_masked)
        true_norm = normalize(true_masked)
        product = np.sum(pred_norm * true_norm, axis=1)
        
        
        arccos = np.arccos(product)
        spectral_distance = 2 * arccos / np.pi

        return spectral_distance

    for s in sequence:
        print(s)
    sequence_lengths = [np.count_nonzero(s) - 2 for s in sequence]
    #print(sequence_lengths)
    intensities = np.array(prediction)
    intensities_raw = np.array(target)
    charge = np.array(charge)
    charges = list(charge.argmax(axis=1) + 1)

    intensities[intensities < 0] = 0
    intensities = normalize_base_peak(intensities)
    intensities = reshape_dims(intensities)
    intensities = mask_outofrange(intensities, sequence_lengths)
    intensities = mask_outofcharge(intensities, charges)
    intensities = reshape_flat(intensities)


    spectral_angle = 1 - masked_spectral_distance(intensities_raw, intensities)
    spectral_angle = np.nan_to_num(spectral_angle)
    return np.nanmedian(spectral_angle)

@registry.register_metric('mae')
def mean_absolute_error(target: Sequence[float],
                        prediction: Sequence[float]) -> float:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return np.mean(np.abs(target_array - prediction_array))


@registry.register_metric('spearmanr')
def spearmanr(target: Sequence[float],
              prediction: Sequence[float]) -> float:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return scipy.stats.spearmanr(target_array, prediction_array).correlation


@registry.register_metric('accuracy')
def accuracy(target: Union[Sequence[int], Sequence[Sequence[int]]],
             prediction: Union[Sequence[float], Sequence[Sequence[float]]]) -> float:
    if isinstance(target[0], int):
        # non-sequence case
        return np.mean(np.asarray(target) == np.asarray(prediction).argmax(-1))
    else:
        correct = 0
        total = 0
        for label, score in zip(target, prediction):
            label_array = np.asarray(label)
            pred_array = np.asarray(score).argmax(-1)
            mask = label_array != -1
            is_correct = label_array[mask] == pred_array[mask]
            correct += is_correct.sum()
            total += is_correct.size
        return correct / total
