from typing import Sequence, Union
import numpy as np
import scipy.stats

from .registry import registry


@registry.register_metric('mse')
def mean_squared_error(target: Sequence[float],
                       prediction: Sequence[float]) -> float:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return np.mean(np.square(target_array - prediction_array))

@registry.register_metric('spectral_angle')
def spectral_angle(target: Sequence[float],
                       prediction: Sequence[float]) -> float:


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


    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return np.mean(np.square(target_array - prediction_array))

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
