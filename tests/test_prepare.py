
import numpy as np
import torch

from bikes_data import make_rentals
from prepare import make_loaders, make_splits


def test_dataset_shape_dtype_and_length(sample):
    features, answer = sample.train_dataset[0]
    assert len(sample.train_dataset) == 180
    assert features.shape == (sample.input_size,)
    assert answer.shape == (1,)
    assert features.dtype == torch.float32
    assert answer.dtype == torch.float32