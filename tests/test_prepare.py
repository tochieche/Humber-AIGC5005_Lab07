
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

def test_split_sizes_and_no_overlap():
    train, val, test = make_splits(make_rentals(), seed=42)
    assert (len(train), len(val), len(test)) == (3600, 1200, 1200)
    assert len(set(train.index) | set(val.index) | set(test.index)) == 6000
    assert set(train.index).isdisjoint(val.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(val.index).isdisjoint(test.index)