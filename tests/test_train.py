"""Make sure the network really learns, not merely runs without an error."""

import copy

import torch
from torch.utils.data import DataLoader, TensorDataset

from model import BikeNet
from prepare import make_loaders
from train import run_epoch, set_seed, train_many


def test_one_step_changes_every_layer(sample):
    set_seed()
    net = BikeNet(sample.input_size)
    layers = (net.layers[0], net.layers[2], net.layers[4])
    before = [layer.weight.detach().clone() for layer in layers]
    loader, _, _ = make_loaders(sample, batch_size=16)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.003)
    run_epoch(net, loader, torch.nn.MSELoss(), optimizer)
    for old, layer in zip(before, layers, strict=True):
        assert not torch.equal(old, layer.weight)