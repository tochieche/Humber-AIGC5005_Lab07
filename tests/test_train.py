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


def test_network_can_memorize_one_small_batch():
    set_seed()
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32)
    y = 2 * x + 1
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    net = BikeNet(1, hidden_size=16)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.03)
    start = run_epoch(net, loader, loss_fn)["loss"]
    for _ in range(350):
        run_epoch(net, loader, loss_fn, optimizer)
    end = run_epoch(net, loader, loss_fn)["loss"]
    assert end < start * 0.01