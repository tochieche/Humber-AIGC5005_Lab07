"""Run the whole project with one command: python train.py.

The network guesses a bike count, checks how wrong it was, and gently changes its numbers. It does this many times.
"""

import copy
import csv
import json
import random
from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from bikes_data import TARGET, make_rentals
from model import BikeNet
from prepare import make_loaders, prepare_data

matplotlib.use("Agg")  # Save graphs on computers without a screen.
REPORTS = Path(__file__).resolve().parent / "reports"
SEED = 42
EPOCHS = 90

def set_seed(seed=SEED):
    """Make the random choices repeatable so experiments are fair."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)  # Reproducible and fast for this small CPU lab.
    torch.use_deterministic_algorithms(True)
    
    
def run_epoch(model, loader, loss_fn, optimizer=None):
    """Train one pass if optimizer is given; otherwise only check accuracy.

    MAE means 'mean absolute error': if MAE = 5, guesses miss by about 5 bikes.
    The MSE loss punishes large mistakes more than tiny ones.
    """
    training = optimizer is not None
    model.train(training)
    total_squared_error = 0.0
    total_absolute_error = 0.0
    count = 0
    for features, actual in loader:  # Take a small batch of examples.
        with torch.set_grad_enabled(training):
            predicted = model(features)  # Forward the network guesses.
            if predicted.shape != actual.shape:
                raise ValueError("Targets and guesses must have the same shape")
            loss = loss_fn(predicted, actual)  # Find how wrong it is.
            if training:
                optimizer.zero_grad()  # Clear yesterday's gradient notes.
                loss.backward()  # Backward: find which weights to change.
                optimizer.step()  # Actually change the weights.
        # Keep totals weighted by batch size, including the last shorter batch.
        total_squared_error += torch.sum((predicted - actual) ** 2).item()
        total_absolute_error += torch.sum(torch.abs(predicted - actual)).item()
        count += actual.numel()
    return {"loss": total_squared_error / count, "mae": total_absolute_error / count}

def train_many(model, train_loader, val_loader, epochs=EPOCHS, lr=0.003):
    """Practice for epochs rounds; restore the best validation weights."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_weights = None
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, loss_fn, optimizer)
        val_metrics = run_epoch(model, val_loader, loss_fn)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
        }
        history.append(row)
        if row["val_loss"] < best_loss:
            best_loss = row["val_loss"]
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_weights)  # The best quiz score wins.
    return history, best_epoch


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
