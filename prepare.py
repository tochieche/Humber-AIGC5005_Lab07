"""Split rows safely and turn them into small batches of tensors.

Imagine studying for a quiz. Training rows are your practice questions,
validation rows help you pick a study method, and test rows are the final quiz.
We must not peek at the final quiz while studying!
"""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from bikes_data import TARGET, make_rentals

NUMERIC = ["hour", "temperature_c", "rain_mm", "wind_kph", "is_weekend"]
CATEGORICAL = ["station_type"]
