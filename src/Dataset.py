import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

# Which columns from the CSV are used as model features (in this order).
# Edit this list if you want to add/drop features - everything downstream
# (input size, output size, un-normalizing) adapts automatically.
FEATURE_COLUMNS = ["hum", "uvi", "tavg", "wspd", "wdir", "pres", "prcp"]


class WeatherDataset(Dataset):
    """
    Loads a daily weather CSV with columns:
        YEAR, MO, DY, hum, uvi, tavg, wspd, wdir, pres, prcp
    and turns it into overlapping windows of length `day_range` for
    sequence-to-sequence / RNN style training.

    Each item returned by __getitem__ is:
        day   - int, the day-of-month of the first day in the window
        month - int, the month of the first day in the window
        data_seq - FloatTensor [day_range, n_features], normalized

    Normalization (mean/std) is always computed from the TRAIN split
    (rows before split_date) so that train and test are scaled the same
    way and the test set never leaks into the statistics.
    """

    def __init__(self, csv_file, day_range=15, split_date="2025-01-01", train_test="train"):
        # ---- load + build a proper datetime index ----
        raw = pd.read_csv(csv_file)
        raw["date"] = pd.to_datetime(dict(year=raw["YEAR"], month=raw["MO"], day=raw["DY"]))
        raw = raw.sort_values("date").set_index("date")

        split_date = pd.to_datetime(split_date)

        # Features only, in a fixed column order
        features = raw[FEATURE_COLUMNS].astype(np.float32)

        # ---- normalization stats always come from the train portion ----
        train_features = features[features.index < split_date]
        self.mean = torch.tensor(train_features.mean().values, dtype=torch.float32)
        self.std = torch.tensor(train_features.std().values, dtype=torch.float32)

        normed = (features - train_features.mean()) / train_features.std()

        # ---- split ----
        if train_test == "train":
            self.dataset = normed[normed.index < split_date]
        elif train_test == "test":
            self.dataset = normed[normed.index >= split_date]
        else:
            raise ValueError("train_test must be 'train' or 'test'")

        self.day_range = day_range
        self.values = torch.tensor(self.dataset.values, dtype=torch.float32)
        self.dates = self.dataset.index

    def __len__(self):
        # number of complete windows of length day_range we can form
        return max(0, len(self.dataset) - self.day_range + 1)

    def __getitem__(self, idx):
        window = self.values[idx: idx + self.day_range]
        first_date = self.dates[idx]
        return first_date.day, first_date.month, window
