import qiskit
import qiskit_machine_learning
import pandas as pd
import torch
import numpy as np

data = pd.read_csv("boston_weather_data.csv")
X = data[["tavg", "wdir", "wspd"]].values

tavg_col = X[:, 0]
wdir_col = X[:, 1]
wspd_col = X[:, 2]

tavg_mean = np.mean(tavg_col)
wdir_mean = np.mean(wdir_col)
wspd_mean = np.mean(wspd_col)

print("MEANS", tavg_mean, wdir_mean, wspd_mean)

tavg_stdev = np.std(tavg_col)
wdir_stdev = np.std(wdir_col)
wspd_stdev = np.std(wspd_col)

print("STANDARD DEVIATIONS", tavg_stdev, wdir_stdev, wspd_stdev)

tavg_zscr = (X[:, 0] - tavg_mean) / tavg_stdev
wdir_zscr = (X[:, 1] - wdir_mean) / wdir_stdev
wspd_zscr = (X[:, 2] - wspd_mean) / wspd_stdev
Z = np.column_stack([tavg_zscr, wdir_zscr, wspd_zscr])

print("Z ARRAY SIZE", Z.shape)
print("Z ARRAY SAMPLE", Z[:5])

max_abs = np.max(np.abs(Z), axis = 0)

print("HIGHEST Z SCORE", max_abs)

alphas = np.pi / max_abs

print("SCALING FACTORS", alphas)

rotations = Z * alphas

print("ROTATIONS ARRAY SIZE", rotations.shape)
print("ROTATIONS ARRAY SAMPLE", rotations[:5])