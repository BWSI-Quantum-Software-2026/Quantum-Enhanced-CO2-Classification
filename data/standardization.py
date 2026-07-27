import pandas as pd
import numpy as np

data = pd.read_csv("boston_weather_data_2.csv")
X = data[["tavg", "hum", "wspd"]].values

tavg_col = X[:, 0]
hum_col = X[:, 1]
wspd_col = X[:, 2]

tavg_mean = np.mean(tavg_col)
hum_mean = np.mean(hum_col)
wspd_mean = np.mean(wspd_col)

print("MEANS", tavg_mean, hum_mean, wspd_mean)

tavg_stdev = np.std(tavg_col)
hum_stdev = np.std(hum_col)
wspd_stdev = np.std(wspd_col)

print("STANDARD DEVIATIONS", tavg_stdev, hum_stdev, wspd_stdev)

tavg_zscr = (X[:, 0] - tavg_mean) / tavg_stdev
hum_zscr = (X[:, 1] - hum_mean) / hum_stdev
wspd_zscr = (X[:, 2] - wspd_mean) / wspd_stdev
Z = np.column_stack([tavg_zscr, hum_zscr, wspd_zscr])

print("Z ARRAY SIZE", Z.shape)
print("Z ARRAY SAMPLE", Z[:5])

max_abs = np.max(np.abs(Z), axis = 0)

print("HIGHEST Z SCORE", max_abs)

alphas = np.pi / max_abs

print("SCALING FACTORS", alphas)

rotations = Z * alphas

print("ROTATIONS ARRAY SIZE", rotations.shape)
print("ROTATIONS ARRAY SAMPLE", rotations[:5])