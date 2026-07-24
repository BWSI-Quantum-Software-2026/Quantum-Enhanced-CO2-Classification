import qiskit
import qiskit_machine_learning
import pandas as pd
import torch
import numpy as np

data = pd.read_csv("boston_weather_data.csv")
X = data[["prcp", "wspd", "tavg"]].values

prcp_col = X[:, 0]
wspd_col = X[:, 1]
tavg_col = X[:, 2]
prcp_mean = np.mean(prcp_col)
wspd_mean = np.mean(wspd_col)
tavg_mean = np.mean(tavg_col)

print(prcp_mean, wspd_mean, tavg_mean)

