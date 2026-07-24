import qiskit
import qiskit_machine_learning
import pandas as pd
import torch

data = pd.read_csv("boston_weather_data.csv")
X = data[["tavg", "prcp", "pres"]].values
