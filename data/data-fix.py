
import pandas as pd

data = pd.read_csv("boston_weather_data.csv")
data["pres"] = data["pres"].interpolate(method="linear").round(1)

data.to_csv("boston_weather_data.csv", index=False)
