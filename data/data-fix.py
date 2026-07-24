
import pandas as pd
import numpy as np

data = pd.read_csv("boston_weather_data.csv")

#calculate the missing pres column using linear interpolation
data["pres"] = data["pres"].interpolate(method="linear").round(1)
data.to_csv("boston_weather_data.csv", index=False)

#delete the rows missing wind dir


#delete the first 2921 rows (8 years)
data = data.drop(data.index[0:2921])
data.to_csv("boston_weather_data.csv")


