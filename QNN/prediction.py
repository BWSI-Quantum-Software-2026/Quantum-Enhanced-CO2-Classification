import numpy as np
from datetime import timedelta
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from data.standardization import load_and_prepare
from qnn_temperature import qnn_temperature
from qnn_humidity import qnn_humidity
from qnn_wind_speed import qnn_wind_speed

w_temp, forward = qnn_temperature()
w_hum = qnn_humidity()
w_wspd = qnn_wind_speed()
X, Ytemp, Yhum, Ywspd, tavgy_mean, tavgy_stdev, humy_mean, humy_stdev, wspdy_mean, wspdy_stdev, alphasx, alphasytemp, alphasyhum, alphasywspd, dates = load_and_prepare()

def qnn_to_celsius(pred_scaled, meanY, stdevY, alphaY): 
    pred_z = pred_scaled / alphaY # reversing scaling
    pred_celsius = pred_z * stdevY + meanY # reversing z-score
    return pred_celsius 

def qnn_to_humidity(pred_scaled, meanY, stdevY, alphaY): 
    pred_z = pred_scaled / alphaY # reversing scaling
    pred_humidity = pred_z * stdevY + meanY # reversing z-score
    return pred_humidity

def qnn_to_wind_speed(pred_scaled, meanY, stdevY, alphaY): 
    pred_z = pred_scaled / alphaY # reversing scaling
    pred_wind_speed = pred_z * stdevY + meanY # reversing z-score
    return pred_wind_speed

def forecast_multivariate(num_days, w_temp, w_hum, w_wspd, lag_window = 4):
    future_temp = []
    future_hum  = []
    future_wspd = []

    raw_window = X[-5:, :3].copy()

    for step in range(num_days):

        current_X_raw = raw_window[::-1].flatten()
        current_X_z = (current_X_raw - X.mean(axis = 0)[:15]) / X.std(axis = 0)[:15]
        current_X_scaled = current_X_z * alphasx[:15]

        temp_next_scaled = forward(current_X_scaled, w_temp)
        hum_next_scaled = forward(current_X_scaled, w_hum)
        wspd_next_scaled = forward(current_X_scaled, w_wspd)

        future_temp.append(temp_next_scaled)
        future_hum.append(hum_next_scaled)
        future_wspd.append(wspd_next_scaled)

        temp_next = qnn_to_celsius(temp_next_scaled, tavgy_mean, tavgy_stdev, alphasytemp)
        hum_next = qnn_to_humidity(hum_next_scaled, humy_mean, humy_stdev, alphasyhum)
        wspd_next = qnn_to_wind_speed(wspd_next_scaled, wspdy_mean, wspdy_stdev, alphasywspd)

        next_row = np.array([temp_next, hum_next, wspd_next])

        raw_window = np.vstack([raw_window[1:], next_row])

    return future_temp, future_hum, future_wspd

prediction_days = 7

future_temp_scaled, future_hum_scaled, future_wspd_scaled = forecast_multivariate(prediction_days, w_temp, w_hum, w_wspd, lag_window = 4)

future_temp_celsius = [qnn_to_celsius(p, tavgy_mean, tavgy_stdev, alphasytemp) for p in future_temp_scaled]
future_hum_percent = [qnn_to_humidity(p, humy_mean, humy_stdev, alphasyhum) for p in future_hum_scaled]
future_wspd_real = [qnn_to_wind_speed(p, wspdy_mean, wspdy_stdev, alphasywspd) for p in future_wspd_scaled]

last_date = dates.iloc[-1]
future_dates = [last_date + timedelta(days = i + 1) for i in range(prediction_days)]

print(f"{prediction_days}-day forecast:")
for date, t, h, w in zip(future_dates, future_temp_celsius, future_hum_percent, future_wspd_real):
    print(f"{date.date()}: Temp = {t:.2f} Celsius, Hum = {h:.2f} %, Wind = {w:.2f} m/s")
