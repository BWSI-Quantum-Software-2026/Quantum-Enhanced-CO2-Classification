import pandas as pd
import numpy as np

def create_lag_features(df, features, lag_window):
    df = df.copy()
    for feature in features:
        for lag in range(1, lag_window + 1):
            df[f"{feature}_lag{lag}"] = df[feature].shift(lag)
    return df

def load_and_prepare():
    data = pd.read_csv("data/boston_weather_data_2.csv")

    day_col = data["DY"]
    month_col = data["MO"]
    year_col = data["YEAR"]
    dates = pd.to_datetime(dict(year = year_col, month = month_col, day = day_col))

    print("DATES SAMPLE", dates[:5])

    base_features = ["tavg", "hum", "wspd"]
    data = create_lag_features(data, base_features, lag_window = 4)

    lagged_cols = (
        base_features +
        [f"{f}_lag{l}" for f in base_features for l in range(1, 5)]
    )

    Ytavg = data["tavg"].shift(-1).dropna().values
    Yhum = data["hum"].shift(-1).dropna().values
    Ywspd = data["wspd"].shift(-1).dropna().values
    data = data.dropna()

    X = data[lagged_cols].values

    Ytavg = Ytavg[-len(X):]
    Yhum = Yhum[-len(X):]
    Ywspd = Ywspd[-len(X):]

    tavgy_col = Ytavg
    humy_col = Yhum
    wspdy_col = Ywspd

    X_mean = np.mean(X, axis=0) 
    tavgy_mean = np.mean(tavgy_col)
    humy_mean = np.mean(humy_col)
    wspdy_mean = np.mean(wspdy_col)

    print("MEANS", X_mean, tavgy_mean, humy_mean, wspdy_mean)

    X_stdev = np.std(X, axis=0)
    tavgy_stdev = np.std(tavgy_col)
    humy_stdev = np.std(humy_col)
    wspdy_stdev = np.std(wspdy_col)


    print("STANDARD DEVIATIONS", X_stdev, tavgy_stdev, humy_stdev, wspdy_stdev)

    X_zscr = (X - X_mean) / X_stdev

    tavgy_zscr = (Ytavg - tavgy_mean) / tavgy_stdev
    ZYtemp = tavgy_zscr
    humy_zscr = (Yhum - humy_mean) / humy_stdev
    ZYhum = humy_zscr
    wspdy_zscr = (Ywspd - wspdy_mean) / wspdy_stdev
    ZYwspd = wspdy_zscr

    print("Z INPUT ARRAY SIZE", X_zscr.shape)
    print("Z INPUT ARRAY SAMPLE", X_zscr[:5])

    print("Z EXPECTED TEMP ARRAY SIZE", ZYtemp.shape)
    print("Z EXPECTED TEMP ARRAY SAMPLE", ZYtemp[:5])
    print("Z EXPECTED HUM ARRAY SIZE", ZYhum.shape)
    print("Z EXPECTED HUM ARRAY SAMPLE", ZYhum[:5])
    print("Z EXPECTED WSPD ARRAY SIZE", ZYwspd.shape)
    print("Z EXPECTED WSPD ARRAY SAMPLE", ZYwspd[:5])

    max_absx = np.max(np.abs(X_zscr), axis = 0)
    max_absytemp = np.max(np.abs(ZYtemp), axis = 0)
    max_absyhum = np.max(np.abs(ZYhum), axis = 0)
    max_absywspd = np.max(np.abs(ZYwspd), axis = 0)

    print("HIGHEST Z INPUT SCORE", max_absx)
    print("HIGHEST Z EXPECTED TEMP SCORE", max_absytemp)
    print("HIGHEST Z EXPECTED HUM SCORE", max_absyhum)
    print("HIGHEST Z EXPECTED WSPD SCORE", max_absywspd)

    alphasx = np.pi / max_absx
    alphasytemp = np.pi / max_absytemp
    alphasyhum = np.pi / max_absyhum
    alphasywspd = np.pi / max_absywspd

    print("SCALING FACTORS INPUT", alphasx)
    print("SCALING FACTORS EXPECTED TEMP", alphasytemp)
    print("SCALING FACTORS EXPECTED HUM", alphasyhum)
    print("SCALING FACTORS EXPECTED WSPD", alphasywspd)

    rotationsx = X_zscr * alphasx
    rotationsytemp = ZYtemp * alphasytemp
    rotationsyhum = ZYhum * alphasyhum
    rotationsywspd = ZYwspd * alphasywspd

    print("ROTATIONS INPUT ARRAY SIZE", rotationsx.shape)
    print("ROTATIONS INPUT ARRAY SAMPLE", rotationsx[:5])
    print("ROTATIONS EXPECTED TEMP ARRAY SIZE", rotationsytemp.shape)
    print("ROTATIONS EXPECTED TEMP ARRAY SAMPLE", rotationsytemp[:5])
    print("ROTATIONS EXPECTED HUM ARRAY SIZE", rotationsyhum.shape)
    print("ROTATIONS EXPECTED HUM ARRAY SAMPLE", rotationsyhum[:5])
    print("ROTATIONS EXPECTED WSPD ARRAY SIZE", rotationsywspd.shape)
    print("ROTATIONS EXPECTED WSPD ARRAY SAMPLE", rotationsywspd[:5])

    print("X/Y Temp aligned:", len(rotationsx) == len(rotationsytemp))
    print("X/Y Hum aligned:", len(rotationsx) == len(rotationsyhum))
    print("X/Y Wspd aligned:", len(rotationsx) == len(rotationsywspd))
    
    print("Rotations shape:", rotationsx.shape)
    print("Y Temp shape:", rotationsytemp.shape)
    print("Y Hum shape:", rotationsyhum.shape)
    print("Y Wspd shape:", rotationsywspd.shape)

    print("Sample rotations:", rotationsx[:3])
    print("Sample Y Temp:", rotationsytemp[:3])
    print("Sample Y Hum:", rotationsyhum[:3])
    print("Sample Y Wspd:", rotationsywspd[:3])

    return rotationsx, rotationsytemp, rotationsyhum, rotationsywspd, tavgy_mean, tavgy_stdev, humy_mean, humy_stdev, wspdy_mean, wspdy_stdev, alphasx, alphasytemp, alphasyhum, alphasywspd, dates

