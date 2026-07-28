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

    Y = data["tavg"].shift(-1).dropna().values
    data = data.dropna()

    X = data[lagged_cols].values

    Y = Y[-len(X):]

    tavg_col = X[:, 0]
    hum_col = X[:, 1]
    wspd_col = X[:, 2]
    tavgy_col = Y

    tavg_mean = np.mean(tavg_col)
    hum_mean = np.mean(hum_col)
    wspd_mean = np.mean(wspd_col)
    tavgy_mean = np.mean(tavgy_col)

    print("MEANS", tavg_mean, hum_mean, wspd_mean, tavgy_mean)

    tavg_stdev = np.std(tavg_col)
    hum_stdev = np.std(hum_col)
    wspd_stdev = np.std(wspd_col)
    tavgy_stdev = np.std(tavgy_col)

    print("STANDARD DEVIATIONS", tavg_stdev, hum_stdev, wspd_stdev, tavgy_stdev)

    tavg_zscr = (X[:, 0] - tavg_mean) / tavg_stdev
    hum_zscr = (X[:, 1] - hum_mean) / hum_stdev
    wspd_zscr = (X[:, 2] - wspd_mean) / wspd_stdev
    ZX = np.column_stack([tavg_zscr, hum_zscr, wspd_zscr])

    tavgy_zscr = (Y - tavgy_mean) / tavgy_stdev
    ZY = tavgy_zscr

    print("Z INPUT ARRAY SIZE", ZX.shape)
    print("Z INPUT ARRAY SAMPLE", ZX[:5])

    print("Z EXPECTED ARRAY SIZE", ZY.shape)
    print("Z EXPECTED ARRAY SAMPLE", ZY[:5])

    max_absx = np.max(np.abs(ZX), axis = 0)
    max_absy = np.max(np.abs(ZY), axis = 0)

    print("HIGHEST Z INPUT SCORE", max_absx)
    print("HIGHEST Z EXPECTED SCORE", max_absy)

    alphasx = np.pi / max_absx
    alphasy = np.pi / max_absy

    print("SCALING FACTORS INPUT", alphasx)
    print("SCALING FACTORS EXPECTED", alphasy)

    rotationsx = ZX * alphasx
    rotationsy = ZY * alphasy

    print("ROTATIONS INPUT ARRAY SIZE", rotationsx.shape)
    print("ROTATIONS INPUT ARRAY SAMPLE", rotationsx[:5])
    print("ROTATIONS EXPECTED ARRAY SIZE", rotationsy.shape)
    print("ROTATIONS EXPECTED ARRAY SAMPLE", rotationsy[:5])

    print("X/Y aligned:", len(rotationsx) == len(rotationsy))
    print("Rotations shape:", rotationsx.shape)
    print("Y shape:", rotationsy.shape)
    print("Sample rotations:", rotationsx[:3])
    print("Sample Y:", rotationsy[:3])

    return rotationsx, rotationsy, tavg_mean, tavgy_mean, tavg_stdev, tavgy_stdev, alphasx, alphasy, dates

