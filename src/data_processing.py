import pandas as pd
import numpy as np

def create_lag_features(df, features, lag_window, target_col = 'tavg'):
#create lagged features
#df: dataframe 
#features:list of feature columns 
#lag_window: number of previous days 
#target_col: prediction target

    df = df.copy()

    for feature in features:
        for lag in range(1, lag_window+1):
            df[f'{feature}_lag{lag}'] = df[feature].shift(lag)

    df['target_tomorrow_tavg'] = df[target_col].shift(-1)

    return df

def standardize_features(df, feature_columns):
    X = df[feature_columns].values

    means = np.mean(X, axis = 0)
    # print(means)
    stds = np.std(X, axis = 0)

    Zscr = (X-means)/stds

    max_abs = np.max(np.abs(Zscr), axis = 0)

    alphas = np.pi/max_abs

    rotations = Zscr*alphas

    return rotations, means, stds, alphas

