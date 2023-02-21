import numpy as np


def preprocessing_correlation(prices, size):
    """
    Returns a dataset of open and close prices information in the form of a history of covariance matrices of shape 2*size x 2*size
    :param prices:      --list(pd.Dataframe), list of prices historical data for all tokens. The list is a list of pandas dataframes inclusing at least one of closing prices, open prices, high prices, low prices, etc...
    :param size:        --int, size of the covariance matrices, it will take size number of days historical data to calculate the covariance matrix
    :return: X          --list, returns a list of 2D covariance matrices
    """
    X = []

    n = prices[0].shape[0] - size
    for i in range(0, n):
        x = []
        for k in range(0, len(prices)):
            price_view = prices[k].iloc[i:i + size].T
            x_transformed = np.nan_to_num(np.corrcoef(price_view), nan=0)
            x.append(x_transformed)
        X.append(x)

    return X


def preprocessing_snapshots(prices, size):
    """
    Returns a dataset snapshot of prices of in batches of size timesteps
    :param prices:      --list(pd.Dataframe), list of prices historical data for all tokens. The list is a list of pandas dataframes inclusing at least one of closing prices, open prices, high prices, low prices, etc...
    :param size:        --int, size of the covariance matrices, it will take size number of days historical data to calculate the covariance matrix
    :return: X          --list, returns a list of 2D covariance matrices
    """
    print("PROCESSING SNAPSHOTS")
    X = []

    print(prices[0])
    n = len(prices[0]) - size
    print(n)
    for i in range(0, n):
        x = []
        for k in range(0, len(prices)):
            prices_view = prices[k].iloc[i:i + size].values
            print(prices_view)
            price_snapshot = np.nan_to_num(prices_view, nan=0)
            print(price_snapshot)
            x.append(price_snapshot)
        print(x)
        X.append(x)
    print(X[-1])

    return X

def prepare_dataset(tokens_to_use, token_prices, use, lookback):
    """
    :param tokens_to_use        --list, it gives you the tokens to take into consideration i.e. asset space or portfolio
    :param token_prices:        --pd.Dataframe, it gets a dataframe with the token open prices
    :param use:                 --int, 1->use change 2->use covariance 3->use snapshot
    :param lookback:            --int, when using covariance, it tells how many days back to look at. To trade for n days, it would be necessary to have n+lookback days as input.
    :return: X                  --np.array, returns the list of 2D input matrices
    """
    print("PREPARE DATASET")
    print(f"use = {use}. type: {type(use)}")
    X_matrices = None

    token_prices = token_prices[tokens_to_use].astype(np.float64)
    token_prices = token_prices.pct_change().drop([token_prices.index[0]], axis=0)

    if use == 2:
        size = lookback
        X_matrices = preprocessing_correlation([token_prices], size)
        X_matrices = np.asanyarray(X_matrices)

    if use == 3:
        size = lookback
        X_matrices = preprocessing_snapshots([token_prices], size)
        print(X_matrices[:5])
        X_matrices = np.asanyarray(X_matrices)
        print(X_matrices[:5])

    return X_matrices
