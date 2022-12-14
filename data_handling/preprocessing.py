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


def prepare_dataset(tokens_prices, use_change, use_covariance, lookback):
    """

    :param tokens_prices:       --pd.Dataframe, it gets a dataframe with the token open prices
    :param use_change:          --boolean, if true, then use price change else use raw prices
    :param use_covariance:      --boolean, if true, then use covariance else use open product
    :param lookback:            --int, when using covariance, it tells how many days back to look at. To trade for n days, it would be necessary to have n+lookback days as input.
    :return: X                  --np.array, returns the list of 2D input matrices
    """
    X_matrices = None

    tokens_prices = tokens_prices.astype(np.float64)
    print("token prices shape:", tokens_prices.shape)

    if use_change:
        tokens_prices = tokens_prices.pct_change().drop([tokens_prices.index[0]], axis=0)

    if use_covariance:
        size = lookback
        X_matrices = preprocessing_correlation([tokens_prices], size)
        X_matrices = np.asanyarray(X_matrices)

    if not use_covariance:
        print("Use Outer Product")
        X_matrices = preprocessing_open_product([tokens_prices], tokens_prices)
        X_matrices = np.asanyarray(X_matrices)

    return X_matrices
