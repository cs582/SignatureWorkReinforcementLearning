import numpy as np


def preprocessing_correlation(op, cl, size):
    """
    Returns a dataset of open and close prices information in the form of a history of covariance matrices of shape 2*size x 2*size
    :param op:      --pd.Dataframe, open prices historical data for all tokens
    :param cl:      --pd.Dataframe, close prices historical data for all tokens
    :param size:    --int, size of the covariance matrices, it will take size number of days historical data to calculate the covariance matrix
    :return: X      --list, returns a list of 2D covariance matrices
    """
    X = []

    n = op.shape[0] - size
    for i in range(0, n):
        op_view = op.iloc[i:i + size]
        cl_view = cl.ilos[i:i + size]
        x = np.corrcoef(op_view, cl_view)
        X.append(x)

    return X


def preprocessing_open_product(op, cl):
    """

    :param op:
    :param cl:
    :return:
    """
    X = []

    n = op.shape[0]

    for i in range(0, n):
        x = np.matmul(op.iloc[i].T, cl.iloc[i])
        X.append(x)

    return X


def prepare_dataset(tokens_open_prices, tokens_close_prices, use_change, use_covariance, lookback):
    """

    :param tokens_open_prices:  --pd.Dataframe, it gets a dataframe with the token open prices
    :param tokens_close_prices: --pd.Dataframe, it gets a dataframe with the token close prices
    :param use_change:          --boolean, if true, then use price change else use raw prices
    :param use_covariance:      --boolean, if true, then use covariance else use open product
    :param lookback:            --int, when using covariance, it tells how many days back to look at. To trade for n days, it would be necessary to have n+lookback days as input.
    :return: X                  --np.array, returns the list of 2D input matrices
    """
    X_matrices = None

    if use_change:
        tokens_close_prices = tokens_close_prices.pct_change().drop([0], axis=0)
        tokens_open_prices = tokens_open_prices.pct_change().drop([0], axis=0)

    if use_covariance:
        size = lookback
        X_matrices = preprocessing_correlation(tokens_open_prices, tokens_close_prices, size)
        X_matrices = np.asanyarray(X_matrices)

    if not use_covariance:
        X_matrices = preprocessing_open_product(tokens_open_prices, tokens_close_prices)
        X_matrices = np.asanyarray(X_matrices)
        X_matrices = (X_matrices - X_matrices.mean(axis=0))/X_matrices.std(axis=0)

    return X_matrices
