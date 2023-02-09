import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

import logging

logger = logging.getLogger("src/data_handling/retrieve_prices")

def retrieve_online_gas_prices(address):
    # Read csv and convert to dataframe
    df = pd.read_csv(address)
    df.sort_index(inplace=True)
    return df


def retrieve_offline_gas_prices(avg_price, std_deviation, n_trading_days):
    # Generate a random starting price between 0 and 1
    prices_gas = np.random.normal(avg_price, std_deviation, n_trading_days)
    return prices_gas.tolist()


def retrieve_token_prices(filename):
    """
    Retrieves historical data and converts csv to a list of dictionaries where the last element in the list is the
    latest historical price.

    :param filename:    --string, name of the filename that contains the historical token data
    :return:            --list, list of dictionaries that map to the price of each token at each time t
    """
    # Read csv and convert to dataframe
    df = pd.read_csv(filename, index_col=0)
    df.sort_index(inplace=True)
    df = df.drop(df.index[0], axis=0)

    df_values = KNNImputer(n_neighbors=5).fit_transform(df)
    for i, col in enumerate(df.columns):
        df[col] = df_values[:, i]

    assert df.isnull().sum().sum() == 0, f"there are {df.isnull().sum().sum()} missing values."

    return df
