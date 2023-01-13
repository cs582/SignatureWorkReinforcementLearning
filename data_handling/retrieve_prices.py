import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


def retrieve_online_gas_prices(address):
    # Read csv and convert to dataframe
    df = pd.read_csv(address)
    df.sort_index(inplace=True)
    return df


def retrieve_offline_gas_prices(avg_price, std_deviation, n_trading_days):
    # Generate a random starting price between 0 and 1
    prices_gas = np.random.normal(avg_price, std_deviation, n_trading_days)
    return prices_gas.tolist()


def retrieve_online_token_prices(filename):
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

    # Get token names and prices
    token_names = df.columns.tolist()

    assert df.isnull().sum().sum() == 0, f"there are {df.isnull().sum().sum()} missing values."

    return len(token_names), token_names, df


def retrieve_offline_token_prices(starting_price, n_defi_tockens, n_trading_days):
    # Generate names of each tocken
    token_names = [str(x) for x in range(1,n_defi_tockens+1)]

    # Markov process to simulate stock price time series data
    historical_prices = []
    for day in range(0, n_trading_days):
        if day == 0:
            # Generate a random starting price around the given starting price
            curr_prices = np.random.randint(low=starting_price - 1, high=starting_price + 1, size=(1, n_defi_tockens))
        else:
            # Move randomly from the previous day's price i.e. random walk from previous price
            prev_prices = [x for x in historical_prices[-1].values()]
            curr_prices = np.asanyarray(prev_prices) + (np.random.rand(1, n_defi_tockens) * 20.0 - 10.0)

        # Convert numpy array to list
        curr_prices = curr_prices.tolist()[0]

        # Map each fake token to its corresponding fake price
        cur_token_price_map = {}
        for token_name, cur_token_price in zip(token_names, curr_prices):
            cur_token_price_map[token_name] = cur_token_price

        historical_prices.append(cur_token_price_map)

    return token_names, historical_prices
