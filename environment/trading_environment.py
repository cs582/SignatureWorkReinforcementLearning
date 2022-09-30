import pandas as pd
import numpy as np

from retrieve_data.retrieve_prices import retrieve_offline_token_prices, retrieve_offline_gas_prices, retrieve_online_prices
from transaction_functions import portfolio_management, token_management

class Environment:
    def __init__(self, trading_days=365, token_prices_address=None, gas_address=None, initial_cash=100000, buy_limit=1000000, sell_limit=1000000):
        # Trading Boundaries
        self.buy_limit = buy_limit
        self.sell_limit = sell_limit
        self.trading_days = trading_days

        self.token_prices = None
        self.gas_prices = None
        self.nDeFiTokens = -1

        self.token_prices_address = token_prices_address
        self.gas_address = gas_address

        # Endowment
        self.curr_cash = initial_cash
        self.curr_units_value = 0
        self.curr_net_worth = self.curr_cash
        self.portfolio = {}

        # Endowment history
        self.cash_history = [self.curr_cash]
        self.units_value_history = [self.curr_units_value]
        self.net_worth_history = [self.curr_net_worth]

        # Performance metrics history
        self.daily_roi_history = [0]
        self.gross_roi_history = [0]
        self.sharpe_history = [0]

        self.token_names = None

        # Done
        self.done = False

    def import_token_prices(self, starting_price=90, n_defi_tokens=25):
        if self.token_prices_address is not None:
            self.token_names, self.token_prices = retrieve_online_prices(self.token_prices_address)
        else:
            self.token_prices = retrieve_offline_token_prices(starting_price=starting_price, n_defi_tockens=n_defi_tokens, n_trading_days=self.trading_days)

    def import_gas_prices(self, avg_price, std_deviation, online=False):
        if self.gas_address is not None:
            self.gas_prices = retrieve_online_prices(self.gas_address)
        else:
            self.gas_prices = retrieve_offline_gas_prices(avg_price=avg_price, std_deviation=std_deviation, n_trading_days=self.trading_days)

    def retrieve_data(self):
        self.import_token_prices()
        self.import_gas_prices()

    def initialize_portfolio(self):
        for token in self.token_names:
            self.portfolio[token] = 0

    def take_actions(self, actions):
        self.curr_net_worth, self.curr_cash, self.curr_units_value = portfolio_management(
            cash=self.curr_cash,
            token_portfolio=self.portfolio,
            current_token_prices=self.current_prices,
            current_gas_price=self.current_gas,
            actions=actions,
            buy_limit=self.buy_limit,
            sell_limit=self.sell_limit
        )

        # Store the current net_worth, cash, and units value
        self.net_worth_history.append(self.curr_net_worth)
        self.cash_history.append(self.curr_cash)
        self.units_value_history.append(self.curr_units_value)

        # Calculate roi and sharpe ratio
        daily_roi = (1-self.net_worth_history[-1]/self.net_worth_history[-2])
        gross_roi = (1-self.net_worth_history[-1]/self.net_worth_history[0])
        self.daily_roi_history.append(daily_roi)
        self.gross_roi_history.append(gross_roi)

        # Obtain the current number of days since the beginning of the transaction period
        n_days = len(self.net_worth_history)

        # Calculate sharpe ratio
        sharpe = (n_days**0.5)*np.mean(self.daily_roi_history)/np.std(self.daily_roi_history)
        self.sharpe_history.append(float(sharpe))

    def get_reward(self, metric):
        if metric=="daily roi":
            return self.daily_roi_history[-1]
        if metric=="sharpe":
            return self.sharpe_history[-1]
        if metric=="gross roi":
            return self.gross_roi_history[-1]