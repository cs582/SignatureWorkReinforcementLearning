import numpy as np

from data_handling.retrieve_prices import retrieve_offline_token_prices, retrieve_offline_gas_prices, \
    retrieve_online_token_prices
from trading_environment import portfolio_management
from data_handling.preprocessing import prepare_dataset
from trading_environment.portfolio_management import portfolio_management

import torch
import logging


class Environment:
    def __init__(self, trading_days=365, token_prices_address=None, gas_address=None,
                 initial_cash=100000, buy_limit=100000, sell_limit=1000000, use_change=True, use_covariance=True,
                 reward_metric="sharpe", print_transactions=True, device=None):
        logging.info("Initializing Environment")

        # Trading Boundaries
        self.n_defi_tokens = -1
        self.curr_transactions = 0
        self.buy_limit = buy_limit
        self.sell_limit = sell_limit
        self.trading_days = trading_days
        self.reward_metric = reward_metric
        self.print_transactions = print_transactions

        self.use_covariance = use_covariance
        self.use_change = use_change

        self.database = None
        self.token_prices = None
        self.gas_prices = None

        self.token_prices_address = token_prices_address
        self.gas_address = gas_address

        self.curr_prices = None
        self.curr_gas = None
        self.curr_prices_image = None

        # Endowment
        self.initial_cash = initial_cash
        self.curr_cash = self.initial_cash
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

        self.data_index = 0

        # Done
        self.done = False

        # Hardware to use
        self.device = device

    def initialize_portfolio(self, starting_price=None, n_defi_tokens=None):
        logging.info("Environment called method initialize_portfolio")

        if self.token_prices_address is not None:
            logging.info("Retrieving token prices from online address: {}".format(self.token_prices_address))
            self.n_defi_tokens, self.token_names, self.token_prices = retrieve_online_token_prices(self.token_prices_address)

            # Reset trading days to be the number of available days if trading days is larger than that
            self.trading_days = min(self.trading_days, len(self.token_prices))
        else:
            logging.info("Getting offline token prices")
            self.token_prices = retrieve_offline_token_prices(starting_price=starting_price,
                                                              n_defi_tockens=n_defi_tokens,
                                                              n_trading_days=self.trading_days)

        if self.gas_address is not None:
            logging.info("Retrieving gas prices from online address: {}".format(self.gas_address))
            self.gas_prices = retrieve_online_token_prices(self.gas_address)
        else:
            logging.info("Getting offline gas prices")
            self.gas_prices = retrieve_offline_gas_prices(avg_price=100, std_deviation=25,
                                                          n_trading_days=self.trading_days)

        logging.info("Preparing the dataset")
        self.database = prepare_dataset(self.token_prices, use_change=self.use_change, use_covariance=self.use_covariance, lookback=10)

        for token in self.token_names:
            self.portfolio[token] = 0

        logging.info("Converting token prices to a dictionary")
        self.token_prices = self.token_prices.to_dict("records")

    def trade(self, actions=None):
        logging.info("Environment called method trade")
        if actions is None:
            logging.debug("Actions is None")

            # Update environment current state
            reward = None
            done = len(self.token_prices) == 0

            # Retrieving the current prices, input image, and gas price
            logging.debug("Retrieving the current prices, input image, and gas price.")
            self.curr_prices = self.token_prices[self.data_index] if not done else None
            self.curr_prices_image = torch.tensor(np.array([self.database[self.data_index]]), dtype=torch.double, device=self.device) if not done else None
            self.curr_gas = self.gas_prices[self.data_index] if not done else None

            self.data_index += 1

            return reward, self.curr_prices_image, done

        logging.debug("Actions is not None")

        assert self.n_defi_tokens==len(actions.reshape(-1,1)), f"actions don't match size expected {self.n_defi_tokens}, got {len(actions)}"

        # Sort indexes and get trading vector
        logging.debug("Transform actions to list")
        trading_vector = actions.tolist()
        logging.debug(f"Trading Vector: {trading_vector}")

        # Performing portfolio management
        logging.info("Performing Portfolio Management")
        self.portfolio, self.curr_net_worth, self.curr_cash, self.curr_units_value = portfolio_management(
            cash=self.curr_cash,
            token_portfolio=self.portfolio,
            current_token_prices=self.curr_prices,
            current_gas_price=self.curr_gas,
            actions=trading_vector,
            buy_limit=self.buy_limit,
            sell_limit=self.sell_limit,
            print_transactions=self.print_transactions
        )
        logging.info(f"Completed Portfolio Management!!!")
        logging.info(f"Current net worth = {self.curr_net_worth}")
        logging.info(f"Current cash = {self.curr_cash}")
        logging.info(f"Current assets value = {self.curr_units_value}")
        logging.debug(f"Portfolio = {self.portfolio}")

        # Store the current net_worth, cash, and units value
        self.net_worth_history.append(self.curr_net_worth)
        self.cash_history.append(self.curr_cash)
        self.units_value_history.append(self.curr_units_value)

        # Calculate roi and sharpe ratio
        daily_roi = (self.net_worth_history[-1] / self.net_worth_history[-2]) - 1
        gross_roi = (self.net_worth_history[-1] / self.initial_cash) - 1
        logging.info(f"Current daily roi = {daily_roi}")
        logging.info(f"Current gross roi = {gross_roi}")

        self.daily_roi_history.append(daily_roi)
        self.gross_roi_history.append(gross_roi)

        # Obtain the current number of days since the beginning of the transaction period
        n_days = len(self.net_worth_history)
        logging.info(f"Number of days since beginning of trading: {n_days}")

        # Calculate sharpe ratio
        sharpe = (n_days ** 0.5) * np.mean(self.daily_roi_history) / np.std(self.daily_roi_history)
        self.sharpe_history.append(float(sharpe))
        logging.info(f"Sharpe Ratio: {sharpe}")

        # Update environment current state
        reward = self.sharpe_history[-1] if self.reward_metric == "sharpe" else self.daily_roi_history[-1]
        done = len(self.database) == 0
        logging.info(f"Reinforcement Learning Reward: {self.reward_metric} = {reward}")

        # Move to next prices
        self.curr_prices_image = torch.tensor(np.array([self.database[self.data_index]]), dtype=torch.double, device=self.device) if not done else None
        self.curr_gas = self.gas_prices[self.data_index] if not done else None
        self.data_index += 1
        logging.debug(f"Data index: {self.data_index}")

        return reward, self.curr_prices_image, done
