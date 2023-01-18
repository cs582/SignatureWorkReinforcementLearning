import numpy as np
import json

from data_handling.retrieve_prices import retrieve_offline_token_prices, retrieve_offline_gas_prices, \
    retrieve_online_token_prices
from trading_environment import portfolio_management
from data_handling.preprocessing import prepare_dataset
from trading_environment.portfolio_management import portfolio_management

import torch
import logging

logger = logging.getLogger("trading_environment/environment")


class Environment:
    def __init__(self, trading_days=365, token_prices_address=None, gas_address=None,
                 initial_cash=100000, buy_limit=100000, sell_limit=1000000,
                 priority_fee=2, gas_limit=21000, use_change=True, use_covariance=True,
                 portfolio_json=None, portfolio_to_use=1,
                 reward_metric="sharpe", device=None):
        logger.info("Initializing Environment")

        # Trading Boundaries
        self.n_defi_tokens = -1
        self.curr_transactions = 0
        self.gas_limit = gas_limit
        self.buy_limit = buy_limit
        self.sell_limit = sell_limit
        self.trading_days = trading_days
        self.reward_metric = reward_metric

        self.use_covariance = use_covariance
        self.use_change = use_change

        self.database = None
        self.token_prices = None
        self.gas_prices = None

        self.token_prices_address = token_prices_address
        self.gas_address = gas_address
        self.portfolio_json = portfolio_json

        self.priority_fee = priority_fee

        self.portfolio_to_use = portfolio_to_use

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

        self.tokens_in_portfolio = None

        self.data_index = 0

        # Done
        self.done = False

        # Hardware to use
        self.device = device

    def start_game(self):
        self.done = False

        self.daily_roi_history = [0]
        self.gross_roi_history = [0]
        self.sharpe_history = [0]

        self.curr_prices_image = None
        self.curr_gas = None
        self.data_index = 0

        self.curr_cash = self.initial_cash
        self.curr_units_value = 0
        self.curr_net_worth = self.curr_cash

        self.cash_history = [self.curr_cash]
        self.units_value_history = [self.curr_units_value]
        self.net_worth_history = [self.curr_net_worth]

        for tkn in self.tokens_in_portfolio:
            self.portfolio[tkn] = 0.0

    def initialize_portfolio(self, starting_price=None, n_defi_tokens=None):
        logger.info("Environment called method initialize_portfolio")

        # Open JSON file with portfolio options
        self.tokens_in_portfolio = json.loads(open('portfolios/portfolios.json', "r").read())[f"Portfolio {self.portfolio_to_use}"]
        self.n_defi_tokens = len(self.tokens_in_portfolio)

        if self.token_prices_address is not None:
            logger.info("Retrieving token prices from online address: {}".format(self.token_prices_address))
            _, _, self.full_token_prices = retrieve_online_token_prices(self.token_prices_address)

        else:
            logger.info("Getting offline token prices")
            self.full_token_prices = retrieve_offline_token_prices(starting_price=starting_price, n_defi_tockens=n_defi_tokens, n_trading_days=self.trading_days)

        if self.gas_address is not None:
            logger.info("Retrieving gas prices from online address: {}".format(self.gas_address))
            self.gas_prices = retrieve_online_token_prices(self.gas_address)
        else:
            logger.info("Getting offline gas prices")
            self.gas_prices = retrieve_offline_gas_prices(avg_price=25, std_deviation=5,
                                                          n_trading_days=self.trading_days)

        logger.info("Preparing the dataset")
        self.database = prepare_dataset(tokens_to_use=self.tokens_in_portfolio, tokens_prices=self.full_token_prices, use_change=self.use_change, use_covariance=self.use_covariance, lookback=10)
        self.token_prices = self.full_token_prices.iloc[-len(self.database):]

        for token in self.tokens_in_portfolio:
            self.portfolio[token] = 0

        logger.debug(f"database size: {self.database.shape}, prices size: {self.token_prices.shape}")
        self.trading_days = min(self.trading_days, len(self.token_prices))

        logger.info("Converting token prices to a dictionary")
        self.token_prices = self.token_prices.to_dict("records")

    def trade(self, actions=None):
        logger.info("Environment called method trade")
        if actions is None:
            logger.debug("Actions is None")

            # Update environment current state
            reward = None
            done = len(self.token_prices) == 0

            # Retrieving the current prices, input image, and gas price
            logger.debug("Retrieving the current prices, input image, and gas price.")
            self.curr_prices = self.token_prices[self.data_index] if not done else None
            self.curr_prices_image = torch.tensor(np.array([self.database[self.data_index]]), dtype=torch.double, device=self.device) if not done else None
            self.curr_gas = self.gas_prices[self.data_index] if not done else None

            self.data_index += 1

            return reward, self.curr_prices_image, done

        logger.debug("Actions is not None")

        assert self.n_defi_tokens==len(actions.reshape(-1,1)), f"actions don't match size expected {self.n_defi_tokens}, got {len(actions)}"

        # Sort indexes and get trading vector
        logger.debug("Transform actions to list")
        trading_vector = actions.tolist()
        logger.debug(f"Trading Vector: {trading_vector}")

        # Performing portfolio management
        logger.info("Performing Portfolio Management")
        self.portfolio, self.curr_net_worth, self.curr_cash, self.curr_units_value = portfolio_management(
            cash=self.curr_cash,
            token_portfolio=self.portfolio,
            current_token_prices=self.curr_prices,
            current_gas_price=self.curr_gas,
            priority_fee=self.priority_fee,
            gas_limit=self.gas_limit,
            actions=trading_vector,
            buy_limit=self.buy_limit,
            sell_limit=self.sell_limit
        )
        logger.info(f"Completed Portfolio Management!!!")
        logger.info(f"Current net worth = {self.curr_net_worth}")
        logger.info(f"Current cash = {self.curr_cash}")
        logger.info(f"Current assets value = {self.curr_units_value}")
        logger.debug(f"Portfolio = {self.portfolio}")

        # Store the current net_worth, cash, and units value
        self.net_worth_history.append(self.curr_net_worth)
        self.cash_history.append(self.curr_cash)
        self.units_value_history.append(self.curr_units_value)

        # Calculate roi and sharpe ratio
        yesterday_net_worth = self.net_worth_history[-2] if self.net_worth_history[-2] != 0.0 else 1
        today_net_worth = self.net_worth_history[-1]
        daily_roi = (today_net_worth / yesterday_net_worth) - 1
        gross_roi = (today_net_worth / self.initial_cash) - 1
        logger.info(f"Current daily roi = {daily_roi}")
        logger.info(f"Current gross roi = {gross_roi}")

        self.daily_roi_history.append(daily_roi)
        self.gross_roi_history.append(gross_roi)

        # Obtain the current number of days since the beginning of the transaction period
        n_days = len(self.net_worth_history)
        logger.info(f"Number of days since beginning of trading: {n_days}")

        # Calculate sharpe ratio
        sharpe = (n_days ** 0.5) * np.mean(self.daily_roi_history) / np.std(self.daily_roi_history)
        self.sharpe_history.append(float(sharpe))
        logger.info(f"Sharpe Ratio: {sharpe}")

        # Update environment current state
        reward = self.sharpe_history[-1] if self.reward_metric == "sharpe" else self.daily_roi_history[-1]
        done = (self.curr_net_worth <= self.initial_cash*0.25) or (self.data_index >= len(self.database)-1)
        logger.info(f"Reinforcement Learning Reward: {self.reward_metric} = {reward}. Done? {done}")

        # If not done, then move to next prices
        if not done:
            self.curr_prices_image = torch.tensor(np.array([self.database[self.data_index]]), dtype=torch.double, device=self.device)
            self.curr_gas = self.gas_prices[self.data_index]
            self.data_index += 1
        if done:
            self.curr_prices_image = None
            self.curr_gas = None
            self.data_index = 0

            self.curr_cash = self.initial_cash
            self.curr_units_value = 0
            self.curr_net_worth = self.curr_cash

            for tkn in self.tokens_in_portfolio:
                self.portfolio[tkn] = 0.0


        logger.debug(f"Next data index: {self.data_index}")

        return reward, self.curr_prices_image, done
