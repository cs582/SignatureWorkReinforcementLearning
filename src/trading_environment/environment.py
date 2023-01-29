import numpy as np
import json

from src.data_handling.retrieve_prices import retrieve_token_prices, retrieve_offline_gas_prices, retrieve_online_gas_prices
from src.trading_environment import portfolio_management
from src.data_handling.preprocessing import prepare_dataset
from src.trading_environment.portfolio_management import portfolio_management
from src.utils.logging_tools.dataframe_logs import prices_and_gas_preview, images_preview

import torch
import logging

logger = logging.getLogger("src/trading_environment/environment")


class Environment:
    def __init__(
            self,
            trading_days: int = 365,
            token_prices_address: str = None,
            gas_address: str = None,
            initial_cash: float = 100000.0,
            buy_limit: float = 100000.0,
            sell_limit: float = 1000000.0,
            priority_fee: float = 2.0,
            gas_limit: int = 21000,
            use_change: bool = True,
            use_covariance: bool = True,
            portfolio_json: dict = None,
            portfolio_to_use: int = 1,
            reward_metric: str = "sharpe",
            device: str = None,
    ):
        """Initialize the trading environment.
        Args:
            trading_days (int, optional): The number of trading days in the simulation. Defaults to 365.
            token_prices_address (str, optional): The address of the file containing token prices. Defaults to None.
            gas_address (str, optional): The address of the file containing gas prices. Defaults to None.
            initial_cash (float, optional): The initial cash endowment of the agent. Defaults to 100000.0.
            buy_limit (float, optional): The maximum amount of cash that can be used to buy tokens. Defaults to 100000.0.
            sell_limit (float, optional): The maximum number of tokens that can be sold. Defaults to 1000000.0.
            priority_fee (float, optional): The fee applied to priority trades. Defaults to 2.0.
            gas_limit (int, optional): The maximum amount of gas that can be used in trades. Defaults to 21000.
            use_change (bool, optional): Whether to use the change in token prices as a feature. Defaults to True.
            use_covariance (bool, optional): Whether to use the covariance between tokens as a feature. Defaults to True.
            portfolio_json (dict, optional): The initial portfolio of tokens. Defaults to None.
            portfolio_to_use (int, optional): The index of the portfolio to use. Defaults to 1.
            reward_metric (str, optional): The metric used to evaluate the agent's performance. Defaults to "sharpe".
            device (str, optional): The device to use for computations. Defaults to None.
        """
        logger.info("Initializing the trading environment")
        self.trading_days = trading_days
        self.token_prices_address = token_prices_address
        self.gas_address = gas_address
        self.initial_cash = initial_cash
        self.buy_limit = buy_limit
        self.sell_limit = sell_limit
        self.priority_fee = priority_fee
        self.gas_limit = gas_limit
        self.use_change = use_change
        self.use_covariance = use_covariance
        self.portfolio_json = portfolio_json
        self.portfolio_to_use = portfolio_to_use
        self.reward_metric = reward_metric
        self.device = device

        self.n_defi_tokens = -1
        self.curr_transactions = 0
        self.database = None
        self.token_prices = None
        self.gas_prices = None
        self.curr_prices = None
        self.curr_gas = None
        self.curr_prices_image = None
        self.curr_cash = self.initial_cash
        self.curr_units_value = 0
        self.curr_net_worth = self.curr_cash
        self.portfolio = {}
        self.cash_history = [self.curr_cash]
        self.units_value_history = [self.curr_units_value]
        self.net_worth_history = [self.curr_net_worth]
        self.daily_roi_history = [0]
        self.gross_roi_history = [0]
        self.sharpe_history = [0]
        self.tokens_in_portfolio = None
        self.data_index = 0
        self.done = False

    def start_game(self):
        """Resets the environment to its initial state.
        """
        logger.info("Starting/Restarting the game")
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
        logger.info("Resetting token values in portfolio")
        for token in self.tokens_in_portfolio:
            self.portfolio[token] = 0.0
        logger.info("Game restarted successfully")

    def preload_prices(self, fake_avg_gas: float = 25.0, fake_gas_std: float = 5.0):
        """Initializes the portfolio with the given parameters.
        Args:
            fake_avg_gas (float, optional): The average gas price to create.
            fake_gas_std (float, optional): The standard deviation of the gas prices to create.
        """
        logger.info("Preloading the prices")
        with open('portfolios/portfolios.json', "r") as file:
            portfolio_options = json.load(file)
        self.tokens_in_portfolio = portfolio_options[f"Portfolio {self.portfolio_to_use}"]
        self.n_defi_tokens = len(self.tokens_in_portfolio)
        self.token_prices = retrieve_token_prices(self.token_prices_address)
        self.database = prepare_dataset(tokens_to_use=self.tokens_in_portfolio, token_prices=self.token_prices, use_change=self.use_change, use_covariance=self.use_covariance, lookback=10)
        self.trading_days = min(len(self.database), self.trading_days)
        self.token_prices = self.token_prices.iloc[-len(self.database):].to_dict("records")
        logger.info("Token Prices Successfully Loaded!!!")
        self.gas_prices = retrieve_online_gas_prices(self.gas_address) if self.gas_address is not None else retrieve_offline_gas_prices(avg_price=fake_avg_gas, std_deviation=fake_gas_std, n_trading_days=self.trading_days)
        logger.info("Gas Prices Successfully Loaded!!!")

        # Checking the token prices and gas prices in the log file
        prices_and_gas_preview(logger, self.token_prices, self.gas_prices)

        # Checking the database in the log file
        images_preview(logger, self.database)

    def trade(self, actions=None):
        """Executes the corresponding trades on the current day's prices.
        :param actions: (np.array, optional) the actions to take.
        :return: (float) reward for this trade.
        """
        if self.data_index >= self.trading_days:
            logger.debug("Game Over!!!")
            self.done = True
            return None, None, self.done

        if actions is None:
            logger.debug("Getting Initial State")
            self.curr_prices = self.token_prices[self.data_index]
            self.curr_prices_image = torch.from_numpy(np.array([self.database[self.data_index]])).to(self.device).double()
            self.curr_gas = self.gas_prices[self.data_index]
            self.data_index += 1
            return None, self.curr_prices_image, None

        assert self.n_defi_tokens==len(actions.reshape(-1,1)), f"actions don't match size expected {self.n_defi_tokens}, got {len(actions)}"

        # Sort indexes and get trading vector
        trading_vector = actions.tolist()

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
        logger.info(f"""Completed Portfolio Management!!!
        Current net worth = {self.curr_net_worth}
        Current cash = {self.curr_cash}
        Current assets value = {self.curr_units_value}
        Portfolio = {self.portfolio}
        """)

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
            self.curr_prices = self.token_prices[self.data_index]
            self.curr_prices_image = torch.tensor(np.array([self.database[self.data_index]]), dtype=torch.double, device=self.device)
            self.curr_gas = self.gas_prices[self.data_index]
            self.data_index += 1

        logger.debug(f"Next data index: {self.data_index}")

        return reward, self.curr_prices_image, done
