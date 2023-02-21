import numpy as np
import json

from src.preprocessing.data_handling.retrieve_prices import retrieve_token_prices, retrieve_offline_gas_prices, retrieve_online_gas_prices
from src.environment.trading_environment import portfolio_management
from src.preprocessing.data_handling.preprocessing import prepare_dataset
from src.environment.trading_environment.portfolio_management import portfolio_management
from src.utils.logging_tools.dataframe_logs import prices_and_gas_preview, images_preview
from src.utils.environment_tools import map_actions_to_tokens
from src.environment.utils import get_trading_action
from src.utils.visualization.custom_messages import show_rewards


import torch
import logging

logger_main = logging.getLogger("src/trading_environment/environment")


class Environment:
    def __init__(
            self,
            trading_days: int = 365,
            token_prices_address: str = None,
            gas_address: str = None,
            initial_cash: float = 100000.0,
            priority_fee: float = 2.0,
            gas_limit: int = 21000,
            use_change: bool = True,
            use_covariance: bool = True,
            portfolio_json: dict = None,
            portfolio_to_use: int = 1,
            reward_metric: str = "sharpe",
            device: str = None,
            prev_action: int = -1
    ):
        """Initialize the trading environment.
        Args:
            trading_days (int, optional): The number of trading days in the simulation. Defaults to 365.
            token_prices_address (str, optional): The address of the file containing token prices. Defaults to None.
            gas_address (str, optional): The address of the file containing gas prices. Defaults to None.
            initial_cash (float, optional): The initial cash endowment of the agent. Defaults to 100000.0.
            priority_fee (float, optional): The fee applied to priority trades. Defaults to 2.0.
            gas_limit (int, optional): The maximum amount of gas that can be used in trades. Defaults to 21000.
            use_change (bool, optional): Whether to use the change in token prices as a feature. Defaults to True.
            use_covariance (bool, optional): Whether to use the covariance between tokens as a feature. Defaults to True.
            portfolio_json (dict, optional): The initial portfolio of tokens. Defaults to None.
            portfolio_to_use (int, optional): The index of the portfolio to use. Defaults to 1.
            reward_metric (str, optional): The metric used to evaluate the agent's performance. Defaults to "sharpe".
            device (str, optional): The device to use for computations. Defaults to None.
            curr_action (int, optional): Initial default action, must always be -1 for training or eval.
        """
        logger_main.info("Initializing the trading environment")
        self.trading_days = trading_days
        self.token_prices_address = token_prices_address
        self.gas_address = gas_address
        self.initial_cash = initial_cash
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

        self.database_train = None
        self.token_prices_train = None
        self.gas_prices_train = None

        self.database_eval = None
        self.token_prices_eval = None
        self.gas_prices_eval = None

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
        self.action_map = None

        self.data_index = 0
        self.done = False

        self.prev_action = prev_action

    def start_game(self, mode=None):
        """Resets the environment to its initial state.
        :param mode (str) set to train for training or eval to evaluate
        """
        logger_main.info("Starting/Restarting the game")
        self.done = False
        self.daily_roi_history = [0]
        self.gross_roi_history = [0]
        self.sharpe_history = [0]
        self.curr_prices_image = None
        self.curr_gas = None
        self.prev_action = -1
        self.data_index = 0
        self.curr_cash = self.initial_cash
        self.curr_units_value = 0
        self.curr_net_worth = self.curr_cash
        self.cash_history = [self.curr_cash]
        self.units_value_history = [self.curr_units_value]
        self.net_worth_history = [self.curr_net_worth]
        logger_main.info("Resetting token values in portfolio")
        for token in self.tokens_in_portfolio:
            self.portfolio[token] = 0.0
        logger_main.info("Game restarted successfully")

        # Assign the corresponding prices to the variables
        self.token_prices = self.token_prices_train if mode == "train" else self.token_prices_eval
        self.database = self.database_train if mode == "train" else self.database_eval
        self.gas_prices = self.gas_prices_train if mode == "train" else self.gas_prices_eval

    def preload_prices(self, fake_avg_gas: float = 25.0, fake_gas_std: float = 5.0):
        """Initializes the portfolio with the given parameters.
        Args:
            fake_avg_gas (float, optional): The average gas price to create.
            fake_gas_std (float, optional): The standard deviation of the gas prices to create.
        """
        # Load Prices
        logger_main.info("Preloading the prices")
        with open(self.portfolio_json, "r") as file:
            portfolio_options = json.load(file)[f"Portfolio {self.portfolio_to_use}"]
        self.tokens_in_portfolio = portfolio_options["tokens"]
        self.action_map = portfolio_options['action_map']
        self.n_defi_tokens = len(self.tokens_in_portfolio)
        self.n_classes = len(self.action_map.keys())+1

        # RETRIEVING WHOLE DATA
        token_prices = retrieve_token_prices(self.token_prices_address)
        database = prepare_dataset(tokens_to_use=self.tokens_in_portfolio, token_prices=token_prices, use_change=self.use_change, use_covariance=self.use_covariance, lookback=10)
        self.trading_days = min(len(database), self.trading_days)
        token_prices = token_prices.iloc[-len(database):].to_dict("records")
        logger_main.info("Token Prices Successfully Loaded!!!")
        gas_prices = retrieve_online_gas_prices(self.gas_address) if self.gas_address is not None else retrieve_offline_gas_prices(avg_price=fake_avg_gas, std_deviation=fake_gas_std, n_trading_days=self.trading_days)
        logger_main.info("Gas Prices Successfully Loaded!!!")

        # Checking the token prices and gas prices in the log file
        prices_and_gas_preview(logger_main, token_prices, gas_prices)

        # Checking the database in the log file
        images_preview(logger_main, database)

        # Create TRAINING DATA
        training_days = int(self.trading_days * 0.75)
        self.token_prices_train = token_prices[:training_days]
        self.database_train = database[:training_days]
        self.gas_prices_train = gas_prices[:training_days]

        # Create EVALUATION DATA
        self.token_prices_eval = token_prices[training_days:]
        self.database_eval = database[training_days:]
        self.gas_prices_eval = gas_prices[training_days:]

    def trade(self, mode="UNK", trading_day=0, action=None):
        """Executes the corresponding trades on the current day's prices.
        :param mode: (str, optional) trading mode can be either training or evaluating
        :param action: (int, required) the actions to take.
        :return: (float) reward for this trade.
        """

        if self.data_index >= len(self.database)-1:
            logger_main.debug("Game Over!!!")
            self.done = True
            return None, None, self.done

        if action is None:
            logger_main.debug("Getting Initial State")
            self.curr_prices = self.token_prices[self.data_index]
            self.curr_prices_image = torch.from_numpy(np.array([self.database[self.data_index]])).to(self.device).double()
            self.curr_gas = self.gas_prices[self.data_index]
            self.data_index += 1
            return None, self.curr_prices_image, None

        # Sort indexes and get tokens to trade
        tokens_to_buyorhold = map_actions_to_tokens(action, self.action_map)
        tokens_to_sell = [tkn for tkn in self.tokens_in_portfolio if self.portfolio[tkn] > 0]
        logger_main.info(f"Tokens to buy/hold: {tokens_to_buyorhold}")
        logger_main.info(f"Tokens to sell: {tokens_to_sell}")

        # Get the current position
        position = get_trading_action(prev_action=self.prev_action, curr_action=action)

        # Performing portfolio management
        self.portfolio, self.curr_net_worth, self.curr_cash, self.curr_units_value = portfolio_management(
            mode=mode,
            day=trading_day,
            position=position,
            prev_action=self.prev_action,
            curr_action=action,
            cash=self.curr_cash,
            token_portfolio=self.portfolio,
            current_token_prices=self.curr_prices,
            current_gas_price=self.curr_gas,
            priority_fee=self.priority_fee,
            gas_limit=self.gas_limit,
            tokens_to_buyorhold=tokens_to_buyorhold,
            tokens_to_sell=tokens_to_sell
        )

        logger_main.info(f"""
        Completed Portfolio Management!!!
        Current net worth: {self.curr_net_worth}
        Current cash: {self.curr_cash}
        Current tokens value: {self.curr_units_value}
        Portfolio: {self.portfolio}
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
        logger_main.info(f"Current daily roi = {daily_roi}")
        logger_main.info(f"Current gross roi = {gross_roi}")

        self.daily_roi_history.append(daily_roi)
        self.gross_roi_history.append(gross_roi)

        # Obtain the current number of days since the beginning of the transaction period
        n_days = len(self.net_worth_history)

        # Calculate sharpe ratio
        r_std = np.std(self.daily_roi_history)
        r_mean = (n_days ** 0.5) * np.mean(self.daily_roi_history)
        sharpe = r_mean/r_std if r_std!=0.0 else r_mean
        self.sharpe_history.append(float(sharpe))
        logger_main.info(f"Sharpe Ratio: {sharpe}")

        # Calculate REWARD
        metric_today = self.sharpe_history[-1] if self.reward_metric == "sharpe" else self.daily_roi_history[-1]
        metric_max = max(self.sharpe_history) if self.reward_metric == "sharpe" else max(self.daily_roi_history)
        reward = metric_today - metric_max

        # Show the reward on screen in CYAN
        show_rewards(mode=mode, day=trading_day, roi=gross_roi, sharpe=sharpe, reward=reward, hist_max=metric_max)

        # Move to next prices
        self.curr_prices = self.token_prices[self.data_index]
        self.curr_prices_image = torch.tensor(np.array([self.database[self.data_index]]), dtype=torch.double, device=self.device)
        self.curr_gas = self.gas_prices[self.data_index]
        self.data_index += 1
        logger_main.debug(f"Next data index: {self.data_index}. Max index: {len(self.database)-1}")

        # Check if done
        done = (self.curr_net_worth <= self.initial_cash*0.25) or (self.data_index >= len(self.database)-1)
        logger_main.info(f"Reinforcement Learning Reward: {self.reward_metric} = {reward}. Done? {done}")

        self.prev_action = action

        return reward, self.curr_prices_image, done
