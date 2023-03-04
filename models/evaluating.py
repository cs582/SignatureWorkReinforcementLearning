import os

import numpy as np
import torch
import time

from models.models import DQN, DuelingDQN, ViT
from models.saving_tools import load_specific_model
from src.environment.trading_environment.agent import Agent
from src.environment.trading_environment.environment import Environment
from src.utils.visualization.timeseries_cashflow import TradingCycleCashFlow
from logs.logger_file import logger_eval


def eval(portfolio_to_use, images_saving_path, n_trading_days, n_tokens, epsilon, initial_cash, priority_fee, gas_limit,  episodes, reward_metric, use=3, lookback=10, dropout=0.2, vector_size=128, nhead=8, device=None, token_prices_address=None, model_path=None, model_name=None, algorithm=None, portfolio_json=None):
    with torch.autograd.set_detect_anomaly(True):
        timeseries_linechart = TradingCycleCashFlow(saving_path=images_saving_path)

        # Initialize environment and portfolio
        environment = Environment(
            trading_days=n_trading_days,
            token_prices_address=token_prices_address,
            gas_address=None,
            gas_limit=gas_limit,
            portfolio_json=portfolio_json,
            portfolio_to_use=portfolio_to_use,
            initial_cash=initial_cash,
            priority_fee=priority_fee,
            use=use,
            lookback=lookback,
            reward_metric=reward_metric,
            device=device
        )
        logger_eval.info("Environment Initialized!")

        # If the filenames are given, no parameters are necessary for method preload_prices
        environment.preload_prices()
        logger_eval.info("Prices are Preloaded!")

        # Calculate in-size and n_tokens
        n_tokens = environment.n_defi_tokens if n_tokens is None else n_tokens
        in_size = (lookback, n_tokens) if use == 3 else (n_tokens, n_tokens)
        logger_eval.info(f"Input size {in_size}. N tokens: {n_tokens}")

        # Calculate out-size
        out_size = environment.n_classes

        # Initialize replay memory D to capacity N
        agent = Agent(
            n_tokens=environment.n_defi_tokens,
            memory_size=0,
            min_epsilon=0,
            decay_rate=1
        )
        logger_eval.info("Agent Initialized")

        # Initialize action-value function Q with random weights
        set_inplace = True
        set_bias = False

        # Set model to use
        if algorithm == "Single_DQN" and model_name == "CNN":
            logger_eval.info("Using Single Stream DQN model")
            q = DQN(in_size=in_size, n_classes=out_size, inplace=set_inplace, bias=set_bias).double().to(device=device)
        elif algorithm == "Dueling_DQN" and model_name == "CNN":
            logger_eval.info("Using Dueling model")
            q = DuelingDQN(in_size=in_size, n_classes=out_size, inplace=set_inplace, bias=set_bias).double().to(device=device)
        elif algorithm == "Single_DQN" and model_name == "ViT":
            q = ViT(in_size=in_size, n_classes=out_size, dropout=dropout, vector_size=vector_size, nhead=nhead).double().to(device=device)

        logger_eval.debug(f"""
        Chosen Model {model_name} with {algorithm} Algorithm :
        {q}
        """)

        # load model
        q = load_specific_model(model_path, q)

        # Set Q-function to train mode
        q.eval()

        # Initiate training
        starting_time = time.time()
        for episode in range(0, episodes):
            os.environ['EPISODE'] = f"{episode}"

            mode = "TRAINING"

            # Start new training episode
            environment.start_game(mode='train')
            logger_eval.info(f"Training episode {episode}")

            # Initialize the current state
            logger_eval.info("Initial Trade call")
            _, cur_state, _ = environment.trade()
            rewards = []

            final_reward = None
            done = False
            current_trading_day = 0

            # Start the trading loop
            while not done:
                logger_eval.info(f"Trading Day {current_trading_day+1}")

                # Predict select random action
                y_hat = q(cur_state)
                cur_action = agent.get_action(y_hat, epsilon, episode)

                # Execute the action and get the reward and next state
                cur_reward, next_image, done = environment.trade(mode=mode, trading_day=current_trading_day, action=cur_action)

                # Store the experience in memory
                cur_experience = (cur_state, cur_action, cur_reward, next_image)
                agent.store(cur_experience)

                # Update the cash flow information to the real time chart
                timeseries_linechart.update(environment.cash_history[-1], environment.units_value_history[-1], environment.net_worth_history[-1], mode='train')

                # Update the current state
                cur_state = next_image

                rewards.append(cur_reward)
                if done:
                    final_reward = environment.gross_roi_history[-1]

                current_trading_day += 1

            # Print
            print(f"EPISODE {episode}. Last Trading day: {current_trading_day-1}.\nFINAL REWARD: {final_reward}. ELAPSED TIME: {time.time() - starting_time} seconds.")

            #####################
            #  EVALUATING LOOP  #
            #####################

            mode = "EVAL"

            # Restart Environment in Evaluating Mode (eval data)
            environment.start_game(mode='eval')

            # Initialize the current state
            logger_eval.info("Initial Trade EVAL call")
            _, cur_state, _ = environment.trade()
            rewards_eval = []

            final_reward_eval = None
            done_eval = False
            current_trading_day_eval = 0

            # Start the trading loop
            while not done_eval:
                logger_eval.info(f"EVAL Trading Day {current_trading_day_eval+1}")

                # Predict select random action from target function
                y_hat = q(cur_state)
                cur_action = agent.get_action(y_hat, epsilon, episode)

                # Execute the action and get the reward and next state
                cur_reward, next_image, done_eval = environment.trade(mode=mode, trading_day=current_trading_day_eval, action=cur_action)

                # Update the cash flow information to the real time chart
                timeseries_linechart.update(environment.cash_history[-1], environment.units_value_history[-1], environment.net_worth_history[-1], mode='eval')

                # Store current evaluating reward
                rewards_eval.append(cur_reward)
                if done_eval:
                    final_reward_eval = environment.gross_roi_history[-1]

                current_trading_day_eval += 1

            # Calculate the average loss and reward of the episode
            average_rewd_eval = np.mean(rewards_eval)

            # Print
            print(f"EPISODE {episode+1}. Last Trading day: {current_trading_day-1}.\nFINAL REWARD: {final_reward_eval}. ELAPSED TIME: {time.time() - starting_time} seconds.")

            # Reset the cash flow history for a new episode
            timeseries_linechart.reset()

        return q