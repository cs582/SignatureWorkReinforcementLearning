import numpy as np
import torch
import time
import logging
from datetime import datetime

from models.models import DQN, DuelingDQN
from models.optimization import optimize_dqn
from models.saving_tools import save_model, load_model
from src.environment.trading_environment.agent import Agent
from src.environment.trading_environment.environment import Environment
from src.utils.visualization.timeseries_cashflow import TradingCycleCashFlow
from logs.logger_file import logger_main


def train(portfolio_to_use, images_saving_path, n_trading_days, n_tokens, min_epsilon, decay_rate, initial_cash, priority_fee, gas_limit, loss_function, episodes, batch_size, memory_size, lr, epsilon, gamma, momentum, reward_metric, use=3, lookback=10, device=None, token_prices_address=None, save_path=None, model_name=None, portfolio_json=None, load_from_checkpoint=True):
    with torch.autograd.set_detect_anomaly(True):
        timeseries_linechart = TradingCycleCashFlow(saving_path=images_saving_path)

        train_history = {"metric_history": [], "metric_history_eval": [], "avg_loss": []}

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
        logger_main.info("Environment Initialized!")

        # If the filenames are given, no parameters are necessary for method preload_prices
        environment.preload_prices()
        logger_main.info("Prices are Preloaded!")

        # Calculate in-size and n_tokens
        n_tokens = environment.n_defi_tokens if n_tokens is None else n_tokens
        in_size = (lookback, n_tokens) if use==3 else (n_tokens, n_tokens)
        logger_main.info(f"Input size {in_size}. N tokens: {n_tokens}")

        # Calculate out-size
        out_size = environment.n_classes

        # Initialize replay memory D to capacity N
        agent = Agent(
            n_tokens=environment.n_defi_tokens,
            memory_size=memory_size,
            min_epsilon=min_epsilon,
            decay_rate=decay_rate
        )
        logger_main.info("Agent Initialized")

        # Initialize action-value function Q with random weights
        set_inplace = True
        set_bias = False

        # Set model to use
        if model_name == "Single_DQN" or model_name == "Double_DQN":
            logger_main.info("Using Single Stream DQN model")
            q = DQN(in_size=in_size, n_classes=out_size, inplace=set_inplace, bias=set_bias).double().to(device=device)
            t = DQN(in_size=in_size, n_classes=out_size, inplace=set_inplace, bias=set_bias).double().to(device=device)
        else:
            logger_main.info("Using Dueling model")
            q = DuelingDQN(in_size=in_size, n_classes=out_size, inplace=set_inplace, bias=set_bias).double().to(device=device)
            t = DuelingDQN(in_size=in_size, n_classes=out_size, inplace=set_inplace, bias=set_bias).double().to(device=device)

        logger_main.debug(f"""
        Chosen Model {model_name}:
        {q}
        """)

        # Setting optimizer
        optimizer = torch.optim.SGD(q.parameters(), lr=lr, momentum=momentum)

        # Setting starting episode
        starting_episode = 0

        # If using a checkpoint, load from checkpoint
        if load_from_checkpoint:
            q, optimizer, train_history, starting_episode = load_model(save_path, model_name, q, optimizer)

        # Load weights from the q to the t model
        t.load_state_dict(q.state_dict())

        # Initiate training
        starting_time = time.time()
        for episode in range(starting_episode, episodes):
            mode = "TRAINING"

            # Set models in train mode
            q.train()
            t.train()

            # Start new training episode
            environment.start_game(mode='train')
            logger_main.info(f"Training episode {episode}")

            # Initialize the current state
            logger_main.info("Initial Trade call")
            _, cur_state, _ = environment.trade()
            rewards = []
            episode_loss = []

            final_reward = None
            done = False
            current_trading_day = 0

            # Start the trading loop
            while not done:
                logger_main.info(f"Trading Day {current_trading_day+1}")

                # Initialize gradient
                optimizer.zero_grad()

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

                # Sample a batch of experiences from memory
                experience_batch = agent.draw(batch_size=batch_size)

                # Perform the optimization step
                loss = optimize_dqn(dqn=q, target=t, batch=experience_batch, loss_fn=loss_function, gamma=gamma, optimizer=optimizer, device=device)

                # Append the current loss and reward to history
                episode_loss.append(loss)
                rewards.append(cur_reward)
                if done:
                    final_reward = environment.gross_roi_history[-1]

                current_trading_day += 1

            # Calculate the average loss and reward of the episode
            average_loss = np.mean(loss)
            average_rewd = np.mean(rewards)

            # Print
            print(f"EPISODE {episode}. Last Trading day: {current_trading_day-1}.\nLOSS: {average_loss}. FINAL REWARD: {final_reward}. ELAPSED TIME: {time.time() - starting_time} seconds.")

            # Append the final reward and average loss for this episode to the training history
            train_history["metric_history"].append(average_rewd)
            train_history["avg_loss"].append(average_loss)

            # Save the model every 10 episodes
            if (episode+1) % 10 == 0:
                logger_main.info(f"Saving model at episode {episode}")
                current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                filename = f"{model_name}_{episode}_{current_time}.pt"
                save_model(model=q, episode=episode, optimizer=optimizer, train_history=train_history, PATH=save_path, filename=filename)

            #####################
            #  EVALUATING LOOP  #
            #####################
            mode = "EVAL"

            environment.start_game(mode='eval')

            # Set models in evaluation mode
            q.eval()
            t.eval()

            # Initialize the current state
            logger_main.info("Initial Trade EVAL call")
            _, cur_state, _ = environment.trade()
            rewards_eval = []

            final_reward_eval = None
            done_eval = False
            current_trading_day_eval = 0

            # Start the trading loop
            while not done_eval:
                logger_main.info(f"EVAL Trading Day {current_trading_day_eval+1}")

                # Predict select random action
                y_hat = q(cur_state)
                cur_action = agent.get_action(y_hat, min_epsilon, episode)

                # Execute the action and get the reward and next state
                cur_reward, next_image, done_eval = environment.trade(mode=mode, trading_day=current_trading_day_eval, action=cur_action)

                # Update the cash flow information to the real time chart
                timeseries_linechart.update(environment.cash_history[-1], environment.units_value_history[-1], environment.net_worth_history[-1], mode='eval')

                # Store current evaluating reward
                rewards_eval.append(cur_reward)
                if done:
                    final_reward_eval = environment.gross_roi_history[-1]

                current_trading_day_eval += 1

            # Calculate the average loss and reward of the episode
            average_rewd_eval = np.mean(rewards_eval)

            # Print
            print(f"EPISODE {episode+1}. Last Trading day: {current_trading_day-1}.\nFINAL REWARD: {final_reward_eval}. ELAPSED TIME: {time.time() - starting_time} seconds.")

            # Append the final reward and average loss for this episode to the training history
            train_history["metric_history_eval"].append(average_rewd_eval)

            # Reset the cash flow history for a new episode
            timeseries_linechart.reset()

        return q, train_history