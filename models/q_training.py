import numpy as np
import torch
import time
import logging
from datetime import datetime

from models.q_models import DQN, DuelingDQN
from models.q_optimization import optimize_dqn
from models.saving_tools import save_model
from src.trading_environment.agent import Agent
from src.trading_environment.environment import Environment

logger = logging.getLogger("reinforcement_learning/q_training.py")


def train(portfolio_to_use, n_trading_days, n_tokens, n_transactions, initial_cash, priority_fee, gas_limit, buy_limit, sell_limit, loss_function, episodes, batch_size, memory_size, lr, epsilon, gamma, momentum, reward_metric, use_change=True, use_covariance=True, device=None, token_prices_address=None, save_path=None, model_name=None, portfolio_json=None):
    with torch.autograd.set_detect_anomaly(True):
        train_history = {"metric_history": [], "avg_loss": []}

        # Initialize environment and portfolio
        environment = Environment(
            trading_days=n_trading_days,
            token_prices_address=token_prices_address,
            gas_address=None,
            gas_limit=gas_limit,
            portfolio_json=portfolio_json,
            portfolio_to_use=portfolio_to_use,
            initial_cash=initial_cash,
            buy_limit=buy_limit,
            sell_limit=sell_limit,
            priority_fee=priority_fee,
            use_change=use_change,
            use_covariance=use_covariance,
            reward_metric=reward_metric,
            device=device
        )
        logger.info("Environment Initialized!")

        # If the filenames are given, no parameters are necessary for method preload_prices
        environment.preload_prices()
        logger.info("Prices are Preloaded!")

        n_tokens = environment.n_defi_tokens if n_tokens is None else n_tokens
        logger.info(f"Number of DeFi tokens: {n_tokens}")

        # Initialize replay memory D to capacity N
        agent = Agent(
            n_transactions=n_transactions,
            n_tokens=environment.n_defi_tokens,
            memory_size=memory_size
        )
        logger.info("Agent Initialized")

        # Initialize action-value function Q with random weights
        set_inplace = True
        set_bias = False

        # Set model to use
        if model_name == "Single_DQN" or model_name == "Double_DQN":
            logger.info("Using Single Stream DQN model")
            q = DQN(n_classes=n_tokens, inplace=set_inplace, bias=set_bias).double().to(device=device)
            t = DQN(n_classes=n_tokens, inplace=set_inplace, bias=set_bias).double().to(device=device)
        else:
            logger.info("Using Dueling model")
            q = DuelingDQN(n_classes=n_tokens, inplace=set_inplace, bias=set_bias).double().to(device=device)
            t = DuelingDQN(n_classes=n_tokens, inplace=set_inplace, bias=set_bias).double().to(device=device)

        # Load weights from the q to the t model
        t.load_state_dict(q.state_dict())

        # Setting optimizer
        optimizer = torch.optim.SGD(q.parameters(), lr=lr, momentum=momentum)

        # Initiate training
        starting_time = time.time()
        for episode in range(0, episodes):

            # Start new episode
            environment.start_game()
            logger.info(f"Training episode {episode}")

            # Initialize the current state
            logger.info("Initial Trade call")
            _, cur_state, _ = environment.trade()
            rewards = []
            episode_loss = []

            final_reward = None
            done = False
            current_trading_day = 0

            # Start the trading loop
            while not done:
                logger.info(f"Trading Day {current_trading_day+1}")

                # Initialize gradient
                optimizer.zero_grad()

                # Predict select random action
                y_hat = q(cur_state)
                cur_action = agent.get_action(y_hat, epsilon)

                # Execute the action and get the reward and next state
                cur_reward, next_image, done = environment.trade(cur_action)

                # Store the experience in memory
                cur_experience = (cur_state, cur_action, cur_reward, next_image)
                agent.store(cur_experience)

                # Update the current state
                cur_state = next_image

                # Sample a batch of experiences from memory
                experience_batch = agent.draw(batch_size=batch_size)

                # Perform the optimization step
                loss = optimize_dqn(dqn=q, target=t, experience_batch=experience_batch, loss_function=loss_function, gamma=gamma, optimizer=optimizer, device=device, model_name=model_name)

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
                logger.info(f"Saving model at episode {episode}")
                current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                file_path = f"{save_path}/{model_name}_{episode}_{current_time}.pt"
                save_model(model=q, episode=episode, optimizer=optimizer, train_history=train_history, PATH=file_path)

        return q, train_history