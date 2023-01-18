import numpy as np
import torch
import logging

from datetime import datetime

from ReinforcementLearning.q_models import DQN, DuelingDQN
from ReinforcementLearning.q_optimization import optimize_dqn

from ReinforcementLearning.saving_tools import save_model

from trading_environment.agent import Agent
from trading_environment.environment import Environment

logger = logging.getLogger("ReinforcementLearning/q_training.py")


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

        # If the filenames are given, no parameters are necessary for method initialize portfolio
        environment.initialize_portfolio()
        logger.info("Portfolio Initialized!")

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

        if model_name == "Single_DQN" or model_name == "Double_DQN":
            logger.info(f"Using Single Stream DQN model")
            q = DQN(n_classes=n_tokens, inplace=set_inplace, bias=set_bias).double().to(device=device)
            t = DQN(n_classes=n_tokens, inplace=set_inplace, bias=set_bias).double().to(device=device)
        else:
            logger.info("Using Dueling model")
            q = DuelingDQN(n_classes=n_tokens, inplace=set_inplace, bias=set_bias).double().to(device=device)
            t = DuelingDQN(n_classes=n_tokens, inplace=set_inplace, bias=set_bias).double().to(device=device)

        t.load_state_dict(q.state_dict())

        optimizer = torch.optim.SGD(q.parameters(), lr=lr, momentum=momentum)

        for episode in range(0, episodes):
            logger.info(f"Training episode {episode}")
            print(f"Training episode {episode}")

            logger.info("Initial Trade call")
            _, cur_state, _ = environment.trade()
            final_reward = 0
            episode_loss = []
            done = False
            current_trading_day = 0
            while not done:
                logger.info(f"Trading Day {current_trading_day+1}")

                # Initialize gradient to zero
                optimizer.zero_grad()

                # Predict or randomly choose an action
                y_hat = q(cur_state)

                cur_action = agent.get_action(y_hat, epsilon)
                if cur_action is None:
                    logger.warning(f"at episode {episode}, cur_action is None")

                # Trade portfolio with the given instructions
                cur_reward, next_image, done = environment.trade(cur_action)

                # Store experience in memory
                logger.debug("Creating current experience")
                cur_experience = (cur_state, cur_action, cur_reward, next_image)

                agent.store(cur_experience)

                # Update current state
                cur_state = next_image

                # Get a random minibatch of transitions from memory
                experience_batch = agent.draw(batch_size=batch_size)

                loss = optimize_dqn(dqn=q, target=t, experience_batch=experience_batch, loss_function=loss_function, gamma=gamma, optimizer=optimizer, device=device, model_name=model_name)

                episode_loss.append(loss)

                final_reward = cur_reward
                current_trading_day += 1

            print(f"Last Trading day: {current_trading_day-1}")

            train_history["metric_history"].append(final_reward)
            train_history["avg_loss"].append(np.mean(loss))

            if (episode+1)%50 == 0:
                logger.info(f"Saving model at episode {episode}")
                current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                file_path = f"{save_path}/{model_name}_{episode}_{current_time}.pt"
                save_model(model=q, episode=episode, optimizer=optimizer, train_history=train_history, PATH=file_path)

        return q, train_history
