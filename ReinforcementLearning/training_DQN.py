import numpy as np
import torch

from ReinforcementLearning.model import DQN
from ReinforcementLearning.optimizing_dqn import optimize_dqn
from trading_environment.agent import Agent
from trading_environment.environment import Environment


def train(n_trading_days, n_tokens, n_transactions, initial_cash, buy_limit, sell_limit, loss_function, episodes, batch_size, memory_size, lr, epsilon, gamma, momentum, reward_metric, use_change=True, use_covariance=True, print_transactions=False, device=None):
    train_history = {"metric_history": [], "loss": []}

    # Initialize environment and portfolio
    environment = Environment(
        trading_days=n_trading_days,
        n_transactions=n_transactions,
        token_prices_address="data//ClosePriceData_2022-10-01_to_2022-08-21.csv",
        gas_address=None,
        initial_cash=initial_cash,
        buy_limit=buy_limit,
        sell_limit=sell_limit,
        use_change=use_change,
        use_covariance=use_covariance,
        reward_metric=reward_metric,
        print_transactions=print_transactions,
        device=device
    )

    # If the filenames are given, no parameters are necessary for method initialize portfolio
    environment.initialize_portfolio()

    print("Number of DeFi tokens:", environment.n_defi_tokens)
    n_tokens = environment.n_defi_tokens if n_tokens is None else n_tokens

    # Initialize replay memory D to capacity N
    agent = Agent(
        n_transactions=n_transactions,
        n_tokens=environment.n_defi_tokens,
        memory_size=memory_size
    )

    # Initialize action-value function Q with random weights
    q = DQN(n_classes=n_tokens).double().to(device=device)

    optimizer = torch.optim.SGD(q.parameters(), lr=lr, momentum=momentum)

    for episode in range(0, episodes):
        print(f"Training episode {episode}")
        _, cur_state, _ = environment.trade()
        final_reward = 0
        episode_loss = []
        done = False
        while not done:
            # Initialize gradient to zero
            optimizer.zero_grad()

            # Predict or randomly choose an action
            y_hat = q(cur_state)
            cur_action = agent.get_action(y_hat, epsilon)
            if cur_action is None:
                print(f"at episode {episode}, cur_action is None")

            # Trade portfolio with the given instructions
            cur_reward, next_image, done = environment.trade(cur_action)
            print(f"cur reward {cur_reward[0]}")

            # Store experience in memory
            cur_experience = (cur_state, cur_action, cur_reward, next_image)
            agent.store(cur_experience)

            # Update current state
            cur_state = next_image

            # Get a random minibatch of transitions from memory
            experience_batch = agent.draw(batch_size=batch_size)

            loss = optimize_dqn(dqn=q, target=q, experience_batch=experience_batch, loss_function=loss_function, gamma=gamma, optimizer=optimizer)

            episode_loss.append(loss)

        train_history["metric_history"].append(final_reward)
        train_history["avg_loss"].append(np.mean(loss))

    return q, train_history
