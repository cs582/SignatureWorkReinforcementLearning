import torch.nn as nn
import matplotlib.pyplot as plt

from ReinforcementLearning.training_DQN import train

if __name__ == "__main__":
    loss_function = nn.MSELoss()

    reward_metric = "roi"

    episodes = 500
    epsilon = 0.05
    gamma = 0.8
    lr = 1e-4
    momentum = 0.001

    n_transactions = 4
    n_trading_days = 1000

    initial_cash = 100000
    buy_limit = 100000
    sell_limit = 100000

    batch_size = 64
    memory_size = 100000

    q, history_dqn = train(
        n_trading_days=n_trading_days,
        n_tokens=None,
        n_transactions=n_transactions,
        initial_cash=initial_cash,
        buy_limit=buy_limit,
        sell_limit=sell_limit,
        loss_function=loss_function,
        episodes=episodes,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
        memory_size=memory_size,
        epsilon=epsilon,
        gamma=gamma,
        reward_metric=reward_metric,
        print_transactions=False
    )

    plt.title("Total reward history")
    plt.plot(history_dqn["total_reward"], color="red")
    plt.show()

    plt.title("Total average loss")
    plt.plot(history_dqn["avg_loss"], color="blue")

    plt.show()