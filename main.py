import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    filename='log.txt',
    format='%(levelname)s %(asctime)s: %(name)s - %(message)s ',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO
)

from ReinforcementLearning.training_DQN import train

if __name__ == "__main__":
    data_file = "data//OP1_2022-10-01_to_2022-08-21.csv"

    device = torch.device("cuda:0") if torch.cuda.is_available() else None

    loss_function = nn.MSELoss()

    reward_metric = "roi"

    episodes = 500
    epsilon = 0.05
    gamma = 0.8
    lr = 1e-4
    momentum = 0.001

    n_transactions = 20
    n_trading_days = 1000

    initial_cash = 100000
    buy_limit = 100000
    sell_limit = 100000

    batch_size = 64
    memory_size = 100000

    training_info = f"""
    Training DQN with
        data_file = {data_file}
        device = {"CPU" if not torch.cuda.is_available() else torch.cuda.get_device_name(device=device)}
        loss_function = {loss_function}
        reward_metric = {reward_metric}
        
        episodes = {episodes}
        epsilon = {epsilon}
        gamma = {gamma}
        lr = {lr}
        momentum = {momentum}
        
        n_transactions = {n_transactions}
        n_trading_days = {n_trading_days}
        
        initial_cash = {initial_cash}
        buy_limit = {buy_limit}
        sell_limit = {sell_limit}
        
        batch_size = {batch_size}
        memory_size = {memory_size}
    """

    logging.info(training_info)

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
        print_transactions=False,
        device=device,
        token_prices_address=data_file
    )

    logging.info("Training Complete!")

    plt.title("Total reward history")
    plt.plot(history_dqn["total_reward"], color="red")
    plt.show()

    plt.title("Total average loss")
    plt.plot(history_dqn["avg_loss"], color="blue")

    plt.show()