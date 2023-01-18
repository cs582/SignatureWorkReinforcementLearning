import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime

from ReinforcementLearning.q_training import train

import logging
import argparse

logging.basicConfig(
    filename='log.txt',
    format='%(levelname)s %(asctime)s: %(name)s - %(message)s ',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG
)

parser = argparse.ArgumentParser(
    prog='DQN Trainer',
    description='DQN Training algorithm for Portfolio Management',
    epilog='Source code for Carlos Gustavo Salas Flores Signature Work at Duke University & Duke Kunshan University '
           'for the B.S. in Data Science undergrduate degree. '
)

logger = logging.getLogger("main")

parser.add_argument('-model', type=str, default="Single_DQN", help='Model to use.')
parser.add_argument('-reward', type=str, default='roi', help="Reward metric to use in training.")

parser.add_argument('-episodes', type=int, default=1000, help="Number of episodes to train.")
parser.add_argument('-e', type=float, default=0.1, help="Epsilon to train.")
parser.add_argument('-g', type=float, default=0.8, help="Gamma value for training.")
parser.add_argument('-lr', type=float, default=1e-4, help="Learning rate.")
parser.add_argument('-m', type=float, default=0.001, help="Momentum for training.")

parser.add_argument('-tr', type=int, default=4, help="Number of transactions per day.")
parser.add_argument('-d', type=int, default=1000, help="Number of trading days.")

parser.add_argument('-ic', type=int, default=100000, help="Set initial cash.")
parser.add_argument('-bl', type=int, default=100000, help="Buy units upperbound.")
parser.add_argument('-sl', type=int, default=100000, help="Sell units upperbound.")

parser.add_argument('-batch', type=int, default=128, help="Batch size.")
parser.add_argument('-memory', type=int, default=10000, help="Replay memory size.")

args = parser.parse_args()

if __name__ == "__main__":
    data_file = "data//OP1_2022-10-01_to_2022-08-21.csv"
    save_path = "ReinforcementLearning/saved_models"

    device = torch.device("cuda:0") if torch.cuda.is_available() else None
    loss_function = nn.MSELoss()

    model_name = args.model
    reward_metric = args.reward

    episodes = args.episodes
    epsilon = args.e
    gamma = args.g
    lr = args.lr
    momentum = args.m

    n_transactions = args.tr
    n_trading_days = args.d

    initial_cash = args.ic
    buy_limit = args.bl
    sell_limit = args.sl

    batch_size = args.batch
    memory_size = args.memory

    training_info = f"""
    Training {model_name} with
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
    print(training_info)

    logger.info(training_info)

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
        device=device,
        token_prices_address=data_file,
        save_path=save_path
    )

    logger.info("Training Complete!")

    plt.title("Total reward history")
    plt.plot(history_dqn["metric_history"], color="red")
    plt.show()

    plt.title("Total average loss")
    plt.plot(history_dqn["avg_loss"], color="blue")

    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    plt.savefig(f"ReinforcementLearning/figures/{model_name}_training_{current_time}.png")